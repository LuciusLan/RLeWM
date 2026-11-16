"""Paired native collection without a WM; simulator/oracle details stay here."""

from contextlib import contextmanager
from importlib import metadata
import inspect
from pathlib import Path

import h5py
import numpy as np
from tqdm.auto import tqdm

from diagnostics.common import require, sha, write
from diagnostics.cube_replay.cube import CubeReplay
from .protocol import array_digest, episode_paths, episode_receipt, seal_write, seed_for


@contextmanager
def numpy_stream(stream):
    previous = np.random.get_state()
    np.random.set_state(stream.get_state())
    try:
        yield
    finally:
        stream.set_state(np.random.get_state())
        np.random.set_state(previous)


class PilotCube(CubeReplay):
    """Reuse audited native stepping/rendering; independently seed each reset."""

    def variation_draw(self, rng, names):
        return {f'cube.{name}': rng.uniform(self.native.variation_space['cube'][name].low,
                                           self.native.variation_space['cube'][name].high) for name in names}

    def fresh_reset(self, case):
        seed = case['environment_seed']
        self.native.np_random = np.random.default_rng(seed)
        values = self.variation_draw(np.random.default_rng(seed), ('start_position', 'start_yaw', 'goal_position', 'goal_yaw'))
        with numpy_stream(np.random.RandomState(seed)):
            self.env.reset(seed=seed, options={'variation': [], 'variation_values': values})
        require((self.native._model.nq, self.native._model.nv, self.native._model.nu) == (21, 20, 7), 'Wrong Cube topology')
        require(abs(self.native._control_timestep-.05) < 1e-12, 'Expected 20Hz raw controls')
        return {key: value.tolist() for key, value in values.items()}

    def retarget(self, case, index):
        rng = np.random.default_rng(seed_for(case['target_seed'], index, 'target'))
        values = self.variation_draw(rng, ('goal_position', 'goal_yaw'))
        self.native.variation_space.set_value(values)
        self.native.set_new_target(p_stack=0)

    def capture(self):
        info, data = self.native.compute_ob_info(), self.native._data
        distance = np.linalg.norm(info['privileged/block_0_pos']-info['privileged/target_block_pos'])
        result = {key.replace('/', '_'): np.asarray(info[key]).copy() for key in
                  ('qpos', 'qvel', 'control', 'observation') if key in info}
        result['observation'] = np.asarray(self.native.compute_observation()).copy()
        for key in ('privileged/block_0_pos', 'privileged/block_0_quat', 'privileged/target_block_pos',
                    'privileged/target_block_yaw', 'proprio/effector_pos', 'proprio/gripper_opening', 'proprio/gripper_contact'):
            result[key.replace('/', '_')] = np.asarray(info[key]).copy()
        result.update(physics_time=np.asarray(data.time), mocap_pos=data.mocap_pos.copy(),
                      mocap_quat=data.mocap_quat.copy(), qacc_warmstart=data.qacc_warmstart.copy(),
                      collection_goal_distance_m=np.asarray(distance), collection_goal_success=np.asarray(distance <= .04, dtype=np.uint8),
                      pixels=self.render())
        require(result['pixels'].shape == (224, 224, 3) and result['pixels'].dtype == np.uint8, 'Unexpected RGB')
        require(all(np.isfinite(value).all() for value in result.values()), 'Nonfinite simulator output')
        return result


class Oracle:
    def __init__(self, backend, name, options, case):
        from ogbench.manipspace.oracles.markov.cube_markov import CubeMarkovOracle
        from ogbench.manipspace.oracles.plan.cube_plan import CubePlanOracle
        self.backend, self.name, self.case = backend, name, case
        self.stream = np.random.RandomState(case['oracle_seed'])
        self.rng = np.random.default_rng(case['noise_seed'])
        self.index, self.events = 0, []
        if name == 'markov':
            self.agent = CubeMarkovOracle(env=backend.native, min_norm=options['min_norm'])
            self.noise = self.rng.uniform(0, options['action_noise'])*np.array([1, 1, 1, 3, 10])
        else:
            self.agent = CubePlanOracle(env=backend.native, **options)
            self.noise = np.zeros(5)  # Plan noise is added to its interpolated plan, not to executed actions again.
        self.reset(0)

    def reset(self, step):
        info = self.backend.native.compute_ob_info()
        with numpy_stream(self.stream):
            self.agent.reset(None, info)
        self.events.append({'raw_step': step, 'target_sequence_index': self.index,
                            'position': info['privileged/target_block_pos'].tolist(),
                            'yaw': np.asarray(info['privileged/target_block_yaw']).tolist()})

    def prepare(self, step):
        if self.agent.done:
            self.index += 1
            self.backend.retarget(self.case, self.index)
            self.reset(step)

    def action(self):
        with numpy_stream(self.stream):
            action = np.asarray(self.agent.select_action(None, self.backend.native.compute_ob_info()), dtype=np.float64)
        if self.name == 'markov':
            action += self.rng.normal(size=5)*self.noise
        require(action.shape == (5,) and np.isfinite(action).all(), 'Invalid oracle command')
        # Same native action boundary as the SWM ExpertPolicy wrapper.
        return np.clip(action, -1, 1).astype(np.float32)

    def description(self):
        classes = type(self.agent).__mro__[:-1]
        files = {Path(inspect.getfile(cls)).resolve() for cls in classes}
        return {'class': f'{type(self.agent).__module__}.{type(self.agent).__name__}',
                'installed_sources': {str(path): sha(path) for path in sorted(files)},
                'markov_episode_noise_std': self.noise.tolist(), 'scipy': metadata.version('scipy')}


def write_episode(path, frames, actions, case):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.partial.h5')
    # Only this pilot's uncommitted file may be replaced on resume.
    with h5py.File(temporary, 'w', libver='latest') as stream:
        for key in frames[0]:
            values = np.stack([frame[key] for frame in frames])
            options = {'compression': 'gzip', 'compression_opts': 1}
            if key == 'pixels':
                options['chunks'] = (1, 224, 224, 3)
            stream.create_dataset(key, data=values, **options)
        stream['action'] = np.concatenate([np.stack(actions), np.zeros((1, 5), dtype=np.float32)])
        stream['action_valid'] = np.r_[np.ones(len(actions), dtype=np.uint8), np.uint8(0)]
        stream['ep_len'], stream['ep_offset'] = np.array([len(frames)]), np.array([0])
        stream['episode_idx'] = np.full(len(frames), case['episode_id'], dtype=np.int64)
        stream['step_idx'] = np.arange(len(frames), dtype=np.int64)
        stream.attrs['action_alignment'] = 'action[t] -> state/pixels[t+1]; terminal action invalid'
        stream.attrs['split'] = 'calibration'
    temporary.replace(path)


def check_pair(root, case, manifest):
    receipts = [episode_receipt(root, name, case, manifest) for name in ('markov', 'plan')]
    require(receipts[0]['initial_physics_sha256'] == receipts[1]['initial_physics_sha256'],
            f'Initial physical states/targets differ for pair {case["index"]}')
    with h5py.File(episode_paths(root, 'markov', case)[0], 'r', swmr=True) as first, \
         h5py.File(episode_paths(root, 'plan', case)[0], 'r', swmr=True) as second:
        error = first['pixels'][0].astype(np.float64)-second['pixels'][0]
        rgb_rmse = float(np.sqrt(np.mean(error**2)))
    require(rgb_rmse <= 2, 'Initial paired RGB differs materially')
    events = [receipt['target_events'] for receipt in receipts]
    for left, right in zip(*events):
        require(left['target_sequence_index'] == right['target_sequence_index']
                and left['position'] == right['position'] and left['yaw'] == right['yaw'], 'Common target sequence differs')
    return {'case': case['index'], 'initial_physics_identical': True, 'initial_rgb_rmse_255': rgb_rmse,
            'common_target_prefix_checked': min(map(len, events))}


def collect(root, manifest, args):
    for name, version in manifest['config']['expected_physics'].items():
        require(metadata.version(name) == version, f'Use the pinned physics-mj36 child venv: require {name}=={version}')
    config, cases = manifest['config'], manifest['cases']
    report = {'completed': False, 'manifest': manifest['fingerprint'], 'episodes': [], 'pairs': []}
    write(root/'collection-report.json', report)
    backend = PilotCube(config['raw_steps'], (224, 224, 3), 'front_pixels')
    try:
        with tqdm(total=len(cases)*2*config['raw_steps'], desc='Cube policy collection', unit='control', disable=args.quiet) as bar:
            for case in cases:
                for name, options in config['policies'].items():
                    path, receipt_path = episode_paths(root, name, case)
                    if receipt_path.exists():
                        receipt = episode_receipt(root, name, case, manifest)
                        for package, version in receipt['environment']['packages'].items():
                            require(metadata.version(package) == version, f'Collection runtime changed: {package}; use a new pilot root')
                        sources = {**receipt['environment']['installed_sources'], **receipt['oracle']['installed_sources']}
                        require(all(Path(p).is_file() and sha(Path(p)) == digest for p, digest in sources.items())
                                and metadata.version('scipy') == receipt['oracle']['scipy'], 'Installed collection code changed; use a new root')
                        report.setdefault('environment', receipt['environment'])
                        bar.update(config['raw_steps'])
                    else:
                        require(not path.exists(), f'Uncommitted final episode exists: {path}; preserve/move it before resuming')
                        bar.set_postfix(case=case['index'], policy=name)
                        variations = backend.fresh_reset(case)
                        report['active'] = {'case': case['index'], 'policy': name}
                        report.setdefault('environment', backend.description())
                        write(root/'collection-report.json', report)
                        oracle = Oracle(backend, name, options, case)
                        frames, actions = [], []
                        for step in range(config['raw_steps']):
                            oracle.prepare(step)
                            frames.append(backend.capture())
                            action = oracle.action()
                            actions.append(action)
                            backend.step(action)
                            bar.update()
                        frames.append(backend.capture())
                        write_episode(path, frames, actions, case)
                        initial = {key: frames[0][key] for key in
                                   ('qpos', 'qvel', 'control', 'mocap_pos', 'mocap_quat', 'physics_time', 'qacc_warmstart')}
                        receipt = seal_write(receipt_path, {'manifest': manifest['fingerprint'], 'case': case,
                            'policy': name, 'path': str(path), 'sha256': sha(path), 'reset_variations': variations,
                            'initial_physics_sha256': array_digest(initial), 'initial_rgb_sha256': array_digest({'pixels': frames[0]['pixels']}),
                            'target_events': oracle.events, 'oracle': oracle.description(), 'environment': backend.description()})
                    report['episodes'].append({'policy': name, 'case': case['index'], 'receipt': str(receipt_path),
                                               'sha256': receipt['sha256']})
                    write(root/'collection-report.json', report)
                report['pairs'].append(check_pair(root, case, manifest))
        report.pop('active', None)
        report['completed'] = True
        write(root/'collection-report.json', report)
    except BaseException as error:
        report['error'] = {'type': type(error).__name__, 'message': str(error)}
        write(root/'collection-report.json', report)
        raise
    finally:
        backend.close()
    from .reporting import behavior_report, previews
    behavior_report(root, manifest)
    previews(root, manifest, args.previews)
    print(f'Paired calibration collection complete: {root}/collection-summary.md')

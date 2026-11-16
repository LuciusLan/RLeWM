"""Restricted numeric AdaJEPA segment loading and explicit SWM replay port."""

import io
import pickle
import random
from pathlib import Path

import numpy as np

from .common import code_identity, headless, output, read, require, sha, write

UPSTREAM = 'https://github.com/agentic-learning-ai-lab/adajepa/blob/main/plan.py'


class NumericUnpickler(pickle.Unpickler):
    """Permit only the three NumPy constructors present in this numeric file."""
    def find_class(self, module, name):
        from numpy.core.multiarray import _reconstruct
        allowed = {('numpy', 'ndarray'): np.ndarray, ('numpy', 'dtype'): np.dtype,
                   ('numpy.core.multiarray', '_reconstruct'): _reconstruct,
                   ('numpy._core.multiarray', '_reconstruct'): _reconstruct}
        require((module, name) in allowed, f'Unsupported pickle global {module}.{name}')
        return allowed[(module, name)]


def load(path):
    path = Path(path).expanduser().resolve()
    require(path.stat().st_size <= 16*1024*1024, 'Expected a small numeric target file, at most 16 MiB')
    stream = io.BytesIO(path.read_bytes())
    data = NumericUnpickler(stream).load()
    require(not stream.read(), 'Trailing data after pickle payload')
    require(type(data) is dict and set(data) == {'segments', 'n_samples', 'traj_len', 'shape', 'seed'}, 'Unexpected target schema')
    segments = data['segments']
    require(type(segments) is list and segments and data['n_samples'] == len(segments), 'Invalid target count')
    require(type(data['traj_len']) is int and data['traj_len'] == 25 and data['shape'] == 'T', 'This port supports the supplied T/+25 targets')
    for s in segments:
        require(type(s) is dict and set(s) == {'states', 'actions', 'ep_idx', 'offset', 'shape'}, 'Unexpected segment schema')
        require(type(s['states']) is np.ndarray and type(s['actions']) is np.ndarray
                and s['states'].shape == (26, 5) and s['actions'].shape == (25, 2)
                and s['states'].dtype == np.float32 and s['actions'].dtype == np.float32
                and np.isfinite(s['states']).all() and np.isfinite(s['actions']).all(), 'Invalid finite numeric segment arrays')
        require(s['shape'] == 'T' and isinstance(s['ep_idx'], (int, np.integer))
                and isinstance(s['offset'], (int, np.integer)) and s['ep_idx'] >= 0 and s['offset'] >= 0,
                'Invalid segment provenance')
    return data


def inspect(path):
    path = Path(path).expanduser().resolve()
    data = load(path)
    actions = np.stack([s['actions'] for s in data['segments']])
    states = np.stack([s['states'] for s in data['segments']])
    return {'path': str(Path(path).resolve()), 'sha256': sha(path), 'segments': len(actions),
        'traj_len': data['traj_len'], 'shape': data['shape'], 'file_seed': data['seed'],
        'state_shape_per_segment': list(states.shape[1:]), 'action_shape_per_segment': list(actions.shape[1:]),
        'dtype': 'float32', 'state_columns': ['agent_x', 'agent_y', 'block_x', 'block_y', 'block_angle'],
        'unique_source_episodes': len({int(s['ep_idx']) for s in data['segments']}),
        'stored_action_min': actions.min((0, 1)).tolist(), 'stored_action_max': actions.max((0, 1)).tolist(),
        'environment_action_min_after_dividing_100': (actions/100).min((0, 1)).tolist(),
        'environment_action_max_after_dividing_100': (actions/100).max((0, 1)).tolist(),
        'rgb_present': False, 'velocities_present': False,
        'action_convention': 'pixel-space relative commands; divide by 100 before relative SWM step; never clip',
        'goal_convention': 'upstream replays actions from the zero-velocity padded initial state and uses replay endpoint RGB/state',
        'history_limit': 'one observed frame gives H5; no pre-initial history is supplied',
        'provenance_limit': 'ep_idx/offset do not establish overlap or independence from LeWM training data',
        'upstream_loader': UPSTREAM, 'restricted_numpy_loader': True}


def state_errors(actual, stored):
    difference = actual[:, :5]-stored
    difference[:, 4] = (difference[:, 4]+np.pi) % (2*np.pi)-np.pi
    return {'maximum_abs_position_error_pixels': float(np.abs(difference[:, :4]).max()),
            'maximum_abs_angle_error_radians': float(np.abs(difference[:, 4]).max()),
            'endpoint_agent_error_pixels': float(np.linalg.norm(difference[-1, :2])),
            'endpoint_block_error_pixels': float(np.linalg.norm(difference[-1, 2:4])),
            'endpoint_angle_error_radians': float(abs(difference[-1, 4]))}


def replay(args):
    """User-run CPU physics, not executed by the coding agent."""
    headless()
    from lewm_tta.simulation.environments.fresh_pusht import FreshPushT
    from PIL import Image
    data = load(args.file)
    require(0 < args.samples <= len(data['segments']) and args.seed >= 0, 'Invalid sample count/seed')
    selected = random.Random(args.seed).sample(range(len(data['segments'])), args.samples)
    directory = output(args.output)
    environment = FreshPushT(224, {})
    pixels, states, actions, stored, rows = [], [], [], [], []
    try:
        for position, index in enumerate(selected):
            s = data['segments'][index]
            seed = args.seed*position+1  # AdaJEPA's eval_seed convention.
            environment.reset(seed)
            native = environment.env.unwrapped
            native.agent.velocity = (0., 0.)
            native.block.velocity = (0., 0.)
            native.block.angular_velocity = 0.
            native._set_state(np.r_[s['states'][0].astype(np.float64), 0., 0.])
            observations = [environment._observe(native._get_obs())]
            raw = (s['actions'].astype(np.float64)/100).astype(np.float32)
            for action in raw:
                # Recorded controls must not pass through the collector/planner
                # clipping/bounds interface. The upstream segment loader also
                # executes these finite commands unchanged after dividing 100.
                obs, _, _, truncated, _ = environment.env.step(action)
                observations.append(environment._observe(obs['state'], False, truncated))
            rgb = np.stack([o.rgb for o in observations])
            physical = np.stack([o.state for o in observations])
            pixels.append(rgb); states.append(physical); actions.append(raw); stored.append(s['states'])
            rows.append({'segment_index': index, 'ep_idx': int(s['ep_idx']), 'offset': int(s['offset']),
                         'reset_seed': seed, 'stored_state_replay_errors': state_errors(physical, s['states']),
                         'recorded_actions_executed_unchanged': True,
                         'outside_nominal_action_box_steps': int((np.abs(raw) > 1+1e-6).any(axis=1).sum()),
                         'maximum_absolute_raw_action': float(np.abs(raw).max())})
            # Replay only: these previews must never be labelled as a planned trajectory.
            panels = np.concatenate((rgb[0], rgb[-1]), axis=1)
            Image.fromarray(panels).save(directory/f'segment-{index:04d}-start-goal.png')
            frames = [Image.fromarray(frame) for frame in rgb]
            frames[0].save(directory/f'segment-{index:04d}-recorded-replay.gif', save_all=True,
                           append_images=frames[1:], duration=100, loop=0)
            print(f'Replayed segment {index}: endpoint block error vs stored {rows[-1]["stored_state_replay_errors"]["endpoint_block_error_pixels"]:.4g} px', flush=True)
        payload = directory/'replay.npz'
        np.savez_compressed(payload, pixels=np.stack(pixels), states=np.stack(states), actions=np.stack(actions),
                            stored_states=np.stack(stored), segment_indices=np.asarray(selected, dtype=np.int64))
        report = {'stage': 'AdaJEPA-SWM-replay-diagnostic', 'completed': True, 'targets': inspect(args.file),
            'samples': args.samples, 'seed': args.seed, 'segments': rows, 'environment': environment.description(),
            'code': code_identity(), 'payload': str(payload), 'payload_sha256': sha(payload),
            'protocol': 'port of AdaJEPA segment initial-state/action replay into current SWM; actual replay frames are evaluation truth',
            'limits': ['not an exact AdaJEPA simulator reproduction', 'not a held-out-data provenance claim',
                       'not CEM or a planning success benchmark', 'stored velocities/contact state are absent',
                       'changing MuJoCo does not change this Pymunk task']}
        write(directory/'report.json', report)
        (directory/'summary.md').write_text('# AdaJEPA target replay in SWM\n\n'
            'Actions were divided by 100, without clipping. Initial velocities were zero. '
            'Goal frames are the actual replay endpoints, as in the upstream loader. '
            'PNGs/GIFs show recorded-action replay, not planned or decoded imagination.\n\n'
            'Stored-state deviations are diagnostics of cross-environment replay compatibility. '
            'This port does not establish exact AdaJEPA parity or independence from WM pretraining.\n')
        print(f'Replay payload: {payload}', flush=True)
    finally:
        environment.close()


def load_replay(path):
    path = Path(path).resolve()
    report = read(path.with_name('report.json'))
    require(report['stage'] == 'AdaJEPA-SWM-replay-diagnostic' and report['completed']
            and report['payload_sha256'] == sha(path), 'Replay identity/checksum differs')
    data = np.load(path, allow_pickle=False)
    n = report['samples']
    require(data['pixels'].shape == (n, 26, 224, 224, 3) and data['pixels'].dtype == np.uint8
            and data['states'].shape == (n, 26, 7) and data['actions'].shape == (n, 25, 2)
            and np.isfinite(data['states']).all() and np.isfinite(data['actions']).all(), 'Invalid replay payload')
    return data, report

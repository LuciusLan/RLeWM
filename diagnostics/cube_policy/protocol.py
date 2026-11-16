"""Model-free protocol, immutable reservations and shared artifact contracts."""

from contextlib import contextmanager
import fcntl
import hashlib
from pathlib import Path

from diagnostics.common import ROOT, checked, identity, read, require, sha, write


def seed_for(seed, case, stream):
    return int(identity(['cube-policy-calibration-v1', seed, case, stream])[:8], 16)


def code_identity(collection_only=False):
    if collection_only:
        # Evaluator/report/test fixes must not invalidate already collected RGB.
        files = [Path(__file__).parent/name for name in ('collection.py', 'protocol.py')]
        files += [ROOT/'codes/diagnostics/common.py', ROOT/'codes/diagnostics/cube_replay/cube.py']
    else:
        files = list(Path(__file__).parent.glob('*.py'))
        files += [ROOT/'codes/diagnostics'/name for name in ('cli.py', 'common.py', 'run.sh', 'frozen.py')]
        files += list((ROOT/'codes/diagnostics/cube_replay').glob('*.py'))
    return {str(path.relative_to(ROOT)): sha(path) for path in sorted(files)}


def resolve(args):
    config = read(args.config)
    require(set(config) == {'version', 'episodes', 'seed', 'raw_steps', 'window_starts', 'history', 'frameskip',
                           'horizon', 'expected_physics', 'policies', 'frozen_reference', 'frozen_reference_sha256', 'repo'},
            'Unexpected/missing Cube policy profile keys')
    for key in ('episodes', 'seed'):
        if getattr(args, key) is not None:
            config[key] = getattr(args, key)
    require(config['version'] == 1 and type(config['episodes']) is int and 1 <= config['episodes'] <= 64
            and type(config['seed']) is int and config['seed'] >= 0, 'Use 1–64 calibration pairs and a nonnegative seed')
    require(config['history'] == 3 and config['frameskip'] == 5 and config['horizon'] == 5,
            'Pilot fixes C3, frameskip5 and H5; H1 is its first prediction endpoint')
    require(type(config['raw_steps']) is int and config['raw_steps'] >= 35, 'Need at least 35 raw controls')
    starts = config['window_starts']
    require(isinstance(starts, list) and starts and all(type(s) is int and s >= 0 for s in starts)
            and starts == sorted(set(starts)) and starts[-1]+35 <= config['raw_steps'], 'Invalid fixed query-window starts')
    require(config['expected_physics'] == {'mujoco': '3.6.0', 'dm-control': '1.0.38'}, 'Keep the audited physics pair')
    policies = config['policies']
    require(set(policies) == {'markov', 'plan'} and set(policies['markov']) == {'min_norm', 'action_noise'}
            and set(policies['plan']) == {'noise', 'noise_smoothing', 'segment_dt'}, 'Invalid oracle settings')
    import math
    require(all(type(v) in (int, float) and math.isfinite(v) and v >= 0 for p in policies.values() for v in p.values())
            and policies['markov']['min_norm'] > 0 and policies['plan']['segment_dt'] > 0
            and policies['plan']['noise_smoothing'] > 0,
            'Oracle options must be finite/nonnegative; min_norm/segment_dt/noise_smoothing positive')
    require(args.previews >= 0 and args.cpu_threads > 0 and args.run_name not in ('', '.', '..')
            and Path(args.run_name).name == args.run_name, 'Invalid previews/threads/evaluation name')
    reference_path = (ROOT/config['frozen_reference']).resolve(strict=True)
    require(sha(reference_path) == config['frozen_reference_sha256'], 'Pinned Frozen reference changed')
    reference = read(reference_path)
    require(reference['completed'] and reference['passed'] and reference['dataset'] == 'cube', 'Need passed Cube Frozen reference')
    require(len(reference['normalization']['mean']) == 5 and len(reference['normalization']['std']) == 5
            and all(math.isfinite(v) for v in reference['normalization']['mean'])
            and all(math.isfinite(v) and v > 0 for v in reference['normalization']['std']), 'Invalid reference action statistics')
    cases = [{'index': i, 'split': 'calibration', 'episode_id': 1_500_000_000+i,
              'environment_seed': seed_for(config['seed'], i, 'environment'),
              'oracle_seed': seed_for(config['seed'], i, 'oracle'),
              'noise_seed': seed_for(config['seed'], i, 'noise'),
              'target_seed': seed_for(config['seed'], i, 'targets')} for i in range(config['episodes'])]
    require(len({c['environment_seed'] for c in cases}) == len(cases), 'Seed collision')
    root = (args.output or ROOT/f"artifacts/diagnostics/cube-policy/paired-e{config['episodes']}-s{config['seed']}").expanduser().resolve()
    manifest = {'version': 1, 'diagnostic': 'cube-policy-calibration', 'config': config, 'cases': cases,
                'code_identity': code_identity(collection_only=True), 'protocol': {
                    'pairing': 'identical reset draws/initial physical state and ordered targets',
                    'retargeting': 'each oracle advances the common target sequence on its own done signal; times may differ',
                    'goals': 'native collection targets; prediction goals are recorded futures +25 raw controls',
                    'acceptance': 'all prespecified cases/windows; no WM error or task-success filtering',
                    'model_feedback_during_collection': False, 'test_set': False,
                    'action_alignment': 'action[t] takes state/pixels[t] to state/pixels[t+1]; terminal action invalid'}}
    manifest['fingerprint'] = identity(manifest)
    return config, reference, cases, root, manifest


@contextmanager
def writer_lock(root):
    root.mkdir(parents=True, exist_ok=True)
    with (root/'.writer.lock').open('a') as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ValueError(f'Another writer is using {root}') from error
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def ensure_manifest(root, expected, create=False):
    path = root/'manifest.json'
    if path.exists():
        require(checked(path) == expected, 'Pilot profile/code changed; choose a new output root')
    else:
        require(create, 'Run cube-policy collect first')
        require(set(p.name for p in root.iterdir()) <= {'.writer.lock'}, 'Refusing to adopt a nonempty pilot root')
        write(path, expected)


def episode_paths(root, policy, case):
    base = root/'episodes'/policy/f"calibration-{case['index']:05d}"
    return base.with_suffix('.h5'), base.with_suffix('.json')


def episode_receipt(root, policy, case, manifest):
    path, receipt_path = episode_paths(root, policy, case)
    require(path.is_file() and receipt_path.is_file(), f'Missing {policy} case {case["index"]}; run collect')
    receipt = checked(receipt_path)
    require(receipt['manifest'] == manifest['fingerprint'] and receipt['case'] == case and receipt['policy'] == policy
            and receipt['sha256'] == sha(path), f'Changed committed episode: {path}')
    return receipt


def seal_write(path, value):
    value = {**value, 'fingerprint': identity(value)}
    write(path, value)
    return value


def array_digest(values):
    """Stable small-array receipts; numpy is imported only in execution paths."""
    import numpy as np
    digest = hashlib.sha256()
    for key, value in sorted(values.items()):
        value = np.ascontiguousarray(value)
        digest.update(identity([key, value.dtype.str, list(value.shape)]).encode())
        digest.update(value.tobytes())
    return digest.hexdigest()

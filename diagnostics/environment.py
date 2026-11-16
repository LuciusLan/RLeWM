"""Metadata audit and optional user-run subprocess-isolated simulator smoke checks."""

from importlib import metadata, util
import json
from pathlib import Path
import subprocess
import sys

from packaging.requirements import Requirement
from packaging.version import Version

from .common import ROOT, code_identity, headless, require, sha

FAST_REQUIREMENTS = 'https://github.com/Yuntian-Gao/Fast-LeWorldModel/blob/main/requirements.txt'
FAST_README = 'https://github.com/Yuntian-Gao/Fast-LeWorldModel#installation'
FAST_PINS = {'stable-worldmodel': '0.0.6', 'stable-pretraining': '0.1.6',
    'mujoco': '3.6.0', 'dm-control': '1.0.38', 'gymnasium': '1.2.3',
    'torch': '2.10.0+cu128', 'torchvision': '0.25.0+cu128', 'numpy': '2.2.6',
    'pymunk': '7.2.0', 'pygame': '2.6.1', 'shapely': '2.1.2', 'transformers': '5.3.0'}
MODULES = {'stable-worldmodel': 'stable_worldmodel', 'stable-pretraining': 'stable_pretraining',
           'dm-control': 'dm_control', 'gymnasium-robotics': 'gymnasium_robotics'}


def package(name):
    try:
        item = metadata.distribution(name)
    except metadata.PackageNotFoundError:
        return {'installed': False}
    module = util.find_spec(MODULES.get(name, name.replace('-', '_')))
    return {'installed': True, 'version': item.version, 'location': str(item.locate_file('')),
            'module_origin': module.origin if module else None, 'requires': item.requires or [],
            'direct_url': json.loads(item.read_text('direct_url.json')) if item.read_text('direct_url.json') else None}


def active_requirements(rows, proposed=None):
    problems = []
    proposed = proposed or {}
    for name, row in rows.items():
        for text in row.get('requires', []):
            r = Requirement(text)
            if r.marker and not r.marker.evaluate({'extra': ''}):
                continue
            try:
                version = proposed.get(r.name, metadata.version(r.name))
            except metadata.PackageNotFoundError:
                problems.append({'package': name, 'requires': text, 'resolved': None})
                continue
            if r.specifier and Version(version) not in r.specifier:
                problems.append({'package': name, 'requires': text, 'resolved': version})
    return problems


def snapshot():
    rows = {name: package(name) for name in [*FAST_PINS, 'ogbench', 'gymnasium-robotics']}
    recorded = {}
    for task in ('pusht', 'cube', 'reacher', 'tworooms'):
        path = ROOT/f'datasets/fresh_data/{task}/source/episodes/validation-00000.json'
        if path.exists():
            value = json.loads(path.read_text())
            recorded[task] = {'record': str(path), 'sha256': sha(path), 'environment': value['environment']}
    return {'stage': 'environment-compatibility-audit', 'python': sys.version, 'executable': sys.executable,
        'installed': rows, 'fast_lewm_documented_pins': FAST_PINS,
        'upstream_requirements': FAST_REQUIREMENTS, 'upstream_warning': FAST_README,
        'active_base_dependency_problems': active_requirements(rows),
        'problems_if_only_mujoco_is_set_to_3_6_0': active_requirements(rows, {'mujoco': '3.6.0'}),
        'recorded_collection_environments': recorded, 'code': code_identity(),
        'scope': 'metadata only; no simulator/model imports, weights, stepping, or numerical equivalence claims',
        'caveats': ['Fast-LeWM pins are not a recovered lockfile of original LeWM training',
                    'optional extras are not fully checked; simulator smoke checks remain necessary',
                    'a physics overlay keeps SWM/Torch fixed and is not the full Fast-LeWM environment',
                    'existing recorded RGB/latents do not change when a simulator package is replaced']}


def worker(task):
    """Invoked only by the user's --smoke command, in a fresh process per task."""
    headless()
    import numpy as np
    if task == 'cube':
        from lewm_tta.simulation.environments.fresh_cube import FreshCube as Factory
    elif task == 'reacher':
        from lewm_tta.simulation.environments.fresh_reacher import FreshReacher as Factory
    else:
        from lewm_tta.simulation.environments.fresh_pusht import FreshPushT as Factory
    env = Factory(224, {})
    try:
        first = env.reset(12345)
        action = np.zeros(env.env.action_space.shape, dtype=np.float32)
        second = env.step(action)
        require(first.rgb.shape == second.rgb.shape == (224, 224, 3)
                and second.rgb.dtype == np.uint8 and np.isfinite(second.state).all(), 'Simulator RGB/state contract failed')
        bindings = {}
        if task in ('cube', 'reacher'):
            import mujoco
            bindings = {'version': mujoco.__version__, 'native_library_version': mujoco.mj_versionString(),
                        'module': mujoco.__file__}
            require(mujoco.__version__ == metadata.version('mujoco'), 'Imported MuJoCo differs from resolved metadata')
        return {'passed': True, 'task': task, 'environment': env.description(), 'bindings': bindings,
                'state_width': len(second.state), 'rgb_shape': list(second.rgb.shape),
                'scope': 'one reset, one zero-action control and RGB render; no model inference or source-physics parity proof'}
    finally:
        env.close()


def smoke(tasks):
    results = []
    for task in tasks:
        print(f'Simulator smoke: {task}', file=sys.stderr, flush=True)
        command = [sys.executable, '-X', 'faulthandler', '-u', str(ROOT/'codes/diagnostics/cli.py'),
                   'environment', '--worker', task]
        try:
            result = subprocess.run(command, capture_output=True, text=True, timeout=60)
            marker = 'DIAGNOSTIC_JSON='
            lines = [line[len(marker):] for line in result.stdout.splitlines() if line.startswith(marker)]
            value = json.loads(lines[-1]) if lines else {'passed': False}
            value.update(returncode=result.returncode, stderr=result.stderr[-6000:])
            value['passed'] = bool(value['passed'] and result.returncode == 0)
        except subprocess.TimeoutExpired:
            value = {'passed': False, 'task': task, 'error': '60-second worker timeout'}
        results.append(value)
    return results

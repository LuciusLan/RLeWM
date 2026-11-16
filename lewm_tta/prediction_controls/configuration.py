"""Resolved inputs and provenance, independent of simulator-specific code."""

from copy import deepcopy
from itertools import product
import math
from pathlib import Path

from ablations.saved_runs import checked, immutable_json, read, require, sha

STAGE = 'P2-memory-prediction-controls'
BASELINES = ({'control': 'frozen', 'selector': None, 'gain': None},
             {'control': 'persistence', 'selector': None, 'gain': None})


def key(case):
    if case['selector'] is None:
        return case['control']
    return f"{case['control']}/{case['selector']}/gain={case['gain']:g}"


def cases(spec, chosen=None):
    if chosen is None:
        options = [{'control': c, 'selector': s, 'gain': float(g)}
                   for c, s, g in product(spec['controls'], spec['selectors'], spec['gains'])]
    else:
        options = [dict(item) for item in chosen]
        if spec['test_unit_gain']:
            options += [{'control': c, 'selector': s, 'gain': 1.0} for c, s in product(spec['controls'], spec['selectors'])]
    return [*map(dict, BASELINES), *{key(c): c for c in options}.values()]


def resolve(root, path, reference, output=None):
    spec = read(path)
    require(spec['dataset'] == 'pusht' and spec['support_size'] == 32 and spec['horizon'] == 5, 'This profile fixes PushT B32/H5')
    require(spec['selection_horizon'] in (1, 3, 5), 'Select gains using H1, H3 or H5')
    require(spec['reference_schedule'] == 'temporal-u1', 'Use the matched temporal-U1 TTT reference')
    require(spec['selectors'] and len(set(spec['selectors'])) == len(spec['selectors'])
            and set(spec['selectors']) <= {'uniform', 'forward', 'reverse', 'fr-fixed'}, 'Invalid selectors')
    require(spec['controls'] == ['residual-transfer', 'delta-transfer'], 'Keep both prediction controls')
    require(spec['test_unit_gain'] is True, 'Keep the canonical gain-1 controls on test')
    require(len(set(spec['gains'])) == len(spec['gains']) and 0 in spec['gains'] and 1 in spec['gains']
            and all(type(g) in (int, float) and math.isfinite(g) and g >= 0 for g in spec['gains']), 'Use distinct nonnegative gains including zero and one')
    require(all(type(spec['cache'][k]) is int and spec['cache'][k] > 0 for k in ('batch_rows', 'checkpoint_rows')), 'Invalid cache batch/checkpoint size')
    require(type(spec['evaluation']['bootstrap_samples']) is int and spec['evaluation']['bootstrap_samples'] > 0, 'Bootstrap draws must be positive')
    require(len(cases(spec)) == 2+len(spec['controls'])*len(spec['selectors'])*len(spec['gains']), 'Gain names collide; use more separated grid values')
    spec['experiment_dir'] = str((output or root/spec['experiment_dir']).resolve())
    spec['cache_dir'] = str((root/spec['cache_dir']).resolve())
    c = reference['config']
    require(reference['stage'] == 'P2-followup-iteration' and reference['completed'] and not reference['limited_run']
            and reference['split'] == 'validation', 'Require a completed full fresh P2 validation reference')
    require(c['scopes'] == ['last-block'] and c['p1']['evaluation']['seeds'] == spec['seeds'], 'Reference scope/seeds changed')
    a = {**c['p1']['adaptation'], **c['adaptation']}
    require(a['support_size'] == 32 and a['learning_rate'] == 5e-5 and set(spec['selectors']) <= set(a['methods']), 'Reference selectors/budget/LR changed')
    require(c['schedules'][spec['reference_schedule']] == {'placement': 'each-step', 'updates': 1, 'persistence': 'trajectory'}, 'Reference schedule changed')
    require(c['p1']['retrieval']['pool_size'] == 32 and c['p1']['retrieval']['max_windows_per_episode'] == 4
            and c['p1']['retrieval']['min_start_step_gap'] == 20, 'Reference diversity settings changed')
    q = checked(c['query_dataset']['manifest'])
    require(q['external_queries'] and not q['episodes']['memory'], 'Require independent fresh evaluation episodes')
    require(c['p1']['data']['history'] == 3 and c['p1']['data']['frameskip'] == 5, 'Temporal convention differs')
    base = deepcopy(c)
    base['experiment_dir'] = spec['experiment_dir']
    base['adaptation']['methods'] = ['frozen', *spec['selectors']]
    base['schedule_names'] = [spec['reference_schedule']]
    base['schedules'] = {spec['reference_schedule']: deepcopy(c['schedules'][spec['reference_schedule']])}
    base['evaluation'].update(spec['evaluation'])
    base['evaluation'].update(records='compact', query_limit=None)
    return spec, base


def implementation(root, reference):
    files = sorted((root/'codes/lewm_tta/prediction_controls').glob('*.py'))
    files += [root/'codes/ablations/memory_prediction.py', root/'codes/ablations/saved_runs.py']
    return {'numerical_core': reference['code'],
            'prediction_controls': {str(p.relative_to(root)): sha(p) for p in files}}


def save_plan(spec, base, code):
    path = Path(spec['experiment_dir'])
    immutable_json(path/'config.json', {'specification': spec, 'base_config': base, 'code': code})


def load_lock(spec, code):
    path = Path(spec['experiment_dir'])/'locks/validation.json'
    lock = checked(path)
    require(lock['stage'] == STAGE and lock['specification'] == spec and lock['code'] == code, 'Validation lock config/code changed')
    report = read(lock['validation_report'])
    require(sha(Path(lock['validation_report'])) == lock['validation_sha256'] and report['completed']
            and report['split'] == 'validation' and not report['limited_run'], 'Locked validation report changed')
    require(report['selected_gains'] == lock['selected_gains'] and report['test_cases'] == lock['test_cases'], 'Locked gains/cases differ')
    from .reporting import selection
    require(selection(report['summary'], spec) == lock['selected_gains']
            and cases(spec, lock['selected_gains']) == lock['test_cases'], 'Gains do not follow the prespecified validation rule')
    return lock

"""Fresh-validation adaptation scopes with fixed support and a bounded LR grid.

User-run orchestration only. Reuse measured last-block controls and the exact
execution sources for each reference. --dry-run/--analyze-only make no model
calls. The visual encoder/projector, memory, normalization and indexes stay fixed.
"""

import argparse
from copy import deepcopy
from itertools import product
import json
import math
from pathlib import Path
import shlex
import subprocess
import sys

if __package__:
    from .saved_runs import ROOT, checked, executor, identity, immutable_json, read, record_rows, require, sha
    from .support_size import collect as collect_fresh_records
else:
    from saved_runs import ROOT, checked, executor, identity, immutable_json, read, record_rows, require, sha
    from support_size import collect as collect_fresh_records


def numerical_sources(report):
    return {name: digest for name, digest in report['code'].items() if name.startswith('codes/lewm_tta/')}


def references(spec):
    """Validate fresh source controls; frontend-only differences stay explicit."""
    require(spec['stage'] in ('p2', 'p3') and spec['dataset'] == 'pusht', 'This pilot uses PushT P2/P3')
    require(spec['support_size'] == 32, 'Keep B32 fixed for the scope ablation')
    require(set(spec['scopes']) <= set(spec['available_scopes']) <= {'last-block', 'predictor', 'predictor-action'},
            'Invalid adaptation scopes')
    require(spec['learning_rates'] and len(set(spec['learning_rates'])) == len(spec['learning_rates'])
            and all(math.isfinite(rate) and rate > 0 for rate in spec['learning_rates']), 'Invalid LR grid')
    require(set(spec['default_learning_rates']) <= set(spec['learning_rates']), 'Default LRs must belong to the grid')
    require(len(spec['methods']) == len(set(spec['methods'])) and {'frozen', 'uniform', 'fr-fixed'} <= set(spec['methods']),
            'Include Frozen, random support and FR-fixed')
    found = {}
    for entry in spec['references']:
        path = ROOT/entry['report']
        require(sha(path) == entry['sha256'], f'Pinned reference changed: {path}')
        report = read(path)
        rate, c = entry['learning_rate'], report['config']
        a = {**c['p1']['adaptation'], **c['adaptation']}
        stage = 'P2-followup-iteration' if spec['stage'] == 'p2' else 'P3c-offline-factorial'
        require(report['completed'] and not report['limited_run'] and report['split'] == 'validation'
                and report['stage'] == stage, 'Require a completed full validation reference')
        require(rate in spec['learning_rates'] and rate not in found, 'Duplicate or undeclared reference LR')
        require(c['scopes'] == ['last-block'] and c['p1']['data'].get('dataset_id', 'pusht') == 'pusht',
                'Reference task/scope mismatch')
        require(c['p1']['data']['history'] == 3 and c['p1']['data']['frameskip'] == 5,
                'This pilot retains the original history3/frameskip5 convention')
        require(a['support_size'] == 32 and a['learning_rate'] == rate
                and c['p1']['evaluation']['seeds'] == spec['seeds'], 'Reference budget/LR/seeds mismatch')
        require(set(spec['methods']) <= set(a['methods']), 'Reference does not cover every requested selector')
        r = c['p1']['retrieval']
        require(r['backend'] == 'cuvs' and r['pool_size'] == 32 and r['max_windows_per_episode'] == 4
                and r['min_start_step_gap'] == 20, 'Reference retrieval settings differ')
        if spec['stage'] == 'p2':
            require(c['horizons'] == [1, 3, 5] and c['schedules'][spec['schedule']] == {
                'placement': 'each-step', 'updates': 1, 'persistence': 'trajectory'}, 'Use temporal-U1/H5')
        else:
            require(a['updates'] == 1 and c['planning']['horizon'] == 5, 'Use one initial P3 update/H5')
        q = checked(c['query_dataset']['manifest'])
        audit = checked(Path(c['query_dataset']['manifest']).with_name('audit.json'))
        require(q['external_queries'] and not q['episodes']['memory'] and audit['passed']
                and audit['query_manifest'] == q['fingerprint'], 'Fresh query/audit mismatch')
        require(report['cohort']['queries'] == 256 and report['cohort']['episodes'] == 64
                and report['cohort']['raw_start_rows'] == q['evaluation_windows']['validation']['raw_start_rows'],
                'Expected 256 fresh queries from 64 validation episodes')
        if found:
            first = next(iter(found.values()))
            for field in ('cohort', 'model', 'p1_prepared_fingerprint'):
                require(report[field] == first[field], f'Reference mismatch: {field}')
            require(numerical_sources(report) == numerical_sources(first), 'Reference numerical code changed')
            for field in ('paths', 'data', 'retrieval', 'evaluation'):
                require(c['p1'][field] == first['config']['p1'][field], f'Reference p1.{field} differs')
            require(c['query_dataset'] == first['config']['query_dataset'], 'Reference query source/cache differs')
            previous = {**first['config']['p1']['adaptation'], **first['config']['adaptation']}
            require({k: v for k, v in a.items() if k not in ('methods', 'learning_rate')}
                    == {k: v for k, v in previous.items() if k not in ('methods', 'learning_rate')},
                    'Reference optimization settings differ beyond LR')
        found[rate] = report
    require(found, 'At least one measured last-block reference is required')
    return found


def configurations(spec, source, scope, rate, directory):
    p1 = deepcopy(source['config']['p1'])
    p1_path = directory/'configs'/f'p1-{identity(p1)[:16]}.json'
    config = deepcopy(source['config'])
    config.pop('p1')
    config['p1_config'] = str(p1_path)
    config['experiment_dir'] = str(directory)
    config['scopes'] = [scope]
    config['adaptation'].update(methods=spec['methods'], support_size=32, learning_rate=rate)
    config['evaluation'].update(records='compact', query_limit=None)
    if spec['stage'] == 'p2':
        config['schedule_names'] = [spec['schedule']]
        config['schedules'] = {spec['schedule']: deepcopy(config['schedules'][spec['schedule']])}
    else:
        config['reuse'] = {}
    path = directory/'configs'/f"{spec['stage']}-{scope}-lr{rate:g}-{identity(config)[:16]}.json"
    expected = {**deepcopy(config), 'p1': deepcopy(p1)}
    return p1_path, p1, path, config, expected


def scope_description(report, spec, scope):
    description = report['scope_parameters']
    if spec['stage'] == 'p2':
        description = description[scope]
    roots = {'predictor'} if scope == 'last-block' else {'predictor', 'pred_proj'}
    if scope == 'predictor-action':
        roots.add('action_encoder')
    require(description['scope'] == scope and set(description['trainable_module_roots']) == roots
            and set(description['parameters_by_module']) == roots, 'Unexpected trainable modules')
    require(description['visual_encoder_and_latent_projector_frozen'] and description['normalization_statistics_fixed'],
            'Visual representation or normalization was adapted')
    names = description['parameter_names']
    require(names and all(name.split('.')[0] in roots for name in names), 'Unexpected parameter names')
    require(description['trainable_parameters'] > 0
            and description['trainable_parameters'] == sum(description['parameters_by_module'].values()),
            'Scope parameter count mismatch')
    return description


def collect(report, source, spec, scope, rate, *, control=False, expected=None):
    # The shared support-budget collector also audits cohort/provenance, exact
    # targets, record hashes, B32, update/LR budgets, weights and episode spacing.
    audit_spec = {**spec, 'learning_rate': rate}
    values = collect_fresh_records(report, source, audit_spec, 32, control=control, expected=expected)
    description = scope_description(report, spec, scope)
    visited = set()
    for row in record_rows(report):
        if control and (row['method'] not in spec['methods'] or (
                spec['stage'] == 'p2' and row['schedule'] != spec['schedule'])):
            continue
        require(row.get('scope', report['config']['scopes'][0]) == scope, 'Record scope differs')
        cell = row['method'] if spec['stage'] == 'p2' else row['method']+'/'+row['proposal_source']
        key = cell, row['query']['raw_observation_rows'][0], row['seed']
        visited.add(key)
        a = row['adaptation']
        support = (a['rounds'][0]['support'] if a['rounds'] else []) if spec['stage'] == 'p2' else a['support']
        values[key]['initial_support'] = identity([{k: item[k] for k in ('transition_id', 'weight', 'channels')}
                                                  for item in support])
        values[key]['query_fingerprint'] = identity(row['query'])
        if spec['stage'] == 'p2':
            require(a['reset_count'] == 1 and len(a['positions']) == 5, 'Temporal reset/position policy changed')
            first = a['positions'][0]
            require(first['context_source'] == 'observed_initial_history'
                    and all(p['context_source'] == 'shifted_predicted_history' for p in a['positions'][1:]),
                    'Autoregressive context policy changed')
            values[key]['initial_input'] = identity([first['context_fingerprint'], first['action_fingerprint'],
                                                     first['query_mse_before'], a['query_rng_seed']])
            modules = a['module_updates']
            require(set(modules) == set(description['trainable_module_roots']), 'Missing per-module diagnostics')
            for name, module in modules.items():
                require(module['trainable_parameters'] == description['parameters_by_module'][name],
                        'Module parameter-count mismatch')
                require(math.isfinite(module['displacement_from_checkpoint_mse'])
                        and module['displacement_from_checkpoint_mse'] >= 0, 'Invalid parameter displacement')
                if row['method'] != 'frozen':
                    require(module['parameter_tensors_with_gradient'] > 0, 'An intended module received no gradient')
                else:
                    require(module['displacement_from_checkpoint_mse'] == 0, 'Frozen weights changed')
        else:
            values[key]['initial_input'] = identity([row['provisional_fingerprint'], a['reset_seed']])
        values[key]['displacement'] = a['parameter_displacement_mse']
        require(math.isfinite(values[key]['displacement']) and values[key]['displacement'] >= 0,
                'Invalid aggregate parameter displacement')
    require(visited == set(values), 'Scope audit omitted rows')
    return {'rows': values, 'description': description, 'report': report}


def completed(path, source, spec, scope, rate, expected):
    if not path.exists():
        require(not path.parent.exists(), f'Partial run preserved: {path.parent}; use another --output-dir')
        return None
    return collect(read(path), source, spec, scope, rate, expected=expected)


def check_pairing(data, base):
    require(set(data['rows']) == set(base['rows']), 'Scope/LR query matrices differ')
    for key, row in data['rows'].items():
        original = base['rows'][key]
        for field in ('episode', 'query_fingerprint', 'initial_input', 'initial_support'):
            require(row[field] == original[field], f'Unmatched initial scope/LR inputs: {field}')
        if key[0].split('/')[0] == 'frozen':
            require(row['errors'] == original['errors'], 'Frozen forecasts changed across scope/LR')


def analyze(spec, measured, directory):
    import csv
    import numpy as np

    base = measured['last-block', spec['references'][0]['learning_rate']]
    starts, seeds = base['report']['cohort']['raw_start_rows'], spec['seeds']
    cells = sorted({key[0] for key in base['rows']})
    groups_by_query = np.array([base['rows'][cells[0], start, seeds[0]]['episode'] for start in starts])
    groups = [np.flatnonzero(groups_by_query == episode) for episode in np.unique(groups_by_query)]
    draws = np.random.default_rng(20261110).integers(len(groups), size=(10000, len(groups)))
    tolerance = base['report']['config']['p1']['evaluation']['harm_tolerance']

    def array(data, cell, metric):
        return np.asarray([[data['rows'][cell, start, seed][metric] for seed in seeds] for start in starts])

    def statistic(values):
        by_episode = np.array([values[group].mean() for group in groups])
        return {'mean': float(by_episode.mean()), 'ci95': np.quantile(by_episode[draws].mean(1), [.025, .975]).tolist()}

    means, per_seed, paired, catalog = [], [], [], []
    for (scope, rate), data in sorted(measured.items()):
        report = data['report']
        reuse = scope == 'last-block' and any(rate == e['learning_rate'] for e in spec['references'])
        report_path = (ROOT/next(e['report'] for e in spec['references'] if e['learning_rate'] == rate) if reuse
                       else directory/'runs'/f'scope-{scope}-lr{rate:g}-validation'/'report.json')
        catalog.append({'scope': scope, 'learning_rate': rate, 'report': str(report_path), 'sha256': sha(report_path),
                        'records_sha256': report['records_sha256'], 'reused_reference': reuse,
                        'code_fingerprint': identity(report['code']), 'scope_description': data['description']})
        check_pairing(data, base)
        for cell in cells:
            method, *proposal = cell.split('/')
            frozen_cell = 'frozen' + ('/'+proposal[0] if proposal else '')
            errors = array(data, cell, 'errors')
            frozen = array(base, frozen_cell, 'errors')
            require(np.array_equal(array(data, frozen_cell, 'errors'), frozen), 'Frozen forecasts changed across scope/LR')

            def metrics(seed_index=None):
                sl = slice(None) if seed_index is None else slice(seed_index, seed_index+1)
                delta = errors[:, sl, -1]-frozen[:, sl, -1]
                return {'dataset': spec['dataset'], 'split': 'validation', 'stage': spec['stage'], 'scope': scope,
                        'learning_rate': rate, 'setting': cell, 'seed': 'mean' if seed_index is None else seeds[seed_index],
                        'support_size': 32, 'reused_reference': reuse,
                        'scope_parameters': data['description']['trainable_parameters'],
                        **{f'H{h}': float(errors[:, sl, h-1].mean()) for h in (1, 3, 5)},
                        'path_mse': float(errors[:, sl].mean()), 'harm_fraction': float((delta > tolerance).mean()),
                        'positive_harm_mass': float(np.maximum(delta, 0).mean()),
                        'seconds': float(array(data, cell, 'seconds')[:, sl].mean()),
                        'effective_sample_size': None if method == 'frozen' else float(array(data, cell, 'ESS')[:, sl].mean()),
                        'parameter_displacement_mse': float(array(data, cell, 'displacement')[:, sl].mean()),
                        'common_frozen_sse': None if spec['stage'] == 'p2' else float(array(data, cell, 'common_sse')[:, sl].mean()),
                        'own_model_sse': None if spec['stage'] == 'p2' else float(array(data, cell, 'own_sse')[:, sl].mean()),
                        'report': str(report_path)}

            means.append(metrics())
            per_seed.extend(metrics(i) for i in range(len(seeds)))
            paired.append({'scope': scope, 'learning_rate': rate, 'setting': cell, 'contrast': 'minus frozen',
                           'H5_delta': statistic((errors[:, :, -1]-frozen[:, :, -1]).mean(1))})
            scope_controls = ['last-block'] if scope != 'last-block' else []
            if scope == 'predictor-action':
                scope_controls.append('predictor')
            for control_scope in scope_controls:
                comparison = measured.get((control_scope, rate))
                if comparison is None:
                    continue
                contrast = {'scope': scope, 'learning_rate': rate, 'setting': cell,
                            'contrast': f'minus {control_scope} at same LR',
                            'H5_delta': statistic((errors[:, :, -1]-array(comparison, cell, 'errors')[:, :, -1]).mean(1))}
                if spec['stage'] == 'p3':
                    contrast['common_frozen_sse_delta'] = statistic(
                        (array(data, cell, 'common_sse')-array(comparison, cell, 'common_sse')).mean(1))
                paired.append(contrast)
            if method == 'fr-fixed':
                suffix = '/'+proposal[0] if proposal else ''
                for control in ('uniform', 'forward', 'reverse'):
                    if control+suffix in cells:
                        paired.append({'scope': scope, 'learning_rate': rate, 'setting': cell, 'contrast': 'minus '+control,
                                       'H5_delta': statistic((errors[:, :, -1]-array(data, control+suffix, 'errors')[:, :, -1]).mean(1))})
    directory.mkdir(parents=True, exist_ok=True)
    result = {'dataset': spec['dataset'], 'stage': spec['stage'], 'split': 'validation', 'queries': 256, 'episodes': 64,
              'seeds': seeds, 'schedule': spec.get('schedule', 'one shared initial update per CEM call'),
              'support_size': 32, 'declared_learning_rates': spec['learning_rates'], 'declared_scopes': spec['available_scopes'],
              'harm_tolerance': tolerance, 'support_episode_cap': 4, 'support_start_gap': 20,
              'catalog': catalog, 'means': means, 'per_seed': per_seed, 'paired': paired,
              'checks': {'fresh_inputs_and_observed_targets_matched': True, 'frozen_forecasts_identical': True,
                         'first_support_identical_across_scope_and_lr': True, 'later_support_allowed_to_follow_adapted_predictions': True},
              'intervals': '10000 paired episode-bootstrap draws after seed averaging; pointwise descriptive 95%; seed20261110',
              'interpretation': 'Scope sensitivity at each measured LR. Report the complete LR grid; a fixed-LR result does not establish tuned scope superiority. P3 search costs are model proxies, not physical success.'}
    (directory/'overview.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    for name, rows in (('means.csv', means), ('per-seed.csv', per_seed)):
        with (directory/name).open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    lines = ['# PushT fresh-validation adaptation-scope ablation', '', result['interpretation'], '',
             '| Scope | LR | Setting | H5 MSE | Harm | Parameters | Seconds | Reused |',
             '|---|---:|---|---:|---:|---:|---:|---|']
    for row in means:
        lines.append(f"| {row['scope']} | {row['learning_rate']:g} | {row['setting']} | {row['H5']:.7f} | "
                     f"{row['harm_fraction']:.2%} | {row['scope_parameters']:,} | {row['seconds']:.3f} | {row['reused_reference']} |")
    lines += ['', 'Frozen performs no updates. The parameter column describes the adaptable scope for that row.',
              'Last-block is the final transformer block, rather than a single linear output layer.',
              'First supports are matched; temporal supports after adaptation may differ across scopes and LRs.',
              'Paired scope/method contrasts and confidence intervals are in overview.json; per-seed rows are in per-seed.csv.',
              'Latencies come from separately executed runs. No test evaluation, validation lock or automatic scope/LR selection is performed.']
    (directory/'overview.md').write_text('\n'.join(lines)+'\n')
    print(f'Saved scope overview and per-seed CSV: {directory}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--scopes', nargs='+', choices=('last-block', 'predictor', 'predictor-action'))
    parser.add_argument('--learning-rates', type=float, nargs='+', help='Subset of the declared LR grid')
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--cpu-threads', type=int, default=8)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--dry-run', action='store_true')
    mode.add_argument('--analyze-only', action='store_true')
    args = parser.parse_args()
    spec = read(args.config)
    sources = references(spec)
    scopes = spec['scopes'] if args.scopes is None else args.scopes
    rates = spec['default_learning_rates'] if args.learning_rates is None else args.learning_rates
    require(scopes and len(set(scopes)) == len(scopes) and set(scopes) <= set(spec['available_scopes']), 'Invalid scope subset')
    require(rates and len(set(rates)) == len(rates) and set(rates) <= set(spec['learning_rates']), 'Invalid LR subset')
    require(args.cpu_threads > 0, 'cpu-threads must be positive')
    directory = (args.output_dir or ROOT/spec['experiment_dir']).resolve()
    backend = 'p2_followup' if spec['stage'] == 'p2' else 'p3c'
    prefixes = {rate: executor(source, backend) for rate, source in sources.items()}
    base_rate = spec['references'][0]['learning_rate']
    plan = {'specification': spec, 'executors_by_reference_lr': {str(rate): p for rate, p in prefixes.items()},
            'cpu_threads': args.cpu_threads, 'runner_sha256': sha(Path(__file__)),
            'helpers_sha256': {name: sha(Path(__file__).with_name(name)) for name in ('saved_runs.py', 'support_size.py')}}
    plan_path = directory/'scope-plan.json'
    if plan_path.exists():
        require(read(plan_path) == plan, 'Existing scope plan differs; choose another --output-dir')
    print(f'PushT fresh validation: scopes={scopes}, LRs={rates}, B32, seeds={spec["seeds"]}', flush=True)
    measured = {}
    for rate, source in sources.items():
        measured['last-block', rate] = collect(source, source, spec, 'last-block', rate, control=True)
        print(f'Reuse last-block LR={rate:g}: {source["records_file"]}', flush=True)
    base = measured['last-block', base_rate]
    for data in measured.values():
        check_pairing(data, base)
    jobs = []
    for scope, rate in product(spec['available_scopes'], spec['learning_rates']):
        if (scope, rate) in measured:
            continue
        source = sources.get(rate, sources[base_rate])
        blueprint = configurations(spec, source, scope, rate, directory)
        path = directory/'runs'/f'scope-{scope}-lr{rate:g}-validation'/'report.json'
        done = completed(path, source, spec, scope, rate, blueprint[-1])
        jobs.append((scope, rate, source, blueprint, path, done))
        if done is not None:
            check_pairing(done, base)
            measured[scope, rate] = done
            print('Skip verified completed:', path, flush=True)
    pending = [job for job in jobs if job[0] in scopes and job[1] in rates and job[-1] is None]

    def run(command):
        print('RUN', shlex.join(command), flush=True)
        if not args.dry_run:
            subprocess.run(command, cwd=ROOT, check=True)

    if not args.dry_run and not args.analyze_only:
        immutable_json(plan_path, plan)
    if pending and not args.analyze_only:
        for _, _, _, (pp, p1, cp, c, _), _, _ in pending:
            if not args.dry_run:
                immutable_json(pp, p1)
                immutable_json(cp, c)
        first = pending[0]
        prefix = prefixes[first[1]] if first[1] in prefixes else prefixes[base_rate]
        run([*prefix, 'prepare', '--config', str(first[3][2])])
        for scope, rate, source, (_, _, cp, _, expected), path, _ in pending:
            prefix = prefixes[rate] if rate in prefixes else prefixes[base_rate]
            run([*prefix, 'evaluate', '--config', str(cp), '--split', 'validation', '--run-name', path.parent.name,
                 '--seeds', *map(str, spec['seeds']), '--device', 'cuda', '--cpu-threads', str(args.cpu_threads)])
            if not args.dry_run:
                result = completed(path, source, spec, scope, rate, expected)
                require(result is not None, 'Evaluation did not complete')
                check_pairing(result, base)
                measured[scope, rate] = result
    if args.dry_run:
        print('Dry run complete: no model calls or output writes.', flush=True)
    else:
        analyze(spec, measured, directory)


if __name__ == '__main__':
    try:
        main()
    except (ValueError, KeyError, OSError, subprocess.CalledProcessError) as error:
        print(f'Adaptation-scope ablation failed: {error}', file=sys.stderr)
        raise SystemExit(1)

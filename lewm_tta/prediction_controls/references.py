"""Project verified saved TTT results without treating them as new inference."""

from pathlib import Path

import numpy as np

from ablations.saved_runs import identity, read, record_rows, require, sha


def load(root, spec, split, validation):
    entry = spec['references'][split]
    path = root/entry['report']
    require(sha(path) == entry['sha256'], f'Pinned {split} report changed')
    report = read(path)
    require(report['completed'] and not report['limited_run'] and report['split'] == split
            and report['stage'] == validation['stage'], 'Incomplete/incompatible TTT reference')
    for field in ('config', 'code', 'model', 'manifest_fingerprint', 'p1_prepared_fingerprint', 'query_source_fingerprint'):
        require(report[field] == validation[field], f'TTT validation/test reference differs: {field}')
    if split == 'test':
        # The old test has already been evaluated. Preserve its original validation lock.
        lock_path = Path(report['config']['experiment_dir'])/'locks/fresh-validation.json'
        lock = read(lock_path)
        require(lock['config'] == validation['config'] and lock['code'] == validation['code']
                and lock['manifest_fingerprint'] == validation['manifest_fingerprint']
                and lock['report_sha256'] == spec['references']['validation']['sha256'],
                'The reused test has no matching original validation lock')
    starts = report['cohort']['raw_start_rows']
    allowed = set(starts)
    methods = {'frozen', *spec['selectors']}
    output, count = {}, 0
    for row in record_rows(report):
        count += 1
        if row.get('schedule') != spec['reference_schedule'] or row['method'] not in methods:
            continue
        start = int(row['query']['raw_observation_rows'][0])
        seed, method = row['seed'], row['method']
        identifier = (start, seed, method)
        require(start in allowed and seed in spec['seeds'] and identifier not in output, 'Duplicate/unknown reference row')
        require(row['scope'] == 'last-block' and all(row['checks'].values())
                and all(row['adaptation']['checks'].values()), 'TTT reference row failed checks')
        require(row['target_raw_rows'] == [start+10+5*h for h in range(1, 6)], 'Reference horizon alignment changed')
        errors = np.asarray(row['mse_by_block'], dtype=np.float64)
        require(errors.shape == (5,) and np.isfinite(errors).all(), 'Invalid TTT reference errors')
        a = row['adaptation']
        require(a['actual_updates'] == (0 if method == 'frozen' else 5)
                and a['support_presentations'] == (0 if method == 'frozen' else 160), 'Unmatched TTT reference budget')
        support = a['rounds'][0]['support'] if method != 'frozen' else None
        if support is not None:
            require(len(support) == 32, 'Unmatched first support budget')
        output[identifier] = {'query': row['query'], 'target_raw_rows': row['target_raw_rows'],
            'mse_by_block': errors.tolist(), 'total_seconds': row['total_seconds'],
            'first_support_fingerprint': identity(support) if support is not None else None,
            'support_presentations': a['support_presentations'], 'optimizer_steps': a['actual_updates'],
            'mean_effective_support_size': float(np.mean([
                1/sum(item['weight']**2 for item in update['support']) for update in a['rounds']])) if a['rounds'] else 0.0}
    require(count == report['record_count'] and len(output) == len(starts)*len(spec['seeds'])*len(methods),
            'Incomplete saved TTT matrix')
    return report, output


def verify_runtime(resources, manifest, reference, split):
    require(resources.adapter.identity == reference['model'], 'Frozen model/runtime differs from saved TTT')
    require(resources.prepared['fingerprint'] == reference['p1_prepared_fingerprint'], 'Prepared source changed')
    require(manifest['splits'][split] == reference['cohort'], 'Fresh query cohort differs from saved TTT')
    require(manifest['query_source_fingerprint'] == reference['query_source_fingerprint'], 'Fresh source changed')


def projection(reference_rows, selectors, starts):
    """Small numeric projection for reporting; no support lists or raw latents."""
    allowed = set(starts)
    rows = []
    for (start, seed, method), row in reference_rows.items():
        if start in allowed:
            rows.append({'case': 'reference/frozen' if method == 'frozen' else f'ttt/{method}',
                'seed': seed, 'start': start, 'episode': row['query']['source_episode_position'],
                'mse_by_block': row['mse_by_block'], 'total_seconds': row['total_seconds'],
                'optimizer_steps': row['optimizer_steps'], 'support_presentations': row['support_presentations'],
                'mean_effective_support_size': row['mean_effective_support_size']})
    require({r['case'] for r in rows} == {'reference/frozen', *[f'ttt/{s}' for s in selectors]}, 'Missing projected reference')
    return rows

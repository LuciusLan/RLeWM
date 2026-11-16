"""Small CSV/JSON tables and episode-clustered intervals; no model imports."""

from collections import defaultdict
import csv
import gzip
from pathlib import Path
import statistics

from diagnostics.common import checked, identity, read, require, sha, write
from .run import load_query, records_per_query


GROUP = ('dataset', 'scope', 'method', 'learning_rate', 'comparison', 'selection', 'k', 'position', 'seed')
META = {'query_id', 'episode_id', *GROUP}


def group_rows(rows):
    groups = defaultdict(list)
    for row in rows:
        # Full seed-average and individual-seed tables, overall and per depth.
        for position in ('all', row['position']):
            for seed in ('all', row['seed']):
                key = tuple(position if name == 'position' else seed if name == 'seed' else
                            row.get('scope', 'last-block') if name == 'scope' else row[name] for name in GROUP)
                groups[key].append(row)
    return groups


def summarize(rows, bootstrap_samples):
    import numpy as np
    result = []
    for key, values in group_rows(rows).items():
        group = dict(zip(GROUP, key))
        episodes = sorted({row['episode_id'] for row in values})
        by_episode = {episode: i for i, episode in enumerate(episodes)}
        rng = np.random.default_rng(int(identity(list(key))[:16], 16))
        draws = rng.integers(len(episodes), size=(bootstrap_samples, len(episodes))) if bootstrap_samples else None
        fields = sorted(set(values[0])-META)
        for metric in fields:
            valid = [row for row in values if row[metric] is not None]
            if not valid:
                continue
            data = np.asarray([row[metric] for row in valid], dtype=np.float64)
            require(np.isfinite(data).all(), f'Nonfinite saved metric: {metric}')
            total, count = np.zeros(len(episodes)), np.zeros(len(episodes))
            for row in valid:
                index = by_episode[row['episode_id']]
                total[index] += row[metric]
                count[index] += 1
            interval = [None, None]
            if draws is not None and len(episodes) > 1:
                denominators = count[draws].sum(1)
                estimates = total[draws].sum(1)[denominators > 0]/denominators[denominators > 0]
                if len(estimates):
                    interval = np.quantile(estimates, [.025, .975]).tolist()
            result.append({**group, 'metric': metric, 'observations': len(data),
                'queries': len({row['query_id'] for row in valid}),
                'episodes': len({row['episode_id'] for row in valid}),
                'seeds': len({row['seed'] for row in valid}),
                'mean': float(data.mean()), 'std': float(data.std()),
                'median': float(np.median(data)), 'p05': float(np.quantile(data, .05)),
                'p95': float(np.quantile(data, .95)), 'ci95_low': interval[0], 'ci95_high': interval[1]})
    return result


def csv_file(path, rows):
    require(rows, f'Cannot write an empty statistics table: {path}')
    temporary = path.with_suffix(path.suffix+'.tmp')
    opener = gzip.open if path.suffix == '.gz' else open
    with opener(temporary, 'wt', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def headline(dataset, scope, request, rows):
    lines = []
    for method in request['methods']:
        for lr in request['learning_rates']:
            selected = [row for row in rows if row['scope'] == scope and row['method'] == method and row['learning_rate'] == lr]
            k = 32 if 32 in request['ks'] else request['ks'][0]

            def mean(comparison, selection, metric, position=None):
                values = [row[metric] for row in selected if row['comparison'] == comparison
                          and row['selection'] == selection and row['k'] == (request['support_size'] if selection == 'actual_fr' else k)
                          and (position is None or row['position'] == position) and row[metric] is not None]
                return statistics.mean(values) if values else None

            def display(value, percent=False):
                return '—' if value is None else f'{value*100:.1f}%' if percent else f'{value:.2f}'

            initial = mean('forward_reverse_before', 'raw', 'intersection', 0)
            overlap = mean('forward_reverse_before', 'raw', 'intersection')
            churn = mean('reverse_after_update', 'raw', 'new_fraction')
            diverse = mean('reverse_after_update', 'diverse', 'new_fraction')
            fr = mean('fr_support_after_update', 'actual_fr', 'added')
            tv = mean('fr_support_after_update', 'actual_fr', 'weight_total_variation')
            lines.append(f'| {dataset} | {scope} | {method} | {lr:g} | {k} | {display(initial)} | {display(overlap)} | '
                         f'{display(churn, True)} | {display(diverse, True)} | {display(fr)} | {display(tv)} |')
    return lines


def saved_dataset(directory, dataset):
    request = checked(directory/'manifest.json')
    fingerprint = request.pop('fingerprint')
    complete = read(directory/'measurement.json')
    require(complete.get('completed') and complete['request'] == fingerprint
            and len(complete['files']) == request['queries'], 'Require complete matching measurement receipts')
    require(sha(directory/'runtime.json') == complete['runtime_sha256'], 'Runtime receipt changed')
    records = []
    for index in range(request['queries']):
        name = f'queries/{index:05d}.json.gz'
        require(sha(directory/name) == complete['files'][name], f'Changed compressed statistics: {directory/name}')
        checkpoint = load_query(directory/name, fingerprint, index)
        require(len(checkpoint['records']) == records_per_query(request), 'Incomplete per-query probe matrix')
        for row in checkpoint['records']:
            require(row.get('scope', request['scope']) == request['scope'], 'Saved row uses another trainable scope')
            records.append({**row, 'scope': request['scope'], 'dataset': dataset})
    require(records and len({row['query_id'] for row in records}) == request['queries'], 'Incomplete query coverage')
    receipt = {'measurement_sha256': sha(directory/'measurement.json'), 'request': fingerprint}
    return request, records, receipt, read(directory/'runtime.json')


def analyze(args):
    args.output = Path(args.output).resolve()
    all_rows, requests, receipts, baselines, runtimes, differences = [], {}, {}, {}, {}, []
    for dataset in args.datasets:
        request, records, receipt, runtime = saved_dataset(args.output/dataset, dataset)
        requests[dataset], receipts[dataset] = request, receipt
        all_rows.extend(records)
        if args.compare_to is not None:
            from .comparison import paired_rows, validate_pair, validate_runtime
            previous, old_records, old_receipt, old_runtime = saved_dataset(args.compare_to/dataset, dataset)
            validate_pair(request, previous)
            validate_runtime(runtime, old_runtime)
            differences.extend(paired_rows(records, old_records))
            all_rows.extend(old_records)
            baselines[dataset] = {'root': str(args.compare_to), 'request': previous, 'receipt': old_receipt}
            runtimes[dataset] = {'predictor': runtime, 'last-block': old_runtime}
    summary = summarize(all_rows, args.bootstrap_samples)
    csv_file(args.output/'per-position.csv.gz', all_rows)
    csv_file(args.output/'summary.csv', [row for row in summary if row['seed'] == 'all'])
    csv_file(args.output/'per-seed.csv.gz', [row for row in summary if row['seed'] != 'all'])
    table_names = ['per-position.csv.gz', 'summary.csv', 'per-seed.csv.gz']
    if args.compare_to is not None:
        from .comparison import format_comparison
        paired = summarize(differences, args.bootstrap_samples)
        csv_file(args.output/'scope-comparison.csv.gz', paired)
        (args.output/'scope-comparison.md').write_text(format_comparison(summary, paired, runtimes))
        table_names += ['scope-comparison.csv.gz', 'scope-comparison.md']
    analysis_paths = [Path(__file__)]
    if args.compare_to is not None:
        analysis_paths.append(Path(__file__).with_name('comparison.py'))
    write(args.output/'summary.json', {'completed': True, 'references': receipts, 'requests': requests,
        'reused_last_block': baselines,
        'limited_run': any(request['limit'] is not None for request in requests.values()),
        'bootstrap_samples': args.bootstrap_samples,
        'interval_unit': 'episode cluster; all seeds/windows/depths for an episode resampled together',
        'analysis_code': {str(path): sha(path) for path in analysis_paths},
        'tables': {name: {'path': str(args.output/name), 'sha256': sha(args.output/name)}
                   for name in table_names}})
    lines = ['# Validation retrieval statistics', '',
        'The first overlap column uses the original observed context, before any TTT. The next averages '
        'over all five positions of the adapted autoregressive trajectory. Reverse churn compares '
        'before/after one update at the same position, context and action. The Update samples column '
        'identifies the strategy supplying the actual gradient step (reverse-only by default). '
        'No second update is executed.', '',
        '| Dataset | Scope | Update samples | LR | Prefix k | Initial F∩R count | Mean F∩R count | Raw R replaced | Diverse R replaced | FR samples added | FR weight TV |',
        '|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for dataset in args.datasets:
        selected = [row for row in all_rows if row['dataset'] == dataset]
        for scope in dict.fromkeys(row['scope'] for row in selected):
            lines += headline(dataset, scope, requests[dataset], selected)
    lines += ['', 'Raw = nearest transition prefixes. Diverse = independently selected prefixes with the '
        'actual per-episode cap/start-gap constraints. FR = the complete weighted support selected with '
        'the unchanged forward ranking and refreshed reverse ranking. Prefix k never changes the '
        'configured TTT support size. Weight TV is half the L1 distance between normalized support '
        'weight maps (0 unchanged, 1 disjoint). For reverse-only updates, FR statistics are hypothetical '
        'selections along the reverse-adapted trajectory, not measurements after an FR-trained update.', '',
        '`summary.csv` reports seed-average means, quantiles and 95% episode-cluster bootstrap intervals; '
        '`per-seed.csv.gz` retains individual seeds. `per-position.csv.gz` contains only numeric statistics, '
        'without predicted latents, full rankings, weights or transition lists. Compressed per-query '
        'checkpoints support exact resume.', '',
        '`reverse_between_positions` is a separate descriptive control: post-update R at h versus '
        'pre-update R at h+1. Context and action change there, so it is not a within-position update effect. '
        'These statistics describe membership and weighting, not proof that refreshing improves accuracy.']
    if args.compare_to is not None:
        lines += ['', 'The last-block baseline is reused read-only. See [the paired scope comparison](scope-comparison.md) '
                  'for predictor-minus-last-block effects and episode-clustered intervals. The predictor scope '
                  'updates predictor + pred_proj; action/visual encoders stay frozen.']
    if any(request['limit'] is not None for request in requests.values()):
        lines += ['', '**Limited smoke cohort: do not report these tables as full validation results.**']
    (args.output/'summary.md').write_text('\n'.join(lines)+'\n')
    print(f"Retrieval statistics: {args.output/'summary.md'}", flush=True)

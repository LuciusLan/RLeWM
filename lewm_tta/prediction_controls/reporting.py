"""Paired query/seed statistics with episode-clustered confidence intervals."""

from collections import defaultdict
import csv
from pathlib import Path

import numpy as np

from ablations.saved_runs import require
from .configuration import key


def compact(row):
    return {k: row[k] for k in ('case', 'seed', 'start', 'episode', 'mse_by_block', 'total_seconds',
                               'optimizer_steps', 'support_presentations', 'mean_effective_support_size')}


def metrics(row):
    values = row['mse_by_block']
    require(len(values) == 5 and np.isfinite(values).all(), 'Expected finite H5 errors')
    return [values[0], values[2], values[4], float(np.mean(values))]


def selection(summary, spec):
    selected = []
    h = f"h{spec['selection_horizon']}_mse"
    for control in spec['controls']:
        for selector in spec['selectors']:
            options = [{'control': control, 'selector': selector, 'gain': float(g)} for g in spec['gains']]
            selected.append(min(options, key=lambda c: (summary[key(c)][h], c['gain'])))
    return selected


def summarize(rows, reference, spec):
    all_rows = [*rows, *reference]
    groups = defaultdict(dict)
    episodes = {}
    for row in all_rows:
        pair = (row['seed'], row['start'])
        require(pair not in groups[row['case']], 'Duplicate numeric result')
        groups[row['case']][pair] = row
        require(row['start'] not in episodes or episodes[row['start']] == row['episode'], 'Query episode mismatch')
        episodes[row['start']] = row['episode']
    pairs = sorted(groups['frozen'])
    require(pairs and all(set(g) == set(pairs) for g in groups.values()), 'Incomplete paired comparison matrix')
    starts = sorted(episodes)
    require(set(pairs) == {(seed, start) for seed in spec['seeds'] for start in starts}, 'Incomplete seed/query matrix')
    names = ('h1_mse', 'h3_mse', 'h5_mse', 'path_mse')
    tensors = {name: np.asarray([metrics(g[pair]) for pair in pairs], dtype=np.float64) for name, g in groups.items()}
    frozen = tensors['frozen']
    require(np.allclose(frozen, tensors['reference/frozen'], rtol=1e-5, atol=1e-7), 'Saved frozen reference differs')
    clusters = sorted(set(episodes.values()))
    labels = np.asarray([episodes[start] for _, start in pairs])
    # Each bootstrap sample draws episodes; seeds and q4 windows stay in their episode.
    draws = np.random.default_rng(20261111).integers(len(clusters), size=(spec['evaluation']['bootstrap_samples'], len(clusters)))

    def ci(values):
        per_episode = np.stack([values[labels == episode].mean(0) for episode in clusters])
        estimates = per_episode[draws].mean(1)
        return np.percentile(estimates, [2.5, 97.5], axis=0).T.tolist()

    summary, per_seed = {}, []
    tolerance = 1e-8
    for name, tensor in sorted(tensors.items()):
        if name == 'reference/frozen':
            continue
        values = groups[name]
        delta = tensor-frozen
        row = {**dict(zip(names, tensor.mean(0).tolist())),
            'ci95': dict(zip(names, ci(tensor))), 'queries': len(starts), 'episodes': len(clusters),
            'seeds': spec['seeds'], 'query_seed_pairs': len(pairs),
            'mean_seconds': float(np.mean([r['total_seconds'] for r in values.values()])),
            'mean_optimizer_steps': float(np.mean([r['optimizer_steps'] for r in values.values()])),
            'mean_support_presentations': float(np.mean([r['support_presentations'] for r in values.values()])),
            'mean_effective_support_size': float(np.mean([r['mean_effective_support_size'] for r in values.values()])),
            'reused_reference': name.startswith('ttt/'),
            'vs_frozen': {**dict(zip(names, delta.mean(0).tolist())),
                'ci95': dict(zip(names, ci(delta))),
                'harm_fraction': dict(zip(names, (delta > tolerance).mean(0).tolist())),
                'positive_harm': dict(zip(names, np.maximum(delta, 0).mean(0).tolist()))}}
        if name.startswith(('residual-transfer/', 'delta-transfer/')):
            selector = name.split('/')[1]
            difference = tensor-tensors[f'ttt/{selector}']
            row['vs_matched_ttt'] = {**dict(zip(names, difference.mean(0).tolist())),
                                     'ci95': dict(zip(names, ci(difference)))}
        summary[name] = row
        for seed in spec['seeds']:
            selected = np.asarray([p[0] == seed for p in pairs])
            per_seed.append({'case': name, 'seed': seed, 'queries': int(selected.sum()),
                **dict(zip(names, tensor[selected].mean(0).tolist())),
                'h5_harm_fraction': float((delta[selected, 2] > tolerance).mean()),
                'h5_positive_harm': float(np.maximum(delta[selected, 2], 0).mean()),
                'mean_seconds': float(np.mean([r['total_seconds'] for (s, _), r in values.items() if s == seed])),
                'mean_effective_support_size': float(np.mean([r['mean_effective_support_size'] for (s, _), r in values.items() if s == seed])),
                'reused_reference': name.startswith('ttt/')})
    return summary, per_seed


def write_tables(directory, report, per_seed):
    directory = Path(directory)
    with (directory/'per-seed.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(per_seed[0]))
        writer.writeheader()
        writer.writerows(per_seed)
    lines = [f"# Frozen-memory prediction controls — {report['split']}", '',
        'H1/H3/H5 are autoregressive endpoint MSE; path averages all five endpoints. Lower is better.',
        'Confidence intervals resample episodes, retaining all query windows and seeds within each episode.',
        'TTT rows reuse the pinned temporal-U1, last-block, LR5e-5 results. No TTT was rerun.', '',
        '| Method | H1 | H3 | H5 | Path | H5 Δ vs frozen [95% CI] | H5 harm | Seconds/query |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for name, row in report['summary'].items():
        d = row['vs_frozen']
        lo, hi = d['ci95']['h5_mse']
        lines.append(f"| {name} | {row['h1_mse']:.6f} | {row['h3_mse']:.6f} | {row['h5_mse']:.6f} | "
            f"{row['path_mse']:.6f} | {d['h5_mse']:+.6f} [{lo:+.6f}, {hi:+.6f}] | "
            f"{d['harm_fraction']['h5_mse']:.2%} | {row['mean_seconds']:.4f} |")
    if 'selected_gains' in report:
        lines += ['', f"Gains selected on validation mean H{report['specification']['selection_horizon']} (ties: smaller gain):", '']
        lines += [f"- {c['control']} / {c['selector']}: {c['gain']:g}" for c in report['selected_gains']]
    cache = report['prediction_cache']
    lines += ['', f"One-time source-memory prediction cache: {cache['bytes']/2**30:.3f} GiB; "
        f"{cache['construction_seconds']:.1f} seconds. Excluded from query timing; CPU cache lookups and GPU transfers are included.",
        'Saved TTT and new controls ran at different times; latency is descriptive, not a paired hardware benchmark.',
        'Support fingerprints match TTT at the initial position. Subsequent support may differ as predictions diverge.',
        'The full gain sweep is retained; test reports both the validation-selected gain and gain 1.', '']
    (directory/'summary.md').write_text('\n'.join(lines))

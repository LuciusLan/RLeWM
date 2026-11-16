"""Average actions/seeds within query, windows within episode, then bootstrap episodes."""

from __future__ import annotations

from collections import defaultdict
import csv
from pathlib import Path

import numpy as np

from .metrics import group_metrics


def summarize(records: list[dict], config: dict) -> dict:
    grouped = defaultdict(lambda: defaultdict(list))
    for record in records:
        values = group_metrics(record, config["prefix_sizes"], config["percentages"], config["support_size"])
        baseline = [values["p/100/fr/unique_retained_fraction"], values["p/100/fr/channel_loss_mass_retained"]]
        baseline.extend(values[f"p/100/topk/{k}/{direction}_coverage"] for k in config["prefix_sizes"]
                        for direction in ("forward", "reverse"))
        if not np.allclose(baseline, 1., rtol=0, atol=1e-10) or not np.isclose(
                values["p/100/pair/consensus_retained_count"], values["pair/consensus_original_count"]):
            raise ValueError("Coverage 100% baseline failed to reproduce unfiltered support")
        grouped[record["mode"]][record["query"]["transition_id"]].append((record, values))
    summary = {}
    for mode, queries in grouped.items():
        keys, query_values, episodes, seed_counts = None, [], [], []
        for group in queries.values():
            episode_ids = {item[0]["query"]["source_episode_position"] for item in group}
            seeds = [item[0]["seed"] for item in group]
            if len(episode_ids) != 1 or len(set(seeds)) != len(seeds):
                raise ValueError("Duplicate seeds or inconsistent query episode IDs")
            if keys is None:
                keys = sorted(group[0][1])
            if any(set(values) != set(keys) for _, values in group):
                raise ValueError("Coverage metrics differ between paired action groups")
            query_values.append([np.mean([values[key] for _, values in group]) for key in keys])
            episodes.append(next(iter(episode_ids)))
            seed_counts.append(len(seeds))
        values = np.asarray(query_values, dtype=np.float64)
        if not np.isfinite(values).all():
            raise ValueError("Nonfinite coverage statistic")
        episode_ids = np.unique(episodes)
        by_episode = np.asarray([values[np.asarray(episodes) == episode].mean(0) for episode in episode_ids])
        intervals = None
        samples = config["evaluation"]["bootstrap_samples"]
        if len(episode_ids) > 1 and samples:
            rng = np.random.default_rng(config["evaluation"]["bootstrap_seed"])
            draws = rng.integers(len(episode_ids), size=(samples, len(episode_ids)))
            weights = np.zeros((samples, len(episode_ids)), dtype=np.float64)
            np.add.at(weights, (np.arange(samples)[:, None], draws), 1.)
            bootstraps = weights @ by_episode / len(episode_ids)
            intervals = np.quantile(bootstraps, [.025, .975], axis=0)
        estimates = {}
        for column, key in enumerate(keys):
            estimates[key] = {"mean": float(by_episode[:, column].mean()),
                              "mean_ci95": intervals[:, column].tolist() if intervals is not None else None,
                              "query_mean_median": float(np.median(values[:, column])),
                              "query_mean_p90": float(np.quantile(values[:, column], .9))}
        summary[mode] = {"queries": len(queries), "episodes": len(episode_ids),
                         "groups": sum(seed_counts), "seeds_per_query": sorted(set(seed_counts)),
                         "action_pairs": sum(len(item[0]["pairs"]) for group in queries.values() for item in group),
                         "statistics": estimates,
                         "aggregation": "equal action weight per group; equal seed weight per query; equal window weight per episode; equal episode weight",
                         "uncertainty": "paired source-episode percentile bootstrap; pointwise 95% CIs, no multiplicity adjustment",
                         "episode_means": {str(ep): dict(zip(keys, row.tolist())) for ep, row in zip(episode_ids, by_episode)}}
    return summary


def write_tables(summary: dict, config: dict, directory: Path) -> None:
    directory = Path(directory)
    with (directory / "summary.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["mode", "metric", "mean", "ci95_low", "ci95_high", "query_mean_median", "query_mean_p90", "episodes", "queries"])
        for mode, group in summary.items():
            for key, row in group["statistics"].items():
                interval = row["mean_ci95"] or [None, None]
                writer.writerow([mode, key, row["mean"], *interval, row["query_mean_median"], row["query_mean_p90"], group["episodes"], group["queries"]])
    lines = ["# Frozen-model rank coverage", "", "Coverage only: no adaptation, CEM or query-target scoring.", ""]
    for mode, group in summary.items():
        lines.extend([f"## {mode}", "", f"{group['queries']} queries / {group['episodes']} source episodes / {group['action_pairs']} action pairs.", "",
                      "| Opposite top-p% | Mutual pool, mean count | Consensus full-B fraction | FR sample retention | FR channel loss mass retained |", "|---:|---:|---:|---:|---:|"])
        for percent in config["percentages"]:
            stats = group["statistics"]
            prefix = f"p/{percent:g}"
            value = lambda suffix: stats[f"{prefix}/{suffix}"]["mean"]
            lines.append(f"| {percent:g} | {value('pair/mutual_pool_count'):.1f} | {value('pair/consensus_full_batch'):.1%} | {value('fr/unique_retained_fraction'):.1%} | {value('fr/channel_loss_mass_retained'):.1%} |")
        lines.extend(["", "CIs and directional top-k coverage are in `summary.csv` and `report.json`. P3 consensus metrics average per-action diagnostics; they do not define a shared P3 consensus selector.", ""])
    (directory / "summary.md").write_text("\n".join(lines) + "\n")

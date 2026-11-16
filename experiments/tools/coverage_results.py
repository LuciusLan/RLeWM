#!/usr/bin/env python3
"""Audit saved coverage results and export statistics/figures, without model imports."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
from pathlib import Path

import numpy as np

from lewm_tta.rank_coverage.analysis import validate_matrix
from lewm_tta.rank_coverage.metrics import cutoff, group_metrics
from lewm_tta.rank_coverage.plotting import plot as plot_overview
from lewm_tta.rank_coverage.statistics import summarize
from lewm_tta.runtime import file_hash, read_json, write_json


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_REPORT = ROOT / "artifacts/pilot-rank-coverage-v64-t150-q4/runs/recorded-q4/report.json"


def audit(report_path: Path):
    report = read_json(report_path)
    if report.get("stage") != "rank-coverage" or not report.get("completed"):
        raise ValueError("A completed coverage report is required")
    records_path = Path(report["records_file"])
    if file_hash(records_path) != report["records_sha256"]:
        raise ValueError("Coverage record checksum differs from the report")
    with gzip.open(records_path, "rt", encoding="utf-8") as stream:
        records = [json.loads(line) for line in stream if line.strip()]
    if len(records) != report["record_count"]:
        raise ValueError("Record count differs from the report")
    matrix = validate_matrix(records, report["config"])
    if matrix["memory_windows"] != report["memory_windows"]:
        raise ValueError("Corpus size differs from the report")
    if sorted({row["query"]["source_episode_position"] for row in records}) != report["query_episode_positions"]:
        raise ValueError("Query episodes differ from the report")
    if summarize(records, report["config"]) != report["summary"]:
        raise ValueError("Summary or bootstrap intervals do not reproduce exactly")
    # Check saved support metadata independently of the runtime acceptance flags.
    settings = report["protocol"]["p1_retrieval"]
    cap, gap = settings["max_windows_per_episode"], settings["min_start_step_gap"]
    for record in records:
        query_episode = record["query"]["source_episode_position"]
        supports = [record["fr_fixed_support"]] + [pair["consensus_sources"] for pair in record["pairs"]]
        for support in supports:
            indices = [row["index"] for row in support]
            if len(indices) != len(set(indices)) or any(not 0 <= i < report["memory_windows"] for i in indices):
                raise ValueError("Repeated or invalid support indices")
            episodes = [row["source_episode_position"] for row in support]
            if query_episode in episodes:
                raise ValueError("Query episode appears in selected support")
            for episode in set(episodes):
                steps = sorted(row["start_in_episode"] for row in support if row["source_episode_position"] == episode)
                if (cap is not None and len(steps) > cap) or any(b - a < gap for a, b in zip(steps, steps[1:])):
                    raise ValueError("Support diversity constraints are violated")
        for pair in record["pairs"]:
            expected = {row[0] for row in pair["consensus_support"]}
            if expected != {row["index"] for row in pair["consensus_sources"]}:
                raise ValueError("Consensus ranks and source metadata disagree")
            for direction, position in (("forward", 1), ("reverse", 2)):
                prefix = pair[f"{direction}_prefix"]
                if [row[position] for row in prefix] != list(range(1, len(prefix) + 1)):
                    raise ValueError("Saved prefixes are not consecutive exact own-ranks")
    return report, records, {**matrix, "records_checksum_verified": True,
        "saved_statistics_and_cis_reproduce_exactly": True, "support_metadata_diversity_verified": True}


def save_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def recorded_diagnostics(report: dict, records: list[dict]) -> tuple[list[dict], list[dict], dict]:
    """Per-query counts are defined only for the single-seed recorded-action mode."""
    records = [row for row in records if row["mode"] == "recorded-action"]
    if not records:
        return [], [], {}
    config, size = report["config"], report["memory_windows"]
    values = [group_metrics(row, config["prefix_sizes"], config["percentages"], config["support_size"]) for row in records]
    stats = report["summary"]["recorded-action"]["statistics"]
    thresholds = []
    for percent in config["percentages"]:
        key = f"p/{percent:g}/"
        mean = lambda metric: stats[key + metric]["mean"]
        row = {"percent": percent, "rank_cutoff": cutoff(size, percent),
            "mean_mutual_pool_count": mean("pair/mutual_pool_count"),
            "mean_mutual_overlap_fraction": mean("pair/mutual_top_percent_coverage"),
            "independent_rank_expected_intersection": cutoff(size, percent)**2 / size,
            "raw_full_budget_queries": int(sum(v[key + "pair/mutual_pool_has_budget"] for v in values)),
            "consensus_full_budget_queries": int(sum(v[key + "pair/consensus_full_batch"] for v in values)),
            "consensus_zero_support_queries": int(sum(v[key + "pair/consensus_zero_support"] for v in values)),
            "consensus_retained_mean_count": mean("pair/consensus_retained_count"),
            "consensus_original_samples_removed": int(round(sum(config["support_size"] - v[key + "pair/consensus_retained_count"] for v in values))),
            "consensus_changed_episodes": len({r["query"]["source_episode_position"] for r, v in zip(records, values) if not v[key + "pair/consensus_full_batch"]}),
            "fr_retained_mean_count": mean("fr/unique_retained_count"),
            "fr_sample_retained_fraction": mean("fr/unique_retained_fraction"),
            "fr_original_sample_weight_retained": mean("fr/original_sample_weight_retained"),
            "fr_channel_loss_mass_retained": mean("fr/channel_loss_mass_retained"),
            "fr_full_budget_queries": int(sum(v[key + "fr/all_original_samples_survive"] for v in values)),
            "fr_zero_support_queries": int(sum(v[key + "fr/zero_support"] for v in values))}
        row["raw_full_but_diversity_shortfall_queries"] = row["raw_full_budget_queries"] - row["consensus_full_budget_queries"]
        for metric in ("pair/consensus_full_batch", "fr/unique_retained_fraction", "fr/channel_loss_mass_retained",
                       "topk/32/forward_coverage", "topk/32/reverse_coverage", "topk/32/forward_minus_reverse_coverage"):
            label = metric.replace("/", "_")
            row[label] = mean(metric)
            row[label + "_ci_low"], row[label + "_ci_high"] = stats[key + metric]["mean_ci95"] or (None, None)
        thresholds.append(row)
    diagnostics = []
    for record, value in zip(records, values):
        query, pair = record["query"], record["pairs"][0]
        row = {key: query[key] for key in ("transition_id", "source_episode_position", "episode_id", "start_in_episode")}
        row.update({"consensus_cover_all_percent": value["pair/consensus_cover_all_rank_percent"],
                    "fr_cover_all_percent": value["fr/cover_all_samples_opposite_rank_percent"],
                    "fr_cover_all_channel_mass_percent": value["fr/cover_all_channel_mass_rank_percent"]})
        for direction, opposite in (("forward", 2), ("reverse", 1)):
            row[f"{direction}_top32_max_opposite_rank"] = max(item[opposite] for item in pair[f"{direction}_prefix"][:32])
        diagnostics.append(row)
    fractions = [0, .5, .75, .9, .95, .99, 1]
    distributions = {key: {f"q{100 * q:g}": float(v) for q, v in zip(fractions, np.quantile([row[key] for row in diagnostics], fractions))}
        for key in ("consensus_cover_all_percent", "fr_cover_all_percent", "fr_cover_all_channel_mass_percent")}
    return thresholds, diagnostics, distributions


def figures(report: dict, diagnostics: list[dict], directory: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
        "pdf.fonttype": 42, "ps.fonttype": 42, "axes.spines.top": False, "axes.spines.right": False})
    plot_overview(Path(report["_report_path"]), directory / "coverage-overview")

    def save(figure, stem):
        figure.savefig(directory / f"{stem}.png", dpi=240)
        figure.savefig(directory / f"{stem}.pdf")
        plt.close(figure)

    for mode, summary in report["summary"].items():
        stats = summary["statistics"]
        percentages = [p for p in report["config"]["percentages"] if p < 100]
        prefixes = report["config"]["prefix_sizes"]
        fig, axes = plt.subplots(1, 2, figsize=(13, 4.1), constrained_layout=True, sharey=True)
        for axis, direction, title in zip(axes, ("forward", "reverse"),
                ("Forward top-L passing the reverse gate", "Reverse top-L passing the forward gate")):
            matrix = np.array([[stats[f"p/{p:g}/topk/{k}/{direction}_coverage"]["mean"] for p in percentages] for k in prefixes])
            handle = axis.imshow(matrix, aspect="auto", vmin=0, vmax=1, cmap="YlGnBu")
            for row in range(len(prefixes)):
                for col in range(len(percentages)):
                    axis.text(col, row, f"{100 * matrix[row, col]:.1f}", ha="center", va="center",
                              fontsize=8, color="white" if matrix[row, col] > .64 else "black")
            axis.set(xticks=range(len(percentages)), xticklabels=[f"{p:g}" for p in percentages],
                     yticks=range(len(prefixes)), yticklabels=prefixes, xlabel="Opposite top-p (%)", title=title)
        axes[0].set_ylabel("Raw own-direction prefix L")
        fig.colorbar(handle, ax=axes, label="Episode-averaged retained fraction", shrink=.85)
        fig.suptitle(f"{mode}: directional coverage (%) | {summary['queries']} contexts / {summary['episodes']} episodes")
        save(fig, f"directional-coverage-{mode}")

        fig, axes = plt.subplots(1, 2, figsize=(11, 4.1), constrained_layout=True)
        p = np.array(percentages, dtype=float)
        rows = [stats[f"p/{v:g}/pair/mutual_top_percent_coverage"] for v in p]
        means = np.array([row["mean"] for row in rows])
        ci = np.array([row["mean_ci95"] or [row["mean"], row["mean"]] for row in rows])
        axes[0].plot(p, means, "o-", color="C0", label="Observed overlap / T")
        axes[0].fill_between(p, ci[:, 0], ci[:, 1], color="C0", alpha=.15)
        null_fraction = np.array([cutoff(report["memory_windows"], v) / report["memory_windows"] for v in p])
        axes[0].plot(p, null_fraction, "--", color="0.4", label="Independent-rank reference T/N")
        axes[0].set(xscale="log", yscale="log", xlabel="Common top-p (%)", ylabel="Intersection size / T",
                    title="Whole-corpus agreement is substantial")
        axes[0].legend(fontsize=8, loc="lower right")
        axes[1].plot(p, [stats[f"p/{v:g}/pair/mutual_pool_has_budget"]["mean"] for v in p],
                     "o--", color="0.5", label="At least B raw mutual windows")
        axes[1].plot(p, [stats[f"p/{v:g}/pair/consensus_full_batch"]["mean"] for v in p],
                     "o-", color="C2", label="Full B after diversity constraints")
        axes[1].set(xscale="log", ylim=(-.02, 1.02), xlabel="Common top-p (%)", ylabel="Fraction of contexts",
                    title="Raw pool size does not ensure B=32")
        axes[1].yaxis.set_major_formatter(PercentFormatter(1))
        axes[1].legend(fontsize=8, loc="lower right")
        for axis in axes:
            axis.grid(True, alpha=.2)
        fig.suptitle(f"{mode}: exact ranks over N={report['memory_windows']:,}; reference is descriptive")
        save(fig, f"mutual-overlap-{mode}")

    if diagnostics:
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.3), constrained_layout=True)
        consensus = np.array([row["consensus_cover_all_percent"] for row in diagnostics])
        fr = np.array([row["fr_cover_all_percent"] for row in diagnostics])
        for values, color, label in ((consensus, "C2", "Complete original consensus B32"),
                                     (fr, "C3", "All original FR-fixed samples")):
            ordered = np.sort(values)
            axes[0].step(np.r_[1e-3, ordered, 100.], np.r_[0., np.arange(1, len(values) + 1) / len(values), 1.],
                         where="post", color=color, label=label)
        for threshold in (.03, .1):
            axes[0].axvline(threshold, color="0.5", lw=.8, ls=":")
            axes[0].text(threshold, .18, f" {threshold:g}%", fontsize=8, rotation=90, va="bottom")
        axes[0].set(xscale="log", xlim=(1e-3, 100), ylim=(0, 1.03), xlabel="Opposite top-p (%)",
                    ylabel="Fraction of recorded-action contexts", title="Largest required rank across contexts")
        axes[0].set_xlabel("Largest required rank (% of corpus)")
        axes[0].yaxis.set_major_formatter(PercentFormatter(1))
        axes[0].legend(fontsize=8, loc="lower right")
        axes[1].scatter(consensus, fr, s=20, color="C0", alpha=.55, edgecolors="none")
        reference = np.geomspace(1e-3, 100, 100)
        axes[1].plot(reference, reference, color="0.5", lw=1, ls="--", label="Equal largest rank percentages")
        axes[1].set(xscale="log", yscale="log", xlim=(1e-3, 100), ylim=(1e-3, 100),
                    xlabel="Consensus largest rank (% of corpus)", ylabel="FR-fixed largest opposite rank (% of corpus)",
                    title="Rare FR candidates have very large opposite ranks")
        axes[1].legend(fontsize=8, loc="lower right")
        for transition, offset in (("pusht:14029:82", (12, -25)), ("pusht:14029:86", (10, -15))):
            for row in diagnostics:
                if row["transition_id"] == transition:
                    axes[1].annotate(transition, (row["consensus_cover_all_percent"], row["fr_cover_all_percent"]),
                        xytext=offset, textcoords="offset points", fontsize=8,
                        arrowprops={"arrowstyle": "-", "color": "0.4", "lw": .7})
        for axis in axes:
            axis.grid(True, alpha=.2)
        episodes = len({row["source_episode_position"] for row in diagnostics})
        fig.suptitle(f"Recorded-action support tails: {len(diagnostics)} contexts / {episodes} episodes; descriptive distributions")
        save(fig, "support-threshold-tails-recorded-action")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    path = args.report.resolve()
    report, records, checks = audit(path)
    output = args.output.resolve() if args.output else path.parent.parent.parent / f"analysis-{path.parent.name}"
    output.mkdir(parents=True, exist_ok=False)
    figure_dir = output / "figures"
    figure_dir.mkdir()
    thresholds, diagnostics, distributions = recorded_diagnostics(report, records)
    if thresholds:
        save_csv(output / "thresholds.csv", thresholds)
        save_csv(output / "query-diagnostics.csv", diagnostics)
    result = {"source_report": str(path), "source_report_sha256": file_hash(path),
        "records_sha256": report["records_sha256"], "analysis_script_sha256": file_hash(Path(__file__)),
        "audit": checks, "thresholds": thresholds, "across_query_required_threshold_quantiles_percent": distributions,
        "tail_examples": sorted(diagnostics, key=lambda row: row["fr_cover_all_percent"], reverse=True)[:10],
        "statistics": "Original episode-weighted estimates and 2,000 paired episode-bootstrap CIs reproduce exactly; threshold quantiles/scatter are descriptive across contexts",
        "scope": "Frozen recorded/provisional inputs present in the report only; no prediction quality or planning effect is inferred"}
    write_json(output / "analysis.json", result)
    figures({**report, "_report_path": str(path)}, diagnostics, figure_dir)
    print("Audited saved coverage records; analysis and figures:", output)


if __name__ == "__main__":
    main()

"""Standalone PNG/PDF coverage curves from saved estimates; no model imports."""

from pathlib import Path

import numpy as np

from ..runtime import read_json


def plot(report_path: Path, output: Path | None = None, prefix_size: int = 32) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    report = read_json(Path(report_path))
    if report.get("stage") != "rank-coverage" or not report.get("completed"):
        raise ValueError("Plotting requires a completed rank-coverage report")
    if prefix_size not in report["config"]["prefix_sizes"]:
        raise ValueError("Plot prefix size must be included in the saved protocol")
    target = Path(output) if output else Path(report_path).parent / "coverage"
    if any(Path(f"{target}-{mode}{suffix}").exists()
           for mode in report["summary"] for suffix in (".png", ".pdf")):
        raise FileExistsError("Coverage plot already exists; choose another output stem")
    percentages = np.asarray(report["config"]["percentages"], dtype=float)
    target.parent.mkdir(parents=True, exist_ok=True)
    for mode, group in report["summary"].items():
        fig, axes = plt.subplots(1, 3, figsize=(12, 3.5), constrained_layout=True)
        panels = [
            [(f"topk/{prefix_size}/forward_coverage", "Forward candidates in reverse", "C0"),
             (f"topk/{prefix_size}/reverse_coverage", "Reverse candidates in forward", "C1")],
            [("pair/consensus_full_batch", "Full B after diversity constraints", "C2"),
             ("pair/mutual_pool_has_budget", "Raw intersection has at least B", "C7")],
            [("fr/unique_retained_fraction", "Original FR samples retained", "C3"),
             ("fr/channel_loss_mass_retained", "Original FR channel mass retained", "C4")]]
        for axis, curves, title in zip(axes, panels, [f"Directional top-{prefix_size}", "Consensus coverage", "FR-fixed retention without refill"]):
            for metric, label, color in curves:
                rows = [group["statistics"][f"p/{p:g}/{metric}"] for p in percentages]
                means = np.asarray([row["mean"] for row in rows])
                axis.plot(percentages, means, marker="o", markersize=3, color=color, label=label)
                if all(row["mean_ci95"] is not None for row in rows):
                    ci = np.asarray([row["mean_ci95"] for row in rows])
                    axis.fill_between(percentages, ci[:, 0], ci[:, 1], color=color, alpha=.15)
            axis.set(xscale="log", ylim=(-.02, 1.02), xlabel="Opposite-direction top-p (%)", title=title)
            axis.grid(True, alpha=.2)
            axis.legend(fontsize=7, loc="lower right")
        axes[0].set_ylabel("Episode-averaged fraction")
        fig.suptitle(f"{mode}: {group['queries']} contexts / {group['episodes']} episodes; pointwise 95% CIs", fontsize=10)
        stem = f"{target}-{mode}"
        fig.savefig(f"{stem}.png", dpi=220)
        fig.savefig(f"{stem}.pdf")
        plt.close(fig)
        print("Coverage figures:", f"{stem}.pdf", flush=True)

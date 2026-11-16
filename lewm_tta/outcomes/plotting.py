"""Standalone publication-editable success/budget figures from saved records."""

from pathlib import Path

from .pipeline import verified_report


def plot(path, directory):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    report = verified_report(path)
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    curves = report["summary"]["curves"]
    figure, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    for label in sorted({point["condition"] for point in curves}):
        points = sorted((point for point in curves if point["condition"] == label), key=lambda point: point["iterations"])
        rates = [point["final_success_rate"]["mean"] for point in points]
        for axis, coordinate in zip(axes, ("iterations", "planning_seconds")):
            x = [point[coordinate] if coordinate == "iterations" else point[coordinate]["mean"] for point in points]
            line, = axis.plot(x, rates, "o-", label=label)
            for xi, point in zip(x, points):
                interval = point["final_success_rate"]["ci95"]
                if interval is not None:
                    # Bootstrap percentile CIs need not enclose the point estimate.
                    axis.vlines(xi, *interval, color=line.get_color(), alpha=.65)
            axis.set(ylim=(-.02, 1.02), ylabel="Final physical success rate")
            axis.grid(alpha=.2)
    axes[0].set_xlabel("CEM iterations J")
    axes[1].set_xlabel("Mean measured planning time (s)")
    axes[1].legend(fontsize="small")
    title = report["metric_contract"]["backend"]+" · fixed-horizon supplied-goal execution"
    figure.suptitle(title+(" · LIMITED RUN" if report["limited_run"] else ""))
    for extension in ("png", "pdf"):
        figure.savefig(directory/f"success-tradeoff.{extension}", dpi=200)
    plt.close(figure)

"""Explicit physical saved-plan replay; never runs a planner or predictor."""

import argparse
import os
from pathlib import Path
import sys

from ..runtime import configure_logging, read_json


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0].startswith("--") and argv[0] != "--help":
        argv.insert(0, "evaluate")
    parser = argparse.ArgumentParser(description="P3 physical success and planning-budget tradeoffs")
    commands = parser.add_subparsers(dest="command", required=True)
    evaluate = commands.add_parser("evaluate")
    evaluate.add_argument("--reports", nargs="+", required=True, type=Path)
    evaluate.add_argument("--output-dir", required=True, type=Path)
    evaluate.add_argument("--conditions", nargs="+")
    evaluate.add_argument("--limit", type=int)
    evaluate.add_argument("--pusht-success", choices=("native-goal", "coverage"), default="native-goal")
    evaluate.add_argument("--coverage-threshold", type=float, default=.95)
    evaluate.add_argument("--state-atol", type=float, default=1e-5)
    evaluate.add_argument("--rgb-mae-atol", type=float, default=.1)
    evaluate.add_argument("--success-targets", nargs="+", type=float, default=[.5, .8, .9])
    evaluate.add_argument("--deadlines", nargs="+", type=float, default=[.1, .5, 1., 2., 5., 10.])
    evaluate.add_argument("--bootstrap-samples", type=int, default=2000)
    evaluate.add_argument("--bootstrap-seed", type=int, default=20261113)
    evaluate.add_argument("--dry-run", action="store_true")
    for name in ("analyze", "plot"):
        command = commands.add_parser(name)
        command.add_argument("--report", type=Path, required=True)
        command.add_argument("--output" if name == "analyze" else "--output-dir", type=Path, required=name == "plot")
    args = parser.parse_args(argv)
    configure_logging()
    if args.command in ("analyze", "plot"):
        from experiments.provenance import replay_recorded_code
        replay_recorded_code("p3_success", read_json(args.report), argv)
        if args.command == "analyze":
            from .pipeline import analyze
            analyze(args.report, args.output)
        else:
            from .plotting import plot
            plot(args.report, args.output_dir)
        return
    from math import isfinite
    settings = {name: getattr(args, name) for name in ("pusht_success", "coverage_threshold", "state_atol",
                "rgb_mae_atol", "success_targets", "deadlines", "bootstrap_samples", "bootstrap_seed")}
    if (args.limit is not None and args.limit < 1) or args.bootstrap_samples < 0 or args.bootstrap_seed < 0:
        raise ValueError("Require positive limit and nonnegative bootstrap counts/seed")
    if (not 0 < args.coverage_threshold <= 1 or not 0 <= args.state_atol < float("inf")
            or not 0 <= args.rgb_mae_atol < float("inf")
            or any(not isfinite(x) or not 0 < x <= 1 for x in args.success_targets)
            or any(not isfinite(x) or x <= 0 for x in args.deadlines)):
        raise ValueError("Require finite tolerances >=0, thresholds/targets in (0,1], deadlines >0")
    settings["success_targets"] = sorted(set(args.success_targets))
    settings["deadlines"] = sorted(set(args.deadlines))
    if args.dry_run:
        import json
        print(json.dumps({"reports": [str(p.resolve()) for p in args.reports], "output": str(args.output_dir.resolve()),
                          "settings": settings, "conditions": args.conditions, "limit": args.limit,
                          "note": "Plan only: no input verification, writes, simulator or model execution"}, indent=2))
        return
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    os.environ.setdefault("MUJOCO_GL", "egl")
    from .pipeline import evaluate as run
    run(args.reports, args.output_dir, settings, args.conditions, args.limit)

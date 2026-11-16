"""Opt-in episode and single-plan GIF/contact-sheet inspection."""

import argparse
from pathlib import Path
import traceback

from ..config import ROOT
from ..collection.config import headless
from ..runtime import configure_logging


def parser():
    root = argparse.ArgumentParser(description=__doc__)
    commands = root.add_subparsers(dest="command", required=True)
    for name in ("collection", "cem"):
        command = commands.add_parser(name)
        command.add_argument("--collection-config", type=Path, default=ROOT/"codes/configs/collect/default.json")
        command.add_argument("--episodes", "--episode", nargs="+", type=int, default=[0])
        command.add_argument("--split", choices=("validation", "test"), default="validation")
        command.add_argument("--regime", default="source")
        position = command.add_mutually_exclusive_group()
        position.add_argument("--start-step", type=int)
        position.add_argument("--window-index", type=int)
        command.add_argument("--every", type=int, default=1, help="Keep every Nth frame, preserving playback speed and the final frame")
        command.add_argument("--scale", type=int, default=2, help="Integer display magnification (1-4)")
        command.add_argument("--output", type=Path)
        command.add_argument("--verbose", action="store_true")
        if name == "collection":
            command.add_argument("--goal-protocol", help="Declared observed-future goal preset; e.g. lewm-paper for TwoRoom +100")
        if name == "cem":
            command.add_argument("--config", type=Path, default=ROOT/"codes/configs/sim/fresh.json", help="Planner profile; its fresh cache paths are not needed for this preview")
            command.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
            command.add_argument("--cpu-threads", type=int, default=8)
            command.add_argument("--seed", type=int, default=42)
            command.add_argument("--method", choices=("frozen", "uniform", "fr-fixed"), default="frozen")
            command.add_argument("--proposal", choices=("gaussian", "context-fragments"), default="gaussian")
    green = commands.add_parser("green", help="Render saved green-target MPC execution; simulator only, no learned model")
    green.add_argument("--report", type=Path, required=True)
    green.add_argument("--episodes", "--episode", type=int, nargs="+", default=[0])
    green.add_argument("--method", "--methods", choices=("frozen", "uniform", "fr-fixed"), nargs="+", default=["frozen"])
    green.add_argument("--proposal", "--proposals", choices=("gaussian", "context-fragments"), nargs="+", default=["gaussian"])
    green.add_argument("--seed", type=int, help="One evaluated seed; defaults to the first saved seed")
    green.add_argument("--every", type=int, default=1)
    green.add_argument("--scale", type=int, default=2)
    green.add_argument("--output", type=Path)
    green.add_argument("--verbose", action="store_true")
    audit = commands.add_parser("green-audit", help="Diagnose saved validation Frozen/Gaussian plans against real physics; runs the frozen model and CEM")
    audit.add_argument("--report", type=Path, required=True)
    audit.add_argument("--episodes", "--episode", type=int, nargs="+", default=[0])
    audit.add_argument("--replans", type=int, nargs="+", default=[0])
    audit.add_argument("--seed", type=int)
    audit.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    audit.add_argument("--cpu-threads", type=int, default=8)
    audit.add_argument("--refine-iterations", type=int, default=30)
    audit.add_argument("--refine-population", type=int, default=300)
    audit.add_argument("--every", type=int, default=1)
    audit.add_argument("--scale", type=int, default=2)
    audit.add_argument("--output", type=Path)
    audit.add_argument("--verbose", action="store_true")
    mpc = commands.add_parser("green-mpc", help="Run only Frozen/Gaussian green-target MPC with explicit planning/goal settings")
    mpc.add_argument("--config", type=Path, default=ROOT/"codes/configs/sim/green-lewm.json")
    mpc.add_argument("--episodes", "--episode", type=int, nargs="+", default=[0])
    mpc.add_argument("--seed", type=int)
    mpc.add_argument("--context-frames", type=int, choices=(1, 3), default=1,
                     help="Initial observed predictor context; imagined history then grows up to H3. Native policy defaults to 1; existing experiments use 3")
    mpc.add_argument("--goal-agent-position", type=float, nargs=2, metavar=("X", "Y"),
                     help="Override only the fixed goal-image pusher, in native 512-scene coordinates")
    mpc.add_argument("--execute-blocks", type=int, choices=(1, 2, 3, 4, 5))
    mpc.add_argument("--max-raw-steps", type=int)
    mpc.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    mpc.add_argument("--cpu-threads", type=int, default=8)
    mpc.add_argument("--every", type=int, default=2)
    mpc.add_argument("--scale", type=int, default=2)
    mpc.add_argument("--output", type=Path)
    mpc.add_argument("--verbose", action="store_true")
    return root


def main():
    args = parser().parse_args()
    configure_logging(args.verbose)
    headless()
    try:
        if args.every < 1 or not 1 <= args.scale <= 4:
            raise ValueError("every must be positive; scale must be from 1 to 4")
        if getattr(args, "cpu_threads", 1) < 1 or (getattr(args, "seed", None) is not None and args.seed < 0):
            raise ValueError("cpu-threads must be positive and seed nonnegative")
        if args.command == "collection":
            from .episodes import preview
            preview(args.collection_config, args.split, args.episodes, args.regime, args.output or "/home/ubuntu/project/preview",
                    start_step=args.start_step, window_index=args.window_index, every=args.every, scale=args.scale,
                    goal_protocol=args.goal_protocol)
        elif args.command == "green":
            from .green import preview
            preview(args)
        elif args.command == "green-audit":
            if (not args.replans or min(args.replans) < 0 or len(set(args.replans)) != len(args.replans)
                    or args.refine_iterations < 1 or args.refine_population < 2):
                raise ValueError("Choose distinct nonnegative replans and positive CEM iterations/population")
            from .green_audit import audit
            audit(args)
        elif args.command == "green-mpc":
            from .green_mpc import preview
            preview(args)
        else:
            from .plans import preview
            preview(args)
    except Exception:
        traceback.print_exc()
        raise SystemExit(1)

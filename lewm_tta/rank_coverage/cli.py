"""Independent coverage pilot: prepare metadata, evaluate retrieval, analyze or plot saved records."""

import argparse
import faulthandler
from pathlib import Path
import sys
import traceback

from ..config import ROOT
from ..runtime import configure_logging
from .config import MODES, load_config, prepare, validate_config


def main() -> None:
    faulthandler.enable(all_threads=True)
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("prepare", "evaluate"):
        command = commands.add_parser(name)
        command.add_argument("--config", type=Path, help="Resolved by codes/run.sh")
        command.add_argument("--verbose", action="store_true")
    evaluate = commands.choices["evaluate"]
    evaluate.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    evaluate.add_argument("--split", choices=("validation", "test"), default="validation")
    evaluate.add_argument("--run-name")
    evaluate.add_argument("--limit", type=int)
    evaluate.add_argument("--query-modes", nargs="+", choices=MODES)
    evaluate.add_argument("--seeds", nargs="+", type=int)
    evaluate.add_argument("--percentages", nargs="+", type=float, help="Percent units: 0.1 means top 0.1%%")
    evaluate.add_argument("--validation-report", type=Path)
    analysis = commands.add_parser("analyze")
    analysis.add_argument("--report", required=True, type=Path)
    analysis.add_argument("--output", type=Path)
    analysis.add_argument("--verbose", action="store_true")
    figures = commands.add_parser("plot")
    figures.add_argument("--report", required=True, type=Path)
    figures.add_argument("--output", type=Path, help="New output filename stem")
    figures.add_argument("--prefix-size", type=int, default=32)
    figures.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    configure_logging(args.verbose)
    try:
        if args.command == "analyze":
            from .analysis import analyze
            analyze(args.report, args.output)
            return
        if args.command == "plot":
            from .plotting import plot
            plot(args.report, args.output, args.prefix_size)
            return
        if args.config is None:
            raise ValueError("Use bash codes/run.sh coverage COMMAND")
        value = load_config(args.config)
        if args.command == "prepare":
            prepare(value)
            return
        if args.limit is not None and args.limit < 1:
            raise ValueError("Query limit must be positive")
        for key in ("query_modes", "seeds", "percentages"):
            if getattr(args, key) is not None:
                value[key] = getattr(args, key)
        validate_config(value)
        from .pipeline import evaluate as run
        run(value, args.device, args.split, args.run_name, args.limit, args.validation_report)
    except Exception as error:
        if args.verbose:
            traceback.print_exc()
        else:
            print(f"Rank coverage failed: {type(error).__name__}: {error}", file=sys.stderr)
            print("Use --verbose for the full traceback.", file=sys.stderr)
        raise SystemExit(1)

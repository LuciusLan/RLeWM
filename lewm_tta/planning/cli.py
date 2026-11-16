"""Thin P3-a CLI; simulation and future-action proposal controls are separate."""

import argparse
import faulthandler
import os
from pathlib import Path
import sys
import traceback

from ..config import ROOT
from ..runtime import configure_logging, read_json
from ..rollout.resources import checked_json
from ..rollout.provenance import replay_recorded_code
from ..rollout.record_io import RECORD_MODES
from .config import METHODS, load_config, validate_config


def main():
    faulthandler.enable(all_threads=True)
    parser = argparse.ArgumentParser(description="P3-a: offline shared-TTT CEM on cached held-out episodes")
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("prepare", "check", "evaluate"):
        command = commands.add_parser(name)
        command.add_argument("--config", type=Path, help="Resolved by codes/run.sh")
        command.add_argument("--experiment-dir", type=Path)
        command.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
        command.add_argument("--cpu-threads", type=int, default=8)
        command.add_argument("--verbose", action="store_true")
        command.add_argument("--trace-native", action="store_true")
    evaluate = commands.choices["evaluate"]
    evaluate.add_argument("--split", choices=("validation", "test"), default="validation")
    evaluate.add_argument("--locked-config", type=Path)
    evaluate.add_argument("--run-name")
    evaluate.add_argument("--methods", nargs="+", choices=METHODS)
    evaluate.add_argument("--learning-rate", type=float)
    evaluate.add_argument("--updates", type=int)
    evaluate.add_argument("--seeds", nargs="+", type=int)
    evaluate.add_argument("--limit", type=int)
    evaluate.add_argument("--records", choices=RECORD_MODES,
                          help="compact: gzip; full: JSONL; none: summaries only, without record-based locking")
    for name in ("analyze", "lock"):
        command = commands.add_parser(name)
        command.add_argument("--report", type=Path, required=True)
        command.add_argument("--output", type=Path, required=name == "lock")
        command.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    configure_logging(args.verbose)
    if getattr(args, "trace_native", False):
        os.environ["LEWM_CUVS_TRACE"] = "1"
    try:
        if args.command in ("analyze", "lock"):
            replay_recorded_code("p3a", read_json(args.report), sys.argv[1:])
            from .pipeline import analyze, lock_validation
            if args.command == "lock":
                lock_validation(args.report, args.output)
            else:
                analyze(args.report, args.output)
            return
        if args.cpu_threads < 1:
            raise ValueError("cpu_threads must be positive")
        lock = None
        if args.command == "evaluate" and args.locked_config:
            overrides = ("experiment_dir", "methods", "learning_rate", "updates", "seeds", "limit", "records")
            if args.split != "test" or any(getattr(args, name) is not None for name in overrides):
                raise ValueError("Use a locked configuration only for test, without experiment overrides")
            lock = checked_json(args.locked_config)
            replay_recorded_code("p3a", lock, sys.argv[1:])
            config = lock["config"]
        else:
            if args.config is None:
                raise ValueError("Use bash codes/run.sh p3 COMMAND --variant shared")
            config = load_config(args.config)
        if args.experiment_dir:
            config["experiment_dir"] = str(args.experiment_dir.resolve())
        if args.command == "evaluate":
            for key in ("methods", "learning_rate", "updates"):
                value = getattr(args, key)
                if value is not None:
                    config["adaptation"][key] = value
            if args.seeds is not None:
                config["p1"]["evaluation"]["seeds"] = args.seeds
            if args.limit is not None:
                config["evaluation"]["query_limit"] = args.limit
            if args.records is not None:
                config["evaluation"]["records"] = args.records
        validate_config(config)
        if args.command == "prepare":
            from .cohorts import prepare
            prepare(config)
        elif args.command == "check":
            from .pipeline import check
            check(config, args.device, args.cpu_threads)
        else:
            from .pipeline import evaluate as run
            run(config, args.device, args.cpu_threads, args.split, args.run_name, lock)
    except Exception as error:
        if args.verbose:
            traceback.print_exc()
        else:
            print(f"P3-a failed: {type(error).__name__}: {error}", file=sys.stderr)
        raise SystemExit(1)

"""Proposal experiment CLI; model and action-source choices are separate axes."""

import argparse
import faulthandler
import os
from pathlib import Path
import sys
import traceback

from ..config import ROOT
from ..runtime import configure_logging, read_json
from ..rollout.record_io import RECORD_MODES
from ..rollout.resources import checked_json
from ..planning.config import METHODS
from .config import SOURCES, load_config, validate_config
from .provenance import replay_recorded_code


def main(program="p3b"):
    if program not in ("p3b", "p3c"):
        raise ValueError("Unknown proposal experiment entrypoint")
    faulthandler.enable(all_threads=True)
    description = ("P3c: shared TTT x action-proposal factorial comparison" if program == "p3c" else
                   "P3b: reusable memory-action proposals for offline CEM")
    parser = argparse.ArgumentParser(description=description)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("prepare", "check", "evaluate"):
        command = commands.add_parser(name)
        command.add_argument("--config", type=Path)
        command.add_argument("--experiment-dir", type=Path)
        command.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
        command.add_argument("--cpu-threads", type=int, default=8)
        command.add_argument("--verbose", action="store_true")
        command.add_argument("--trace-native", action="store_true")
    evaluate = commands.choices["evaluate"]
    evaluate.add_argument("--split", choices=("validation", "test"), default="validation")
    evaluate.add_argument("--locked-config", type=Path)
    evaluate.add_argument("--run-name")
    evaluate.add_argument("--sources", nargs="+", choices=SOURCES)
    evaluate.add_argument("--injection-slots", type=int)
    evaluate.add_argument("--noise-std", type=float)
    evaluate.add_argument("--goal-weight", type=float)
    evaluate.add_argument("--methods", nargs="+", choices=METHODS)
    evaluate.add_argument("--lr", type=float)
    evaluate.add_argument("--updates", type=int)
    evaluate.add_argument("--seeds", nargs="+", type=int)
    evaluate.add_argument("--limit", type=int)
    evaluate.add_argument("--records", choices=RECORD_MODES)
    evaluate.add_argument("--no-reuse", action="store_true",
                          help="Run all cells afresh instead of importing configured Gaussian results")
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
            replay_recorded_code(program, read_json(args.report), sys.argv[1:])
            from .pipeline import analyze, lock_validation
            (lock_validation if args.command == "lock" else analyze)(args.report, args.output)
            return
        if args.cpu_threads < 1:
            raise ValueError("cpu_threads must be positive")
        lock = None
        if args.command == "evaluate" and args.locked_config:
            overrides = ("config", "experiment_dir", "sources", "injection_slots", "noise_std", "goal_weight",
                         "methods", "lr", "updates", "seeds", "limit", "records")
            if args.split != "test" or args.no_reuse or any(getattr(args, name) is not None for name in overrides):
                raise ValueError("Use the validation lock only for test, without overrides")
            lock = checked_json(args.locked_config)
            replay_recorded_code(program, lock, sys.argv[1:])
            config = lock["config"]
        else:
            if args.config is None:
                raise ValueError("Use bash codes/run.sh p3 COMMAND to select a stage profile")
            config = load_config(args.config)
        if args.experiment_dir:
            config["experiment_dir"] = str(args.experiment_dir.resolve())
        if args.command == "evaluate":
            if args.no_reuse:
                config.pop("reuse", None)
            for key in ("sources", "injection_slots", "noise_std", "goal_weight"):
                if getattr(args, key) is not None:
                    config["proposals"][key] = getattr(args, key)
            for flag, key in (("methods", "methods"), ("lr", "learning_rate"), ("updates", "updates")):
                if getattr(args, flag) is not None:
                    config["adaptation"][key] = getattr(args, flag)
            for key, destination in (("seeds", config["p1"]["evaluation"]), ("records", config["evaluation"])):
                if getattr(args, key) is not None:
                    destination[key] = getattr(args, key)
            if args.limit is not None:
                config["evaluation"]["query_limit"] = args.limit
        validate_config(config)
        if program == "p3c" and config["comparison"] != "factorial":
            raise ValueError("The P3c entrypoint requires a factorial config")
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
            print(f"{program.upper()} failed: {type(error).__name__}: {error}", file=sys.stderr)
        raise SystemExit(1)

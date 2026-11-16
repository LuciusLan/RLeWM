"""Shared command parsing for the thin recorded-rollout stage entrypoints."""

from __future__ import annotations

import argparse
import faulthandler
import os
from pathlib import Path
import sys
import traceback

from ..config import METHODS, ROOT
from ..runtime import configure_logging, read_json
from .config import P2_STAGE, SCOPE_STAGE, SCHEDULE_STAGE, SCOPES, entrypoint, load_config, validate_config
from .provenance import replay_recorded_code
from .record_io import RECORD_MODES


def main(program: str) -> None:
    if program not in ("p2", "p25", "p2_followup"):
        raise ValueError("Unknown rollout program")
    label = {"p2": "P2", "p25": "P2.5", "p2_followup": "P2 follow-up"}[program]
    faulthandler.enable(all_threads=True)
    parser = argparse.ArgumentParser(description=f"{label}: recorded-trajectory rollout evaluation")
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("prepare", "check", "evaluate"):
        command = commands.add_parser(name)
        command.add_argument("--config", type=Path, help="Resolved by codes/run.sh")
        command.add_argument("--experiment-dir", type=Path)
        command.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
        command.add_argument("--cpu-threads", type=int, default=8)
        command.add_argument("--verbose", action="store_true")
        command.add_argument("--trace-native", action="store_true")
        if program in ("p25", "p2_followup") and name != "prepare":
            command.add_argument("--scopes", nargs="+", choices=SCOPES)
        if program == "p2_followup" and name != "prepare":
            command.add_argument("--schedules", nargs="+", help="Names from the config schedule catalog")
    evaluate = commands.choices["evaluate"]
    evaluate.add_argument("--split", choices=("validation", "test"), default="validation")
    evaluate.add_argument("--locked-config", type=Path)
    evaluate.add_argument("--run-name")
    evaluate.add_argument("--methods", nargs="+", choices=METHODS)
    if program != "p2_followup":
        evaluate.add_argument("--updates", type=int)
    evaluate.add_argument("--learning-rate", type=float,
                          help="Global LR; explicit schedule learning_rate values take precedence")
    evaluate.add_argument("--seeds", nargs="+", type=int)
    evaluate.add_argument("--limit", type=int, help="Integration only; limited runs cannot be locked")
    evaluate.add_argument("--records", choices=RECORD_MODES,
                          help="compact: gzip without optional detailed diagnostics; full: JSONL; none: summaries only")
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
        if args.command in ("lock", "analyze"):
            report = read_json(args.report)
            expected_stage = {"p2": P2_STAGE, "p25": SCOPE_STAGE, "p2_followup": SCHEDULE_STAGE}[program]
            if report.get("stage") != expected_stage:
                raise ValueError(f"Expected a {label} report")
            replay_recorded_code(program, report, sys.argv[1:])
            from .pipeline import analyze, lock_validation
            if args.command == "lock":
                lock_validation(args.report, args.output)
            else:
                analyze(args.report, args.output)
            return
        if args.cpu_threads < 1:
            raise ValueError("cpu_threads must be positive")
        locked = None
        if args.command == "evaluate" and args.locked_config:
            overrides = ("scopes", "schedules", "methods", "updates", "learning_rate", "seeds", "limit", "records", "experiment_dir")
            if args.split != "test" or any(getattr(args, name, None) is not None for name in overrides):
                raise ValueError("Use a locked configuration only for test without experiment overrides")
            locked = read_json(args.locked_config)
            replay_recorded_code(program, locked, sys.argv[1:])
            config = locked["config"]
        else:
            if args.config is None:
                raise ValueError("Use bash codes/run.sh p2 COMMAND to select a stage profile")
            config = load_config(args.config)
        if args.experiment_dir:
            config["experiment_dir"] = str(args.experiment_dir.resolve())
        if getattr(args, "scopes", None) is not None:
            config["scopes"] = args.scopes
        if getattr(args, "schedules", None) is not None:
            config["schedule_names"] = args.schedules
        if args.command == "evaluate":
            for name in ("methods", "updates", "learning_rate"):
                value = getattr(args, name, None)
                if value is not None:
                    config["adaptation"][name] = value
            if args.seeds is not None:
                config["p1"]["evaluation"]["seeds"] = args.seeds
            if args.limit is not None:
                config["evaluation"]["query_limit"] = args.limit
            if args.records is not None:
                config["evaluation"]["records"] = args.records
        validate_config(config)
        if entrypoint(config) != program:
            raise ValueError(f"Use {entrypoint(config)}.py for this configuration")
        if args.command == "prepare":
            from .cohorts import prepare
            prepare(config)
        elif args.command == "check":
            from .pipeline import check
            check(config, args.device, args.cpu_threads)
        else:
            from .pipeline import evaluate as run_evaluation
            run_evaluation(config, args.device, args.cpu_threads, args.split, args.run_name, locked)
    except Exception as error:
        if args.verbose:
            traceback.print_exc()
        else:
            print(f"{label} failed: {type(error).__name__}: {error}", file=sys.stderr)
        raise SystemExit(1)

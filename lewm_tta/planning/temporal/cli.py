"""Stage-compatible CLI for the optional expensive temporal planning treatment."""

import argparse
import faulthandler
import os
from pathlib import Path
import sys
import traceback

from ...runtime import configure_logging, read_json
from ...rollout.provenance import replay_recorded_code
from ...rollout.record_io import RECORD_MODES
from ...rollout.resources import checked_json
from .config import MODES, SELECTORS, STAGE, load_config, override_seeds, validate_config


def main():
    faulthandler.enable(all_threads=True)
    parser = argparse.ArgumentParser(description="P3: shared-depth and isolated candidate-temporal CEM")
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
    evaluate.add_argument("--modes", nargs="+", choices=MODES)
    evaluate.add_argument("--selector", choices=SELECTORS)
    evaluate.add_argument("--persistence", choices=("trajectory", "position"))
    evaluate.add_argument("--learning-rate", type=float)
    evaluate.add_argument("--updates", type=int)
    evaluate.add_argument("--seeds", nargs="+", type=int,
                          help="Use these seeds for every mode, overriding the seeds_by_mode JSON policy")
    limit = evaluate.add_mutually_exclusive_group()
    limit.add_argument("--limit", type=int)
    limit.add_argument("--full-cohort", action="store_true", help="Explicitly remove the small pilot query limit")
    evaluate.add_argument("--records", choices=RECORD_MODES)
    for flag in ("population", "iterations", "elites", "reference-sequences"):
        evaluate.add_argument("--"+flag, type=int)
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
            source = read_json(args.report)
            if source.get("stage") != STAGE:
                raise ValueError("Expected a temporal planning report")
            replay_recorded_code("p3_temporal", source, sys.argv[1:])
            from .pipeline import analyze, lock_validation
            (lock_validation if args.command == "lock" else analyze)(args.report, args.output)
            return
        if args.cpu_threads < 1:
            raise ValueError("cpu_threads must be positive")
        lock = None
        if args.command == "evaluate" and args.locked_config:
            overrides = ("experiment_dir", "modes", "selector", "persistence", "learning_rate", "updates",
                         "seeds", "limit", "records", "population", "iterations", "elites", "reference_sequences")
            if args.split != "test" or args.full_cohort or any(getattr(args, key) is not None for key in overrides):
                raise ValueError("Use a validation lock only for test without experiment overrides")
            lock = checked_json(args.locked_config)
            replay_recorded_code("p3_temporal", lock, sys.argv[1:])
            config = lock["config"]
        else:
            if args.config is None:
                raise ValueError("Use bash codes/run.sh p3 COMMAND --variant temporal")
            config = load_config(args.config)
        if args.experiment_dir:
            config["experiment_dir"] = str(args.experiment_dir.resolve())
        if args.command == "evaluate":
            for key in ("modes", "selector", "persistence"):
                if getattr(args, key) is not None:
                    config["temporal"][key] = getattr(args, key)
            if args.selector is not None:
                config["adaptation"]["methods"] = ["frozen", args.selector]
            for key in ("learning_rate", "updates"):
                if getattr(args, key) is not None:
                    config["adaptation"][key] = getattr(args, key)
            for key in ("population", "iterations", "elites"):
                if getattr(args, key) is not None:
                    config["planning"][key] = getattr(args, key)
            if args.reference_sequences is not None:
                config["planning"]["provisional_actions"] = args.reference_sequences
            if args.seeds is not None:
                override_seeds(config, args.seeds)
            if args.full_cohort:
                config["evaluation"]["query_limit"] = None
            elif args.limit is not None:
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
            from .pipeline import evaluate
            evaluate(config, args.device, args.cpu_threads, args.split, args.run_name, lock)
    except Exception as error:
        if args.verbose:
            traceback.print_exc()
        else:
            print(f"P3 temporal failed: {type(error).__name__}: {error}", file=sys.stderr)
        raise SystemExit(1)

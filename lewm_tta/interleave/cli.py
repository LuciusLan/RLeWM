"""Explicit opt-in P3 interleave CLI; no changes to old P3 engine arguments."""

import argparse
import faulthandler
import os
from pathlib import Path
import sys
import traceback

from ..rollout.config import SCOPES
from ..rollout.record_io import RECORD_MODES
from ..rollout.resources import checked_json
from ..runtime import configure_logging, read_json
from .config import ARMS, SUPPORT_METHODS, ADAPTATION_SCHEDULES, load_config, validate_config


def main():
    faulthandler.enable(all_threads=True)
    if len(sys.argv) > 1 and sys.argv[1] == "evaluate-snapshots":
        from .anytime import main as snapshots
        snapshots(sys.argv[2:])
        return
    parser = argparse.ArgumentParser(description="P3: query-local predictor updates between CEM phases")
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
    evaluate.add_argument("--arms", nargs="+", choices=tuple(ARMS))
    evaluate.add_argument("--checkpoints", nargs="*", type=int)
    evaluate.add_argument("--support-method", choices=SUPPORT_METHODS)
    evaluate.add_argument("--adaptation-schedule", choices=ADAPTATION_SCHEDULES)
    evaluate.add_argument("--reference-source", choices=("elite", "random"))
    evaluate.add_argument("--continuation-context", choices=("imagined", "observed"))
    evaluate.add_argument("--initial-source", choices=("gaussian", "uniform-fragments", "context-fragments", "goal-context-fragments"))
    for name in ("references", "support-pairs", "fragments-per-pair", "injection-slots"):
        evaluate.add_argument("--"+name, type=int)
    for name in ("adaptation-depths", "continuation-depths"):
        evaluate.add_argument("--"+name, nargs="+", type=int)
    for name in ("noise-std", "goal-weight"):
        evaluate.add_argument("--"+name, type=float)
    evaluate.add_argument("--replay", choices=("current", "current-plus-replay"))
    evaluate.add_argument("--replay-fraction", type=float)
    evaluate.add_argument("--replay-capacity", type=int)
    evaluate.add_argument("--lr", "--learning-rate", dest="learning_rate", type=float)
    evaluate.add_argument("--updates", type=int,
                          help="Updates per checkpoint, or per selected depth with sequential-depth")
    evaluate.add_argument("--scope", choices=SCOPES)
    evaluate.add_argument("--seeds", nargs="+", type=int)
    evaluate.add_argument("--limit", type=int)
    evaluate.add_argument("--records", choices=RECORD_MODES)
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
            from experiments.provenance import replay_recorded_code
            replay_recorded_code("p3_interleave", read_json(args.report), sys.argv[1:])
            from .pipeline import analyze, lock_validation
            (lock_validation if args.command == "lock" else analyze)(args.report, args.output)
            return
        if args.cpu_threads < 1:
            raise ValueError("cpu_threads must be positive")
        lock = None
        interleave_keys = ("arms", "checkpoints", "support_method", "adaptation_schedule", "reference_source", "continuation_context",
                           "initial_source", "references", "support_pairs", "fragments_per_pair", "injection_slots",
                           "adaptation_depths", "continuation_depths", "noise_std", "goal_weight")
        if args.command == "evaluate" and args.locked_config:
            overrides = ("config", "experiment_dir", "learning_rate", "updates", "scope", "seeds", "limit", "records",
                         "replay", "replay_fraction", "replay_capacity", *interleave_keys)
            if args.split != "test" or any(getattr(args, key) is not None for key in overrides):
                raise ValueError("Use a validation lock only for test, without scientific overrides")
            lock = checked_json(args.locked_config)
            from experiments.provenance import replay_recorded_code
            replay_recorded_code("p3_interleave", lock, sys.argv[1:])
            config = lock["config"]
        else:
            if args.config is None:
                raise ValueError("Use bash codes/run.sh p3 COMMAND --variant interleave")
            config = load_config(args.config)
        if args.experiment_dir:
            config["experiment_dir"] = str(args.experiment_dir.resolve())
        if args.command == "evaluate" and lock is None:
            for key in interleave_keys:
                if getattr(args, key) is not None:
                    config["interleave"][key] = getattr(args, key)
            for flag, key in (("replay", "policy"), ("replay_fraction", "fraction"), ("replay_capacity", "capacity")):
                if getattr(args, flag) is not None:
                    config["interleave"]["replay"][key] = getattr(args, flag)
            for key in ("learning_rate", "updates"):
                if getattr(args, key) is not None:
                    config["adaptation"][key] = getattr(args, key)
            if args.scope is not None:
                config["scopes"] = [args.scope]
            if args.seeds is not None:
                config["p1"]["evaluation"]["seeds"] = args.seeds
                config["evaluation"]["seeds"] = args.seeds
            for flag, key in (("limit", "query_limit"), ("records", "records")):
                if getattr(args, flag) is not None:
                    config["evaluation"][key] = getattr(args, flag)
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
            print(f"P3 interleave failed: {type(error).__name__}: {error}", file=sys.stderr)
        raise SystemExit(1)

"""Separate core backend; existing interleave and saved-plan commands are intact."""

import argparse
from pathlib import Path

from ..interleave.config import load_config
from ..runtime import configure_logging, file_hash, read_json
from ..rollout.resources import checked_json
from .config import STAGE, validate


def main():
    parser = argparse.ArgumentParser(description="P3 core physical comparison; four cells by default")
    commands = parser.add_subparsers(dest="command", required=True)
    evaluate = commands.add_parser("evaluate")
    evaluate.add_argument("--config", type=Path)
    evaluate.add_argument("--locked-config", type=Path)
    evaluate.add_argument("--split", choices=("validation", "test"), default="validation")
    evaluate.add_argument("--run-name")
    evaluate.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    evaluate.add_argument("--cpu-threads", type=int, default=8)
    evaluate.add_argument("--verbose", action="store_true")
    evaluate.add_argument("--resume", action="store_true")
    for name in ("analyze", "lock"):
        command = commands.add_parser(name)
        command.add_argument("--report", required=True, type=Path)
        command.add_argument("--output", type=Path, required=name == "lock")
        command.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    configure_logging(args.verbose)
    if args.command in ("analyze", "lock"):
        import sys
        from experiments.provenance import replay_recorded_code
        replay_recorded_code("p3_core", read_json(args.report), sys.argv[1:])
        from .pipeline import analyze, lock_validation
        (analyze if args.command == "analyze" else lock_validation)(args.report, args.output)
        return
    if args.cpu_threads < 1:
        raise ValueError("CPU threads must be positive")
    lock = None
    if args.locked_config:
        if args.split != "test" or args.config is not None:
            raise ValueError("Use a core validation lock for test, without config overrides")
        lock = checked_json(args.locked_config)
        if lock.get("stage") != STAGE or file_hash(Path(lock["validation_report"])) != lock["report_sha256"]:
            raise ValueError("Core validation lock/report changed")
        import sys
        from experiments.provenance import replay_recorded_code
        replay_recorded_code("p3_core", lock, sys.argv[1:])
        config = lock["config"]
    elif args.config is not None and args.split == "validation":
        config = load_config(args.config)
    else:
        raise ValueError("Use p3 core for validation or a verified validation lock for test")
    validate(config)
    from .pipeline import evaluate as run
    run(config, args.device, args.cpu_threads, args.split, args.run_name, lock, args.resume)

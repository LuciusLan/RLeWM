"""Thin P1 CLI. Actual preparation/inference/adaptation is user-run."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import traceback

from lewm_tta.config import METHODS, ROOT, load_config, validate_config
from lewm_tta.runtime import configure_logging, read_json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("prepare", "evaluate"):
        command = commands.add_parser(name)
        command.add_argument("--config", type=Path, default=ROOT / "codes/configs/p1.json")
        command.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
        command.add_argument("--cpu-threads", type=int, default=8)
        command.add_argument("--encode-batch-size", type=int, default=32,
            help="Raw images per encoder batch for full cache; windows per batch for sampled profiles")
        command.add_argument("--verbose", action="store_true")
    evaluate = commands.choices["evaluate"]
    evaluate.add_argument("--split", choices=("validation", "test"), default="validation")
    evaluate.add_argument("--locked-config", type=Path)
    evaluate.add_argument("--run-name")
    evaluate.add_argument("--methods", nargs="+", choices=METHODS)
    evaluate.add_argument("--updates", type=int)
    evaluate.add_argument("--learning-rate", type=float)
    evaluate.add_argument("--seeds", nargs="+", type=int)
    lock = commands.add_parser("lock")
    lock.add_argument("--report", required=True, type=Path)
    lock.add_argument("--output", required=True, type=Path)
    lock.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    configure_logging(args.verbose)
    try:
        from lewm_tta.pipeline import evaluate, lock_validation, prepare
        if args.command == "lock":
            lock_validation(args.report, args.output)
            return
        if args.cpu_threads < 1 or args.encode_batch_size < 1:
            raise ValueError("Thread and encoding batch counts must be positive")
        locked = None
        if args.command == "evaluate" and args.locked_config:
            if args.split != "test" or any(value is not None for value in
                (args.methods, args.updates, args.learning_rate, args.seeds)):
                raise ValueError("A locked config is used only for test, without experiment overrides")
            locked = read_json(args.locked_config)
            config = locked["config"]
        else:
            config = load_config(args.config)
        if args.command == "prepare":
            prepare(config, args.device, args.cpu_threads, args.encode_batch_size)
        else:
            if args.methods is not None:
                config["adaptation"]["methods"] = args.methods
            if args.updates is not None:
                config["adaptation"]["updates"] = args.updates
            if args.learning_rate is not None:
                config["adaptation"]["learning_rate"] = args.learning_rate
            if args.seeds is not None:
                config["evaluation"]["seeds"] = args.seeds
            validate_config(config)
            evaluate(config, args.device, args.cpu_threads, args.encode_batch_size,
                     args.split, args.run_name, locked)
    except Exception as error:
        if args.verbose:
            traceback.print_exc()
        else:
            print(f"P1 failed: {type(error).__name__}: {error}", file=sys.stderr)
            print("Use --verbose for the full traceback.", file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()

"""Unified simulator pilot commands; simulator/model work runs only when requested."""

import argparse
import os
from pathlib import Path
import traceback

from ..runtime import configure_logging
from .config import load_config, validate_config


def main():
    parser = argparse.ArgumentParser(description="Observed simulator outcomes: saved-plan replay and six-cell MPC pilot")
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("prepare", "check", "preview", "replay", "evaluate"):
        command = commands.add_parser(name)
        command.add_argument("--config", type=Path)
        command.add_argument("--verbose", action="store_true")
        if name != "prepare":
            command.add_argument("--limit", type=int)
            command.add_argument("--seeds", type=int, nargs="+")
        command.add_argument("--split", choices=("validation", "test"))
        if name == "preview":
            command.add_argument("--output", type=Path)
        if name == "evaluate":
            command.add_argument("--locked-config", type=Path)
        if name in ("replay", "evaluate"):
            command.add_argument("--run-name")
            command.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
            command.add_argument("--cpu-threads", type=int, default=8)
            command.add_argument("--records", choices=("compact", "none"))
            command.add_argument("--call-records", choices=("compact", "none"))
            command.add_argument("--budget-hours", type=float)
    analyze = commands.add_parser("analyze")
    analyze.add_argument("--report", type=Path, required=True)
    analyze.add_argument("--output", type=Path)
    lock = commands.add_parser("lock")
    lock.add_argument("--report", type=Path, required=True)
    lock.add_argument("--output", type=Path)
    args = parser.parse_args()
    configure_logging(getattr(args, "verbose", False))
    # Set these before importing environment or rendering libraries; no .bashrc edits.
    for name, value in {"SDL_VIDEODRIVER": "dummy", "SDL_AUDIODRIVER": "dummy",
                        "PYGAME_HIDE_SUPPORT_PROMPT": "1", "MUJOCO_GL": "egl", "PYOPENGL_PLATFORM": "egl"}.items():
        os.environ.setdefault(name, value)
    try:
        if args.command == "analyze":
            from .pipeline import analyze as run
            run(args.report, args.output)
            return
        if args.command == "lock":
            from .green import lock_validation
            lock_validation(args.report, args.output)
            return
        from .green import enabled, locked_test_config
        validation_lock = None
        if getattr(args, "locked_config", None):
            if args.config or args.split not in (None, "test") or any(
                    getattr(args, key, None) is not None for key in ("limit", "seeds", "records", "call_records", "budget_hours")):
                raise ValueError("Locked green test supplies its protocol; omit --config and evaluation overrides")
            config, validation_lock = locked_test_config(args.locked_config)
        else:
            if args.config is None:
                raise ValueError("Provide --config, or --locked-config for green test evaluation")
            config = load_config(args.config)
            if args.split is not None:
                if not enabled(config):
                    raise ValueError("Explicit validation/test splits belong to the green benchmark")
                config["simulation"]["split"] = args.split
        if getattr(args, "limit", None) is not None:
            config["evaluation"]["query_limit"] = args.limit
        if getattr(args, "seeds", None) is not None:
            config["simulation"]["seeds"] = args.seeds
        for flag, key in (("records", "records"), ("call_records", "call_records")):
            if getattr(args, flag, None) is not None:
                config["evaluation"][key] = getattr(args, flag)
        if getattr(args, "budget_hours", None) is not None:
            config["simulation"]["runtime_budget_hours"] = args.budget_hours
        if getattr(args, "cpu_threads", 1) < 1:
            raise ValueError("cpu_threads must be positive")
        validate_config(config)
        if args.command == "prepare":
            from .cohorts import prepare
            prepare(config)
        elif args.command == "check":
            from .checks import check
            check(config)
        elif args.command == "preview":
            from .preview import preview
            preview(config, args.output)
        else:
            from .pipeline import evaluate
            evaluate(config, "replay" if args.command == "replay" else "mpc", args.device, args.cpu_threads, args.run_name, validation_lock)
    except Exception:
        traceback.print_exc()
        raise SystemExit(1)

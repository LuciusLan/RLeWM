"""Materialize the core matrix, or run its locked test phase."""

import argparse
from pathlib import Path
import shlex
import subprocess
import sys

from lewm_tta.runtime import object_hash, read_json
from .catalog import ROOT
from .config import materialize, query_dataset_settings, resolve
from .fresh import effective_config


def core(selected, arguments, dry_run=False, *, dataset=None, cache_only=False):
    parser = argparse.ArgumentParser(description="Four-cell P3 core matrix; one seed and shared first-call snapshots by default")
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--phase", choices=("validation", "test"), default="validation")
    parser.add_argument("--locked-config", type=Path)
    parser.add_argument("--iterations", nargs="+", type=int)
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--checkpoints", nargs="+", type=int)
    parser.add_argument("--adaptation-schedule", choices=("pooled-depth", "sequential-depth"),
                        help="Checkpoint support/update schedule; default pooled-depth")
    parser.add_argument("--mid-search", action="store_true", default=None,
                        help="Restore the historical six-cell matrix with mid-search action retrieval")
    parser.add_argument("--control-budget", type=int, choices=(25, 50))
    parser.add_argument("--runtime-budget-hours", type=float)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--updates", type=int,
                        help="Updates per checkpoint (pooled-depth) or per selected depth (sequential-depth); default 1")
    parser.add_argument("--collection-config", type=Path, help="Explicit fresh query collection profile on validation")
    parser.add_argument("--python", type=Path, help="Model/physics interpreter; defaults to task-specific core profile")
    parser.add_argument("--bootstrap-samples", type=int)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--cpu-threads", type=int, default=8)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--resume", action="store_true", help="Continue a verified clean budget-stopped run at the next complete query")
    args = parser.parse_args(arguments)
    if args.cpu_threads < 1:
        raise ValueError("CPU threads must be positive")
    if args.phase == "test":
        if args.locked_config is None or any(getattr(args, key) is not None for key in (
                "output_root", "iterations", "seeds", "checkpoints", "control_budget", "runtime_budget_hours",
                "limit", "learning_rate", "updates", "adaptation_schedule", "mid_search", "bootstrap_samples", "collection_config")) or dataset is not None:
            raise ValueError("Core test requires --locked-config only, without dataset/scientific/output overrides")
        engine = ["--locked-config", str(args.locked_config.resolve())]
        runtime_config = read_json(args.locked_config)["config"]
        if cache_only and not runtime_config.get("cache_only", False):
            raise ValueError("Core test loading mode is fixed by its validation lock; create cache-only validation and a new lock first")
    else:
        if args.locked_config or args.output_root is None:
            raise ValueError("Core validation requires --output-root and no test lock")
        value = resolve(selected, dataset=dataset, cache_only=cache_only)
        if "core" not in value:
            raise ValueError("Use --variant core or a config extending codes/configs/p3/core.json")
        value["experiment_dir"] = str(args.output_root.resolve())
        settings = value["core"]
        collection = args.collection_config or settings.get("collection_by_dataset", {}).get(dataset)
        if collection:
            value["query_dataset"] = query_dataset_settings({"collection_config": str(collection), "regime": "source",
                "batch_size": value["query_dataset"].get("batch_size", 128)})
        if args.iterations is not None:
            settings["iterations"] = sorted(args.iterations)
        value["planning"]["iterations"] = max(settings["iterations"], default=0)
        if args.checkpoints is not None:
            value["interleave"]["checkpoints"] = args.checkpoints
        if args.adaptation_schedule is not None:
            value["interleave"]["adaptation_schedule"] = args.adaptation_schedule
        if args.mid_search is not None:
            settings["mid_search"] = args.mid_search
        for key in ("control_budget", "runtime_budget_hours", "bootstrap_samples"):
            if getattr(args, key) is not None:
                settings[key] = getattr(args, key)
        value["evaluation"].update(query_limit=args.limit, records="compact")
        for key in ("learning_rate", "updates"):
            if getattr(args, key) is not None:
                value["adaptation"][key] = getattr(args, key)
        effective = effective_config(value)
        seeds = args.seeds if args.seeds is not None else value["evaluation"].get("seeds", effective["p1"]["evaluation"]["seeds"])[:1]
        value["evaluation"]["seeds"] = list(seeds)
        effective = effective_config(value)
        effective["p1"]["evaluation"]["seeds"] = list(seeds)
        from lewm_tta.core.config import cells, primary_endpoint, validate
        validate(effective)
        selected_cells = cells(effective)
        runtime_config = effective
        print(f"P3 core: {len(selected_cells)} cells, seeds={seeds}, J={settings['iterations']}, "
              f"controls={settings['control_budget']}, primary={primary_endpoint(effective)}, "
              f"runtime cap={settings['runtime_budget_hours']}h (checked between complete queries).", flush=True)
        print("Cells: "+", ".join(selected_cells.values()), flush=True)
        from lewm_tta.interleave.config import adaptation_schedule, checkpoint_updates
        print(f"Adaptation: {adaptation_schedule(effective)}, {checkpoint_updates(effective)} requested updates "
              "per active checkpoint; one final predictor scores all depths.", flush=True)
        print("First call shares maximum-J search; later MPC calls use each budget's own real observations.", flush=True)
        path = ROOT/"artifacts/experiment-configs"/f"{object_hash(value)}.json" if dry_run else materialize(value)
        engine = ["--config", str(path)]
    task = runtime_config["p1"]["data"].get("dataset_id", "pusht")
    interpreters = runtime_config["core"].get("python_by_dataset", {})
    python = args.python or Path(interpreters.get(task, interpreters.get("default", sys.executable)))
    # Preserve the venv's executable path: resolving its symlink to /usr/bin
    # would silently discard pyvenv.cfg and select the wrong package versions.
    python = (ROOT/python).absolute()
    command = [str(python), "-X", "faulthandler", "-u", str(ROOT/"codes/experiment.py"),
               "legacy", "p3_core", "evaluate", *engine, "--split", args.phase,
               "--device", args.device, "--cpu-threads", str(args.cpu_threads), *( ["--verbose"] if args.verbose else [])]
    if args.resume:
        command.append("--resume")
    print(shlex.join(command), flush=True)
    if not dry_run:
        subprocess.run(command, cwd=ROOT, check=True)

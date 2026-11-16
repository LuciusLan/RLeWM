"""Single stage interface; resolves configs before calling existing engines."""

import argparse
import importlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

from lewm_tta.runtime import read_json
from .catalog import BACKENDS, REPORT_STAGES, ROOT, STAGES, infer_backend, profile, stage_for_backend
from .config import materialize, resolve
from .datasets import DATASETS
from .provenance import archived_command, backend_identity


HELP = """Usage: bash codes/run.sh STAGE COMMAND [--variant NAME] [engine options]

Stages:
  p0         prepare | smoke | evaluate
  p1         prepare | check-retrieval | evaluate | analyze-ranks | lock
             variants: default (PushT), tworooms, reacher, cube (source memory only)
  p2         prepare | check | evaluate | analyze | lock | sweep
             variants: default (H5 temporal controls), initial, followup, fresh, fresh-temporal
             sweep [--learning-rates LR ...] compares four temporal schedules on fresh validation
  p3         prepare | check | evaluate | analyze | lock | ablate | budget | success | core
             variants: default (TTT x proposals), shared, proposals, temporal, fresh, fresh-temporal, fresh-proposals, interleave, interleave-lewm-budget, core
             interleave is a separate checkpoint-RCA / intermediate-continuation comparison
             compare [--phase validation|test|all] completes/resumes the six-cell default
             ablate compares Frozen proposal sources, injection shares and CEM budgets on fresh validation
             budget --iterations 2 4 6 --output-root DIR captures interleave stopping plans in one maximum-J search
             --execution-mode independent retains separate budget runs; other P3 variants use them by default
             success --reports REPORT... --output-dir DIR executes saved fresh-query plans in the simulator
             core --dataset TASK --output-root DIR runs four paired cells, one seed, shared J snapshots
             core --mid-search restores the historical six-cell matrix
             core --control-budget 50 enables real interleave MPC; default 25 is one open-loop plan
  coverage   prepare | evaluate | analyze | plot | figures
  sim        prepare | check | preview | replay | evaluate | analyze | lock
             variants: default/fresh future goals, green fixed-target MPC
  collect    prepare | collect | check | preview-rendering
             fresh simulator trajectories; variants: default, shifts, tworooms, reacher, cube, cube-plan, reacher-checkerboard
  split      create
  fresh      status | prepare | check | smoke | validation | test | all
             consolidated fresh P2/P3 suite; --dataset pusht|tworooms|reacher|cube
             optional --jobs p2 p3 temporal; same dataset selector works for p2/p3
  preprocess inspect | extract | views | prepare [--datasets tworooms reacher cube]
             source archives -> NVMe HDF5 -> persistent latents and cuVS keys
  appearance prepare | preview | synthesize | check
             PushT/Reacher color and background shifts of recorded trajectories;
             unchanged actions/states, inherited episode splits; no WM required
  shifted    status | prepare | check | smoke | validation | test | all
             appearance pools -> new latents/cuVS assets -> paired source/shifted
             memory P2/P3 runs; optional --jobs temporal; separate validation locks

Utilities:
  list                       List active profiles and their outputs.
  STAGE config               Print the inherited configuration without running a model.
  prepare-cohorts             Prepare source-corpus P2/P3 cohorts, without inference or CEM.
  legacy ENGINE COMMAND      Use a migrated historical config (e.g. legacy p25 evaluate).

Common interface options: --variant NAME, --config PATH, --dry-run.
P2/P3 also accept --cache-only: reuse completed latent/action caches and the
stored query audit without raw-source checks or re-encoding. Real simulator
evaluation still requires fresh replay payloads. Missing caches are errors.
P2/P3: --dataset NAME binds the matching source, fresh queries and action bounds;
without --variant it selects fresh (sweep/ablate retain their fresh presets).
All remaining options are passed to the engine. Default profiles use the
64-validation / 150-test split and /home/ubuntu/lewm_tta/pusht_split memory cache.
Fresh profiles use the collection-reserved query splits (currently 64/100 episodes).
See docs/stages/ for protocols and commands.
Complete command/argument reference: docs/cli-arguments.md
"""


def flag_value(arguments, flag):
    for position, argument in enumerate(arguments):
        if argument == flag:
            if position + 1 == len(arguments):
                raise ValueError(f"{flag} requires a value")
            return arguments[position + 1]
        if argument.startswith(flag + "="):
            return argument.split("=", 1)[1]
    return None


def show_profiles():
    print("Stage / variant          Backend          Experiment directory")
    for stage, variants in STAGES.items():
        for variant in variants:
            backend, path = profile(stage, variant)
            config = resolve(path)
            output = config.get("experiment_dir", config.get("output_dir", config.get("paths", {}).get("experiment_dir")))
            print(f"{stage + '/' + variant:<24} {backend:<16} {output}")


def tool(stage, action, arguments, dry_run):
    if stage == "p0":
        if action == "prepare":
            command = ["bash", str(ROOT / "codes/experiments/tools/p0_prepare.sh"), *arguments]
        elif action in ("smoke", "evaluate"):
            command = [sys.executable, "-B", str(ROOT / "codes/experiment.py"), "_p0",
                       "--mode", "smoke" if action == "smoke" else "baseline", "--samples",
                       "3" if action == "smoke" else "30", "--device", "cuda", *arguments]
        else:
            raise ValueError("P0 supports prepare, smoke and evaluate")
    elif stage == "split" and action == "create":
        command = [sys.executable, "-B", str(ROOT / "codes/experiment.py"), "_split", *arguments]
    elif stage == "coverage" and action == "figures":
        command = [sys.executable, "-B", str(ROOT / "codes/experiment.py"), "_figures", *arguments]
    else:
        return False
    if dry_run:
        print(shlex.join(command))
    else:
        subprocess.run(command, check=True)
    return True


def dispatch(argv):
    if not argv or argv[0] in ("-h", "--help", "help"):
        print(HELP)
        return
    if (len(argv) >= 2 and not {"--help", "-h"}.intersection(argv)
            and any(value.split("=", 1)[0] in ("--learning-rates", "--lrs") for value in argv[2:])
            and tuple(argv[:2]) != ("p2", "sweep")):
        raise ValueError("--learning-rates/--lrs belongs to 'p2 sweep'. Use: "
                         "bash codes/run.sh p2 sweep --dataset reacher --learning-rates 1e-5 5e-5 1e-4 "
                         "--methods frozen uniform fr-fixed forward reverse --seeds 41 42 43. "
                         "For a single setting use --learning-rate LR. See docs/cli-arguments.md.")
    if argv[0] == "list":
        show_profiles()
        return
    if argv[:2] == ["p3", "success"]:
        from lewm_tta.outcomes.cli import main as success
        success(argv[2:])
        return
    if argv[0] == "preprocess":
        from .preprocess import main as preprocess
        preprocess(argv[1:])
        return
    if argv[0] == "appearance":
        from lewm_tta.appearance.cli import main as appearance
        appearance(argv[1:])
        return
    if argv[0] == "shifted":
        from .shifted import main as shifted
        shifted(argv[1:])
        return
    if argv[0] == "fresh":
        from .fresh import main as fresh
        if len(argv) < 2:
            raise ValueError("Use fresh status|prepare|check|smoke|validation|test|all")
        arguments = argv[2:]
        fresh(argv[1], [value for value in arguments if value != "--dry-run"], "--dry-run" in arguments)
        return
    if argv[0] in ("_p0", "_split", "_figures"):
        module = {"_p0": "experiments.backends.p0", "_split": "experiments.tools.resplit",
                  "_figures": "experiments.tools.coverage_results"}[argv[0]]
        sys.argv = [sys.argv[0], *argv[1:]]
        importlib.import_module(module).main()
        return
    if argv[0] == "prepare-cohorts":
        if set(argv[1:]) - {"--dry-run", "--verbose"}:
            raise ValueError("prepare-cohorts accepts only --dry-run and --verbose")
        for stage in ("p2", "p3"):
            for variant in STAGES[stage]:
                if resolve(profile(stage, variant)[1]).get("query_dataset"):
                    # Fresh cohorts have their own collection/audit prerequisites.
                    continue
                command = [sys.executable, "-B", str(ROOT / "codes/experiment.py"), stage, "prepare", "--variant", variant]
                if "--verbose" in argv:
                    command.append("--verbose")
                if "--dry-run" in argv:
                    dispatch([stage, "prepare", "--variant", variant, "--dry-run"])
                else:
                    subprocess.run(command, check=True)
        return
    legacy = argv[0] == "legacy"
    if legacy:
        if len(argv) < 3 or argv[1] not in BACKENDS:
            raise ValueError("Use legacy ENGINE COMMAND; engine must be one of " + ", ".join(BACKENDS))
        backend, action, remaining = argv[1], argv[2], argv[3:]
        stage = stage_for_backend(backend)
    else:
        if len(argv) < 2:
            raise ValueError("Provide a stage and command; use --help for the catalog")
        stage, action, remaining = argv[0], argv[1], argv[2:]
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    default_variant = {("p2", "sweep"): "fresh-temporal", ("p3", "ablate"): "fresh-proposals",
                       ("p3", "budget"): "interleave", ("p3", "core"): "core"}
    parser.add_argument("--variant")
    if stage == "p0":
        parser.set_defaults(dataset=None)  # P0's backend owns --dataset HDF5_PATH.
    else:
        parser.add_argument("--dataset", choices=DATASETS)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--cache-only", action="store_true")
    options, arguments = parser.parse_known_args(remaining)
    if options.cache_only and stage not in ("p2", "p3"):
        raise ValueError("--cache-only is supported for P2/P3 evaluation workflows; P1 preparation still requires raw data")
    if options.dataset is not None and (legacy or stage not in ("p2", "p3")):
        raise ValueError("--dataset selects fresh P2/P3 evaluations; P1/collect use --variant TASK")
    options.variant = options.variant or default_variant.get((stage, action), "fresh" if options.dataset else "default")
    if tool(stage, action, arguments, options.dry_run):
        return
    if legacy:
        if options.variant != "default":
            raise ValueError("Legacy engines use --config, not named variants")
        selected = options.config or ROOT / "codes/legacy/configs" / ("rank-coverage.json" if backend == "rank_coverage" else backend + ".json")
    else:
        backend, selected = profile(stage, options.variant)
        selected = options.config or selected
    if stage == "p3" and action == "compare":
        from .workflows import compare
        compare(selected, arguments, options.dry_run, dataset=options.dataset, cache_only=options.cache_only)
        return
    if stage == "p2" and action == "sweep":
        from .p2_sweep import sweep
        sweep(selected, arguments, options.dry_run, dataset=options.dataset, cache_only=options.cache_only)
        return
    if stage == "p3" and action == "ablate":
        from .p3_ablations import ablate
        ablate(selected, arguments, options.dry_run, dataset=options.dataset, cache_only=options.cache_only)
        return
    if stage == "p3" and action == "budget":
        from .p3_budgets import budget
        budget(selected, arguments, options.dry_run, dataset=options.dataset, cache_only=options.cache_only)
        return
    if stage == "p3" and action == "core":
        from .p3_core import core
        core(selected, arguments, options.dry_run, dataset=options.dataset, cache_only=options.cache_only)
        return
    source_path = flag_value(arguments, "--report") if action in ("analyze", "analyze-ranks", "lock", "plot") else None
    source_path = source_path or flag_value(arguments, "--locked-config")
    source_path = source_path or (flag_value(arguments, "--validation-report") if stage == "coverage" else None)
    source = read_json(Path(source_path)) if source_path else None
    if options.cache_only and source is not None and not source.get("config", {}).get("cache_only", False):
        raise ValueError("A saved report/lock supplies its own loading mode. Create cache-only validation and a new lock; do not override historical reports/locks")
    if source is not None and options.dataset is not None:
        raise ValueError("Reports and validation locks supply their own dataset/config; omit --dataset")
    if source is not None:
        inferred = REPORT_STAGES.get(source.get("stage"), backend)
        if stage_for_backend(inferred) != stage:
            raise ValueError("The report belongs to a different stage")
        backend = inferred
    if flag_value(arguments, "--locked-config") and options.config is not None:
        raise ValueError("A validation lock supplies its own config; omit --config")
    needs_config = action not in ("analyze", "analyze-ranks", "lock", "plot") and not flag_value(arguments, "--locked-config")
    config = resolve(selected, dataset=options.dataset, cache_only=options.cache_only) if needs_config or action == "config" else None
    if config is not None:
        actual = infer_backend(stage, config)
        if not legacy and options.variant != "default" and actual != backend:
            raise ValueError("Explicit config does not match the selected variant")
        backend = actual
    if action == "config":
        print(json.dumps(config, indent=2))
        return
    if backend in ("p3b", "p3c"):
        arguments = ["--lr" + arg[len("--learning-rate"):] if arg == "--learning-rate" or arg.startswith("--learning-rate=") else arg for arg in arguments]
    if config is not None:
        # Dry runs describe a stable future path without creating it.
        if options.dry_run:
            from lewm_tta.runtime import object_hash
            config_path = ROOT / "artifacts/experiment-configs" / f"{object_hash(config)}.json"
        else:
            config_path = materialize(config)
        arguments = ["--config", str(config_path), *arguments]
    engine_arguments = [action, *arguments]
    if backend == "collect" and action == "collect":
        from .collection_resume import recorded_collection_code
        recorded = recorded_collection_code(config)
        if recorded is not None and recorded != backend_identity("collect"):
            command = archived_command("collect", recorded)
            if command is None:
                raise ValueError("Existing collection source has no verified archive; refusing mixed-source collection")
            command += engine_arguments
            if options.dry_run:
                print(shlex.join(command))
            else:
                print("Resuming collection with its verified archived implementation.", flush=True)
                os.execv(sys.executable, command)
            return
    # Saved-rank analysis records its own code/input hashes and does not replay
    # the original evaluation or create a validation lock.
    if source and action != "analyze-ranks" and source.get("code") and source["code"] != backend_identity(backend):
        command = archived_command(backend, source["code"])
        if command is None:
            raise ValueError("No verified archive matches this historical report")
        command += engine_arguments
        if options.dry_run:
            print(shlex.join(command))
        else:
            print("Replaying the report's verified source archive.", flush=True)
            if action == "evaluate":
                from .progress import replay_with_progress
                replay_with_progress(command, source)
            else:
                os.execv(sys.executable, command)
        return
    if options.dry_run:
        module, positional = BACKENDS[backend]
        print(f"{stage}/{options.variant} -> {module}.main{positional}: " + shlex.join(engine_arguments))
        print("Output:", config.get("experiment_dir", config.get("output_dir", config.get("paths", {}).get("experiment_dir"))) if config else "saved report/lock")
        return
    module, positional = BACKENDS[backend]
    sys.argv = [str(ROOT / "codes/experiment.py"), *engine_arguments]
    importlib.import_module(module).main(*positional)


def main():
    try:
        dispatch(sys.argv[1:])
    except (ValueError, FileNotFoundError, subprocess.CalledProcessError) as error:
        print(f"Experiment failed: {error}", file=sys.stderr)
        raise SystemExit(1)

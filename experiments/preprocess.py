"""User-run multi-task preparation, delegating encoding and indexing to P1."""

import argparse
import math
from pathlib import Path
import shlex
import subprocess
import sys

from lewm_tta.config import load_config
from lewm_tta.runtime import object_hash, write_json
from lewm_tta.preprocessing.archives import extract_hdf5
from lewm_tta.preprocessing.inspection import inspect_task
from lewm_tta.preprocessing.hdf5 import prepare_source_view, raw_dataset_path
from lewm_tta.preprocessing.storage import GIB, require_space, reserve_cache, storage_plan
from .catalog import ROOT, profile
from .config import materialize

TASKS = ("tworooms", "reacher", "cube")


def select_configs(args):
    if args.config:
        if args.datasets:
            raise ValueError("Use --config for one task, or --datasets for catalog profiles")
        configs = [load_config(args.config)]
    else:
        names = args.datasets or TASKS
        if len(set(names)) != len(names):
            raise ValueError("Choose distinct dataset names")
        configs = [load_config(profile("p1", name)[1]) for name in names]
    # Resolve user path overrides into the immutable effective P1 configuration.
    for config in configs:
        name = config["data"]["dataset_id"]
        config["preprocessing"]["archive"] = str((ROOT / config["preprocessing"]["archive"]).resolve())
        if "raw_dataset" in config["preprocessing"]:
            config["preprocessing"]["raw_dataset"] = str((ROOT / config["preprocessing"]["raw_dataset"]).resolve())
        if args.cache_root:
            config["paths"]["cache_dir"] = str((args.cache_root / name).resolve())
        if args.output_root:
            config["paths"]["experiment_dir"] = str((args.output_root / name).resolve())
            if "raw_dataset" in config["preprocessing"]:
                config["paths"]["dataset"] = str((args.output_root / name / "source.h5").resolve())
        if args.scratch_root:
            raw = str((args.scratch_root / f"lewm_{name}" / config["preprocessing"]["member"]).resolve())
            if "raw_dataset" in config["preprocessing"]:
                config["preprocessing"]["raw_dataset"] = raw
            else:
                config["paths"]["dataset"] = raw
    return configs


def inspect_jobs(configs):
    return [(config, inspect_task(config)) for config in configs]


def describe(jobs, volumes):
    for _, item in jobs:
        mark = "provisional episode/row estimate" if item["estimate_before_extraction"] else "exact metadata counts"
        print(f"{item['dataset']}: {item['cache']['total_bytes']/GIB:.2f} GiB cache ({mark}); "
              f"{item['extraction_bytes']/GIB:.2f} GiB HDF5 to extract", flush=True)
        summary = item["source_summary"]
        if summary:
            print(f"  {summary['episodes']:,} episodes, {summary['rows']:,} observations; "
                  f"{item['checkpoint']['raw_action_dim']} raw action coordinates", flush=True)
            print(f"  episode column: {summary['metadata_columns']['episode_idx']}; "
                  f"terminal action padding: {summary['action_audit']['terminal_padding_rows']:,} rows", flush=True)
        print(f"  cache: {item['paths']['cache_dir']}", flush=True)
    for group in volumes:
        state = "fits" if group["sufficient"] else "does not fit"
        print(f"Volume {group['filesystem_example']}: {group['available_bytes']/GIB:.2f} GiB free; "
              f"need {group['additional_bytes']/GIB:.2f} + reserve {group['reserve_bytes']/GIB:.1f} GiB: {state}", flush=True)
    if any(item["estimate_before_extraction"] for _, item in jobs):
        print("Cache estimates use provisional row budgets. Preparation checks exact episode/window counts after extraction, before encoding.", flush=True)


def prepare_command(config, args, *, dry_run):
    path = (ROOT / "artifacts/experiment-configs" / f"{object_hash(config)}.json"
            if dry_run else materialize(config))
    # A fresh process per task frees model/GPU/native handles between datasets.
    command = [sys.executable, "-X", "faulthandler", "-u", str(ROOT / "codes/experiment.py"),
               "p1", "prepare", "--config", str(path), "--device", args.device,
               "--cpu-threads", str(args.cpu_threads), "--encode-batch-size", str(args.encode_batch_size)]
    if args.verbose:
        command.append("--verbose")
    return command


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("inspect", "extract", "views", "prepare"))
    parser.add_argument("--datasets", nargs="+", choices=TASKS)
    parser.add_argument("--config", type=Path, help="Custom source-memory P1 profile for one task")
    parser.add_argument("--cache-root", type=Path, help="Cache root; each task receives its own subdirectory")
    parser.add_argument("--output-root", type=Path, help="Root for task manifests/prepared records")
    parser.add_argument("--scratch-root", type=Path, help="Root for extracted lewm_TASK/MEMBER HDF5 files")
    parser.add_argument("--keep-free-gib", type=float, default=6)
    parser.add_argument("--scratch-keep-free-gib", type=float, default=10)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--cpu-threads", type=int, default=8)
    parser.add_argument("--encode-batch-size", type=int, default=64)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)
    if min(args.cpu_threads, args.encode_batch_size) < 1 or any(
            not math.isfinite(value) or value < 0 for value in (args.keep_free_gib, args.scratch_keep_free_gib)):
        raise ValueError("Batch/thread counts must be positive; disk reserves must be finite and nonnegative")
    configs = select_configs(args)
    jobs = inspect_jobs(configs)
    space_args = {"keep_free_gib": args.keep_free_gib, "scratch_keep_free_gib": args.scratch_keep_free_gib}
    volumes = storage_plan(jobs, include_cache=args.command not in ("extract", "views"), **space_args)
    describe(jobs, volumes)
    if args.command == "inspect" or args.dry_run:
        if args.dry_run:
            for config, item in jobs:
                if not item["source_exists"]:
                    print(f"Extract {config['preprocessing']['archive']} -> {raw_dataset_path(config)}")
                if args.command in ("views", "prepare"):
                    if config.get("preprocessing", {}).get("raw_dataset"):
                        print(f"Prepare canonical metadata/action view: {config['paths']['dataset']}")
                if args.command == "prepare":
                    print(shlex.join(prepare_command(config, args, dry_run=True)))
        return
    # Only extraction sizes are exact until HDF5 metadata exists. Never reject
    # a feasible preparation solely on the provisional observation estimate.
    require_space(storage_plan(jobs, include_cache=False, **space_args))
    for config, item in jobs:
        if item["archive"] is not None:
            extract_hdf5(config, item["archive"])
    jobs = inspect_jobs(configs)
    if args.command == "extract":
        describe(jobs, storage_plan(jobs, **space_args))
        return
    if args.command == "views":
        for config, _ in jobs:
            prepare_source_view(config)
        print("Canonical source views ready; no model loaded or cache encoded.", flush=True)
        return
    # Check the SUM for the selected tasks before any GPU encoding is launched.
    volumes = storage_plan(jobs, **space_args)
    describe(jobs, volumes)
    require_space(volumes)
    for config, item in jobs:
        # Recheck current free space (e.g. if another process consumed disk).
        require_space(storage_plan([(config, item)], **space_args))
        reserve_cache(config, item)
        prepare_source_view(config)
        directory = Path(config["paths"]["experiment_dir"])
        write_json(directory / "source-inspection.json", item)
        write_json(directory / "preparation-config.json", config)
        subprocess.run(prepare_command(config, args, dry_run=False), check=True)
    print("Prepared all selected source memories. Fresh validation/test collection remains a separate step.", flush=True)

"""Reserve new test episodes using saved metadata; never loads a model or CUDA."""

from __future__ import annotations

import argparse
from pathlib import Path

from lewm_tta.config import ROOT, load_config, validate_config
from lewm_tta.runtime import file_hash, read_json
from lewm_tta.splits.episodes import collect_exposure, resplit_manifest, validate_parent, write_profile


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-config", type=Path, default=ROOT / "codes/configs/p1/default.json")
    parser.add_argument("--parent-manifest", type=Path, default=ROOT / "artifacts/p1-full-q16/manifest.json")
    parser.add_argument("--test-episodes", type=int, default=150)
    parser.add_argument("--windows-per-episode", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20261026)
    parser.add_argument("--output-config", type=Path)
    parser.add_argument("--split-dir", type=Path)
    parser.add_argument("--experiment-dir", type=Path)
    args = parser.parse_args()
    config = load_config(args.source_config)
    parent = read_json(args.parent_manifest)
    validate_parent(parent)
    source = Path(config["paths"]["dataset"]).resolve(strict=True)
    stat = source.stat()
    if str(source) != parent["source"]["path"] or (stat.st_size, stat.st_mtime_ns) != (
            parent["source"]["bytes"], parent["source"]["mtime_ns"]):
        raise ValueError("Dataset source identity changed since the cached manifest")
    validation_n = len(parent["episodes"]["validation"])
    name = f"v{validation_n}-t{args.test_episodes}-q{args.windows_per_episode}"
    split_dir = (args.split_dir or ROOT / f"artifacts/dataset-splits/pusht-{name}").resolve()
    output = (args.output_config or ROOT / f"codes/configs/p1/{name}.json").resolve()
    experiment = (args.experiment_dir or ROOT / f"artifacts/p1-{name}").resolve()
    config["paths"]["experiment_dir"] = str(experiment)
    data = config["data"]
    data.update(validation_episodes=validation_n, test_episodes=args.test_episodes,
                memory_windows="all", windows_per_query_episode=args.windows_per_episode,
                split_seed=args.seed, split_from=str(split_dir / "manifest.json"))
    data["inspected_report_globs"] = [str(ROOT / "artifacts/p*/runs/*/report.json"),
                                      str(ROOT / "artifacts/p*/runs/*/query-episodes.json")]
    validate_config(config)
    excluded, exposure = collect_exposure(parent, data, [str(ROOT / "artifacts/p*/manifest.json"),
        str(ROOT / "artifacts/dataset-splits/*/manifest.json")])
    manifest, audit = resplit_manifest(parent, data, excluded, args.parent_manifest)
    audit["exposure_sources"] = exposure
    audit["generator_code_sha256"] = {str(path.relative_to(ROOT)): file_hash(path) for path in (
        Path(__file__).resolve(), ROOT / "codes/lewm_tta/splits/episodes.py")}
    write_profile(config, manifest, audit, output, split_dir)
    print("Saved episode split:", audit["episode_counts"])
    print("Sampled one-step queries:", audit["evaluation_query_counts"])
    print("Validation IDs preserved; new test excludes all known prior query episodes.")
    print("Prior retrieval-support exposure and unknown pretraining overlap are recorded in audit.json.")
    print("Manifest:", split_dir / "manifest.json")
    print("Prepare assets (user-run): bash codes/run.sh p1 prepare --config", output.relative_to(ROOT)
          if output.is_relative_to(ROOT) else output)


if __name__ == "__main__":
    main()

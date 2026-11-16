"""Re-render existing trajectories with appearance shifts and unchanged labels."""

import argparse
import math
from pathlib import Path

from ..collection.config import headless
from ..collection.manifest import sealed
from ..runtime import configure_logging
from .settings import PATTERNS, settings_from_args


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("command", choices=("prepare", "preview", "synthesize", "check"))
    parser.add_argument("--dataset", choices=("pusht", "reacher"))
    parser.add_argument("--source-manifest", type=Path, help="P1 manifest (memory only) or fresh queries.json")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split", choices=("memory", "validation", "test", "all"),
                        help="Default memory for P1, all for fresh query manifests")
    parser.add_argument("--limit-episodes", type=int, help="Fixed pilot subset; use a separate output directory")
    parser.add_argument("--episode-seed", type=int, default=0, help="Seed for a shared random episode subset when limiting the cohort")
    for flag, description in (("object", "PushT T block or Reacher arm"), ("agent", "PushT robot disk"),
                              ("goal", "PushT decorative goal marker"), ("background", "Background primary color")):
        parser.add_argument(f"--{flag}-color", nargs=3, type=int, metavar=("R", "G", "B"), help=description + ", RGB 0..255")
    parser.add_argument("--background-color2", nargs=3, type=int, metavar=("R", "G", "B"), help="Pattern secondary RGB 0..255")
    parser.add_argument("--background-pattern", choices=PATTERNS, default="preserve")
    parser.add_argument("--pattern-size", type=int, default=64, help="Tile/stripe width in 512-unit scene/texture coordinates")
    parser.add_argument("--compression", choices=("gzip", "lzf"), default="gzip")
    parser.add_argument("--min-free-gib", type=float, default=2, help="Stop before this output-volume free-space reserve is exhausted")
    parser.add_argument("--workers", type=int, default=1, help="Synthesis processes, each with its own renderer; start with 4 for PushT on 8 CPU cores")
    parser.add_argument("--write-batch-size", type=int, default=16, help="Frames per HDF5 write; bounded per-worker RGB buffer")
    parser.add_argument("--preview-episodes", type=int, default=2)
    parser.add_argument("--dry-run", action="store_true", help="Validate/show recipe only; no renderer or writes")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)
    configure_logging(args.verbose)
    directory = args.output_dir.resolve()
    if args.command == "check":
        if args.dry_run:
            parser.error("check reads committed payloads; omit --dry-run")
        from .pipeline import check
        check(directory)
        return
    if args.dataset is None or args.source_manifest is None:
        parser.error("--dataset and --source-manifest are required except for check")
    if (args.episode_seed < 0 or not math.isfinite(args.min_free_gib) or args.min_free_gib < 0
            or args.workers < 1 or args.write_batch_size < 1):
        parser.error("--episode-seed/--min-free-gib must be nonnegative; --workers/--write-batch-size must be positive")
    args.source_manifest = args.source_manifest.resolve(strict=True)
    from .source import specification
    recipe, parent = specification(args, settings_from_args(args))
    recipe.pop("fingerprint")
    recipe = sealed({**recipe, "compression": args.compression})
    if args.command in ("prepare", "synthesize"):
        from .compatibility import resume_recipe
        recipe = resume_recipe(directory, recipe)
    if args.dry_run:
        rows = sum(ep["rows"] for ep in recipe["episodes"])
        print(f"{args.command}: {args.dataset}, {len(recipe['episodes']):,} episodes, {rows:,} RGB224 frames")
        print(f"Settings: {recipe['settings']}")
        print(f"Output: {directory}; recipe {recipe['fingerprint']}; no rendering requested")
        return
    from .pipeline import exclusive, prepare, preview, synthesize
    if args.command == "prepare":
        with exclusive(directory):
            prepare(directory, recipe)
    else:
        headless()
        if args.command == "preview":
            preview(directory, recipe, parent, args.preview_episodes)
        else:
            synthesize(directory, recipe, parent, args.compression, args.min_free_gib,
                       args.workers, args.write_batch_size)

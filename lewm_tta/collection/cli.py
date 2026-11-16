"""Reserve, collect and audit fresh evaluation trajectories without a model."""

import argparse
from pathlib import Path
import traceback

from ..runtime import configure_logging
from .config import headless, load_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "collect", "check", "preview-rendering"))
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--split", choices=("all", "validation", "test"), default="all")
    parser.add_argument("--regimes", nargs="+")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--source-rows", nargs="+", type=int, help="preview-rendering: released-source rows; default 0 50 100")
    parser.add_argument("--output-dir", type=Path, help="preview-rendering: new output directory")
    args = parser.parse_args()
    configure_logging(args.verbose)
    headless()
    try:
        if args.limit is not None and args.limit < 1:
            raise ValueError("limit must be positive")
        config = load_config(args.config)
        if args.command != "preview-rendering" and (args.source_rows is not None or args.output_dir is not None):
            raise ValueError("--source-rows/--output-dir belong to preview-rendering")
        if args.command == "prepare":
            from .manifest import prepare
            prepare(config)
        elif args.command == "collect":
            from .pipeline import collect
            collect(config, args.split, args.regimes, args.limit)
        elif args.command == "preview-rendering":
            if args.limit is not None or args.split != "all" or args.regimes is not None:
                raise ValueError("preview-rendering uses --source-rows; omit --limit/--split/--regimes")
            from .render_preview import preview
            preview(config, args.output_dir or Path("preview/reacher-background"),
                    args.source_rows if args.source_rows is not None else [0, 50, 100])
        else:
            if args.limit is not None or args.split != "all":
                raise ValueError("check audits the complete exported regime; omit --limit/--split")
            from .pipeline import check
            check(config, args.regimes)
    except Exception:
        traceback.print_exc()
        raise SystemExit(1)

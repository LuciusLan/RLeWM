"""Small gzip records; avoid duplicating support losses, frames or CEM populations."""

import gzip
import json
from pathlib import Path


class Writer:
    def __init__(self, directory, name, mode):
        self.path = Path(directory)/f"{name}.jsonl.gz" if mode == "compact" else None
        self.stream = None

    def __enter__(self):
        if self.path:
            self.stream = gzip.open(self.path, "xt", encoding="utf-8", compresslevel=6)
        return self

    def append(self, row):
        if self.stream:
            self.stream.write(json.dumps(row, allow_nan=False, separators=(",", ":"))+"\n")

    def flush(self):
        if self.stream:
            self.stream.flush()

    def __exit__(self, *exc):
        if self.stream:
            self.stream.close()

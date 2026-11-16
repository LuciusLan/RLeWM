"""Optional full or compressed records shared by rollout and planning stages."""

from __future__ import annotations

import gzip
import json
from pathlib import Path

from ..runtime import file_hash


RECORD_MODES = ("compact", "full", "none")


def record_mode(config: dict) -> str:
    # Older resolved configs keep their original uncompressed record behavior.
    mode = config["evaluation"].get("records", "full")
    if mode not in RECORD_MODES:
        raise ValueError(f"evaluation.records must be one of {RECORD_MODES}")
    return mode


def compact_record(row: dict) -> dict:
    """Omit optional diagnostics; retain every input to the saved summaries.

    Support IDs, weights and memberships stay available for trace validation.
    Per-item losses and per-round module breakdowns are optional diagnostics;
    round-level support losses and final module metrics remain available.
    """
    result = dict(row)
    adaptation = dict(row["adaptation"])
    adaptation.pop("initial_rank_summary", None)
    adaptation.pop("post_adaptation_rank_summary", None)
    rounds = []
    for step in adaptation["rounds"]:
        values = {key: value for key, value in step.items() if key != "module_updates"}
        if "support" in step:
            values["support"] = [{key: value for key, value in item.items()
                                  if key not in ("loss_before", "loss_after")}
                                 for item in step["support"]]
        rounds.append(values)
    adaptation["rounds"] = rounds
    result["adaptation"] = adaptation
    return result


class RecordWriter:
    def __init__(self, directory: Path, config: dict):
        self.mode = record_mode(config)
        self.path = (None if self.mode == "none" else
                     Path(directory)/("records.jsonl.gz" if self.mode == "compact" else "records.jsonl"))
        self.stream = None

    def __enter__(self):
        if self.path is not None:
            self.stream = (gzip.open(self.path, "xt", encoding="utf-8", compresslevel=6)
                           if self.mode == "compact" else self.path.open("x", encoding="utf-8"))
        return self

    def append(self, row: dict) -> dict:
        value = compact_record(row) if self.mode == "compact" else row
        if self.stream is not None:
            options = {"separators": (",", ":")} if self.mode == "compact" else {}
            self.stream.write(json.dumps(value, allow_nan=False, **options)+"\n")
        return value

    def flush(self):
        if self.stream is not None:
            self.stream.flush()

    def __exit__(self, exc_type, exc, traceback):
        if self.stream is not None:
            self.stream.close()

    def metadata(self) -> dict:
        return {"mode": self.mode, "bytes": self.path.stat().st_size if self.path else 0,
                "compression": "gzip" if self.mode == "compact" else None,
                "paired_reanalysis_available": self.path is not None,
                "omitted_diagnostics": (["per-support-item losses", "per-round module breakdowns",
                                          "optional initial/final rank summaries"] if self.mode == "compact" else []),
                "note": ("Summary-only run; per-query reanalysis and validation locking are unavailable"
                         if self.mode == "none" else "Checksummed records preserve the summary inputs")}


def open_records(path: Path):
    path = Path(path)
    return gzip.open(path, "rt", encoding="utf-8") if path.suffix == ".gz" else path.open(encoding="utf-8")


def read_saved_records(report: dict) -> list[dict]:
    if not report.get("records_file"):
        raise ValueError("This summary-only run saved no records. Use --records compact or full "
                         "for paired reanalysis and validation locking.")
    path = Path(report["records_file"])
    if not path.is_file() or not path.stat().st_size or file_hash(path) != report["records_sha256"]:
        raise ValueError("Missing, empty, or changed saved records")
    with open_records(path) as stream:
        records = [json.loads(line) for line in stream if line.strip()]
    if len(records) != report["record_count"]:
        raise ValueError("Saved record count differs from report")
    return records


def check_record_roundtrip(records: list[dict], config: dict, summarize) -> None:
    """User-run fixture: serialized compact/full records reproduce the summary."""
    from tempfile import TemporaryDirectory

    reference = summarize(records, config)
    with TemporaryDirectory(prefix="lewm-record-check-") as directory:
        for mode in RECORD_MODES:
            output = Path(directory)/mode
            output.mkdir()
            selected = {**config, "evaluation": {**config["evaluation"], "records": mode}}
            with RecordWriter(output, selected) as writer:
                retained = [writer.append(row) for row in records]
            if summarize(retained, config) != reference:
                raise AssertionError("Record projection changed evaluation statistics")
            report = {"records_file": str(writer.path) if writer.path else None,
                      "records_sha256": file_hash(writer.path) if writer.path else None,
                      "record_count": len(records)}
            if mode == "none":
                if list(output.iterdir()):
                    raise AssertionError("Summary-only mode unexpectedly wrote records")
                try:
                    read_saved_records(report)
                except ValueError:
                    pass
                else:
                    raise AssertionError("Summary-only runs must reject record reanalysis")
            else:
                loaded = read_saved_records(report)
                if summarize(loaded, config) != reference:
                    raise AssertionError("Record round-trip changed evaluation statistics")
                with writer.path.open("ab") as stream:
                    stream.write(b"changed")
                try:
                    read_saved_records(report)
                except ValueError:
                    pass
                else:
                    raise AssertionError("Record checksum corruption was not rejected")

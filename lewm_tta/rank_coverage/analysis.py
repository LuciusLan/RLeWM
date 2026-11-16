"""Checksummed reanalysis of compact pilot records, without Torch or model imports."""

from __future__ import annotations

import gzip
import json
from pathlib import Path

from ..runtime import file_hash, read_json, write_json
from .statistics import summarize, write_tables


def validate_matrix(records: list[dict], value: dict, expected_queries: int | None = None) -> dict:
    seen, queries, sizes, counts = set(), {}, set(), {}
    expected_seeds = {mode: set(value["seeds"][:1] if mode == "recorded-action" else value["seeds"])
                      for mode in value["query_modes"]}
    for row in records:
        mode, query, seed = row["mode"], row["query"]["transition_id"], row["seed"]
        key = (mode, query, seed)
        if key in seen or mode not in expected_seeds or seed not in expected_seeds[mode]:
            raise ValueError("Repeated or unexpected coverage action group")
        seen.add(key)
        episode = row["query"]["source_episode_position"]
        if query in queries and queries[query] != episode:
            raise ValueError("A transition ID appears in multiple source episodes")
        queries[query] = episode
        counts.setdefault((mode, query), set()).add(seed)
        required_pairs = 1 if mode == "recorded-action" else value["provisional"]["provisional_actions"]
        if len(row["pairs"]) != required_pairs or len(row["actions_normalized"]) != required_pairs:
            raise ValueError("Coverage action-pair count differs from protocol")
        for pair in row["pairs"]:
            sizes.add(pair["memory_size"])
            if [item["percent"] for item in pair["full_corpus"]] != value["percentages"]:
                raise ValueError("Record threshold grid differs from protocol")
            if pair["full_corpus"][-1]["intersection_count"] != pair["memory_size"]:
                raise ValueError("The 100% full-corpus sanity baseline failed")
    if not seen or len(sizes) != 1:
        raise ValueError("Empty records or inconsistent retrieval corpus sizes")
    if expected_queries is not None and len(queries) != expected_queries:
        raise ValueError("Coverage query count differs from the declared cohort")
    for mode, seeds in expected_seeds.items():
        if any(counts.get((mode, query)) != seeds for query in queries):
            raise ValueError("Coverage seed/mode/query matrix is incomplete")
    return {"groups": len(seen), "queries": len(queries), "episodes": len(set(queries.values())),
            "memory_windows": next(iter(sizes)), "complete_record_matrix": True}


def analyze(report_path: Path, output: Path | None = None) -> dict:
    report_path = Path(report_path).resolve()
    report = read_json(report_path)
    if report.get("stage") != "rank-coverage" or not report.get("completed"):
        raise ValueError("Analysis requires a completed rank-coverage report")
    records_path = Path(report["records_file"])
    if file_hash(records_path) != report["records_sha256"]:
        raise ValueError("Coverage record checksum mismatch")
    with gzip.open(records_path, "rt", encoding="utf-8") as stream:
        records = [json.loads(line) for line in stream if line.strip()]
    if len(records) != report["record_count"]:
        raise ValueError("Coverage record count differs from the completed report")
    matrix = validate_matrix(records, report["config"])
    if matrix["memory_windows"] != report["memory_windows"]:
        raise ValueError("Coverage corpus size differs from the completed report")
    episodes = sorted({row["query"]["source_episode_position"] for row in records})
    if "query_episode_positions" in report and episodes != report["query_episode_positions"]:
        raise ValueError("Coverage query episodes differ from the completed report")
    if any(matrix["queries"] != group["queries"] or matrix["episodes"] != group["episodes"]
           for group in report.get("summary", {}).values()):
        raise ValueError("Coverage query/episode counts differ from the completed summary")
    directory = Path(output).resolve() if output else report_path.parent / "reanalysis"
    directory.mkdir(parents=True, exist_ok=False)
    summary = summarize(records, report["config"])
    result = {**report, "summary": summary, "source_report": str(report_path),
              "source_report_sha256": file_hash(report_path), "reanalysis_matrix_check": matrix,
              "analysis_code": {path.name: file_hash(path) for path in Path(__file__).parent.glob("*.py")}}
    write_tables(summary, report["config"], directory)
    write_json(directory / "report.json", result)
    print("Recomputed coverage:", directory / "report.json", flush=True)
    return result

"""Top-k rank statistics and CPU-only analysis of saved evaluation records."""

from __future__ import annotations

from collections import defaultdict
import hashlib
import json
from pathlib import Path
from statistics import fmean

from .rank_diagnostics import candidate_cross_ranks, summarize_saved_topk
from .runtime import file_hash, read_json, write_json


def summarize_topk_records(records, statistics, *, k: int | None = None, phase: str = "initial") -> dict:
    """Keep episode clusters; average seeds per query for discrepancy/fractions."""
    if phase not in ("initial", "final"):
        raise ValueError("Rank phase must be initial or final")
    grouped = defaultdict(lambda: defaultdict(list))
    key = f"rank_diagnostics_{phase}"
    for record in records:
        diagnostic = record[key]
        if diagnostic["memory_size"] and "forward_candidates_exact_cross_ranks" in diagnostic:
            grouped[record["method"]][record["query"]["transition_id"]].append(record)
    result = {}
    for method, queries in grouped.items():
        ranks = {direction: [] for direction in ("forward", "reverse")}
        rank_episodes = {direction: [] for direction in ranks}
        fractions = {direction: [] for direction in ranks}
        counts = {direction: [] for direction in ranks}
        query_episodes, gaps, requested, rules, seed_counts = [], [], set(), set(), set()
        for group in queries.values():
            episode = group[0]["query"]["source_episode_position"]
            if any(record["query"]["source_episode_position"] != episode for record in group):
                raise ValueError("A query ID cannot span multiple episodes")
            seed_counts.add(len(group))
            # Initial retrieval is deterministic and identical across seeds.
            # Final retrieval can change with a seed's adaptation support.
            snapshots = group[:1] if phase == "initial" else group
            summaries = []
            for record in snapshots:
                diagnostic = record[key]
                summary = summarize_saved_topk(diagnostic, k)
                summaries.append(summary)
                requested.add(summary["requested_k"])
                rules.add(summary["selection_rule"])
                for direction in ranks:
                    candidates = candidate_cross_ranks(diagnostic[f"{direction}_candidates_exact_cross_ranks"],
                        direction, diagnostic["memory_size"], summary["requested_k"])
                    ranks[direction].extend(row["opposite_rank"] for row in candidates)
                    rank_episodes[direction].extend([episode] * len(candidates))
            query_episodes.append(episode)
            gaps.append(fmean(summary["mean_bidirectional_absolute_log2_rank_ratio"] for summary in summaries))
            for direction in ranks:
                fractions[direction].append(fmean(summary[direction]["opposite_topk_fraction"] for summary in summaries))
                counts[direction].append(fmean(summary[direction]["count"] for summary in summaries))
        result[method] = {"phase": phase, "queries": len(queries), "requested_k_values": sorted(requested),
            "selection_rules": sorted(rules), "seeds_per_query": sorted(seed_counts),
            "forward_candidates_in_reverse": statistics(ranks["forward"], rank_episodes["forward"]),
            "reverse_candidates_in_forward": statistics(ranks["reverse"], rank_episodes["reverse"]),
            "bidirectional_mean_absolute_log2_rank_ratio": statistics(gaps, query_episodes),
            "forward_opposite_topk_fraction": statistics(fractions["forward"], query_episodes),
            "reverse_opposite_topk_fraction": statistics(fractions["reverse"], query_episodes),
            "forward_candidate_count": statistics(counts["forward"], query_episodes),
            "reverse_candidate_count": statistics(counts["reverse"], query_episodes),
            "aggregation": "opposite ranks pooled descriptively; gaps/fractions average seeds per query; episode bootstrap"}
    return result


def format_topk_analysis(phases: dict) -> str:
    lines = []
    for phase, methods in phases.items():
        if not methods:
            continue
        lines.append(f"Exact {phase} opposite-direction ranks over the complete memory")
        lines.append("Method           k       Median F->R  Median R->F  Mean |log2 R/F|  F in R top-k  R in F top-k")
        for method, row in methods.items():
            k_label = ",".join(str(value) for value in row["requested_k_values"])
            lines.append(f"{method:<16} {k_label:<7} {row['forward_candidates_in_reverse']['median']:<12.6g} "
                f"{row['reverse_candidates_in_forward']['median']:<12.6g} "
                f"{row['bidirectional_mean_absolute_log2_rank_ratio']['mean']:<16.6g} "
                f"{row['forward_opposite_topk_fraction']['mean']:<13.1%} "
                f"{row['reverse_opposite_topk_fraction']['mean']:.1%}")
        for rule in sorted({rule for row in methods.values() for rule in row["selection_rules"]}):
            lines.append(f"Candidate selection: {rule}")
    return "\n".join(lines)


def analyze_saved_ranks(report_path: Path, *, k: int | None = None, phase: str = "both",
                        methods: list[str] | None = None, output: Path | None = None) -> dict:
    """Read predictions/ranks already scored by the user; no model or GPU work."""
    from .reporting import episode_statistics
    if k is not None and k < 1:
        raise ValueError("k must be positive")
    if phase not in ("initial", "final", "both"):
        raise ValueError("phase must be initial, final, or both")
    report_path = Path(report_path).resolve()
    report = read_json(report_path)
    if not report.get("completed"):
        raise ValueError("Rank analysis requires a completed evaluation report")
    output = Path(output).resolve() if output else report_path.with_name(f"rank-topk-{k or 'saved'}-{phase}.json")
    if output.exists():
        raise FileExistsError(f"Analysis output already exists: {output}; choose a different --output")
    records_path = Path(report["records_file"])
    checksum, records, seen = hashlib.sha256(), [], 0
    with records_path.open("rb") as stream:
        for line in stream:
            checksum.update(line)
            if not line.strip():
                continue
            record = json.loads(line)
            seen += 1
            if methods is not None and record["method"] not in methods:
                continue
            records.append({key: record[key] for key in ("method", "seed", "query",
                "rank_diagnostics_initial", "rank_diagnostics_final")})
    if checksum.hexdigest() != report["records_sha256"]:
        raise ValueError("Saved evaluation records no longer match their report checksum")
    if not records:
        raise ValueError("No saved records match the requested methods")
    available = {record["method"] for record in records}
    if methods is not None and set(methods) - available:
        raise ValueError(f"Methods missing from saved run: {sorted(set(methods)-available)}")
    phases = ("initial", "final") if phase == "both" else (phase,)
    statistics = lambda values, episodes: episode_statistics(values, episodes, report["config"])
    summaries = {name: summarize_topk_records(records, statistics, k=k, phase=name) for name in phases}
    if not any(summaries.values()):
        raise ValueError("The run does not contain saved directional candidate ranks")
    result = {"kind": "saved_record_topk_rank_analysis", "source_report": str(report_path),
        "source_report_sha256": file_hash(report_path), "source_records": str(records_path),
        "source_records_sha256": checksum.hexdigest(), "source_split": report["split"],
        "source_manifest_fingerprint": report["manifest_fingerprint"],
        "source_prepared_fingerprint": report["prepared_fingerprint"], "source_records_seen": seen,
        "records_analyzed": len(records), "parameters": {"k": k, "phase": phase, "methods": methods},
        "analysis_code": {name: file_hash(Path(__file__).parent / name)
                          for name in ("rank_reporting.py", "rank_diagnostics.py", "reporting.py")},
        "phases": summaries,
        "limits": ["Only saved directional candidates are analyzed; absent neighbors cannot be reconstructed",
                   "Archived support-filtered candidates retain their original selection rule",
                   "This is derived analysis, not a new evaluation or validation lock"]}
    write_json(output, result)
    print(format_topk_analysis(summaries) + f"\nRank analysis: {output}", flush=True)
    return result

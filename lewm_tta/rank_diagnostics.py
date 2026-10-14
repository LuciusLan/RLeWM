"""Compare both ranks of the same transition under the same query.

Inputs must cover the full, identical memory set. Query targets are not accepted.
"""

from __future__ import annotations

import math
from statistics import fmean, median
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch


def exact_distance_rank(distances: torch.Tensor, candidate_index: int) -> int:
    """One-based whole-corpus rank, with stable memory-row tie breaking.

    A candidate can be arbitrarily far outside top-k. Counting better scores
    certifies its rank without requesting or sorting a complete result list.
    """
    import torch
    with torch.no_grad():
        if distances.ndim != 1 or not 0 <= candidate_index < len(distances):
            raise ValueError("Candidate must index a one-dimensional distance array")
        if not torch.isfinite(distances).all():
            raise ValueError("Exact ranks require finite distances")
        threshold = distances[candidate_index]
        better = int((distances < threshold).sum())
        tied_before = int((distances[:candidate_index] == threshold).sum())
        return 1 + better + tied_before


def candidate_cross_ranks(rows: list[dict], direction: str, memory_size: int, k: int) -> list[dict]:
    """Enrich separate directional candidates using their exact global ranks."""
    if direction not in ("forward", "reverse") or k < 1 or memory_size < 1:
        raise ValueError("Cross-ranks require a direction, positive k, and nonempty memory")
    result, ids, previous_rank = [], set(), 0
    for position, original in enumerate(rows[:k], 1):
        row = dict(original)
        ranks = [row["forward_rank"], row["reverse_rank"]]
        if any(isinstance(rank, bool) or not isinstance(rank, int) or not 1 <= rank <= memory_size for rank in ranks):
            raise ValueError("Candidate ranks must be one-based integers within the complete memory")
        own = row[f"{direction}_rank"]
        opposite = row["reverse_rank" if direction == "forward" else "forward_rank"]
        if row["transition_id"] in ids or own <= previous_rank:
            raise ValueError("Directional candidates must be unique and ordered by own rank")
        ids.add(row["transition_id"])
        previous_rank = own
        signed_gap = math.log2(row["reverse_rank"]/row["forward_rank"])
        row.update(selected_by=direction, selected_position=position, own_rank=own, opposite_rank=opposite,
                   opposite_rank_percentile=(opposite-1)/(memory_size-1) if memory_size > 1 else 0.0,
                   signed_log2_rank_ratio=signed_gap, absolute_log2_rank_ratio=abs(signed_gap),
                   in_opposite_topk=opposite <= min(k, memory_size))
        result.append(row)
    return result


def topk_cross_rank_summary(forward_rows: list[dict], reverse_rows: list[dict], memory_size: int,
                           k: int, selection_rule: str = "raw_directional_topk") -> dict:
    """Each candidate's opposite rank is global; no top-k union is formed."""
    if k < 1:
        raise ValueError("Diagnostic top-k must be positive")
    if memory_size < 1:
        return {"requested_k": k, "memory_size": 0, "reason": "empty_memory"}
    def describe(rows, direction):
        candidates = candidate_cross_ranks(rows, direction, memory_size, k)
        if not candidates:
            raise ValueError("Nonempty memory requires saved candidates in both directions")
        if selection_rule == "raw_directional_topk" and [row["own_rank"] for row in candidates] != list(range(1, len(candidates)+1)):
            raise ValueError("Raw top-k candidates must be a complete own-rank prefix")
        opposite = sorted(row["opposite_rank"] for row in candidates)
        position = (len(opposite)-1) * .9
        lower, upper = math.floor(position), math.ceil(position)
        p90 = opposite[lower] + (opposite[upper]-opposite[lower]) * (position-lower)
        return {"count": len(candidates), "shortfall": max(0, min(k, memory_size)-len(candidates)),
                "mean_opposite_rank": fmean(opposite), "median_opposite_rank": median(opposite),
                "p90_opposite_rank": p90,
                "mean_opposite_rank_percentile": fmean(row["opposite_rank_percentile"] for row in candidates),
                "mean_signed_log2_rank_ratio": fmean(row["signed_log2_rank_ratio"] for row in candidates),
                "mean_absolute_log2_rank_ratio": fmean(row["absolute_log2_rank_ratio"] for row in candidates),
                "opposite_topk_fraction": fmean(row["in_opposite_topk"] for row in candidates)}
    forward, reverse = describe(forward_rows, "forward"), describe(reverse_rows, "reverse")
    return {"requested_k": k, "memory_size": memory_size, "selection_rule": selection_rule,
            "rank_scope": "complete identical memory; ties by memory row",
            "forward": forward, "reverse": reverse,
            "mean_bidirectional_absolute_log2_rank_ratio":
                (forward["mean_absolute_log2_rank_ratio"]+reverse["mean_absolute_log2_rank_ratio"])/2}


def summarize_saved_topk(diagnostic: dict, k: int | None = None) -> dict:
    """Use archived ranks as-is; never label a filtered prefix as raw top-k."""
    rows_f = diagnostic["forward_candidates_exact_cross_ranks"]
    rows_r = diagnostic["reverse_candidates_exact_cross_ranks"]
    metadata = diagnostic.get("topk_cross_ranks", diagnostic.get("supplementary_topk", {}))
    requested = k if k is not None else metadata.get("requested_k", metadata.get("k", min(len(rows_f), len(rows_r))))
    if k is not None and min(len(rows_f), len(rows_r)) < min(k, diagnostic["memory_size"]):
        raise ValueError(f"Only {min(len(rows_f), len(rows_r))} candidates per direction were saved; "
                         f"cannot reconstruct k={k}. A new evaluation with --rank-topk {k} is needed.")
    selection = metadata.get("selection_rule", "archived_directional_candidates")
    return topk_cross_rank_summary(rows_f, rows_r, diagnostic["memory_size"], requested, selection)


def top1_cross_rank_summary(forward_candidate: dict, reverse_candidate: dict, memory_size: int) -> dict:
    """Compare the forward winner in reverse and the reverse winner in forward."""
    if memory_size < 1:
        return {"memory_size": 0, "reason": "empty_memory"}
    candidates = []
    for direction, original in (("forward", forward_candidate), ("reverse", reverse_candidate)):
        row = dict(original)
        if row[f"{direction}_rank"] != 1:
            raise ValueError("A direction's top-1 candidate must have own rank one")
        opposite = row["reverse_rank" if direction == "forward" else "forward_rank"]
        if not 1 <= opposite <= memory_size:
            raise ValueError("Opposite rank lies outside the complete memory")
        row.update(selected_by=direction, opposite_rank=opposite,
                   opposite_rank_percentile=(opposite-1)/(memory_size-1) if memory_size > 1 else 0.0,
                   log2_opposite_rank=math.log2(opposite),
                   signed_log2_rank_ratio=math.log2(row["reverse_rank"]/row["forward_rank"]))
        candidates.append(row)
    forward, reverse = candidates
    return {"memory_size": memory_size,
            "comparison": "exact opposite-direction rank of each direction's top-1 observed transition",
            "forward_top1": forward, "reverse_top1": reverse,
            "forward_top1_reverse_rank": forward["reverse_rank"],
            "reverse_top1_forward_rank": reverse["forward_rank"],
            "mean_top1_log2_opposite_rank": (forward["log2_opposite_rank"]+reverse["log2_opposite_rank"])/2,
            "same_top1_transition": forward["transition_id"] == reverse["transition_id"]}


def cross_rank_diagnostics(forward_ids, reverse_ids, k: int = 32) -> dict:
    forward, reverse = list(forward_ids), list(reverse_ids)
    if len(set(forward)) != len(forward) or len(set(reverse)) != len(reverse) or set(forward) != set(reverse):
        raise ValueError("Rankings must be permutations of the same complete memory IDs")
    if k < 1:
        raise ValueError("k must be positive")
    if not forward:
        return {"memory_size": 0, "k": 0, "rows": [], "reason": "empty_memory"}
    k = min(k, len(forward))
    rank_f = {item: rank for rank, item in enumerate(forward, 1)}
    rank_r = {item: rank for rank, item in enumerate(reverse, 1)}
    top_f, top_r = set(forward[:k]), set(reverse[:k])
    def row(item):
        return {"transition_id": item, "forward_rank": rank_f[item], "reverse_rank": rank_r[item]}
    result = top1_cross_rank_summary(row(forward[0]), row(reverse[0]), len(forward))
    result["forward_candidates_exact_cross_ranks"] = candidate_cross_ranks([row(item) for item in forward[:k]], "forward", len(forward), k)
    result["reverse_candidates_exact_cross_ranks"] = candidate_cross_ranks([row(item) for item in reverse[:k]], "reverse", len(forward), k)
    result["topk_cross_ranks"] = topk_cross_rank_summary(result["forward_candidates_exact_cross_ranks"],
        result["reverse_candidates_exact_cross_ranks"], len(forward), k)
    result["supplementary_topk"] = {"k": k, "overlap_fraction": len(top_f & top_r)/k,
        "jaccard": len(top_f & top_r)/len(top_f | top_r),
        "median_reverse_rank_of_forward_topk": median(rank_r[item] for item in forward[:k]),
        "median_forward_rank_of_reverse_topk": median(rank_f[item] for item in reverse[:k])}
    return result


def consensus_indices(forward_indices, reverse_indices, budget: int) -> list[int]:
    """Minimize worst rank over complete memory; break ties by sum, then ID."""
    forward, reverse = list(forward_indices), list(reverse_indices)
    if set(forward) != set(reverse) or len(set(forward)) != len(forward) or len(set(reverse)) != len(reverse):
        raise ValueError("Consensus requires complete rankings over one memory set")
    if budget < 1:
        raise ValueError("Consensus requires a positive support budget")
    rank_f = {item: rank for rank, item in enumerate(forward, 1)}
    rank_r = {item: rank for rank, item in enumerate(reverse, 1)}
    return sorted(forward, key=lambda item: (max(rank_f[item], rank_r[item]),
                  rank_f[item] + rank_r[item], item))[:budget]

"""Compare both ranks of the same transition under the same query.

Inputs must cover the full, identical memory set. Query targets are not accepted.
"""

from __future__ import annotations

import math
from statistics import mean, median


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
    candidates = list(dict.fromkeys(forward[:k] + reverse[:k]))
    rows = [{"transition_id": item, "forward_rank": rank_f[item], "reverse_rank": rank_r[item],
             "signed_log2_rank_ratio": math.log2(rank_r[item] / rank_f[item]),
             "absolute_log2_rank_gap": abs(math.log2(rank_r[item] / rank_f[item])),
             "worst_rank": max(rank_f[item], rank_r[item])} for item in candidates]
    return {"memory_size": len(forward), "k": k,
            "overlap_fraction": len(top_f & top_r) / k,
            "jaccard": len(top_f & top_r) / len(top_f | top_r),
            "forward_top1_reverse_rank": rank_r[forward[0]],
            "reverse_top1_forward_rank": rank_f[reverse[0]],
            "median_reverse_rank_of_forward_topk": median(rank_r[item] for item in forward[:k]),
            "median_forward_rank_of_reverse_topk": median(rank_f[item] for item in reverse[:k]),
            "mean_absolute_log2_rank_gap": mean(row["absolute_log2_rank_gap"] for row in rows),
            "rows": rows}


def consensus_indices(forward_indices, reverse_indices, budget: int, pool_size: int) -> list[int]:
    """Minimize worst rank within the top-k union; break ties by sum, then ID."""
    forward, reverse = list(forward_indices), list(reverse_indices)
    if set(forward) != set(reverse) or len(set(forward)) != len(forward) or len(set(reverse)) != len(reverse):
        raise ValueError("Consensus requires complete rankings over one memory set")
    if budget < 1 or pool_size < budget:
        raise ValueError("Consensus pool must cover a positive support budget")
    rank_f = {item: rank for rank, item in enumerate(forward, 1)}
    rank_r = {item: rank for rank, item in enumerate(reverse, 1)}
    candidates = set(forward[:pool_size]) | set(reverse[:pool_size])
    return sorted(candidates, key=lambda item: (max(rank_f[item], rank_r[item]),
                  rank_f[item] + rank_r[item], item))[:budget]

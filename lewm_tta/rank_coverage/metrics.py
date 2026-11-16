"""Exact rank coverage and counterfactual support retention, without model code."""

from __future__ import annotations

import math
import numpy as np


def cutoff(memory_size: int, percent: float) -> int:
    if memory_size < 1 or not math.isfinite(percent) or not 0 < percent <= 100:
        raise ValueError("Require nonempty memory and 0 < percentage <= 100 (percent units)")
    return min(memory_size, max(1, math.ceil(memory_size * percent / 100)))


def pair_record(forward_ranks, reverse_ranks, forward_prefix, reverse_prefix,
                consensus_indices, percentages: list[float]) -> dict:
    """Store small raw prefixes/supports and full-corpus intersection counts only."""
    forward, reverse = np.asarray(forward_ranks), np.asarray(reverse_ranks)
    size = len(forward)
    if size < 1 or forward.shape != reverse.shape or forward.ndim != 1:
        raise ValueError("Directional ranks must describe the same nonempty corpus")
    for ranks in (forward, reverse):
        if ranks.dtype.kind not in "iu" or ranks.min() != 1 or ranks.max() != size:
            raise ValueError("Ranks must be one-based integer permutations")
        if not np.all(np.bincount(ranks.astype(np.int64), minlength=size + 1)[1:] == 1):
            raise ValueError("Directional rank arrays contain repeated or missing ranks")
    worst = np.maximum(forward, reverse)
    histogram = np.bincount(worst.astype(np.int64), minlength=size + 1)
    cumulative = np.cumsum(histogram)

    def rows(indices):
        return [[int(i), int(forward[i]), int(reverse[i])] for i in indices]

    fp, rp = rows(forward_prefix), rows(reverse_prefix)
    if ([row[1] for row in fp] != list(range(1, len(fp) + 1)) or
            [row[2] for row in rp] != list(range(1, len(rp) + 1))):
        raise ValueError("Saved directional candidates must be raw, exact own-rank prefixes")
    consensus = rows(consensus_indices)
    if len({row[0] for row in consensus}) != len(consensus):
        raise ValueError("Consensus support contains duplicate windows")
    return {"memory_size": size, "row_columns": ["memory_index", "forward_rank", "reverse_rank"],
            "forward_prefix": fp, "reverse_prefix": rp, "consensus_support": consensus,
            "full_corpus": [{"percent": float(p), "cutoff": cutoff(size, p),
                             "intersection_count": int(cumulative[cutoff(size, p)])} for p in percentages]}


def pair_metrics(record: dict, prefix_sizes: list[int], budget: int) -> dict[str, float]:
    """All values are per action pair; they are averaged within query before CIs."""
    size, result = record["memory_size"], {}
    consensus = record["consensus_support"]
    result["pair/consensus_original_count"] = float(len(consensus))
    result["pair/consensus_cover_all_rank_percent"] = (
        100 * max(max(row[1:]) for row in consensus) / size if consensus else 100.0)
    for requested in prefix_sizes:
        directional = {name: record[f"{name}_prefix"][:min(requested, size)] for name in ("forward", "reverse")}
        if any(len(rows) != min(requested, size) for rows in directional.values()):
            raise ValueError("Saved rank prefix is too short for the requested analysis")
        for name, rows in directional.items():
            opposite = [row[2 if name == "forward" else 1] for row in rows]
            for label, quantile in (("median", .5), ("p90", .9), ("p95", .95), ("max", 1.)):
                value = float(np.quantile(opposite, quantile))
                result[f"topk/{requested}/{name}_opposite_{label}_rank"] = value
                result[f"topk/{requested}/{name}_opposite_{label}_percent"] = 100 * value / size
        for threshold in record["full_corpus"]:
            percent, top = threshold["percent"], threshold["cutoff"]
            key = f"p/{percent:g}/topk/{requested}"
            passed = {name: {row[0] for row in rows if row[2 if name == "forward" else 1] <= top}
                      for name, rows in directional.items()}
            for name in passed:
                result[f"{key}/{name}_coverage"] = len(passed[name]) / len(directional[name])
                result[f"{key}/{name}_survivors"] = float(len(passed[name]))
            result[f"{key}/forward_minus_reverse_coverage"] = (
                result[f"{key}/forward_coverage"] - result[f"{key}/reverse_coverage"])
            result[f"{key}/gated_seed_union_count"] = float(len(passed["forward"] | passed["reverse"]))
            result[f"{key}/gated_seed_intersection_count"] = float(len(passed["forward"] & passed["reverse"]))
    for threshold in record["full_corpus"]:
        percent, top, count = threshold["percent"], threshold["cutoff"], threshold["intersection_count"]
        key = f"p/{percent:g}/pair"
        if top != cutoff(size, percent) or not max(0, 2 * top - size) <= count <= top:
            raise ValueError("Invalid full-corpus intersection count")
        retained = sum(max(row[1:]) <= top for row in consensus)
        result.update({f"{key}/mutual_pool_count": float(count), f"{key}/mutual_top_percent_coverage": count / top,
                       f"{key}/mutual_jaccard": count / (2 * top - count),
                       f"{key}/mutual_pool_has_budget": float(count >= budget),
                       f"{key}/consensus_retained_count": float(retained),
                       f"{key}/consensus_budget_fill_fraction": retained / budget,
                       f"{key}/consensus_full_batch": float(retained >= budget),
                       f"{key}/consensus_zero_support": float(retained == 0)})
        if "consensus_sources" in record:
            selected = {row[0] for row in consensus if max(row[1:]) <= top}
            result[f"{key}/consensus_retained_episode_count"] = float(len({
                row["source_episode_position"] for row in record["consensus_sources"] if row["index"] in selected}))
    return result


def validate_ledger(ledger: list[dict], size: int) -> None:
    if not ledger or len({row["index"] for row in ledger}) != len(ledger):
        raise ValueError("FR support must be nonempty and unique")
    total = 0.
    for row in ledger:
        origins = row["origins"]
        if not origins or not math.isfinite(row["weight"]) or row["weight"] <= 0:
            raise ValueError("FR sample has no positive loss contribution")
        if not math.isclose(sum(item["loss_contribution"] for item in origins), row["weight"], abs_tol=1e-10):
            raise ValueError("Recorded channel contributions do not reproduce the original FR weight")
        for item in origins:
            if item["direction"] not in ("forward", "reverse") or any(
                    not 1 <= item[key] <= size for key in ("forward_rank", "reverse_rank")):
                raise ValueError("Invalid origin direction or global rank")
            if not math.isfinite(item["loss_contribution"]) or item["loss_contribution"] <= 0:
                raise ValueError("Loss contributions must be positive")
        total += row["weight"]
    if not math.isclose(total, 1., abs_tol=1e-10):
        raise ValueError("Original FR support weights must sum to one")


def fr_metrics(ledger: list[dict], size: int, percentages: list[float], budget: int) -> dict[str, float]:
    """Mask the ORIGINAL FR support, without refill, reweighting or training."""
    validate_ledger(ledger, size)
    result = {"fr/original_count": float(len(ledger)),
              "fr/cover_all_samples_opposite_rank_percent": 100 * max(
                  min(origin["reverse_rank" if origin["direction"] == "forward" else "forward_rank"]
                      for origin in row["origins"]) for row in ledger) / size,
              "fr/cover_all_channel_mass_rank_percent": 100 * max(
                  origin["reverse_rank" if origin["direction"] == "forward" else "forward_rank"]
                  for row in ledger for origin in row["origins"]) / size}
    for percent in percentages:
        top, key = cutoff(size, percent), f"p/{percent:g}/fr"
        count, strict, sample_mass, channel_mass = 0, 0, 0., 0.
        retained_episodes = set()
        counts = {name: [0, 0] for name in ("forward", "reverse")}
        for row in ledger:
            passes, both = [], []
            for origin in row["origins"]:
                direction = origin["direction"]
                passed = origin["reverse_rank" if direction == "forward" else "forward_rank"] <= top
                counts[direction][0] += int(passed)
                counts[direction][1] += 1
                channel_mass += origin["loss_contribution"] if passed else 0.
                passes.append(passed)
                both.append(max(origin["forward_rank"], origin["reverse_rank"]) <= top)
            count += int(any(passes))
            strict += int(any(both))
            sample_mass += row["weight"] if any(passes) else 0.
            if any(passes):
                retained_episodes.add(row["source_episode_position"])
        result.update({f"{key}/unique_retained_count": float(count),
                       f"{key}/unique_retained_fraction": count / len(ledger),
                       f"{key}/full_budget_survives": float(count >= budget),
                       f"{key}/all_original_samples_survive": float(count == len(ledger)),
                       f"{key}/zero_support": float(count == 0),
                       f"{key}/original_sample_weight_retained": sample_mass,
                       f"{key}/channel_loss_mass_retained": channel_mass,
                       f"{key}/retained_episode_count": float(len(retained_episodes)),
                       f"{key}/both_ranks_unique_retained_count": float(strict)})
        for direction, (passed, total) in counts.items():
            if total:
                result[f"{key}/{direction}_origin_coverage"] = passed / total
    return result


def group_metrics(record: dict, prefix_sizes: list[int], percentages: list[float], budget: int) -> dict[str, float]:
    pair_values = [pair_metrics(pair, prefix_sizes, budget) for pair in record["pairs"]]
    if not pair_values or any(set(row) != set(pair_values[0]) for row in pair_values):
        raise ValueError("An action group needs matching pair measurements")
    result = {key: float(np.mean([row[key] for row in pair_values])) for key in pair_values[0]}
    result.update(fr_metrics(record["fr_fixed_support"], record["pairs"][0]["memory_size"], percentages, budget))
    return result

"""Exact, scaled retrieval over observed memory transitions only."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

import torch

from .cache import LatentBank
from .model import Context
from .rank_diagnostics import candidate_cross_ranks, top1_cross_rank_summary, topk_cross_rank_summary


@dataclass
class Ranking:
    indices: np.ndarray                   # complete permutation, best first
    distances: torch.Tensor               # indexed by memory row, not rank
    components: dict[str, torch.Tensor]    # raw component MSE
    _ranks: np.ndarray | None = None

    def __len__(self):
        return len(self.indices)

    @property
    def order(self):
        return self.indices

    def rank_of(self, index: int) -> int:
        return int(self.ranks[index])

    @property
    def ranks(self):
        if self._ranks is None:
            self._ranks = np.empty(len(self.indices), dtype=np.int64)
            self._ranks[self.indices] = np.arange(1, len(self.indices)+1, dtype=np.int64)
        return self._ranks


@dataclass(frozen=True)
class Support:
    indices: list[int]
    weights: list[float]
    channels: dict[int, list[str]]
    details: dict = field(default_factory=dict)


def calibrate_scales(memory: LatentBank, pairs: int, seed: int, floor: float) -> dict:
    if memory.split != "memory":
        raise ValueError("Distance calibration may only use memory")
    fields = {"context": memory.contexts, "past_actions": memory.past_actions,
              "action": memory.actions, "successor": memory.successors}
    if len(memory) < 2:
        return {"scales": {name: 1.0 for name in fields}, "pairs": 0,
                "rule": "unit fallback for fewer than two memory windows"}
    generator = torch.Generator(device="cpu").manual_seed(seed)
    first = torch.randint(len(memory), (pairs,), generator=generator)
    second = (first + torch.randint(1, len(memory), (pairs,), generator=generator)) % len(memory)
    scales, nonzero = {}, {}
    for name, values in fields.items():
        difference = (values[first] - values[second]).flatten(1)
        distances = difference.square().mean(1) if difference.shape[1] else torch.zeros(pairs)
        positive = distances[distances > floor]
        scales[name] = max(float(positive.median()), floor) if len(positive) else 1.0
        nonzero[name] = len(positive)
    return {"scales": scales, "nonzero_pairs": nonzero, "pairs": pairs, "seed": seed,
            "rule": "median nonzero raw MSE of distinct memory pairs", "floor": floor}


class ExactRetriever:
    def __init__(self, memory: LatentBank, calibration: dict, settings: dict):
        if memory.split != "memory":
            raise ValueError("Retrieval cannot index validation or test transitions")
        self.memory, self.scales, self.settings = memory, calibration["scales"], settings
        self.episode_positions = memory.episode_positions
        self.step_positions = memory.step_positions
        episodes, counts = np.unique(self.episode_positions, return_counts=True)
        self.episode_set = set(episodes.tolist())
        self.episode_cap = settings.get("max_windows_per_episode")
        self.min_step_gap = settings.get("min_start_step_gap", 0)
        self.max_support = int(np.minimum(counts, self.episode_cap).sum()) if self.episode_cap else len(memory)

    def _mse(self, values: torch.Tensor, query: torch.Tensor) -> torch.Tensor:
        pieces = []
        for start in range(0, len(values), self.settings["chunk_size"]):
            difference = (values[start:start+self.settings["chunk_size"]] - query).flatten(1)
            pieces.append(difference.square().mean(1) if difference.shape[1] else
                          torch.zeros(len(difference), device=values.device))
        return torch.cat(pieces) if pieces else torch.empty(0, device=values.device)

    def _context(self, context: Context) -> dict:
        if context.latents.shape[0] != 1:
            raise ValueError("ExactRetriever currently accepts one independent query")
        return {"context": self._mse(self.memory.contexts, context.latents),
                "past_actions": self._mse(self.memory.past_actions, context.past_actions)}

    def _rank(self, components: dict, extra: str | None = None, beta: float = 1.0) -> Ranking:
        distance = components["context"] / self.scales["context"]
        distance = distance + components["past_actions"] / self.scales["past_actions"]
        if extra:
            distance = distance + beta * components[extra] / self.scales[extra]
        if not torch.isfinite(distance).all():
            raise ValueError("Nonfinite retrieval distances")
        return Ranking(torch.argsort(distance, stable=True).cpu().numpy(), distance, components)

    @torch.no_grad()
    def state(self, context: Context) -> Ranking:
        return self._rank(self._context(context))

    @torch.no_grad()
    def forward(self, context: Context, action: torch.Tensor) -> Ranking:
        components = self._context(context)
        components["action"] = self._mse(self.memory.actions, action)
        return self._rank(components, "action", self.settings["beta_action"])

    @torch.no_grad()
    def reverse(self, context: Context, prediction: torch.Tensor) -> Ranking:
        # Candidate A is intentionally absent. Past observed actions belong to x.
        components = self._context(context)
        components["successor"] = self._mse(self.memory.successors, prediction)
        return self._rank(components, "successor", self.settings["beta_successor"])

    def diagnostics(self, forward: Ranking, reverse: Ranking) -> dict:
        size = len(forward)
        if not size:
            return {"memory_size": 0, "k": 0, "rows": [], "reason": "empty_memory"}
        requested_k = self.settings.get("diagnostic_topk", self.settings["pool_size"])
        k = min(requested_k, size)
        if self.settings.get("diagnostic_selection", "raw") == "raw":
            top_f = [int(index) for index in forward.order[:k]]
            top_r = [int(index) for index in reverse.order[:k]]
            selection_rule = "raw_directional_topk"
        else:
            k = min(k, self.max_support)
            top_f = diverse_prefix(forward.order, k, self.episode_positions, self.episode_cap,
                                   self.step_positions, self.min_step_gap)
            top_r = diverse_prefix(reverse.order, k, self.episode_positions, self.episode_cap,
                                   self.step_positions, self.min_step_gap)
            selection_rule = "directional prefixes respecting support episode and step limits"
        k = min(k, len(top_f), len(top_r))
        top_f, top_r = top_f[:k], top_r[:k]
        forward_index, reverse_index = int(forward.order[0]), int(reverse.order[0])
        forward_row = {**self.memory.items[forward_index], "forward_rank": 1,
            "reverse_rank": reverse.rank_of(forward_index),
            **self.describe(forward_index, forward, reverse)}
        reverse_row = {**self.memory.items[reverse_index],
            "forward_rank": forward.rank_of(reverse_index), "reverse_rank": 1,
            **self.describe(reverse_index, forward, reverse)}
        result = top1_cross_rank_summary(forward_row, reverse_row, size)
        result["rank_scope"] = "complete identical memory; ties by memory row"
        # Keep directional candidate lists separate. Every opposite rank is
        # looked up in the complete ordering, including ranks far beyond k.
        def candidate_row(index):
            return {**self.memory.items[index], "forward_rank": forward.rank_of(index),
                    "reverse_rank": reverse.rank_of(index),
                    **self.describe(index, forward, reverse)}
        result["forward_candidates_exact_cross_ranks"] = candidate_cross_ranks(
            [candidate_row(index) for index in top_f], "forward", size, requested_k)
        result["reverse_candidates_exact_cross_ranks"] = candidate_cross_ranks(
            [candidate_row(index) for index in top_r], "reverse", size, requested_k)
        result["topk_cross_ranks"] = topk_cross_rank_summary(result["forward_candidates_exact_cross_ranks"],
            result["reverse_candidates_exact_cross_ranks"], size, requested_k, selection_rule)
        sf, sr = set(top_f), set(top_r)
        result["supplementary_topk"] = {"k": k,
            "episode_cap": self.episode_cap if selection_rule != "raw_directional_topk" else None,
            "min_start_step_gap": self.min_step_gap if selection_rule != "raw_directional_topk" else 0,
            "selection_rule": selection_rule,
            "overlap_fraction": len(sf & sr)/k,
            "jaccard": len(sf & sr)/len(sf | sr)}
        return result

    def describe(self, index: int, forward: Ranking, reverse: Ranking) -> dict:
        return {"forward_distance": float(forward.distances[index]),
                "reverse_distance": float(reverse.distances[index]),
                "raw_component_mse": {name: float(value[index]) for name, value in
                    (forward.components | reverse.components).items()}}


def diverse_prefix(order, budget: int, episode_positions=None, cap: int | None = None,
                   step_positions=None, min_step_gap: int = 0) -> list[int]:
    if not budget:
        return []
    if cap is None and not min_step_gap:
        return [int(index) for index in order[:budget]]
    selected, counts, steps = [], {}, {}
    for value in order:
        index = int(value)
        episode = int(episode_positions[index])
        step = int(step_positions[index]) if min_step_gap else 0
        if cap is not None and counts.get(episode, 0) >= cap:
            continue
        if min_step_gap and any(abs(step-previous) < min_step_gap for previous in steps.get(episode, [])):
            continue
        if cap is None or counts.get(episode, 0) < cap:
            selected.append(index)
            counts[episode] = counts.get(episode, 0) + 1
            steps.setdefault(episode, []).append(step)
            if len(selected) == budget:
                break
    return selected


def select_support(method: str, budget: int, pool_size: int, alpha: float,
                   forward: Ranking, reverse: Ranking, generator: torch.Generator,
                   state: Ranking | None = None, episode_positions=None,
                   max_per_episode: int | None = None, max_support: int | None = None,
                   step_positions=None, min_step_gap: int = 0) -> Support:
    size = min(budget, len(forward))
    if max_support is not None:
        size = min(size, max_support)
    if size == 0:
        return Support([], [], {})
    if method == "uniform":
        order = torch.randperm(len(forward), generator=generator).numpy()
        selected = diverse_prefix(order, size, episode_positions, max_per_episode, step_positions, min_step_gap)
    elif method == "state":
        selected = diverse_prefix(state.order, size, episode_positions, max_per_episode, step_positions, min_step_gap)
    elif method == "forward" or (method.startswith("fr-") and alpha == 0):
        selected = diverse_prefix(forward.order, size, episode_positions, max_per_episode, step_positions, min_step_gap)
    elif method == "reverse" or (method.startswith("fr-") and alpha == 1):
        selected = diverse_prefix(reverse.order, size, episode_positions, max_per_episode, step_positions, min_step_gap)
    elif method == "rank-consensus":
        # All transitions receive the minimax score, including candidates
        # outside both top-k pools. Partial sorting is only a work reduction;
        # including every boundary tie preserves the complete global order.
        worst = np.maximum(forward.ranks, reverse.ranks)
        rank_sum = forward.ranks + reverse.ranks
        limit = len(worst)
        count = min(max(pool_size, size), limit)
        starting_count = count
        expansions = 0
        while True:
            boundary = int(np.partition(worst, count-1)[count-1])
            candidates = np.flatnonzero(worst <= boundary)
            order = np.lexsort((candidates, rank_sum[candidates], worst[candidates]))
            ordered = candidates[order]
            selected = diverse_prefix(ordered, size, episode_positions, max_per_episode, step_positions, min_step_gap)
            if len(selected) >= size or count >= limit:
                break
            count = min(count*2, limit)
            expansions += 1
        return Support(selected, [1/len(selected)]*len(selected), {index: [method] for index in selected},
                       {"rank_scope": "complete_memory", "scored_memory_windows": limit,
                        "initial_working_prefix_size": starting_count,
                        "working_prefix_candidates": len(candidates), "boundary_worst_rank": boundary,
                        "prefix_expansions": expansions})
    elif method.startswith("fr-"):
        if size == 1:
            chosen = int(forward.order[0])
            return Support([chosen], [1.0], {chosen: ["fallback_single_available_support"]})
        channels = {"forward": [], "reverse": []}
        selected, episode_counts, episode_steps = [], {}, {}
        def add(channel, value):
            index = int(value)
            if index in channels[channel]:
                return False
            if index not in selected:
                episode = int(episode_positions[index]) if max_per_episode or min_step_gap else None
                if max_per_episode and episode_counts.get(episode, 0) >= max_per_episode:
                    return False
                step = int(step_positions[index]) if min_step_gap else 0
                if min_step_gap and any(abs(step-previous) < min_step_gap for previous in episode_steps.get(episode, [])):
                    return False
                selected.append(index)
                if max_per_episode or min_step_gap:
                    episode_counts[episode] = episode_counts.get(episode, 0) + 1
                    episode_steps.setdefault(episode, []).append(step)
            channels[channel].append(index)
            return True
        rankings = {"forward": forward.order, "reverse": reverse.order}
        cursors = {"forward": 0, "reverse": 0}
        for channel, quota in (("forward", size//2), ("reverse", size-size//2)):
            while len(channels[channel]) < quota and cursors[channel] < len(rankings[channel]):
                value = rankings[channel][cursors[channel]]
                cursors[channel] += 1
                add(channel, value)
        # Deduplication preserves both contributions. Fill vacancies from the
        # two complete lists, alternating channel, without repeating a window.
        turn = 0
        while len(selected) < size:
            if all(cursors[name] >= len(rankings[name]) for name in rankings):
                break
            channel = ("forward", "reverse")[turn % 2]
            turn += 1
            while cursors[channel] < len(rankings[channel]):
                index = int(rankings[channel][cursors[channel]])
                cursors[channel] += 1
                if index not in selected and add(channel, index):
                    break
        memberships = {index: [channel for channel, indices in channels.items() if index in indices]
                       for index in selected}
        weights = [(1-alpha) * (index in channels["forward"]) / len(channels["forward"]) +
                   alpha * (index in channels["reverse"]) / len(channels["reverse"]) for index in selected]
        if abs(sum(weights) - 1) > 1e-6 or len(set(selected)) != len(selected):
            raise AssertionError("Invalid FR support weights or unique budget")
        return Support(selected, weights, memberships)
    else:
        raise ValueError(f"Unsupported support method: {method}")
    return Support(selected, [1 / len(selected)] * len(selected), {index: [method] for index in selected})


def prepare_retrieval(memory, calibration: dict, settings: dict, directory, device) -> dict:
    """Prepare backend assets independently of the expensive encoder cache."""
    if settings.get("backend", "torch") == "torch":
        return {"backend": "torch", "memory_windows": len(memory)}
    from .cuvs_retrieval import prepare_cuvs
    return prepare_cuvs(memory, calibration, settings, directory, device)


def make_retriever(memory, calibration: dict, settings: dict, directory):
    if settings.get("backend", "torch") == "torch":
        return ExactRetriever(memory, calibration, settings)
    from .cuvs_retrieval import CuVSRetriever
    return CuVSRetriever(memory, calibration, settings, directory)

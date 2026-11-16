"""Continuous, episode-safe action fragments from observed memory only."""

from dataclasses import dataclass
import numpy as np
import torch

from ..runtime import object_hash
from .context_search import exact_prefix


@dataclass
class FragmentBatch:
    actions: torch.Tensor             # (sources, horizon, ordered_action_dim)
    sources: list[dict]
    diagnostics: dict

    @property
    def fingerprint(self):
        return object_hash({"sources": self.sources, "actions": self.actions.cpu().tolist()})


class FragmentStore:
    def __init__(self, memory, horizon, coordinates):
        if memory.split != "memory":
            raise ValueError("Action proposals must come from the memory split")
        self.memory, self.horizon, self.coordinates = memory, horizon, coordinates
        self.episodes = memory.episode_positions
        self.steps = memory.step_positions
        self.starts = np.asarray(memory.items.starts)
        offsets = np.asarray(memory.manifest["episode_metadata"]["offsets"])
        lengths = np.asarray(memory.manifest["episode_metadata"]["lengths"])
        self.current_rows = self.starts+(memory.history-1)*memory.frameskip
        self.final_rows = self.current_rows+horizon*memory.frameskip
        self.eligible = self.final_rows < (offsets+lengths)[self.episodes]
        allowed = set(memory.manifest["episodes"]["memory"])
        if set(np.unique(self.episodes).tolist())-allowed:
            raise ValueError("Fragment bank includes non-memory episodes")

    def eligible_indices(self, excluded_episode):
        return np.flatnonzero(self.eligible & (self.episodes != excluded_episode))

    def actions_at(self, index):
        if not self.eligible[index]:
            raise ValueError("Fragment final observation crosses an episode boundary")
        first, final = int(self.current_rows[index]), int(self.final_rows[index])
        actions = self.memory.store.actions[first:final].reshape(self.horizon, -1)
        if actions.shape != (self.horizon, self.coordinates.width):
            raise ValueError("Fragment action-block alignment mismatch")
        return actions.detach()

    @torch.no_grad()
    def select(self, order, count, excluded_episode, cap, gap):
        selected, sources, actions, episodes, starts = [], [], [], {}, {}
        rejected = {"short_future": 0, "query_episode": 0, "episode_cap": 0,
                    "start_spacing": 0, "invalid_or_out_of_bounds_actions": 0}
        scanned = 0
        for index in order:
            if len(selected) == count:
                break
            index = int(index)
            scanned += 1
            if not 0 <= index < len(self.memory):
                raise ValueError("Fragment candidate index outside memory")
            episode, step = int(self.episodes[index]), int(self.steps[index])
            if not self.eligible[index]:
                rejected["short_future"] += 1
                continue
            if episode == excluded_episode:
                rejected["query_episode"] += 1
                continue
            if episodes.get(episode, 0) >= cap:
                rejected["episode_cap"] += 1
                continue
            if index in selected or any(abs(step-old) < gap for old in starts.get(episode, [])):
                rejected["start_spacing"] += 1
                continue
            fragment = self.actions_at(index)
            # Preserve an exact recorded seed; reject rather than silently clip it.
            if (not torch.isfinite(fragment).all() or (fragment < self.coordinates.low).any()
                    or (fragment > self.coordinates.high).any()):
                rejected["invalid_or_out_of_bounds_actions"] += 1
                continue
            selected.append(index)
            actions.append(fragment)
            sources.append({**self.memory.items[index], "memory_index": index,
                            "fragment_action_start_row": int(self.current_rows[index]),
                            "fragment_action_stop_row_exclusive": int(self.final_rows[index]),
                            "fragment_final_observation_row": int(self.final_rows[index]),
                            "horizon": self.horizon})
            episodes[episode] = episodes.get(episode, 0)+1
            starts.setdefault(episode, []).append(step)
        shape = (0, self.horizon, self.coordinates.width)
        values = torch.stack(actions) if actions else torch.empty(shape, device=self.coordinates.low.device)
        return FragmentBatch(values, sources, {
            "requested_fragments": count, "selected_fragments": len(selected), "scanned_candidates": scanned,
            "fragment_shortfall": count-len(selected), "rejections": rejected,
            "unique_source_episodes": len(episodes), "max_fragments_per_episode": cap, "min_start_step_gap": gap})


class FragmentProvider:
    """Model-independent proposals; only the goal-context source reads a supplied goal."""
    def __init__(self, store, search, settings, goal_search=None):
        self.store, self.search, self.settings = store, search, settings
        self.goal_search = goal_search

    def retrieve(self, context, metadata, source, seed, goal=None):
        if source not in ("uniform-fragments", "context-fragments", "goal-context-fragments"):
            raise ValueError("Unknown memory-action proposal source")
        episode = metadata["source_episode_position"]
        eligible = self.store.eligible_indices(episode)
        limit = min(self.settings["max_scan_candidates"], len(eligible))
        distances = None
        if source in ("context-fragments", "goal-context-fragments"):
            if source == "goal-context-fragments":
                if self.goal_search is None:
                    raise ValueError("Goal-conditioned proposal source requires a configured goal search")
                distances = self.goal_search.distances(context, goal)
            else:
                distances = self.search.distances(context)
            order = exact_prefix(distances, eligible, limit) if limit else []
        else:
            generator = torch.Generator(device="cpu").manual_seed(seed)
            permutation = torch.randperm(len(eligible), generator=generator).numpy()[:limit]
            order = eligible[permutation].tolist()
        result = self.store.select(order, self.settings["fragments"], episode,
                                   self.settings["max_fragments_per_episode"], self.settings["min_start_step_gap"])
        indices = [row["memory_index"] for row in result.sources]
        scores = (distances[indices].cpu().tolist() if source == "context-fragments" else
                  self.search.selected_distances(context, indices))
        for row, score in zip(result.sources, scores):
            row["context_distance"] = float(score)
        if source == "goal-context-fragments":
            endpoints = self.goal_search.endpoint_distances(goal, indices).cpu().tolist()
            for row, endpoint in zip(result.sources, endpoints):
                row.update(goal_endpoint_distance=float(endpoint),
                           retrieval_distance=float(distances[row["memory_index"]]))
        result.diagnostics.update(source=source, eligible_horizon_safe_windows=len(eligible),
                                  max_scan_candidates=self.settings["max_scan_candidates"],
                                  selection_seed=seed if source == "uniform-fragments" else None,
                                  distance_scope="horizon-safe memory; exclude query episode before selection",
                                  query_episode_excluded=True)
        if source == "goal-context-fragments":
            result.diagnostics.update(goal_weight=self.goal_search.weight, endpoint_scale=self.goal_search.scale,
                                      endpoint_calibration_fingerprint=self.goal_search.calibration["fingerprint"],
                                      goal_supplied=True, endpoint_horizon=self.store.horizon)
        return result

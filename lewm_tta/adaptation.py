"""One predictor per independent query; only observed memory provides labels."""

from __future__ import annotations

import time

import torch

from .cache import LatentBank, Query
from .model import Context, LeWMAdapter
from .retrieval import ExactRetriever, select_support
from .runtime import object_hash


class AdaptationEngine:
    def __init__(self, adapter: LeWMAdapter, memory: LatentBank,
                 retriever: ExactRetriever, settings: dict):
        if memory.split != "memory" or retriever.memory is not memory:
            raise ValueError("Adaptation must share the retriever's observed memory")
        self.adapter, self.memory, self.retriever, self.settings = adapter, memory, retriever, settings

    def _sync(self):
        if self.adapter.device.type == "cuda":
            torch.cuda.synchronize(self.adapter.device)

    def run(self, query: Query, method: str, seed: int) -> dict:
        """There is no query target argument. The evaluator scores returned predictions."""
        query_episode = query.metadata.get("source_episode_position")
        if query_episode in self.retriever.episode_set:
            raise ValueError("Query episode is present in retrieval memory")
        self._sync()
        started = time.perf_counter()
        # Reproducible per-query randomness without depending on query order.
        # Uniform replay must not reuse one identical subset for every query.
        query_seed = int(object_hash([seed, query.metadata["transition_id"]])[:8], 16)
        self.adapter.reset(query_seed)
        reset_finished = time.perf_counter()
        initial = self.adapter.predict(query.context, query.action).detach()
        self._sync()
        prediction_seconds = time.perf_counter() - reset_finished
        search_started = time.perf_counter()
        forward = self.retriever.forward(query.context, query.action)
        reverse = self.retriever.reverse(query.context, initial)
        state = self.retriever.state(query.context) if method == "state" else None
        diagnostics = self.retriever.diagnostics(forward, reverse)
        self._sync()
        search_seconds = time.perf_counter() - search_started
        generator = torch.Generator(device="cpu").manual_seed(query_seed)
        optimizer = torch.optim.AdamW(self.adapter.parameters, lr=self.settings["learning_rate"], weight_decay=0)
        rounds, cumulative, previous_ids = [], set(), set()
        prediction, fixed_support = initial, None
        updates = 0 if method == "frozen" or not len(self.memory) else self.settings["updates"]
        update_seconds, selection_seconds, support_evaluation_seconds = 0.0, 0.0, 0.0
        for step in range(updates):
            selection_started = time.perf_counter()
            if method == "fr-refresh" and step > 0:
                reverse = self.retriever.reverse(query.context, prediction)
            if fixed_support is None or method == "fr-refresh":
                support = select_support(method, self.settings["support_size"], self.retriever.settings["pool_size"],
                    self.settings["alpha"], forward, reverse, generator, state,
                    self.retriever.episode_positions, self.retriever.episode_cap, self.retriever.max_support,
                    self.retriever.step_positions, self.retriever.min_step_gap)
                if method != "fr-refresh":
                    fixed_support = support
            else:
                support = fixed_support
            self._sync()
            selection_seconds += time.perf_counter() - selection_started
            indices = torch.tensor(support.indices, device=self.adapter.device, dtype=torch.long)
            weights = torch.tensor(support.weights, device=self.adapter.device)
            support_context = Context(self.memory.contexts[indices], self.memory.past_actions[indices])
            support_action = self.memory.actions[indices]
            target = self.memory.successors[indices].detach()
            update_started = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            observed_prediction = self.adapter.predict(support_context, support_action, gradients=True)
            per_item = (observed_prediction - target).square().mean(1)
            observed_loss = (weights * per_item).sum()
            anchor = self.adapter.anchor_loss()
            loss = observed_loss + self.settings["anchor_weight"] * anchor
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite adaptation loss")
            loss.backward()
            selected_names = set(self.adapter.parameter_names)
            if any(value.grad is not None for name, value in self.adapter.model.named_parameters()
                   if name not in selected_names):
                raise AssertionError("A frozen parameter received an adaptation gradient")
            gradient_norm = torch.nn.utils.clip_grad_norm_(self.adapter.parameters, self.settings["clip_norm"],
                                                         error_if_nonfinite=True)
            optimizer.step()
            self._sync()
            update_seconds += time.perf_counter() - update_started
            support_evaluation_started = time.perf_counter()
            after_per_item = (self.adapter.predict(support_context, support_action) - target).square().mean(1)
            support_loss_after = float((weights * after_per_item).sum())
            self._sync()
            support_evaluation_seconds += time.perf_counter() - support_evaluation_started
            query_started = time.perf_counter()
            before = prediction
            prediction = self.adapter.predict(query.context, query.action).detach()
            self._sync()
            prediction_seconds += time.perf_counter() - query_started
            ids = {self.memory.items[index]["transition_id"] for index in support.indices}
            support_episodes = [int(self.retriever.episode_positions[index]) for index in support.indices]
            support_steps = [int(self.retriever.step_positions[index]) for index in support.indices]
            if self.retriever.episode_cap is not None and any(
                support_episodes.count(episode) > self.retriever.episode_cap for episode in set(support_episodes)):
                raise AssertionError("Support exceeds the per-episode window cap")
            for left in range(len(support.indices)):
                for right in range(left+1, len(support.indices)):
                    if support_episodes[left] == support_episodes[right] and abs(
                        support_steps[left]-support_steps[right]) < self.retriever.min_step_gap:
                        raise AssertionError("Support windows are too close within an episode")
            cumulative |= ids
            rounds.append({"update": step+1, "prediction_before": before.cpu().flatten().tolist(),
                "prediction_after": prediction.cpu().flatten().tolist(),
                "support_loss_before": float(observed_loss.detach()), "support_loss_after": support_loss_after,
                "anchor_before": float(anchor.detach()),
                "gradient_norm_before_clipping": float(gradient_norm),
                "parameter_displacement_mse": float(self.adapter.anchor_loss().detach()),
                "prediction_correction_mse": float((prediction-before).square().mean()),
                "unique_support": len(ids), "cumulative_unique_support": len(cumulative),
                "requested_support": self.settings["support_size"],
                "support_budget_shortfall": max(0, self.settings["support_size"]-len(ids)),
                "unique_support_episodes": len(set(support_episodes)),
                "max_windows_per_episode": self.retriever.episode_cap,
                "min_start_step_gap": self.retriever.min_step_gap, "selection": support.details,
                "presentations_so_far": sum(row["unique_support"] for row in rounds) + len(ids),
                "turnover_fraction": len(ids - previous_ids) / len(ids) if previous_ids else None,
                "support": [{**self.memory.items[index], "weight": support.weights[position],
                    "channels": support.channels[index], "loss_before": float(per_item[position].detach()),
                    "loss_after": float(after_per_item[position]),
                    **self.retriever.describe(index, forward, reverse)}
                    for position, index in enumerate(support.indices)]})
            previous_ids = ids
        final_search_started = time.perf_counter()
        final_reverse = self.retriever.reverse(query.context, prediction) if updates else reverse
        final_diagnostics = self.retriever.diagnostics(forward, final_reverse) if updates else diagnostics
        self._sync()
        search_seconds += time.perf_counter() - final_search_started
        self.adapter.check_fixed_state()
        if not torch.isfinite(prediction).all():
            raise ValueError("Nonfinite query prediction after adaptation")
        return {"method": method, "seed": seed, "query_rng_seed": query_seed, "query": query.metadata,
            "prediction_initial": initial.cpu().flatten().tolist(),
            "prediction_final": prediction.cpu().flatten().tolist(), "rounds": rounds,
            "rank_diagnostics_initial": diagnostics, "rank_diagnostics_final": final_diagnostics,
            "support_presentations": sum(row["unique_support"] for row in rounds),
            "extra_support_evaluation_presentations": sum(row["unique_support"] for row in rounds),
            "cumulative_unique_support": len(cumulative),
            "parameter_displacement_mse": float(self.adapter.anchor_loss().detach()),
            "skip_reason": "empty_memory" if not len(self.memory) else None,
            "timing_seconds": {"reset": reset_finished-started, "query_prediction": prediction_seconds,
                "diagnostic_search": search_seconds, "support_selection": selection_seconds,
                "gradient_updates": update_seconds, "support_loss_evaluation": support_evaluation_seconds,
                "total_with_checks": time.perf_counter()-started},
            "checks": {"frozen_parameters_and_buffers": True, "query_target_absent_from_adapter": True}}

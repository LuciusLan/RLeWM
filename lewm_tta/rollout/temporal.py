"""Detached autoregression with independent update placement and support refresh.

No evaluator target enters this module. Each trajectory owns a fresh parameter,
optimizer and RNG session; only observed memory transitions supervise updates.
"""

from __future__ import annotations

import time

import torch

from ..model import Context
from ..retrieval import select_support
from ..runtime import object_hash
from .config import settings_for_schedule, updates_at, update_budget
from .support_updates import observed_support_step, synchronize, validate_support


class AdaptationSession:
    """Own the optimizer across positions without changing the pretrained anchor."""

    def __init__(self, engine, query, seed: int, schedule: dict | None = None):
        self.adapter, self.memory = engine.adapter, engine.memory
        self.retriever = engine.retriever
        self.settings = settings_for_schedule(engine.settings, schedule)
        if self.memory.split != "memory" or self.retriever.memory is not self.memory:
            raise ValueError("Session and retriever must share observed memory")
        if query.metadata.get("source_episode_position") in self.retriever.episode_set:
            raise ValueError("Query episode is present in retrieval memory")
        self.query_seed = int(object_hash([seed, query.metadata["transition_id"]])[:8], 16)
        self.cumulative, self.previous_ids, self.rounds = set(), set(), []
        self.presentations, self.resets, self.optimizer_updates = 0, 0, 0
        self.search_calls = {"forward": 0, "reverse": 0, "state": 0}
        self.timing = {key: 0.0 for key in ("reset", "query_prediction", "retrieval_and_selection",
                                          "gradient_updates", "support_loss_evaluation")}
        self.reset()

    def reset(self):
        synchronize(self.adapter)
        started = time.perf_counter()
        self.adapter.reset(self.query_seed)
        self.optimizer = torch.optim.AdamW(self.adapter.parameters, lr=self.settings["learning_rate"], weight_decay=0)
        self.optimizer_updates = 0
        self.resets += 1
        synchronize(self.adapter)
        self.timing["reset"] += time.perf_counter()-started

    def predict(self, context, action):
        synchronize(self.adapter)
        started = time.perf_counter()
        # Even between adaptation steps the prediction operation is read-only.
        with torch.no_grad():
            prediction = self.adapter.predict(context, action).detach()
        self.adapter.check_fixed_state(check_parameters=False)
        if not torch.isfinite(prediction).all():
            raise ValueError("Nonfinite scheduled query prediction")
        synchronize(self.adapter)
        self.timing["query_prediction"] += time.perf_counter()-started
        return prediction

    def check_optimizer_clock(self):
        steps = [int(state["step"].item()) for state in self.optimizer.state.values() if "step" in state]
        if (self.optimizer_updates and not steps) or any(step != self.optimizer_updates for step in steps):
            raise AssertionError("AdamW state does not match the session's update clock")

    def adapt_position(self, context, action, method: str, updates: int, position: int) -> dict:
        initial = prediction = self.predict(context, action)
        start_round = len(self.rounds)
        self.check_optimizer_clock()
        optimizer_before = self.optimizer_updates
        displacement_before = float(self.adapter.anchor_loss().detach())
        generator_seed = (self.query_seed if position == 0 else
                          int(object_hash([self.query_seed, "position", position])[:8], 16))
        generator = torch.Generator(device="cpu").manual_seed(generator_seed)
        budget = 0 if method == "frozen" else updates
        skip_reason = "empty_memory" if budget and not len(self.memory) else None
        forward = reverse = state = support = None
        previous_local = set()
        if budget and len(self.memory):
            synchronize(self.adapter)
            started = time.perf_counter()
            # Rebuilt at every adapting position; fixed within a position.
            forward = self.retriever.forward(context, action)
            reverse = self.retriever.reverse(context, prediction)
            self.search_calls["forward"] += 1
            self.search_calls["reverse"] += 1
            if method == "state":
                state = self.retriever.state(context)
                self.search_calls["state"] += 1
            synchronize(self.adapter)
            self.timing["retrieval_and_selection"] += time.perf_counter()-started
            for local in range(budget):
                started = time.perf_counter()
                if method == "fr-refresh" and local > 0:
                    reverse = self.retriever.reverse(context, prediction)
                    self.search_calls["reverse"] += 1
                if support is None or method == "fr-refresh":
                    support = select_support(method, self.settings["support_size"], self.retriever.settings["pool_size"],
                        self.settings["alpha"], forward, reverse, generator, state,
                        self.retriever.episode_positions, self.retriever.episode_cap, self.retriever.max_support,
                        self.retriever.step_positions, self.retriever.min_step_gap)
                synchronize(self.adapter)
                self.timing["retrieval_and_selection"] += time.perf_counter()-started
                if not support.indices:
                    skip_reason = "no_eligible_support"
                    break
                validate_support(self.memory, self.retriever, support, self.settings["support_size"])
                values = observed_support_step(self.adapter, self.memory, support, self.optimizer, self.settings)
                self.optimizer_updates += 1
                self.check_optimizer_clock()
                self.timing["gradient_updates"] += values.pop("gradient_seconds")
                self.timing["support_loss_evaluation"] += values.pop("support_evaluation_seconds")
                before, prediction = prediction, self.predict(context, action)
                before_losses, after_losses = values.pop("losses_before"), values.pop("losses_after")
                items = [self.memory.items[i] for i in support.indices]
                ids = {item["transition_id"] for item in items}
                self.cumulative |= ids
                self.presentations += len(ids)
                self.rounds.append({**values, "update": len(self.rounds)+1, "position": position,
                    "update_at_position": local+1, "optimizer_step": self.optimizer_updates,
                    "learning_rate": self.settings["learning_rate"],
                    "prediction_before": before.cpu().flatten().tolist(),
                    "prediction_after": prediction.cpu().flatten().tolist(),
                    "prediction_correction_mse": float((prediction-before).square().mean()),
                    "unique_support": len(ids), "cumulative_unique_support": len(self.cumulative),
                    "unique_support_episodes": len({item["source_episode_position"] for item in items}),
                    "requested_support": self.settings["support_size"],
                    "support_budget_shortfall": self.settings["support_size"]-len(ids),
                    "turnover_fraction": len(ids-self.previous_ids)/len(ids) if self.previous_ids else None,
                    "within_position_turnover": len(ids-previous_local)/len(ids) if previous_local else None,
                    "presentations_so_far": self.presentations, "selection": support.details,
                    "support": [{key: item[key] for key in
                                 ("transition_id", "source_episode_position", "start_in_episode")} |
                                {"weight": support.weights[j], "channels": support.channels[i],
                                 "loss_before": before_losses[j], "loss_after": after_losses[j]}
                                for j, (i, item) in enumerate(zip(support.indices, items))]})
                self.previous_ids, previous_local = ids, ids
        return {"position": position, "requested_updates": budget,
                "actual_updates": len(self.rounds)-start_round, "round_start": start_round,
                "learning_rate": self.settings["learning_rate"],
                "round_end": len(self.rounds), "optimizer_steps_before": optimizer_before,
                "optimizer_steps_after": self.optimizer_updates, "support_rng_seed": generator_seed,
                "parameter_displacement_before": displacement_before,
                "parameter_displacement_after": float(self.adapter.anchor_loss().detach()),
                "prediction_initial": initial.cpu().flatten().tolist(),
                "prediction_final": prediction.cpu().flatten().tolist(),
                "prediction_tensor": prediction, "skip_reason": skip_reason}


def run_scheduled_rollout(engine, query, actions, method: str, seed: int, schedule: dict):
    """Return predictions and an audit record; future observations are not inputs."""
    adapter = engine.adapter
    if actions.ndim != 3 or actions.shape[0] != 1 or actions.shape[2] != adapter.action_dim or actions.shape[1] < 1:
        raise ValueError("Scheduled evaluation expects (1, horizon, action_dim) recorded controls")
    torch.testing.assert_close(actions[:, 0], query.action, rtol=0, atol=0)
    synchronize(adapter)
    started = time.perf_counter()
    session = AdaptationSession(engine, query, seed, schedule)
    context = Context(query.context.latents.detach(), query.context.past_actions.detach())
    predictions, positions = [], []
    for h, action in enumerate(actions.detach().unbind(1)):
        if h and schedule["persistence"] == "position":
            session.reset()
        if context.latents.requires_grad or context.past_actions.requires_grad:
            raise AssertionError("Predicted query contexts must be detached")
        if h:
            torch.testing.assert_close(context.latents[:, -1], predictions[-1], rtol=0, atol=0)
            torch.testing.assert_close(context.past_actions[:, -1], actions[:, h-1], rtol=0, atol=0)
        row = session.adapt_position(context, action, method, updates_at(schedule, h), h)
        if h and schedule["persistence"] == "trajectory":
            if row["parameter_displacement_before"] != positions[-1]["parameter_displacement_after"]:
                raise AssertionError("Parameters changed between carried trajectory positions")
        prediction = row.pop("prediction_tensor")
        row["context_fingerprint"] = object_hash([context.latents.cpu().tolist(), context.past_actions.cpu().tolist()])
        row["action_fingerprint"] = object_hash(action.cpu().tolist())
        row["context_source"] = "observed_initial_history" if h == 0 else "shifted_predicted_history"
        row["reset_before_position"] = h == 0 or schedule["persistence"] == "position"
        positions.append(row)
        predictions.append(prediction)
        context = Context(torch.cat((context.latents[:, 1:], prediction.unsqueeze(1)), dim=1).detach(),
                          torch.cat((context.past_actions[:, 1:], action.unsqueeze(1)), dim=1).detach())
    adapter.check_fixed_state()
    synchronize(adapter)
    record = {"query_rng_seed": session.query_seed, "schedule": schedule,
              "learning_rate": session.settings["learning_rate"],
              "actual_learning_rate_sum": len(session.rounds)*session.settings["learning_rate"],
              "rounds": session.rounds, "positions": positions,
              "prediction_initial": positions[0]["prediction_initial"],
              "prediction_final": positions[0]["prediction_final"],
              "requested_updates": 0 if method == "frozen" else update_budget(schedule, actions.shape[1]),
              "actual_updates": len(session.rounds), "reset_count": session.resets,
              "support_presentations": session.presentations,
              "extra_support_evaluation_presentations": session.presentations,
              "cumulative_unique_support": len(session.cumulative),
              "parameter_displacement_mse": float(adapter.anchor_loss().detach()),
              "module_updates": adapter.module_metrics(), "retrieval_calls": session.search_calls,
              "timing_seconds": session.timing | {"total_with_checks": time.perf_counter()-started},
              "checks": {"observed_memory_targets_only": True, "no_query_targets_in_prediction": True,
                         "detached_predicted_context_feedback": True, "frozen_parameters_and_buffers": True}}
    return torch.stack(predictions, dim=1), record

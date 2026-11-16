"""One model/optimizer lifetime per query, shared by all candidates in each phase."""

import time

import torch

from ..planning.actions import stream_seed
from ..rollout.inference import fixed_predictor
from ..rollout.support_updates import observed_support_step, synchronize, validate_support
from ..runtime import object_hash
from .cem import CheckpointResult
from .config import ARMS, adaptation_schedule, checkpoint_updates, validate_depth_updates
from .continuations import retrieve_continuations
from .references import balanced_pairs, pair_tensors, rollout_references, select_parents
from .sequential import adapt_sequential
from .supports import ReplayBuffer, directions, select_support, support_change


class CheckpointSession:
    def __init__(self, engine, context, metadata, goal, coordinates, providers, config, arm, seed):
        self.engine, self.adapter = engine, engine.adapter
        if engine.memory.split != "memory" or engine.retriever.memory is not engine.memory:
            raise ValueError("Checkpoint supervision must use the observed-memory bank")
        if metadata["source_episode_position"] in engine.retriever.episode_set:
            raise ValueError("Query episode occurs in adaptation memory")
        self.context, self.metadata, self.goal = context, metadata, goal
        self.coordinates, self.providers, self.config = coordinates, providers, config
        self.arm, self.seed = arm, seed
        self.adapt_enabled, self.proposals_enabled = ARMS[arm]
        self.version, self.rounds, self.unique_support, self.previous_support = 0, [], set(), None
        self.buffer = ReplayBuffer(config["interleave"]["replay"]["capacity"])
        self.counters = {name: 0 for name in (
            "reference_sequence_evaluations", "reference_transition_predictions", "successor_query_predictions",
            "forward_retrievals", "reverse_retrievals", "initial_action_retrievals", "continuation_retrievals", "actual_updates",
            "support_presentations", "support_evaluation_presentations", "retained_candidate_rescorings")}
        self.timings = {name: 0. for name in (
            "reset_and_optimizer", "initial_proposals", "cem_scoring", "reference_rollouts",
            "support_selection", "replay_selection", "gradient_updates", "support_evaluation", "continuations")}
        self.reset_seed = stream_seed(seed, metadata["transition_id"], "model-reset")
        synchronize(self.adapter)
        started = time.perf_counter()
        self.adapter.reset(self.reset_seed)
        self.optimizer = torch.optim.AdamW(self.adapter.parameters,
            lr=engine.settings["learning_rate"], weight_decay=0) if self.adapt_enabled else None
        synchronize(self.adapter)
        self.timings["reset_and_optimizer"] = time.perf_counter()-started

    def rng(self, name, checkpoint):
        # Arm is excluded deliberately: matched treatments share exogenous draws.
        return stream_seed(self.seed, self.metadata["transition_id"], f"{name}-{checkpoint}")

    def roll(self, actions):
        synchronize(self.adapter)
        started = time.perf_counter()
        with fixed_predictor(self.adapter):
            result = rollout_references(self.adapter, self.context, actions)
        synchronize(self.adapter)
        self.timings["reference_rollouts"] += time.perf_counter()-started
        self.counters["reference_sequence_evaluations"] += len(actions)
        self.counters["reference_transition_predictions"] += len(actions)*actions.shape[1]
        return result

    def score(self, actions):
        from ..planning.cem import score_sequences
        synchronize(self.adapter)
        started = time.perf_counter()
        with fixed_predictor(self.adapter):
            result = score_sequences(self.adapter, self.context, self.goal, actions,
                                     self.config["planning"]["scoring_batch_size"])
        synchronize(self.adapter)
        self.timings["cem_scoring"] += time.perf_counter()-started
        return result

    def adapt_pairs(self, checkpoint, contexts, actions, record, *, depth=None):
        """Train on one query batch; keep pooled RNG streams unchanged."""
        s, engine = self.config["interleave"], self.engine
        support, forward_cache = None, None
        for update in range(engine.settings["updates"]):
            if update == 0 or s["support_method"] == "fr-refresh":
                synchronize(self.adapter)
                selection_started = time.perf_counter()
                support, forward_cache, calls = select_support(engine, contexts, actions,
                    s["support_method"], self.rng("support" if depth is None else f"support-depth-{depth}", checkpoint),
                    self.config["planning"]["support_scan_limit"], forward_cache)
                synchronize(self.adapter)
                self.timings["support_selection"] += time.perf_counter()-selection_started
                for name in ("forward", "reverse"):
                    self.counters[name+"_retrievals"] += calls[name]
                self.counters["successor_query_predictions"] += calls["successor_predictions"]
            current = support
            replay = s["replay"]
            if replay["policy"] == "current-plus-replay":
                replay_started = time.perf_counter()
                support_batch = self.buffer.mix(current, engine.retriever, engine.settings["support_size"],
                    replay["fraction"], checkpoint, self.rng(f"replay-{update}" if depth is None else f"replay-depth-{depth}-{update}", checkpoint))
                self.timings["replay_selection"] += time.perf_counter()-replay_started
            else:
                support_batch = current
            if not support_batch.indices:
                break
            validate_support(engine.memory, engine.retriever, support_batch, engine.settings["support_size"])
            with torch.enable_grad():
                values = observed_support_step(self.adapter, engine.memory, support_batch,
                                               self.optimizer, engine.settings)
            self.timings["gradient_updates"] += values["gradient_seconds"]
            self.timings["support_evaluation"] += values["support_evaluation_seconds"]
            self.version += 1
            self.counters["actual_updates"] += 1
            self.counters["support_presentations"] += len(support_batch.indices)
            self.counters["support_evaluation_presentations"] += len(support_batch.indices)
            expected = self.counters["actual_updates"]
            if any(int(state["step"].item()) != expected for state in self.optimizer.state.values() if "step" in state):
                raise AssertionError("Checkpoint optimizer did not carry its update clock")
            items = [{**engine.memory.items[index], "memory_index": int(index),
                      "weight": float(weight), "channels": support_batch.channels[index]}
                     for index, weight in zip(support_batch.indices, support_batch.weights)]
            row = {"checkpoint": checkpoint, "update": expected, "model_version": self.version,
                   "support": items, "selection": support_batch.details,
                   "current_support_ids": [int(i) for i in current.indices],
                   "support_fingerprint": object_hash(items),
                   "support_change": support_change(self.previous_support, support_batch),
                   **{key: value for key, value in values.items() if key not in ("losses_before", "losses_after")}}
            if depth is not None:
                row["depth"] = depth
            record["rounds"].append(row)
            self.rounds.append(row)
            record["actual_updates"] += 1
            record["support_presentations"] += len(items)
            self.unique_support.update(support_batch.indices)
            self.previous_support = support_batch
            # Same-checkpoint rows are never sampled as older replay.
            self.buffer.insert(current, checkpoint)

    def __call__(self, checkpoint, candidates, costs, elite_indices):
        synchronize(self.adapter)
        started = time.perf_counter()
        timing_before = dict(self.timings)
        s, engine = self.config["interleave"], self.engine
        record = {"checkpoint": checkpoint, "model_version_before": self.version,
                  "population_fingerprint": object_hash(candidates.cpu().tolist()),
                  "requested_updates": checkpoint_updates(self.config, engine.settings["updates"]) if self.adapt_enabled else 0,
                  "actual_updates": 0, "support_presentations": 0, "buffer_before": len(self.buffer.rows),
                  "adaptation_pairs": [], "rounds": [], "continuations": None,
                  "retained_candidate_rescorings": 0,
                  "retained_score_policy": "Old costs are discarded; old mean/variance are proposal state only"}
        if adaptation_schedule(self.config) == "sequential-depth":
            record.update(adaptation_schedule="sequential-depth",
                          updates_per_depth=engine.settings["updates"] if self.adapt_enabled else 0, depths=[])
        active_adaptation = self.adapt_enabled and engine.settings["updates"] > 0 and len(engine.memory) > 0
        active_proposals = self.proposals_enabled and s["injection_slots"] > 0
        parents, parent_ids = select_parents(candidates, elite_indices, s["references"], "elite",
                                            self.rng("parent-selection", checkpoint))
        record["parent_candidate_ids"] = parent_ids
        record["parent_actions_fingerprint"] = object_hash(parents.cpu().tolist())
        parent_contexts, ref_contexts, references, ref_ids = None, None, parents, parent_ids
        if active_adaptation:
            if s["reference_source"] == "random":
                references, ref_ids = select_parents(candidates, elite_indices, s["references"], "random",
                                                     self.rng("reference-selection", checkpoint))
            record["adaptation_reference_source"] = s["reference_source"]
            record["adaptation_reference_candidate_ids"] = ref_ids
            if adaptation_schedule(self.config) == "sequential-depth":
                adapt_sequential(self, checkpoint, references, ref_ids, record)
            else:
                ref_contexts, _ = self.roll(references)
                if ref_ids == parent_ids:
                    parent_contexts = ref_contexts
                channels = 1 if s["support_method"] == "uniform" else len(directions(engine.settings, s["support_method"]))
                # Keep the same query-pair count for Forward/Reverse/FR controls.
                # Reserve room for both directions even when a control uses one.
                limit = s["support_pairs"] or max(1, engine.settings["support_size"]//2)
                limit = min(limit, engine.settings["support_size"]//channels)
                pairs = balanced_pairs(len(references), s["adaptation_depths"], limit,
                                       self.rng("adaptation-pairs", checkpoint))
                contexts, actions = pair_tensors(ref_contexts, references, pairs)
                record["adaptation_pairs"] = [{"parent_candidate_id": int(ref_ids[p]), "depth": h}
                                              for p, h in pairs]
                record["adaptation_query_fingerprint"] = object_hash({
                    "latents": contexts.latents.cpu().tolist(), "past_actions": contexts.past_actions.cpu().tolist(),
                    "actions": actions.cpu().tolist()})
                self.adapt_pairs(checkpoint, contexts, actions, record)
        injection = None
        if active_proposals:
            if parent_contexts is None or record["actual_updates"]:
                parent_contexts, _ = self.roll(parents)
            synchronize(self.adapter)
            continuation_started = time.perf_counter()
            with fixed_predictor(self.adapter):
                injection, record["continuations"] = retrieve_continuations(self.providers, self.context,
                    parent_contexts, parents, parent_ids, self.metadata, self.goal, self.seed, checkpoint)
            synchronize(self.adapter)
            self.timings["continuations"] += time.perf_counter()-continuation_started
            self.counters["continuation_retrievals"] += record["continuations"]["retrieval_calls"]
        self.adapter.check_fixed_state()
        synchronize(self.adapter)
        record.update(model_version_after=self.version, buffer_after=len(self.buffer.rows),
                      buffer_entries=[{"memory_index": int(i), **row} for i, row in self.buffer.rows.items()],
                      skip_reason=("zero_updates" if self.adapt_enabled and not engine.settings["updates"] else
                                   "empty_memory" if self.adapt_enabled and not len(engine.memory) else
                                   "no_eligible_support" if active_adaptation and record["actual_updates"] < record["requested_updates"] else None),
                      seconds_with_checks=time.perf_counter()-started)
        record["timing_seconds"] = {name: value-timing_before[name] for name, value in self.timings.items()}
        validate_depth_updates(record, self.config, engine.settings["updates"])
        return CheckpointResult(injection, record, self.version)

    def adaptation_record(self):
        return {"rounds": self.rounds, "updates": self.counters["actual_updates"],
                "support_presentations": self.counters["support_presentations"],
                "cumulative_unique_support": len(self.unique_support), "reset_seed": self.reset_seed,
                "model_version": self.version, "reset_count": 1,
                "parameter_displacement_mse": float(self.adapter.anchor_loss().detach())}

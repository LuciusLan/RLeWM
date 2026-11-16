"""CEM scorers with a common deterministic mapping within each treatment."""

import hashlib
import time

import torch

from experiments.progress import progress

from ...model import Context
from ...runtime import object_hash
from ...rollout.support_updates import synchronize
from ..shared_ttt import adapt_shared
from .config import MODES
from .rollouts import build_depth_models, candidate_rollout
from .state import ParameterSnapshot, advance, fresh_counters


class TemporalScorer:
    def __init__(self, engine, context, metadata, references, mode, seed, config):
        if mode not in MODES or engine.memory.split != "memory" or engine.retriever.memory is not engine.memory:
            raise ValueError("Invalid temporal treatment or observed-memory source")
        if metadata["source_episode_position"] in engine.retriever.episode_set:
            raise ValueError("Query episode appears in temporal planning support memory")
        self.engine, self.context, self.metadata = engine, context, metadata
        self.mode, self.seed, self.config = mode, seed, config
        self.method, self.counters = config["temporal"]["selector"], fresh_counters()
        self.audit_digest, self.last_audit = hashlib.sha256(), None
        self.snapshots, self.setup = [], {"rounds": []}
        synchronize(engine.adapter)
        started = time.perf_counter()
        if mode == "shared-depth":
            self.snapshots, self.setup = build_depth_models(engine, context, metadata, references,
                                                          self.method, seed, config, self.counters)
        elif mode in ("frozen", "shared-initial"):
            record = adapt_shared(engine, context, metadata, references[:, 0],
                "frozen" if mode == "frozen" else self.method, seed, config["planning"]["support_scan_limit"])
            self.snapshots = [ParameterSnapshot(engine.adapter)]*config["planning"]["horizon"]
            self.setup = record
            self.counters["requested_updates"] = 0 if mode == "frozen" else engine.settings["updates"]
            self.counters["actual_updates"] = record["updates"]
            self.counters["support_presentations"] = record["support_presentations"]
            self.counters["support_positions"] = int(bool(self.counters["requested_updates"]))
            self.counters["support_shortfall_positions"] = int(bool(self.counters["requested_updates"])
                and record["unique_support"] < engine.settings["support_size"])
            self.counters["forward_retrievals"] = record["search_calls"]["forward"]
            self.counters["reverse_retrievals"] = record["search_calls"]["reverse"]
            self.counters["reset_count"] = 1
        else:
            # Each scored sequence owns its reset, optimizer and support RNG.
            self.setup["reset_policy"] = "Fresh reset at every scored sequence; no setup adaptation"
        synchronize(engine.adapter)
        self.setup_seconds = time.perf_counter()-started
        self.setup["reference_fingerprint"] = object_hash(references.cpu().tolist())
        if self.snapshots:
            self.setup["model_fingerprints"] = [snapshot.fingerprint for snapshot in self.snapshots]

    def rollout(self, actions):
        adapter = self.engine.adapter
        horizon = self.config["planning"]["horizon"]
        if (actions.ndim != 3 or len(actions) < 1 or actions.shape[1:] != (horizon, adapter.action_dim)
                or not torch.isfinite(actions).all()):
            raise ValueError("Temporal scorer expects finite (N, horizon, action_dim) actions")
        predictions = []
        if self.mode == "candidate-temporal":
            with progress(actions.detach().split(1), total=len(actions), config=self.config,
                          desc="Candidate-temporal population", unit="sequence", show=len(actions) >= 32,
                          position=1, leave=False) as candidates:
                for sequence in candidates:
                    prediction, self.last_audit = candidate_rollout(self.engine, self.context, self.metadata,
                        sequence, self.method, self.seed, self.config, self.counters)
                    self.audit_digest.update(object_hash(self.last_audit).encode())
                    predictions.append(prediction.detach())
                    candidates.set_postfix(updates=self.counters["actual_updates"], refresh=False)
        else:
            for chunk in actions.detach().split(self.config["planning"]["scoring_batch_size"]):
                current = Context(self.context.latents.expand(len(chunk), -1, -1).detach(),
                                  self.context.past_actions.expand(len(chunk), -1, -1).detach())
                local = []
                for depth, action in enumerate(chunk.unbind(1)):
                    self.snapshots[depth].restore(adapter)
                    with torch.no_grad():
                        prediction = adapter.predict(current, action).detach()
                    local.append(prediction)
                    current = advance(current, action, prediction)
                predictions.append(torch.stack(local, 1))
            adapter.check_fixed_state()
        result = torch.cat(predictions)
        if not torch.isfinite(result).all():
            raise ValueError("Nonfinite adaptive CEM rollout")
        self.counters["sequence_evaluations"] += len(actions)
        self.counters["transition_predictions"] += len(actions)*horizon
        return result

    def costs(self, actions, goal):
        # Goals affect only this terminal scoring operation, never support TTT.
        prediction = self.rollout(actions)
        return (prediction[:, -1]-goal).square().sum(-1).detach()

    def accounting(self):
        return {**self.counters, "audit_sha256": self.audit_digest.hexdigest()}

"""Exact context-plus-achieved-endpoint search over the existing observation cache."""

import fcntl
import math
from pathlib import Path

import numpy as np
import torch

from ..runtime import file_hash, object_hash, read_json, write_json


def endpoint_scale(store, settings):
    """Median positive endpoint MSE of distinct eligible memory-anchor pairs."""
    eligible = np.flatnonzero(store.eligible)
    count = len(eligible)
    if count < 2:
        return {"scale": 1.0, "pairs": 0, "positive_pairs": 0,
                "rule": "unit fallback for fewer than two horizon-safe memory anchors"}
    generator = torch.Generator(device="cpu").manual_seed(settings["goal_calibration_seed"])
    pairs = settings["goal_calibration_pairs"]
    first = torch.randint(count, (pairs,), generator=generator).numpy()
    second = (first+torch.randint(1, count, (pairs,), generator=generator).numpy()) % count
    device = store.memory.store.latents.device
    values = []
    for start in range(0, pairs, settings["context_chunk_rows"]):
        end = min(start+settings["context_chunk_rows"], pairs)
        left = torch.as_tensor(store.final_rows[eligible[first[start:end]]], device=device)
        right = torch.as_tensor(store.final_rows[eligible[second[start:end]]], device=device)
        difference = store.memory.store.latents[left]-store.memory.store.latents[right]
        mse = difference.square().mean(1)
        if not torch.isfinite(mse).all():
            raise ValueError("Nonfinite memory endpoint calibration distances")
        values.append(mse.cpu())
    distances = torch.cat(values)
    positive = distances[distances > settings["goal_distance_floor"]]
    scale = max(float(positive.median()), settings["goal_distance_floor"]) if len(positive) else 1.0
    return {"scale": scale, "pairs": pairs, "positive_pairs": len(positive),
            "rule": "median positive raw MSE of distinct horizon-safe memory-anchor endpoint pairs; unit fallback if none"}


class GoalContextSearch:
    """The horizon endpoint is observed memory data, never a predicted query future."""

    def __init__(self, context_search, store, settings, calibration_directory=None):
        self.context, self.store, self.settings = context_search, store, settings
        if context_search.memory is not store.memory or store.memory.split != "memory":
            raise ValueError("Goal retrieval and calibration require the same memory-only bank")
        self.weight = settings["goal_weight"]
        identity = {"version": 1, "memory": store.memory.identity, "horizon": store.horizon,
                    "frameskip": store.memory.frameskip, "eligible_memory_anchors": int(store.eligible.sum()),
                    "endpoint_rule": "start + (history - 1 + horizon) * frameskip, inside source episode",
                    "calibration_settings": {key: settings[key] for key in (
                        "goal_calibration_pairs", "goal_calibration_seed", "goal_distance_floor")},
                    "implementation_sha256": file_hash(Path(__file__))}
        self.calibration_path = None

        def calibrate():
            value = {"identity": identity, **endpoint_scale(store, settings)}
            value["fingerprint"] = object_hash(value)
            return value

        if calibration_directory is None:
            calibration = calibrate()
        else:
            directory = Path(calibration_directory)
            directory.mkdir(parents=True, exist_ok=True)
            self.calibration_path = directory/f"goal-h{store.horizon}-{object_hash(identity)[:20]}.json"
            with self.calibration_path.with_suffix(".lock").open("a+") as lock:
                fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
                if self.calibration_path.exists():
                    calibration = read_json(self.calibration_path)
                else:
                    calibration = calibrate()
                    write_json(self.calibration_path, calibration)
        if (calibration.get("identity") != identity or calibration.get("fingerprint") != object_hash(
                {key: value for key, value in calibration.items() if key != "fingerprint"})):
            raise ValueError("Goal endpoint calibration identity/checksum mismatch")
        self.calibration = calibration
        self.scale = calibration["scale"]
        if not math.isfinite(self.scale) or self.scale <= 0:
            raise ValueError("Goal endpoint distance scale must be finite and positive")

    def checked_goal(self, goal):
        latents = self.store.memory.store.latents
        if not isinstance(goal, torch.Tensor) or goal.shape != (1, latents.shape[1]) or not torch.isfinite(goal).all():
            raise ValueError("Goal retrieval requires one finite supplied goal latent")
        value = goal.detach().to(device=latents.device, dtype=torch.float32).contiguous()
        if not torch.isfinite(value).all():
            raise ValueError("Retrieval goal must remain finite in float32")
        return value

    @torch.no_grad()
    def endpoint_distances(self, goal, indices):
        goal = self.checked_goal(goal)
        indices = np.asarray(indices, dtype=np.int64)
        if (indices < 0).any() or (indices >= len(self.store.memory)).any():
            raise ValueError("Goal retrieval memory index outside the bank")
        if not self.store.eligible[indices].all():
            raise ValueError("Goal retrieval endpoint crosses a source episode boundary")
        if not len(indices):
            return goal.new_empty(0)
        rows = torch.as_tensor(self.store.final_rows[indices], device=goal.device)
        endpoints = self.store.memory.store.latents[rows].to(dtype=torch.float32).contiguous()
        if self.context.cuvs:
            result = self.context.base.runtime.distances(endpoints, goal)/(endpoints.shape[1]*self.scale)
        else:
            result = (endpoints-goal).square().mean(1)/self.scale
        if not torch.isfinite(result).all():
            raise ValueError("Nonfinite goal endpoint retrieval distances")
        return result

    @torch.no_grad()
    def distances(self, context, goal):
        goal = self.checked_goal(goal)
        result = self.context.distances(context)
        # Preserve exact context rankings for the zero-weight control.
        if self.weight == 0:
            return result
        chunk = self.settings["context_chunk_rows"]
        for start in range(0, len(self.store.memory), chunk):
            end = min(start+chunk, len(self.store.memory))
            eligible = np.flatnonzero(self.store.eligible[start:end])+start
            if len(eligible):
                rows = torch.as_tensor(eligible, device=result.device)
                result[rows] += self.weight*self.endpoint_distances(goal, eligible)
                if not torch.isfinite(result[rows]).all():
                    raise ValueError("Nonfinite combined context/goal retrieval distances")
            invalid = np.flatnonzero(~self.store.eligible[start:end])+start
            if len(invalid):
                result[torch.as_tensor(invalid, device=result.device)] = float("inf")
        return result

    def description(self):
        context = self.context.description()
        width = self.store.memory.store.latents.shape[1]
        specification = {"calibration_fingerprint": self.calibration["fingerprint"],
                         "context_scales": self.context.base.scales, "goal_weight": self.weight}
        return {"enabled": True, "backend": "cuvs_exact_chunked" if self.context.cuvs else "torch_exact_chunked",
                "metric": "context_MSE/context_scale + past_action_MSE/past_action_scale + goal_weight*endpoint_MSE/endpoint_scale",
                "query_fields": ["context_latents", "past_observed_actions", "supplied_goal_latent"],
                "key_fields": ["memory_context_latents", "memory_past_actions", "observed_H_block_endpoint_latent"],
                "horizon": self.store.horizon, "raw_horizon": self.store.horizon*self.store.memory.frameskip,
                "goal_weight": self.weight, "endpoint_scale": self.scale, "calibration": self.calibration,
                "calibration_file": str(self.calibration_path.resolve()) if self.calibration_path else None,
                "fingerprint": object_hash(specification), "concatenated_key_width": context["context_key_width"]+width,
                "additional_persistent_key_bytes": 0,
                "endpoint_chunk_bytes_upper_bound": min(self.settings["context_chunk_rows"], len(self.store.memory))*width*4,
                "note": "Context and endpoint distances add exactly in real arithmetic; separate float32 kernels may round differently. Scratch excludes native workspace."}

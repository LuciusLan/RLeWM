"""Explicit raw/normalized action contract and independent random streams."""

import torch

from ..runtime import object_hash


def stream_seed(seed, transition_id, stream):
    return int(object_hash([int(seed), transition_id, stream])[:8], 16)


class ActionCoordinates:
    def __init__(self, normalization, planning, frameskip, device):
        self.mean = torch.tensor(normalization["mean"], device=device, dtype=torch.float32)
        self.std = torch.tensor(normalization["std"], device=device, dtype=torch.float32)
        self.raw_low = torch.tensor(planning["raw_action_low"], device=device, dtype=torch.float32)
        self.raw_high = torch.tensor(planning["raw_action_high"], device=device, dtype=torch.float32)
        self.frameskip = frameskip
        if (self.mean.ndim != 1 or self.mean.shape != self.std.shape or self.std.shape != self.raw_low.shape
                or self.std.shape != self.raw_high.shape or not torch.isfinite(self.mean).all()
                or not torch.isfinite(self.std).all() or (self.std <= 0).any()):
            raise ValueError("Raw action coordinates disagree with cached normalization")
        self.low = ((self.raw_low-self.mean)/self.std).repeat(frameskip)
        self.high = ((self.raw_high-self.mean)/self.std).repeat(frameskip)
        self.width = self.low.numel()

    def clip(self, blocks):
        return blocks.clamp(self.low, self.high)

    def to_raw(self, blocks):
        return blocks.reshape(*blocks.shape[:-1], self.frameskip, -1)*self.std+self.mean

    def from_raw(self, actions):
        return ((actions-self.mean)/self.std).flatten(-2)

    def description(self):
        return {"raw_low": self.raw_low.tolist(), "raw_high": self.raw_high.tolist(),
                "normalized_block_low": self.low.tolist(), "normalized_block_high": self.high.tolist(),
                "raw_steps_per_block": self.frameskip, "normalization_mean": self.mean.tolist(),
                "normalization_std": self.std.tolist(),
                "normalization_fit": "memory episodes only", "proposal_clipping": "mapped raw bounds",
                "initial_mean": "zero in normalized coordinates (memory mean in raw coordinates)",
                "raw_bounds_source": "explicit config; simulator execution contract remains unverified"}


def provisional_actions(coordinates, count, std, seed):
    generator = torch.Generator(device="cpu").manual_seed(seed)
    actions = torch.randn(count, coordinates.width, generator=generator)*std
    actions[0].zero_()
    return coordinates.clip(actions.to(coordinates.low.device))

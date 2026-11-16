"""Generate a fixed-size set of exact/noisy seeds; missing seeds keep Gaussian slots."""

from dataclasses import dataclass
import torch

from ..runtime import object_hash


@dataclass
class Injection:
    slots: list[int]
    unbounded_actions: torch.Tensor
    provenance: list[dict]
    requested_slots: int

    @property
    def fingerprint(self):
        return object_hash({"slots": self.slots, "actions": self.unbounded_actions.cpu().tolist(),
                            "provenance": self.provenance, "requested_slots": self.requested_slots})


def build_injection(fragments, settings, seed):
    requested, count = settings["injection_slots"], settings["fragments"]
    if not 0 <= len(fragments.sources) <= count or len(fragments.actions) != len(fragments.sources):
        raise ValueError("Fragment batch exceeds the configured source count")
    horizon, width = fragments.actions.shape[1:]
    generator = torch.Generator(device="cpu").manual_seed(seed)
    # Draw the nominal budget even on shortages: controls share each slot's jitter.
    noise = torch.randn(requested, horizon, width, generator=generator).to(fragments.actions)
    slots, actions, provenance = [], [], []
    offset = 0
    for source in range(count):
        quota = requested//count + (source < requested % count)
        if source < len(fragments.sources):
            for copy in range(quota):
                nominal = offset+copy
                value = fragments.actions[source].clone()
                if copy:
                    value += settings["noise_std"]*noise[nominal]
                # Slot zero remains the ordinary mean, including any future warm start.
                slots.append(nominal+1)
                actions.append(value)
                provenance.append({"slot": nominal+1, "fragment": source,
                                   "transition_id": fragments.sources[source]["transition_id"],
                                   "kind": "exact" if copy == 0 else "perturbed"})
        offset += quota
    values = torch.stack(actions) if actions else fragments.actions.new_empty((0, horizon, width))
    return Injection(slots, values, provenance, requested)

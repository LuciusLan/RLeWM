"""Snapshots contain only adaptable parameters; observed encodings stay unchanged."""

import hashlib

import torch

from ...model import Context


def advance(context, action, prediction):
    return Context(torch.cat((context.latents[:, 1:], prediction.detach().unsqueeze(1)), 1).detach(),
                   torch.cat((context.past_actions[:, 1:], action.detach().unsqueeze(1)), 1).detach())


class ParameterSnapshot:
    def __init__(self, adapter):
        self.names = tuple(adapter.parameter_names)
        self.values = [value.detach().clone() for value in adapter.parameters]
        digest = hashlib.sha256()
        for name, value in zip(self.names, self.values):
            digest.update(name.encode())
            digest.update(value.cpu().contiguous().numpy().tobytes())
        self.fingerprint = digest.hexdigest()

    @torch.no_grad()
    def restore(self, adapter):
        if self.names != tuple(adapter.parameter_names):
            raise ValueError("Adaptation scope changed after preparing a depth model")
        for parameter, value in zip(adapter.parameters, self.values):
            parameter.copy_(value)
        adapter.model.zero_grad(set_to_none=True)


def fresh_counters():
    return {key: 0 for key in (
        "requested_updates", "actual_updates", "support_presentations", "support_positions",
        "support_shortfall_positions", "forward_retrievals", "reverse_retrievals", "reset_count",
        "sequence_evaluations", "transition_predictions", "reference_transition_predictions",
        "candidate_trajectories", "candidate_unique_support_sum")}

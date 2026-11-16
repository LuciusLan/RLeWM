"""One observed-memory gradient step, independent of reset and rollout schedules.

The legacy P1/P2 engine stays unchanged for prepared-cache/source compatibility.
New schedules use this operation with an explicitly owned optimizer lifetime.
"""

from __future__ import annotations

import time
import math

import torch

from ..model import Context


def synchronize(adapter) -> None:
    if adapter.device.type == "cuda":
        torch.cuda.synchronize(adapter.device)


def validate_support(memory, retriever, support, budget: int) -> None:
    indices = support.indices
    if not indices or len(indices) != len(set(indices)) or len(indices) > budget:
        raise ValueError("Support must be nonempty, unique and within the update budget")
    if any(i < 0 or i >= len(memory) for i in indices):
        raise ValueError("Support index outside observed memory")
    if (len(support.weights) != len(indices) or any(not math.isfinite(w) or w < 0 for w in support.weights)
            or abs(sum(support.weights)-1) > 1e-6):
        raise ValueError("Support weights must sum to one")
    episodes = [int(retriever.episode_positions[i]) for i in indices]
    steps = [int(retriever.step_positions[i]) for i in indices]
    if retriever.episode_cap is not None and any(
            episodes.count(ep) > retriever.episode_cap for ep in set(episodes)):
        raise AssertionError("Support exceeds the per-episode cap")
    for left in range(len(indices)):
        for right in range(left+1, len(indices)):
            if episodes[left] == episodes[right] and abs(steps[left]-steps[right]) < retriever.min_step_gap:
                raise AssertionError("Support violates within-episode start spacing")


def observed_support_step(adapter, memory, support, optimizer, settings: dict) -> dict:
    """Supervision is taken only from memory; no query or query target is accepted."""
    if memory.split != "memory":
        raise ValueError("Only observed memory may supervise adaptation")
    indices = torch.tensor(support.indices, device=adapter.device, dtype=torch.long)
    weights = torch.tensor(support.weights, device=adapter.device)
    context = Context(memory.contexts[indices], memory.past_actions[indices])
    action, target = memory.actions[indices], memory.successors[indices].detach()
    synchronize(adapter)
    started = time.perf_counter()
    optimizer.zero_grad(set_to_none=True)
    observed_prediction = adapter.predict(context, action, gradients=True)
    per_item = (observed_prediction-target).square().mean(1)
    observed_loss = (weights*per_item).sum()
    anchor = adapter.anchor_loss()
    loss = observed_loss + settings["anchor_weight"]*anchor
    if not torch.isfinite(loss):
        raise ValueError("Nonfinite observed-support loss")
    loss.backward()
    selected = set(adapter.parameter_names)
    if any(value.grad is not None for name, value in adapter.model.named_parameters() if name not in selected):
        raise AssertionError("A frozen parameter received a gradient")
    norm = torch.nn.utils.clip_grad_norm_(adapter.parameters, settings["clip_norm"], error_if_nonfinite=True)
    optimizer.step()
    synchronize(adapter)
    update_seconds = time.perf_counter()-started
    started = time.perf_counter()
    after_per_item = (adapter.predict(context, action)-target).square().mean(1)
    after_loss = (weights*after_per_item).sum()
    if not torch.isfinite(after_per_item).all():
        raise ValueError("Nonfinite observed-support prediction after update")
    synchronize(adapter)
    evaluation_seconds = time.perf_counter()-started
    modules = adapter.module_metrics()
    if any(not row["parameter_tensors_with_gradient"] for row in modules.values()):
        raise AssertionError("An intended trainable module received no gradient")
    return {"support_loss_before": float(observed_loss.detach()), "support_loss_after": float(after_loss),
            "anchor_before": float(anchor.detach()), "gradient_norm_before_clipping": float(norm),
            "parameter_displacement_mse": float(adapter.anchor_loss().detach()), "module_updates": modules,
            "losses_before": per_item.detach().cpu().tolist(), "losses_after": after_per_item.cpu().tolist(),
            "gradient_seconds": update_seconds, "support_evaluation_seconds": evaluation_seconds}

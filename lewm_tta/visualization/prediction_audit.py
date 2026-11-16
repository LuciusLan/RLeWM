"""Small prediction/physics comparisons shared by task-specific diagnostics.

Observed future states are diagnostic labels only. They never enter CEM or TTT.
"""

import numpy as np
import torch

from ..model import Context


def control_probes(selected, magnitude=.2):
    """Keep the selected plan and add raw-zero and four signed-axis controls."""
    selected = np.asarray(selected, dtype=np.float32)
    if selected.ndim != 2 or selected.shape[1] != 2 or not len(selected) or not np.isfinite(selected).all():
        raise ValueError("Expected a nonempty finite (raw steps, 2) plan")
    probes = {"saved-cem": selected.copy(), "raw-zero": np.zeros_like(selected)}
    for axis, name in enumerate(("x", "y")):
        for sign, label in ((1, "plus"), (-1, "minus")):
            actions = np.zeros_like(selected)
            actions[:, axis] = sign*magnitude
            probes[f"{label}-{name}"] = actions
    return probes


def execute_probe(environment, restore, actions, frameskip):
    """Independently restore the same physical context for every action plan."""
    observation = restore()
    frames, states, metrics, keyframes = [observation.rgb], [observation.state.tolist()], [observation.metrics], []
    if len(actions) % frameskip:
        raise ValueError("Probe actions must span complete ordered blocks")
    for step, action in enumerate(actions, 1):
        observation = environment.step(action)
        if observation.truncated:
            raise ValueError("Diagnostic continuation truncated unexpectedly")
        frames.append(observation.rgb)
        states.append(observation.state.tolist())
        metrics.append(observation.metrics)
        if step % frameskip == 0:
            keyframes.append(observation.rgb)
    return {"frames": frames, "keyframes": keyframes, "states": states, "metrics": metrics}


@torch.no_grad()
def compare_prediction(adapter, coordinates, context, raw_actions, actual_latents, goals):
    """Separate one-block error from autoregressive drift and goal-score error."""
    raw = torch.as_tensor(raw_actions, dtype=torch.float32, device=adapter.device)
    blocks = raw.reshape(-1, coordinates.frameskip, len(coordinates.mean))
    actions = coordinates.from_raw(blocks).unsqueeze(0)
    if actual_latents.shape != (1, len(blocks), adapter.latent_dim):
        raise ValueError("Actual keyframe latents and control horizon disagree")
    predicted = adapter.rollout(context, actions)
    teacher, persistence = [], []
    current = context
    for index, action in enumerate(actions.unbind(1)):
        teacher.append(adapter.predict(current, action))
        persistence.append(current.latents[:, -1])
        current = Context(torch.cat((current.latents[:, 1:], actual_latents[:, index:index+1]), 1),
                          torch.cat((current.past_actions[:, 1:], action[:, None]), 1))
    teacher = torch.stack(teacher, 1)
    persistence = torch.stack(persistence, 1)
    per_step = lambda value: value.square().mean(-1)[0].cpu().tolist()
    return {"raw_roundtrip_max_abs_error": float((coordinates.to_raw(actions)[0].flatten(0, 1)-raw).abs().max()),
            "autoregressive_mse_by_block": per_step(predicted-actual_latents),
            "autoregressive_persistence_mse_by_block": per_step(context.latents[:, -1:]-actual_latents),
            "teacher_forced_mse_by_block": per_step(teacher-actual_latents),
            "teacher_forced_persistence_mse_by_block": per_step(persistence-actual_latents),
            "goal_sse": {name: {"initial_observed": float((context.latents[:, -1]-goal).square().sum()),
                                "predicted_terminal": float((predicted[:, -1]-goal).square().sum()),
                                "observed_terminal": float((actual_latents[:, -1]-goal).square().sum())}
                         for name, goal in goals.items()}}


@torch.no_grad()
def native_rollout_parity(adapter, coordinates, context, images, transform, raw_actions):
    """Check the local latent rollout against LeWM's original pixel-based API."""
    from ..simulation.history import preprocess_frame
    pixels = torch.stack([preprocess_frame(image, transform) for image in images]).to(adapter.device)
    raw = torch.as_tensor(raw_actions, dtype=torch.float32, device=adapter.device)
    actions = coordinates.from_raw(raw.reshape(-1, coordinates.frameskip, len(coordinates.mean)))[None]
    sequence = torch.cat((context.past_actions, actions), 1)[:, None]
    native = adapter.model.rollout({"pixels": pixels[None, None]}, sequence, history_size=adapter.history)
    expected = adapter.rollout(context, actions)
    actual = native["predicted_emb"][:, 0, -len(actions[0]):]
    return float((actual-expected).abs().max())

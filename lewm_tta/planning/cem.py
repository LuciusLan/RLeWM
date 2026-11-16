"""Bounded diagonal-Gaussian CEM with an explicit scoring callback.

One query per call. The callback owns the fixed model and goal. Future proposal
providers can replace population slots here without coupling search to TTT.
"""

from dataclasses import dataclass
import torch

from ..model import Context
from ..runtime import object_hash


@dataclass
class Plan:
    actions: torch.Tensor
    cost: float
    trace: list
    candidate_evaluations: int
    first_population_fingerprint: str


@torch.no_grad()
def score_sequences(adapter, context, goal, actions, batch_size):
    if (actions.ndim != 3 or actions.shape[2] != adapter.action_dim or len(actions) < 1
            or context.latents.shape[0] != 1 or goal.shape != (1, adapter.latent_dim)):
        raise ValueError("Expected one context/goal and (candidates, horizon, action_dim) actions")
    costs = []
    for start in range(0, len(actions), batch_size):
        chunk = actions[start:start+batch_size]
        repeated = Context(context.latents.expand(len(chunk), -1, -1),
                           context.past_actions.expand(len(chunk), -1, -1))
        predicted = adapter.rollout(repeated, chunk)
        # Native LeWM criterion sums squared error over final-step coordinates.
        costs.append((predicted[:, -1]-goal).square().sum(-1))
    result = torch.cat(costs)
    if not torch.isfinite(result).all():
        raise ValueError("Nonfinite CEM goal cost")
    return result


@torch.no_grad()
def solve_cem(score, settings, coordinates, seed, initial_mean=None):
    horizon, population = settings["horizon"], settings["population"]
    shape = (horizon, coordinates.width)
    mean = torch.zeros(shape, device=coordinates.low.device)
    if initial_mean is not None:
        # A later MPC caller may supply its unexecuted prefix; zero-pad the tail.
        if initial_mean.ndim != 2 or initial_mean.shape[1] != shape[1] or len(initial_mean) > horizon:
            raise ValueError("Warm start must contain <= horizon action blocks")
        mean[:len(initial_mean)] = initial_mean.to(mean)
    mean = coordinates.clip(mean)
    std = torch.full_like(mean, settings["initial_std"])
    generator = torch.Generator(device="cpu").manual_seed(seed)
    trace, signature, best = [], None, float("inf")
    for iteration in range(settings["iterations"]):
        # Local RNG makes CEM independent of support sampling, updates and batching.
        noise = torch.randn(population, *shape, generator=generator).to(mean)
        unbounded = mean[None]+std[None]*noise
        unbounded[0] = mean
        candidates = coordinates.clip(unbounded)
        if signature is None:
            signature = object_hash(candidates.cpu().tolist())
        costs = score(candidates)
        if costs.shape != (population,) or not torch.isfinite(costs).all():
            raise ValueError("Scorer must return one finite cost per CEM candidate")
        indices = torch.argsort(costs, stable=True)[:settings["elites"]]
        elites = candidates[indices]
        best = min(best, float(costs[indices[0]]))
        trace.append({"iteration": iteration+1, "population": population,
                      "mean_candidate_cost": float(costs[0]), "minimum_cost": float(costs.min()),
                      "best_scored_cost_so_far": best, "population_mean_cost": float(costs.mean()),
                      "elite_mean_cost": float(costs[indices].mean()),
                      "sampling_std_mean": float(std.mean()),
                      "clipped_coordinate_fraction": float((candidates != unbounded).float().mean()),
                      "candidate_coordinate_std_mean": float(candidates.std(0).mean())})
        mean = elites.mean(0)
        std = elites.std(0, correction=1).clamp_min(settings["minimum_std"])
    # Match the native returned-mean rule, but actually score that returned mean.
    final_cost = score(mean.unsqueeze(0))
    if final_cost.shape != (1,) or not torch.isfinite(final_cost).all():
        raise ValueError("Nonfinite final-mean score")
    return Plan(mean, float(final_cost[0]), trace, population*settings["iterations"]+1, signature)

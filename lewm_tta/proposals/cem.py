"""CEM proposal hook shared by P3b and later TTT x proposal comparisons.

P3-a's recorded solver is preserved while it is running. This extension keeps
its Gaussian/RNG/refit rules; no-injection calls delegate to that solver and
user-run acceptance fixtures verify the injected loop's zero-effect parity.
"""

from dataclasses import dataclass
import torch

from ..planning.cem import Plan, solve_cem
from ..runtime import object_hash


@dataclass
class GuidedPlan:
    plan: Plan
    base_first_population_fingerprint: str
    injection_fingerprint: str | None
    injected_slots: int
    direct_elite_source_fraction: float


@torch.no_grad()
def solve_guided_cem(score, settings, coordinates, seed, injection=None, initial_mean=None):
    if injection is None or not injection.slots:
        plan = solve_cem(score, settings, coordinates, seed, initial_mean)
        return GuidedPlan(plan, plan.first_population_fingerprint,
                          injection.fingerprint if injection is not None else None, 0, 0.0)
    horizon, population = settings["horizon"], settings["population"]
    if (len(set(injection.slots)) != len(injection.slots) or min(injection.slots) < 1
            or max(injection.slots) >= population
            or injection.unbounded_actions.shape != (len(injection.slots), horizon, coordinates.width)
            or not torch.isfinite(injection.unbounded_actions).all()):
        raise ValueError("Injection must replace distinct non-mean slots within the fixed population")
    mean = torch.zeros(horizon, coordinates.width, device=coordinates.low.device)
    if initial_mean is not None:
        if initial_mean.ndim != 2 or initial_mean.shape[1] != coordinates.width or len(initial_mean) > horizon:
            raise ValueError("Warm start must contain <= horizon action blocks")
        mean[:len(initial_mean)] = initial_mean.to(mean)
    mean = coordinates.clip(mean)
    std = torch.full_like(mean, settings["initial_std"])
    generator = torch.Generator(device="cpu").manual_seed(seed)
    trace, signature, base_signature, best = [], None, None, float("inf")
    first_elite_fraction = 0.0
    slots = torch.as_tensor(injection.slots, device=mean.device)
    for iteration in range(settings["iterations"]):
        noise = torch.randn(population, horizon, coordinates.width, generator=generator).to(mean)
        unbounded = mean[None]+std[None]*noise
        unbounded[0] = mean
        if iteration == 0:
            base_signature = object_hash(coordinates.clip(unbounded).cpu().tolist())
            unbounded[slots] = injection.unbounded_actions.to(mean)
        candidates = coordinates.clip(unbounded)
        if signature is None:
            signature = object_hash(candidates.cpu().tolist())
        costs = score(candidates)
        if costs.shape != (population,) or not torch.isfinite(costs).all():
            raise ValueError("Scorer must return one finite cost per CEM candidate")
        indices = torch.argsort(costs, stable=True)[:settings["elites"]]
        elites = candidates[indices]
        best = min(best, float(costs[indices[0]]))
        direct_fraction = float(torch.isin(indices, slots).float().mean()) if iteration == 0 else 0.0
        if iteration == 0:
            first_elite_fraction = direct_fraction
        trace.append({"iteration": iteration+1, "population": population,
                      "mean_candidate_cost": float(costs[0]), "minimum_cost": float(costs.min()),
                      "best_scored_cost_so_far": best, "population_mean_cost": float(costs.mean()),
                      "elite_mean_cost": float(costs[indices].mean()), "sampling_std_mean": float(std.mean()),
                      "clipped_coordinate_fraction": float((candidates != unbounded).float().mean()),
                      "candidate_coordinate_std_mean": float(candidates.std(0).mean()),
                      "directly_injected_candidates": len(slots) if iteration == 0 else 0,
                      "directly_injected_elite_fraction": direct_fraction,
                      "directly_injected_elite_slots": [int(i) for i in indices.cpu().tolist() if i in injection.slots] if iteration == 0 else [],
                      "injected_coordinate_clipping_fraction": float((candidates[slots] != unbounded[slots]).float().mean()) if iteration == 0 else 0.0})
        mean = elites.mean(0)
        std = elites.std(0, correction=1).clamp_min(settings["minimum_std"])
    final = score(mean[None])
    if final.shape != (1,) or not torch.isfinite(final).all():
        raise ValueError("Nonfinite returned-mean cost")
    plan = Plan(mean, float(final[0]), trace, population*settings["iterations"]+1, signature)
    return GuidedPlan(plan, base_signature, injection.fingerprint, len(slots), first_elite_fraction)

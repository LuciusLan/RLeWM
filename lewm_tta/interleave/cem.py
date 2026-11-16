"""CEM's existing sampler/refit, with checkpoint callbacks outside scoring graphs."""

from dataclasses import dataclass
import torch

from ..planning.cem import Plan
from ..runtime import object_hash


@dataclass
class CheckpointResult:
    injection: object
    record: dict
    model_version: int


def solve_interleaved_cem(score, settings, coordinates, seed, checkpoints=(), checkpoint=None,
                          initial_injection=None, initial_mean=None, *, snapshot_iterations=(), on_snapshot=None):
    horizon, population = settings["horizon"], settings["population"]
    snapshots = list(snapshot_iterations)
    if (snapshots != sorted(set(snapshots))
            or any(type(j) is not int or not 1 <= j <= settings["iterations"] for j in snapshots)
            or bool(snapshots) != (on_snapshot is not None)):
        raise ValueError("Choose sorted distinct stopping iterations and a snapshot callback")
    checkpoints = list(checkpoints)
    if checkpoints != sorted(set(checkpoints)) or any(type(j) is not int or not 1 <= j < settings["iterations"] for j in checkpoints):
        raise ValueError("Every checkpoint must leave at least one subsequent CEM iteration")
    shape = (horizon, coordinates.width)
    mean = torch.zeros(shape, device=coordinates.low.device)
    if initial_mean is not None:
        if initial_mean.ndim != 2 or initial_mean.shape[1] != shape[1] or len(initial_mean) > horizon:
            raise ValueError("Warm start must contain <= horizon action blocks")
        mean[:len(initial_mean)] = initial_mean.to(mean)
    mean = coordinates.clip(mean)
    std = torch.full_like(mean, settings["initial_std"])
    generator = torch.Generator(device="cpu").manual_seed(seed)
    trace, records, signature, best = [], [], None, float("inf")
    version, best_version, injection = 0, 0, initial_injection
    for iteration in range(settings["iterations"]):
        noise = torch.randn(population, *shape, generator=generator).to(mean)
        unbounded = mean[None]+std[None]*noise
        unbounded[0] = mean
        slots = [] if injection is None else injection.slots
        if slots:
            if (len(set(slots)) != len(slots) or min(slots) < 1 or max(slots) >= population
                    or injection.unbounded_actions.shape != (len(slots), *shape)
                    or not torch.isfinite(injection.unbounded_actions).all()):
                raise ValueError("Invalid checkpoint injection; preserve mean and population budgets")
            unbounded[torch.as_tensor(slots, device=mean.device)] = injection.unbounded_actions.to(mean)
        candidates = coordinates.clip(unbounded)
        if signature is None:
            signature = object_hash(candidates.cpu().tolist())
        with torch.no_grad():
            costs = score(candidates)
        if costs.shape != (population,) or not torch.isfinite(costs).all():
            raise ValueError("Scorer must return one finite cost per CEM candidate")
        indices = torch.argsort(costs, stable=True)[:settings["elites"]]
        elites = candidates[indices]
        if best_version != version:
            best, best_version = float("inf"), version
        best = min(best, float(costs[indices[0]]))
        trace.append({"iteration": iteration+1, "population": population, "model_version": version,
                      "mean_candidate_cost": float(costs[0]), "minimum_cost": float(costs.min()),
                      "best_scored_cost_so_far": best, "best_cost_model_version": best_version,
                      "population_mean_cost": float(costs.mean()), "elite_mean_cost": float(costs[indices].mean()),
                      "sampling_std_mean": float(std.mean()),
                      "clipped_coordinate_fraction": float((candidates != unbounded).float().mean()),
                      "directly_injected_candidates": len(slots),
                      "directly_injected_elite_fraction": float(torch.isin(indices, torch.as_tensor(
                          slots, dtype=torch.long, device=indices.device)).float().mean()),
                      "population_fingerprint": object_hash(candidates.cpu().tolist()),
                      "injection_fingerprint": injection.fingerprint if injection is not None else None})
        mean = elites.mean(0)
        std = elites.std(0, correction=1).clamp_min(settings["minimum_std"])
        injection = None
        # A stop after j iterations precedes a checkpoint whose effects are
        # intended for iteration j+1. The final stop reuses the regular score.
        if iteration+1 in snapshots and iteration+1 < settings["iterations"]:
            on_snapshot(iteration+1, mean, trace, records, version, None, signature)
        if iteration+1 in checkpoints:
            if checkpoint is None:
                raise ValueError("Checkpoint callback is missing")
            result = checkpoint(iteration+1, candidates.detach(), costs.detach(), indices.detach())
            if result.model_version < version:
                raise ValueError("Model version cannot move backwards")
            version, injection = result.model_version, result.injection
            records.append(result.record)
    with torch.no_grad():
        final = score(mean[None])
    if final.shape != (1,) or not torch.isfinite(final).all():
        raise ValueError("Nonfinite final-mean cost")
    if settings["iterations"] in snapshots:
        on_snapshot(settings["iterations"], mean, trace, records, version, float(final[0]), signature)
    return Plan(mean, float(final[0]), trace, population*settings["iterations"]+1, signature), records

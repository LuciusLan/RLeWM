"""One adaptation phase followed by an explicitly fixed autoregressive model."""

from __future__ import annotations

from contextlib import contextmanager

import numpy as np
import torch


@contextmanager
def fixed_predictor(adapter):
    parameters = list(adapter.model.parameters())
    flags = [parameter.requires_grad for parameter in parameters]
    references = [parameter.detach().clone() for parameter in adapter.parameters]
    adapter.model.requires_grad_(False)
    try:
        with torch.no_grad():
            yield
    finally:
        try:
            for parameter, reference in zip(adapter.parameters, references):
                torch.testing.assert_close(parameter.detach(), reference, rtol=0, atol=0)
            adapter.check_fixed_state()
        finally:
            for parameter, flag in zip(parameters, flags):
                parameter.requires_grad_(flag)


def latent_errors(prediction: torch.Tensor, target: torch.Tensor) -> list[float]:
    prediction = prediction.detach().cpu().numpy().astype(np.float64)
    target = target.detach().cpu().numpy().astype(np.float64)
    if prediction.shape != target.shape or prediction.ndim != 3 or prediction.shape[0] != 1:
        raise ValueError("Expected matching (1, horizon, latent_dim) evaluator tensors")
    values = np.mean((prediction - target) ** 2, axis=-1).reshape(-1)
    if not np.isfinite(values).all():
        raise ValueError("Nonfinite P2 rollout errors")
    return values.tolist()


def compact_adaptation(record: dict, first_target: torch.Tensor) -> dict:
    """Keep audit/analysis values, not all latent vectors or ranked candidates."""
    target = first_target.detach().cpu().numpy().reshape(-1).astype(np.float64)
    rounds = []
    for row in record["rounds"]:
        values = {key: row[key] for key in (
            "update", "support_loss_before", "support_loss_after", "gradient_norm_before_clipping",
            "parameter_displacement_mse", "prediction_correction_mse", "unique_support",
            "cumulative_unique_support", "unique_support_episodes", "support_budget_shortfall",
            "turnover_fraction", "presentations_so_far")}
        values["query_mse_after"] = float(np.mean((np.asarray(row["prediction_after"], dtype=np.float64)-target)**2))
        values["support"] = [{key: item[key] for key in (
            "transition_id", "source_episode_position", "start_in_episode", "weight", "channels")}
            for item in row["support"]]
        rounds.append(values)
    return {"query_rng_seed": record["query_rng_seed"], "rounds": rounds,
            "support_presentations": record["support_presentations"],
            "cumulative_unique_support": record["cumulative_unique_support"],
            "parameter_displacement_mse": record["parameter_displacement_mse"],
            "initial_rank_summary": record["rank_diagnostics_initial"].get("topk_cross_ranks"),
            "post_adaptation_rank_summary": record["rank_diagnostics_final"].get("topk_cross_ranks"),
            "timing_seconds": record["timing_seconds"], "checks": record["checks"]}


def compact_schedule_adaptation(record: dict, targets: torch.Tensor) -> dict:
    """Score per-position/round predictions only after the trajectory completes."""
    targets = targets.detach().cpu().numpy().astype(np.float64)[0]
    result = {key: value for key, value in record.items()
              if key not in ("prediction_initial", "prediction_final", "positions", "rounds")}
    result["rounds"], result["positions"] = [], []
    for kind in ("rounds", "positions"):
        for row in record[kind]:
            compact = {key: value for key, value in row.items() if not key.startswith("prediction_")}
            target = targets[row["position"]]
            for source, name in (("prediction_before", "query_mse_before"),
                                 ("prediction_after", "query_mse_after"),
                                 ("prediction_initial", "query_mse_before"),
                                 ("prediction_final", "query_mse_after")):
                if source in row:
                    compact[name] = float(np.mean((np.asarray(row[source], dtype=np.float64)-target)**2))
            if "prediction_correction_mse" in row:
                compact["prediction_correction_mse"] = row["prediction_correction_mse"]
            result[kind].append(compact)
    return result

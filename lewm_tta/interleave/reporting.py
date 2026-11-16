"""Complete paired arm matrices, episode uncertainty, and version-aware budgets."""

from itertools import product
import numpy as np

from ..planning.reporting import interval
from .config import ARMS, STAGE, adaptation_schedule, checkpoint_updates, validate_depth_updates


def summarize(records, config):
    arms, seeds = config["interleave"]["arms"], config["p1"]["evaluation"]["seeds"]
    ids = sorted({row["query"]["transition_id"] for row in records})
    lookup = {(row["query"]["transition_id"], row["seed"], row["arm"]): row for row in records}
    if not ids or len(lookup) != len(records) or set(lookup) != set(product(ids, seeds, arms)):
        raise ValueError("Incomplete or duplicated interleave query/seed/arm matrix")
    horizon, planning = config["planning"]["horizon"], config["planning"]
    checkpoints = config["interleave"]["checkpoints"]
    expected_evaluations = planning["population"]*planning["iterations"]+1
    for query, seed in product(ids, seeds):
        reference = lookup[query, seed, "baseline"]
        for arm in arms:
            row = lookup[query, seed, arm]
            if (row["query"] != reference["query"] or row["goal_raw_row"] != reference["goal_raw_row"]
                    or row["cem_seed"] != reference["cem_seed"]
                    or row["cem_first_population_fingerprint"] != reference["cem_first_population_fingerprint"]
                    or row["initial_injection_fingerprint"] != reference["initial_injection_fingerprint"]):
                raise ValueError("Paired query, goal, initial proposals, or randomness differs")
            warmup = checkpoints[0] if checkpoints else planning["iterations"]
            if ([step["population_fingerprint"] for step in row["cem_trace"][:warmup]] !=
                    [step["population_fingerprint"] for step in reference["cem_trace"][:warmup]]):
                raise ValueError("Warm-up populations must agree across arms")
            if (row["cem_candidate_evaluations"] != expected_evaluations
                    or row["cem_transition_predictions"] != expected_evaluations*horizon
                    or len(row["cem_trace"]) != planning["iterations"]
                    or [cp["checkpoint"] for cp in row["checkpoints"]] != checkpoints):
                raise ValueError("Interleave population, horizon, or checkpoint budget mismatch")
            if len(row["selected_actions_normalized"]) != horizon or len(row["recorded_action_mse_by_block"]) != horizon:
                raise ValueError("Returned plan or recorded diagnostic has the wrong horizon")
            adapted = row["adaptation"]
            maximum = len(checkpoints)*checkpoint_updates(config) if ARMS[arm][0] else 0
            if (adapted["updates"] != len(adapted["rounds"]) or adapted["updates"] > maximum
                    or adapted["model_version"] != adapted["updates"] or adapted["reset_count"] != 1
                    or adapted["support_presentations"] != sum(len(r["support"]) for r in adapted["rounds"])):
                raise ValueError("Interleave update or model lifetime mismatch")
            if row["counters"]["actual_updates"] != adapted["updates"]:
                raise ValueError("Update counters differ")
            if adapted["updates"] < maximum and not any(cp["skip_reason"] for cp in row["checkpoints"]):
                raise ValueError("Missing updates require an explicit skip reason")
            for r in adapted["rounds"]:
                support = r["support"]
                if (not support or len(support) > config["adaptation"]["support_size"]
                        or len({i["memory_index"] for i in support}) != len(support)
                        or abs(sum(i["weight"] for i in support)-1) > 1e-6):
                    raise ValueError("Invalid saved global support budget/weights")
            version = 0
            for cp in row["checkpoints"]:
                validate_depth_updates(cp, config)
                if cp["model_version_before"] != version or cp["model_version_after"] != version+cp["actual_updates"]:
                    raise ValueError("Checkpoint model versions are discontinuous")
                version = cp["model_version_after"]
                if len(cp["rounds"]) != cp["actual_updates"]:
                    raise ValueError("Checkpoint update audit is incomplete")
            for step in row["cem_trace"]:
                expected_version = sum(cp["actual_updates"] for cp in row["checkpoints"]
                                       if cp["checkpoint"] < step["iteration"])
                if step["model_version"] != expected_version or step["best_cost_model_version"] != expected_version:
                    raise ValueError("A search score belongs to a stale model")
            if not all(row["checks"].values()):
                raise ValueError("Per-record acceptance check failed")
    episodes = np.asarray([lookup[q, seeds[0], "baseline"]["query"]["source_episode_position"] for q in ids])
    draws, tolerance = config["evaluation"]["bootstrap_samples"], config["p1"]["evaluation"]["harm_tolerance"]
    forecasts = {arm: np.asarray([[lookup[q, seed, arm]["recorded_action_mse_by_block"]
                                  for seed in seeds] for q in ids]) for arm in arms}
    if any(not np.isfinite(value).all() or (value < 0).any() for value in forecasts.values()):
        raise ValueError("Invalid forecast diagnostic")
    common = {arm: np.asarray([[lookup[q, seed, arm]["selected_plan_frozen_model_goal_sse"]
                               for seed in seeds] for q in ids]).mean(1) for arm in arms}
    if any(not np.isfinite(value).all() for value in common.values()):
        raise ValueError("Nonfinite common-reference cost")

    def estimate(values):
        return {"mean_difference": float(values.mean()), "ci95": interval(values, episodes, draws)}

    groups = {}
    for arm in arms:
        rows = [lookup[q, seed, arm] for q, seed in product(ids, seeds)]
        delta = forecasts[arm]-forecasts["baseline"]
        groups[arm] = {"recorded_endpoint_mse": float(forecasts[arm][:, :, -1].mean()),
            "recorded_path_mse": float(forecasts[arm].mean()),
            "forecast_delta_vs_baseline": estimate(delta[:, :, -1].mean(1)),
            "harm_fraction": float((delta[:, :, -1] > tolerance).mean()),
            "positive_harm_mass": float(np.maximum(delta[:, :, -1], 0).mean()),
            "common_frozen_sse": float(common[arm].mean()),
            "common_delta_vs_baseline": estimate(common[arm]-common["baseline"]),
            "own_model_sse": float(np.mean([r["selected_model_goal_sse"] for r in rows])),
            "primary_seconds": float(np.mean([r["primary_seconds_with_checks"] for r in rows])),
            "primary_seconds_p95": float(np.quantile([r["primary_seconds_with_checks"] for r in rows], .95)),
            "cem_candidate_evaluations": expected_evaluations,
            "mean_injected_candidates": float(np.mean([sum(s["directly_injected_candidates"] for s in r["cem_trace"]) for r in rows])),
            "timing_seconds": {name: float(np.mean([r["timing_seconds"][name] for r in rows]))
                               for name in rows[0].get("timing_seconds", {})},
            "compute": {name: float(np.mean([r["counters"][name] for r in rows])) for name in rows[0]["counters"]}}
    interaction = (estimate(common["combined"]-common["rca"]-common["proposals"]+common["baseline"])
                   if set(ARMS).issubset(arms) else None)
    return {"queries": len(ids), "episodes": len(set(episodes.tolist())), "rows": len(records), "seeds": seeds,
            "arms": groups, "common_frozen_proxy_interaction": interaction,
            "physical_success": None, "physical_interaction": None,
            "uncertainty": "Seed-average paired episode bootstrap; pointwise exploratory 95% percentile intervals",
            "interpretation": "Forecasts use recorded controls after search. Search costs are model proxies; no selected plan is executed."}


def format_summary(report):
    summary = report["summary"]
    lines = [f"{STAGE}: checkpoint RCA and intermediate action continuations", "",
             f"{summary['queries']} queries / {summary['episodes']} episodes; seeds {summary['seeds']}.", "",
             "| Arm | Recorded endpoint MSE | Harm | Own-model SSE | Common Frozen SSE | Seconds/call | p95 seconds |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    if adaptation_schedule(report["config"]) == "sequential-depth":
        lines[2:2] = [f"Adaptation: sequential-depth, {report['config']['adaptation']['updates']} updates per "
                      f"selected depth; {checkpoint_updates(report['config'])} per checkpoint. "
                      "Final weights score every rollout depth.", ""]
    for arm, row in summary["arms"].items():
        lines.append(f"| {arm} | {row['recorded_endpoint_mse']:.7f} | {row['harm_fraction']:.1%} | "
                     f"{row['own_model_sse']:.5f} | {row['common_frozen_sse']:.5f} | "
                     f"{row['primary_seconds']:.3f} | {row['primary_seconds_p95']:.3f} |")
    lines += ["", "Paired differences use the common Frozen reference (lower is better):", "",
              "| Contrast | Mean SSE difference | 95% episode interval |", "|---|---:|---|"]
    for arm, row in summary["arms"].items():
        value = row["common_delta_vs_baseline"]
        lines.append(f"| {arm} minus baseline | {value['mean_difference']:+.6f} | {value['ci95']} |")
    value = summary["common_frozen_proxy_interaction"]
    if value:
        lines.append(f"| Combined − RCA − proposals + baseline | {value['mean_difference']:+.6f} | {value['ci95']} |")
    lines += ["", "Compute per standalone call:", "",
              "| Arm | CEM sequences incl. final mean | Reference transition predictions | Updates | Support presentations | Continuation lookups |",
              "|---|---:|---:|---:|---:|---:|"]
    for arm, row in summary["arms"].items():
        c = row["compute"]
        lines.append(f"| {arm} | {row['cem_candidate_evaluations']} | {c['reference_transition_predictions']:.1f} | "
                     f"{c['actual_updates']:.1f} | {c['support_presentations']:.1f} | {c['continuation_retrievals']:.1f} |")
    lines += ["", summary["interpretation"],
              "Own-model costs use different predictors across RCA arms. Frozen rescoring can favor its own plans; it does not identify physical synergy.",
              "CEM best-so-far costs reset after model changes. Primary latency includes reset, retrieval, adaptation, rerolls, injection and scoring checks.",
              "A single checkpoint tests one correction/injection stage; repeated proposal-to-adaptation feedback requires multiple checkpoints."]
    return "\n".join(lines)+"\n"

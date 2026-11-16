"""User-run acceptance checks for persistent/reset temporal adaptation.

These use the selected checkpoint and observed memory. They are deliberately
not executed by the agent; the check/evaluate commands run them on the user's GPU.
"""

from __future__ import annotations

from copy import deepcopy

import torch

from .cohorts import rollout_inputs
from .config import scope_names
from .inference import compact_schedule_adaptation, fixed_predictor
from .schedule_reporting import summarize_schedules
from .temporal import run_scheduled_rollout
from .record_io import check_record_roundtrip


def _support(round_record):
    return [{key: item[key] for key in ("transition_id", "weight", "channels")}
            for item in round_record["support"]]


def _statistics_check(config, query, target_rows, outputs, scope, seed):
    """Known scalar errors plus actual traces exercise pairing/budget checks."""
    fixture = deepcopy(config)
    fixture["scopes"] = [scope]
    fixture["horizons"] = [1, 3]
    fixture["schedule_names"] = ["initial-three", "temporal-one"]
    fixture["schedules"] = {"initial-three": {"placement": "initial", "updates": 3, "persistence": "trajectory"},
                            "temporal-one": {"placement": "each-step", "updates": 1, "persistence": "trajectory"}}
    fixture["adaptation"]["methods"] = ["frozen", "forward"]
    fixture["p1"]["evaluation"]["seeds"] = [seed]
    fixture["evaluation"]["bootstrap_samples"] = 8
    rows = []
    for name in fixture["schedule_names"]:
        for method in fixture["adaptation"]["methods"]:
            prediction, trace = outputs[(name, method)]
            values = [1., 1., 1.] if method == "frozen" else ([.8, .8, .8] if name == "initial-three" else [.6, .6, .6])
            compact = compact_schedule_adaptation(trace, torch.zeros_like(prediction))
            for position, value in zip(compact["positions"], values):
                position["query_mse_after"] = value
            rows.append({"scope": scope, "schedule": name, "method": method, "seed": seed, "query": query.metadata,
                         "target_raw_rows": target_rows.tolist(), "mse_by_block": values,
                         "adaptation": compact, "total_seconds": .1})
    summary = summarize_schedules(rows, fixture)
    check_record_roundtrip(rows, fixture, summarize_schedules)
    pair = summary["cross_schedule"]["temporal-one-minus-initial-three"][scope]["forward"]
    assert pair["3"]["matched_actual_updates"] and pair["3"]["matched_support_presentations"]
    assert not pair["1"]["matched_actual_updates"]
    assert abs(pair["3"]["endpoint_delta"]["mean"]+.2) < 1e-12
    for broken in (rows[:-1], rows+[rows[0]]):
        try:
            summarize_schedules(broken, fixture)
        except ValueError:
            pass
        else:
            raise AssertionError("Incomplete/duplicate schedule matrices must be rejected")
    corrupted = deepcopy(rows)
    corrupted[-1]["adaptation"]["support_presentations"] += 1
    try:
        summarize_schedules(corrupted, fixture)
    except ValueError:
        pass
    else:
        raise AssertionError("Corrupt optimization accounting must be rejected")


def run_temporal_checks(resources, queries, config: dict) -> dict:
    # The default cohort is H5-safe; require H3 for the matched-placement fixture.
    query, actions, targets = rollout_inputs(queries, 0, 3)
    adapter, engine = resources.adapter, resources.engine
    seed = resources.config["evaluation"]["seeds"][0]
    settings = engine.settings
    checks = {}
    initial = {"placement": "initial", "updates": 3, "persistence": "trajectory"}
    temporal = {"placement": "each-step", "updates": 1, "persistence": "trajectory"}
    try:
        for scope in scope_names(config):
            adapter.set_scope(scope)
            outputs = {(name, method): run_scheduled_rollout(engine, query, actions, method, seed, schedule)
                       for name, schedule in (("initial-three", initial), ("temporal-one", temporal))
                       for method in ("frozen", "forward")}
            torch.testing.assert_close(outputs[("initial-three", "frozen")][0],
                                       outputs[("temporal-one", "frozen")][0], rtol=0, atol=0)
            # Compare the new session's initial-only updates/rollout with the
            # established engine at two budgets, not just a self-comparison.
            for updates in (1, 3):
                scheduled, trace = (outputs[("initial-three", "forward")] if updates == 3 else
                                    run_scheduled_rollout(engine, query, actions, "forward", seed, initial | {"updates": updates}))
                engine.settings = settings | {"updates": updates}
                legacy = engine.run(query, "forward", seed)
                with fixed_predictor(adapter):
                    expected = adapter.rollout(query.context, actions)
                torch.testing.assert_close(scheduled, expected, rtol=1e-6, atol=1e-7)
                assert [_support(row) for row in trace["rounds"]] == [_support(row) for row in legacy["rounds"]]
                assert trace["actual_updates"] == updates
            engine.settings = settings
            # An explicit schedule LR must reach AdamW without leaking into
            # another schedule or the engine's shared fallback settings.
            override_rate = settings["learning_rate"]/5
            override_schedule = initial | {"updates": 2, "learning_rate": override_rate}
            overridden, override_trace = run_scheduled_rollout(engine, query, actions, "forward", seed, override_schedule)
            assert engine.settings is settings
            assert override_trace["learning_rate"] == override_rate
            assert override_trace["actual_learning_rate_sum"] == 2*override_rate
            assert all(row["learning_rate"] == override_rate
                       for row in override_trace["positions"]+override_trace["rounds"])
            engine.settings = settings | {"updates": 2, "learning_rate": override_rate}
            legacy = engine.run(query, "forward", seed)
            with fixed_predictor(adapter):
                expected = adapter.rollout(query.context, actions)
            torch.testing.assert_close(overridden, expected, rtol=1e-6, atol=1e-7)
            assert [_support(row) for row in override_trace["rounds"]] == [_support(row) for row in legacy["rounds"]]
            engine.settings = settings
            fixed, fixed_trace = run_scheduled_rollout(engine, query, actions, "fr-fixed", seed, temporal)
            refreshed, refresh_trace = run_scheduled_rollout(engine, query, actions, "fr-refresh", seed, temporal)
            torch.testing.assert_close(fixed, refreshed, rtol=1e-6, atol=1e-7)
            assert [_support(row) for row in fixed_trace["rounds"]] == [_support(row) for row in refresh_trace["rounds"]]
            assert fixed_trace["actual_updates"] == 3
            assert [row["optimizer_steps_before"] for row in fixed_trace["positions"]] == [0, 1, 2]
            assert fixed_trace["reset_count"] == 1
            assert fixed_trace["retrieval_calls"] == {"forward": 3, "reverse": 3, "state": 0}
            _, reset = run_scheduled_rollout(engine, query, actions, "forward", seed, temporal | {"persistence": "position"})
            assert reset["reset_count"] == 3
            assert all(row["optimizer_steps_before"] == 0 and row["parameter_displacement_before"] == 0
                       for row in reset["positions"])
            combined, combined_trace = run_scheduled_rollout(engine, query, actions, "fr-refresh", seed,
                                                            temporal | {"updates": 2})
            assert combined_trace["actual_updates"] == 6
            assert combined_trace["retrieval_calls"] == {"forward": 3, "reverse": 6, "state": 0}
            torch.testing.assert_close(torch.tensor(combined_trace["rounds"][0]["prediction_after"]),
                                       torch.tensor(fixed_trace["rounds"][0]["prediction_after"]), rtol=1e-6, atol=1e-7)
            # A different schedule has changed the live model; replay must reset
            # parameters, optimizer and support RNG completely.
            repeat, repeated_trace = run_scheduled_rollout(engine, query, actions, "fr-fixed", seed, temporal)
            torch.testing.assert_close(repeat, fixed, rtol=0, atol=0)
            assert [_support(row) for row in repeated_trace["rounds"]] == [_support(row) for row in fixed_trace["rounds"]]
            _statistics_check(config, query, targets, outputs, scope, seed)
            checks[scope] = {"initial_only_matches_legacy_u1_and_u3": True,
                             "schedule_learning_rate_override_matches_legacy": True,
                             "schedule_learning_rate_does_not_change_global_settings": True,
                             "frozen_schedule_equivalence": True, "u1_fixed_refresh_equivalence": True,
                             "first_update_equivalence": True, "optimizer_persists_within_trajectory": True,
                             "reset_per_position_control": True, "deterministic_reset_between_trajectories": True,
                             "both_pools_rebuilt_at_changed_context": True, "inner_reverse_refresh_clock": True,
                             "matched_h3_budgets_but_unmatched_h1_budgets": True,
                             "compact_full_none_record_roundtrip_and_checksums": True,
                             "incomplete_duplicate_and_corrupt_records_rejected": True}
    finally:
        engine.settings = settings
        adapter.reset(seed)
    return {"iteration_acceptance": checks}

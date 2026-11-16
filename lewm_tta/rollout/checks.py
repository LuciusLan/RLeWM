"""User-run fixtures for action alignment, autoregression, and fixed parameters."""

from __future__ import annotations

from copy import deepcopy

import numpy as np
import torch

from ..checks import run_checks as p1_checks
from ..model import Context, LeWMAdapter
from .inference import fixed_predictor
from .reporting import summarize
from .cohorts import horizon_rows, filter_cohort, sample_cohort, rollout_inputs
from ..adaptation import AdaptationEngine


def check_alignment() -> dict:
    actions, targets = horizon_rows(7, 3, 5, 5)
    np.testing.assert_array_equal(actions, np.arange(17, 42))
    np.testing.assert_array_equal(targets, [22, 27, 32, 37, 42])
    # The last target must exist, even when all preceding action rows exist.
    source = {"data_config": {"history": 3, "frameskip": 5},
              "episode_metadata": {"offsets": [0, 50], "lengths": [50, 50]},
              "episodes": {"validation": [0]},
              "evaluation_windows": {"validation": {"mode": "subset", "raw_start_rows": [14, 15]}}}
    cohort = filter_cohort(source, "validation", 5)
    assert cohort["raw_start_rows"] == [14] and cohort["dropped_for_short_future"] == 1
    return {"ordered_future_action_blocks": True, "target_frame_alignment": True,
            "last_target_episode_boundary": True, "common_horizon_cohort": True}


def check_autoregression() -> dict:
    class Recurrence:
        def __init__(self):
            self.contexts = []

        def predict(self, context, action):
            self.contexts.append(context.latents.detach().clone())
            return context.latents[:, -1] + context.past_actions[:, -1] + action

    toy = Recurrence()
    context = Context(torch.tensor([[[0.], [10.], [20.]]]), torch.tensor([[[2.], [3.]]]))
    actions = torch.tensor([[[5.], [7.], [11.]]])
    predictions = LeWMAdapter.rollout(toy, context, actions)
    torch.testing.assert_close(predictions, torch.tensor([[[28.], [40.], [58.]]]), rtol=0, atol=0)
    assert toy.contexts[1][0, -1, 0].item() == 28
    assert toy.contexts[2][0, -1, 0].item() == 40
    return {"predicted_context_feedback": True, "past_action_history_shift": True,
            "no_intermediate_target_in_rollout_interface": True}


def check_statistics(p1: dict) -> dict:
    config = {"p1": deepcopy(p1), "adaptation": {"methods": ["frozen", "forward"]},
              "evaluation": {"bootstrap_samples": 16, "log_every": 0}, "horizons": [1, 3, 5]}
    config["p1"]["evaluation"]["seeds"] = [42]
    records = [{"method": method, "seed": 42,
                "query": {"transition_id": str(ep), "source_episode_position": ep},
                "mse_by_block": [1., 4., 9., 16., 25.] if method == "frozen" else [.5, 2., 18., 8., 12.5],
                "adaptation": {"support_presentations": 0 if method == "frozen" else 32,
                               "cumulative_unique_support": 0 if method == "frozen" else 32},
                "total_seconds": 0.1} for ep in (10, 20) for method in ("frozen", "forward")]
    summary = summarize(records, config)
    h3 = summary["methods"]["forward"]["horizons"]["3"]
    assert h3["endpoint_mse"]["mean"] == 18 and h3["paired_delta_vs_frozen"]["mean"] == 9
    assert h3["harm_fraction"] == 1 and np.isclose(h3["mean_path_mse"]["mean"], 20.5/3)
    assert summary["methods"]["forward"]["horizons"]["1"]["improvement_fraction"] == 1
    try:
        summarize(records[:-1], config)
    except ValueError:
        pass
    else:
        raise AssertionError("An unpaired/incomplete comparison must be rejected")
    return {"endpoint_distinct_from_path_metric": True, "paired_horizon_specific_harm": True,
            "incomplete_comparisons_rejected": True}


def run_rollout_checks(resources, queries, horizon: int, seed: int) -> dict:
    checks = check_alignment() | check_autoregression() | check_statistics(resources.config)
    checks["p1_acceptance"] = p1_checks(resources.engine, queries, seed)
    query, actions, _ = rollout_inputs(queries, 0, horizon)
    resources.adapter.reset(seed)
    expected = resources.adapter.predict(query.context, query.action)
    with fixed_predictor(resources.adapter):
        actual = resources.adapter.rollout(query.context, actions)
        torch.testing.assert_close(actual[:, 0], expected, rtol=1e-6, atol=1e-7)
        repeated = Context(query.context.latents.repeat(2, 1, 1), query.context.past_actions.repeat(2, 1, 1))
        batch = resources.adapter.rollout(repeated, actions.repeat(2, 1, 1))
        torch.testing.assert_close(batch[0], actual[0], rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(batch[1], actual[0], rtol=1e-5, atol=1e-6)
    assert all(parameter.requires_grad for parameter in resources.adapter.parameters)
    resources.adapter.reset(seed)
    checks.update(one_step_rollout_equivalence=True, batched_rollout_equivalence=True,
                  parameters_and_buffers_fixed_during_rollout=True, adaptation_grad_flags_restored=True)
    return checks


def check_cohort() -> dict:
    source = {"data_config": {"history": 3, "frameskip": 5},
        "episode_metadata": {"offsets": [0, 100], "lengths": [100, 100]},
        "episodes": {"validation": [0], "test": [1]}}
    selection = {"windows_per_episode": 4, "seed": 123}
    cohort = sample_cohort(source, "validation", 5, selection)
    assert cohort == sample_cohort(source, "validation", 5, selection)
    assert cohort["queries"] == 4 and cohort["episodes"] == 1
    assert max(cohort["raw_start_rows"])+35 < 100
    assert len(set(cohort["raw_start_rows"])) == 4
    other = sample_cohort(source, "test", 5, selection)
    assert min(other["raw_start_rows"]) >= 100 and max(other["raw_start_rows"])+35 < 200
    return {"q4_horizon_safe_sampling": True, "deterministic_cohort": True,
            "episode_partition_preserved": True}


def check_scope_statistics(config: dict) -> dict:
    fixture = deepcopy(config)
    fixture.pop("schedules", None)
    fixture.pop("schedule_names", None)
    fixture["scopes"] = ["last-block", "predictor"]
    fixture["adaptation"]["methods"] = ["frozen", "forward"]
    fixture["p1"]["evaluation"]["seeds"] = [42]
    fixture["evaluation"]["bootstrap_samples"] = 16
    # Exercise the active horizon and both standard profiles. These records
    # are synthetic reporting fixtures, independent of cached trajectory length.
    profiles = list(dict.fromkeys((tuple(config["horizons"]), (1,), (1, 3), (1, 3, 5))))
    for horizons in profiles:
        fixture["horizons"] = list(horizons)
        horizon = max(horizons)
        records = [{"scope": scope, "method": method, "seed": 42,
            "query": {"transition_id": str(ep), "source_episode_position": ep},
            "mse_by_block": ([1.] if method == "frozen" else [.8 if scope == "last-block" else .5])*horizon,
            "adaptation": {"support_presentations": 0, "cumulative_unique_support": 0, "module_updates": {}},
            "total_seconds": 0.1}
            for scope in fixture["scopes"] for ep in (10, 20) for method in fixture["adaptation"]["methods"]]
        result = summarize(records, fixture)
        assert result["horizons"] == list(horizons)
        for h in horizons:
            assert np.isclose(result["cross_scope"]["predictor-minus-last-block"]["forward"][str(h)]["endpoint_delta"]["mean"], -.3)
        try:
            summarize(records[:-1], fixture)
        except ValueError:
            pass
        else:
            raise AssertionError("Missing scope/method/query/seed rows must be rejected")
        malformed = deepcopy(records)
        for row in malformed:
            row["mse_by_block"].append(row["mse_by_block"][-1])
        try:
            summarize(malformed, fixture)
        except ValueError as error:
            if "horizon/record mismatch" not in str(error):
                raise
        else:
            raise AssertionError("Wrong-length horizon records must still be rejected")
    return {"paired_scope_statistics": True, "incomplete_scope_matrix_rejected": True,
            "scope_statistics_horizon_profiles": [list(horizons) for horizons in profiles],
            "wrong_length_horizon_records_rejected": True}


def run_scope_checks(resources, queries, config: dict) -> dict:
    checks = check_cohort() | check_scope_statistics(config)
    seed, adapter = resources.config["evaluation"]["seeds"][0], resources.adapter
    original_settings = resources.engine.settings
    reference = None
    per_scope = {}
    try:
        for scope in config["scopes"]:
            adapter.set_scope(scope)
            adapter.reset(seed)
            query = queries.query(0)
            frozen = adapter.predict(query.context, query.action)
            if reference is None:
                reference = frozen.detach().clone()
            torch.testing.assert_close(frozen, reference, rtol=0, atol=0)
            flags = {name for name, parameter in adapter.model.named_parameters() if parameter.requires_grad}
            assert flags == set(adapter.parameter_names)
            assert not any(name.startswith(("encoder.", "projector.")) for name in flags)
            action_trainable = any(name.startswith("action_encoder.") for name in flags)
            assert action_trainable == (scope == "predictor-action")
            per_scope[scope] = run_rollout_checks(resources, queries, max(config["horizons"]), seed)
            # An actual observed-support backward pass, executed by the user,
            # verifies the action encoder isn't silently behind no_grad.
            resources.engine.settings = original_settings | {"updates": 1}
            updated = resources.engine.run(query, "forward", seed)
            assert updated["rounds"], "The scope fixture requires nonempty memory"
            assert all(row["parameter_tensors_with_gradient"] > 0 and
                       row["last_update_gradient_l2_after_clipping"] > 0 and
                       row["displacement_from_checkpoint_mse"] > 0
                       for row in updated["module_updates"].values())
            if scope == "last-block":
                # Share model storage but call the original adapter methods.
                # Both engines reset fully, avoiding a second GPU model copy.
                legacy = LeWMAdapter.__new__(LeWMAdapter)
                legacy.__dict__.update(adapter.__dict__)
                expected = AdaptationEngine(legacy, resources.memory, resources.engine.retriever,
                    resources.engine.settings).run(query, "forward", seed)
                assert adapter.parameter_names == list(adapter._legacy_names)
                torch.testing.assert_close(torch.tensor(updated["prediction_final"]),
                                           torch.tensor(expected["prediction_final"]), rtol=0, atol=0)
                assert updated["rounds"][0]["support"] == expected["rounds"][0]["support"]
                per_scope[scope]["legacy_last_block_update_equivalence"] = True
            per_scope[scope].update(exact_parameter_scope=True, intended_module_gradients_present=True,
                                   action_encoder_gradient_route=True)
    finally:
        resources.engine.settings = original_settings
        adapter.reset(seed)
    checks["per_scope"] = per_scope
    checks["frozen_predictions_identical_across_scopes"] = True
    return checks


def run_checks(resources, queries, config: dict) -> dict:
    if "schedules" in config:
        from .temporal_checks import run_temporal_checks
        return run_scope_checks(resources, queries, config) | run_temporal_checks(resources, queries, config)
    if "scopes" in config:
        return run_scope_checks(resources, queries, config)
    return run_rollout_checks(resources, queries, max(config["horizons"]), resources.config["evaluation"]["seeds"][0])

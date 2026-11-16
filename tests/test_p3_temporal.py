"""Model-free config, dispatch and synthetic-budget checks for temporal CEM."""

from copy import deepcopy
import contextlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from experiments.catalog import profile
from experiments.cli import dispatch
from experiments.config import resolve
from lewm_tta.runtime import write_json
from lewm_tta.planning.temporal.config import (
    load_config, nominal_budget, override_seeds, seed_schedule, seeds_for_mode, validate_config,
)
from lewm_tta.planning.temporal.reporting import format_summary, summarize


def fixture_records(config):
    rows = []
    planning = config["planning"]
    for episode in (100, 200):
        for mode in config["temporal"]["modes"]:
            budget = nominal_budget(config, mode)
            counters = {"requested_updates": budget["requested_gradient_updates"],
                        "actual_updates": budget["requested_gradient_updates"],
                        "support_presentations": budget["requested_support_presentations"],
                        "support_shortfall_positions": 0,
                        "sequence_evaluations": budget["cem_sequence_evaluations"],
                        "transition_predictions": budget["cem_transition_predictions"], "rounds": [], "audit_sha256": "fixture"}
            row = {"query": {"transition_id": str(episode), "source_episode_position": episode}, "seed": 42,
                         "mode": mode, "goal_raw_row": episode+25, "reference_fingerprint": "same", "cem_seed": 17,
                         "cem_first_population_fingerprint": "same", "cem_trace": [{}]*planning["iterations"],
                         "adaptation": counters, "selected_actions_normalized": [[0.]*10]*planning["horizon"],
                         "recorded_action_mse_by_block": [.1 if mode == "frozen" else .08]*planning["horizon"],
                         "selected_model_goal_sse": 2., "selected_plan_frozen_model_goal_sse": 2.,
                         "primary_seconds_with_checks": .5, "diagnostic_seconds_with_checks": .1,
                         "diagnostic_common_frozen_sequence_evaluations": 1,
                         "diagnostic_common_frozen_transition_predictions": planning["horizon"],
                         "diagnostic_adaptation": {"actual_updates": 0, "requested_updates":
                             3*planning["horizon"]*config["adaptation"]["updates"] if mode == "candidate-temporal" else 0,
                             "sequence_evaluations": 3, "transition_predictions": 3*planning["horizon"]},
                         "checks": {"fixture": True}}
            for seed in seeds_for_mode(config, mode):
                saved = deepcopy(row)
                saved["seed"] = seed
                rows.append(saved)
    return rows


class P3TemporalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)/"resolved.json"
            write_json(path, resolve(profile("p3", "temporal")[1]))
            cls.config = load_config(path)

    def test_temporal_profile_is_bounded_and_reuses_source(self):
        self.assertEqual(self.config["evaluation"]["query_limit"], 8)
        self.assertEqual(self.config["p1"]["evaluation"]["seeds"], [42, 43, 44])
        self.assertEqual(seeds_for_mode(self.config, "candidate-temporal"), [42])
        self.assertEqual(seeds_for_mode(self.config, "shared-depth"), [42, 43, 44])
        self.assertEqual(self.config["p1"]["data"]["test_episodes"], 150)
        self.assertEqual(self.config["adaptation"]["learning_rate"], 5e-5)
        self.assertEqual(self.config["temporal"]["persistence"], "trajectory")

    def test_nominal_counts_include_final_mean_and_all_depths(self):
        self.assertEqual(nominal_budget(self.config, "frozen")["requested_gradient_updates"], 0)
        self.assertEqual(nominal_budget(self.config, "shared-initial")["requested_gradient_updates"], 1)
        self.assertEqual(nominal_budget(self.config, "shared-depth")["requested_gradient_updates"], 5)
        budget = nominal_budget(self.config, "candidate-temporal")
        self.assertEqual(budget["cem_sequence_evaluations"], 1537)
        self.assertEqual(budget["requested_gradient_updates"], 7685)
        self.assertEqual(budget["requested_support_presentations"], 245920)
        config = deepcopy(self.config)
        config["adaptation"]["updates"] = 2
        self.assertEqual(nominal_budget(config, "candidate-temporal")["requested_gradient_updates"], 15370)

    def test_invalid_modes_selector_and_persistence_rejected(self):
        for key, value in (("modes", ["shared-depth"]), ("modes", ["frozen", "frozen"]),
                           ("modes", ["frozen", "unknown"]), ("selector", "fr-refresh"),
                           ("persistence", "episode")):
            with self.subTest(key=key, value=value):
                config = deepcopy(self.config)
                config["temporal"][key] = value
                with self.assertRaises(ValueError):
                    validate_config(config)

    def test_synthetic_summary_and_corrupt_budgets(self):
        rows = fixture_records(self.config)
        result = summarize(rows, self.config)
        self.assertEqual(result["queries"], 2)
        self.assertEqual(result["episodes"], 2)
        self.assertAlmostEqual(result["modes"]["shared-depth"]["path_mse"]["mean"], .08)
        for broken in (rows[:-1], rows+[rows[0]]):
            with self.assertRaises(ValueError):
                summarize(broken, self.config)
        corrupt = deepcopy(rows)
        corrupt[-1]["adaptation"]["requested_updates"] -= 5
        with self.assertRaises(ValueError):
            summarize(corrupt, self.config)
        corrupt = deepcopy(rows)
        corrupt[-1]["cem_seed"] += 1
        with self.assertRaises(ValueError):
            summarize(corrupt, self.config)

    def test_temporal_dry_run_never_loads_runtime(self):
        output = io.StringIO()
        with patch("experiments.cli.importlib.import_module", side_effect=AssertionError("Unexpected numerical import")):
            with contextlib.redirect_stdout(output):
                dispatch(["p3", "evaluate", "--variant", "temporal", "--limit", "1", "--dry-run"])
        self.assertIn("lewm_tta.planning.temporal.cli", output.getvalue())
        self.assertNotIn("--lr", output.getvalue())

    def test_seed_schedule_skips_only_the_expensive_extra_seeds(self):
        schedule = seed_schedule(self.config)
        self.assertEqual(schedule[0], (42, ["frozen", "shared-initial", "shared-depth", "candidate-temporal"]))
        self.assertEqual(schedule[1], (43, ["frozen", "shared-initial", "shared-depth"]))
        self.assertEqual(schedule[2], (44, ["frozen", "shared-initial", "shared-depth"]))
        self.assertTrue(all(modes[0] == "frozen" for seed, modes in schedule))
        self.assertEqual(sum(len(modes) for seed, modes in schedule), 10)
        self.assertEqual(len(fixture_records(self.config)), 20)

    def test_mixed_seed_comparisons_use_only_matching_pairs(self):
        rows = fixture_records(self.config)
        for row in rows:
            if row["seed"] == 42:
                error = {"frozen": .1, "shared-initial": .08, "shared-depth": .06, "candidate-temporal": .05}[row["mode"]]
                cost = {"frozen": 10., "shared-initial": 9., "shared-depth": 8., "candidate-temporal": 7.}[row["mode"]]
            else:
                error = {"frozen": 1., "shared-initial": .9, "shared-depth": .8}[row["mode"]]
                cost = {"frozen": 100., "shared-initial": 90., "shared-depth": 80.}[row["mode"]]
            row["recorded_action_mse_by_block"] = [error]*self.config["planning"]["horizon"]
            row["selected_plan_frozen_model_goal_sse"] = row["selected_model_goal_sse"] = cost
        result = summarize(rows, self.config)
        candidate = result["modes"]["candidate-temporal"]
        shared = result["modes"]["shared-depth"]
        self.assertEqual(candidate["rows"], 2)
        self.assertEqual(shared["rows"], 6)
        self.assertEqual(set(shared["by_seed"]), {"42", "43", "44"})
        self.assertAlmostEqual(candidate["recorded_action_prediction"]["5"]["delta_vs_frozen"]["mean"], -.05)
        pair = result["paired_modes"]["candidate-temporal-minus-shared-depth"]
        self.assertEqual(pair["seeds"], [42])
        self.assertEqual(pair["rows"], 2)
        self.assertAlmostEqual(pair["recorded_action_prediction"]["5"]["endpoint_delta"]["mean"], -.01)
        self.assertAlmostEqual(pair["common_frozen_plan_score_delta"]["mean"], -1.)
        cheap = result["paired_modes"]["shared-depth-minus-shared-initial"]
        self.assertEqual(cheap["seeds"], [42, 43, 44])
        self.assertEqual(cheap["rows"], 6)
        formatted = format_summary({"summary": result, "config": self.config})
        self.assertIn("common seeds only", formatted)
        self.assertIn("42,43,44", formatted)

    def test_per_mode_matrix_rejects_unexpected_or_missing_seeds(self):
        rows = fixture_records(self.config)
        missing = [row for row in rows if not (row["mode"] == "shared-depth" and row["seed"] == 43)]
        with self.assertRaises(ValueError):
            summarize(missing, self.config)
        extra = deepcopy(next(row for row in rows if row["mode"] == "candidate-temporal"))
        extra["seed"] = 43
        with self.assertRaises(ValueError):
            summarize(rows+[extra], self.config)

    def test_disjoint_mode_seeds_are_reported_without_fake_pairs(self):
        config = deepcopy(self.config)
        config["evaluation"]["seeds_by_mode"].update({"shared-initial": [43], "shared-depth": [44]})
        validate_config(config)
        result = summarize(fixture_records(config), config)
        pair = result["paired_modes"]["candidate-temporal-minus-shared-depth"]
        self.assertEqual(pair["seeds"], [])
        self.assertEqual(pair["rows"], 0)
        self.assertFalse(pair["available"])
        self.assertNotIn("recorded_action_prediction", pair)

    def test_cli_seed_override_and_legacy_uniform_policy(self):
        config = deepcopy(self.config)
        override_seeds(config, [43])
        validate_config(config)
        self.assertNotIn("seeds_by_mode", config["evaluation"])
        for mode in config["temporal"]["modes"]:
            self.assertEqual(seeds_for_mode(config, mode), [43])
        result = summarize(fixture_records(config), config)
        self.assertTrue(all(pair["seeds"] == [43] for pair in result["paired_modes"].values()))

    def test_invalid_seed_policies_and_incomplete_frozen_coverage(self):
        for policy in (None, {"typo": [42]}, {"candidate-temporal": []}, {"candidate-temporal": [42, 42]},
                       {"candidate-temporal": [45]}, {"candidate-temporal": [True]},
                       {"candidate-temporal": [42.]}, {"candidate-temporal": [-1]},
                       {"candidate-temporal": "42"}, {"frozen": [42]}):
            with self.subTest(policy=policy):
                config = deepcopy(self.config)
                config["evaluation"]["seeds_by_mode"] = policy
                with self.assertRaises(ValueError):
                    validate_config(config)
        config = deepcopy(self.config)
        config["temporal"]["modes"].remove("candidate-temporal")
        validate_config(config)
        self.assertTrue(all(len(modes) == 3 for seed, modes in seed_schedule(config)))


if __name__ == "__main__":
    unittest.main()

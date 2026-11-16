"""Model-free checks for schedule LR precedence and optimization accounting."""

from copy import deepcopy
import math
from pathlib import Path
import tempfile
import unittest

from experiments.catalog import profile
from experiments.config import resolve
from lewm_tta.runtime import write_json
from lewm_tta.rollout.config import (
    effective_p1, load_config, manifest_spec, settings_for_schedule, updates_at, validate_config,
)
from lewm_tta.rollout.schedule_reporting import _validate_trace, summarize_schedules


def synthetic_records(config):
    """Complete scalar traces; no model, tensors, dataset or GPU is involved."""
    records = []
    for name in config["schedule_names"]:
        schedule = config["schedules"][name]
        rate = settings_for_schedule(effective_p1(config)["adaptation"], schedule)["learning_rate"]
        for method in config["adaptation"]["methods"]:
            rounds, positions = [], []
            optimizer_steps = 0
            for h in range(max(config["horizons"])):
                if schedule["persistence"] == "position":
                    optimizer_steps = 0
                count = 0 if method == "frozen" else updates_at(schedule, h)
                start = len(rounds)
                for local in range(count):
                    rounds.append({
                        "position": h, "update_at_position": local+1, "update": len(rounds)+1,
                        "optimizer_step": optimizer_steps+local+1, "learning_rate": rate,
                        "support": [{"transition_id": 99, "weight": 1., "channels": ["forward"]}],
                        "unique_support": 1, "presentations_so_far": len(rounds)+1,
                        "cumulative_unique_support": 1, "turnover_fraction": 0. if rounds else None,
                        "within_position_turnover": 0. if local else None,
                        "support_loss_before": 1., "support_loss_after": .9,
                        "gradient_norm_before_clipping": .1, "prediction_correction_mse": .01,
                        "parameter_displacement_mse": .001, "support_budget_shortfall": 0,
                    })
                positions.append({
                    "position": h, "learning_rate": rate, "requested_updates": count,
                    "actual_updates": count, "skip_reason": None,
                    "round_start": start, "round_end": len(rounds),
                    "optimizer_steps_before": optimizer_steps, "optimizer_steps_after": optimizer_steps+count,
                    "reset_before_position": h == 0 or schedule["persistence"] == "position",
                    "parameter_displacement_before": float(optimizer_steps),
                    "parameter_displacement_after": float(optimizer_steps+count),
                    "query_mse_before": 1., "query_mse_after": 1. if method == "frozen" else .8,
                })
                optimizer_steps += count
            records.append({
                "scope": "last-block", "schedule": name, "method": method,
                "seed": 42, "query": {"transition_id": 7, "source_episode_position": 12},
                "target_raw_rows": [20, 25, 30],
                "mse_by_block": [1. if method == "frozen" else .8]*max(config["horizons"]),
                "total_seconds": .01,
                "adaptation": {
                    "schedule": deepcopy(schedule), "learning_rate": rate,
                    "actual_learning_rate_sum": len(rounds)*rate,
                    "positions": positions, "rounds": rounds,
                    "actual_updates": len(rounds), "requested_updates": len(rounds),
                    "reset_count": 1 if schedule["persistence"] == "trajectory" else max(config["horizons"]),
                    "support_presentations": len(rounds), "extra_support_evaluation_presentations": len(rounds),
                    "cumulative_unique_support": int(bool(rounds)), "module_updates": {},
                    "retrieval_calls": {"forward": int(bool(rounds)), "reverse": 0, "state": 0},
                    "timing_seconds": {"total_with_checks": .01},
                },
            })
    return records


class ScheduleLearningRateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        resolved = resolve(profile("p2", "default")[1])
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)/"config.json"
            write_json(path, resolved)
            cls.config = load_config(path)

    def fixture(self):
        config = deepcopy(self.config)
        config["schedule_names"] = ["initial-u1", "initial-u5", "initial-u5-1e-5"]
        config["adaptation"]["methods"] = ["frozen", "forward"]
        config["horizons"] = [1, 3]
        config["p1"]["evaluation"]["seeds"] = [42]
        config["evaluation"]["bootstrap_samples"] = 0
        return config

    def test_fallback_override_and_settings_isolation(self):
        settings = {"learning_rate": 5e-5, "support_size": 32}
        original = deepcopy(settings)
        low = settings_for_schedule(settings, {"learning_rate": 1e-5})
        fallback = settings_for_schedule(settings, {"updates": 5})
        low["learning_rate"] = 2e-5
        self.assertEqual(settings, original)
        self.assertEqual(fallback, original)
        self.assertIsNot(fallback, settings)
        # A CLI/global override changes only schedules with no explicit LR.
        global_override = settings | {"learning_rate": 1e-4}
        self.assertEqual(settings_for_schedule(global_override, {})["learning_rate"], 1e-4)
        self.assertEqual(settings_for_schedule(global_override, {"learning_rate": 1e-5})["learning_rate"], 1e-5)

    def test_catalog_is_opt_in_and_names_do_not_set_the_rate(self):
        self.assertEqual(self.config["schedules"]["initial-u5-1e-5"]["learning_rate"], 1e-5)
        config = self.fixture()
        # Users may select the control in their active profile; selection and
        # the catalog remain independent, rather than freezing user defaults.
        config["schedule_names"] = ["initial-u1"]
        validate_config(config)
        self.assertIn("initial-u5-1e-5", config["schedules"])
        del config["schedules"]["initial-u5-1e-5"]["learning_rate"]
        validate_config(config)
        self.assertEqual(settings_for_schedule(effective_p1(config)["adaptation"],
            config["schedules"]["initial-u5-1e-5"])["learning_rate"], 5e-5)

    def test_lr_controls_preserve_the_prepared_cohort(self):
        config = self.fixture()
        before = manifest_spec(config)
        config["schedules"]["initial-u5"]["learning_rate"] = 3e-5
        config["adaptation"]["learning_rate"] = 1e-4
        validate_config(config)
        self.assertEqual(manifest_spec(config), before)

    def test_invalid_schedule_rates_and_unknown_fields_are_rejected(self):
        for rate in (0, -1e-5, math.nan, math.inf, -math.inf, True, "1e-5", None):
            with self.subTest(rate=rate):
                config = self.fixture()
                config["schedules"]["initial-u5"]["learning_rate"] = rate
                with self.assertRaises(ValueError):
                    validate_config(config)
        config = self.fixture()
        config["schedules"]["initial-u5"]["lr"] = 1e-5
        with self.assertRaises(ValueError):
            validate_config(config)

    def test_summary_distinguishes_updates_rates_and_summed_lr(self):
        config = self.fixture()
        summary = summarize_schedules(synthetic_records(config), config)
        comparisons = summary["cross_schedule"]
        equal_sum = comparisons["initial-u5-1e-5-minus-initial-u1"]["last-block"]["forward"]["3"]
        self.assertTrue(equal_sum["matched_learning_rate_sums"])
        self.assertFalse(equal_sum["matched_actual_updates"])
        self.assertFalse(equal_sum["matched_per_update_learning_rate"])
        equal_count = comparisons["initial-u5-1e-5-minus-initial-u5"]["last-block"]["forward"]["3"]
        self.assertTrue(equal_count["matched_actual_updates"])
        self.assertFalse(equal_count["matched_learning_rate_sums"])
        self.assertEqual(summary["schedule_learning_rates"]["initial-u5-1e-5"], 1e-5)
        self.assertAlmostEqual(summary["nominal_learning_rate_sums"]["initial-u5-1e-5"]["3"], 5e-5)
        frozen = summary["schedules"]["initial-u5-1e-5"]["scopes"]["last-block"]["methods"]["frozen"]
        self.assertEqual(frozen["iteration"]["mean_actual_learning_rate_sum"], 0.)

    def test_corrupt_or_missing_explicit_rate_metadata_is_rejected(self):
        config = self.fixture()
        row = next(row for row in synthetic_records(config)
                   if row["schedule"] == "initial-u5-1e-5" and row["method"] == "forward")
        _validate_trace(row, config)
        for location in (None, "positions", "rounds"):
            for missing in (False, True):
                with self.subTest(location=location, missing=missing):
                    corrupt = deepcopy(row)
                    entry = corrupt["adaptation"] if location is None else corrupt["adaptation"][location][0]
                    if missing:
                        del entry["learning_rate"]
                    else:
                        entry["learning_rate"] = 5e-5
                    with self.assertRaises(ValueError):
                        _validate_trace(corrupt, config)
        row["adaptation"]["actual_learning_rate_sum"] += 1e-5
        with self.assertRaises(ValueError):
            _validate_trace(row, config)

    def test_historical_fallback_traces_need_no_new_metadata(self):
        config = self.fixture()
        row = next(row for row in synthetic_records(config)
                   if row["schedule"] == "initial-u1" and row["method"] == "forward")
        del row["adaptation"]["actual_learning_rate_sum"]
        for entry in [row["adaptation"], *row["adaptation"]["positions"], *row["adaptation"]["rounds"]]:
            del entry["learning_rate"]
        _validate_trace(row, config)


if __name__ == "__main__":
    unittest.main()

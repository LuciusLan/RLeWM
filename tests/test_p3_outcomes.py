"""User-run success contracts; no checkpoints, GPU inference or native physics.

The coding agent compiles these fixtures only (docs/research/plan.md).
"""

from copy import deepcopy
import contextlib
import gzip
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from experiments.cli import dispatch
from experiments.catalog import stage_for_backend
from experiments.p3_budgets import budget_configs
from lewm_tta.collection.manifest import sealed
from lewm_tta.outcomes.inputs import load_sources, raw_plan
from lewm_tta.outcomes.metrics import PlanOutcome, goal_metrics
from lewm_tta.outcomes.pipeline import PROTOCOL, analyze, identity, verified_report
from lewm_tta.outcomes.replay import FreshReplay
from lewm_tta.outcomes.reporting import summarize
from lewm_tta.runtime import file_hash, object_hash, write_json


SETTINGS = {"bootstrap_samples": 100, "bootstrap_seed": 11, "deadlines": [.5, 2.],
            "success_targets": [.5, .8, 1.], "state_atol": 1e-5, "rgb_mae_atol": .1,
            "pusht_success": "native-goal", "coverage_threshold": .95}


def observation(state, fields=None):
    return SimpleNamespace(state=np.asarray(state, dtype=float), fields=fields or {},
                           rgb=np.zeros((2, 2, 3), dtype=np.uint8), truncated=False)


def record(query, episode, iterations, label, success, seed=41, initial=False, seconds=1.):
    initial_metrics = {"success": initial, "goal_distance": 0. if initial else 1., "planning_score": 0. if initial else -1.}
    final = {"success": success, "goal_distance": 0. if success else 2., "planning_score": 0. if success else -2.}
    outcome = PlanOutcome(initial_metrics)
    outcome.append(final)
    return {"kind": "planned", "case": {"case_id": query, "query": {"episode_id": episode}},
            "iterations": iterations, "condition": label, "seed": seed, "planning_seconds": seconds,
            "compute": {"cem_candidate_evaluations": iterations*8+1}, **outcome.record()}


class PhysicalMetrics(unittest.TestCase):
    def test_native_thresholds_and_unwrapped_reacher(self):
        goal = {"state64": np.zeros(2)}
        self.assertTrue(goal_metrics("tworooms", observation([15.999, 0]), goal)["success"])
        self.assertFalse(goal_metrics("tworooms", observation([16, 0]), goal)["success"])
        goal = {"state64": np.zeros(4), "qpos": np.zeros(2)}
        self.assertTrue(goal_metrics("reacher", observation([.049, -.049, 4, 4]), goal)["success"])
        self.assertFalse(goal_metrics("reacher", observation([.05, 0, 0, 0]), goal)["success"])
        self.assertFalse(goal_metrics("reacher", observation([2*np.pi, 0, 0, 0]), goal)["success"])

    def test_cube_position_success_and_orientation_sign(self):
        goal = {"state64": np.zeros(41), "privileged_block_0_pos": [0, 0, 0], "privileged_block_0_quat": [1, 0, 0, 0]}
        obs = observation(np.zeros(41), {"privileged_block_0_pos": [.04, 0, 0], "privileged_block_0_quat": [-1, 0, 0, 0]})
        metrics = goal_metrics("cube", obs, goal)
        self.assertTrue(metrics["success"])
        self.assertEqual(metrics["orientation_error_radians"], 0.)
        obs.fields["privileged_block_0_pos"] = [.04001, 0, 0]
        self.assertFalse(goal_metrics("cube", obs, goal)["success"])

    def test_pusht_native_and_coverage_are_distinct(self):
        goal = {"state64": np.zeros(5)}
        obs = observation([100, 100, 0, 0, 0])
        native = goal_metrics("pusht", obs, goal, coverage=1.)
        covered = goal_metrics("pusht", obs, goal, coverage=1., pusht_success="coverage")
        self.assertFalse(native["success"])
        self.assertTrue(covered["success"])
        self.assertEqual(covered["planning_score"], 1.)
        self.assertFalse(goal_metrics("pusht", observation(np.zeros(5)), goal, coverage=.949,
                                      pusht_success="coverage")["success"])
        with self.assertRaises(ValueError):
            goal_metrics("pusht", obs, goal)

    def test_endpoint_ever_and_initial_success_are_not_conflated(self):
        outcome = PlanOutcome({"success": False, "goal_distance": 4., "planning_score": -4.})
        outcome.append({"success": True, "goal_distance": 0., "planning_score": 0.})
        outcome.append({"success": False, "goal_distance": 2., "planning_score": -2.})
        value = outcome.record()
        self.assertFalse(value["final_success"])
        self.assertTrue(value["ever_success"])
        self.assertEqual(value["first_success_raw_step"], 1)
        self.assertEqual(value["planning_score_improvement"], 2.)
        self.assertEqual(PlanOutcome({"success": True, "goal_distance": 0., "planning_score": 0.}).record()["first_success_raw_step"], 0)

    def test_raw_action_layout_and_corruption(self):
        config = {"planning": {"horizon": 2, "raw_action_low": [-1, -1], "raw_action_high": [1, 1]},
                  "p1": {"data": {"frameskip": 2}}}
        norm = {"mean": [.1, .2], "std": [.5, .5]}
        row = {"selected_actions_normalized": [[0, 0, 0, 0], [1, 1, 1, 1]],
               "selected_raw_actions": [[[.1, .2], [.1, .2]], [[.6, .7], [.6, .7]]]}
        self.assertEqual(raw_plan(row, config, norm).shape, (4, 2))
        row["selected_raw_actions"][0][0][0] = .2
        with self.assertRaises(ValueError):
            raw_plan(row, config, norm)


class PhysicalSummaries(unittest.TestCase):
    def test_nonmonotonic_target_crossings_and_deadline_misses(self):
        rows = [record(f"q{q}", q//2, j, "baseline", q < wins, seconds=1.)
                for j, wins in ((2, 8), (4, 2), (6, 9)) for q in range(10)]
        summary = summarize(rows, SETTINGS)
        target = next(point for point in summary["targets"] if point["target_success_rate"] == .8)
        self.assertEqual(target["minimum_observed_iterations"], 2)
        self.assertIsNone(next(point for point in summary["targets"] if point["target_success_rate"] == 1.)["minimum_observed_iterations"])
        self.assertEqual(summary["curves"][0]["success_by_deadline"][0]["mean"], 0.)
        self.assertEqual(summary["curves"][0]["success_by_deadline"][1]["mean"], .8)

    def test_episode_clustering_and_seed_averaging(self):
        rows = [record(f"q{q}", q//4, 2, "baseline", q < 4) for q in range(8)]
        one = summarize(rows, SETTINGS)["curves"][0]["final_success_rate"]
        duplicated = rows+[dict(row, seed=42) for row in rows]
        two = summarize(duplicated, SETTINGS)["curves"][0]["final_success_rate"]
        self.assertEqual(one, two)
        self.assertEqual(one["episodes"], 2)
        self.assertEqual(one["queries"], 8)
        self.assertEqual(one["ci95"], [0., 1.])

    def test_nontrivial_denominator_and_empty_interval(self):
        rows = [record("q0", 0, 2, "baseline", True, initial=True), record("q1", 0, 2, "baseline", False)]
        point = summarize(rows, SETTINGS)["curves"][0]
        self.assertEqual(point["final_success_rate"]["mean"], .5)
        self.assertEqual(point["nontrivial_final_success_rate"]["mean"], 0.)
        self.assertEqual(point["nontrivial_final_success_rate"]["queries"], 1)
        self.assertIsNone(point["final_success_rate"]["ci95"])

    def test_paired_factorial_physical_interaction(self):
        rows = [record(f"q{q}", q, 6, label, label == "combined")
                for q in range(3) for label in ("baseline", "rca", "proposals", "combined")]
        interaction = next(item for item in summarize(rows, SETTINGS)["paired_contrasts"]
                           if item["contrast"].startswith("factorial"))
        self.assertEqual(interaction["final_success_rate_difference"]["mean"], 1.)
        self.assertEqual(interaction["paired_query_seed_rows"], 3)

    def test_saved_report_reanalysis_and_checksum(self):
        rows = [record("q0", 0, 2, "baseline", True), record("q1", 1, 2, "baseline", False)]
        with TemporaryDirectory() as directory:
            root = Path(directory)
            path = root/"records.jsonl.gz"
            with gzip.open(path, "wt") as stream:
                for row in rows:
                    stream.write(json.dumps(row)+"\n")
            report = {"stage": "P3-executed-plan-outcomes", "completed": True, "sources": [],
                      "protocol": PROTOCOL, "metric_contract": {"success": "test"}, "limited_run": True,
                      "config": {"settings": SETTINGS}, "records_file": str(path), "records_sha256": file_hash(path),
                      "record_count": len(rows), "record_identity_sha256": object_hash([identity(row) for row in rows]),
                      "summary": summarize(rows, SETTINGS)}
            filename = root/"report.json"
            write_json(filename, report)
            self.assertEqual(verified_report(filename)["summary"], report["summary"])
            with self.assertRaises(ValueError):
                analyze(filename, filename)
            path.write_bytes(b"corrupt")
            with self.assertRaises(ValueError):
                verified_report(filename)


class SourceAndReplayContracts(unittest.TestCase):
    def test_budget_schedule_prefix_and_old_mode_availability(self):
        base = {"planning": {"iterations": 6}, "interleave": {"checkpoints": [2, 4]},
                "adaptation": {"learning_rate": 1e-4}, "reuse": {"validation": ["old"]}}
        saved = deepcopy(base)
        jobs = budget_configs(base, [6, 2, 4], "/tmp/fixture-budget")
        self.assertEqual([job["interleave"]["checkpoints"] for job in jobs], [[], [2], [2, 4]])
        self.assertEqual(base, saved)
        self.assertTrue(all(job["adaptation"]["learning_rate"] == 1e-4 and job["reuse"] == {} for job in jobs))
        self.assertEqual(stage_for_backend("p3c"), "p3")
        self.assertEqual(stage_for_backend("p3_success"), "p3")

    def test_success_dry_run_never_constructs_simulator(self):
        with patch("lewm_tta.outcomes.replay.FreshReplay", side_effect=AssertionError("simulator")), contextlib.redirect_stdout(io.StringIO()) as output:
            dispatch(["p3", "success", "--reports", "absent.json", "--output-dir", "/tmp/not-created", "--dry-run"])
        self.assertIn("no input verification", output.getvalue())

    def test_reset_replays_complete_past_and_plan_has_no_future_controls(self):
        replay = FreshReplay.__new__(FreshReplay)
        replay.settings, replay.backend = SETTINGS, "tworooms"
        replay.config = {"p1": {"data": {"frameskip": 1, "history": 2}}}
        state = {"value": 0.}
        used = []
        def reset(seed):
            state["value"] = 0.
            used.clear()
            return observation([0., 0.])
        def step(action):
            state["value"] += float(action[0])
            used.append(float(action[0]))
            return observation([state["value"], 0.])
        native = SimpleNamespace(_reset_next_step=False, _set_goal_state=lambda goal: None)
        replay.environment = SimpleNamespace(env=SimpleNamespace(unwrapped=native), reset=reset, step=step)
        replay.description = {"test": True}
        replay.query = {"cases": [{"episode_id": 10, "reset_seed": 3}], "episode_files": [{"environment": replay.description}]}
        def rows(key, row):
            return {"state64": np.array([row, 0.]), "pixels": np.zeros((2, 2, 3), dtype=np.uint8),
                    "action": np.array([1., 0.]), "target_pos": np.zeros(2)}[key]
        replay.dataset = SimpleNamespace(rows=rows, validate_case=lambda *args: None)
        replay.fields = []
        case = {"case_id": "q", "query": {"episode_id": 10}, "episode_position": 0,
                "episode_start_raw_row": 0, "start_raw_row": 2, "current_raw_row": 3, "goal_raw_row": 5}
        planned = replay.execute(case, np.array([[-1., 0.], [-1., 0.]]))
        self.assertEqual(used, [1., 1., 1., -1., -1.])
        self.assertEqual(planned["raw_steps"], 2)
        control = replay.execute(case)
        self.assertEqual(used, [1., 1., 1., 1., 1.])
        self.assertEqual(control["recorded_future_reconstruction"]["state_max_abs_error"], 0.)
        replay.settings = dict(SETTINGS, state_atol=0.)
        with self.assertRaises(ValueError):
            replay.reconstruction(observation([2., 1.]), 2)

    def test_sources_pair_exact_budget_matrix_and_allow_temporal_receipt_fallback(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            query = {"fingerprint": "fresh", "episode_metadata": {"ids": [10], "offsets": [0], "lengths": [20]},
                     "episodes": {"validation": [0]}, "evaluation_windows": {"validation": {"raw_start_rows": [0]}}}
            config = {"experiment_dir": str(root/"j2"), "query_dataset": {"manifest": "fixture"},
                      "p1": {"data": {"history": 2, "frameskip": 1}}, "evaluation": {"query_limit": None},
                      "planning": {"horizon": 2, "iterations": 2, "raw_action_low": [-1, -1], "raw_action_high": [1, 1]}}
            cohort = {"raw_start_rows": [0]}
            reports = []
            for budget in (2, 4):
                selected = deepcopy(config)
                selected["planning"]["iterations"] = budget
                selected["experiment_dir"] = str(root/f"j{budget}")
                folder = Path(selected["experiment_dir"])
                folder.mkdir()
                manifest = sealed({"query_source_fingerprint": "fresh", "splits": {"validation": cohort}})
                write_json(folder/"manifest.json", manifest)
                row = {"mode": "frozen", "query": {"transition_id": "q", "episode_id": 10, "start_in_episode": 0},
                       "seed": 41, "goal_raw_row": 3, "checks": {"ok": True},
                       "selected_actions_normalized": [[0, 0], [0, 0]], "selected_raw_actions": [[0, 0], [0, 0]]}
                records = folder/"records.jsonl"
                records.write_text(json.dumps(row)+"\n")
                report = {"stage": "P3-temporal-cem", "completed": True, "split": "validation", "code": {"fixture": "ok"},
                          "config": selected, "p1_prepared_fingerprint": "prepared", "model": {}, "cohort": cohort,
                          "manifest_fingerprint": manifest["fingerprint"], "records_file": str(records),
                          "records_sha256": file_hash(records), "record_count": 1}
                filename = folder/"report.json"
                write_json(filename, report)
                reports.append(filename)
            prepared = {"fingerprint": "prepared", "normalization": {"mean": [0, 0], "std": [1, 1]}}
            with patch("lewm_tta.rollout.resources.source_metadata", return_value=({}, prepared)), patch("lewm_tta.collection.queries.query_source", return_value=query), patch("experiments.provenance.backend_identity", return_value={"fixture": "ok"}):
                bundle = load_sources(reports)
                self.assertEqual(bundle["budgets"], [2, 4])
                self.assertEqual(len(bundle["plans"]), 2)
                with self.assertRaises(ValueError):
                    load_sources([reports[0], reports[0]])
                altered = json.loads((root/"j4"/"records.jsonl").read_text())
                altered["seed"] = 42
                record_file = root/"j4"/"records.jsonl"
                record_file.write_text(json.dumps(altered)+"\n")
                report = json.loads(reports[1].read_text())
                report["records_sha256"] = file_hash(record_file)
                write_json(reports[1], report)
                with self.assertRaises(ValueError):
                    load_sources(reports)


if __name__ == "__main__":
    unittest.main()

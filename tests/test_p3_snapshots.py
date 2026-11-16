"""User-run snapshot parity/provenance contracts; coding agent syntax-checks only."""

from copy import deepcopy
import contextlib
import io
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from experiments.cli import dispatch
from lewm_tta.budget_receipts import STAGE, verify_budget_capture
from lewm_tta.interleave.anytime import stopped_config, stopped_retrieval_identity
from lewm_tta.interleave.cem import solve_interleaved_cem
from lewm_tta.interleave.continuations import ContinuationProviders
from lewm_tta.interleave.session import CheckpointSession
from lewm_tta.interleave.snapshots import SnapshotRecorder
from lewm_tta.runtime import file_hash, object_hash, write_json
import test_p3_interleave as interleave_contracts


class P3SnapshotTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        interleave_contracts.P3InterleaveTests.setUpClass()

    def setUp(self):
        fixture = interleave_contracts.P3InterleaveTests()
        fixture.setUp()
        for name in ("current", "metadata", "goal", "memory", "adapter", "engine", "coordinates"):
            setattr(self, name, getattr(fixture, name))
        self.config = deepcopy(interleave_contracts.P3InterleaveTests.config)
        self.config["planning"].update(population=8, elites=2, iterations=6)
        self.config["interleave"].update(checkpoints=[2, 4], references=2, injection_slots=7)

    def test_snapshots_equal_independent_stops_and_preserve_final_search(self):
        with tempfile.TemporaryDirectory() as directory:
            self.config["experiment_dir"] = directory
            providers = ContinuationProviders(SimpleNamespace(engine=self.engine, memory=self.memory), self.config, self.coordinates)
            for arm in ("baseline", "rca", "proposals", "combined"):
                session = CheckpointSession(self.engine, self.current, self.metadata, self.goal, self.coordinates,
                                            providers, self.config, arm, 42)
                recorder = SnapshotRecorder(session, time.perf_counter())
                final, _ = solve_interleaved_cem(session.score, self.config["planning"], self.coordinates, 42,
                    [2, 4], session, snapshot_iterations=[1, 2, 4, 6], on_snapshot=recorder)
                self.assertEqual(recorder.extra_terminal_probes, 3)
                self.assertEqual(set(recorder.snapshots), {1, 2, 4, 6})
                for j, snapshot in recorder.snapshots.items():
                    settings = self.config["planning"] | {"iterations": j}
                    local = stopped_config(self.config, j, directory)
                    independent = CheckpointSession(self.engine, self.current, self.metadata, self.goal,
                        self.coordinates, providers, local, arm, 42)
                    plan, checkpoints = solve_interleaved_cem(independent.score, settings, self.coordinates, 42,
                                                              local["interleave"]["checkpoints"], independent)
                    torch.testing.assert_close(snapshot.plan.actions, plan.actions, rtol=0, atol=0)
                    self.assertAlmostEqual(snapshot.plan.cost, plan.cost, places=7)
                    self.assertEqual(snapshot.plan.trace, plan.trace)
                    self.assertEqual([cp["checkpoint"] for cp in snapshot.checkpoints], [k for k in [2, 4] if k < j])
                    self.assertEqual(snapshot.plan.candidate_evaluations, 8*j+1)
                    expected_version = sum(k < j for k in [2, 4]) if arm in ("rca", "combined") else 0
                    self.assertEqual(snapshot.model_version, expected_version)
                    self.assertEqual(snapshot.counters["actual_updates"], expected_version)
                    recorder.restore(snapshot)
                    reproduced = session.score(snapshot.plan.actions[None])
                    self.assertAlmostEqual(float(reproduced[0]), snapshot.plan.cost, places=7)
                torch.testing.assert_close(final.actions, recorder.snapshots[6].plan.actions, rtol=0, atol=0)

    def test_terminal_probe_cannot_advance_rng_or_contaminate_later_phase_times(self):
        clock = SimpleNamespace(now=10.)
        parameter = torch.nn.Parameter(torch.tensor(.1))
        def checked():
            clock.now += .05
        def adapted():
            clock.now += .3  # Existing standalone adaptation audit is included.
            return {"model_version": 0}
        adapter = SimpleNamespace(device=torch.device("cpu"), parameters=[parameter], check_fixed_state=checked)
        session = SimpleNamespace(adapter=adapter, config={"planning": {"population": 8}},
            timings={"cem_scoring": 1.}, counters={}, adaptation_record=adapted)
        def score(actions):
            torch.rand(5)
            clock.now += .2
            session.timings["cem_scoring"] += .2
            return torch.tensor([1.])
        session.score = score
        recorder = SnapshotRecorder(session, 0.)
        rng = torch.random.get_rng_state().clone()
        with patch("lewm_tta.interleave.snapshots.time.perf_counter", side_effect=lambda: clock.now):
            recorder(2, torch.zeros(5, 2), [], [], 0, None, "population")
            torch.testing.assert_close(torch.random.get_rng_state(), rng, rtol=0, atol=0)
            self.assertAlmostEqual(recorder.snapshots[2].primary_seconds, 10.55)
            self.assertEqual(session.timings["cem_scoring"], 1.)
            self.assertAlmostEqual(recorder.snapshots[2].timing_seconds["cem_scoring"], 1.2)
            self.assertAlmostEqual(recorder.overhead_seconds, .55)
            clock.now += 1.
            entered = clock.now
            recorder(4, torch.zeros(5, 2), [], [], 0, 1., "population")
            self.assertAlmostEqual(recorder.snapshots[4].primary_seconds, entered-.55+.35)
            self.assertEqual(recorder.extra_terminal_probes, 1)

    def test_snapshot_callback_validation_and_no_snapshot_api_unchanged(self):
        score = lambda actions: actions.square().sum((1, 2))
        for values in ([2, 2], [4, 2], [0], [7]):
            with self.assertRaises(ValueError):
                solve_interleaved_cem(score, self.config["planning"], self.coordinates, 42,
                                      snapshot_iterations=values, on_snapshot=lambda *args: None)
        with self.assertRaises(ValueError):
            solve_interleaved_cem(score, self.config["planning"], self.coordinates, 42, snapshot_iterations=[2])
        plan, checkpoints = solve_interleaved_cem(score, self.config["planning"], self.coordinates, 42)
        self.assertEqual(plan.candidate_evaluations, 49)
        self.assertEqual(checkpoints, [])

    def test_shared_receipt_pins_records_config_and_implementation(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"master.json"
            config = {"planning": {"iterations": 2}}
            report = {"config": config, "code": {"fixture": "code"}, "records_file": "records.gz",
                      "records_sha256": "hash", "record_count": 3, "manifest_fingerprint": "cohort"}
            master = {"completed": True, "stage": STAGE, "iterations": [2, 4], "code": report["code"],
                "reports": [{"iterations": 2, "records_file": "records.gz", "records_sha256": "hash", "record_count": 3,
                             "manifest_fingerprint": "cohort", "config_fingerprint": object_hash(config)}]}
            write_json(path, master)
            report["budget_capture"] = {"mode": "single-pass", "master_report": str(path), "master_report_sha256": file_hash(path),
                                        "max_iterations": 4, "requested_iterations": [2, 4]}
            self.assertEqual(verify_budget_capture(report), master)
            changed = deepcopy(report)
            changed["records_sha256"] = "changed"
            with self.assertRaises(ValueError):
                verify_budget_capture(changed)
            changed = deepcopy(report)
            changed["config"]["planning"]["population"] = 99
            with self.assertRaises(ValueError):
                verify_budget_capture(changed)
            path.write_text("{}")
            with self.assertRaises(ValueError):
                verify_budget_capture(report)

    def test_frontend_emits_one_search_or_explicit_independent_searches(self):
        for mode, search_count in (("single-pass", 1), ("independent", 4)):
            with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()) as output:
                with patch("experiments.p3_budgets.completed_report", return_value=False), patch("experiments.p3_budgets.subprocess.run", side_effect=AssertionError("execution")):
                    dispatch(["p3", "budget", "--variant", "interleave", "--iterations", "2", "4", "8", "16",
                              "--execution-mode", mode, "--output-root", directory, "--planning-only", "--dry-run"])
                commands = output.getvalue().splitlines()
                self.assertEqual(sum(" p3 evaluate-snapshots " in line for line in commands), 1 if mode == "single-pass" else 0)
                self.assertEqual(sum(" p3 evaluate " in line for line in commands), search_count if mode == "independent" else 0)
                self.assertEqual(list(Path(directory).iterdir()), [])

    def test_endpoint_calibration_remains_lock_compatible_for_each_budget(self):
        identity = {"suffix_horizons": {"4": {"goal": {"calibration_file": "/old/retrieval/goal-h4.json",
                    "calibration": {"fingerprint": "memory-only", "scale": 1.}}}}}
        with tempfile.TemporaryDirectory() as directory:
            stopped = stopped_retrieval_identity(identity, directory)
            self.assertEqual(stopped["suffix_horizons"]["4"]["goal"]["calibration_file"], str(Path(directory)/"retrieval/goal-h4.json"))
            self.assertTrue((Path(directory)/"retrieval/goal-h4.json").is_file())
            self.assertEqual(identity["suffix_horizons"]["4"]["goal"]["calibration_file"], "/old/retrieval/goal-h4.json")
            self.assertEqual(stopped_retrieval_identity(identity, directory), stopped)


if __name__ == "__main__":
    unittest.main()

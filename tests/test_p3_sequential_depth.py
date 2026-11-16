"""User-run synthetic contracts; the coding agent performs syntax checks only."""

from copy import deepcopy
import contextlib
import io
from pathlib import Path
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from experiments.cli import dispatch
from experiments.provenance import archive_entry
from lewm_tta.interleave.anytime import stopped_config
from lewm_tta.interleave.cem import solve_interleaved_cem
from lewm_tta.interleave.config import adaptation_schedule, checkpoint_updates, validate_config
from lewm_tta.interleave.reporting import summarize
from lewm_tta.interleave.session import CheckpointSession
from lewm_tta.interleave.snapshots import SnapshotRecorder
from lewm_tta.rollout.record_io import check_record_roundtrip
from lewm_tta.runtime import read_json
import test_p3_interleave as fixtures


class SequentialDepthTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fixtures.P3InterleaveTests.setUpClass()

    def setUp(self):
        fixture = fixtures.P3InterleaveTests()
        fixture.setUp()
        for key in ("current", "metadata", "goal", "memory", "adapter", "engine", "coordinates"):
            setattr(self, key, getattr(fixture, key))
        self.config = deepcopy(fixtures.P3InterleaveTests.config)
        self.config["planning"].update(population=8, elites=2, iterations=6)
        self.config["interleave"].update(arms=["baseline", "rca"], references=2,
            checkpoints=[2, 4], injection_slots=0, adaptation_schedule="sequential-depth")
        self.config["p1"]["evaluation"]["seeds"] = [42]
        self.config["evaluation"]["bootstrap_samples"] = 0

    def session(self, config=None, arm="rca", updates=1):
        engine = SimpleNamespace(**{**vars(self.engine), "settings": self.engine.settings | {"updates": updates}})
        return CheckpointSession(engine, self.current, self.metadata, self.goal, self.coordinates,
                                 None, config or self.config, arm, 42)

    def solve(self, config=None, arm="rca", updates=1):
        selected = config or self.config
        session = self.session(selected, arm, updates)
        plan, checkpoints = solve_interleaved_cem(session.score, selected["planning"], self.coordinates,
                                                 42, selected["interleave"]["checkpoints"], session)
        return session, plan, checkpoints

    def test_legacy_absence_and_explicit_pooled_schedule_match(self):
        config = deepcopy(self.config)
        config["interleave"].pop("adaptation_schedule")
        self.assertEqual(adaptation_schedule(config), "pooled-depth")
        first, plan, _ = self.solve(config)
        fingerprints = [r["support_fingerprint"] for r in first.rounds]
        config["interleave"]["adaptation_schedule"] = "pooled-depth"
        second, again, _ = self.solve(config)
        torch.testing.assert_close(plan.actions, again.actions, rtol=0, atol=0)
        self.assertEqual(plan.cost, again.cost)
        self.assertEqual(fingerprints, [r["support_fingerprint"] for r in second.rounds])
        self.assertEqual(checkpoint_updates(config), 1)
        config["interleave"]["adaptation_schedule"] = "unknown"
        with self.assertRaisesRegex(ValueError, "schedule"):
            validate_config(config)

    def test_each_depth_rebuilds_prefix_with_latest_weights_and_carries_optimizer(self):
        candidates = torch.stack([torch.full((5, 2), value) for value in (.2, .3, .4, .5)])
        for method, updates in (("fr-fixed", 1), ("fr-fixed", 2), ("fr-refresh", 2)):
            with self.subTest(method=method, updates=updates):
                self.config["interleave"]["support_method"] = method
                session = self.session(updates=updates)
                original, seen = session.adapt_pairs, []

                def inspect(checkpoint, contexts, actions, record, *, depth=None):
                    before = float(self.adapter.model.weight.detach())
                    # Constant per-trajectory actions make a complete prefix
                    # under the latest scalar predictor equal h*w*a.
                    torch.testing.assert_close(contexts.latents[:, -1], depth*before*actions)
                    self.assertFalse(contexts.latents.requires_grad)
                    self.assertFalse(actions.requires_grad)
                    seen.append((depth, session.version, before))
                    return original(checkpoint, contexts, actions, record, depth=depth)

                with patch.object(session, "adapt_pairs", side_effect=inspect):
                    result = session(2, candidates, torch.zeros(4), torch.tensor([0, 1]))
                self.assertEqual([(h, v) for h, v, _ in seen], [(h, h*updates) for h in range(5)])
                self.assertTrue(all(weight > seen[0][2] for _, _, weight in seen[1:]))
                self.assertEqual(result.record["requested_updates"], 5*updates)
                self.assertEqual(session.version, 5*updates)
                self.assertEqual(session.counters["support_presentations"], 160*updates)
                self.assertEqual(session.counters["reference_transition_predictions"], 2*sum(range(5)))
                self.assertEqual(session.counters["forward_retrievals"], 10)
                self.assertEqual(session.counters["reverse_retrievals"], 10*(updates if method == "fr-refresh" else 1))
                self.assertEqual(int(session.optimizer.state[self.adapter.model.weight]["step"]), 5*updates)
                if updates == 2 and method == "fr-fixed":
                    for depth in range(5):
                        rows = [r for r in session.rounds if r["depth"] == depth]
                        self.assertEqual(rows[0]["support_fingerprint"], rows[1]["support_fingerprint"])
                for row in session.rounds:
                    self.assertEqual(len(row["support"]), 32)
                    self.assertTrue(all(item["source_episode_position"] >= 0 for item in row["support"]))
                # Scoring uses one final model at all five steps and performs
                # no training or snapshot switching within candidate rollout.
                before = session.version
                endpoint = self.adapter.model.weight.detach()*candidates.sum(1)
                expected = (endpoint-self.goal).square().sum(-1)
                torch.testing.assert_close(session.score(candidates), expected)
                self.assertEqual(session.version, before)

    def test_zero_updates_are_frozen_and_depth_subsets_have_explicit_budgets(self):
        _, frozen, _ = self.solve(arm="baseline")
        session, zero, checkpoints = self.solve(updates=0)
        torch.testing.assert_close(zero.actions, frozen.actions, rtol=0, atol=0)
        self.assertEqual(zero.cost, frozen.cost)
        self.assertEqual(session.version, 0)
        self.assertTrue(all(cp["skip_reason"] == "zero_updates" for cp in checkpoints))
        config = deepcopy(self.config)
        config["interleave"]["adaptation_depths"] = [0, 2, 4]
        self.assertEqual(checkpoint_updates(config), 3)
        session, _, checkpoints = self.solve(config)
        self.assertEqual(session.version, 6)
        self.assertEqual([stage["depth"] for stage in checkpoints[0]["depths"]], [0, 2, 4])

    def test_snapshots_match_independent_stops_before_equal_checkpoints(self):
        session = self.session()
        recorder = SnapshotRecorder(session, time.perf_counter())
        solve_interleaved_cem(session.score, self.config["planning"], self.coordinates, 42,
            [2, 4], session, snapshot_iterations=[2, 4, 6], on_snapshot=recorder)
        for j, snapshot in recorder.snapshots.items():
            config = stopped_config(self.config, j, self.config["experiment_dir"])
            _, plan, _ = self.solve(config)
            torch.testing.assert_close(snapshot.plan.actions, plan.actions, rtol=0, atol=0)
            self.assertAlmostEqual(snapshot.plan.cost, plan.cost, places=7)
            self.assertEqual(snapshot.model_version, 5*sum(k < j for k in (2, 4)))
            recorder.restore(snapshot)
            self.assertAlmostEqual(float(session.score(plan.actions[None])[0]), plan.cost, places=7)

    def test_reports_roundtrip_and_reject_stale_depth_receipts(self):
        rows = []
        for arm in ("baseline", "rca"):
            session, plan, checkpoints = self.solve(arm=arm)
            rows.append({"arm": arm, "seed": 42, "query": self.metadata, "goal_raw_row": 7,
                "cem_seed": 42, "initial_injection_fingerprint": None,
                "cem_first_population_fingerprint": plan.first_population_fingerprint,
                "cem_trace": plan.trace, "checkpoints": checkpoints, "adaptation": session.adaptation_record(),
                "selected_actions_normalized": plan.actions.tolist(), "counters": dict(session.counters),
                "cem_candidate_evaluations": plan.candidate_evaluations,
                "cem_transition_predictions": 5*plan.candidate_evaluations,
                "recorded_action_mse_by_block": [.1]*5, "selected_plan_frozen_model_goal_sse": plan.cost,
                "selected_model_goal_sse": plan.cost, "primary_seconds_with_checks": 1., "checks": {"synthetic": True}})
        check_record_roundtrip(rows, self.config, summarize)
        broken = deepcopy(rows)
        broken[-1]["checkpoints"][0]["depths"][1]["reference_prefix_model_version"] = 0
        with self.assertRaisesRegex(ValueError, "stale"):
            summarize(broken, self.config)

    def test_core_cli_and_historical_source_archive(self):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            dispatch(["p3", "core", "--dataset", "pusht", "--output-root", "/tmp/p3-sequential-contract",
                      "--iterations", "30", "--checkpoints", "15", "--updates", "1",
                      "--adaptation-schedule", "sequential-depth", "--dry-run"])
        self.assertIn("sequential-depth, 5 requested updates", output.getvalue())
        with self.assertRaises(ValueError):
            dispatch(["p3", "core", "--phase", "test", "--locked-config", "/tmp/lock.json",
                      "--adaptation-schedule", "sequential-depth", "--dry-run"])
        root = Path(__file__).resolve().parents[2]
        archive = root/"artifacts/code-archives/e705c3727408ae6ec401"
        recorded = read_json(archive/"manifest.json")["entrypoints"]["p3_core"]["code"]
        self.assertEqual(archive_entry("p3_core", recorded, root=root),
                         (archive/"codes/experiment.py", ["legacy", "p3_core"]))


if __name__ == "__main__":
    unittest.main()

"""User-run green benchmark contracts; no real simulator or learned inference.

PYTHONPATH=codes python -m unittest discover -s codes/tests -p test_green_target.py
"""

from copy import deepcopy
from itertools import product
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from experiments.catalog import ROOT
from experiments.config import resolve
from lewm_tta.runtime import write_json
from lewm_tta.simulation import green
from lewm_tta.simulation.dataset import ResetCases
from lewm_tta.simulation.environments.base import Observation
from lewm_tta.simulation.environments.fresh_pusht import FreshPushT
from lewm_tta.simulation.environments.green_pusht import GreenTargetPushT, align_to_marker, polygon
from lewm_tta.simulation.execution import mpc
from lewm_tta.simulation.history import bootstrap


class ToyEnvironment:
    def reset(self, case, seed, *, history_start):
        self.position = 0
        self.goal_image = np.full((2, 2, 3), 99, dtype=np.uint8)
        self.goal_contract = {"marker": "fixed"}
        return self.observe()

    def observe(self):
        return Observation(np.full((2, 2, 3), self.position, dtype=np.uint8), np.array([self.position]),
                           {"coverage": min(max((self.position-10)/7, 0), 1),
                            "success": self.position >= 17, "out_of_frame": False})

    def step(self, action, *, recorded=False):
        self.position += 1
        return self.observe()

    def reconstruction(self, observation, row):
        return {}

    def description(self):
        return {"metric_directions": {"coverage": 1, "success": 1, "out_of_frame": -1}}


class ToyHistory:
    def __init__(self, adapter, coordinates, images, blocks):
        self.last = int(images[-1][0, 0, 0])

    def encode(self, image):
        return torch.tensor([[float(image[0, 0, 0])]])

    def append(self, image, block):
        self.last = int(image[0, 0, 0])

    def context(self):
        return self.last


class GreenContracts(unittest.TestCase):
    def config(self):
        config = resolve(ROOT/"codes/configs/sim/green.json")
        config["simulation"].update(episodes=2, test_episodes=3)
        return config

    def manifest(self, config):
        source = {"fingerprint": "memory", "data_config": {"history": 3, "frameskip": 5}}
        with patch.object(green, "source_metadata", return_value=(source, {"fingerprint": "prepared"})):
            return green.manifest_value(config)

    def test_reset_suites_are_disjoint_and_test_selection_does_not_rereserve(self):
        config = self.config()
        first = self.manifest(config)
        config["simulation"]["split"] = "test"
        self.assertEqual(first, self.manifest(config))
        validation = {case["episode_reset_seed"] for case in first["splits"]["validation"]}
        test = {case["episode_reset_seed"] for case in first["splits"]["test"]}
        self.assertFalse(validation & test)
        for case in sum(first["splits"].values(), []):
            ResetCases().validate_case(case, 5, 3)
            self.assertIsNone(case["goal_raw_row"])
        with self.assertRaises(ValueError):
            ResetCases().rows("action", 10, 35)

    def test_render_transform_matches_marker_despite_nonzero_center_of_gravity(self):
        import pymunk
        body = pymunk.Body(1, 10)
        body.center_of_gravity = (4., 7.)
        body.position, body.angle = (75, 120), -.8
        shape = pymunk.Poly(body, [(-10, -5), (10, -5), (10, 5), (-10, 5)])
        marker = pymunk.Body(1, 10)
        marker.position, marker.angle = (256, 256), .7
        align_to_marker(body, marker)
        self.assertAlmostEqual(polygon(body, [shape]).symmetric_difference(polygon(marker, [shape])).area, 0., places=9)

    def test_marker_success_ignores_pusher_pose_and_uses_overlap(self):
        import pymunk
        block = pymunk.Body(1, 10)
        block.position = (256, 256)
        shape = pymunk.Poly(block, [(-10, -10), (10, -10), (10, 10), (-10, 10)])
        agent = pymunk.Body(body_type=pymunk.Body.KINEMATIC)
        agent.position = (450, 450)
        circle = pymunk.Circle(agent, 15)
        adapter = GreenTargetPushT.__new__(GreenTargetPushT)
        adapter.env = SimpleNamespace(unwrapped=SimpleNamespace(block=block, agent=agent, window_size=512))
        adapter.target_polygon = polygon(block, [shape])
        adapter.goal = np.array([64, 64, 256, 256, 0, 0, 0.])
        adapter.shift, adapter.settings = {}, {"success_coverage": .95}
        state = np.array([450, 450, 256, 256, 0, 0, 0.])
        self.assertTrue(adapter._metrics(state)["success"])
        self.assertEqual(adapter._metrics(state)["coverage"], 1.)
        block.position = (300, 256)
        state[2] = 300
        self.assertFalse(adapter._metrics(state)["success"])
        # Keep shape references alive until the geometry assertions complete.
        self.assertEqual(circle.body, agent)

    def test_native_random_state_termination_does_not_stop_marker_task(self):
        adapter = GreenTargetPushT.__new__(GreenTargetPushT)
        observed = Observation(np.zeros((2, 2, 3), dtype=np.uint8), np.zeros(7),
                               {"success": False}, terminated=True)
        with patch.object(FreshPushT, "step", return_value=observed):
            self.assertFalse(adapter.step([0, 0]).terminated)
        observed.metrics["success"], observed.terminated = True, False
        with patch.object(FreshPushT, "step", return_value=observed):
            self.assertTrue(adapter.step([0, 0]).terminated)

    def test_zero_history_and_fixed_rendered_goal_never_read_a_future_dataset(self):
        config = self.config()
        case = self.manifest(config)["splits"]["validation"][0]
        images, blocks, current = bootstrap(ToyEnvironment(), ResetCases(), case, 42, 3, 5)
        self.assertEqual([int(image[0, 0, 0]) for image in images], [0, 5, 10])
        seen = []
        class Controller:
            resources = SimpleNamespace(adapter=SimpleNamespace(device=torch.device("cpu")))
            coordinates = None
            def plan(self, context, goal, metadata, method, source, seed, warm):
                seen.append((context, goal.clone()))
                return torch.zeros((2, 10)), np.zeros((2, 5, 2)), {"planning_seconds": .001}
        with patch("lewm_tta.simulation.execution.ObservedHistory", ToyHistory):
            result = mpc(ToyEnvironment(), ResetCases(), case, 42, "frozen", "gaussian", Controller(),
                         {"history": 3, "frameskip": 5}, config, [])
        self.assertEqual([context for context, goal in seen], [10, 15])
        self.assertTrue(all(goal.tolist() == [[99.]] for context, goal in seen))
        self.assertEqual(result["first_success_raw_step"], 7)
        self.assertEqual(result["raw_steps"], 7)

    def test_test_evaluation_requires_lock_before_loading_resources(self):
        from lewm_tta.simulation.pipeline import evaluate
        config = self.config()
        config["simulation"]["split"] = "test"
        with patch("lewm_tta.simulation.pipeline.source_dataset") as source:
            with self.assertRaisesRegex(ValueError, "locked-config"):
                evaluate(config, "mpc", "cpu", 1)
            source.assert_not_called()

    def test_lock_binds_the_complete_six_cell_validation_and_report_bytes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = self.config()
            config["experiment_dir"] = str(root)
            manifest = self.manifest(config)
            cases = manifest["splits"]["validation"]
            rows = [{"case_id": case["case_id"], "query": case["query"], "seed": seed,
                     "method": method, "proposal_source": proposal}
                    for case, seed, method, proposal in product(cases, config["simulation"]["seeds"],
                        config["adaptation"]["methods"], config["proposals"]["sources"])]
            report = {"config": config, "code": {"sim.py": "hash"}, "stage": "simulator-evaluation", "task": "green-target",
                      "mode": "mpc", "split": "validation", "completed": True, "limited_run": False,
                      "budget_exhausted": False, "manifest_fingerprint": manifest["fingerprint"],
                      "case_ids": [case["case_id"] for case in cases], "summary": {"cases": len(cases)},
                      "record_count": len(rows)}
            path = root/"report.json"
            write_json(path, report)
            with patch("experiments.provenance.backend_identity", return_value=report["code"]), patch(
                    "lewm_tta.simulation.cohorts.load_manifest", return_value=manifest), patch(
                    "lewm_tta.rollout.record_io.read_saved_records", return_value=rows) as read, patch(
                    "lewm_tta.simulation.reporting.summarize", return_value=report["summary"]):
                green.lock_validation(path)
                locked, info = green.locked_test_config(root/"green-validation-lock.json")
                self.assertEqual(locked["simulation"]["split"], "test")
                self.assertEqual(config["simulation"]["split"], "validation")
                read.return_value = rows[:-1]
                with self.assertRaisesRegex(ValueError, "matrix"):
                    green.lock_validation(path, root/"invalid.json")
                path.write_text(path.read_text()+"\n")
                with self.assertRaises(ValueError):
                    green.locked_test_config(root/"green-validation-lock.json")


if __name__ == "__main__":
    unittest.main()

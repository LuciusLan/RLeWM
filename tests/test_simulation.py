"""Synthetic contracts without a checkpoint, cuVS index or real simulator.

PYTHONPATH=codes python -m unittest discover -s codes/tests -p test_simulation.py
"""

from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
import numpy as np
import torch

from lewm_tta.simulation.environments.base import Observation
from lewm_tta.simulation.environments.pusht import PushTEnvironment, goal_metrics
from lewm_tta.simulation.execution import mpc, replay
from lewm_tta.simulation.history import bootstrap, preprocess_frame
from lewm_tta.simulation.reporting import summarize, format_summary


class PastOnlyDataset:
    def __init__(self):
        self.reads = []

    def rows(self, key, start, stop):
        if key != "action" or start != 0 or stop != 10:
            raise AssertionError("MPC accessed recorded future actions")
        self.reads.append((key, start, stop))
        return np.zeros((10, 2), dtype=np.float32)


class Environment:
    def reset(self, case, seed, *, history_start):
        self.position = 0 if history_start else 10
        self.action_origins = []
        return self.observe()

    def observe(self):
        return Observation(np.full((2, 2, 3), self.position, dtype=np.uint8),
                           np.array([self.position], dtype=float),
                           {"distance": float(abs(17-self.position)), "success": self.position >= 17},
                           terminated=self.position >= 17)

    def step(self, action, *, recorded=False):
        self.action_origins.append(recorded)
        self.position += 1
        return self.observe()

    def reconstruction(self, observation, row):
        return {"diagnostic": 0.}

    def description(self):
        return {"metric_directions": {"distance": -1, "success": 1}}


class History:
    def __init__(self, adapter, coordinates, images, blocks):
        self.last = int(images[-1][0, 0, 0])

    def append(self, image, block):
        if len(block) != 5:
            raise AssertionError("Partial action block entered history")
        self.last = int(image[0, 0, 0])

    def context(self):
        return self.last


class Controller:
    def __init__(self):
        self.resources = SimpleNamespace(adapter=SimpleNamespace(device=torch.device("cpu")),
            memory=SimpleNamespace(store=SimpleNamespace(latents=torch.arange(50.)[:, None])))
        self.coordinates, self.seen = None, []

    def plan(self, context, goal, metadata, method, source, seed, warm):
        self.seen.append((context, goal.clone(), dict(metadata), None if warm is None else warm.clone()))
        return torch.zeros((2, 10)), np.zeros((2, 5, 2)), {"planning_seconds": 0.001}


CASE = {"case_id": "case-1", "start_raw_row": 0, "current_raw_row": 10, "goal_raw_row": 35,
        "query": {"transition_id": "case-1", "source_episode_position": 1, "episode_id": 1}}


class SimulationContracts(unittest.TestCase):
    def test_recorded_out_of_box_command_reaches_native_controller_unchanged(self):
        adapter = PushTEnvironment.__new__(PushTEnvironment)
        native = Mock()
        native.step.return_value = ({"state": np.zeros(7)}, 0., False, False, {})
        adapter.env = native
        adapter._observation = Mock(return_value="observed")
        action = np.array([1.082123041152954, -0.3141365051269531], dtype=np.float32)
        self.assertEqual(adapter.step(action, recorded=True), "observed")
        np.testing.assert_array_equal(native.step.call_args.args[0], action)

    def test_planner_bounds_still_enforced_before_native_execution(self):
        adapter = PushTEnvironment.__new__(PushTEnvironment)
        adapter.env = Mock()
        with self.assertRaises(ValueError):
            adapter.step([1.082123, -.3141365])
        adapter.env.step.assert_not_called()

    def test_recorded_commands_must_still_have_finite_valid_shape(self):
        adapter = PushTEnvironment.__new__(PushTEnvironment)
        adapter.env = Mock()
        for action in ([float("nan"), 0], [0, float("inf")], [[0, 0]], [0]):
            with self.subTest(action=action), self.assertRaises(ValueError):
                adapter.step(action, recorded=True)
        adapter.env.step.assert_not_called()

    def test_coverage_respects_nonzero_body_center_of_gravity(self):
        import pymunk
        body = pymunk.Body(1, 10)
        body.center_of_gravity = (.4, .8)
        body.position = (5, 6)
        body.angle = .5
        shape = pymunk.Poly(body, [(-1, -1), (1, -1), (1, 1), (-1, 1)])
        adapter = PushTEnvironment.__new__(PushTEnvironment)
        adapter.env = SimpleNamespace(unwrapped=SimpleNamespace(block=body))
        adapter.goal = np.array([0, 0, body.position.x, body.position.y, body.angle, 0, 0.])
        self.assertAlmostEqual(adapter._coverage(), 1., places=12)
        adapter.goal[2] += 20
        self.assertEqual(adapter._coverage(), 0.)

    def test_rgb_spatial_layout_matches_native_hdf5_loader(self):
        image = np.arange(2*4*3, dtype=np.uint8).reshape(2, 4, 3)
        result = preprocess_frame(image, lambda sample: sample)
        self.assertEqual(tuple(result.shape), (3, 2, 4))
        np.testing.assert_array_equal(result.numpy(), image.transpose(2, 0, 1))

    def test_angle_wrap_and_native_success_both_constraints(self):
        goal = np.array([0, 0, 0, 0, 0.01, 0, 0.])
        state = goal.copy()
        state[4] = 2*np.pi-0.01
        self.assertTrue(goal_metrics(state, goal)["success"])
        state[0] = 21
        self.assertFalse(goal_metrics(state, goal)["success"])

    def test_bootstrap_reads_only_past_actions_at_block_boundaries(self):
        dataset = PastOnlyDataset()
        environment = Environment()
        images, blocks, last = bootstrap(environment, dataset, CASE, 42, 3, 5)
        self.assertEqual([int(image[0, 0, 0]) for image in images], [0, 5, 10])
        self.assertEqual([len(block) for block in blocks], [5, 5])
        self.assertEqual(last.state.tolist(), [10])
        self.assertEqual(dataset.reads, [("action", 0, 10)])
        self.assertEqual(environment.action_origins, [True]*10)

    def test_mpc_fixed_goal_observed_history_warm_start_and_terminal_block(self):
        controller, calls = Controller(), []
        with patch("lewm_tta.simulation.execution.ObservedHistory", History):
            row = mpc(Environment(), PastOnlyDataset(), CASE, 42, "fr-fixed", "gaussian", controller,
                      {"history": 3, "frameskip": 5}, {"simulation": {"max_raw_steps": 20}}, calls)
        self.assertEqual((row["raw_steps"], row["planning_calls"]), (7, 2))
        self.assertEqual([item[0] for item in controller.seen], [10, 15])
        self.assertTrue(all(item[1].tolist() == [[35.]] for item in controller.seen))
        self.assertIsNone(controller.seen[0][3])
        self.assertEqual(tuple(controller.seen[1][3].shape), (1, 10))
        self.assertEqual([len(item["executed_raw_actions"]) for item in calls], [5, 2])
        self.assertEqual([item[2]["transition_id"] for item in controller.seen],
                         ["case-1/real-replan-0", "case-1/real-replan-1"])
        self.assertTrue(row["ever_success"])

    def test_replay_keeps_fixed_horizon_after_early_success(self):
        environment = Environment()
        row = replay(environment, CASE, 42, np.zeros((25, 2)))
        self.assertEqual((row["raw_steps"], row["final_state"]), (25, [35.]))
        self.assertTrue(row["ever_success"])
        self.assertEqual(environment.action_origins, [False]*25)

    def test_recorded_control_replay_marks_commands_as_recorded(self):
        environment = Environment()
        row = replay(environment, CASE, 42, np.full((25, 2), 1.082123), recorded=True)
        self.assertEqual(environment.action_origins, [True]*25)
        self.assertEqual(row["action_origin"], "recorded")

    def test_paired_statistics_and_generic_metric_names(self):
        config = {"adaptation": {"methods": ["frozen", "uniform", "fr-fixed"]},
                  "proposals": {"sources": ["gaussian", "context-fragments"]},
                  "evaluation": {"bootstrap_samples": 20}, "simulation": {"cohort_seed": 1}}
        rows = []
        for episode in (1, 2, 3):
            for seed in (42, 43):
                for method, effect in (("frozen", 0), ("uniform", -1), ("fr-fixed", -3)):
                    for source in config["proposals"]["sources"]:
                        distance = 10+episode+effect-(2 if source == "context-fragments" else 0)
                        rows.append({"case_id": f"case-{episode}", "query": {"source_episode_position": episode},
                            "seed": seed, "method": method, "proposal_source": source,
                            "initial_metrics": {"distance": 20, "success": False},
                            "final_metrics": {"distance": distance, "success": False},
                            "best_metrics": {"distance": distance, "success": 0},
                            "ever_success": False, "initial_success": False, "raw_steps": 20,
                            "planning_calls": 4, "wall_seconds": 1., "planning_seconds": .8, "context_encoding_seconds": .1})
        summary = summarize(rows, config)
        self.assertEqual(summary["episodes"], 3)
        self.assertEqual(summary["paired_contrasts"]["fr-fixed-minus-uniform/gaussian"]["final_distance"]["mean"], -2.)
        self.assertEqual(summary["paired_contrasts"]["interaction/fr-fixed-minus-uniform"]["final_distance"]["mean"], 0.)
        text = format_summary({"mode": "mpc", "completed": True, "summary": summary,
                               "environment": {"metric_directions": {"distance": -1, "success": 1}}})
        self.assertIn("final_distance", text)
        with self.assertRaises(ValueError):
            summarize(rows[:-1], config)


if __name__ == "__main__":
    unittest.main()

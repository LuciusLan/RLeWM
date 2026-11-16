"""User-run saved-trajectory contracts; no real simulator or learned model.

PYTHONPATH=codes python -m unittest discover -s codes/tests -p test_green_preview.py
"""

from copy import deepcopy
import unittest

import numpy as np

from lewm_tta.simulation.dataset import ResetCases
from lewm_tta.simulation.environments.base import Observation
from lewm_tta.simulation.execution import history_fingerprint
from lewm_tta.visualization.green import replay_saved, selections


def frame(position):
    return np.full((8, 8, 3), position, dtype=np.uint8)


def metrics(position, target=13):
    return {"coverage": min(position/target, 1.), "success": position >= target, "out_of_frame": False}


class SyntheticEnvironment:
    def __init__(self, target=13):
        self.target, self.steps = target, []
        self.goal_contract = {"goal": "fixed"}
        self.goal_image = frame(99)

    def reset(self, case, seed, *, history_start):
        self.position = 0
        self.steps = []
        return self.observe()

    def observe(self):
        return Observation(frame(self.position), np.array([self.position], dtype=float), metrics(self.position, self.target))

    def step(self, action, *, recorded=False):
        self.steps.append((np.asarray(action).copy(), recorded))
        self.position += 1
        return self.observe()

    def description(self):
        return {"metric_directions": {"coverage": 1, "success": 1, "out_of_frame": -1}}


class GreenPreviewContracts(unittest.TestCase):
    def case(self):
        return {"case_id": "case0", "start_raw_row": 0, "current_raw_row": 10,
                "goal_raw_row": None, "warmup_raw_actions": [[0., 0.]]*10,
                "query": {"episode_id": 2000000000}}

    def row(self, steps=3):
        target = 10+steps
        return {"case_id": "case0", "query": self.case()["query"], "seed": 42,
                "method": "frozen", "proposal_source": "gaussian", "raw_steps": steps,
                "executed_raw_actions": [[.125, -.25]]*steps, "planning_calls": int(steps > 0),
                "bootstrap_fingerprint": history_fingerprint([frame(0), frame(5), frame(10)], [np.zeros((5, 2), dtype=np.float32)]*2),
                "goal_contract": {"goal": "fixed"}, "initial_state": [10.], "final_state": [float(target)],
                "initial_metrics": metrics(10, target), "final_metrics": metrics(target, target),
                "best_metrics": {"coverage": 1., "success": 1., "out_of_frame": 0.},
                "ever_success": True, "initial_success": steps == 0,
                "first_success_raw_step": steps, "ever_out_of_frame": False}

    def test_replay_applies_only_saved_controls_after_causal_warmup(self):
        environment = SyntheticEnvironment()
        trace = replay_saved(environment, ResetCases(), self.case(), self.row(),
                             {"frameskip": 5, "history": 3},
                             {"simulation": {"execute_blocks": 1, "max_raw_steps": 300}})
        self.assertEqual(trace["states"], [[10.], [11.], [12.], [13.]])
        self.assertEqual(trace["replan_start_steps"], [0])
        self.assertEqual(trace["verification"]["final_state_max_abs_error"], 0.)
        self.assertEqual([recorded for action, recorded in environment.steps], [True]*10+[False]*3)
        np.testing.assert_array_equal(trace["goal_image"], frame(99))
        np.testing.assert_array_equal(trace["actions"], self.row()["executed_raw_actions"])

    def test_initial_success_is_a_valid_single_frame_preview(self):
        trace = replay_saved(SyntheticEnvironment(target=10), ResetCases(), self.case(), self.row(0),
                             {"frameskip": 5, "history": 3},
                             {"simulation": {"execute_blocks": 1, "max_raw_steps": 300}})
        self.assertEqual(len(trace["frames"]), 1)
        self.assertEqual(trace["actions"], [])
        self.assertEqual(trace["replan_start_steps"], [])

    def test_state_goal_metrics_and_action_count_drift_are_rejected(self):
        changed = []
        row = self.row()
        for key, value in (("final_state", [14.]), ("goal_contract", {"goal": "different"}),
                           ("bootstrap_fingerprint", "different"), ("raw_steps", 2)):
            changed.append({**deepcopy(row), key: value})
        row["final_metrics"]["coverage"] = .1
        changed.append(row)
        for saved in changed:
            with self.subTest(saved=saved), self.assertRaises(ValueError):
                replay_saved(SyntheticEnvironment(), ResetCases(), self.case(), saved,
                             {"frameskip": 5, "history": 3},
                             {"simulation": {"execute_blocks": 1, "max_raw_steps": 300}})

    def test_selection_uses_reserved_split_indices_and_requires_saved_treatment(self):
        cases = [self.case(), {**self.case(), "case_id": "case1", "query": {"episode_id": 2000000001}}]
        manifest = {"task": "green-target", "fingerprint": "manifest", "splits": {"validation": cases}}
        report = {"stage": "simulator-evaluation", "task": "green-target", "mode": "mpc", "split": "validation",
                  "manifest_fingerprint": "manifest", "case_ids": ["case0"], "config": {
                      "simulation": {"seeds": [42]}, "adaptation": {"methods": ["frozen", "fr-fixed"]},
                      "proposals": {"sources": ["gaussian", "context-fragments"]}}}
        selected = selections(report, manifest, [self.row()], [0], ["frozen"], ["gaussian"], None)
        self.assertEqual(selected[0][1], self.case())
        for episodes, methods, proposals, seed in (([1], ["frozen"], ["gaussian"], 42),
                ([0], ["fr-fixed"], ["gaussian"], 42), ([0], ["frozen"], ["gaussian"], 43),
                ([0, 0], ["frozen"], ["gaussian"], 42)):
            with self.assertRaises(ValueError):
                selections(report, manifest, [self.row()], episodes, methods, proposals, seed)


if __name__ == "__main__":
    unittest.main()

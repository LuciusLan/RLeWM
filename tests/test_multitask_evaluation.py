"""User-run multi-task binding and synthetic action-layout regression checks.

PYTHONPATH=codes python -m unittest discover -s codes/tests -p test_multitask_evaluation.py
No real checkpoint/model/simulator is executed by these tests.
"""

import contextlib
import io
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from experiments.catalog import ROOT, profile
from experiments.cli import dispatch
from experiments.config import resolve
from experiments.fresh import effective_config
from lewm_tta.collection.contracts import validate_evaluation_config, validate_query_task
from lewm_tta.frame_bank import FrameLatentBank
from lewm_tta.planning.actions import ActionCoordinates
from lewm_tta.planning.cem import solve_cem
from lewm_tta.planning.temporal.config import validate_config as validate_temporal
from lewm_tta.proposals.config import validate_config as validate_proposals
from lewm_tta.proposals.fragments import FragmentStore
from lewm_tta.rollout.config import validate_config as validate_rollouts


class MultiTaskEvaluationContracts(unittest.TestCase):
    def test_shared_stage_profiles_bind_sources_and_keep_budgets(self):
        for task, width in (("tworooms", 2), ("reacher", 2), ("cube", 5)):
            for stage, variant, validator in (("p2", "fresh", validate_rollouts),
                    ("p2", "fresh-temporal", validate_rollouts), ("p3", "fresh", validate_proposals),
                    ("p3", "fresh-proposals", validate_proposals), ("p3", "fresh-temporal", validate_temporal)):
                with self.subTest(task=task, stage=stage, variant=variant):
                    path = profile(stage, variant)[1]
                    original = resolve(path)
                    bound = resolve(path, dataset=task)
                    effective = effective_config(bound)
                    validator(effective)
                    self.assertEqual(effective["p1"]["data"]["dataset_id"], task)
                    self.assertEqual(effective["p1"]["preprocessing"]["raw_action_dim"], width)
                    self.assertIn(f"fresh_data/{task}/source/", bound["query_dataset"]["manifest"])
                    self.assertIn(f"evaluation/{task}/", bound["experiment_dir"])
                    self.assertEqual(original["adaptation"], bound["adaptation"])
                    if "planning" in bound:
                        for key in ("population", "iterations", "elites", "horizon"):
                            self.assertEqual(original["planning"][key], bound["planning"][key])
                        self.assertEqual(bound["planning"]["raw_action_low"], [-1]*width)
                        self.assertEqual(bound["planning"]["raw_action_high"], [1]*width)
                        self.assertFalse(bound.get("reuse"))

    def test_pusht_and_unbound_resolutions_stay_identical(self):
        for stage, variant in (("p2", "fresh"), ("p3", "fresh"), ("p3", "fresh-temporal")):
            path = profile(stage, variant)[1]
            self.assertEqual(resolve(path), resolve(path, dataset="pusht"))
        with self.assertRaisesRegex(ValueError, "fresh P2/P3"):
            resolve(profile("p3", "default")[1], dataset="cube")

    def test_same_width_different_task_queries_are_rejected(self):
        config = effective_config(resolve(profile("p2", "fresh")[1], dataset="reacher"))
        memory = {"data_config": {"dataset_id": "reacher"}, "source": {"schema": {"action": {"shape": [100, 2]}}}}
        query = {"backend": "tworooms", "source": {"schema": {"action": {"shape": [50, 2]}}}}
        with self.assertRaisesRegex(ValueError, "backend"):
            validate_query_task(config, memory, query)
        query["backend"] = "reacher"
        validate_query_task(config, memory, query)
        query["source"]["schema"]["action"]["shape"][-1] = 5
        with self.assertRaisesRegex(ValueError, "action widths"):
            validate_query_task(config, memory, query)

    def test_memory_only_queries_and_cube_bounds_are_checked_before_inference(self):
        config = effective_config(resolve(profile("p3", "fresh")[1], dataset="cube"))
        config["planning"]["raw_action_low"] = [-1, -1]
        with self.assertRaisesRegex(ValueError, "5 coordinates"):
            validate_evaluation_config(config)
        config.pop("query_dataset")
        with self.assertRaisesRegex(ValueError, "independent query_dataset"):
            validate_evaluation_config(config)

    def test_dataset_selector_uses_fresh_without_changing_old_defaults(self):
        for stage, task in (("p2", "reacher"), ("p3", "cube")):
            stream = io.StringIO()
            with contextlib.redirect_stdout(stream):
                dispatch([stage, "prepare", "--dataset", task, "--dry-run"])
            self.assertIn(f"{stage}/fresh", stream.getvalue())
            self.assertIn(f"artifacts/evaluation/{task}/{stage}", stream.getvalue())
        with self.assertRaisesRegex(ValueError, "P1/collect"):
            dispatch(["p1", "prepare", "--dataset", "cube", "--dry-run"])

    def test_six_cell_children_receive_the_resolved_task_configuration(self):
        from lewm_tta.runtime import object_hash
        bound = resolve(profile("p3", "fresh")[1], dataset="cube")
        with tempfile.TemporaryDirectory() as temporary:
            bound["experiment_dir"] = str(Path(temporary)/"cube")
            expected = str(ROOT/"artifacts/experiment-configs"/f"{object_hash(bound)}.json")
            output = io.StringIO()
            with patch("experiments.workflows.resolve", return_value=bound), contextlib.redirect_stdout(output):
                dispatch(["p3", "compare", "--dataset", "cube", "--phase", "validation", "--dry-run"])
            commands = [line for line in output.getvalue().splitlines() if "--config" in line]
            self.assertTrue(commands)
            self.assertTrue(all(expected in line for line in commands))
            self.assertTrue(all("codes/configs/p3/fresh.json" not in line for line in commands))

    def test_five_coordinate_fragments_stay_aligned_and_inside_episodes(self):
        raw_width, frameskip = 5, 5
        rows = 90
        actions = torch.arange(rows*raw_width, dtype=torch.float32).reshape(rows, raw_width)/1000
        manifest = {"source": {"path": "synthetic"}, "data_config": {"history": 3, "frameskip": frameskip},
            "episode_metadata": {"offsets": [0, 45], "lengths": [45, 45], "ids": [7, 8]},
            "episodes": {"memory": [0, 1], "validation": [], "test": []}}
        store = SimpleNamespace(latents=torch.zeros(rows, 3), actions=actions)
        memory = FrameLatentBank(store, manifest, "memory", np.asarray([0, 9, 10, 45], dtype=np.int64), {})
        coordinates = ActionCoordinates({"mean": [0]*raw_width, "std": [1]*raw_width},
            {"raw_action_low": [-1]*raw_width, "raw_action_high": [1]*raw_width}, frameskip, "cpu")
        fragments = FragmentStore(memory, 5, coordinates)
        self.assertEqual(fragments.eligible.tolist(), [True, True, False, True])
        first = fragments.actions_at(0)
        self.assertEqual(first.shape, (5, 25))
        torch.testing.assert_close(coordinates.to_raw(first), actions[10:35].reshape(5, 5, 5))
        torch.testing.assert_close(coordinates.from_raw(coordinates.to_raw(first)), first)
        with self.assertRaises(ValueError):
            fragments.actions_at(2)

    def test_analytic_cem_accepts_25_coordinate_blocks_without_a_world_model(self):
        coordinates = ActionCoordinates({"mean": [0]*5, "std": [1]*5},
            {"raw_action_low": [-1]*5, "raw_action_high": [1]*5}, 5, "cpu")
        settings = {"horizon": 5, "population": 8, "iterations": 2, "elites": 2,
                    "initial_std": 1., "minimum_std": 0.}
        score = lambda actions: (actions-.1).square().flatten(1).sum(1)
        plan = solve_cem(score, settings, coordinates, 42)
        self.assertEqual(plan.actions.shape, (5, 25))
        self.assertEqual(plan.candidate_evaluations, 17)
        raw = coordinates.to_raw(plan.actions)
        self.assertEqual(raw.shape, (5, 5, 5))
        self.assertTrue(bool((raw.abs() <= 1).all()))


if __name__ == "__main__":
    unittest.main()

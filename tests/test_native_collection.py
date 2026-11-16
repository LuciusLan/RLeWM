"""User-run collection contracts; synthetic frames only, no simulator or model.

PYTHONPATH=codes python -m unittest discover -s codes/tests -p test_native_collection.py
"""

from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import h5py
import numpy as np

from experiments.catalog import ROOT
from experiments.config import resolve
from lewm_tta.collection.config import validate_config
from lewm_tta.collection.goals import available_starts
from lewm_tta.collection.manifest import checked, prepare
from lewm_tta.collection.pipeline import collect, check
from lewm_tta.collection.storage import EpisodeWriter
from lewm_tta.simulation.environments import make_collection_environment
from lewm_tta.simulation.environments.base import Observation
from lewm_tta.simulation.environments.fresh_native import numpy_stream
from lewm_tta.simulation.environments.fresh_cube import FreshCube
from lewm_tta.simulation.environments.fresh_reacher import FreshReacher
from lewm_tta.visualization.episodes import load_episode
from lewm_tta.runtime import write_json


class SyntheticNative:
    collection_schema = staticmethod(FreshReacher.collection_schema)
    goal_contract = staticmethod(FreshReacher.goal_contract)
    goal_diagnostics = staticmethod(FreshReacher.goal_diagnostics)
    reference_state = staticmethod(FreshReacher.reference_state)

    def __init__(self, image_size, shift):
        self.size = image_size

    def description(self):
        return {"synthetic": True, "control_hz": 25}

    def reset(self, seed):
        self.position = float(seed % 10000)+10
        self.time = 0
        return self.observe()

    def observe(self):
        qpos, qvel = np.array([self.position, 1.]), np.array([.5, 0.])
        return Observation(np.full((self.size, self.size, 3), self.time, dtype=np.uint8), np.r_[qpos, qvel], {},
            fields={"qpos": qpos, "qvel": qvel, "observation": np.zeros(6), "finger_pos": np.zeros(2),
                    "target_pos": np.zeros(2), "physics_time": np.array(self.time*.04), "control": np.zeros(2)})

    def collection_action(self, observation, rng, step, settings):
        return np.array([.5, 0], dtype=np.float32)

    def step(self, action):
        self.position += action[0]
        self.time += 1
        return self.observe()

    def close(self):
        pass


class NativeCollectionContracts(unittest.TestCase):
    def test_cube_plan_profile_is_independent_and_strict(self):
        plan = resolve(ROOT/"codes/configs/collect/cube-plan.json")
        markov = resolve(ROOT/"codes/configs/collect/cube.json")
        validate_config(plan)
        self.assertNotEqual(plan["seed"], markov["seed"])
        self.assertNotEqual(plan["output_dir"], markov["output_dir"])
        self.assertEqual(plan["splits"], {"validation": 64, "test": 100})
        self.assertEqual(plan["policy"], {"name": "native-plan-oracle", "noise": .1,
                                       "noise_smoothing": .5, "segment_dt": .4})
        self.assertEqual(available_starts(plan), 166)
        with patch("lewm_tta.simulation.environments.fresh_environment_class") as registry:
            registry.return_value.collection_options.return_value = FreshCube.collection_options(plan)
            make_collection_environment(plan, "source")
            registry.return_value.assert_called_once_with(224, {}, policy=plan["policy"],
                                                         expected_physics=plan["expected_physics"])
        with self.assertRaisesRegex(ValueError, "requires mujoco"):
            validate_config({**plan, "expected_physics": {"mujoco": "3.15.0", "dm-control": "1.0.38"}})
        with self.assertRaisesRegex(ValueError, "Expected policy"):
            validate_config({**plan, "policy": {**plan["policy"], "action_noise": .1}})
        with self.assertRaisesRegex(ValueError, "noise_smoothing"):
            validate_config({**plan, "policy": {**plan["policy"], "noise_smoothing": 0}})

    def test_cube_physics_mismatch_fails_before_simulator_construction(self):
        config = resolve(ROOT/"codes/configs/collect/cube-plan.json")
        with patch("lewm_tta.simulation.environments.fresh_cube.version", return_value="wrong"):
            with self.assertRaisesRegex(ValueError, "Use .venvs/physics-mj36/bin/python"):
                FreshCube(224, {}, **FreshCube.collection_options(config))

    def test_cube_plan_controls_retarget_and_do_not_add_action_noise(self):
        class FakePlan:
            def __init__(self, env, **options):
                self.options, self.done, self.resets = options, False, 0

            def reset(self, observation, info):
                np.random.uniform()  # Exercise isolated legacy oracle RNG.
                self.resets += 1
                self.done = False

            def select_action(self, observation, info):
                np.random.uniform()
                return np.array([2., -.2, .3, -3., .125])

        class NoActionNoise:
            def __init__(self):
                self.rng = np.random.default_rng(123)

            def integers(self, *args, **kwargs):
                return self.rng.integers(*args, **kwargs)

            def uniform(self, *args, **kwargs):
                return self.rng.uniform(*args, **kwargs)

            def normal(self, *args, **kwargs):
                raise AssertionError("Plan controls must not receive extra Gaussian noise")

        cube = object.__new__(FreshCube)  # No simulator or installed oracle import.
        cube.policy_settings = resolve(ROOT/"codes/configs/collect/cube-plan.json")["policy"]
        cube.policy, cube.assets = None, {"target_events": []}
        cube.info = {"privileged/target_block_pos": np.array([.4, .1, .02]),
                     "privileged/target_block_yaw": np.array([0.])}
        cube.env = SimpleNamespace(action_space=SimpleNamespace(shape=(5,)))
        changes = []
        cube.native = SimpleNamespace(variation_space=SimpleNamespace(set_value=changes.append),
            set_new_target=lambda p_stack: (None, {"privileged/target_block_pos": np.array([.5, 0., .02]),
                                                  "privileged/target_block_yaw": np.array([.25])}))
        previous = np.random.get_state()
        rng = NoActionNoise()
        with patch.object(cube, "behavior_classes", return_value=[FakePlan]), \
             patch.object(cube, "sample_variations", return_value={"cube.goal_yaw": .25}):
            first = cube.collection_action(None, rng, 0, cube.policy_settings)
            cube.policy.done = True
            second = cube.collection_action(None, rng, 89, cube.policy_settings)
        np.testing.assert_array_equal(first, np.array([1., -.2, .3, -1., .125], dtype=np.float32))
        np.testing.assert_array_equal(first, second)
        self.assertEqual(cube.policy.options, {"noise": .1, "noise_smoothing": .5, "segment_dt": .4})
        self.assertEqual(cube.policy.resets, 2)
        self.assertEqual(changes, [{"cube.goal_yaw": .25}])
        self.assertEqual([event["raw_step"] for event in cube.assets["target_events"]], [0, 89])
        self.assertEqual([event["target_sequence_index"] for event in cube.assets["target_events"]], [0, 1])
        self.assertFalse(cube.assets["extra_action_noise"])
        after = np.random.get_state()
        self.assertEqual(previous[0], after[0])
        np.testing.assert_array_equal(previous[1], after[1])
        self.assertEqual(previous[2:], after[2:])

    def test_task_profiles_and_goal_protocols(self):
        for task in ("tworooms", "reacher", "cube"):
            config = resolve(ROOT/f"codes/configs/collect/{task}.json")
            validate_config(config)
            self.assertEqual(config["splits"], {"validation": 64, "test": 100})
            self.assertEqual(config["goal_offset_steps"], 25)
            self.assertEqual(available_starts(config), 91 if task == "tworooms" else 166)
            config["regimes"]["unsupported"] = {"gravity": 0}
            with self.assertRaisesRegex(ValueError, "source dynamics only"):
                validate_config(config)

    def test_export_native_columns_action_alignment_and_audit(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = resolve(ROOT/"codes/configs/collect/reacher.json")
            config.update(output_dir=str(root/"fresh"), reference_dataset=str(root/"original.h5"),
                          raw_steps=35, windows_per_episode=1, splits={"validation": 1, "test": 1})
            with h5py.File(config["reference_dataset"], "w") as stream:
                stream["qpos"] = np.full((2, 2), -100, dtype=np.float64)
                stream["qvel"] = np.ones((2, 2))
                stream["ep_offset"], stream["ep_len"], stream["ep_idx"] = [0], [2], [0, 0]
                stream["action"] = np.zeros((2, 2))
            prepare(config)
            with patch("lewm_tta.simulation.environments.fresh_environment_class", return_value=SyntheticNative):
                collect(config)
                audit = check(config)[0]
                self.assertTrue(audit["passed"])
                self.assertEqual(audit["exact_duplicate_complete_episodes"], 0)
                with h5py.File(root/"fresh/source/dataset.h5", "r") as stream:
                    np.testing.assert_array_equal(stream["qpos"][:], stream["state64"][:, :2])
                    np.testing.assert_array_equal(stream["qpos"][1]-stream["qpos"][0], stream["action"][0])
                    self.assertFalse(stream["action_valid"][35])
                    self.assertTrue(stream["action_valid"][36])
                    self.assertEqual(stream["physics_time"][36], 0)
                catalog = checked(root/"fresh/source/goal-protocols.json")
                self.assertEqual(len(catalog["cases"]), 2)
                for case in catalog["cases"]:
                    self.assertEqual(case["goal_rows"]["lewm-paper"]-case["current_row"], 25)
                query = checked(root/"fresh/source/queries.json")
                self.assertEqual(query["episodes"]["memory"], [])
                self.assertTrue(set(query["episodes"]["validation"]).isdisjoint(query["episodes"]["test"]))
                # Resume must preserve the VDS identity used by observation caches.
                collect(config)
                self.assertEqual(query, checked(root/"fresh/source/queries.json"))

    def test_alternative_goal_preserves_the_same_reserved_anchor(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = resolve(ROOT/"codes/configs/collect/tworooms.json")
            config.update(output_dir=str(root/"fresh"), splits={"validation": 1, "test": 1})
            manifest = prepare(config)
            from lewm_tta.collection.storage import episode_paths
            from lewm_tta.collection.manifest import sealed
            from lewm_tta.runtime import file_hash
            path, metadata = episode_paths(config, "source", manifest["cases"][0])
            path.parent.mkdir(parents=True)
            with h5py.File(path, "w") as stream:
                stream["pixels"] = np.zeros((201, 1, 1, 3), dtype=np.uint8)
            write_json(metadata, sealed({"case": manifest["cases"][0], "regime": "source",
                "collection_manifest": manifest["fingerprint"], "sha256": file_hash(path),
                "environment": {"control_hz": 10}}))
            profile = root/"config.json"
            write_json(profile, config)
            repository = load_episode(profile, "validation", 0, "source")
            paper = load_episode(profile, "validation", 0, "source", goal_protocol="lewm-paper")
            self.assertEqual(repository["start"], paper["start"])
            self.assertEqual(repository["goal"]-repository["current"], 25)
            self.assertEqual(paper["goal"]-paper["current"], 100)
            self.assertLessEqual(paper["goal"], 200)

    def test_nonfinite_native_replay_fields_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            writer = EpisodeWriter(Path(temporary)/"episode.h5", {"raw_steps": 1, "image_size": 2, "compression": "gzip"}, {},
                                   {"state_width": 1, "action_width": 1, "proprio_indices": [0], "fields": {"qvel": [1]}})
            try:
                with self.assertRaisesRegex(ValueError, "qvel"):
                    writer.append(Observation(np.zeros((2, 2, 3), dtype=np.uint8), np.zeros(1), {}, fields={"qvel": [np.nan]}))
            finally:
                writer.close()

    def test_legacy_numpy_oracle_rng_is_isolated_even_on_failure(self):
        previous = np.random.get_state()
        stream = np.random.RandomState(123)
        expected = np.random.RandomState(123).uniform(size=4)
        try:
            with numpy_stream(stream):
                actual = np.random.uniform(size=4)
                raise RuntimeError("oracle error")
        except RuntimeError:
            pass
        np.testing.assert_array_equal(actual, expected)
        after = np.random.get_state()
        np.testing.assert_array_equal(previous[1], after[1])
        self.assertEqual(previous[2:], after[2:])
        self.assertNotEqual(stream.get_state()[2], 624)


if __name__ == "__main__":
    unittest.main()

"""Synthetic fresh-data contracts; no simulator, checkpoint, TTT or cuVS.

User-run: PYTHONPATH=codes python -m unittest discover -s codes/tests -p test_fresh_collection.py
"""

import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import h5py
import numpy as np

from experiments.config import resolve
from experiments.catalog import ROOT
from lewm_tta.collection.manifest import cases, prepare, checked
from lewm_tta.collection.pipeline import collect, check, state_keys
from lewm_tta.collection.queries import namespace_bank
from lewm_tta.simulation.environments.base import Observation
from lewm_tta.simulation.environments.fresh_pusht import FreshPushT, validate_shift
from lewm_tta.simulation.environments.fresh_recorded_pusht import FreshRecordedPushT


class SyntheticEnvironment:
    collection_schema = staticmethod(FreshPushT.collection_schema)
    def __init__(self, image_size, shift):
        self.size = image_size

    def description(self):
        return {"synthetic": True}

    def reset(self, seed):
        self.position = float(seed % 100000)+100
        self.t = 0
        return self.observe()

    def observe(self):
        state = np.array([self.position, self.t, 20+self.t, 30, .1, 0, 0.])
        return Observation(np.full((self.size, self.size, 3), self.t, dtype=np.uint8), state, {})

    def collection_action(self, observation, rng, step, settings):
        return np.array([.25, -.125], dtype=np.float32)

    def step(self, action):
        self.position += float(action[0])
        self.t += 1
        return self.observe()

    def close(self):
        pass


class FreshCollectionContracts(unittest.TestCase):
    def config(self, root):
        value = resolve(ROOT/"codes/configs/collect/default.json")
        value.update(output_dir=str(root/"fresh"), reference_dataset=str(root/"original.h5"),
                     raw_steps=35, windows_per_episode=1, splits={"validation": 1, "test": 1})
        value.pop("quality", None)
        value["policy"] = {"name": "contact-exploration"}
        return value

    def test_membership_and_rngs_are_reserved_independently_of_regime(self):
        value = resolve(ROOT/"codes/configs/collect/default.json")
        original = cases(value)
        value["regimes"] = {"source": {}, "shape-I": {"geometry": {"block_shape": "I"}}}
        self.assertEqual(original, cases(value))
        self.assertEqual(len({row["reset_seed"] for row in original}), len(original))
        validation = {row["case_id"] for row in original if row["split"] == "validation"}
        test = {row["case_id"] for row in original if row["split"] == "test"}
        self.assertFalse(validation & test)

    def test_episode_commit_resume_export_and_action_alignment(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            value = self.config(root)
            with h5py.File(root/"original.h5", "w") as stream:
                stream["state"] = np.full((2, 7), -123, dtype=np.float32)
                stream["episode_idx"] = [0, 0]
                stream["ep_offset"] = [0]
            prepare(value)
            with patch("lewm_tta.simulation.environments.fresh_pusht.FreshPushT", SyntheticEnvironment):
                collect(value)
                first = checked(root/"fresh/source/queries.json")
                collect(value)  # Must preserve virtual file mtime/content identity.
                self.assertEqual(first, checked(root/"fresh/source/queries.json"))
            with h5py.File(root/"fresh/source/dataset.h5", "r") as stream:
                self.assertEqual(stream["pixels"].shape, (72, 224, 224, 3))
                np.testing.assert_array_equal(stream["ep_len"][:], [36, 36])
                np.testing.assert_array_equal(stream["ep_offset"][:], [0, 36])
                self.assertEqual(int(stream["action_valid"][35]), 0)
                self.assertEqual(int(stream["action_valid"][36]), 1)
                self.assertAlmostEqual(float(stream["state64"][1, 0]-stream["state64"][0, 0]), .25)
                np.testing.assert_array_equal(stream["action"][0], [.25, -.125])
                self.assertEqual(int(stream["step_idx"][36]), 0)
            self.assertTrue(check(value)[0]["passed"])
            # Mutating a source episode must be detected even though VDS metadata did not change.
            episode = Path(first["episode_files"][0]["path"])
            with h5py.File(episode, "r+") as stream:
                stream["state"][0, 0] = -999
            with self.assertRaises(ValueError):
                check(value)

    def test_signed_zeros_do_not_evade_source_state_duplicate_check(self):
        positive = np.zeros((1, 7), dtype=np.float32)
        negative = -positive
        np.testing.assert_array_equal(state_keys(positive), state_keys(negative))

    def test_external_metadata_never_collides_with_original_episode_positions(self):
        import torch
        from lewm_tta.frame_bank import FrameLatentBank
        metadata = {"external_queries": True, "regime": "source", "collection_manifest": "abc",
                    "source": {"path": "fresh.h5"}, "data_config": {"history": 3, "frameskip": 5},
                    "episode_metadata": {"offsets": [0], "lengths": [36], "ids": [1_000_000_000]},
                    "episodes": {"validation": [0]}}
        store = SimpleNamespace(latents=torch.zeros(36, 2), actions=torch.zeros(36, 2))
        bank = namespace_bank(FrameLatentBank(store, metadata, "validation", np.array([0], dtype=np.int64), {}))
        self.assertEqual(bank.episode_positions.tolist(), [0])
        self.assertLess(bank.query(0).metadata["source_episode_position"], 0)
        self.assertEqual(bank.query(0).metadata["local_episode_position"], 0)

    def test_fresh_reconstruction_replays_only_the_complete_past_prefix(self):
        environment = FreshRecordedPushT.__new__(FreshRecordedPushT)
        observation = Observation(np.zeros((2, 2, 3), dtype=np.uint8), np.zeros(7), {})
        environment.dataset = Mock()
        environment.dataset.rows.side_effect = lambda key, start, stop=None: np.zeros(7) if key == "state64" else np.zeros((stop-start, 2))
        environment.set_goal = Mock()
        environment._observe = Mock(return_value=observation)
        case = {"episode_reset_seed": 12345, "episode_start_raw_row": 100,
                "start_raw_row": 110, "current_raw_row": 120, "goal_raw_row": 145}
        with patch.object(FreshPushT, "reset", return_value=observation) as reset, patch.object(FreshPushT, "step", return_value=observation) as step:
            environment.reset(case, 42, history_start=True)
            reset.assert_called_once_with(12345)
            self.assertEqual(step.call_count, 10)
        environment.dataset.rows.assert_any_call("action", 100, 110)

    def test_unknown_or_invalid_shifts_fail_before_simulator_creation(self):
        for shift in ({"dynamics": {"block_mass_scale": -1}}, {"visual": {"block_color": [256, 0, 0]}},
                      {"geometry": {"block_shape": "unknown"}}, {"dynamics": {"silent_typo": 2}}):
            with self.subTest(shift=shift), self.assertRaises(ValueError):
                validate_shift(shift)
        validate_shift({"geometry": {"block_shape": "I"}, "visual": {"agent_color": [255, 0, 0]}})


if __name__ == "__main__":
    unittest.main()

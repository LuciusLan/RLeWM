"""User-run synthetic contracts: no real simulator, pretrained model or CEM.

PYTHONPATH=codes python -m unittest discover -s codes/tests -p test_collection_quality.py
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
from lewm_tta.collection.manifest import attempt_case, checked, prepare
from lewm_tta.collection.pipeline import collect, check
from lewm_tta.collection.quality import assess_episode, geometry_visible, select_starts
from lewm_tta.simulation.environments.base import Observation
from lewm_tta.simulation.environments.fresh_pusht import FreshPushT
from lewm_tta.simulation.environments.pusht_collection import agent_block_pair, body_bounds
from lewm_tta.visualization.episodes import load_episode
from lewm_tta.visualization.plans import physical_case
from lewm_tta.runtime import write_json


class SyntheticQualityEnvironment:
    collection_schema = staticmethod(FreshPushT.collection_schema)
    goal_diagnostics = staticmethod(FreshPushT.goal_diagnostics)

    def __init__(self, image_size, shift):
        self.size, self.resets = image_size, 0

    def description(self):
        return {"synthetic": True, "control_hz": 10}

    def reset(self, seed):
        self.resets += 1
        self.bad = self.resets == 1
        self.base, self.t = 100+(seed % 100000)/1000, 0
        return self.observe()

    def observe(self):
        state = np.array([self.base+self.t, 100, self.base+self.t, 150, 0., 0., 0.])
        return Observation(np.full((self.size, self.size, 3), self.t, dtype=np.uint8), state, {})

    def begin_collection(self, policy):
        pass

    def collection_measurement(self):
        x = self.base+self.t
        bounds = np.array([[x-10, 90, x+10, 110], [x-20, 140, x+20, 160]])
        if self.bad:
            bounds[1, 0] = -1
        return {"bounds": bounds, "contacts": int(self.t > 0), "extent": 512.}

    def collection_action(self, observation, rng, step, settings):
        return np.array([.1, 0], dtype=np.float32)

    def step(self, action):
        self.t += 1
        return self.observe()

    def close(self):
        pass


class CollectionQualityContracts(unittest.TestCase):
    def config(self, root):
        value = resolve(ROOT/"codes/configs/collect/default.json")
        value.update(output_dir=str(root/"fresh"), reference_dataset=str(root/"original.h5"),
                     raw_steps=35, windows_per_episode=1, splits={"validation": 1, "test": 1})
        return value

    def test_gate_rejects_invisible_intermediate_frames_and_aligns_contact_actions(self):
        value = self.config(Path("/tmp/fixture"))
        states = np.zeros((36, 7))
        states[:, 2] = np.arange(36)
        bounds = np.tile([[[100, 100, 120, 120], [150, 150, 220, 220]]], (36, 1, 1))
        contacts = np.zeros(36)
        diagnose = lambda a, b: FreshPushT.goal_diagnostics(a, b, {})
        contacts[10] = 1  # Into current state; this is NOT a future contact.
        result = assess_episode(value, states, bounds, contacts, 512, diagnose)
        self.assertEqual(result["eligible_start_steps"], [])
        contacts[10], contacts[11] = 0, 1
        self.assertEqual(assess_episode(value, states, bounds, contacts, 512, diagnose)["eligible_start_steps"], [0])
        bounds[20, 1, 2] = 520  # Valid start/goal anchors do not rescue an offscreen intermediate.
        result = assess_episode(value, states, bounds, contacts, 512, diagnose)
        self.assertFalse(result["accepted"])
        self.assertEqual(result["window_failure_counts"]["visibility"], 1)

    def test_static_block_and_already_reached_goal_are_ineligible(self):
        value = self.config(Path("/tmp/fixture"))
        states = np.zeros((36, 7))
        bounds = np.tile([[[100, 100, 120, 120], [150, 150, 220, 220]]], (36, 1, 1))
        result = assess_episode(value, states, bounds, np.ones(36), 512,
                                lambda a, b: FreshPushT.goal_diagnostics(a, b, {}))
        self.assertFalse(result["accepted"])
        self.assertEqual(result["window_failure_counts"]["low_motion"], 1)
        self.assertEqual(result["window_failure_counts"]["initial_success"], 1)

    def test_rotation_only_contact_can_be_eligible_without_block_translation(self):
        value = self.config(Path("/tmp/fixture"))
        states = np.zeros((36, 7))
        states[:, 4] = np.linspace(0, 1, 36)
        bounds = np.tile([[[100, 100, 120, 120], [150, 150, 220, 220]]], (36, 1, 1))
        contacts = np.ones(36)
        contacts[0] = 0
        result = assess_episode(value, states, bounds, contacts, 512,
                                lambda a, b: FreshPushT.goal_diagnostics(a, b, {}))
        self.assertTrue(result["accepted"])

    def test_geometry_uses_full_world_shape_and_contacts_exclude_walls(self):
        class Body:
            def local_to_world(self, point):
                return (490-point[1], 250+point[0])  # Rotated 90 degrees.
        block, agent, wall = Body(), Body(), Body()
        agent.local_to_world = lambda point: (200+point[0], 200+point[1])
        polygon = SimpleNamespace(get_vertices=lambda: [(-10, -35), (10, -35), (10, 35), (-10, 35)], radius=0)
        block.shapes = [polygon]
        agent.shapes = [SimpleNamespace(offset=(0, 0), radius=15)]
        native = SimpleNamespace(block=block, agent=agent)
        self.assertFalse(geometry_visible(body_bounds(native), 512, 8))
        shapes = lambda first, second: SimpleNamespace(shapes=[SimpleNamespace(body=first), SimpleNamespace(body=second)])
        self.assertTrue(agent_block_pair(shapes(agent, block), native))
        self.assertTrue(agent_block_pair(shapes(block, agent), native))
        self.assertFalse(agent_block_pair(shapes(agent, wall), native))

    def test_retries_export_actual_seed_and_preview_uses_the_selected_window(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            value = self.config(root)
            config_path = root/"collection.json"
            write_json(config_path, value)
            with h5py.File(root/"original.h5", "w") as original:
                original["state"] = np.full((2, 7), -123, dtype=np.float32)
                original["episode_idx"] = [0, 0]
                original["ep_offset"] = [0]
            manifest = prepare(value)
            with patch("lewm_tta.simulation.environments.fresh_pusht.FreshPushT", SyntheticQualityEnvironment):
                collect(value)
                exported = checked(root/"fresh/source/queries.json")
                collect(value)  # Committed episodes and VDS are immutable on resume.
                self.assertEqual(exported, checked(root/"fresh/source/queries.json"))
                # Recover a crash after accepted data/attempt commit but before
                # the final episode JSON was published; never reroll that case.
                (root/"fresh/source/episodes/validation-00000.json").unlink()
                collect(value)
                self.assertEqual(exported, checked(root/"fresh/source/queries.json"))
            episode = exported["episode_files"][0]
            self.assertEqual(episode["accepted_attempt"], 1)
            self.assertEqual(exported["cases"][0], attempt_case(manifest["cases"][0], 1))
            self.assertEqual(len(episode["attempts"]), 2)
            self.assertEqual(episode["quality"]["selected_start_steps"], [0])
            preview = load_episode(config_path, "validation", 0, "source")
            self.assertEqual(preview["reserved_starts"], [0])
            self.assertEqual(physical_case(preview)["episode_reset_seed"], exported["cases"][0]["reset_seed"])
            audited = check(value)[0]
            self.assertTrue(audited["passed"])
            self.assertEqual(audited["behavior_quality"]["by_split"]["validation"]["attempts"], 2)

    def test_cap_does_not_publish_failed_episode_or_silently_relax_thresholds(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            value = self.config(root)
            value["quality"]["max_attempts"] = 2
            prepare(value)
            class NeverEligible(SyntheticQualityEnvironment):
                def reset(self, seed):
                    observation = super().reset(seed)
                    self.bad = True
                    return observation
            with patch("lewm_tta.simulation.environments.fresh_pusht.FreshPushT", NeverEligible):
                with self.assertRaisesRegex(ValueError, "attempt cap"):
                    collect(value, split="validation")
            self.assertFalse(list((root/"fresh/source/episodes").glob("*.h5")))
            self.assertEqual(len(list((root/"fresh/source/attempts").rglob("*.json"))), 2)
            self.assertFalse((root/"fresh/source/queries.json").exists())

    def test_sampling_is_seeded_without_replacement_and_settings_are_validated(self):
        case = {"case_id": "fixture"}
        self.assertEqual(select_starts(case, [3, 5, 9, 10], 3), select_starts(case, [3, 5, 9, 10], 3))
        with self.assertRaises(ValueError):
            select_starts(case, [3], 2)
        config = self.config(Path("/tmp/fixture"))
        validate_config(config)
        config["policy"]["max_action_norm"] = 2
        with self.assertRaises(ValueError):
            validate_config(config)


if __name__ == "__main__":
    unittest.main()

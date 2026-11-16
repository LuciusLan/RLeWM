"""User-run visual export contracts; no real model or simulator execution."""

from pathlib import Path
import tempfile
import unittest

import h5py
import numpy as np
from PIL import Image

from experiments.catalog import ROOT
from experiments.config import resolve
from lewm_tta.collection.manifest import prepare, sealed
from lewm_tta.collection.storage import episode_paths
from lewm_tta.runtime import file_hash, write_json
from lewm_tta.visualization.episodes import load_episode
from lewm_tta.visualization.media import Panel, compose, save_gallery
from lewm_tta.visualization.plans import physical_case


class VisualPreviewContracts(unittest.TestCase):
    def test_target_panel_stays_fixed_and_sources_are_not_modified(self):
        frames = [np.full((8, 8, 3), value, dtype=np.uint8) for value in (20, 100)]
        target = np.full((8, 8, 3), [0, 255, 0], dtype=np.uint8)
        before = target.copy()
        panels = [Panel("trajectory", frames, [10, 11]), Panel("goal", [target], [35], fixed=True)]
        image = np.asarray(compose(panels, 1, 1))
        np.testing.assert_array_equal(image[66:, 8:16], target)
        np.testing.assert_array_equal(target, before)
        np.testing.assert_array_equal(frames[0], np.full((8, 8, 3), 20, dtype=np.uint8))

    def test_subsampling_retains_endpoint_and_physical_playback_duration(self):
        frames = [np.full((8, 8, 3), 20*index, dtype=np.uint8) for index in range(6)]
        with tempfile.TemporaryDirectory() as temporary:
            result = save_gallery(Path(temporary), "fixture", [Panel("episode", frames, list(range(6)))],
                                  fps=10, every=4, scale=1)
            self.assertEqual(result["displayed_frames"], 3)  # raw indices0,4,5.
            self.assertEqual(result["playback_milliseconds"], 600)
            with Image.open(Path(temporary)/result["gif"]) as gif:
                gif.seek(gif.n_frames-1)
                np.testing.assert_array_equal(np.asarray(gif.convert("RGB"))[66:], frames[-1])
            self.assertTrue((Path(temporary)/result["contact_sheet"]).is_file())

    def test_committed_episode_preview_does_not_require_full_dataset_or_query_cache(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = resolve(ROOT/"codes/configs/collect/default.json")
            config.update(output_dir=str(root/"collection"), raw_steps=40, windows_per_episode=2,
                          splits={"validation": 1, "test": 1})
            config.pop("quality", None)
            config["policy"] = {"name": "contact-exploration"}
            config_path = root/"collection.json"
            write_json(config_path, config)
            manifest = prepare(config)
            case = manifest["cases"][0]
            path, metadata = episode_paths(config, "source", case)
            path.parent.mkdir(parents=True, exist_ok=True)
            with h5py.File(path, "w") as stream:
                stream["pixels"] = np.zeros((41, 8, 8, 3), dtype=np.uint8)
            write_json(metadata, sealed({"case": case, "regime": "source", "collection_manifest": manifest["fingerprint"],
                                        "sha256": file_hash(path), "environment": {"control_hz": 10}}))
            loaded = load_episode(config_path, "validation", 0, "source", start_step=0)
            self.assertEqual((loaded["start"], loaded["current"], loaded["goal"]), (0, 10, 35))
            self.assertFalse((path.parent.parent/"queries.json").exists())
            physical = physical_case(loaded)
            self.assertEqual(physical["episode_start_raw_row"], 0)
            self.assertLess(physical["query"]["source_episode_position"], 0)
            self.assertEqual(physical["query"]["transition_id"], physical["case_id"]+"/real-replan-0")


if __name__ == "__main__":
    unittest.main()

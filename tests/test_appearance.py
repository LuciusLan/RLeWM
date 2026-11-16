"""Model/simulator-free checks of paired labels, query rows and lineage guards."""

from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

import h5py
import numpy as np

from lewm_tta.appearance.pipeline import check
from lewm_tta.appearance.source import load_parent, specification
from lewm_tta.appearance.storage import export, write_episode
from lewm_tta.appearance.compatibility import resume_recipe
from lewm_tta.collection.manifest import sealed
from lewm_tta.runtime import write_json


class ConstantRenderer:
    description = {"test_renderer": True}

    def render(self, source, row):
        np.testing.assert_array_equal(source["state"][row], np.full(7, row, dtype=np.float32))
        return np.full((224, 224, 3), row, dtype=np.uint8)


class AppearanceTests(unittest.TestCase):
    def parent(self, directory, external=True):
        path = directory/"parent.h5"
        actions = np.arange(20, dtype=np.float64).reshape(10, 2)
        actions[4] = np.nan  # Even a terminal bit pattern must be preserved.
        with h5py.File(path, "w") as stream:
            for key, value in {"pixels": np.zeros((10, 224, 224, 3), dtype=np.uint8),
                               "state": np.repeat(np.arange(10, dtype=np.float32)[:, None], 7, axis=1), "action": actions,
                               "ep_len": [5, 5], "ep_offset": [0, 5],
                               "episode_idx": np.repeat([200, 201], 5), "step_idx": np.tile(np.arange(5), 2)}.items():
                stream.create_dataset(key, data=value)
        stat = path.stat()
        parent = sealed({"version": 2, "backend": "pusht", "external_queries": external,
                         "source": {"path": str(path), "bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns},
                         "episode_metadata": {"lengths": [5, 5], "offsets": [0, 5], "ids": [200, 201]},
                         "episodes": {"memory": [] if external else [0], "validation": [0] if external else [], "test": [1]},
                         "evaluation_windows": {"validation": {"mode": "subset", "raw_start_rows": [0], "count": 1},
                                                "test": {"mode": "subset", "raw_start_rows": [5], "count": 1}},
                         "cases": [{"case_id": "val", "episode_id": 200}, {"case_id": "test", "episode_id": 201}],
                         "data_config": {"history": 1, "frameskip": 1}, "goal_offset_steps": 2,
                         "collection_manifest": "parent-collection", "regime": "source"})
        manifest = directory/"queries.json"
        write_json(manifest, parent)
        write_json(directory/"audit.json", sealed({"passed": True, "query_manifest": parent["fingerprint"],
                   "source_dataset": "immutable-reference.h5", "source_size": 1, "source_mtime_ns": 2}))
        return manifest, parent

    def test_exact_labels_nan_terminal_and_query_remapping(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            path, parent = self.parent(root)
            _, columns, dependencies = load_parent(path, "pusht")
            for positions in ([0, 1], [1]):
                with self.subTest(positions=positions):
                    output = root/("both" if len(positions) == 2 else "test-only")
                    output.mkdir()
                    episodes = [{"parent_episode_position": p, "episode_id": 200+p, "parent_raw_start": p*5,
                                 "rows": 5, "split": "validation" if p == 0 else "test"} for p in positions]
                    recipe = sealed({"backend": "pusht", "parent_manifest": str(path),
                                     "parent_fingerprint": parent["fingerprint"], "dependencies": dependencies,
                                     "columns": columns, "episodes": episodes, "settings": {}})
                    write_json(output/"manifest.json", recipe)
                    with h5py.File(parent["source"]["path"], "r") as source:
                        for ep in episodes:
                            first = write_episode(output, recipe, ep, source, ConstantRenderer(), "gzip")
                            second = write_episode(output, recipe, ep, source, ConstantRenderer(), "gzip")
                            self.assertEqual(first, second)  # Resume preserves the committed identity.
                    value, filename = export(output, recipe, parent)
                    self.assertEqual(filename.name, "queries.json")
                    self.assertEqual(value["version"], parent["version"])
                    self.assertEqual(value["evaluation_windows"]["test"]["raw_start_rows"], [5 if len(positions) == 2 else 0])
                    self.assertEqual(value["episode_metadata"]["ids"], [200+p for p in positions])
                    self.assertEqual(value["cases"], [parent["cases"][p] for p in positions])
                    with h5py.File(output/"dataset.h5", "r") as shifted:
                        np.testing.assert_array_equal(shifted["pixels"][0], np.full((224, 224, 3), positions[0]*5, dtype=np.uint8))
                        self.assertFalse(shifted["action"].is_virtual)
                    self.assertEqual(export(output, recipe, parent)[0], value)
                    check(output)  # Exact parent bytes, shard bytes, VDS and inherited audit.

    def test_released_source_queries_are_rejected(self):
        with TemporaryDirectory() as temporary:
            path, _ = self.parent(Path(temporary), external=False)
            args = SimpleNamespace(source_manifest=path, dataset="pusht", split="test", limit_episodes=None)
            with self.assertRaisesRegex(ValueError, "Released trajectories"):
                specification(args, {})

    def test_io_resume_retains_recipe_but_rejects_renderer_changes(self):
        with TemporaryDirectory() as temporary:
            directory = Path(temporary)
            old = {"code": {"appearance/pusht.py": "same-renderer", "appearance/storage.py": "old-writer"},
                   "native_code": {"native": "same-native"}, "packages": {"package": "same-version"}}
            saved = sealed({"settings": {"object_color": [200, 70, 60]}, "implementation": old})
            write_json(directory/"manifest.json", saved)
            updated = {**old, "code": {**old["code"], "appearance/storage.py": "new-writer", "appearance/workers.py": "new-scheduler"}}
            self.assertEqual(resume_recipe(directory, sealed({**{k: v for k, v in saved.items() if k != "fingerprint"},
                                                              "implementation": updated})), saved)
            updated["code"]["appearance/pusht.py"] = "different-renderer"
            with self.assertRaisesRegex(ValueError, "renderer/inputs changed"):
                resume_recipe(directory, sealed({**{k: v for k, v in saved.items() if k != "fingerprint"},
                                                  "implementation": updated}))

    def test_batched_writes_preserve_pixels_and_read_labels_once(self):
        class CountingSource:
            def __init__(self, source):
                self.source, self.calls = source, {}

            def __getitem__(self, key):
                self.calls[key] = self.calls.get(key, 0)+1
                return self.source[key]

        with TemporaryDirectory() as temporary:
            directory = Path(temporary)
            path, parent = self.parent(directory)
            _, columns, dependencies = load_parent(path, "pusht")
            ep = {"parent_episode_position": 1, "episode_id": 201, "parent_raw_start": 5, "rows": 5, "split": "test"}
            recipe = sealed({"columns": columns, "episodes": [ep]})
            images = []
            with h5py.File(parent["source"]["path"], "r") as source:
                for size in (1, 3, 16):
                    counted = CountingSource(source)
                    record = write_episode(directory/f"batch-{size}", recipe, ep, counted, ConstantRenderer(), "gzip", batch_size=size)
                    self.assertEqual(counted.calls, {key: 1 for key in columns if key != "pixels"})
                    with h5py.File(record["path"], "r") as shifted:
                        images.append(shifted["pixels"][:])
            for pixels in images[1:]:
                np.testing.assert_array_equal(pixels, images[0])


if __name__ == "__main__":
    unittest.main()

"""User-run model-free shifted profile/reference regression tests."""

from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from experiments.catalog import profile
from experiments.shifted_profiles import make_profiles
from lewm_tta.collection.manifest import sealed
from lewm_tta.config import load_config as load_p1
from lewm_tta.runtime import file_hash, read_json, write_json
from lewm_tta.shifted.reference import load_reference, original_source_path, validate_prepared_reference


class ShiftedWorkflowTests(unittest.TestCase):
    def reference_fixture(self, root):
        dataset = root/"shifted.h5"
        dataset.write_bytes(b"immutable fixture identity; no image/model calls")
        source = sealed({"source": {"path": str(root/"released.h5")}, "data_config": {"history": 3, "frameskip": 5},
                         "episode_metadata": {"ids": [10, 11]}, "episodes": {"memory": [0], "validation": [1], "test": []}})
        prepared = sealed({"manifest": source["fingerprint"], "model_fingerprint": "same-model",
                           "normalization": {"mean": [0, 0], "std": [1, 1]},
                           "calibration": {"scales": {"context": 2, "action": 3}},
                           "calibration_settings": {"calibration_pairs": 4, "distance_floor": 1e-12}})
        memory = sealed({"source": {"path": str(dataset), "bytes": dataset.stat().st_size,
                                    "mtime_ns": dataset.stat().st_mtime_ns}, "dataset_sha256": file_hash(dataset),
                         "episode_metadata": {"ids": [10]}, "episodes": {"memory": [0], "validation": [], "test": []},
                         "appearance": {"parent_fingerprint": source["fingerprint"], "fingerprint": "appearance-id"}})
        write_json(root/"source.json", source)
        write_json(root/"prepared.json", prepared)
        write_json(root/"episodes.json", memory)
        write_json(root/"audit.json", sealed({"passed": True, "exact_labels_preserved": True,
                   "appearance_recipe": "appearance-id", "parent_fingerprint": source["fingerprint"]}))
        reference = {"prepared": str(root/"prepared.json"), "prepared_sha256": file_hash(root/"prepared.json"),
                     "prepared_fingerprint": prepared["fingerprint"], "source_manifest": str(root/"source.json"),
                     "source_fingerprint": source["fingerprint"], "memory_export": str(root/"episodes.json"),
                     "memory_export_fingerprint": memory["fingerprint"]}
        config = {"memory_reference": reference, "paths": {"dataset": str(dataset)},
                  "data": {"history": 3, "frameskip": 5, "memory_only": True}, "retrieval": prepared["calibration_settings"]}
        return config, source, prepared, memory

    def test_frozen_reference_and_original_audit_path(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            config, source, prepared, _ = self.reference_fixture(root)
            self.assertEqual(load_reference(config), prepared)
            self.assertEqual(original_source_path(config), Path(source["source"]["path"]))
            target = {**prepared, "memory_reference": config["memory_reference"]}
            validate_prepared_reference(config, target)
            target["calibration"] = {"scales": {"context": 99}}
            with self.assertRaisesRegex(ValueError, "statistics differ"):
                validate_prepared_reference(config, target)
            self.assertEqual(original_source_path({"paths": {"dataset": "ordinary-source.h5"}}), Path("ordinary-source.h5"))

    def test_parent_validation_episode_cannot_enter_memory(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            config, _, _, memory = self.reference_fixture(root)
            changed = {key: value for key, value in memory.items() if key != "fingerprint"}
            changed["episode_metadata"] = {"ids": [11]}
            changed = sealed(changed)
            write_json(root/"episodes.json", changed)
            config["memory_reference"]["memory_export_fingerprint"] = changed["fingerprint"]
            with self.assertRaisesRegex(ValueError, "non-memory"):
                load_reference(config)

    def test_profiles_share_queries_keep_stage_axes_and_preserve_source(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            p1 = load_p1(profile("p1", "default")[1])
            p1["paths"].update(dataset=str(root/"released.h5"), experiment_dir=str(root/"original-prepared"), cache_dir=str(root/"original-cache"))
            original = deepcopy(p1)
            source = {"fingerprint": "source-id", "source": {"schema": {"action": {"shape": [100, 2]}}},
                      "episodes": {"memory": [0, 1]}, "episode_metadata": {"ids": [10, 11]}}
            prepared = {"fingerprint": "prepared-id"}
            write_json(root/"original-prepared/prepared.json", prepared)
            paths = {"output": str(root/"output"), "p1_source": str(root/"output/configs/p1-source.json"),
                     "p1_shifted": str(root/"output/configs/p1-shifted.json"), "memory": str(root/"raw-memory"),
                     "queries": str(root/"raw-queries"), "memory_cache": str(root/"new-memory-cache"),
                     "query_cache": str(root/"new-query-cache"), "p1_output": str(root/"new-prepared"),
                     "source_manifest": str(root/"original-prepared/manifest.json"),
                     "source_prepared": str(root/"original-prepared/prepared.json")}
            ctx = {"name": "fixture", "paths": paths, "condition": {"dataset": "pusht"},
                   "source_p1": p1, "source_manifest": source, "source_prepared": prepared,
                   "binding": {"learning_rate": 5e-5, "seeds": [42, 43, 44], "encode_batch_size": 64,
                               "p2_methods": ["frozen", "uniform", "forward", "reverse", "fr-fixed"],
                               "p2_schedules": ["temporal-u1"]}}
            args = SimpleNamespace(memory=["source", "shifted"], jobs=["p2", "p3", "temporal"])
            def export(ctx, role, required=True):
                return {"fingerprint": role + "-id", "episode_metadata": {"ids": [10, 11]}}
            with patch("experiments.shifted_profiles.exported", side_effect=export):
                jobs, _ = make_profiles(ctx, args, {})
            self.assertEqual(p1, original)
            shifted = read_json(paths["p1_shifted"])
            self.assertTrue(shifted["data"]["memory_only"])
            self.assertEqual(shifted["data"]["validation_episodes"], 0)
            self.assertNotIn("raw_dataset", shifted["preprocessing"])
            self.assertEqual(shifted["memory_reference"]["prepared_fingerprint"], "prepared-id")
            self.assertEqual(len(jobs), 6)
            self.assertEqual(len({job["config"]["experiment_dir"] for job in jobs}), 6)
            self.assertEqual({job["config"]["query_dataset"]["cache_dir"] for job in jobs}, {paths["query_cache"]})
            for job in jobs:
                config = job["config"]
                self.assertNotIn("collection_config", config["query_dataset"])
                self.assertEqual(config["evaluation"]["records"], "compact")
                self.assertIsNone(config["evaluation"]["query_limit"])
                if job["name"] == "p3":
                    self.assertEqual(len(config["adaptation"]["methods"])*len(config["proposals"]["sources"]), 6)
                    self.assertEqual(config["reuse"], {})
                if job["name"] == "temporal":
                    self.assertEqual(config["temporal"]["modes"], ["frozen", "shared-initial", "shared-depth"])
                    self.assertEqual(config["evaluation"]["seeds"], [42, 43, 44])
                    self.assertNotIn("seeds_by_mode", config["evaluation"])


if __name__ == "__main__":
    unittest.main()

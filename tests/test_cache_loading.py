"""User-run CPU contracts. No learned model, CUDA, cuVS or simulator call.

PYTHONPATH=codes python -m unittest discover -s codes/tests -p test_cache_loading.py
"""

from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np

from lewm_tta.cache_loading import (
    cache_only, load_query_observations, observation_identity, report_fields,
    validate_query_receipts, validate_source_receipts,
)
from lewm_tta.collection.manifest import sealed
from lewm_tta.collection.queries import query_source
from lewm_tta.runtime import file_hash, object_hash, read_json, write_json


class CacheOnlyContracts(unittest.TestCase):
    def fixture(self, root):
        directory = root/"collection/source"
        directory.mkdir(parents=True)
        source = {"path": str(root/"omitted/training.h5"), "bytes": 321, "mtime_ns": 123,
                  "rows": 12, "schema": {"action": {"shape": [12, 2]}}}
        data = {"dataset_id": "cube", "history": 3, "frameskip": 5}
        memory = sealed({"source": source, "data_config": data})
        collection = sealed({"config": {}, "cases": []})
        write_json(directory.parent/"manifest.json", collection)
        query = sealed({"backend": "cube", "external_queries": True, "regime": "source",
                        "data_config": data, "episodes": {"memory": [], "validation": [0], "test": [1]},
                        "source": {**source, "path": str(root/"omitted/query.h5")},
                        "dataset_sha256": "omitted-payload", "episode_files": [
                            {"path": str(root/"omitted/episode.h5"), "sha256": "omitted-episode"}],
                        "collection_manifest": collection["fingerprint"],
                        "collection_manifest_path": str(root/"old-location/manifest.json")})
        audit = sealed({"passed": True, "collection_manifest": collection["fingerprint"],
                        "query_manifest": query["fingerprint"], "regime": "source", "backend": "cube",
                        "source_dataset": source["path"], "source_size": source["bytes"],
                        "source_mtime_ns": source["mtime_ns"]})
        write_json(directory/"audit.json", audit)
        write_json(directory/"queries.json", query)
        settings = {"manifest": str(directory/"queries.json"), "cache_dir": str(root/"cache"), "batch_size": 2}
        config = {"cache_only": True, "query_dataset": settings, "p1": {"data": data}}
        adapter = SimpleNamespace(fingerprint="saved-model", latent_dim=3, device="cpu",
                                  encode=Mock(side_effect=AssertionError("Must never encode")))
        normalization = {"mean": [0., 0.], "std": [1., 1.]}
        identity = observation_identity(adapter, query)
        frames = root/"cache"/f"observations-{object_hash(identity)[:16]}"
        frames.mkdir(parents=True)
        latents = np.arange(36, dtype=np.float32).reshape(12, 3)
        actions = np.ones((12, 2), dtype=np.float32)
        np.save(frames/"latents.npy", latents)
        action_path = frames/f"actions-{object_hash(normalization)[:16]}.npy"
        np.save(action_path, actions)
        write_json(frames/"progress.json", {"identity": identity, "complete": True, "encoded_rows": 12,
                                          "latents_sha256": file_hash(frames/"latents.npy")})
        write_json(action_path.with_suffix(".json"), {"normalization": normalization, "sha256": file_hash(action_path)})
        return SimpleNamespace(root=root, directory=directory, memory=memory, query=query, audit=audit,
                               settings=settings, config=config, adapter=adapter, normalization=normalization,
                               frames=frames, action_path=action_path, latents=latents, actions=actions)

    def load(self, f):
        return load_query_observations(f.adapter, f.query, f.normalization, Path(f.settings["cache_dir"]))

    def test_cache_mode_requires_an_explicit_boolean(self):
        self.assertFalse(cache_only({}))
        self.assertTrue(cache_only({"cache_only": True}))
        for value in ("true", 1, None):
            with self.subTest(value=value), self.assertRaises(ValueError):
                cache_only({"cache_only": value})

    def test_existing_arrays_load_without_raw_files_encoder_or_cache_writes(self):
        with tempfile.TemporaryDirectory() as tmp:
            f = self.fixture(Path(tmp))
            before = {str(p): file_hash(p) for p in f.frames.iterdir()}
            with patch("h5py.File", side_effect=AssertionError("No raw HDF5 access")):
                bank = self.load(f)
                np.testing.assert_array_equal(bank.latents.numpy(), f.latents)
                np.testing.assert_array_equal(bank.actions.numpy(), f.actions)
                bank.latents[0, 0] = 999  # copy-on-write map cannot alter the file
            self.assertEqual(before, {str(p): file_hash(p) for p in f.frames.iterdir()})
            f.adapter.encode.assert_not_called()
            self.assertFalse((f.frames/"prepare.lock").exists())

    def test_query_metadata_uses_copied_audit_without_raw_hashes(self):
        with tempfile.TemporaryDirectory() as tmp:
            f = self.fixture(Path(tmp))
            before = object_hash(f.query)
            with patch("lewm_tta.collection.queries.file_hash", side_effect=AssertionError("No raw hashing")):
                self.assertEqual(query_source(f.config, f.memory), f.query)
            self.assertEqual(object_hash(f.query), before)
            receipt = validate_query_receipts(f.settings, f.query, f.memory)
            self.assertFalse(receipt["source_raw_payloads_reverified"])
            self.assertFalse(receipt["query_raw_payloads_reverified"])

    def test_read_only_store_matches_the_existing_completed_cache_loader(self):
        from lewm_tta.observation_cache import build_or_load_observations
        with tempfile.TemporaryDirectory() as tmp:
            f = self.fixture(Path(tmp))
            with patch("h5py.File", side_effect=AssertionError("Completed caches need no raw input")):
                original = build_or_load_observations(f.adapter, f.query, f.normalization,
                                                      Path(f.settings["cache_dir"]), 2)
                cached = self.load(f)
            self.assertEqual(cached.identity, original.identity)
            np.testing.assert_array_equal(cached.latents.numpy(), original.latents.numpy())
            np.testing.assert_array_equal(cached.actions.numpy(), original.actions.numpy())
            f.adapter.encode.assert_not_called()

    def test_missing_latents_or_action_receipts_never_rebuild(self):
        for name in ("latents", "actions", "action-receipt", "progress"):
            with self.subTest(name=name), tempfile.TemporaryDirectory() as tmp:
                f = self.fixture(Path(tmp))
                path = {"latents": f.frames/"latents.npy", "actions": f.action_path,
                        "action-receipt": f.action_path.with_suffix(".json"), "progress": f.frames/"progress.json"}[name]
                path.unlink()
                with patch("h5py.File", side_effect=AssertionError("No fallback")):
                    with self.assertRaisesRegex(FileNotFoundError, "no encoding/rebuild"):
                        self.load(f)
                self.assertFalse(path.exists())
                f.adapter.encode.assert_not_called()

    def test_partial_or_wrong_identity_receipt_is_rejected(self):
        for change in ("incomplete", "rows", "identity"):
            with self.subTest(change=change), tempfile.TemporaryDirectory() as tmp:
                f = self.fixture(Path(tmp))
                path = f.frames/"progress.json"
                status = read_json(path)
                if change == "incomplete": status["complete"] = False
                if change == "rows": status["encoded_rows"] = 11
                if change == "identity": status["identity"]["model"] = "different-model"
                write_json(path, status)
                with self.assertRaisesRegex(ValueError, "incomplete|identity changed"):
                    self.load(f)

    def test_changed_array_bytes_and_normalization_are_rejected(self):
        for change in ("latent-bytes", "action-bytes", "normalization"):
            with self.subTest(change=change), tempfile.TemporaryDirectory() as tmp:
                f = self.fixture(Path(tmp))
                if change == "normalization":
                    path = f.action_path.with_suffix(".json")
                    receipt = read_json(path)
                    receipt["normalization"]["std"][0] = 2.
                    write_json(path, receipt)
                else:
                    path = f.frames/"latents.npy" if change == "latent-bytes" else f.action_path
                    values = np.load(path, mmap_mode="r+")
                    values[0, 0] += 1
                    values.flush()
                    del values
                with self.assertRaisesRegex(ValueError, "checksum|normalization"):
                    self.load(f)

    def test_shape_is_checked_even_if_payload_hash_is_consistent(self):
        with tempfile.TemporaryDirectory() as tmp:
            f = self.fixture(Path(tmp))
            np.save(f.action_path, np.zeros((11, 2), dtype=np.float32))
            write_json(f.action_path.with_suffix(".json"), {"normalization": f.normalization,
                                                          "sha256": file_hash(f.action_path)})
            with self.assertRaisesRegex(ValueError, "shape or precision"):
                self.load(f)

    def test_failed_stale_or_corrupt_prior_audit_is_rejected(self):
        for change in ("failed", "query", "source", "regime", "unsealed"):
            with self.subTest(change=change), tempfile.TemporaryDirectory() as tmp:
                f = self.fixture(Path(tmp))
                audit = {k: v for k, v in f.audit.items() if k != "fingerprint"}
                if change == "failed": audit["passed"] = False
                if change == "query": audit["query_manifest"] = "different-query"
                if change == "source": audit["source_size"] += 1
                if change == "regime": audit["regime"] = "other-shift"
                audit = sealed(audit)
                if change == "unsealed": audit["fingerprint"] = "corrupt"
                write_json(f.directory/"audit.json", audit)
                with self.assertRaises(ValueError):
                    query_source(f.config, f.memory)

    def test_default_mode_still_requires_raw_query_payloads(self):
        with tempfile.TemporaryDirectory() as tmp:
            f = self.fixture(Path(tmp))
            with self.assertRaisesRegex(FileNotFoundError, "Fresh raw"):
                query_source({**f.config, "cache_only": False}, f.memory)

    def test_cache_only_does_not_bypass_task_or_horizon_contracts(self):
        with tempfile.TemporaryDirectory() as tmp:
            f = self.fixture(Path(tmp))
            with self.assertRaisesRegex(ValueError, "backend"):
                query_source({**f.config, "p1": {"data": {**f.config['p1']['data'], "dataset_id": "reacher"}}},
                             {**f.memory, "data_config": {**f.memory['data_config'], "dataset_id": "reacher"}})
            query = {k: v for k, v in f.query.items() if k != "fingerprint"}
            query["goal_offset_steps"] = 100
            write_json(f.directory/"queries.json", sealed(query))
            with self.assertRaisesRegex(ValueError, "horizon"):
                query_source({**f.config, "planning": {"horizon": 5}}, f.memory)

    def test_simulator_cache_mode_still_requires_fresh_raw_payloads(self):
        from lewm_tta.outcomes.replay import FreshReplay
        with tempfile.TemporaryDirectory() as tmp:
            f = self.fixture(Path(tmp))
            with patch("lewm_tta.outcomes.replay.SourceDataset") as reader:
                with self.assertRaisesRegex(FileNotFoundError, "simulator replay"):
                    FreshReplay(f.query, f.config, {})
                reader.assert_not_called()

    def test_source_receipts_require_complete_matching_p1_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            f = self.fixture(Path(tmp))
            observed = {"observation_fingerprint": object_hash(observation_identity(f.adapter, f.query)),
                        "raw_frames": 12, "latents_sha256": file_hash(f.frames/"latents.npy"),
                        "actions_sha256": file_hash(f.action_path)}
            prepared = {"memory_cache": {"observations": observed}, "normalization": f.normalization,
                        "model_fingerprint": f.adapter.fingerprint}
            config = {"paths": {"cache_dir": f.settings["cache_dir"]}}
            validate_source_receipts(config, {"source": f.query["source"]}, prepared)
            prepared["model_fingerprint"] = "wrong"
            with self.assertRaisesRegex(ValueError, "P1 observation/action receipts"):
                validate_source_receipts(config, {"source": f.query["source"]}, prepared)

    def test_config_and_report_mode_are_explicit_and_default_is_unchanged(self):
        from experiments.config import resolve
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/"profile.json"
            write_json(path, {"setting": "unchanged"})
            self.assertEqual(resolve(path), {"setting": "unchanged"})
            self.assertEqual(resolve(path, cache_only=True), {"setting": "unchanged", "cache_only": True})
        self.assertEqual(report_fields(SimpleNamespace()), {})
        fields = report_fields(SimpleNamespace(cache_only=True, loading_provenance={"mode": "cache-only"}), replay=True)
        self.assertTrue(fields["data_loading"]["query_raw_payloads_reverified"])


if __name__ == "__main__":
    unittest.main()

"""User-run cache relocation contracts; no encoder, predictor, CUDA or cuVS.

PYTHONPATH=codes python -m unittest discover -s codes/tests -p test_cache_relocation.py
"""

from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from lewm_tta.cache_relocation import index_array_paths, make_prepared_retriever
from lewm_tta.runtime import file_hash, object_hash, write_json


class CacheRelocationContracts(unittest.TestCase):
    def fixture(self, root):
        identity = {"memory_cache": {"fingerprint": "unchanged"}, "version": 1}
        directory = root/"relocated/retrieval"/f"cuvs-{object_hash(identity)[:16]}"
        directory.mkdir(parents=True)
        arrays = {}
        for direction in ("forward", "reverse"):
            path = directory/f"{direction}.npy"
            np.save(path, np.zeros((3, 2), dtype=np.float32))
            arrays[direction] = {"path": f"/old-nvme/retrieval/{directory.name}/{direction}.npy",
                                "shape": [3, 2], "file_bytes": path.stat().st_size, "sha256": file_hash(path)}
        record = {"identity": identity, "arrays": arrays,
                  "fingerprint": object_hash({"identity": identity, "arrays": arrays}), "numeric_payload_bytes": 48}
        return root/"relocated", directory, record

    def test_current_root_overrides_stale_paths_without_modifying_metadata(self):
        with tempfile.TemporaryDirectory() as temporary:
            root, directory, record = self.fixture(Path(temporary))
            before = object_hash(record)
            paths = index_array_paths(root, record)
            self.assertEqual(paths, {name: directory/f"{name}.npy" for name in ("forward", "reverse")})
            self.assertEqual(object_hash(record), before)
            self.assertTrue(record["arrays"]["forward"]["path"].startswith("/old-nvme/"))

    def test_missing_local_payload_does_not_fall_back_to_the_recorded_path(self):
        with tempfile.TemporaryDirectory() as temporary:
            root, directory, record = self.fixture(Path(temporary))
            (directory/"reverse.npy").unlink()
            with self.assertRaisesRegex(FileNotFoundError, "configured cache"):
                index_array_paths(root, record)

    def test_modified_manifest_and_truncated_payload_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root, directory, record = self.fixture(Path(temporary))
            record["fingerprint"] = "corrupt"
            with self.assertRaisesRegex(ValueError, "fingerprint"):
                index_array_paths(root, record)
            record["fingerprint"] = object_hash({"identity": record["identity"], "arrays": record["arrays"]})
            (directory/"forward.npy").write_bytes(b"truncated")
            with self.assertRaisesRegex(ValueError, "size changed"):
                index_array_paths(root, record)

    def test_non_cuvs_path_keeps_the_existing_loader(self):
        with patch("lewm_tta.retrieval.make_retriever", return_value="original") as original:
            result = make_prepared_retriever([1], {}, {"backend": "torch"}, Path("unused"))
            self.assertEqual(result, "original")
            original.assert_called_once()

    def prepared_fixture(self, root):
        from lewm_tta.indexed_keys import DIRECTIONS, KeyLayout, key_identity
        class Memory:
            split = "memory"
            identity = {"fingerprint": "synthetic-memory"}
            contexts = np.zeros((3, 1, 2), dtype=np.float32)
            past_actions = np.zeros((3, 0, 2), dtype=np.float32)
            actions = np.zeros((3, 2), dtype=np.float32)
            successors = np.zeros((3, 2), dtype=np.float32)

            def __len__(self):
                return 3
        memory = Memory()
        settings = {"backend": "cuvs", "beta_action": 1., "beta_successor": 1., "device_reserve_gib": 0}
        calibration = {"scales": dict.fromkeys(("context", "past_actions", "action", "successor"), 1.)}
        layout = KeyLayout.from_memory(memory, calibration, settings)
        identity = key_identity(memory, layout)
        directory = root/"retrieval"/f"cuvs-{object_hash(identity)[:16]}"
        directory.mkdir(parents=True)
        arrays = {}
        for direction in DIRECTIONS:
            path = directory/f"{direction}.npy"
            np.save(path, np.zeros((3, layout.width(direction)), dtype=np.float32))
            arrays[direction] = {"path": f"/old-nvme/{directory.name}/{direction}.npy", "shape": [3, layout.width(direction)],
                                "file_bytes": path.stat().st_size, "sha256": file_hash(path)}
        record = {"identity": identity, "arrays": arrays,
                  "fingerprint": object_hash({"identity": identity, "arrays": arrays}),
                  "numeric_payload_bytes": sum(3*layout.width(direction)*4 for direction in DIRECTIONS)}
        write_json(directory/"index.json", record)
        return memory, calibration, settings, directory, record

    def test_relocated_loader_preserves_assets_and_uploads_local_payloads(self):
        import torch
        class FakeRetriever:
            def __init__(self, memory, calibration, settings, directory, *, tables):
                assert tables == {}
                self.runtime = SimpleNamespace(device="cpu", build=lambda table: ("index", table.shape))
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            memory, calibration, settings, directory, record = self.prepared_fixture(root)
            before = file_hash(directory/"index.json")
            with patch("lewm_tta.cuvs_retrieval.CuVSRetriever", FakeRetriever), \
                    patch("torch.cuda.empty_cache"), patch("torch.cuda.mem_get_info", return_value=(2**30, 2**30)):
                retriever = make_prepared_retriever(memory, calibration, settings, root)
            self.assertEqual(retriever.assets, record)
            self.assertEqual(file_hash(directory/"index.json"), before)
            for direction in ("forward", "reverse"):
                torch.testing.assert_close(retriever.keys[direction], torch.from_numpy(np.load(directory/f"{direction}.npy")))
                self.assertEqual(retriever.resolved_asset_paths[direction], str(directory/f"{direction}.npy"))

    def test_same_size_content_corruption_is_rejected_before_native_runtime(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            memory, calibration, settings, directory, _ = self.prepared_fixture(root)
            values = np.load(directory/"forward.npy", mmap_mode="r+")
            values[0, 0] = 1
            values.flush()
            del values
            with patch("lewm_tta.cuvs_retrieval.CuVSRetriever") as native:
                with self.assertRaisesRegex(ValueError, "Invalid relocated forward"):
                    make_prepared_retriever(memory, calibration, settings, root)
                native.assert_not_called()


if __name__ == "__main__":
    unittest.main()

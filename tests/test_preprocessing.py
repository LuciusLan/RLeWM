"""User-run metadata/extraction tests; no checkpoint loading or model execution."""

import io
from pathlib import Path
import shutil
import subprocess
import tarfile
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from experiments.config import resolve
from experiments.preprocess import select_configs
from lewm_tta.data import action_statistics, get_manifest, inspect_source
from lewm_tta.preprocessing.archives import archive_header, extract_hdf5
from lewm_tta.preprocessing.banks import SourceWindowItems
from lewm_tta.preprocessing.inspection import cache_bytes, inspect_task
from lewm_tta.preprocessing.hdf5 import action_padding, prepare_source_view, raw_dataset_path
from lewm_tta.preprocessing.storage import GIB, require_space, storage_plan
from lewm_tta.runtime import read_json, write_json

ROOT = Path(__file__).resolve().parents[2]


class PreprocessingTests(unittest.TestCase):
    def test_profiles_are_independent_memory_only_and_checkpoint_specific(self):
        for name, width in (("tworooms", 2), ("reacher", 2), ("cube", 5)):
            config = resolve(ROOT / f"codes/configs/p1/{name}.json")
            self.assertTrue(config["data"]["memory_only"])
            self.assertEqual(config["data"]["dataset_id"], name)
            self.assertEqual(config["data"]["validation_episodes"], 0)
            self.assertEqual(config["data"]["test_episodes"], 0)
            self.assertEqual(config["data"]["historical_inspected_episodes"], [])
            self.assertEqual(config["preprocessing"]["raw_action_dim"], width)
            self.assertIn(f"lewm-{name}", config["paths"]["checkpoint"])
            self.assertEqual(config["paths"]["cache_dir"], f"/home/ubuntu/lewm_tta/{name}")
            self.assertEqual(Path(config["paths"]["dataset"]).name, "source.h5")
            self.assertTrue(str(raw_dataset_path(config)).startswith("/opt/dlami/nvme/ubuntu/"))

    def test_cube_key_width_includes_five_ordered_action_coordinates(self):
        result = cache_bytes(200, 181, 3, 192, 5, 5)
        self.assertEqual(result["key_dimensions"], {"forward": 651, "reverse": 818})
        self.assertEqual(result["total_bytes"], 200*(192+5)*4 + 181*(651+818)*4 + 181*8)

    def test_path_overrides_are_task_specific(self):
        args = SimpleNamespace(config=None, datasets=["reacher", "cube"], cache_root=Path("/tmp/cache"),
                               output_root=Path("/tmp/meta"), scratch_root=Path("/tmp/raw"))
        reacher, cube = select_configs(args)
        self.assertEqual(reacher["paths"]["cache_dir"], "/tmp/cache/reacher")
        self.assertEqual(cube["paths"]["dataset"], "/tmp/meta/cube/source.h5")
        self.assertEqual(str(raw_dataset_path(cube)), "/tmp/raw/lewm_cube/cube_single_expert.h5")

    def test_metadata_ids_and_slice_keep_task_namespace(self):
        import numpy as np
        manifest = {"data_config": {"dataset_id": "cube", "history": 3, "frameskip": 5},
                    "source": {"path": "source.h5"}, "episode_metadata": {"offsets": [0, 30], "ids": [7, 19]}}
        items = SourceWindowItems(manifest, np.array([1, 30]))
        self.assertEqual(items[0]["transition_id"], "cube:7:1")
        self.assertEqual(items[:][1]["transition_id"], "cube:19:0")
        self.assertEqual(items[-1]["candidate_action_rows"], list(range(40, 45)))

    def test_storage_groups_two_directories_on_same_filesystem(self):
        with tempfile.TemporaryDirectory() as temporary:
            jobs = []
            for name in ("reacher", "cube"):
                config = {"paths": {"dataset": f"{temporary}/scratch/{name}.h5", "cache_dir": f"{temporary}/cache/{name}"}}
                item = {"dataset": name, "source": None, "archive": {}, "extraction_bytes": 3*GIB,
                        "cache": {"total_bytes": 2*GIB}}
                jobs.append((config, item))
            with patch("lewm_tta.preprocessing.storage.shutil.disk_usage", return_value=SimpleNamespace(free=19*GIB)):
                groups = storage_plan(jobs, keep_free_gib=6, scratch_keep_free_gib=10)
            self.assertEqual(len(groups), 1)
            self.assertGreater(groups[0]["additional_bytes"], 10*GIB)
            self.assertFalse(groups[0]["sufficient"])
            with self.assertRaisesRegex(ValueError, "Insufficient space"):
                require_space(groups)

    def fixture(self, directory, *, native_schema=False, terminal_padding=False):
        import h5py
        import numpy as np
        config = resolve(ROOT / "codes/configs/p1/reacher.json")
        checkpoint = directory / "checkpoint"
        checkpoint.mkdir()
        # Metadata checks never deserialize weights.
        (checkpoint / "weights.pt").write_bytes(b"not-loaded")
        write_json(checkpoint / "config.json", {"encoder": {"image_size": 224, "pretrained": False},
            "projector": {"output_dim": 192}, "predictor": {"num_frames": 3}, "action_encoder": {"input_dim": 10}})
        source = directory / "original.h5"
        with h5py.File(source, "w") as handle:
            handle["ep_len"] = np.array([30, 30], dtype=np.int64)
            handle["ep_offset"] = np.array([0, 30], dtype=np.int64)
            handle["pixels"] = np.zeros((60, 8, 8, 3), dtype=np.uint8)
            actions = np.arange(120, dtype=np.float64).reshape(60, 2) / 17
            if terminal_padding:
                actions[[29, 59]] = np.nan
            handle["action"] = actions
            handle["ep_idx" if native_schema else "episode_idx"] = np.repeat([7, 19], 30)
            handle["step_idx"] = np.tile(np.arange(30), 2)
            handle["qpos"] = np.zeros((60, 2), dtype=np.float64)
        archive = directory / "reacher.tar.zst"
        data = io.BytesIO()
        with tarfile.open(fileobj=data, mode="w") as tar:
            tar.add(source, arcname="reacher.h5")
        archive.write_bytes(subprocess.run(["zstd", "-q", "-c"], input=data.getvalue(),
                                           check=True, capture_output=True).stdout)
        config["paths"].update(checkpoint=str(checkpoint), dataset=str(directory / "metadata/source.h5"),
                               experiment_dir=str(directory / "metadata"), cache_dir=str(directory / "cache"))
        config["preprocessing"]["archive"] = str(archive)
        config["preprocessing"]["raw_dataset"] = str(directory / "raw/reacher.h5")
        config["evaluation"]["log_every"] = 0
        return config, source

    @unittest.skipUnless(shutil.which("zstd"), "zstd unavailable")
    def test_extract_metadata_and_memory_manifest_without_model(self):
        with tempfile.TemporaryDirectory() as temporary:
            config, original = self.fixture(Path(temporary))
            before = inspect_task(config)
            self.assertTrue(before["estimate_before_extraction"])
            extract_hdf5(config, before["archive"])
            self.assertEqual(raw_dataset_path(config).read_bytes(), original.read_bytes())
            receipt = read_json(Path(config["paths"]["experiment_dir"])/"source-extraction.json")
            self.assertTrue(receipt["sha256"])
            after = inspect_task(config)
            self.assertFalse(after["estimate_before_extraction"])
            self.assertEqual(after["source_summary"]["episodes"], 2)
            self.assertEqual(after["source_summary"]["rows"], 60)
            self.assertIn("qpos", after["source_summary"]["columns"])
            prepare_source_view(config)
            manifest = get_manifest(config)
            self.assertEqual(manifest["episodes"], {"memory": [0, 1], "validation": [], "test": []})
            self.assertEqual(manifest["windows"]["memory"]["count"], 22)
            self.assertEqual(manifest["evaluation_windows"]["test"]["count"], 0)
            self.assertEqual(manifest, get_manifest(config))
            extract_hdf5(config, after["archive"])

    @unittest.skipUnless(shutil.which("zstd"), "zstd unavailable")
    def test_owned_interrupted_extraction_can_restart(self):
        with tempfile.TemporaryDirectory() as temporary:
            config, original = self.fixture(Path(temporary))
            header = archive_header(config["preprocessing"]["archive"], "reacher.h5")
            target = raw_dataset_path(config)
            target.parent.mkdir()
            target.with_name(target.name+".partial").write_bytes(b"unfinished")
            write_json(target.with_name(target.name+".extraction.json"), {"status": "extracting", "header": header})
            extract_hdf5(config, header)
            self.assertEqual(target.read_bytes(), original.read_bytes())

    @unittest.skipUnless(shutil.which("zstd"), "zstd unavailable")
    def test_native_alias_and_terminal_padding_preserve_source_and_statistics(self):
        import h5py
        import numpy as np
        with tempfile.TemporaryDirectory() as temporary:
            config, _ = self.fixture(Path(temporary), native_schema=True, terminal_padding=True)
            extract_hdf5(config, inspect_task(config)["archive"])
            raw = raw_dataset_path(config)
            before = raw.stat()
            inspection = inspect_task(config)
            self.assertEqual(inspection["source_summary"]["metadata_columns"]["episode_idx"], "ep_idx")
            self.assertEqual(inspection["source_summary"]["action_audit"]["terminal_padding_rows"], 2)
            source_identity, _, _, ids = inspect_source(raw)
            self.assertEqual(ids.tolist(), [7, 19])
            self.assertEqual(source_identity["metadata_columns"]["episode_idx"], "ep_idx")
            prepare_source_view(config)
            view_path = Path(config["paths"]["dataset"])
            view_before = view_path.stat()
            prepare_source_view(config)
            self.assertEqual(view_path.stat().st_mtime_ns, view_before.st_mtime_ns)
            self.assertEqual((raw.stat().st_size, raw.stat().st_mtime_ns), (before.st_size, before.st_mtime_ns))
            with h5py.File(raw, "r") as source, h5py.File(view_path, "r") as view:
                self.assertNotIn("episode_idx", source)
                self.assertTrue(np.isnan(source["action"][[29, 59]]).all())
                self.assertIsInstance(view.get("pixels", getlink=True), h5py.ExternalLink)
                self.assertEqual(view["episode_idx"][[0, 30]].tolist(), [7, 19])
                valid = view["action_valid"][:]
                self.assertEqual(np.flatnonzero(~valid).tolist(), [29, 59])
                self.assertTrue(np.isfinite(view["action"][:]).all())
                expected = source["action"][:][valid]
            manifest = get_manifest(config)
            stats = action_statistics(manifest)
            self.assertEqual(stats["raw_rows"], 58)
            self.assertEqual(stats["excluded_terminal_action_rows"], 2)
            np.testing.assert_allclose(stats["mean"], expected.mean(0), rtol=1e-12)
            np.testing.assert_allclose(stats["std"], expected.std(0, ddof=1), rtol=1e-12)
            starts = np.r_[np.arange(11), 30+np.arange(11)]
            for item in SourceWindowItems(manifest, starts):
                self.assertTrue(valid[item["candidate_action_rows"]].all())
            # Exercise the actual native loader with canonical keys, without
            # constructing an encoder, predictor or adaptation engine.
            from stable_worldmodel.data import HDF5Dataset
            dataset = HDF5Dataset(path=str(view_path), frameskip=1, num_steps=1000,
                                  keys_to_load=["pixels", "episode_idx", "step_idx"])
            try:
                chunks = dataset.load_chunk(np.array([0, 1]), np.array([27, 0]), np.array([30, 3]))
                self.assertEqual(chunks[0]["episode_idx"].tolist(), [7, 7, 7])
                self.assertEqual(chunks[1]["step_idx"].tolist(), [0, 1, 2])
                self.assertEqual(tuple(chunks[0]["pixels"].shape), (3, 3, 8, 8))
            finally:
                if dataset.h5_file is not None:
                    dataset.h5_file.close()

    @unittest.skipUnless(shutil.which("zstd"), "zstd unavailable")
    def test_nonterminal_nan_actions_are_not_silently_repaired(self):
        import h5py
        import numpy as np
        with tempfile.TemporaryDirectory() as temporary:
            _, source = self.fixture(Path(temporary), native_schema=True, terminal_padding=True)
            with h5py.File(source, "r+") as handle:
                handle["action"][2] = np.nan
                with self.assertRaisesRegex(ValueError, "Nonfinite actions outside"):
                    action_padding(handle, handle["ep_len"][:], handle["ep_offset"][:])

    @unittest.skipUnless(shutil.which("zstd"), "zstd unavailable")
    def test_reject_archive_traversal_and_wrong_member(self):
        with tempfile.TemporaryDirectory() as temporary:
            data = io.BytesIO()
            with tarfile.open(fileobj=data, mode="w") as tar:
                entry = tarfile.TarInfo("../outside.h5")
                entry.size = 1
                tar.addfile(entry, io.BytesIO(b"x"))
            path = Path(temporary) / "unsafe.tar.zst"
            path.write_bytes(subprocess.run(["zstd", "-q", "-c"], input=data.getvalue(), check=True, capture_output=True).stdout)
            with self.assertRaisesRegex(ValueError, "Expected a regular archive member"):
                archive_header(path, "reacher.h5")


if __name__ == "__main__":
    unittest.main()

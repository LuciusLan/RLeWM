"""Metadata-only regression checks for episode leakage and immutable split profiles."""

from copy import deepcopy
from pathlib import Path
import tempfile
import unittest

from lewm_tta.runtime import object_hash, read_json, write_json
from lewm_tta.splits.episodes import collect_exposure, resplit_manifest, validate_parent, write_profile


def fixture():
    lengths = [40] * 14
    offsets = [40 * ep for ep in range(len(lengths))]
    ids = [100 + ep for ep in range(len(lengths))]
    data = {"history": 3, "frameskip": 5, "memory_windows": "all", "split_seed": 17,
            "validation_episodes": 2, "test_episodes": 2, "windows_per_query_episode": 4}
    parent = {"version": 2, "layout": "shared observation latents with compact window indices",
              "source": {"rows": sum(lengths), "episodes": len(lengths),
                         "metadata_sha256": object_hash([lengths, offsets, ids])},
              "episode_metadata": {"lengths": lengths, "offsets": offsets, "ids": ids},
              "data_config": data, "episodes": {"memory": list(range(4, 14)), "validation": [0, 1], "test": [2, 3]},
              "checkpoint_overlap": "unknown"}
    parent["fingerprint"] = object_hash(parent)
    return parent


class EpisodeSplitTests(unittest.TestCase):
    def test_preserves_validation_excludes_prior_queries_and_has_no_overlap(self):
        parent = fixture()
        original = deepcopy(parent)
        data = dict(parent["data_config"], test_episodes=4)
        manifest, audit = resplit_manifest(parent, data, {4, 5}, Path("parent.json"))
        self.assertEqual(parent, original)
        self.assertEqual(manifest["episodes"]["validation"], [0, 1])
        self.assertTrue(set(manifest["episodes"]["test"]).isdisjoint({0, 1, 2, 3, 4, 5}))
        self.assertEqual(sorted(sum(manifest["episodes"].values(), [])), list(range(14)))
        self.assertEqual(audit["episode_counts"], {"memory": 8, "validation": 2, "test": 4})
        self.assertEqual(audit["new_test_previously_in_memory"], 4)

    def test_reproducible_windows_are_unique_and_cannot_cross_terminals(self):
        parent = fixture()
        first, _ = resplit_manifest(parent, parent["data_config"], set(), Path("parent.json"))
        second, _ = resplit_manifest(parent, parent["data_config"], set(), Path("parent.json"))
        self.assertEqual(first, second)
        for split in ("validation", "test"):
            starts = first["evaluation_windows"][split]["raw_start_rows"]
            self.assertEqual(len(starts), 4 * len(first["episodes"][split]))
            self.assertEqual(len(starts), len(set(starts)))
            for raw in starts:
                ep, start = divmod(raw, 40)
                self.assertIn(ep, first["episodes"][split])
                self.assertLessEqual(start + 20, 40)

    def test_rejects_corrupt_metadata_and_changed_validation_or_temporal_contract(self):
        parent = fixture()
        corrupt = deepcopy(parent)
        corrupt["episode_metadata"]["offsets"][1] += 1
        corrupt["fingerprint"] = object_hash({k: v for k, v in corrupt.items() if k != "fingerprint"})
        with self.assertRaises(ValueError):
            validate_parent(corrupt)
        for override in ({"validation_episodes": 3}, {"frameskip": 4}):
            with self.assertRaises(ValueError):
                resplit_manifest(parent, dict(parent["data_config"], **override), set(), Path("parent.json"))
        with self.assertRaises(ValueError):
            resplit_manifest(parent, parent["data_config"], set(range(14)), Path("parent.json"))

    def test_all_window_profile_and_short_episode_query_shortfall(self):
        parent = fixture()
        manifest, _ = resplit_manifest(parent, dict(parent["data_config"], windows_per_query_episode="all"), set(), Path("p.json"))
        self.assertEqual(manifest["evaluation_windows"]["test"], {"mode": "all", "count": 42})
        manifest, _ = resplit_manifest(parent, dict(parent["data_config"], windows_per_query_episode=30), set(), Path("p.json"))
        self.assertEqual(manifest["evaluation_windows"]["test"]["count"], 42)

    def test_exposure_reads_queries_and_reserved_starts_without_support_ids(self):
        parent = fixture()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_json(root / "report.json", {"query_episode_positions": [4], "records": [
                {"source": {"source_episode_position": 5}, "rounds": [
                    {"support": [{"source_episode_position": 6}]}]}]})
            write_json(root / "manifest.json", {"splits": {"test": {"raw_start_rows": [40 * 7 + 3]}}})
            data = {"inspected_reports": [str(root / "report.json")], "inspected_report_globs": [],
                    "historical_inspected_episodes": [8]}
            excluded, sources = collect_exposure(parent, data, [str(root / "manifest.json")])
            self.assertTrue({0, 1, 2, 3, 4, 5, 7, 8} <= excluded)
            self.assertNotIn(6, excluded)
            self.assertEqual(len(sources), 2)

    def test_profile_refuses_overwrite_or_existing_experiment(self):
        parent = fixture()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data = dict(parent["data_config"], split_from=str(root / "split/manifest.json"))
            manifest, audit = resplit_manifest(parent, data, set(), Path("parent.json"))
            config = {"data": data, "paths": {"experiment_dir": str(root / "experiment")}}
            write_profile(config, manifest, audit, root / "config.json", root / "split")
            self.assertEqual(read_json(root / "split/manifest.json"), manifest)
            with self.assertRaises(FileExistsError):
                write_profile(config, manifest, audit, root / "config.json", root / "split")


if __name__ == "__main__":
    unittest.main()

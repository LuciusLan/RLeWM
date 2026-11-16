"""User-run synthetic fresh-suite contracts; no checkpoint or simulator.

PYTHONPATH=codes python -m unittest discover -s codes/tests -p test_fresh_workflow.py
"""

from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from experiments.config import query_dataset_settings
from experiments.fresh import completed_report, query_ids, verify_lock
from lewm_tta.collection.manifest import sealed
from lewm_tta.runtime import file_hash, write_json


class FreshWorkflowContracts(unittest.TestCase):
    def query(self):
        return {"external_queries": True, "fingerprint": "query-manifest", "regime": "source",
                "collection_manifest": "reserved-collection", "data_config": {"history": 3, "frameskip": 5},
                "episodes": {"memory": [], "validation": [0], "test": [1]},
                "episode_metadata": {"offsets": [0, 100], "lengths": [100, 100], "ids": [1000000000, 1000000001]},
                "evaluation_windows": {"validation": {"mode": "subset", "raw_start_rows": [13, 19]},
                                       "test": {"mode": "subset", "raw_start_rows": [103, 119]}}}

    def test_query_paths_follow_the_collection_output_and_regime(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            collection = root / "collector.json"
            write_json(collection, {"output_dir": str(root / "moved-data"), "regimes": {"source": {}, "shape-I": {}}})
            settings = query_dataset_settings({"collection_config": str(collection), "regime": "shape-I"})
            self.assertEqual(settings["manifest"], str(root / "moved-data/shape-I/queries.json"))
            self.assertEqual(settings["cache_dir"], str(root / "moved-data/shape-I/cache"))
            with self.assertRaises(ValueError):
                query_dataset_settings({"collection_config": str(collection), "manifest": "stale.json"})

    def test_reserved_windows_survive_stage_seed_changes(self):
        from lewm_tta.rollout.cohorts import sample_cohort
        query = self.query()
        first = sample_cohort(query, "validation", 5, {"windows_per_episode": 2, "seed": 0})
        second = sample_cohort(query, "validation", 5, {"windows_per_episode": 2, "seed": 900})
        self.assertEqual(first["raw_start_rows"], [13, 19])
        self.assertEqual(first, second)
        with self.assertRaises(ValueError):
            sample_cohort(query, "validation", 5, {"windows_per_episode": 4, "seed": 0})
        with self.assertRaises(ValueError):
            sample_cohort(query, "validation", 20, {"windows_per_episode": 2, "seed": 0})

    def fixture(self, root):
        query = self.query()
        config = {"experiment_dir": str(root), "p1": {"evaluation": {"seeds": [42]}},
                  "scopes": ["last-block"], "schedule_names": ["initial-u1"],
                  "adaptation": {"methods": ["frozen", "uniform"]}, "evaluation": {"query_limit": None}}
        job = {"name": "p2", "backend": "p2_followup", "report_stage": "P2-followup-iteration", "config": config}
        cohort = {"raw_start_rows": [13, 19], "episodes": 1, "queries": 2}
        manifest = sealed({"splits": {"validation": cohort}, "query_source_fingerprint": query["fingerprint"]})
        write_json(root / "manifest.json", manifest)
        path = root / "runs/validation/report.json"
        path.parent.mkdir(parents=True)
        records = path.with_name("records.jsonl")
        rows = [{"query": {"transition_id": item}, "seed": 42, "scope": "last-block",
                 "schedule": "initial-u1", "method": method}
                for item in sorted(query_ids(query, [13, 19])) for method in ("frozen", "uniform")]
        records.write_text("".join(json.dumps(row)+"\n" for row in rows))
        report = {"stage": job["report_stage"], "completed": True, "limited_run": False,
                  "split": "validation", "config": config, "manifest_fingerprint": manifest["fingerprint"],
                  "cohort": cohort, "code": {"engine.py": "original"}, "record_count": len(rows),
                  "records_file": str(records), "records_sha256": file_hash(records)}
        write_json(path, report)
        return query, job, path, report, rows

    def test_completed_matrix_rejects_duplicate_or_wrong_query_records(self):
        with tempfile.TemporaryDirectory() as temporary:
            query, job, path, report, rows = self.fixture(Path(temporary))
            with patch("experiments.fresh.effective_config", side_effect=deepcopy), patch(
                    "experiments.fresh.backend_identity", return_value=report["code"]):
                self.assertEqual(completed_report(path, job, query, "validation"), report)
                for replacement in (rows[0], {**rows[-1], "query": {"transition_id": "different-case"}}):
                    changed = rows[:-1]+[replacement]
                    Path(report["records_file"]).write_text("".join(json.dumps(row)+"\n" for row in changed))
                    report["records_sha256"] = file_hash(Path(report["records_file"]))
                    write_json(path, report)
                    with self.assertRaises(ValueError):
                        completed_report(path, job, query, "validation")

    def test_limited_validation_and_partial_runs_cannot_be_reused_as_complete(self):
        with tempfile.TemporaryDirectory() as temporary:
            query, job, path, report, _ = self.fixture(Path(temporary))
            report["limited_run"] = True
            write_json(path, report)
            with patch("experiments.fresh.effective_config", side_effect=deepcopy), self.assertRaises(ValueError):
                completed_report(path, job, query, "validation")
            path.unlink()
            with self.assertRaises(ValueError):
                completed_report(path, job, query, "validation")

    def test_lock_requires_the_exact_validation_report(self):
        with tempfile.TemporaryDirectory() as temporary:
            _, _, report_path, report, _ = self.fixture(Path(temporary))
            path = report_path.parent / "lock.json"
            write_json(path, sealed({"stage": report["stage"], "config": report["config"], "code": report["code"],
                                    "manifest_fingerprint": report["manifest_fingerprint"],
                                    "validation_report": str(report_path), "report_sha256": file_hash(report_path)}))
            self.assertTrue(verify_lock(path, report_path, report))
            report_path.write_text(report_path.read_text()+"\n")
            with self.assertRaises(ValueError):
                verify_lock(path, report_path, report)

    def test_established_p2_lock_has_no_self_fingerprint(self):
        with tempfile.TemporaryDirectory() as temporary:
            _, _, report_path, report, _ = self.fixture(Path(temporary))
            report["summary"] = {"accurate_thresholds": {"1": .01, "3": .02, "5": .03}}
            write_json(report_path, report)
            path = report_path.parent / "p2-lock.json"
            lock = {"stage": report["stage"], "config": report["config"], "code": report["code"],
                    "manifest_fingerprint": report["manifest_fingerprint"],
                    "validation_report": str(report_path), "report_sha256": file_hash(report_path),
                    "accurate_thresholds": report["summary"]["accurate_thresholds"]}
            write_json(path, lock)
            self.assertTrue(verify_lock(path, report_path, report))
            lock["accurate_thresholds"] = {"1": .1, "3": .2, "5": .3}
            write_json(path, lock)
            with self.assertRaises(ValueError):
                verify_lock(path, report_path, report)
            lock["stage"] = "P3c-offline-factorial"
            write_json(path, lock)
            with self.assertRaises(ValueError):
                verify_lock(path, report_path, report)


if __name__ == "__main__":
    unittest.main()

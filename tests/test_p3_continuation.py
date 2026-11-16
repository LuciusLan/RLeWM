"""User-run continuation contracts; no model or simulator evaluation."""

from copy import deepcopy
import gzip
import importlib.util
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import test_p3_recovery as recovery_fixtures


SCRIPT = Path(__file__).resolve().parents[1]/"continue-p3-core.py"
SPEC = importlib.util.spec_from_file_location("p3_continuation", SCRIPT)
continuation = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(continuation)
recovery = continuation.recovery


class ContinuationTests(unittest.TestCase):
    def fixture(self, directory):
        run, config, _, streams = recovery_fixtures.RecoveryTests().fixture(directory)
        prepared_root = Path(directory)/"p1"
        prepared_root.mkdir()
        prepared = {"model": {"weights": "test-checkpoint"}}
        prepared["fingerprint"] = recovery.object_digest(prepared)
        recovery.write_json(prepared_root/"prepared.json", prepared)
        config["p1"]["paths"] = {"experiment_dir": str(prepared_root)}
        recovery.write_json(run/"config.json", config)
        manifest_path = Path(config["experiment_dir"])/"manifest.json"
        manifest = recovery.read_json(manifest_path)
        manifest["p1_prepared_fingerprint"] = prepared["fingerprint"]
        manifest.pop("fingerprint")
        manifest["fingerprint"] = recovery.object_digest(manifest)
        recovery.write_json(manifest_path, manifest)
        with patch("sys.stdout", new_callable=io.StringIO):
            partial = recovery.recover(run)
        path = run/"recovered-partial"/"report.json"
        return continuation.load_partial(path), streams

    def segment_fixture(self, source, streams):
        from lewm_tta.core.reporting import summarize
        original = source["report"]
        identity = list(source["expected"])[-1]
        first_id = next(iter(source["expected"]))
        case = deepcopy(next(row["case"] for row in streams["records"] if row["case"]["case_id"] == identity))
        output = Path(original["config"]["experiment_dir"])/"runs"/"validation-continuation-q2"
        output.mkdir()
        values = {name: [] for name in recovery.STREAMS}
        for name, rows in streams.items():
            for row in rows:
                if recovery.case_id(row, name) != first_id:
                    continue
                value = deepcopy(row)
                if name == "records":
                    value["case"] = case
                else:
                    value["query"] = case["query"]
                    if name == "calls":
                        value["case_id"] = identity
                values[name].append(value)
        code, extension = {"backend.py": "stable"}, continuation.extension_identity()
        report = deepcopy(original)
        report.update(stage=continuation.SEGMENT_STAGE, completed=True, completed_queries=1,
                      requested_queries=1, code=code, model={"weights": "test-checkpoint"}, runtime={"device": "cpu"},
                      gpu_memory={}, evaluation_seconds=1., resource_loading_and_checks_seconds=.2,
                      accounting={"search_calls": 4, "actual_cem_sequence_evaluations": 10,
                                  "actual_cem_transition_predictions": 50, "checkpoint_counters": {"actual_updates": 5}},
                      continuation={"source_report": str(source["path"]), "source_report_sha256": source["sha256"],
                                    "selected_case_ids": [identity], "backend_code": code, "extension_code": extension},
                      summary=summarize(values["records"], original["config"]))
        for name, rows in values.items():
            path = output/(name+".jsonl.gz")
            with gzip.open(path, "wt") as stream:
                for row in rows:
                    stream.write(json.dumps(row)+"\n")
            count_key = {"records": "record_count", "plans": "plan_count", "calls": "call_count"}[name]
            report.update({name+"_file": str(path), name+"_sha256": recovery.digest(path), count_key: len(rows)})
        path = output/"report.json"
        recovery.write_json(path, report)
        return path, report, values["records"], code, extension

    def test_inspect_verifies_prefix_without_numerical_execution(self):
        with tempfile.TemporaryDirectory() as directory:
            source, _ = self.fixture(directory)
            with patch.object(continuation, "run_segment", side_effect=AssertionError("Must not evaluate")), \
                    patch("sys.stdout", new_callable=io.StringIO):
                result = continuation.continue_run(source["path"], inspect_only=True)
            self.assertEqual(len(result["expected"])-result["report"]["completed_queries"], 1)

    def test_main_bank_uses_only_suffix_acceptance_checks_keep_original_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            source, _ = self.fixture(directory)
            calls = []
            def original(resources, manifest, split, limit):
                calls.append((manifest, split, limit))
                return SimpleNamespace(items=[{"transition_id": identity} for identity in list(source["expected"])[2:]])
            # SimpleNamespace lacks len(); use an explicit fake bank.
            class Bank:
                items = [{"transition_id": identity} for identity in list(source["expected"])[2:]]
                def __len__(self):
                    return len(self.items)
            def bank(resources, manifest, split, limit):
                original(resources, manifest, split, limit)
                return Bank()
            factory = continuation.suffix_bank_factory(bank, source)
            manifest = deepcopy(source["manifest"])
            factory(None, manifest, "validation", None)
            factory(None, manifest, "validation", 1)
            self.assertEqual(calls[0][0]["splits"]["validation"]["raw_start_rows"], [200])
            self.assertIs(calls[1][0], manifest)
            self.assertEqual(calls[1][2], 1)
            self.assertEqual(manifest, source["manifest"])

    def test_merge_contains_all_queries_and_preserves_prefix_and_time_provenance(self):
        with tempfile.TemporaryDirectory() as directory:
            source, streams = self.fixture(directory)
            path, _, rows, code, extension = self.segment_fixture(source, streams)
            segment, validated = continuation.checked_segment(path, source, code, extension)
            self.assertEqual(rows, validated)
            prefix_path = Path(source["report"]["records_file"])
            prefix = prefix_path.read_bytes()
            output = Path(directory)/"combined"
            with patch("sys.stdout", new_callable=io.StringIO):
                report = continuation.merge(source, path, segment, rows, output)
            self.assertEqual(report["completed_queries"], 3)
            self.assertTrue(report["completed"])
            self.assertEqual(report["record_count"], 27)
            self.assertEqual(report["continuation"]["reused_queries"], 2)
            self.assertEqual(report["continuation"]["new_queries"], 1)
            self.assertEqual(report["accounting"]["search_calls"], 12)
            self.assertEqual(report["accounting"]["actual_cem_transition_predictions"], 50)
            self.assertIsNone(report["evaluation_seconds"])
            self.assertIsNone(report["code"])
            self.assertFalse(report["validation_lock_eligible"])
            self.assertEqual({row["success_rate"]["queries"] for row in report["summary"]["curves"]}, {3})
            self.assertTrue(Path(report["records_file"]).read_bytes().startswith(prefix))
            self.assertEqual(prefix_path.read_bytes(), prefix)

    def test_completed_segment_is_reused_after_merge_was_interrupted(self):
        with tempfile.TemporaryDirectory() as directory:
            source, streams = self.fixture(directory)
            _, _, _, code, _ = self.segment_fixture(source, streams)
            with patch("experiments.provenance.backend_identity", return_value=code), \
                    patch.object(continuation, "run_segment", side_effect=AssertionError("Must reuse segment")), \
                    patch("sys.stdout", new_callable=io.StringIO):
                report = continuation.continue_run(source["path"])
            self.assertEqual(report["completed_queries"], 3)

    def test_changed_segment_identity_or_hash_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            source, streams = self.fixture(directory)
            path, report, _, code, extension = self.segment_fixture(source, streams)
            with self.assertRaisesRegex(ValueError, "differs"):
                continuation.checked_segment(path, source, {"backend.py": "changed"}, extension)
            with Path(report["calls_file"]).open("ab") as stream:
                stream.write(b"changed")
            with self.assertRaisesRegex(ValueError, "Changed calls"):
                continuation.checked_segment(path, source, code, extension)

    def test_changed_prepared_fingerprint_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            source, _ = self.fixture(directory)
            path = Path(source["report"]["config"]["p1"]["paths"]["experiment_dir"])/"prepared.json"
            changed = {"model": {"weights": "different-checkpoint"}}
            changed["fingerprint"] = recovery.object_digest(changed)
            recovery.write_json(path, changed)
            with self.assertRaisesRegex(ValueError, "Prepared model"):
                continuation.load_partial(source["path"])


if __name__ == "__main__":
    unittest.main()

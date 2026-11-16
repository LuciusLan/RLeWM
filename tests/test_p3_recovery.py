"""Offline recovery contracts. User-run; the coding agent checks syntax only."""

from collections import Counter
from copy import deepcopy
import gzip
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch


SCRIPT = Path(__file__).resolve().parents[1]/"recover-p3-core.py"
SPEC = importlib.util.spec_from_file_location("p3_recovery", SCRIPT)
recovery = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(recovery)


class RecoveryTests(unittest.TestCase):
    def read_stream(self, data):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"records.jsonl.gz"
            path.write_bytes(data)
            receipt = {}
            rows = [row for _, row in recovery.json_lines(path, receipt)]
            return rows, receipt

    def test_missing_footer_and_torn_json_keep_only_complete_lines(self):
        data = gzip.compress(b'{"case":1}\n{"case":2}\n{"case":')[:-8]
        rows, receipt = self.read_stream(data)
        self.assertEqual(rows, [{"case": 1}, {"case": 2}])
        self.assertTrue(receipt["truncated_gzip"])
        self.assertEqual(receipt["discarded_tail_bytes"], len(b'{"case":'))

    def test_concatenated_resume_members_and_truncated_last_member(self):
        data = gzip.compress(b'{"case":1}\n')+gzip.compress(b'{"case":2}\n')[:-8]
        rows, receipt = self.read_stream(data)
        self.assertEqual(rows, [{"case": 1}, {"case": 2}])
        self.assertEqual(receipt["finished_gzip_members"], 1)
        self.assertTrue(receipt["truncated_gzip"])

    def test_crc_errors_and_malformed_complete_json_are_fatal(self):
        damaged = bytearray(gzip.compress(b'{"case":1}\n'))
        damaged[-8] ^= 1
        for data in (bytes(damaged), gzip.compress(b'{"case":oops}\n'), gzip.compress(b'{"x":NaN}\n')):
            with self.subTest(data=data), self.assertRaises(ValueError):
                self.read_stream(data)

    def test_non_newline_terminated_row_is_never_committed(self):
        rows, receipt = self.read_stream(gzip.compress(b'{"case":1}\n{"case":2}'))
        self.assertEqual(rows, [{"case": 1}])
        self.assertFalse(receipt["truncated_gzip"])
        self.assertGreater(receipt["discarded_tail_bytes"], 0)

    def fixture(self, directory):
        experiment = Path(directory)/"experiment"
        run = experiment/"runs"/"validation"
        run.mkdir(parents=True)
        query_path = Path(directory)/"queries.json"
        query = {"regime": "source", "collection_manifest": "abc123", "episode_metadata": {
            "ids": [100, 101, 102], "offsets": [0, 100, 200], "lengths": [100, 100, 100]}}
        query["fingerprint"] = recovery.object_digest(query)
        recovery.write_json(query_path, query)
        manifest = {"history": 3, "frameskip": 5, "horizon": 5,
                    "query_source_fingerprint": query["fingerprint"], "p1_prepared_fingerprint": "prepared",
                    "splits": {"validation": {"queries": 3, "raw_start_rows": [0, 100, 200]}}}
        manifest["fingerprint"] = recovery.object_digest(manifest)
        recovery.write_json(experiment/"manifest.json", manifest)
        config = {"experiment_dir": str(experiment), "query_dataset": {"manifest": str(query_path)},
                  "planning": {"horizon": 5}, "evaluation": {"query_limit": None},
                  "p1": {"data": {"history": 3, "frameskip": 5}, "evaluation": {"seeds": [42]}},
                  "core": {"mid_search": False, "iterations": [8, 30], "control_budget": 50,
                           "state_atol": 1e-5, "rgb_mae_atol": .1, "bootstrap_samples": 10,
                           "bootstrap_seed": 4, "deadlines": [1], "success_targets": [.5]},
                  "interleave": {"adaptation_schedule": "sequential-depth", "adaptation_depths": [0, 1, 2, 3, 4]},
                  "adaptation": {"updates": 1}}
        recovery.write_json(run/"config.json", config)
        recovery.write_json(run/"status.json", {"status": "running", "completed_queries": 2,
                                                 "requested_queries": 3, "accounting": {"search_calls": 8}})
        recovery.write_json(run/"acceptance-checks.json", {"gaussian": {"check": True}, "retrieval": {"check": True}})
        expected = recovery.expected_cases(config, manifest, query, "validation")
        labels = ["gaussian/frozen", "gaussian/checkpoint", "init/frozen", "init/checkpoint"]
        streams = {name: [] for name in recovery.STREAMS}
        initial = {"success": False, "planning_score": 0., "goal_distance": 2.}
        final = {"success": True, "planning_score": 1., "goal_distance": 0.}
        state = {"state_max_abs_error": 0., "rgb_mae_0_255": 0.}
        for number, (identity, value) in enumerate(expected.items()):
            case = {k: v for k, v in value.items() if k not in ("episode_id", "start_in_episode")}
            case.update(case_id=identity, query={"transition_id": identity, "episode_id": value["episode_id"],
                                                "start_in_episode": value["start_in_episode"]})
            base = {"case": case, "initial_metrics": initial, "final_metrics": final,
                    "initial_success": False, "final_success": True, "ever_success": True,
                    "first_success_raw_step": 1, "raw_steps": 1, "planning_score_improvement": 1.,
                    "goal_distance_reduction": 2., "prefix_reconstruction": state}
            streams["records"].append({**base, "kind": "recorded-control", "recorded_future_reconstruction": state})
            for j in (8, 30):
                for label in labels:
                    if number == 2 and (j, label) != (8, labels[0]):
                        continue  # A genuinely unfinished query at the tail.
                    key = {"iterations": j, "seed": 42, "condition": label}
                    streams["records"].append({**base, **key, "kind": "planned", "planning_calls": 1,
                        "planning_seconds": .5, "compute": {}, "executed_raw_actions": [[.1, .2]]})
                    streams["plans"].append({**key, "query": case["query"]})
                    streams["calls"].append({**key, "case_id": identity, "query": case["query"],
                        "replan": 0, "raw_steps_before": 0, "first_call_snapshot": True,
                        "executed_raw_actions": [[.1, .2]]})
        for name, rows in streams.items():
            data = b"".join(json.dumps(row).encode()+b"\n" for row in rows)+b'{"torn":'
            (run/(name+".jsonl.gz")).write_bytes(gzip.compress(data)[:-8])
        return run, config, expected, streams

    def test_end_to_end_recovery_preserves_inputs_and_pairs_all_curves(self):
        with tempfile.TemporaryDirectory() as directory:
            run, _, _, _ = self.fixture(directory)
            hashes = {path.name: recovery.digest(path) for path in run.iterdir()}
            with patch("sys.stdout", new_callable=io.StringIO):
                report = recovery.recover(run)
            self.assertEqual(report["completed_queries"], 2)
            self.assertEqual(report["record_count"], 18)
            self.assertEqual(report["plan_count"], 16)
            self.assertEqual(report["call_count"], 16)
            self.assertFalse(report["completed"])
            self.assertIsNone(report["code"])
            self.assertIsNone(report["evaluation_seconds"])
            self.assertEqual({c["success_rate"]["queries"] for c in report["summary"]["curves"]}, {2})
            for name, sha in hashes.items():
                self.assertEqual(recovery.digest(run/name), sha)
            for name in recovery.STREAMS:
                with gzip.open(report[name+"_file"], "rt") as stream:
                    self.assertTrue(all(json.loads(line) for line in stream))  # Valid closing trailers.
            self.assertTrue((run/"recovered-partial"/"success-curves.csv").exists())
            with self.assertRaises(FileExistsError), patch("sys.stdout", new_callable=io.StringIO):
                recovery.recover(run)

    def test_incomplete_cell_excludes_the_entire_last_query(self):
        with tempfile.TemporaryDirectory() as directory:
            run, _, _, _ = self.fixture(directory)
            path = run/"calls.jsonl.gz"
            receipt = {}
            rows = [row for _, row in recovery.json_lines(path, receipt)]
            rows.pop(15)  # Remove one call in the second otherwise complete query.
            path.write_bytes(gzip.compress(b"".join(json.dumps(row).encode()+b"\n" for row in rows)))
            with patch("sys.stdout", new_callable=io.StringIO):
                audit = recovery.recover(run, inspect_only=True)
            self.assertEqual(audit["completed_queries"], 1)
            self.assertIn("incomplete executed-call", audit["stop_reason"])
            self.assertFalse((run/"recovered-partial").exists())

    def test_action_mismatch_is_corruption_not_a_silent_tail_drop(self):
        with tempfile.TemporaryDirectory() as directory:
            run, config, expected, streams = self.fixture(directory)
            identity = next(iter(expected))
            rows = [row for row in streams["records"] if row["case"]["case_id"] == identity]
            keys = {recovery.row_key(row) for row in rows if row["kind"] == "planned"}
            plans = Counter(keys)
            calls = {recovery.row_key(row): [row] for row in streams["calls"] if row["case_id"] == identity}
            calls[next(iter(calls))][0]["executed_raw_actions"] = [[9., 9.]]
            with self.assertRaisesRegex(ValueError, "Executed actions differ"):
                recovery.complete_case(identity, rows, plans, calls, keys, config)

    def test_duplicate_outcomes_and_failed_reconstruction_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            _, config, expected, streams = self.fixture(directory)
            identity = next(iter(expected))
            rows = [row for row in streams["records"] if row["case"]["case_id"] == identity]
            keys = {recovery.row_key(row) for row in rows if row["kind"] == "planned"}
            plans = Counter(keys)
            calls = {recovery.row_key(row): [row] for row in streams["calls"] if row["case_id"] == identity}
            with self.assertRaisesRegex(ValueError, "Duplicate"):
                recovery.complete_case(identity, rows+[rows[-1]], plans, calls, keys, config)
            failed = deepcopy(rows)
            failed[0]["recorded_future_reconstruction"]["state_max_abs_error"] = 1.
            with self.assertRaisesRegex(ValueError, "reconstruction failed"):
                recovery.complete_case(identity, failed, plans, calls, keys, config)

    def test_live_open_writer_is_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"calls.jsonl.gz"
            with path.open("wb"):
                with self.assertRaisesRegex(ValueError, "still being written"):
                    recovery.assert_idle([path])


if __name__ == "__main__":
    unittest.main()

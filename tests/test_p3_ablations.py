"""Model-free job/reuse contracts; user-run, no encoder or CEM invocation."""

from copy import deepcopy
from itertools import product
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from experiments.config import resolve
from experiments.p3_ablations import plan_jobs
from lewm_tta.proposals.reuse_frozen import load_frozen_cells, reusable_sources
from lewm_tta.runtime import file_hash


class P3AblationTests(unittest.TestCase):
    def test_default_plan_changes_one_factor_and_keeps_frozen(self):
        root = Path(__file__).resolve().parents[2]
        base = resolve(root / "codes/configs/p3/fresh-proposals.json")
        original = deepcopy(base)
        jobs = dict(plan_jobs(base, [32, 64, 128], [2, 6, 12], ["controls", "share", "budget"]))
        self.assertEqual(set(jobs), {"controls", "share-32", "share-128", "iterations-2", "iterations-12"})
        self.assertEqual(sum(len(value["proposals"]["sources"]) for value in jobs.values()), 11)
        self.assertEqual(base, original)
        for name, config in jobs.items():
            self.assertEqual(config["adaptation"]["methods"], ["frozen"])
            self.assertEqual(config["planning"]["horizon"], 5)
            self.assertEqual(config["planning"]["population"], 256)
            self.assertEqual(config["planning"]["elites"], 32)
            self.assertEqual(config["proposals"]["noise_std"], .1)
            self.assertEqual(config["proposals"]["fragments"], 8)
            if name.startswith("share-"):
                self.assertEqual(config["planning"], base["planning"])
            if name.startswith("iterations-"):
                self.assertEqual(config["proposals"]["injection_slots"], 64)

    def test_injected_cells_require_matching_parameters_gaussian_does_not(self):
        source = {"config": {"proposals": {"sources": ["gaussian", "context-fragments"],
                                           "injection_slots": 64, "noise_std": .1}}}
        target = {"proposals": {"sources": ["gaussian", "uniform-fragments", "context-fragments"],
                                 "injection_slots": 64, "noise_std": .1}}
        self.assertEqual(reusable_sources(source, target), {"gaussian", "context-fragments"})
        for field, value in (("injection_slots", 32), ("noise_std", .2)):
            changed = deepcopy(target)
            changed["proposals"][field] = value
            self.assertEqual(reusable_sources(source, changed), {"gaussian"})

    def test_goal_is_one_fixed_job_without_changing_or_crossing_existing_jobs(self):
        root = Path(__file__).resolve().parents[2]
        base = resolve(root / "codes/configs/p3/fresh-proposals.json")
        groups = ["controls", "share", "budget"]
        previous = dict(plan_jobs(base, [32, 64, 128], [2, 6, 12], groups))
        extended = dict(plan_jobs(base, [32, 64, 128], [2, 6, 12], groups+["goal"]))
        self.assertEqual(set(extended)-set(previous), {"goal"})
        for label, config in previous.items():
            self.assertEqual(config, extended[label])
            self.assertNotIn("goal-context-fragments", config["proposals"]["sources"])
        goal = extended["goal"]
        self.assertEqual(goal["adaptation"]["methods"], ["frozen"])
        self.assertEqual(goal["planning"]["iterations"], 6)
        self.assertEqual(goal["proposals"]["sources"], ["gaussian", "context-fragments", "goal-context-fragments"])
        for key, value in (("fragments", 8), ("injection_slots", 64), ("noise_std", .1), ("goal_weight", 1.)):
            self.assertEqual(goal["proposals"][key], value)
        # Changing grid entries cannot change the dedicated goal operating point.
        another = dict(plan_jobs(base, [16, 128], [1, 12], ["goal"]))
        self.assertEqual(another["goal"], goal)

    def test_legacy_gaussian_lookup_and_new_proposal_lookup_stay_separate(self):
        from lewm_tta.proposals.reuse import ReusedPlans
        cache = ReusedPlans.__new__(ReusedPlans)
        gaussian = {"mode": "frozen"}
        guided = {"proposal_source": "context-fragments", "proposal": {
            "source": "context-fragments", "fragment_fingerprint": "fragments", "injection_fingerprint": "noise",
            "injected_slots": 64, "requested_slots": 64, "slot_provenance": []}}
        cache.rows = {("query", 42, "frozen"): gaussian}
        cache.origins = {("query", 42, "frozen"): {"mode": "frozen"}}
        cache.proposal_rows = {("query", 42, "frozen", "context-fragments"): guided}
        cache.proposal_origins = {("query", 42, "frozen", "context-fragments"): {"proposal_source": "context-fragments"}}
        task = SimpleNamespace(metadata={"transition_id": "query"})
        self.assertIs(cache.reference(task, 42, "frozen"), gaussian)
        self.assertIs(cache.reference(task, 42, "frozen", "context-fragments"), guided)
        self.assertIsNone(cache.reference(task, 42, "frozen", "uniform-fragments"))
        self.assertEqual(cache.origin(task, 42, "frozen"), {"mode": "frozen"})
        cache.verify_proposal(task, 42, "frozen", "context-fragments", guided["proposal"])
        corrupted = {**guided["proposal"], "injection_fingerprint": "different-noise"}
        with self.assertRaisesRegex(ValueError, "injection_fingerprint"):
            cache.verify_proposal(task, 42, "frozen", "context-fragments", corrupted)

    def source_fixture(self, directory):
        proposals = {"sources": ["gaussian", "context-fragments"], "injection_slots": 64, "noise_std": .1}
        config = {"adaptation": {"methods": ["frozen"]}, "p1": {"evaluation": {"seeds": [42, 43]}},
                  "proposals": proposals}
        rows = []
        for seed, source in product([42, 43], proposals["sources"]):
            rows.append({"query": {"transition_id": "query-1", "source_episode_position": -7},
                "seed": seed, "method": "frozen", "proposal_source": source, "scope": "last-block",
                "current_raw_row": 10, "goal_raw_row": 35, "target_raw_rows": [15, 20, 25, 30, 35],
                "provisional_seed": seed, "provisional_fingerprint": "actions", "cem_seed": seed+1,
                "base_first_population_fingerprint": "base", "recorded_action_mse_by_block": [.1]*5,
                "adaptation": {"updates": 0, "parameter_displacement_mse": 0.}})
        path = directory / "records.jsonl"
        path.write_text("".join(json.dumps(row)+"\n" for row in rows))
        source = {"config": deepcopy(config), "records_file": str(path), "records_sha256": file_hash(path),
                  "record_count": len(rows), "cohort": {"queries": 1, "episodes": 1}}
        return config, rows, source

    def load_fixture(self, directory, source, config):
        with patch("lewm_tta.proposals.reuse_frozen.check_frozen_protocol"), \
             patch("lewm_tta.proposals.reuse_frozen.check_frozen_sources"):
            return load_frozen_cells(directory / "report.json", source, config, "validation", {})

    def rewrite_records(self, source, rows):
        path = Path(source["records_file"])
        path.write_text("".join(json.dumps(row)+"\n" for row in rows))
        source.update(record_count=len(rows), records_sha256=file_hash(path))

    def test_source_matrix_is_complete_before_seed_and_source_filtering(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            config, rows, source = self.source_fixture(directory)
            config["p1"]["evaluation"]["seeds"] = [42]
            config["proposals"]["injection_slots"] = 32
            kept = self.load_fixture(directory, source, config)
            self.assertEqual(set(kept), {("query-1", 42, "frozen", "gaussian")})
            self.rewrite_records(source, rows[:-1])
            with self.assertRaisesRegex(ValueError, "incomplete"):
                self.load_fixture(directory, source, config)

    def test_duplicate_and_adapted_frozen_rows_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            config, rows, source = self.source_fixture(directory)
            self.rewrite_records(source, rows+[rows[0]])
            with self.assertRaisesRegex(ValueError, "Duplicate"):
                self.load_fixture(directory, source, config)
            rows[1]["adaptation"]["updates"] = 1
            self.rewrite_records(source, rows)
            with self.assertRaisesRegex(ValueError, "adaptation"):
                self.load_fixture(directory, source, config)

    def test_goal_pairing_and_record_checksums_are_enforced(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            config, rows, source = self.source_fixture(directory)
            rows[1]["goal_raw_row"] = 36
            self.rewrite_records(source, rows)
            with self.assertRaisesRegex(ValueError, "pairing"):
                self.load_fixture(directory, source, config)
            source["records_sha256"] = "0"*64
            with self.assertRaisesRegex(ValueError, "changed"):
                self.load_fixture(directory, source, config)


if __name__ == "__main__":
    unittest.main()

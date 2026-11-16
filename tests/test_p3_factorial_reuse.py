"""Model-free checks of factorial contrasts, source guards and workflow dispatch."""

from copy import deepcopy
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from experiments.catalog import profile
from experiments.workflows import compare, completed_report
from lewm_tta.proposals.effects import COMMON_SCORE, factorial_effects
from lewm_tta.proposals.reuse import ReusedPlans, check_protocol, pinned_report, validate_reuse
from lewm_tta.runtime import file_hash, write_json


class FactorialReuseTests(unittest.TestCase):
    def test_random_retrieved_and_proposal_effects_are_distinct(self):
        config = {"p1": {"evaluation": {"seeds": [42, 43]}},
                  "adaptation": {"methods": ["frozen", "uniform", "fr-fixed"]},
                  "proposals": {"sources": ["gaussian", "context-fragments"]},
                  "evaluation": {"bootstrap_samples": 20}}
        costs = {("frozen", "gaussian"): 10., ("frozen", "context-fragments"): 6.,
                 ("uniform", "gaussian"): 9., ("uniform", "context-fragments"): 5.,
                 ("fr-fixed", "gaussian"): 8., ("fr-fixed", "context-fragments"): 3.}
        rows = {(q, seed, method, source): {COMMON_SCORE: cost,
                "query": {"source_episode_position": int(q)}}
                for q in ("1", "2") for seed in (42, 43) for (method, source), cost in costs.items()}
        result = factorial_effects(rows, ["1", "2"], config)
        self.assertEqual(result["ttt_minus_frozen_by_proposal"]["uniform"]["gaussian"]["mean_difference"], -1.)
        selection = result["support_selection_minus_uniform_by_proposal"]["fr-fixed"]
        self.assertEqual(selection["gaussian"]["mean_difference"], -1.)
        self.assertEqual(selection["context-fragments"]["mean_difference"], -2.)
        interaction = result["support_selection_difference_in_differences"]["fr-fixed"]
        self.assertEqual(interaction["context-fragments_minus_gaussian"]["mean_difference"], -1.)

    def test_source_hash_is_required_and_checked(self):
        for invalid in ({"validation": "file"}, {"other": []},
                        {"validation": [{"report": "file"}]},
                        {"test": [{"report": "file", "sha256": "x" * 64}]}):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                validate_reuse({"reuse": invalid})
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "report.json"
            write_json(path, {"completed": True})
            entry = {"report": str(path), "sha256": file_hash(path)}
            self.assertEqual(pinned_report(entry)[1], {"completed": True})
            write_json(path, {"completed": False})
            with self.assertRaisesRegex(ValueError, "changed"):
                pinned_report(entry)

    def test_split_budget_memory_and_cohort_mismatch_rejected(self):
        config = {"planning": {"horizon": 5}, "scopes": ["last-block"], "query_selection": {"seed": 7},
                  "adaptation": {"methods": ["frozen", "uniform", "fr-fixed"], "learning_rate": 5e-5},
                  "p1": {"paths": {"checkpoint": "original"}, "data": {"history": 3}, "retrieval": {"backend": "cuvs"}}}
        cohort = {"queries": 4, "episodes": 1, "raw_start_rows": [1, 2, 3, 4]}
        manifest = {"p1_prepared_fingerprint": "prepared", "splits": {"validation": cohort}}
        report = {"completed": True, "limited_run": False, "stage": "P3-temporal-cem", "split": "validation",
                  "config": deepcopy(config), "cohort": deepcopy(cohort), "summary": {"queries": 4, "episodes": 1},
                  "p1_prepared_fingerprint": "prepared"}
        check_protocol(report, config, "validation", manifest)
        for field, value in (("split", "test"), ("limited_run", True), ("p1_prepared_fingerprint", "other")):
            broken = deepcopy(report)
            broken[field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                check_protocol(broken, config, "validation", manifest)
        for field in ("planning", "adaptation", "p1", "cohort"):
            broken = deepcopy(report)
            if field == "planning":
                broken["config"][field]["horizon"] = 3
            elif field == "adaptation":
                broken["config"][field]["learning_rate"] = 1e-4
            elif field == "p1":
                broken["config"][field]["paths"]["checkpoint"] = "other"
            else:
                broken[field]["raw_start_rows"] = [5, 6, 7, 8]
            with self.subTest(field=field), self.assertRaises(ValueError):
                check_protocol(broken, config, "validation", manifest)

    def test_saved_metrics_are_kept_only_after_parity(self):
        source = {"recorded_action_mse_by_block": [.1, .2]}
        self.assertEqual(ReusedPlans.forecast([.1 + 1e-10, .2], source), [.1, .2])
        with self.assertRaises(ValueError):
            ReusedPlans.forecast([.1, .3], source)
        source = {"selected_model_goal_sse": 1., "selected_plan_frozen_model_goal_sse": 2.,
                  "frozen_plan_this_model_goal_sse": 3.}
        row = {key: value + 1e-7 for key, value in source.items()}
        ReusedPlans.retain_scores(row, source)
        self.assertEqual(row, source)
        row["selected_plan_frozen_model_goal_sse"] = 9.
        with self.assertRaises(ValueError):
            ReusedPlans.retain_scores(row, source)

    def test_workflow_dry_run_launches_no_process_or_model(self):
        output = io.StringIO()
        with patch("experiments.workflows.subprocess.run", side_effect=AssertionError("Launched a process")):
            with contextlib.redirect_stdout(output):
                compare(profile("p3", "default")[1], [], dry_run=True)
        text = output.getvalue()
        self.assertIn("factorial-validation", text)
        self.assertIn("--locked-config", text)
        self.assertIn("factorial-test", text)
        self.assertIn("Pinned Gaussian source cells are reused", text)

    def test_completed_matrix_skipped_but_seed_and_duplicate_mismatches_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = {"experiment_dir": directory,
                      "adaptation": {"methods": ["frozen", "uniform", "fr-fixed"]},
                      "proposals": {"sources": ["gaussian", "context-fragments"]},
                      "p1": {"evaluation": {"seeds": [42, 43]}}}
            cohort = {"queries": 2, "episodes": 2}
            write_json(root / "manifest.json", {"fingerprint": "manifest", "splits": {"validation": cohort}})
            rows = [{"query": {"transition_id": str(q), "source_episode_position": q},
                     "seed": seed, "method": method, "proposal_source": source}
                    for q in (1, 2) for seed in (42, 43)
                    for method in config["adaptation"]["methods"] for source in config["proposals"]["sources"]]
            records = root / "records.jsonl"
            records.write_text("".join(json.dumps(row) + "\n" for row in rows))
            report = {"completed": True, "limited_run": False, "split": "validation",
                      "stage": "P3c-offline-factorial", "config": config, "code": {"fixture": "digest"},
                      "manifest_fingerprint": "manifest", "cohort": cohort,
                      "records_file": str(records), "records_sha256": file_hash(records), "record_count": len(rows)}
            path = root / "report.json"
            write_json(path, report)
            with patch("experiments.workflows.backend_identity", return_value=report["code"]):
                self.assertEqual(completed_report(path, config, "validation"), report)
                wrong = deepcopy(report)
                wrong["config"]["p1"]["evaluation"]["seeds"] = [42]
                write_json(path, wrong)
                with self.assertRaises(ValueError):
                    completed_report(path, config, "validation")
                rows.append(rows[0])
                records.write_text("".join(json.dumps(row) + "\n" for row in rows))
                report.update(records_sha256=file_hash(records), record_count=len(rows))
                write_json(path, report)
                with self.assertRaisesRegex(ValueError, "duplicate"):
                    completed_report(path, config, "validation")


if __name__ == "__main__":
    unittest.main()

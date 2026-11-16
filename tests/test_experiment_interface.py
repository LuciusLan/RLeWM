"""Model-free migration checks: cohort settings, dispatch and verified replay."""

import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from experiments.catalog import ROOT, profile
from experiments.cli import dispatch
from experiments.config import materialize, read_profile, resolve
from experiments.provenance import archive_entry, backend_identity
from lewm_tta.runtime import file_hash, read_json, write_json


class ExperimentInterfaceTests(unittest.TestCase):
    def test_plural_learning_rates_route_only_to_the_sweep(self):
        arguments = ["--learning-rates", "1e-5", "5e-5", "1e-4", "--methods",
                     "frozen", "uniform", "fr-fixed", "forward", "reverse", "--seeds", "41", "42", "43"]
        with patch("experiments.p2_sweep.sweep") as sweep:
            dispatch(["p2", "sweep", "--dataset", "reacher", *arguments, "--dry-run"])
            sweep.assert_called_once_with(ROOT/"codes/configs/p2/fresh-temporal.json", arguments, True, dataset="reacher")
        for command in (["fresh", "validation"], ["p2", "evaluate"]):
            with self.assertRaisesRegex(ValueError, "belongs to 'p2 sweep'"):
                dispatch([*command, *arguments, "--dry-run"])
        with self.assertRaisesRegex(ValueError, "p2 sweep"):
            dispatch(["fresh", "validation", "--lrs=1e-5", "--dry-run"])

    def test_p0_dataset_path_is_forwarded_to_its_own_parser(self):
        with patch("experiments.cli.tool", return_value=True) as tool:
            dispatch(["p0", "smoke", "--dataset", "/tmp/example.h5", "--dry-run"])
            tool.assert_called_once_with("p0", "smoke", ["--dataset", "/tmp/example.h5"], True)

    def test_collection_resume_preserves_committed_source_and_rejects_mixing(self):
        from experiments.collection_resume import recorded_collection_code
        from lewm_tta.collection.manifest import sealed
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = {"output_dir": str(root), "regimes": {"source": {}}}
            original = {"codes/example.py": "original"}
            write_json(root/"source/episodes/validation-00000.json", sealed({"code": original}))
            self.assertEqual(recorded_collection_code(config), original)
            write_json(root/"source/episodes/validation-00001.json", sealed({"code": {"codes/example.py": "changed"}}))
            with self.assertRaisesRegex(ValueError, "mixed"):
                recorded_collection_code(config)

    def test_active_profiles_preserve_the_new_split_settings(self):
        mapping = {("p1", "default"): "p1-v64-t150-q4", ("p2", "default"): "p2_final-v64-t150-q4",
            ("p2", "initial"): "p25-v64-t150-q4", ("p2", "followup"): "p2_followup-v64-t150-q4",
            ("p3", "default"): "p3c-goal-v64-t150-q4", ("p3", "shared"): "p3a-v64-t150-q4",
            ("p3", "proposals"): "p3b-goal-v64-t150-q4", ("coverage", "default"): "rank-coverage"}
        for (stage, variant), name in mapping.items():
            with self.subTest(stage=stage, variant=variant):
                current = resolve(profile(stage, variant)[1])
                historical = read_json(ROOT / f"codes/legacy/configs/{name}.json")
                # Experiment controls (methods, schedules, LR) may evolve;
                # the migration must preserve the prepared source/cohorts.
                for key in ("paths", "data", "query_selection", "horizons", "cohort_horizon"):
                    self.assertEqual(current.get(key), historical.get(key), key)
                if "p1_config" in current:
                    self.assertEqual(Path(current["p1_config"]), ROOT / "codes/configs/p1/default.json")

    def test_legacy_source_references_and_immutable_materialization(self):
        value = resolve(ROOT / "codes/legacy/configs/p2_final.json")
        self.assertEqual(value["p1_config"], str(ROOT / "codes/legacy/configs/p1.json"))
        self.assertEqual(resolve(ROOT / "codes/configs/p2_final.json"), value)
        with tempfile.TemporaryDirectory() as temporary:
            with patch("experiments.config.ROOT", Path(temporary)):
                path = materialize(value)
                self.assertEqual(path, materialize(value))
                self.assertEqual(read_json(path), value)
                write_json(path, {"corrupt": True})
                with self.assertRaises(ValueError):
                    materialize(value)

    def test_inheritance_cycle_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "cycle.json"
            write_json(path, {"extends": "cycle.json"})
            with self.assertRaises(ValueError):
                read_profile(path)

    def test_proposal_lr_dispatch_uses_existing_backend_without_inference(self):
        calls = []
        backend = SimpleNamespace(main=lambda *args: calls.append((args, sys.argv[1:])))
        with tempfile.TemporaryDirectory() as temporary:
            with patch("experiments.cli.materialize", return_value=Path(temporary) / "resolved.json"), patch("experiments.cli.importlib.import_module", return_value=backend), patch.object(sys, "argv", ["test"]):
                dispatch(["p3", "evaluate", "--variant", "proposals", "--learning-rate", "1e-5", "--limit", "8"])
        positional, arguments = calls[0]
        self.assertEqual(positional, ())
        self.assertEqual(arguments[0], "evaluate")
        self.assertIn("--lr", arguments)
        self.assertNotIn("--learning-rate", arguments)
        self.assertIn("--config", arguments)

    def test_locked_test_receives_no_injected_config(self):
        calls = []
        backend = SimpleNamespace(main=lambda *args: calls.append((args, sys.argv[1:])))
        with tempfile.TemporaryDirectory() as temporary:
            lock = Path(temporary) / "lock.json"
            write_json(lock, {"stage": "P3c-offline-factorial", "code": backend_identity("p3c")})
            with patch("experiments.cli.importlib.import_module", return_value=backend), patch.object(sys, "argv", ["test"]):
                dispatch(["p3", "evaluate", "--split", "test", "--locked-config", str(lock)])
                with self.assertRaises(ValueError):
                    dispatch(["p3", "evaluate", "--locked-config", str(lock), "--config", "unwanted.json"])
        self.assertEqual(calls[0][0], ("p3c",))
        self.assertNotIn("--config", calls[0][1])

    def test_historical_report_replay_keeps_original_identity(self):
        archive = ROOT / "artifacts/code-archives/0586d22dffc1ca5c7c54"
        metadata = read_json(archive / "manifest.json")
        original = metadata["entrypoints"]["p3c"]["code"]
        script, prefix = archive_entry("p3c", original)
        self.assertEqual(script, archive / "codes/p3c.py")
        self.assertEqual(prefix, [])
        with tempfile.TemporaryDirectory() as temporary:
            report = Path(temporary) / "report.json"
            write_json(report, {"stage": "P3c-offline-factorial", "code": original})
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                dispatch(["p3", "analyze", "--report", str(report), "--dry-run"])
            self.assertIn(str(script), output.getvalue())
            self.assertNotIn("--config", output.getvalue())
            write_json(report, {"stage": "P3c-offline-factorial", "code": {"unavailable.py": "unknown"}})
            with self.assertRaises(ValueError):
                dispatch(["p3", "analyze", "--report", str(report), "--dry-run"])

    def test_dry_run_does_not_load_numerical_backends(self):
        with patch("experiments.cli.importlib.import_module", side_effect=AssertionError("Unexpected runtime import")):
            with contextlib.redirect_stdout(io.StringIO()):
                dispatch(["prepare-cohorts", "--dry-run"])
                dispatch(["legacy", "p2", "prepare", "--dry-run"])
        self.assertNotIn("torch", sys.modules)

    def test_prepared_cache_implementation_is_unchanged(self):
        archive = ROOT / "artifacts/code-archives/0586d22dffc1ca5c7c54"
        for name in ("model", "config", "data", "frame_bank", "observation_cache", "indexed_keys", "cuvs_retrieval", "retrieval"):
            relative = Path(f"codes/lewm_tta/{name}.py")
            self.assertEqual(file_hash(ROOT / relative), file_hash(archive / relative))

    def test_early_p1_rank_analysis_does_not_require_missing_runtime_source(self):
        with tempfile.TemporaryDirectory() as temporary:
            report = Path(temporary) / "report.json"
            write_json(report, {"code": {"unavailable.py": "unknown"}})
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                dispatch(["p1", "analyze-ranks", "--report", str(report), "--dry-run"])
            self.assertIn("experiments.backends.p1", output.getvalue())
            self.assertIn("analyze-ranks", output.getvalue())


if __name__ == "__main__":
    unittest.main()

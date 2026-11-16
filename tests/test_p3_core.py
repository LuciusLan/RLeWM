"""User-run core matrix and causal MPC contracts; agent syntax-checks only."""

from copy import deepcopy
import contextlib
import io
from contextlib import ExitStack
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from experiments.catalog import infer_backend, profile
from experiments.cli import dispatch
from experiments.config import resolve
from lewm_tta.core.config import CELLS, cells, groups, primary_endpoint, validate
from lewm_tta.core.execution import execute
from lewm_tta.core.reporting import summarize
from lewm_tta.interleave.config import load_config
from lewm_tta.outcomes.metrics import PlanOutcome
from lewm_tta.runtime import read_json, write_json


class CoreTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"config.json"
            write_json(path, resolve(profile("p3", "core")[1]))
            cls.config = load_config(path)

    def test_default_four_cells_keep_initialization_and_disable_mid_search(self):
        validate(self.config)
        values = groups(self.config)
        selected = [(name, arm) for name, value in values.items() for arm in value["interleave"]["arms"]]
        self.assertEqual(set(selected), set(cells(self.config)))
        self.assertEqual(len(selected), 4)
        self.assertEqual(set(cells(self.config).values()),
                         {"gaussian/frozen", "gaussian/checkpoint", "init/frozen", "init/checkpoint"})
        self.assertFalse(self.config["core"]["mid_search"])
        self.assertTrue(all(value["interleave"]["injection_slots"] == 0 for value in values.values()))
        self.assertGreater(values["retrieval"]["proposals"]["injection_slots"], 0)
        self.assertEqual(values["gaussian"]["interleave"]["initial_source"], "gaussian")
        self.assertEqual(values["retrieval"]["interleave"]["initial_source"], "context-fragments")
        self.assertEqual(self.config["planning"]["population"], 300)
        self.assertEqual(self.config["planning"]["elites"], 30)
        self.assertEqual(self.config["core"]["iterations"], [2, 4, 8, 16])
        self.assertEqual(infer_backend("p3", self.config), "p3_core")
        old = deepcopy(self.config)
        old.pop("core")
        self.assertEqual(infer_backend("p3", old), "p3_interleave")

    def test_explicit_and_historical_six_cell_configs_remain_available(self):
        for historical in (False, True):
            value = deepcopy(self.config)
            if historical:
                value["core"].pop("mid_search")
            else:
                value["core"]["mid_search"] = True
            validate(value)
            self.assertEqual(cells(value), CELLS)
            self.assertEqual(groups(value)["retrieval"]["interleave"]["arms"],
                             ["baseline", "rca", "proposals", "combined"])
        value["core"]["mid_search"] = "false"
        with self.assertRaises(ValueError):
            validate(value)

    def test_four_cell_providers_load_initial_fragments_without_suffix_indexes(self):
        from lewm_tta.interleave.continuations import ContinuationProviders
        resources = SimpleNamespace(engine=SimpleNamespace(retriever=object()), memory=object())
        coordinates = object()
        with patch("lewm_tta.interleave.continuations.ContextSearch"), \
             patch("lewm_tta.interleave.continuations.FragmentStore") as store, \
             patch("lewm_tta.interleave.continuations.FragmentProvider"):
            for name, value in groups(self.config).items():
                provider = ContinuationProviders(resources, value, coordinates)
                self.assertEqual(provider.providers, {})
                self.assertEqual(provider.initial is None, name == "gaussian")
            self.assertEqual(store.call_count, 1)
            self.assertEqual(store.call_args.args[1], 5)

    def test_previous_six_cell_implementation_has_verified_archive_entrypoint(self):
        from experiments.provenance import archive_entry
        root = Path(__file__).resolve().parents[2]
        archive = root/"artifacts/code-archives/8bea09f8624ca6f8924d"
        manifest = read_json(archive/"manifest.json")
        recorded = manifest["entrypoints"]["p3_core"]["code"]
        self.assertEqual(archive_entry("p3_core", recorded, root=root),
                         (archive/"codes/experiment.py", ["legacy", "p3_core"]))

    def test_control_protocol_and_validation_reject_inactive_treatments(self):
        self.assertEqual(primary_endpoint(self.config), "final")
        value = deepcopy(self.config)
        value["core"]["control_budget"] = 50
        validate(value)
        self.assertEqual(primary_endpoint(value), "ever")
        value["interleave"]["injection_slots"] = 0
        validate(value)
        for changed in ("initial", "mid", "updates", "checkpoints"):
            value = deepcopy(self.config)
            if changed == "initial":
                value["proposals"]["injection_slots"] = 0
            elif changed == "mid":
                value["core"]["mid_search"] = True
                value["interleave"]["injection_slots"] = 0
            elif changed == "updates":
                value["adaptation"]["updates"] = 0
            else:
                value["interleave"]["checkpoints"] = []
            with self.assertRaises(ValueError):
                validate(value)

    def test_frontend_default_is_one_seed_one_engine_no_dry_run_writes(self):
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()) as out:
            with patch("experiments.p3_core.subprocess.run", side_effect=AssertionError("execution")):
                dispatch(["p3", "core", "--dataset", "pusht", "--output-root", directory, "--dry-run"])
            text = out.getvalue()
            self.assertIn("4 cells, seeds=[42]", text)
            self.assertIn("controls=25, primary=final", text)
            self.assertEqual(text.count("legacy p3_core evaluate"), 1)
            self.assertEqual(list(Path(directory).iterdir()), [])
        with contextlib.redirect_stdout(io.StringIO()) as out:
            dispatch(["p3", "core", "--dataset", "pusht", "--output-root", "/tmp/core-six-contract",
                      "--mid-search", "--dry-run"])
            self.assertIn("6 cells, seeds=[42]", out.getvalue())
            self.assertIn("init-mid/checkpoint", out.getvalue())
        with contextlib.redirect_stdout(io.StringIO()) as out:
            dispatch(["p3", "core", "--dataset", "reacher", "--output-root", "/tmp/core-contract",
                      "--control-budget", "50", "--dry-run"])
            self.assertIn("seeds=[41]", out.getvalue())
            self.assertIn("controls=50, primary=ever", out.getvalue())
            self.assertIn("/home/ubuntu/.venv/bin/python", out.getvalue())
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()) as out:
            captured = []
            def materialized(value):
                captured.append(deepcopy(value))
                return Path(directory)/"config.json"
            with patch("experiments.p3_core.materialize", side_effect=materialized), \
                 patch("experiments.p3_core.subprocess.run") as run:
                dispatch(["p3", "core", "--dataset", "cube", "--output-root", directory,
                          "--updates", "7", "--iterations", "30", "--checkpoints", "2", "6", "10", "14", "18", "22", "26"])
            self.assertIn("cube-plan-mj36/source/queries.json", captured[0]["query_dataset"]["manifest"])
            self.assertTrue(run.call_args.args[0][0].endswith(".venvs/physics-mj36/bin/python"))
            self.assertEqual(captured[0]["adaptation"]["updates"], 7)
            self.assertEqual(captured[0]["interleave"]["checkpoints"], [2, 6, 10, 14, 18, 22, 26])
        for option in (["--seeds", "99"], ["--mid-search"], ["--updates", "7"]):
            with self.assertRaises(ValueError):
                dispatch(["p3", "core", "--phase", "test", "--locked-config", "/tmp/lock.json",
                          *option, "--dry-run"])

    def run_execution(self, controls, success_at):
        class Replay:
            def reset(self, case):
                self.steps = 0
                return SimpleNamespace(rgb=np.zeros((1, 1, 3), dtype=np.uint8)), {}
            def goal(self, case):
                return {"state64": np.asarray([999.])}
            def metrics(self, observation, goal):
                return {"success": self.steps >= success_at, "planning_score": float(self.steps),
                        "goal_distance": float(max(0, success_at-self.steps))}
            def step(self, action):
                self.steps += 1
                return SimpleNamespace(rgb=np.full((1, 1, 3), self.steps, dtype=np.uint8))
        class History:
            def __init__(self, *args):
                self.observed_steps = []
            def append(self, rgb, block):
                self.observed_steps.append(int(rgb[0, 0, 0]))
            def context(self):
                return tuple(self.observed_steps)
        config = deepcopy(self.config)
        config["core"]["control_budget"] = controls
        plan = SimpleNamespace(actions=torch.ones(5, 5), candidate_evaluations=33)
        snapshot = SimpleNamespace(iterations=4, plan=plan)
        coordinates = SimpleNamespace(to_raw=lambda actions: actions.reshape(5, 5, 1))
        resources = SimpleNamespace(adapter=SimpleNamespace(device=torch.device("cpu")))
        task = SimpleNamespace(context="past only", metadata={"transition_id": "query", "source_episode_position": -1})
        source = {"primary_seconds_with_checks": .5, "counters": {}, "cem_candidate_evaluations": 33,
                  "cem_transition_predictions": 165}
        goal = torch.tensor([[123.]])
        calls = []
        with patch("lewm_tta.core.execution.LiveHistory", History), \
             patch("lewm_tta.core.execution.call_row", return_value=deepcopy(source)), \
             patch("lewm_tta.core.execution.search", return_value=(SimpleNamespace(snapshots={4: snapshot}), None, None)) as search:
            result = execute(Replay(), {"case_id": "query"}, task, goal, resources, config, None,
                             coordinates, "rca", 42, snapshot, source, {}, calls.append, "init/checkpoint")
        return result, calls, search, goal

    def test_50_controls_replan_from_actual_history_with_unchanged_goal_and_own_J(self):
        result, calls, search, goal = self.run_execution(50, 35)
        self.assertEqual(result["raw_steps"], 35)
        self.assertTrue(result["ever_success"])
        self.assertEqual(result["planning_calls"], 2)
        self.assertEqual([c["raw_steps_before"] for c in calls], [0, 25])
        self.assertEqual(search.call_count, 1)
        args, kwargs = search.call_args
        self.assertEqual(args[1]["planning"]["iterations"], 4)
        self.assertEqual(args[4], (5, 10, 15, 20, 25))
        self.assertIs(args[6], goal)
        self.assertEqual(args[9], [4])
        self.assertEqual(args[5]["simulator_raw_steps"], 25)
        self.assertEqual(len(kwargs["warm"]), 0)
        self.assertAlmostEqual(result["search_seconds"], 1.)

    def test_25_controls_execute_full_plan_without_replanning_after_early_hit(self):
        result, calls, search, _ = self.run_execution(25, 10)
        self.assertEqual(result["raw_steps"], 25)
        self.assertEqual(result["planning_calls"], 1)
        self.assertEqual(search.call_count, 0)
        self.assertEqual(len(calls[0]["executed_raw_actions"]), 25)

    def test_primary_summary_retains_configured_cells_and_real_final_success(self):
        config = deepcopy(self.config)
        config["core"].update(control_budget=50, iterations=[4], bootstrap_samples=0)
        config["p1"]["evaluation"]["seeds"] = [42]
        case = {"case_id": "query", "query": {"episode_id": "episode"}}
        metrics = {"success": False, "planning_score": 0., "goal_distance": 1.}
        outcome = PlanOutcome(metrics)
        outcome.append({**metrics, "success": True})
        outcome.append(metrics)
        base = {"case": case, **outcome.record()}
        records = [{"kind": "recorded-control", **base}]
        records += [{"kind": "planned", "iterations": 4, "condition": label, "seed": 42,
                     "planning_seconds": 1., "planning_calls": 2, "compute": {}, **base} for label in cells(config).values()]
        result = summarize(records, config)
        self.assertEqual(len(result["curves"]), 4)
        self.assertTrue(all(c["success_rate"]["mean"] == 1. for c in result["curves"]))
        self.assertTrue(all(c["final_state_success_rate"]["mean"] == 0. for c in result["curves"]))
        with self.assertRaises(ValueError):
            summarize(records[:-1], config)
        more = deepcopy(records)
        for row in more:
            row["case"]["case_id"] = "initially-successful"
            if row["kind"] == "planned":
                row["initial_success"] = True
                row["compute"] = {}
        for row in records:
            if row["kind"] == "planned":
                row["compute"] = {"cem_candidate_evaluations": 33.}
        combined = summarize(records+more, config)
        self.assertTrue(all(c["compute_means"]["cem_candidate_evaluations"] == 16.5 for c in combined["curves"]))
        legacy = deepcopy(config)
        legacy["core"].pop("mid_search")
        extra = [{**records[1], "condition": label} for label in ("init-mid/frozen", "init-mid/checkpoint")]
        self.assertEqual(len(summarize(records+extra, legacy)["curves"]), 6)
        with self.assertRaises(ValueError):
            summarize(records+extra, config)

    def test_runtime_cap_stops_between_four_cell_queries_and_clean_resume_skips_them(self):
        from lewm_tta.core.pipeline import evaluate, verified_report
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            config = deepcopy(self.config)
            config["experiment_dir"] = directory
            config["planning"]["iterations"] = 4
            config["core"].update(iterations=[4], runtime_budget_hours=.0001, bootstrap_samples=0)
            config["p1"]["evaluation"]["seeds"] = [42]
            query_path = Path(directory)/"queries.json"
            write_json(query_path, {"fixture": True})
            config["query_dataset"]["manifest"] = str(query_path)
            metadata = [{"transition_id": f"query-{i}", "episode_id": f"episode-{i}"} for i in range(2)]
            class Bank:
                items = metadata
                def __len__(self):
                    return len(self.items)
            metrics = {"success": False, "planning_score": 0., "goal_distance": 1.}
            outcome = PlanOutcome(metrics).record()
            class Replay:
                backend, description = "pusht", {"fixture": True}
                def __init__(self, *args):
                    pass
                def reset(self, case):
                    return None, {}
                def goal(self, case):
                    return None
                def metrics(self, observation, goal):
                    return metrics
                def execute(self, case):
                    return {**outcome, "final_success": True}
                def close(self):
                    pass
            adapter = SimpleNamespace(device=torch.device("cpu"), identity={"fixture": "model"}, scope_description=lambda: {})
            resources = SimpleNamespace(adapter=adapter, external_manifest={"fingerprint": "queries"},
                                        prepared={"fingerprint": "memory"})
            manifest = {"fingerprint": "cohort", "splits": {"validation": {"queries": 2}}}
            planned = {**outcome, "planning_seconds": .5, "planning_calls": 1, "compute": {}}
            def searched(*args, **kwargs):
                args[10]["search_calls"] += 1
                return SimpleNamespace(snapshots={4: SimpleNamespace(iterations=4)}), None, {}
            clock = SimpleNamespace(now=0.)
            def tick():
                clock.now += 1.
                return clock.now
            with ExitStack() as stack:
                def mock(name, *args, **kwargs):
                    return stack.enter_context(patch("lewm_tta.core.pipeline."+name, *args, **kwargs))
                mock("headless")
                mock("prepare", return_value=manifest)
                mock("load_manifest", return_value=manifest)
                mock("load_resources", return_value=resources)
                mock("action_coordinates", return_value=SimpleNamespace(description=lambda: {}))
                mock("ContinuationProviders", side_effect=lambda *args: SimpleNamespace(description=lambda: {}))
                mock("evaluation_bank", return_value=Bank())
                mock("FreshReplay", Replay)
                mock("planning_task", side_effect=lambda bank, i, h: SimpleNamespace(metadata=metadata[i], context=None))
                mock("goal_latent", return_value=None)
                mock("case_value", side_effect=lambda bank, i, task, q: {"case_id": metadata[i]["transition_id"], "query": metadata[i]})
                search = mock("search", side_effect=searched)
                mock("first_plan_row", return_value={"fixture": "saved plan"})
                mock("execute", return_value=planned)
                mock("synchronize")
                mock("progress", side_effect=lambda iterable, **kwargs: iterable)
                mock("time.perf_counter", side_effect=tick)
                stack.enter_context(patch("lewm_tta.interleave.checks.run_checks", return_value={"passed": True}))
                first = evaluate(config, "cpu", 1)
                self.assertFalse(first["completed"])
                self.assertEqual(first["completed_queries"], 1)
                self.assertEqual(first["record_count"], 5)
                self.assertEqual(search.call_count, 4)
                self.assertEqual(len(first["cells"]), 4)
                self.assertIn("Four paired cells", first["protocol"])
                with self.assertRaises(FileExistsError):
                    evaluate(config, "cpu", 1)
                second = evaluate(config, "cpu", 1, resume=True)
                self.assertTrue(second["completed"])
                self.assertEqual(second["completed_queries"], 2)
                self.assertEqual(second["record_count"], 10)
                self.assertEqual(second["plan_count"], 8)
                self.assertEqual(search.call_count, 8)
                self.assertEqual(second["segments"], 2)
                self.assertTrue((Path(directory)/"runs/validation/segments/report-1.json").is_file())
                self.assertEqual(verified_report(Path(directory)/"runs/validation/report.json")["summary"], second["summary"])


if __name__ == "__main__":
    unittest.main()

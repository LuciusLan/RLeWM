"""User-run synthetic contracts; no native checkpoints, encoders, or simulators.

The coding agent compiles this file only, as required by docs/research/plan.md.
"""

from copy import deepcopy
import contextlib
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from experiments.catalog import ROOT, infer_backend, profile, stage_for_backend
from experiments.cli import dispatch
from experiments.config import resolve
from experiments.provenance import archive_entry
from lewm_tta.model import Context
from lewm_tta.planning.actions import ActionCoordinates
from lewm_tta.planning.cem import solve_cem
from lewm_tta.planning.temporal.state import advance
from lewm_tta.proposals.cem import solve_guided_cem
from lewm_tta.proposals.injection import Injection
from lewm_tta.retrieval import ExactRetriever, Support
from lewm_tta.rollout.record_io import check_record_roundtrip
from lewm_tta.rollout.support_updates import validate_support
from lewm_tta.runtime import object_hash, write_json
from lewm_tta.interleave.cem import CheckpointResult, solve_interleaved_cem
from lewm_tta.interleave.config import load_config, validate_config
from lewm_tta.interleave.continuations import ContinuationProviders, splice_suffix, suffix_injection, retrieve_continuations
from lewm_tta.interleave.references import balanced_pairs, select_parents, rollout_references
from lewm_tta.interleave.reporting import summarize
from lewm_tta.interleave.session import CheckpointSession
from lewm_tta.interleave.supports import ReplayBuffer, select_support


class ToyItems(list):
    pass


class ToyMemory:
    def __init__(self):
        self.split, self.history, self.frameskip = "memory", 3, 1
        self.episode_positions = np.repeat(np.arange(16), 4)
        self.step_positions = np.tile(np.arange(4), 16)
        starts = self.episode_positions*10+self.step_positions
        actions = torch.sin(torch.arange(320, dtype=torch.float32).reshape(160, 2))*.3
        latents = torch.cat((torch.zeros(1, 2), (.45*actions[:-1]).cumsum(0)))
        self.store = SimpleNamespace(actions=actions, latents=latents)
        self.manifest = {"episodes": {"memory": list(range(16))},
                         "episode_metadata": {"offsets": list(range(0, 160, 10)), "lengths": [10]*16}}
        self.items = ToyItems({"transition_id": f"memory-{i}", "source_episode_position": int(ep),
                               "start_in_episode": int(step)}
                              for i, (ep, step) in enumerate(zip(self.episode_positions, self.step_positions)))
        self.items.starts = starts
        self.contexts = latents[torch.as_tensor(starts[:, None]+np.arange(3))]
        self.past_actions = actions[torch.as_tensor(starts[:, None]+np.arange(2))]
        self.actions = actions[torch.as_tensor(starts+2)]
        self.successors = latents[torch.as_tensor(starts+3)]
        self.identity = {"toy_observed_memory": True}

    def __len__(self):
        return len(self.items)


class ToyAdapter:
    def __init__(self):
        self.device = torch.device("cpu")
        self.action_dim, self.latent_dim = 2, 2
        self.model = torch.nn.Module()
        self.model.register_parameter("weight", torch.nn.Parameter(torch.tensor(.1)))
        self.model.register_parameter("encoder", torch.nn.Parameter(torch.tensor(0.), requires_grad=False))
        self.model.register_buffer("fixed", torch.tensor(7.))
        self.model.eval()
        self.parameters, self.parameter_names = [self.model.weight], ["weight"]

    def reset(self, seed):
        with torch.no_grad():
            self.model.weight.fill_(.1)
        self.model.zero_grad(set_to_none=True)
        self.model.eval()

    def predict(self, context, action, gradients=False):
        with torch.enable_grad() if gradients else torch.no_grad():
            return context.latents[:, -1]+self.model.weight*action

    def rollout(self, context, actions):
        values = []
        for action in actions.unbind(1):
            prediction = self.predict(context, action)
            values.append(prediction)
            context = advance(context, action, prediction)
        return torch.stack(values, 1)

    def anchor_loss(self):
        return (self.model.weight-.1).square()

    def check_fixed_state(self):
        assert not self.model.training
        assert float(self.model.encoder) == 0 and float(self.model.fixed) == 7

    def module_metrics(self):
        return {"predictor": {"parameter_tensors_with_gradient": int(self.model.weight.grad is not None)}}


class P3InterleaveTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"resolved.json"
            write_json(path, resolve(profile("p3", "interleave")[1]))
            cls.config = load_config(path)

    def setUp(self):
        self.current = Context(torch.zeros(1, 3, 2), torch.zeros(1, 2, 2))
        self.metadata = {"transition_id": "query-1", "source_episode_position": -999}
        self.goal = torch.tensor([[.15, .1]])
        self.memory, self.adapter = ToyMemory(), ToyAdapter()
        settings = {"max_windows_per_episode": 4, "min_start_step_gap": 1, "chunk_size": 32,
                    "beta_action": 1., "beta_successor": 1., "pool_size": 64}
        retriever = ExactRetriever(self.memory, {"scales": {key: 1. for key in (
            "context", "past_actions", "action", "successor")}}, settings)
        self.engine = SimpleNamespace(adapter=self.adapter, memory=self.memory, retriever=retriever,
                                      settings=self.config["adaptation"] | {"learning_rate": .001})
        self.coordinates = ActionCoordinates({"mean": [0., 0.], "std": [1., 1.]},
                                             {"raw_action_low": [-1., -1.], "raw_action_high": [1., 1.]}, 1, "cpu")

    def test_separate_dispatch_and_old_profiles(self):
        self.assertEqual(infer_backend("p3", self.config), "p3_interleave")
        self.assertEqual(stage_for_backend("p3_interleave"), "p3")
        for variant, expected in (("default", "p3c"), ("fresh", "p3c"), ("shared", "p3a"),
                                  ("proposals", "p3b"), ("temporal", "p3_temporal")):
            self.assertEqual(profile("p3", variant)[0], expected)
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            dispatch(["p3", "evaluate", "--variant", "interleave", "--dataset", "reacher", "--dry-run"])
        self.assertIn("lewm_tta.interleave.cli", output.getvalue())
        native = resolve(profile("p3", "interleave")[1], dataset="reacher")
        self.assertEqual(native["adaptation"]["learning_rate"], 1e-4)
        self.assertEqual(native["evaluation"]["seeds"], [41, 42, 43])
        self.assertTrue(native["experiment_dir"].endswith("p3-interleave"))

    def test_old_source_archive_is_verified_for_all_p3_modes(self):
        path = ROOT/"artifacts/code-archives/2f1940922fa4e0bd5334/manifest.json"
        manifest = json.loads(path.read_text())
        for backend in ("p3a", "p3b", "p3c", "p3_temporal"):
            found = archive_entry(backend, manifest["entrypoints"][backend]["code"])
            self.assertIsNotNone(found)
            self.assertEqual(found[1], ["legacy", backend])

    def test_invalid_checkpoints_pairs_and_replay_rejected(self):
        for key, value in (("checkpoints", [6]), ("checkpoints", [4, 2]), ("checkpoints", [2, 2]),
                           ("support_pairs", 17), ("references", 33), ("adaptation_depths", [0, 5])):
            config = deepcopy(self.config)
            config["interleave"][key] = value
            with self.assertRaises(ValueError):
                validate_config(config)
        config = deepcopy(self.config)
        config["interleave"]["replay"].update(policy="current-plus-replay", capacity=1)
        with self.assertRaises(ValueError):
            validate_config(config)

    def test_balanced_pairs_prevent_eighty_stream_truncation(self):
        pairs = balanced_pairs(8, list(range(5)), 16, 42)
        self.assertEqual(len(set(pairs)), 16)
        self.assertEqual({p for p, _ in pairs}, set(range(8)))
        self.assertEqual({h for _, h in pairs}, set(range(5)))
        self.assertEqual(pairs, balanced_pairs(8, list(range(5)), 16, 42))
        contexts = Context(self.current.latents.expand(16, -1, -1), self.current.past_actions.expand(16, -1, -1))
        support, cache, calls = select_support(self.engine, contexts, torch.zeros(16, 2), "fr-fixed", 42, 64)
        validate_support(self.memory, self.engine.retriever, support, 32)
        self.assertEqual(len(support.details["streams"]), 32)
        self.assertTrue(all(row["quota"] >= 1 for row in support.details["streams"].values()))
        self.assertEqual(calls["forward"], 16)
        _, _, second = select_support(self.engine, contexts, torch.zeros(16, 2), "fr-refresh", 42, 64, cache)
        self.assertEqual(second["forward"], 0)
        with self.assertRaises(ValueError):
            select_support(self.engine, Context(self.current.latents.expand(40, -1, -1),
                self.current.past_actions.expand(40, -1, -1)), torch.zeros(40, 2), "fr-fixed", 42, 64)

    def test_disabled_solver_and_initial_injection_equal_existing_cem(self):
        settings = self.config["planning"] | {"population": 8, "elites": 2, "iterations": 3}
        score = lambda actions: (actions-.2).square().flatten(1).sum(1)
        expected = solve_cem(score, settings, self.coordinates, 42)
        actual, records = solve_interleaved_cem(score, settings, self.coordinates, 42)
        torch.testing.assert_close(actual.actions, expected.actions, rtol=0, atol=0)
        self.assertEqual(actual.cost, expected.cost)
        self.assertEqual(records, [])
        injection = Injection([1], torch.full((1, 5, 2), .3), [{"kind": "exact"}], 1)
        expected = solve_guided_cem(score, settings, self.coordinates, 42, injection).plan
        actual, _ = solve_interleaved_cem(score, settings, self.coordinates, 42, initial_injection=injection)
        torch.testing.assert_close(actual.actions, expected.actions, rtol=0, atol=0)
        self.assertEqual(actual.cost, expected.cost)

    def test_cost_history_resets_when_model_changes(self):
        version = [0]
        def score(actions):
            return actions.square().flatten(1).sum(1)+version[0]*100
        def checkpoint(completed, candidates, costs, elites):
            version[0] += 1
            return CheckpointResult(None, {"checkpoint": completed}, version[0])
        settings = self.config["planning"] | {"population": 8, "elites": 2, "iterations": 3}
        plan, records = solve_interleaved_cem(score, settings, self.coordinates, 42, [1, 2], checkpoint)
        self.assertEqual([row["model_version"] for row in plan.trace], [0, 1, 2])
        self.assertGreaterEqual(plan.trace[-1]["best_scored_cost_so_far"], 200)
        self.assertEqual(plan.candidate_evaluations, 25)
        self.assertEqual(len(records), 2)

    def test_suffix_alignment_missing_sources_and_prefix_preservation(self):
        parent = torch.arange(10, dtype=torch.float32).reshape(5, 2)/10
        suffix = torch.full((2, 2), -.5)
        candidate = splice_suffix(parent, 3, suffix)
        torch.testing.assert_close(candidate[:3], parent[:3], rtol=0, atol=0)
        seeds = [{"actions": candidate, "depth": 3}, None]
        injection = suffix_injection(seeds, 6, .1, 42, parent)
        self.assertEqual(injection.slots, [1, 3, 5])
        torch.testing.assert_close(injection.unbounded_actions[0], candidate, rtol=0, atol=0)
        for actions in injection.unbounded_actions:
            torch.testing.assert_close(actions[:3], parent[:3], rtol=0, atol=0)
        with self.assertRaises(ValueError):
            splice_suffix(parent, 3, torch.zeros(3, 2))

    def test_replay_excludes_current_rows_and_preserves_global_diversity(self):
        old = Support(list(range(32)), [1/32]*32, {i: ["forward"] for i in range(32)})
        current = Support(list(range(24, 56)), [1/32]*32, {i: ["reverse"] for i in range(24, 56)})
        buffer = ReplayBuffer(64)
        buffer.insert(old, 2)
        mixed = buffer.mix(current, self.engine.retriever, 32, .5, 4, 42)
        validate_support(self.memory, self.engine.retriever, mixed, 32)
        self.assertEqual(mixed.details["current_count"], 16)
        self.assertEqual(mixed.details["replay_count"], 16)
        for i in mixed.indices:
            if i in current.indices:
                self.assertNotIn("replay", mixed.channels[i])
        buffer.insert(current, 4)
        same_checkpoint = buffer.mix(current, self.engine.retriever, 32, .5, 4, 42)
        self.assertTrue(all(buffer.rows[i]["last_selected_checkpoint"] < 4 for i in same_checkpoint.indices
                            if "replay" in same_checkpoint.channels[i]))
        small = ReplayBuffer(2)
        small.insert(Support([1, 2], [.5, .5], {1: ["f"], 2: ["f"]}), 1)
        small.insert(Support([1, 3], [.5, .5], {1: ["f"], 3: ["f"]}), 2)
        self.assertEqual(list(small.rows), [1, 3])

    def test_continuation_sources_obey_global_cap_for_multiple_fragments(self):
        config = deepcopy(self.config)
        config["planning"].update(population=16, elites=4)
        config["interleave"].update(injection_slots=15, fragments_per_pair=3)
        config["proposals"].update(max_fragments_per_episode=2, min_start_step_gap=0)
        parents = torch.zeros(4, 5, 2)
        contexts, _ = rollout_references(self.adapter, self.current, parents)
        with tempfile.TemporaryDirectory() as directory:
            config["experiment_dir"] = directory
            providers = ContinuationProviders(SimpleNamespace(engine=self.engine, memory=self.memory), config, self.coordinates)
            injection, audit = retrieve_continuations(providers, self.current, contexts, parents, [0, 1, 2, 3],
                                                     self.metadata, self.goal, 42, 2)
        exact = [row for row in injection.provenance if row["kind"] == "exact"]
        episodes = [row["source"]["source_episode_position"] for row in exact]
        self.assertTrue(all(episodes.count(ep) <= 2 for ep in set(episodes)))
        for actions, provenance in zip(injection.unbounded_actions, injection.provenance):
            depth, parent = provenance["depth"], provenance["parent_reference_index"]
            torch.testing.assert_close(actions[:depth], parents[parent, :depth], rtol=0, atol=0)
            source = provenance["source"]
            self.assertEqual(source["fragment_action_stop_row_exclusive"]-source["fragment_action_start_row"], 5-depth)
        self.assertEqual(audit["realized_slots"], len(injection.slots))

    def test_continuation_queries_use_post_update_model_and_local_refresh_keeps_forward(self):
        config = deepcopy(self.config)
        config["planning"].update(population=8, elites=2, iterations=2)
        config["interleave"].update(checkpoints=[1], references=2, injection_slots=7,
                                    support_method="fr-refresh")
        config["adaptation"]["updates"] = 2
        engine = SimpleNamespace(**{**vars(self.engine), "settings": self.engine.settings | {"updates": 2}})
        observed = []

        def inspect(providers, initial, contexts, parents, parent_ids, metadata, goal, seed, checkpoint):
            expected, _ = rollout_references(self.adapter, initial, parents)
            for actual, reference in zip(contexts, expected):
                torch.testing.assert_close(actual.latents, reference.latents, rtol=0, atol=0)
                torch.testing.assert_close(actual.past_actions, reference.past_actions, rtol=0, atol=0)
                self.assertFalse(actual.latents.requires_grad)
            observed.append(float(self.adapter.model.weight))
            return retrieve_continuations(providers, initial, contexts, parents, parent_ids,
                                           metadata, goal, seed, checkpoint)

        with tempfile.TemporaryDirectory() as directory:
            config["experiment_dir"] = directory
            providers = ContinuationProviders(SimpleNamespace(engine=engine, memory=self.memory), config, self.coordinates)
            session = CheckpointSession(engine, self.current, self.metadata, self.goal, self.coordinates,
                                        providers, config, "combined", 42)
            with patch("lewm_tta.interleave.session.retrieve_continuations", side_effect=inspect):
                _, records = solve_interleaved_cem(session.score, config["planning"], self.coordinates, 42, [1], session)
        pairs = len(records[0]["adaptation_pairs"])
        self.assertEqual(session.counters["forward_retrievals"], pairs)
        self.assertEqual(session.counters["reverse_retrievals"], 2*pairs)
        self.assertEqual(session.version, 2)
        self.assertEqual(len(observed), 1)
        self.assertGreater(observed[0], .1)

    def test_paired_four_arm_run_records_roundtrip_and_query_reset(self):
        config = deepcopy(self.config)
        config["planning"].update(population=8, elites=2, iterations=3)
        config["interleave"].update(checkpoints=[1, 2], references=2, injection_slots=7)
        config["p1"]["evaluation"]["seeds"] = [42]
        config["evaluation"]["bootstrap_samples"] = 20
        config["interleave"]["replay"]["policy"] = "current-plus-replay"
        resources = SimpleNamespace(engine=self.engine, memory=self.memory)
        rows = []
        with tempfile.TemporaryDirectory() as directory:
            config["experiment_dir"] = directory
            providers = ContinuationProviders(resources, config, self.coordinates)
            final_actions = None
            for arm in config["interleave"]["arms"]:
                session = CheckpointSession(self.engine, self.current, self.metadata, self.goal, self.coordinates,
                                            providers, config, arm, 42)
                plan, checkpoints = solve_interleaved_cem(session.score, config["planning"], self.coordinates,
                                                          42, [1, 2], session)
                adapted = session.adaptation_record()
                rows.append({"arm": arm, "seed": 42, "query": self.metadata, "goal_raw_row": 7, "cem_seed": 42,
                    "initial_injection_fingerprint": None, "cem_first_population_fingerprint": plan.first_population_fingerprint,
                    "cem_trace": plan.trace, "checkpoints": checkpoints, "adaptation": adapted,
                    "selected_actions_normalized": plan.actions.tolist(), "counters": dict(session.counters),
                    "cem_candidate_evaluations": plan.candidate_evaluations,
                    "cem_transition_predictions": plan.candidate_evaluations*5,
                    "recorded_action_mse_by_block": [.1]*5, "selected_plan_frozen_model_goal_sse": plan.cost,
                    "selected_model_goal_sse": plan.cost, "primary_seconds_with_checks": 1., "checks": {"synthetic": True}})
                if arm == "combined":
                    final_actions = plan.actions.clone()
                    self.assertEqual(adapted["updates"], 2)
                    self.assertEqual(adapted["support_presentations"], 64)
                    for update in adapted["rounds"]:
                        for item in update["support"]:
                            self.assertEqual(item["transition_id"], self.memory.items[item["memory_index"]]["transition_id"])
            summary = summarize(rows, config)
            self.assertIsNotNone(summary["common_frozen_proxy_interaction"])
            check_record_roundtrip(rows, config, summarize)
            # Intervening query/treatment work cannot change a restarted query.
            repeated = CheckpointSession(self.engine, self.current, self.metadata, self.goal, self.coordinates,
                                         providers, config, "combined", 42)
            plan, _ = solve_interleaved_cem(repeated.score, config["planning"], self.coordinates, 42, [1, 2], repeated)
            torch.testing.assert_close(plan.actions, final_actions, rtol=0, atol=0)
            broken = deepcopy(rows)
            broken[-1]["cem_trace"][-1]["best_cost_model_version"] = 0
            with self.assertRaisesRegex(ValueError, "stale"):
                summarize(broken, config)


if __name__ == "__main__":
    unittest.main()

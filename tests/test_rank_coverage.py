"""Synthetic rank/weight/statistics checks; no Torch, model or dataset imports."""

import gzip
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np

from lewm_tta.rank_coverage.analysis import analyze, validate_matrix
from lewm_tta.rank_coverage.metrics import cutoff, fr_metrics, group_metrics, pair_metrics, pair_record
from lewm_tta.rank_coverage.pipeline import support_ledger
from lewm_tta.rank_coverage.statistics import summarize, write_tables
from lewm_tta.runtime import file_hash, write_json


def example_pair():
    forward = np.arange(1, 9)
    reverse = np.asarray([8, 1, 2, 3, 4, 5, 6, 7])
    return pair_record(forward, reverse, [0, 1, 2, 3], [1, 2, 3, 4], [1, 2], [25., 50., 100.])


def ledger(episode=2):
    return [{"index": 0, "source_episode_position": episode, "weight": .5,
             "origins": [{"action": 0, "direction": "forward", "forward_rank": 1, "reverse_rank": 8, "loss_contribution": .25},
                         {"action": 0, "direction": "reverse", "forward_rank": 1, "reverse_rank": 8, "loss_contribution": .25}]},
            {"index": 1, "source_episode_position": episode + 1, "weight": .5,
             "origins": [{"action": 0, "direction": "forward", "forward_rank": 2, "reverse_rank": 1, "loss_contribution": .25},
                         {"action": 0, "direction": "reverse", "forward_rank": 2, "reverse_rank": 1, "loss_contribution": .25}]}]


def settings():
    return {"percentages": [25., 50., 100.], "prefix_sizes": [1, 2, 4], "support_size": 2,
            "query_modes": ["recorded-action"], "seeds": [42, 43, 44],
            "provisional": {"provisional_actions": 2},
            "evaluation": {"bootstrap_samples": 100, "bootstrap_seed": 17}}


def group(query="query-a", episode=10):
    return {"query": {"transition_id": query, "source_episode_position": episode},
            "mode": "recorded-action", "seed": 42, "pairs": [example_pair()],
            "actions_normalized": [[0., 0.]], "fr_fixed_support": ledger()}


class RankCoverageTests(unittest.TestCase):
    def test_support_ledger_reconstructs_existing_single_and_shared_weights(self):
        def ranking(values):
            return SimpleNamespace(rank_of=lambda index: values[index])
        pairs = [(ranking([1, 2, 3]), ranking([3, 1, 2])),
                 (ranking([3, 2, 1]), ranking([1, 2, 3]))]
        retriever = SimpleNamespace(episode_positions=[5, 6, 7], step_positions=[10, 20, 30])
        support = SimpleNamespace(indices=[0, 1, 2], weights=[.5, .25, .25], details={},
            channels={0: ["forward", "reverse"], 1: ["forward"], 2: ["reverse"]})
        rows = support_ledger(support, pairs, retriever, .5, shared=False)
        self.assertEqual([origin["loss_contribution"] for origin in rows[0]["origins"]], [.25, .25])
        self.assertEqual(fr_metrics(rows, 3, [100.], 3)["p/100/fr/channel_loss_mass_retained"], 1.)
        masses = {"action-0/forward": .1, "action-0/reverse": .2,
                  "action-1/forward": .3, "action-1/reverse": .4}
        support.weights = [.35, .25, .4]
        support.channels = {0: ["action-0/forward", "action-1/forward"],
                            1: ["action-0/forward", "action-0/reverse"], 2: ["action-1/reverse"]}
        support.details = {"streams": {name: {"effective_loss_mass": mass} for name, mass in masses.items()}}
        rows = support_ledger(support, pairs, retriever, .5, shared=True)
        self.assertEqual(rows[0]["origins"][1]["action"], 1)
        self.assertEqual(rows[0]["origins"][1]["reverse_rank"], 1)
        metrics = fr_metrics(rows, 3, [34., 100.], 3)
        self.assertEqual(metrics["p/34/fr/unique_retained_fraction"], 1.)
        self.assertAlmostEqual(metrics["p/34/fr/channel_loss_mass_retained"], .95)

    def test_raw_pool_size_does_not_imply_diverse_batch_coverage(self):
        pair = example_pair()
        # A valid greedy support when nearby starts in the same episode block indices 2 and 4.
        pair["consensus_support"] = [[1, 2, 1], [3, 4, 3], [5, 6, 5]]
        values = pair_metrics(pair, [2], 3)
        self.assertEqual(values["p/50/pair/mutual_pool_count"], 3)
        self.assertEqual(values["p/50/pair/mutual_pool_has_budget"], 1.)
        self.assertEqual(values["p/50/pair/consensus_retained_count"], 2)
        self.assertEqual(values["p/50/pair/consensus_full_batch"], 0.)

    def test_exact_whole_corpus_counts_and_asymmetric_directional_coverage(self):
        pair = example_pair()
        self.assertEqual(pair["forward_prefix"][0], [0, 1, 8])
        self.assertEqual([row["intersection_count"] for row in pair["full_corpus"]], [1, 3, 8])
        metrics = pair_metrics(pair, [1, 2, 4], 2)
        self.assertEqual(metrics["p/50/topk/2/forward_coverage"], .5)
        self.assertEqual(metrics["p/50/topk/2/reverse_coverage"], 1.)
        self.assertEqual(metrics["p/50/topk/2/forward_minus_reverse_coverage"], -.5)
        self.assertAlmostEqual(metrics["p/25/pair/mutual_jaccard"], 1 / 3)
        self.assertEqual(metrics["p/25/pair/consensus_retained_count"], 1)
        self.assertEqual(metrics["p/50/pair/consensus_full_batch"], 1.)

    def test_channel_overlap_retains_samples_but_can_lose_loss_mass(self):
        metrics = fr_metrics(ledger(), 8, [25., 50., 100.], 2)
        self.assertEqual(metrics["p/25/fr/unique_retained_fraction"], 1.)
        self.assertEqual(metrics["p/25/fr/original_sample_weight_retained"], 1.)
        self.assertEqual(metrics["p/25/fr/channel_loss_mass_retained"], .75)
        self.assertEqual(metrics["p/25/fr/both_ranks_unique_retained_count"], 1)
        self.assertEqual(metrics["p/100/fr/channel_loss_mass_retained"], 1.)

    def test_p3_origins_use_their_own_action_opposite_rank(self):
        rows = ledger()
        rows[0]["origins"][1].update(action=1, forward_rank=2, reverse_rank=7)
        self.assertEqual(fr_metrics(rows, 8, [25., 100.], 2)["p/25/fr/unique_retained_count"], 2)
        rows[0]["origins"][1]["forward_rank"] = 3
        self.assertEqual(fr_metrics(rows, 8, [25., 100.], 2)["p/25/fr/unique_retained_count"], 1)

    def test_direction_swap_preserves_mutual_coverage(self):
        pair = example_pair()
        forward = np.asarray([8, 1, 2, 3, 4, 5, 6, 7])
        reverse = np.arange(1, 9)
        swapped = pair_record(forward, reverse, [1, 2, 3, 4], [0, 1, 2, 3], [1, 2], [25., 50., 100.])
        a, b = pair_metrics(pair, [2], 2), pair_metrics(swapped, [2], 2)
        self.assertEqual(a["p/50/pair/mutual_pool_count"], b["p/50/pair/mutual_pool_count"])
        self.assertEqual(a["p/50/topk/2/forward_coverage"], b["p/50/topk/2/reverse_coverage"])

    def test_strict_thresholds_and_rank_validation(self):
        self.assertEqual(cutoff(1959664, .001), 20)
        self.assertEqual(cutoff(1959664, .1), 1960)
        self.assertEqual(cutoff(8, 100), 8)
        for percent in (0., -1., 101., float("nan")):
            with self.assertRaises(ValueError):
                cutoff(8, percent)
        with self.assertRaises(ValueError):
            pair_record(np.asarray([1, 1, 3]), np.asarray([1, 2, 3]), [0], [0], [0], [100.])
        rows = ledger()
        rows[0]["origins"][0]["loss_contribution"] = .1
        with self.assertRaises(ValueError):
            fr_metrics(rows, 8, [100.], 2)

    def test_100_percent_baseline_and_monotonic_retention(self):
        metrics = group_metrics(group(), [1, 2, 4], [25., 50., 100.], 2)
        for suffix in ("fr/unique_retained_fraction", "fr/channel_loss_mass_retained", "pair/consensus_budget_fill_fraction"):
            values = [metrics[f"p/{p}/{suffix}"] for p in (25, 50, 100)]
            self.assertEqual(values, sorted(values))
            self.assertEqual(values[-1], 1.)

    def test_equal_episode_weighting_and_no_fake_seed_replication(self):
        rows = [group("a", 10), group("b", 10), group("c", 11)]
        # Two contexts in episode 10 retain all samples. One in episode 11 retains none at p=25.
        for sample in rows[-1]["fr_fixed_support"]:
            for origin in sample["origins"]:
                origin.update(forward_rank=7, reverse_rank=8)
        result = summarize(rows, settings())["recorded-action"]
        self.assertEqual(result["statistics"]["p/25/fr/unique_retained_fraction"]["mean"], .5)
        self.assertEqual(result["seeds_per_query"], [1])
        self.assertEqual(result["episodes"], 2)
        self.assertEqual(result, summarize(rows, settings())["recorded-action"])
        self.assertEqual(validate_matrix(rows, settings(), 3)["queries"], 3)
        with self.assertRaises(ValueError):
            validate_matrix(rows + [rows[0]], settings())
        protocol = settings()
        protocol["query_modes"] = ["recorded-action", "provisional-actions"]
        with self.assertRaises(ValueError):
            validate_matrix(rows, protocol)

    def test_exports_csv_and_markdown_without_model_imports(self):
        config = settings()
        result = summarize([group()], config)
        self.assertIsNone(result["recorded-action"]["statistics"]["p/25/fr/unique_retained_fraction"]["mean_ci95"])
        with tempfile.TemporaryDirectory() as temporary:
            write_tables(result, config, Path(temporary))
            self.assertIn("ci95_low", (Path(temporary) / "summary.csv").read_text())
            self.assertIn("Consensus full-B", (Path(temporary) / "summary.md").read_text())

    def test_reanalysis_checksums_and_reproduces_saved_statistics(self):
        rows, config = [group("a", 10), group("b", 11)], settings()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            records = root / "records.jsonl.gz"
            with gzip.open(records, "wt") as stream:
                for row in rows:
                    stream.write(json.dumps(row) + "\n")
            source = {"stage": "rank-coverage", "completed": True, "split": "validation", "config": config,
                      "records_file": str(records), "records_sha256": file_hash(records),
                      "record_count": len(rows), "memory_windows": 8}
            write_json(root / "source.json", source)
            result = analyze(root / "source.json", root / "derived")
            self.assertEqual(result["summary"], summarize(rows, config))
            source["records_sha256"] = "incorrect"
            write_json(root / "source.json", source)
            with self.assertRaises(ValueError):
                analyze(root / "source.json", root / "bad")


if __name__ == "__main__":
    unittest.main()

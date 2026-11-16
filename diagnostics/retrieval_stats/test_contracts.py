"""User-run synthetic metric/trace contracts; no datasets, simulator or pretrained weights.

PYTHONPATH=codes python -m unittest discover -s codes/diagnostics/retrieval_stats -p test_contracts.py
"""

from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

from diagnostics.common import identity, sha, write
from diagnostics.retrieval_stats.metrics import compare, episode_overlap
from diagnostics.retrieval_stats.run import load_query, output_directory, records_per_query, save_query, seal


class Metrics(unittest.TestCase):
    def test_default_updates_use_reverse_support_in_a_separate_directory(self):
        import argparse
        from diagnostics.retrieval_stats import add_parser
        parser = argparse.ArgumentParser()
        add_parser(parser.add_subparsers(dest='command'))
        args = parser.parse_args(['retrieval-stats'])
        self.assertEqual(args.methods, ['reverse'])
        self.assertEqual(args.learning_rates, [1e-4, 1e-5, 1e-6])
        self.assertEqual(args.scope, 'last-block')
        self.assertEqual(output_directory(args).name, 'retrieval-stats-reverse')
        predictor = parser.parse_args(['retrieval-stats', '--scope', 'predictor'])
        self.assertEqual(output_directory(predictor).name, 'retrieval-stats-reverse-predictor')

    def test_intersection_counts_churn_and_reordering(self):
        result = compare([1, 2, 3, 4], [4, 3, 5, 6])
        self.assertEqual(result['intersection'], 2)
        self.assertEqual(result['added'], 2)
        self.assertEqual(result['removed'], 2)
        self.assertEqual(result['new_fraction'], .5)
        self.assertEqual(result['jaccard'], 1/3)
        reordered = compare([1, 2], [2, 1])
        self.assertEqual(reordered['set_changed'], 0)
        self.assertEqual(reordered['order_changed'], 1)
        self.assertEqual(reordered['retained_rank_displacement'], 1)

    def test_weight_changes_without_sample_changes(self):
        result = compare([1, 2], [2, 1], [.75, .25], [.5, .5])
        self.assertEqual(result['set_changed'], 0)
        self.assertEqual(result['weight_total_variation'], .25)
        self.assertEqual(compare([1], [2], [1.], [1.])['weight_total_variation'], 1)
        with self.assertRaises(ValueError):
            compare([1], [1], [.5], [1.])

    def test_empty_shortfall_and_episode_identity(self):
        self.assertIsNone(compare([], [])['jaccard'])
        self.assertEqual(compare([1, 2], [2])['retained_fraction'], .5)
        self.assertEqual(compare([1, 2], [2])['new_fraction'], 0)
        self.assertEqual(episode_overlap([0, 1], [2], [10, 10, 10])['episode_intersection'], 1)
        with self.assertRaises(ValueError):
            compare([1, 1], [2])

    def test_compressed_checkpoint_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'query.json.gz'
            value = {'request': identity('probe'), 'query_index': 7, 'records': [{'intersection': 3}]}
            save_query(path, value)
            self.assertEqual(load_query(path, value['request'], 7), value)
            with self.assertRaises(ValueError):
                load_query(path, value['request'], 8)

    def test_episode_cluster_average_is_not_treated_as_independent_seeds(self):
        from diagnostics.retrieval_stats.analysis import summarize
        rows = [{'dataset': 'synthetic', 'method': 'fr-fixed', 'learning_rate': 1e-4,
                 'comparison': 'reverse_after_update', 'selection': 'raw', 'k': 2, 'position': 0,
                 'seed': seed, 'query_id': f'q{episode}', 'episode_id': episode,
                 'intersection': episode*2} for episode in (0, 1) for seed in (41, 42, 43)]
        summary = summarize(rows, 1000)
        row = next(r for r in summary if r['position'] == 'all' and r['seed'] == 'all')
        self.assertEqual((row['mean'], row['episodes'], row['seeds'], row['observations']), (1, 2, 3, 6))
        self.assertEqual((row['ci95_low'], row['ci95_high']), (0, 2))

    def test_scope_pair_requires_matched_data_settings_and_runtime(self):
        from diagnostics.retrieval_stats.comparison import MATCHED, validate_pair, validate_runtime
        current = dict.fromkeys(MATCHED, 'same') | {'scope': 'predictor', 'code': {'new': 'hash'}}
        previous = dict.fromkeys(MATCHED, 'same') | {'scope': 'last-block', 'code': {'old': 'hash'}}
        validate_pair(current, previous)  # Old diagnostic code identity is preserved, not relabeled.
        with self.assertRaisesRegex(ValueError, 'learning_rates'):
            validate_pair({**current, 'learning_rates': [1e-5]}, previous)
        validate_runtime({'scope': {'scope': 'predictor'}, 'model': 'same'},
                         {'scope': {'scope': 'last-block'}, 'model': 'same'})
        with self.assertRaisesRegex(ValueError, 'query_cache'):
            validate_runtime({'scope': {'scope': 'predictor'}, 'query_cache': 'changed'},
                             {'scope': {'scope': 'last-block'}, 'query_cache': 'original'})

    def test_scope_differences_are_paired_before_episode_bootstrap(self):
        from diagnostics.retrieval_stats.analysis import summarize
        from diagnostics.retrieval_stats.comparison import paired_rows
        base = [{'dataset': 'synthetic', 'scope': 'last-block', 'method': 'reverse', 'learning_rate': 1e-4,
                 'comparison': 'reverse_after_update', 'selection': 'diverse', 'k': 32, 'position': 0,
                 'seed': seed, 'query_id': f'q{episode}', 'episode_id': episode, 'new_fraction': episode*.1,
                 'intersection': 32-episode, 'jaccard': .8, 'set_changed': 1, 'prediction_correction_mse': .01}
                for episode in (0, 1) for seed in (41, 42, 43)]
        current = [{**row, 'scope': 'predictor', 'new_fraction': row['new_fraction']+.125,
                    'prediction_correction_mse': .04} for row in reversed(base)]
        effects = paired_rows(current, base)
        summary = summarize(effects, 1000)
        effect = next(row for row in summary if row['metric'] == 'new_fraction' and row['seed'] == row['position'] == 'all')
        self.assertEqual((effect['mean'], effect['episodes'], effect['seeds']), (.125, 2, 3))
        self.assertEqual((effect['ci95_low'], effect['ci95_high']), (.125, .125))
        with self.assertRaisesRegex(ValueError, 'unmatched'):
            paired_rows(current[:-1], base)

    def test_old_last_block_receipts_are_read_without_rewriting(self):
        from diagnostics.retrieval_stats.analysis import saved_dataset
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            request = {'scope': 'last-block', 'queries': 1, 'horizon': 1, 'ks': [32], 'methods': ['reverse'],
                       'seeds': [42], 'learning_rates': [1e-4]}
            write(root/'manifest.json', seal(request))
            write(root/'runtime.json', {'scope': {'scope': 'last-block'}})
            path = root/'queries/00000.json.gz'
            # Legacy checkpoints intentionally lack a per-row scope field.
            records = [{'query_id': 'q', 'intersection': 30} for _ in range(records_per_query(request))]
            save_query(path, {'request': identity(request), 'query_index': 0, 'query_id': 'q', 'records': records})
            write(root/'measurement.json', {'completed': True, 'request': identity(request),
                'files': {'queries/00000.json.gz': sha(path)}, 'runtime_sha256': sha(root/'runtime.json')})
            before = {str(p): sha(p) for p in root.rglob('*') if p.is_file()}
            _, loaded, _, _ = saved_dataset(root, 'synthetic')
            self.assertTrue(all(row['scope'] == 'last-block' for row in loaded))
            self.assertEqual(before, {str(p): sha(p) for p in root.rglob('*') if p.is_file()})

    def test_scope_pair_checks_preupdate_original_context_retrieval(self):
        from diagnostics.retrieval_stats.comparison import paired_rows
        row = {'dataset': 'synthetic', 'scope': 'last-block', 'query_id': 'q', 'episode_id': 1,
               'seed': 42, 'method': 'reverse', 'learning_rate': 1e-4, 'position': 0,
               'comparison': 'forward_reverse_before', 'selection': 'raw', 'k': 32,
               'before_size': 32, 'after_size': 32, 'intersection': 6, 'jaccard': 6/58,
               'episode_intersection': 5}
        self.assertEqual(paired_rows([{**row, 'scope': 'predictor'}], [row]), [])
        with self.assertRaisesRegex(ValueError, 'Pre-adaptation'):
            paired_rows([{**row, 'scope': 'predictor', 'intersection': 7}], [row])

    def test_probe_keeps_temporal_u1_trajectory_and_optimizer_steps(self):
        import torch
        from lewm_tta.adaptation import AdaptationEngine
        from lewm_tta.cache import LatentBank, Query
        from lewm_tta.model import Context
        from lewm_tta.retrieval import ExactRetriever
        from lewm_tta.rollout.temporal import run_scheduled_rollout
        from diagnostics.retrieval_stats.measure import trace_query

        class Adapter:
            device, action_dim = torch.device('cpu'), 1

            def __init__(self):
                self.model = torch.nn.Linear(1, 1, bias=False)
                with torch.no_grad():
                    self.model.weight.fill_(.5)
                self.parameters = list(self.model.parameters())
                self.parameter_names = ['weight']
                self.anchor = self.model.weight.detach().clone()

            def reset(self, seed):
                with torch.no_grad():
                    self.model.weight.copy_(self.anchor)

            def predict(self, context, action, gradients=False):
                with torch.enable_grad() if gradients else torch.no_grad():
                    return self.model(context.latents[:, -1]+action)

            def check_fixed_state(self, **kwargs):
                pass

            def anchor_loss(self):
                return (self.model.weight-self.anchor).square().mean()

            def module_metrics(self):
                return {'synthetic': {'parameter_tensors_with_gradient': int(self.model.weight.grad is not None)}}

        adapter = Adapter()
        memory = LatentBank('memory', torch.arange(6.).reshape(6, 1, 1), torch.empty(6, 0, 1),
            torch.arange(6.).reshape(6, 1)/10, torch.arange(6.).reshape(6, 1),
            [{'transition_id': f'm{i}', 'source_episode_position': i, 'start_in_episode': 0} for i in range(6)], {})
        settings = {'support_size': 2, 'learning_rate': 1e-4, 'clip_norm': 1., 'alpha': .5, 'anchor_weight': 0}
        retriever = ExactRetriever(memory, {'scales': dict.fromkeys(('context', 'past_actions', 'action', 'successor'), 1.)},
            {'chunk_size': 8, 'pool_size': 2, 'beta_action': 1., 'beta_successor': 1.})
        engine = AdaptationEngine(adapter, memory, retriever, settings)
        resources = SimpleNamespace(engine=engine, memory=memory, adapter=adapter)
        query = Query(Context(torch.tensor([[[1.]]]), torch.empty(1, 0, 1)), torch.tensor([[.2]]),
                      {'transition_id': 'q1', 'source_episode_position': -1, 'episode_id': 99})
        actions = torch.tensor([[[.2], [.3], [.4]]])
        schedule = {'placement': 'each-step', 'updates': 1, 'persistence': 'trajectory'}
        for method in ('reverse', 'fr-fixed'):
            with self.subTest(method=method):
                _, reference = run_scheduled_rollout(engine, query, actions, method, 42, schedule)
                final = adapter.model.weight.detach().clone()
                rows = trace_query(resources, query, actions, method, 42, 1e-4, [2], schedule)
                torch.testing.assert_close(adapter.model.weight, final, rtol=0, atol=0)
                request = {'horizon': 3, 'ks': [2], 'methods': [method], 'seeds': [42], 'learning_rates': [1e-4]}
                self.assertEqual(len(rows), records_per_query(request))
                for position in range(3):
                    values = [row for row in rows if row['position'] == position]
                    self.assertTrue(all(row['prediction_correction_mse'] == reference['rounds'][position]['prediction_correction_mse']
                                        for row in values))
                self.assertEqual(sum(row['comparison'] == 'reverse_between_positions' for row in rows), 4)


if __name__ == '__main__':
    unittest.main()

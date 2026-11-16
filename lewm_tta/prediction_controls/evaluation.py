"""Fresh recorded-rollout orchestration, isolated from prediction and scoring."""

import gzip
import json
from pathlib import Path
import time

import numpy as np
import torch

from ablations.saved_runs import identity, immutable_json, read, record_rows, require
from experiments.progress import progress
from ..rollout.cohorts import evaluation_bank, observed_targets, prepare as prepare_cohort, rollout_inputs
from ..rollout.inference import latent_errors
from ..rollout.resources import load_resources
from ..runtime import file_hash, write_json
from . import cache, references
from .configuration import STAGE, cases, key, load_lock
from .prediction import correction, rollout
from .reporting import compact, selection, summarize, write_tables


def resources_for(base, spec, validation, args):
    manifest = prepare_cohort(base, quiet=True)
    resources = load_resources(base, args.device, args.cpu_threads)
    require(resources.adapter.identity == validation['model'], 'Frozen model/runtime differs from saved TTT')
    require(resources.prepared['fingerprint'] == validation['p1_prepared_fingerprint'], 'Prepared P1 source changed')
    require(manifest['query_source_fingerprint'] == validation['query_source_fingerprint'], 'Fresh query source changed')
    return resources, manifest


def check(resources, manifest, spec, values, metadata, ref_rows):
    """User-run acceptance: formulas, cache parity, support and zero-gain controls."""
    a = resources.adapter
    cache.freeze(a, spec['seeds'][0])
    # Independent small numerical fixtures make direction/sign mistakes visible.
    base = torch.tensor([[1., 2.]], device=a.device)
    current = torch.tensor([[3., 4.]], device=a.device)
    successors = torch.tensor([[4., 8.], [9., 7.]], device=a.device)
    old = torch.tensor([[1., 2.], [3., 4.]], device=a.device)
    predicted = torch.tensor([[2., 3.], [4., 5.]], device=a.device)
    weights = torch.tensor([.25, .75], device=a.device)
    torch.testing.assert_close(correction('residual-transfer', base, current, successors, old, predicted, weights, 1.),
                               torch.tensor([[5.25, 4.75]], device=a.device), rtol=0, atol=0)
    torch.testing.assert_close(correction('delta-transfer', base, current, successors, old, predicted, weights, 1.),
                               torch.tensor([[8.25, 7.75]], device=a.device), rtol=0, atol=0)
    torch.testing.assert_close(correction('residual-transfer', base, current, successors, old, predicted, weights, 0.), base, rtol=0, atol=0)
    torch.testing.assert_close(correction('delta-transfer', base, current, successors, old, predicted, weights, 0.), current, rtol=0, atol=0)
    from ..model import Context
    indices = torch.tensor(sorted(set([0, len(resources.memory)//2, len(resources.memory)-1])), device=a.device)
    memory = resources.memory
    direct = a.predict(Context(memory.contexts[indices], memory.past_actions[indices]), memory.actions[indices])
    cached = torch.from_numpy(np.array(values[indices.cpu().numpy()], copy=True)).to(a.device)
    torch.testing.assert_close(direct, cached, rtol=1e-4, atol=1e-5)
    bank = evaluation_bank(resources, manifest, 'validation', 1)
    query, actions, targets = rollout_inputs(bank, 0, spec['horizon'])
    start, seed = int(bank.start_rows[0]), spec['seeds'][0]
    outputs = {}
    fixtures = cases(spec)
    for item in fixtures:
        if item['selector'] is not None and item['gain'] not in (0, 1):
            continue
        result, audit = rollout(resources, query, actions, item, values, spec, seed)
        outputs[key(item)] = result
        if item['selector'] is not None:
            require(audit['first_support_fingerprint'] == ref_rows[(start, seed, item['selector'])]['first_support_fingerprint'],
                    'Initial support differs from saved TTT')
    for selector in spec['selectors']:
        for control, baseline in (('residual-transfer', 'frozen'), ('delta-transfer', 'persistence')):
            name = key({'control': control, 'selector': selector, 'gain': 0})
            torch.testing.assert_close(outputs[name], outputs[baseline], rtol=0, atol=0)
    # Query truth is accessed by the evaluator only after predictions finish.
    errors = latent_errors(outputs['frozen'], observed_targets(bank, targets))
    require(np.allclose(errors, ref_rows[(start, seed, 'frozen')]['mse_by_block'], rtol=1e-5, atol=1e-7),
            'Frozen rollout differs from saved TTT reference')
    result = {'weighted_correction_formulas': True, 'zero_gain_equivalences': True,
        'cached_vs_direct_memory_predictions': True, 'initial_support_matches_ttt': True,
        'frozen_matches_saved_reference': True, 'model_frozen': True, 'target_isolation': True,
        'prediction_cache': metadata['fingerprint']}
    print(json.dumps(result, indent=2), flush=True)
    return result


def completed(directory, spec, code, split, limit, chosen):
    path = directory/'report.json'
    if not path.is_file():
        return None
    report = read(path)
    require(report['completed'] and report['stage'] == STAGE and report['specification'] == spec
            and report['code'] == code and report['split'] == split and report['query_limit'] == limit
            and report['cases'] == cases(spec, chosen), 'Existing result differs; choose another --run-name/output directory')
    rows = list(record_rows(report))
    require(len(rows) == report['record_count'], 'Saved record count differs')
    require({r['case'] for r in rows} == {key(c) for c in report['cases']}
            and len(rows) == len(report['cases'])*len(spec['seeds'])*report['cohort']['queries'],
            'Saved comparison matrix is incomplete')
    require(all(r['optimizer_steps'] == 0 and all(r['checks'].values()) for r in rows), 'Saved frozen-control checks failed')
    return report, rows


def write_lock(directory, report):
    spec = report['specification']
    if report['split'] == 'validation' and not report['limited_run']:
        lock = {'stage': STAGE, 'specification': spec, 'code': report['code'],
            'validation_report': str(directory/'report.json'), 'validation_sha256': file_hash(directory/'report.json'),
            'selected_gains': report['selected_gains'], 'test_cases': report['test_cases'],
            'prediction_cache': report['prediction_cache']['fingerprint'],
            'cohort_fingerprint': report['manifest_fingerprint'], 'references': spec['references']}
        lock['fingerprint'] = identity(lock)
        immutable_json(Path(spec['experiment_dir'])/'locks/validation.json', lock)


def finalize(directory, report, rows, ref_rows):
    spec = report['specification']
    starts = report['cohort']['raw_start_rows']
    require({r['case'] for r in rows} == {key(c) for c in report['cases']}
            and len(rows) == len(report['cases'])*len(spec['seeds'])*len(starts), 'Incomplete result matrix')
    ref = references.projection(ref_rows, spec['selectors'], starts)
    summary, per_seed = summarize([compact(r) for r in rows], ref, spec)
    report['summary'] = summary
    if report['split'] == 'validation' and not report['limited_run']:
        report['selected_gains'] = selection(summary, spec)
        report['test_cases'] = cases(spec, report['selected_gains'])
    report['completed'] = True
    write_json(directory/'report.json', report)
    write_tables(directory, report, per_seed)
    write_lock(directory, report)
    return report


def evaluate(root, spec, base, code, validation, args, *, analyze=False):
    split = args.phase if not analyze else args.split
    lock = load_lock(spec, code) if split == 'test' else None
    chosen = lock['selected_gains'] if lock else None
    reference, ref_rows = references.load(root, spec, split, validation)
    run_name = args.run_name or (split if args.limit is None else f'{split}-smoke-q{args.limit}')
    require(run_name and Path(run_name).name == run_name and run_name not in ('.', '..'), 'Invalid run name')
    directory = Path(spec['experiment_dir'])/'runs'/run_name
    old = completed(directory, spec, code, split, args.limit, chosen)
    if old:
        report, rows = old
        # Rebuild derived tables from verified rows without replacing the locked report.
        ref = references.projection(ref_rows, spec['selectors'], report['cohort']['raw_start_rows'])
        summary, per_seed = summarize([compact(r) for r in rows], ref, spec)
        require(summary == report['summary'], 'Stored summary differs from verified records')
        if split == 'validation' and not report['limited_run']:
            require(report['selected_gains'] == selection(summary, spec)
                    and report['test_cases'] == cases(spec, report['selected_gains']), 'Saved validation gains differ')
        write_tables(directory, report, per_seed)
        write_lock(directory, report)
        print(f'Using completed {split} run: {directory}/summary.md', flush=True)
        return report
    require(not analyze, f'No completed result to analyze: {directory}')
    require(not directory.exists(), f'Partial run preserved: {directory}; choose another --run-name')
    resources, manifest = resources_for(base, spec, validation, args)
    references.verify_runtime(resources, manifest, reference, split)
    values, cache_metadata = cache.load(resources, spec)
    if lock:
        require(cache_metadata['fingerprint'] == lock['prediction_cache']
                and manifest['fingerprint'] == lock['cohort_fingerprint'], 'Locked cache/cohort changed')
    bank = evaluation_bank(resources, manifest, split, args.limit)
    selected_cases = cases(spec, chosen)
    starts = [int(v) for v in bank.start_rows]
    directory.mkdir(parents=True)
    records_path = directory/'records.jsonl.gz'
    immutable_json(directory/'execution.json', {'specification': spec, 'base_config': base, 'code': code,
        'split': split, 'query_limit': args.limit, 'cases': selected_cases,
        'reference': spec['references'][split], 'prediction_cache': cache_metadata})
    rows, elapsed = [], time.perf_counter()
    checks = {'initial_support_matches_ttt': True, 'zero_gain_equivalences': True,
              'frozen_matches_saved_reference': True, 'all_weights_and_buffers_frozen': True,
              'no_query_target_in_prediction': True, 'complete_paired_matrix': True}
    with gzip.open(records_path, 'xt', encoding='utf-8', compresslevel=6) as stream:
        with progress(range(len(bank)), config=spec, desc=f'Prediction controls {split}') as bar:
            for index in bar:
                query, actions, target_rows = rollout_inputs(bank, index, spec['horizon'])
                start = int(query.metadata['raw_observation_rows'][0])
                for seed in spec['seeds']:
                    predictions = {}
                    audits = {}
                    for item in selected_cases:
                        name = key(item)
                        prediction, audit = rollout(resources, query, actions, item, values, spec, seed)
                        predictions[name], audits[name] = prediction, audit
                        if item['selector']:
                            expected = ref_rows[(start, seed, item['selector'])]
                            require(audit['first_support_fingerprint'] == expected['first_support_fingerprint'],
                                    f'Initial support differs from TTT: {name}/{start}/{seed}')
                        if item['gain'] == 0:
                            baseline = 'frozen' if item['control'] == 'residual-transfer' else 'persistence'
                            torch.testing.assert_close(prediction, predictions[baseline], rtol=0, atol=0)
                    # No intermediate observed frame has entered any rollout above.
                    targets = observed_targets(bank, target_rows)
                    for item in selected_cases:
                        name = key(item)
                        errors = latent_errors(predictions[name], targets)
                        expected = ref_rows[(start, seed, 'frozen')]
                        require(query.metadata == expected['query'] and target_rows.tolist() == expected['target_raw_rows'],
                                'Query/target metadata differs from saved TTT')
                        if name == 'frozen':
                            require(np.allclose(errors, expected['mse_by_block'], rtol=1e-5, atol=1e-7),
                                    'Frozen predictions differ from saved reference')
                        audit = audits[name]
                        row = {'case': name, 'setting': item, 'seed': seed, 'start': start,
                            'episode': query.metadata['source_episode_position'], 'query': query.metadata,
                            'target_raw_rows': target_rows.tolist(), 'mse_by_block': errors, **audit}
                        stream.write(json.dumps(row, separators=(',', ':'), allow_nan=False)+'\n')
                        rows.append(compact(row))
                stream.flush()
    cohort = dict(manifest['splits'][split])
    cohort.update(raw_start_rows=starts, queries=len(starts), episodes=len({r['episode'] for r in rows}))
    report = {'completed': False, 'stage': STAGE, 'split': split, 'specification': spec,
        'config': base, 'code': code, 'cases': selected_cases, 'query_limit': args.limit,
        'limited_run': args.limit is not None, 'manifest_fingerprint': manifest['fingerprint'],
        'model': resources.adapter.identity, 'p1_prepared_fingerprint': resources.prepared['fingerprint'],
        'query_source_fingerprint': manifest['query_source_fingerprint'], 'cohort': cohort,
        'records_file': str(records_path), 'records_sha256': file_hash(records_path), 'record_count': len(rows),
        'prediction_cache': cache_metadata, 'checks': checks, 'reference': spec['references'][split],
        'validation_lock': lock['fingerprint'] if lock else None, 'evaluation_seconds': time.perf_counter()-elapsed,
        'protocol': {'history': 3, 'frameskip': 5, 'horizon': 5, 'context_feedback': 'predicted',
            'support_clock': 'reselect at each imagined position with frozen F0 reverse key',
            'support_matching': 'identical initial support; later support is policy-dependent',
            'ttt_reference': 'last-block temporal-U1 LR5e-5; reused, not rerun',
            'model_updates': 0, 'bootstrap_unit': 'episode; seeds and q4 windows retained together'}}
    result = finalize(directory, report, rows, ref_rows)
    print(f'Results: {directory}/summary.md', flush=True)
    if split == 'validation' and not result['limited_run']:
        print(f"Validation gain lock: {spec['experiment_dir']}/locks/validation.json", flush=True)
    return result

"""Reuse the audited Frozen evaluator and archived WM core; no new rollout math."""

import csv
from pathlib import Path
from types import SimpleNamespace
import sys
import time

import h5py
import numpy as np
from tqdm.auto import tqdm

from diagnostics.common import ROOT, checked, read, require, sha, write
from .protocol import code_identity, episode_paths, episode_receipt, seal_write
from .reporting import evaluation_summary


def evaluate(root, manifest, reference, args):
    collection = read(root/'collection-report.json')
    require(collection['completed'] and collection['manifest'] == manifest['fingerprint']
            and len(collection['pairs']) == len(manifest['cases'])
            and all(pair['initial_physics_identical'] for pair in collection['pairs']), 'Finish verified paired collection first')
    receipts = {(name, case['index']): episode_receipt(root, name, case, manifest)
                for name in ('markov', 'plan') for case in manifest['cases']}
    request = {'manifest': manifest['fingerprint'], 'reference': manifest['config']['frozen_reference_sha256'],
               'device': args.device, 'cpu_threads': args.cpu_threads, 'code_identity': code_identity()}
    directory = root/'evaluations'/args.run_name
    if (directory/'report.json').is_file():
        prior = checked(directory/'report.json')
        require(prior['completed'] and prior['request'] == request
                and sha(directory/'predictions.csv') == prior['predictions_sha256'], 'Existing evaluation differs/is incomplete; use --run-name NEW')
        print(f"Reuse completed evaluation: {directory/'summary.md'}")
        return 0 if prior['passed'] else 2
    require(not directory.exists(), 'Evaluation directory exists without a complete report; use --run-name NEW')
    core = Path(reference['numerical_core_directory']).resolve(strict=True)
    for name, digest in reference['numerical_core'].items():
        path = (core.parent/name).resolve()
        require(path.is_relative_to(core.parent) and path.is_file() and sha(path) == digest, f'Archived numerical core changed: {path}')
    require(sha(ROOT/'codes/diagnostics/frozen.py') == reference['diagnostic_code']['codes/diagnostics/frozen.py'],
            'Audited Frozen helper changed; use a separately audited reference')
    require('lewm_tta' not in sys.modules, 'WM core was imported before archived-core selection; use a fresh process')
    sys.path.insert(0, str(core))
    import torch
    from lewm_tta.data import FrameReader
    from lewm_tta.model import LeWMAdapter
    from diagnostics.frozen import one_case
    torch.set_num_threads(args.cpu_threads)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    config = manifest['config']
    adapter = LeWMAdapter(ROOT/config['repo'], Path(reference['model']['checkpoint']), args.device)
    require(adapter.identity == reference['model'], 'Model/runtime identity differs from the audited Frozen reference')
    require((adapter.history, adapter.action_dim, adapter.image_size, adapter.latent_dim) == (3, 25, 224, 192), 'Wrong Cube model')
    adapter.reset(config['seed'])
    adapter.model.requires_grad_(False)
    directory.mkdir(parents=True)
    report = {'completed': False, 'request': request, 'model': adapter.identity,
              'normalization': reference['normalization'], 'numerical_core_directory': str(core)}
    write(directory/'partial.json', report)
    csv_rows, errors, parity = [], {}, []
    started = time.perf_counter()
    try:
        with tqdm(total=len(manifest['cases'])*2*len(config['window_starts']), desc='Cube Frozen prediction',
                  unit='query', disable=args.quiet) as bar:
            for name in ('markov', 'plan'):
                for case in manifest['cases']:
                    report['active'] = {'policy': name, 'case': case['index']}
                    write(directory/'partial.json', report)
                    path = episode_paths(root, name, case)[0]
                    frame_manifest = {'source': {'path': str(path)}, 'episode_metadata': {
                        'lengths': [config['raw_steps']+1], 'offsets': [0], 'ids': [case['episode_id']]}}
                    reader = FrameReader(frame_manifest, adapter.image_size)
                    values = []
                    try:
                        with h5py.File(path, 'r', swmr=True) as stream:
                            for start in config['window_starts']:
                                pixels = reader.batch(start, start+36)
                                commands = stream['action'][start:start+35]
                                require(commands.shape == (35, 5) and np.isfinite(commands).all()
                                        and stream['action_valid'][start:start+35].all(), 'Invalid prediction action window')
                                item = {'start': start, 'episode': case['episode_id'], 'current_offset': 10}
                                options = SimpleNamespace(histories=[3], horizon=5, seed=config['seed'])
                                row = one_case(adapter, pixels, commands, None, item, reference['normalization'], options, name)[0]
                                native, project = row['mse_by_block']['native_pixel'], row['mse_by_block']['project_live']
                                parity.append(row['checks']['native_pixels_vs_live_latent_rollout'])
                                values.append(native)
                                for depth, (first, second) in enumerate(zip(native, project), 1):
                                    csv_rows.append({'policy': name, 'case': case['index'], 'window_start': start,
                                        'current_raw_step': start+10, 'depth': depth, 'raw_controls_ahead': depth*5,
                                        'native_mse': first, 'project_mse': second,
                                        'input_sha256': row['input_pixels_and_actions_sha256']})
                                bar.update()
                    finally:
                        reader.close()
                    errors[(name, case['index'])] = np.mean(values, axis=0)
        adapter.check_fixed_state()
        require(float(adapter.anchor_loss()) == 0 and all(not p.requires_grad and p.grad is None for p in adapter.model.parameters()),
                'Frozen model changed or received gradients')
        for key, receipt in receipts.items():
            require(episode_receipt(root, key[0], manifest['cases'][key[1]], manifest) == receipt, 'Episode changed during evaluation')
        with (directory/'predictions.csv').open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(csv_rows[0]))
            writer.writeheader()
            writer.writerows(csv_rows)
        summary = {}
        for name in ('markov', 'plan'):
            values = np.stack([errors[(name, case['index'])] for case in manifest['cases']])
            summary[name] = {'episodes': len(values), 'queries': len(values)*len(config['window_starts']),
                             'mean_mse_by_block': values.mean(0).tolist(), 'path_mse': float(values.mean()),
                             'episode_mean_mse_by_block': values.tolist()}
        difference = np.array([errors[('markov', case['index'])][-1]-errors[('plan', case['index'])][-1] for case in manifest['cases']])
        rng = np.random.default_rng(config['seed'])
        bootstrap = difference[rng.integers(len(difference), size=(10000, len(difference)))].mean(1)
        report.pop('active', None)
        report.update(completed=True, passed=all(row['passed'] for row in parity), summary=summary,
            parity={'all_queries_passed': all(row['passed'] for row in parity), 'query_checks': len(parity),
                    'maximum_absolute_prediction_difference': max(row['max_abs'] for row in parity)},
            paired_initial_case_comparison={'mean_h5_markov_minus_plan': float(difference.mean()),
                'h5_markov_minus_plan_by_case': difference.tolist(),
                'bootstrap95_h5_markov_minus_plan': np.quantile(bootstrap, [.025, .975]).tolist(),
                'unit': 'matched initial episode pair; four fixed windows averaged before resampling'},
            predictions_sha256=sha(directory/'predictions.csv'), evaluation_seconds=time.perf_counter()-started,
            protocol={'history': 3, 'autoregressive_blocks': 5, 'raw_controls_per_block': 5, 'latent_mse_dim': 192,
                      'teacher_forcing': False, 'retrieval': False, 'optimizer_steps': 0, 'cache_or_index_required': False,
                      'cohort_pairing': 'shared initial cases; generated queries/actions/futures differ', 'split': 'calibration'})
        seal_write(directory/'report.json', report)
        (directory/'partial.json').unlink()
        evaluation_summary(directory, report)
        print(f"Frozen policy-pilot evaluation: {directory/'summary.md'}")
        return 0 if report['passed'] else 2
    except BaseException as error:
        report['error'] = {'type': type(error).__name__, 'message': str(error)}
        write(directory/'partial.json', report)
        raise

"""Small native-LeWM/cached-project comparisons; no retrieval, CEM or simulation."""

from collections import defaultdict
import csv
import gzip
import json
from pathlib import Path
import time

import h5py
import numpy as np
import torch

from .common import ROOT, checked, code_identity, headless, output, read, require, sha, write
from lewm_tta.data import FrameReader
from lewm_tta.model import Context, LeWMAdapter
from experiments.progress import progress


def compare(actual, expected, rtol=2e-4, atol=2e-4):
    require(actual.shape == expected.shape, f'Comparison shapes differ: {actual.shape}/{expected.shape}')
    delta = (actual-expected).detach().float()
    finite = bool(torch.isfinite(delta).all())
    return {'passed': finite and bool(torch.allclose(actual, expected, rtol=rtol, atol=atol)),
            'max_abs': float(delta.abs().max()) if finite else None,
            'mean_abs': float(delta.abs().mean()) if finite else None, 'rtol': rtol, 'atol': atol}


def mse(prediction, target):
    first = prediction.detach().cpu().numpy().astype(np.float64)
    second = target.detach().cpu().numpy().astype(np.float64)
    require(first.shape == second.shape and first.ndim == 3 and first.shape[0] == 1, 'Invalid scoring shapes')
    values = np.mean((first-second)**2, axis=-1)[0]
    require(np.isfinite(values).all(), 'Nonfinite latent error')
    return values.tolist()


@torch.no_grad()
def project_rollout(adapter, context, actions):
    if context.latents.shape[1] == adapter.history:
        return adapter.rollout(context, actions)  # The actual P2 Frozen implementation.
    # One-frame control: grow imagined context to C=3 without inventing older RGB.
    outputs = []
    for action in actions.unbind(1):
        ordered = torch.cat((context.past_actions, action[:, None]), 1)
        prediction = adapter.model.predict(context.latents, adapter.model.action_encoder(ordered))[:, -1]
        outputs.append(prediction)
        latents = torch.cat((context.latents, prediction[:, None]), 1)[:, -adapter.history:]
        past = ordered[:, -(latents.shape[1]-1):] if latents.shape[1] > 1 else ordered[:, :0]
        context = Context(latents, past)
    return torch.stack(outputs, 1)


def source_cases(manifest, samples, seed, horizon):
    offsets = manifest['episode_metadata']['offsets']
    lengths = manifest['episode_metadata']['lengths']
    span = (manifest['data_config']['history']-1+horizon)*manifest['data_config']['frameskip']
    eligible = [ep for ep in manifest['episodes']['memory'] if lengths[ep] > span]
    require(len(eligible) >= samples, 'Not enough horizon-safe source memory episodes')
    rng = np.random.default_rng(seed)
    cases = []
    for ep in sorted(rng.choice(eligible, samples, replace=False).tolist()):
        start = int(offsets[ep])+int(rng.integers(lengths[ep]-span))
        cases.append({'start': start, 'episode': ep, 'current_offset': 10})
    return cases


def fresh_cases(manifest, split, samples, seed, horizon):
    starts = manifest['evaluation_windows'][split]['raw_start_rows']
    offsets = np.asarray(manifest['episode_metadata']['offsets'])
    lengths = np.asarray(manifest['episode_metadata']['lengths'])
    by_episode = defaultdict(list)
    for start in starts:
        ep = int(np.searchsorted(offsets, start, side='right')-1)
        if start+(2+horizon)*5 < offsets[ep]+lengths[ep]:
            by_episode[ep].append(start)
    require(len(by_episode) >= samples, 'Not enough prespecified fresh windows/episodes for this horizon')
    rng = np.random.default_rng(seed)
    return [{'start': int(rng.choice(by_episode[ep])), 'episode': int(ep), 'current_offset': 10}
            for ep in sorted(rng.choice(sorted(by_episode), samples, replace=False).tolist())]


def cache_for(args, reference, manifest, cohort, adapter, prepared):
    if args.skip_cache_check:
        return None, {'enabled': False}
    if cohort == 'source':
        expected = prepared['memory_cache']['observations']
        directory = Path(reference['config']['p1']['paths']['cache_dir'])/f"observations-{expected['observation_fingerprint'][:16]}"
        candidates = [directory]
    else:
        expected = None
        candidates = list(Path(reference['config']['query_dataset']['cache_dir']).glob('observations-*'))
    matches = []
    for directory in candidates:
        if not (directory/'progress.json').is_file():
            continue
        state = read(directory/'progress.json')
        if (state['identity']['model'] == adapter.fingerprint
                and state['identity']['source'] == manifest['source'] and state['complete']
                and state['encoded_rows'] == manifest['source']['rows']):
            matches.append((directory, state))
    require(len(matches) == 1, f'Expected one completed matching {cohort} cache, found {len(matches)}; restore it or use --skip-cache-check')
    directory, state = matches[0]
    path = directory/'latents.npy'
    if expected:
        require(expected['latents_sha256'] == state['latents_sha256'], 'Source cache receipt differs from prepared metadata')
    if args.verify_cache_sha256:
        require(sha(path) == state['latents_sha256'], 'Full latent-cache checksum differs')
    values = np.load(path, mmap_mode='r', allow_pickle=False)
    require(values.shape == (manifest['source']['rows'], adapter.latent_dim) and values.dtype == np.float32,
            'Invalid cache shape/dtype')
    return values, {'enabled': True, 'path': str(path), 'receipt_sha256': sha(directory/'progress.json'),
                    'expected_latents_sha256': state['latents_sha256'],
                    'full_payload_sha256_verified': args.verify_cache_sha256,
                    'scope': 'selected cached observations compared directly with newly encoded same RGBs'}


def saved_errors(reference, starts):
    from ablations.saved_runs import record_rows
    expected = {}
    for row in record_rows(reference):
        if row['method'] != 'frozen' or row.get('schedule') != 'temporal-u1':
            continue
        start = int(row['query']['raw_observation_rows'][0])
        if start in starts and row['seed'] == reference['config']['p1']['evaluation']['seeds'][0]:
            require(start not in expected, 'Duplicate saved Frozen reference')
            expected[start] = row['mse_by_block']
    require(set(expected) == set(starts), 'Saved Frozen reference omits selected fresh windows')
    return expected


@torch.no_grad()
def one_case(adapter, pixels, raw_actions, cached, case, normalizer, args, cohort, saved=None):
    current, skip = case['current_offset'], 5
    required = sorted(set([current-k*skip for k in range(max(args.histories))]
                          +[current+k*skip for k in range(1, args.horizon+1)]))
    require(required[0] >= 0 and required[-1] < len(pixels), 'Observed/target frame range is invalid')
    images = pixels[required].to(adapter.device)
    live = adapter.encode(images[None])
    lookup = {row: i for i, row in enumerate(required)}
    observed_targets = live[:, [lookup[current+k*skip] for k in range(1, args.horizon+1)]]
    mean = torch.tensor(normalizer['mean'], device=adapter.device)
    std = torch.tensor(normalizer['std'], device=adapter.device)
    commands = torch.from_numpy(np.array(raw_actions, dtype=np.float32, copy=True)).to(adapter.device)
    normalized = (commands-mean)/std
    cached_values = None if cached is None else torch.from_numpy(np.array(cached[case['start']+np.asarray(required)], copy=True))[None].to(adapter.device)
    rows = []
    for history in args.histories:
        first = current-(history-1)*skip
        observed = [first+j*skip for j in range(history)]
        ids = [lookup[i] for i in observed]
        blocks = normalized[first:current+args.horizon*skip].reshape(1, history-1+args.horizon, adapter.action_dim)
        past, future = blocks[:, :history-1], blocks[:, history-1:]
        context = Context(live[:, ids], past)
        predicted = project_rollout(adapter, context, future)
        # The original JEPA.rollout receives only observed RGB and the ordered
        # past/future actions. Its output begins with the observed embeddings.
        native = adapter.model.rollout({'pixels': pixels[observed][None, None].to(adapter.device)},
                                      blocks[:, None], history_size=adapter.history)['predicted_emb']
        require(native.shape == (1, 1, history+args.horizon, adapter.latent_dim), 'Native rollout returned an unexpected timeline')
        native = native[:, 0, history:]
        checks = {'native_pixels_vs_live_latent_rollout': compare(native, predicted)}
        errors = {'native_pixel': mse(native, observed_targets), 'project_live': mse(predicted, observed_targets)}
        if cached_values is not None:
            target_ids = [lookup[current+k*skip] for k in range(1, args.horizon+1)]
            cached_prediction = project_rollout(adapter, Context(cached_values[:, ids], past), future)
            checks['cached_observations_vs_new_encoding'] = compare(cached_values, live)
            checks['native_vs_project_cached_rollout'] = compare(native, cached_prediction)
            errors['project_cached'] = mse(cached_prediction, cached_values[:, target_ids])
            if saved is not None and history == 3:
                count = min(5, args.horizon)
                checks['cached_mse_vs_saved_p2_frozen'] = compare(
                    torch.tensor(errors['project_cached'][:count], dtype=torch.float64),
                    torch.tensor(saved[:count], dtype=torch.float64), rtol=2e-4, atol=1e-6)
        rows.append({'cohort': cohort, 'case': case, 'observed_history': history,
            'current_raw_row': case['start']+current, 'seed': args.seed,
            'target_raw_rows': [case['start']+current+k*skip for k in range(1, args.horizon+1)],
            'mse_by_block': errors, 'checks': checks, 'passed': all(c['passed'] for c in checks.values()),
            'input_pixels_and_actions_sha256': __import__('hashlib').sha256(
                pixels[required].numpy().tobytes()+np.asarray(raw_actions, dtype=np.float32).tobytes()).hexdigest()})
    return rows


def tables(directory, report, records):
    groups = defaultdict(list)
    for row in records:
        for method, errors in row['mse_by_block'].items():
            groups[(row['cohort'], row['observed_history'], method)].append(errors)
    csv_rows, summaries = [], []
    for (cohort, history, method), errors in sorted(groups.items()):
        values = np.asarray(errors, dtype=np.float64)
        item = {'cohort': cohort, 'history': history, 'method': method, 'queries': len(values),
                'mean_mse_by_block': values.mean(0).tolist(), 'median_mse_by_block': np.median(values, axis=0).tolist(),
                'path_mse': float(values.mean())}
        summaries.append(item)
        for h, mean in enumerate(item['mean_mse_by_block'], 1):
            csv_rows.append({'cohort': cohort, 'history': history, 'method': method, 'queries': len(values),
                             'latent_depth': h, 'raw_controls_ahead': h*5, 'mean_mse': mean,
                             'median_mse': item['median_mse_by_block'][h-1]})
    with (directory/'mse-by-step.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(csv_rows[0]))
        writer.writeheader(); writer.writerows(csv_rows)
    report['summary'] = summaries
    lines = ['# Frozen native LeWM / project rollout diagnostic', '',
        f"Numerical parity passed: **{report['passed']}**.", '',
        '| Cohort | History | Implementation | N | H1 MSE | Final MSE | Path MSE |',
        '|---|---:|---|---:|---:|---:|---:|']
    for s in summaries:
        lines.append(f"| {s['cohort']} | {s['history']} | {s['method']} | {s['queries']} | "
                     f"{s['mean_mse_by_block'][0]:.6f} | {s['mean_mse_by_block'][-1]:.6f} | {s['path_mse']:.6f} |")
    lines += ['', 'The native pixel API and latent wrapper use identical weights, input RGBs, normalized ordered actions and history.',
        'Source/fresh are different diagnostic cohorts; their difference is not a paired causal estimate or a held-out benchmark.',
        'History 1 and 3 share the same current time and future controls. Missing history is never repeated or fabricated.',
        'No CEM, retrieval, adaptation, simulator stepping, cache encoding jobs, or cache/index writes were performed.',
        'Action normalization is the saved P2 normalization in all paths; this is not a separate reproduction of upstream whole-source scaler fitting.',
        'Passing this diagnostic does not identify Figure 3 trajectory IDs/checkpoint or establish simulator equivalence.', '']
    (directory/'summary.md').write_text('\n'.join(lines))
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        report['plot'] = 'matplotlib unavailable; use mse-by-step.csv'
        return
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for s in summaries:
        if s['method'] == 'project_live':
            continue  # Redundant with native when parity holds.
        ax.plot(np.arange(1, len(s['mean_mse_by_block'])+1)*5, s['mean_mse_by_block'],
                marker='o', linestyle='--' if s['method'] == 'project_cached' else '-',
                label=f"{s['cohort']} C{s['history']} {s['method']}")
    ax.set(xlabel='Raw controls ahead of current observation', ylabel='Endpoint latent MSE', title=report['dataset'])
    ax.legend(fontsize=8); ax.grid(alpha=.2); fig.tight_layout()
    fig.savefig(directory/'mse-by-step.png', dpi=180); plt.close(fig)


def run(args, reference, entry, core_codes):
    headless()
    from lewm_tta.collection.queries import query_source
    torch.set_num_threads(args.cpu_threads)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(args.seed)
    p1 = reference['config']['p1']
    source = checked(Path(p1['paths']['experiment_dir'])/'manifest.json')
    prepared = checked(Path(p1['paths']['experiment_dir'])/'prepared.json')
    require(source['fingerprint'] == prepared['manifest']
            and prepared['fingerprint'] == reference['p1_prepared_fingerprint'], 'Prepared source provenance changed')
    require(source['data_config']['history'] == 3 and source['data_config']['frameskip'] == 5, 'Expected history3/frameskip5')
    suffix = 'adajepa-replay' if args.target_replay else args.split
    directory = output(args.output or ROOT/f'artifacts/diagnostics/frozen-rollout/{args.dataset}-{suffix}-q{args.samples}-s{args.seed}')
    adapter = LeWMAdapter(Path(p1['paths']['repo']), Path(p1['paths']['checkpoint']), args.device)
    require(adapter.identity == reference['model'], 'Model/runtime differs from the measured reference; preserve the environment or use a separately audited reference')
    require(adapter.fingerprint == prepared['model_fingerprint'], 'Prepared observation model differs from this checkpoint')
    adapter.reset(args.seed); adapter.model.requires_grad_(False)
    records, cohorts = [], {}
    started = time.perf_counter()
    if args.target_replay:
        from diagnostics.targets import load_replay
        from utils import get_img_preprocessor
        data, receipt = load_replay(args.target_replay)
        require(args.samples <= receipt['samples'], 'Replay payload has fewer cases than --samples')
        transform = get_img_preprocessor('pixels', 'pixels', adapter.image_size)
        for index in progress(range(args.samples), config={'evaluation': {'log_every': 1}}, desc='AdaJEPA SWM replay parity'):
            pixels = torch.from_numpy(np.array(data['pixels'][index], copy=True)).permute(0, 3, 1, 2)
            pixels = transform({'pixels': pixels})['pixels']
            case = {'start': 0, 'episode': int(data['segment_indices'][index]), 'current_offset': 0,
                    'replay_position': index}
            records.extend(one_case(adapter, pixels, data['actions'][index], None, case,
                                    prepared['normalization'], args, 'adajepa-swm-replay'))
        data.close()
        cohorts['adajepa-swm-replay'] = {'receipt': receipt, 'scope': 'replayed observations in current SWM; not original AdaJEPA RGB'}
    else:
        for cohort in args.cohorts:
            manifest = source if cohort == 'source' else query_source(reference['config'], source)
            cases = (source_cases(manifest, args.samples, args.seed, args.horizon) if cohort == 'source'
                     else fresh_cases(manifest, args.split, args.samples, args.seed, args.horizon))
            cached, cache_info = cache_for(args, reference, manifest, cohort, adapter, prepared)
            saved = (saved_errors(reference, [c['start'] for c in cases])
                     if cohort == 'fresh' and args.split == 'validation' and cached is not None else {})
            reader = FrameReader(manifest, adapter.image_size)
            try:
                # FrameReader's native SWM reader opens this same file with
                # swmr=True. Concurrent handles must agree on the SWMR flag,
                # including when the native image handle is opened lazily.
                with h5py.File(manifest['source']['path'], 'r', swmr=True) as h5:
                    for case in progress(cases, config={'evaluation': {'log_every': 1}}, desc=f'{args.dataset} {cohort} parity'):
                        start = case['start']; end = start+(2+args.horizon)*5
                        pixels = reader.batch(start, end+1)
                        raw = h5['action'][start:end]
                        require(raw.shape[0] == end-start and np.isfinite(raw).all(), 'Invalid recorded action interval')
                        records.extend(one_case(adapter, pixels, raw, cached, case, prepared['normalization'],
                                                args, cohort, saved.get(start)))
            finally:
                reader.close()
            cohorts[cohort] = {'manifest_fingerprint': manifest['fingerprint'], 'cases': cases, 'cache': cache_info,
                              'label': 'checkpoint training-source diagnostic' if cohort == 'source' else f'fresh {args.split} diagnostic'}
    adapter.check_fixed_state()
    require(float(adapter.anchor_loss()) == 0 and all(not p.requires_grad and p.grad is None for p in adapter.model.parameters()),
            'Frozen model was modified or received gradients')
    path = directory/'records.jsonl.gz'
    with gzip.open(path, 'xt', encoding='utf-8') as stream:
        for row in records:
            stream.write(json.dumps(row, separators=(',', ':'), allow_nan=False)+'\n')
    report = {'stage': 'Frozen-native-LeWM-rollout-diagnostic', 'completed': True,
        'passed': all(row['passed'] for row in records), 'dataset': args.dataset,
        'samples_per_cohort': args.samples, 'seed': args.seed, 'histories': args.histories, 'horizon': args.horizon,
        'reference': entry, 'model': adapter.identity, 'normalization': prepared['normalization'],
        'numerical_core_directory': str(core_codes), 'numerical_core': reference['code'], 'diagnostic_code': code_identity(),
        'cohorts': cohorts, 'records_file': str(path), 'records_sha256': sha(path), 'record_count': len(records),
        'evaluation_seconds': time.perf_counter()-started, 'max_native_vs_live_abs_error': max(
            r['checks']['native_pixels_vs_live_latent_rollout']['max_abs'] or 0 for r in records),
        'protocol': {'metric': 'endpoint mean squared error over 192 projected-latent coordinates',
            'context_feedback': 'autoregressive predicted latents', 'raw_controls_per_depth': 5,
            'weights_and_buffers_frozen': True, 'optimizer_steps': 0, 'retrieval_index_loaded': False,
            'simulator_execution': False, 'new_cache_encodings_written': False,
            'source_vs_fresh_is_paired': False}}
    tables(directory, report, records)
    write(directory/'report.json', report)
    print(f"Parity passed: {report['passed']}; results: {directory}/summary.md", flush=True)
    if not report['passed']:
        raise SystemExit(2)

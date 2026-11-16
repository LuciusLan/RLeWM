"""Use the reference WM/update core unchanged; retrieve again without another step."""

from copy import deepcopy
import hashlib
from importlib.metadata import PackageNotFoundError, version
import inspect
from pathlib import Path
from types import SimpleNamespace
import time

from diagnostics.common import identity, read, require, sha, write
from .metrics import compare, episode_overlap
from .run import load_query, records_per_query, save_query, seal


class CaptureRetriever:
    """Observe the two searches already performed by AdaptationSession."""
    def __init__(self, original):
        self.original = original

    def __getattr__(self, name):
        return getattr(self.original, name)

    def forward(self, context, action):
        self.forward_before = self.original.forward(context, action)
        return self.forward_before

    def reverse(self, context, prediction):
        self.reverse_before = self.original.reverse(context, prediction)
        return self.reverse_before


def existing_queries(resources, config, manifest, limit):
    """Open the exact existing encoding read-only; never invoke the encoder/cache writer."""
    import numpy as np
    import torch
    from lewm_tta.data import FrameReader
    from lewm_tta.frame_bank import FrameLatentBank
    from lewm_tta.observation_cache import ObservationStore, _encode_observations
    from lewm_tta.collection.queries import namespace_bank
    from lewm_tta.runtime import object_hash
    query, adapter = resources.external_manifest, resources.adapter
    observed = {'version': 1, 'source': query['source'], 'model': adapter.fingerprint,
                'latent_dim': adapter.latent_dim, 'dtype': 'float32',
                'frame_reader_sha256': hashlib.sha256(inspect.getsource(FrameReader).encode()).hexdigest(),
                'encoder_loop_sha256': hashlib.sha256(inspect.getsource(_encode_observations).encode()).hexdigest()}
    fingerprint = object_hash(observed)
    directory = Path(config['query_dataset']['cache_dir'])/f'observations-{fingerprint[:16]}'
    require((directory/'progress.json').is_file(), 'Fresh encoding missing; run the matching P2 check first')
    status = read(directory/'progress.json')
    normal = resources.prepared['normalization']
    action_path = directory/f'actions-{object_hash(normal)[:16]}.npy'
    action_status = read(action_path.with_suffix('.json'))
    require(status['identity'] == observed and status.get('complete')
            and status['encoded_rows'] == query['source']['rows']
            and sha(directory/'latents.npy') == status['latents_sha256']
            and action_status['normalization'] == normal and sha(action_path) == action_status['sha256'],
            'Fresh latent/action cache identity or checksum changed')
    latents = np.load(directory/'latents.npy', mmap_mode='c', allow_pickle=False)
    actions = np.load(action_path, mmap_mode='c', allow_pickle=False)
    require(latents.shape == (query['source']['rows'], adapter.latent_dim)
            and actions.shape == tuple(query['source']['schema']['action']['shape'])
            and latents.dtype == actions.dtype == np.float32, 'Invalid fresh cache dimensions/precision')
    store_identity = {'observation_fingerprint': fingerprint, 'latents_sha256': status['latents_sha256'],
                      'normalization': normal, 'actions_sha256': action_status['sha256'],
                      'raw_frames': len(latents), 'directory': str(directory)}
    store = ObservationStore(torch.from_numpy(latents), torch.from_numpy(actions), store_identity, directory).to(adapter.device)
    starts = manifest['splits']['validation']['raw_start_rows']
    if limit is not None:
        starts = starts[:limit]
    bank = FrameLatentBank(store, query, 'validation', np.asarray(starts, dtype=np.int64),
                          {'external_queries': query['fingerprint'], 'observations': store_identity})
    return namespace_bank(bank)


def trace_query(resources, query, actions, method, seed, lr, ks, schedule):
    import torch
    from lewm_tta.model import Context
    from lewm_tta.retrieval import diverse_prefix, select_support
    from lewm_tta.rollout.temporal import AdaptationSession
    captured = CaptureRetriever(resources.engine.retriever)
    engine = SimpleNamespace(adapter=resources.adapter, memory=resources.memory, retriever=captured,
                             settings={**resources.engine.settings, 'learning_rate': lr})
    schedule = {**schedule, 'learning_rate': lr}
    session = AdaptationSession(engine, query, seed, schedule)
    context = Context(query.context.latents.detach(), query.context.past_actions.detach())
    previous, rows, size = {}, [], session.settings['support_size']

    def prefix(ranking, k, constrained):
        if not constrained:
            return [int(i) for i in ranking.order[:min(k, len(ranking))]]
        return diverse_prefix(ranking.order, min(k, captured.max_support), captured.episode_positions,
                              captured.episode_cap, captured.step_positions, captured.min_step_gap)

    def support(forward, reverse, generator_seed):
        return select_support('fr-refresh', size, captured.settings['pool_size'], session.settings['alpha'],
            forward, reverse, torch.Generator(device='cpu').manual_seed(generator_seed), None,
            captured.episode_positions, captured.episode_cap, captured.max_support,
            captured.step_positions, captured.min_step_gap)

    for position, action in enumerate(actions.detach().unbind(1)):
        row = session.adapt_position(context, action, method, 1, position)
        require(row['actual_updates'] == 1, 'Expected one observed-memory update at every position')
        prediction = row['prediction_tensor']
        forward, before = captured.forward_before, captured.reverse_before
        # Same x and a. This read-only search cannot affect the u1 rollout/optimizer.
        after = captured.original.reverse(context, prediction)
        base = {'query_id': query.metadata['transition_id'], 'episode_id': query.metadata['episode_id'],
                'scope': getattr(resources.adapter, 'scope', 'last-block'),
                'seed': seed, 'method': method, 'learning_rate': lr, 'position': position,
                'prediction_correction_mse': session.rounds[-1]['prediction_correction_mse']}

        def add(comparison, selection, k, left, right, weights0=None, weights1=None):
            rows.append({**base, 'comparison': comparison, 'selection': selection, 'k': k,
                         **compare(left, right, weights0, weights1),
                         **episode_overlap(left, right, captured.episode_positions)})

        for k in ks:
            for constrained, selection in ((False, 'raw'), (True, 'diverse')):
                f, r0, r1 = (prefix(ranking, k, constrained) for ranking in (forward, before, after))
                add('forward_reverse_before', selection, k, f, r0)
                add('forward_reverse_after', selection, k, f, r1)
                add('reverse_after_update', selection, k, r0, r1)
                key = (k, selection)
                if key in previous:
                    add('reverse_between_positions', selection, k, previous[key], r0)
                previous[key] = r1
        fr0, fr1 = support(forward, before, row['support_rng_seed']), support(forward, after, row['support_rng_seed'])
        if method in ('reverse', 'fr-fixed'):
            # Verify the measured pre-update set is the one that really trained
            # the predictor, including episode/step constraints and weights.
            indices = prefix(before, size, True) if method == 'reverse' else fr0.indices
            require(indices, 'Measured update support is empty')
            weights = [1/len(indices)]*len(indices) if method == 'reverse' else fr0.weights
            actual = session.rounds[-1]['support']
            require([resources.memory.items[i]['transition_id'] for i in indices]
                    == [item['transition_id'] for item in actual], f'Probe differs from actual {method} first support')
            torch.testing.assert_close(torch.tensor(weights), torch.tensor([item['weight'] for item in actual]), rtol=0, atol=0)
        add('fr_support_after_update', 'actual_fr', size, fr0.indices, fr1.indices, fr0.weights, fr1.weights)
        f0 = [i for i in fr0.indices if 'forward' in fr0.channels[i]]
        f1 = [i for i in fr1.indices if 'forward' in fr1.channels[i]]
        r0 = [i for i in fr0.indices if 'reverse' in fr0.channels[i]]
        r1 = [i for i in fr1.indices if 'reverse' in fr1.channels[i]]
        add('fr_directional_overlap_before', 'actual_fr', size, f0, r0)
        add('fr_directional_overlap_after', 'actual_fr', size, f1, r1)
        add('fr_reverse_channel_after_update', 'actual_fr', size, r0, r1)
        require(session.optimizer_updates == position+1, 'Read-only probe changed the optimizer clock')
        context = Context(torch.cat((context.latents[:, 1:], prediction.unsqueeze(1)), dim=1).detach(),
                          torch.cat((context.past_actions[:, 1:], action.unsqueeze(1)), dim=1).detach())
        # Compact on disk: no latent vectors, support IDs, complete ranks or model weights.
    resources.adapter.check_fixed_state()
    return rows


def measure(args, report, request, core):
    import torch
    from tqdm.auto import tqdm
    from lewm_tta.rollout.cohorts import load_manifest, rollout_inputs
    from lewm_tta.rollout.resources import load_resources
    config = deepcopy(report['config'])
    config['scopes'] = [request['scope']]
    manifest = load_manifest(config)
    directory, fingerprint = args.output/request['dataset'], identity(request)
    path = directory/'manifest.json'
    if path.exists():
        require(read(path) == seal(request), 'Measurement settings/code changed; choose a new output root')
    else:
        require(not (directory/'queries').exists(), 'Unrecognized query checkpoint directory')
        write(path, seal(request))
    complete = directory/'measurement.json'
    if complete.exists():
        saved = read(complete)
        require(saved.get('completed') and saved['request'] == fingerprint, 'Changed completion receipt')
        require(len(saved['files']) == request['queries'] and all(
            sha(directory/name) == digest for name, digest in saved['files'].items()), 'Saved statistics changed')
        require(saved['runtime_sha256'] == sha(directory/'runtime.json'), 'Saved runtime receipt changed')
        print(f"Reuse verified statistics: {request['dataset']}", flush=True)
        return
    resources = load_resources(config, args.device, args.cpu_threads)
    require(resources.external_manifest is not None and len(resources.memory) > 0, 'Fresh queries and nonempty observed memory required')
    scope = resources.adapter.scope_description()
    require(scope['scope'] == request['scope'], 'Requested and actual adaptation scopes disagree')
    if request['scope'] == 'predictor':
        require(set(scope['trainable_module_roots']) == {'predictor', 'pred_proj'},
                'Full-predictor diagnostic must freeze the action and visual encoders')
    queries = existing_queries(resources, config, manifest, args.limit)
    require(len(queries) == request['queries'], 'Validation query count changed')
    runtime = {'core': str(core), 'model': resources.adapter.identity, 'memory': resources.memory.identity,
               'query_cache': queries.store.identity, 'retrieval': resources.engine.retriever.settings,
               'memory_windows': len(resources.memory), 'device': str(resources.adapter.device),
               'scope': scope, 'torch': torch.__version__}
    if resources.engine.retriever.settings.get('backend') == 'cuvs':
        try:
            runtime['cuvs'] = version('cuvs-cu13')
        except PackageNotFoundError:
            runtime['cuvs'] = version('cuvs-cu12')
    runtime_path = directory/'runtime.json'
    if runtime_path.exists():
        require(read(runtime_path) == runtime, 'Runtime/cache changed during resume; choose a new output root')
    else:
        write(runtime_path, runtime)
    schedule = config['schedules']['temporal-u1']
    sessions = len(request['seeds'])*len(request['methods'])*len(request['learning_rates'])
    started = time.perf_counter()
    files = {}
    with tqdm(total=len(queries)*sessions, desc=f"{request['dataset']} temporal-u1", unit='rollout', disable=args.quiet) as bar:
        for index in range(len(queries)):
            checkpoint = directory/'queries'/f'{index:05d}.json.gz'
            if checkpoint.exists():
                value = load_query(checkpoint, fingerprint, index)
                require(value['query_id'] == queries.items[index]['transition_id']
                        and len(value['records']) == records_per_query(request), 'Query order/coverage changed on resume')
                bar.update(sessions)
            else:
                query, actions, _ = rollout_inputs(queries, index, request['horizon'])
                records = []
                for method in request['methods']:
                    for seed in request['seeds']:
                        for lr in request['learning_rates']:
                            bar.set_postfix(query=index, lr=f'{lr:g}', seed=seed, refresh=False)
                            records.extend(trace_query(resources, query, actions, method, seed, lr, request['ks'], schedule))
                            bar.update()
                require(len(records) == records_per_query(request), 'Unexpected per-query probe coverage')
                save_query(checkpoint, {'request': fingerprint, 'query_index': index,
                    'query_id': query.metadata['transition_id'], 'records': records})
            files[str(checkpoint.relative_to(directory))] = sha(checkpoint)
    write(complete, {'completed': True, 'request': fingerprint, 'files': files,
        'runtime_sha256': sha(runtime_path),
        'limited_run': args.limit is not None, 'validation_only': True,
        'elapsed_seconds_this_invocation': time.perf_counter()-started,
        'protocol': {'actual_updates_per_position': 1, 'extra_updates_from_probe': 0,
            'update_methods': request['methods'],
            'raw_and_diverse_lists': True, 'forward_fixed_within_position': True,
            'query_context_and_action_fixed_for_update_probe': True,
            'fr_selector_support_and_weight_probe': True, 'teacher_forcing': False,
            'query_future_labels_used': False, 'simulator_or_cem': False,
            'query_and_optimizer_reset_per_rollout': True, 'encoder_run': False}})
    print(f"Retrieval measurements complete: {directory}", flush=True)

"""Nonparametric prediction controls; no query targets or optimizer enter here."""

import time

import numpy as np
import torch

from ablations.saved_runs import identity, require
from ..model import Context
from ..retrieval import select_support
from ..rollout.support_updates import synchronize, validate_support
from .cache import assert_frozen, freeze


def correction(control, base, current, successors, memory_current, memory_prediction, weights, gain):
    """Transfer observed changes or frozen-model residuals in encoder latent space."""
    if control == 'residual-transfer':
        reference, changes = base, successors-memory_prediction
    elif control == 'delta-transfer':
        reference, changes = current, successors-memory_current
    else:
        raise ValueError(f'Unknown correction: {control}')
    # Zero gain has an exact equivalence even if distant-memory deltas are large.
    if gain == 0:
        return reference
    return reference + gain*(weights[:, None]*changes).sum(0, keepdim=True)


def support_rows(memory, support):
    return [{**{k: memory.items[i][k] for k in ('transition_id', 'source_episode_position', 'start_in_episode')},
             'weight': w, 'channels': support.channels[i]}
            for i, w in zip(support.indices, support.weights)]


def select(resources, context, action, base, selector, rng_seed, budget):
    retriever = resources.engine.retriever
    # Uniform needs only the corpus length; avoid two unused index searches.
    forward, reverse = range(len(resources.memory)), None
    calls = {'forward': 0, 'reverse': 0}
    if selector in ('forward', 'fr-fixed'):
        forward = retriever.forward(context, action)
        calls['forward'] = 1
    if selector in ('reverse', 'fr-fixed'):
        reverse = retriever.reverse(context, base)
        calls['reverse'] = 1
    settings = resources.config
    support = select_support(selector, budget, settings['retrieval']['pool_size'],
        settings['adaptation']['alpha'], forward, reverse,
        torch.Generator(device='cpu').manual_seed(rng_seed),
        episode_positions=retriever.episode_positions, max_per_episode=retriever.episode_cap,
        max_support=retriever.max_support, step_positions=retriever.step_positions,
        min_step_gap=retriever.min_step_gap)
    validate_support(resources.memory, retriever, support, budget)
    require(len(support.indices) == budget, 'Support budget shortfall')
    return support, calls


@torch.no_grad()
def rollout(resources, query, actions, case, predictions, spec, seed):
    """Autoregressive rollout with support reselected at each imagined position."""
    adapter, memory = resources.adapter, resources.memory
    require(query.metadata['source_episode_position'] not in resources.engine.retriever.episode_set,
            'Query episode occurs in memory')
    query_seed = int(identity([seed, query.metadata['transition_id']])[:8], 16)
    synchronize(adapter)
    started = time.perf_counter()
    freeze(adapter, query_seed)
    synchronize(adapter)
    timing = {'reset': time.perf_counter()-started, 'query_prediction': 0.0,
              'retrieval_and_selection': 0.0, 'correction_and_lookup': 0.0}
    context, output, positions = query.context, [], []
    search_calls = {'forward': 0, 'reverse': 0}
    first_support = None
    for h, action in enumerate(actions.unbind(1)):
        tick = time.perf_counter()
        # Delta transfer still uses F0 for its reverse retrieval key, never for supervision.
        base = (adapter.predict(context, action) if case['control'] != 'persistence'
                else context.latents[:, -1])
        synchronize(adapter)
        timing['query_prediction'] += time.perf_counter()-tick
        position = {'position': h, 'support_size': 0, 'effective_support_size': 0.0}
        if case['selector'] is None:
            result = base
        else:
            rng = query_seed if h == 0 else int(identity([query_seed, 'position', h])[:8], 16)
            tick = time.perf_counter()
            support, calls = select(resources, context, action, base, case['selector'], rng, spec['support_size'])
            synchronize(adapter)
            timing['retrieval_and_selection'] += time.perf_counter()-tick
            for name, count in calls.items():
                search_calls[name] += count
            tick = time.perf_counter()
            indices = torch.tensor(support.indices, device=adapter.device, dtype=torch.long)
            weights = torch.tensor(support.weights, device=adapter.device, dtype=base.dtype)
            successors = memory.successors[indices]
            memory_current = memory.contexts[indices][:, -1] if case['control'] == 'delta-transfer' else None
            memory_prediction = (torch.from_numpy(np.array(predictions[support.indices], copy=True)).to(adapter.device)
                                 if case['control'] == 'residual-transfer' else None)
            result = correction(case['control'], base, context.latents[:, -1], successors,
                                memory_current, memory_prediction, weights, case['gain'])
            rows = support_rows(memory, support)
            if h == 0:
                first_support = identity(rows)
            position.update(support_size=len(support.indices), effective_support_size=1/sum(w*w for w in support.weights),
                            support_fingerprint=identity(rows), correction_mse=float((result-base).square().mean()))
            synchronize(adapter)
            timing['correction_and_lookup'] += time.perf_counter()-tick
        require(bool(torch.isfinite(result).all()), 'Nonfinite corrected prediction')
        output.append(result.detach())
        positions.append(position)
        context = Context(torch.cat((context.latents[:, 1:], result[:, None].detach()), dim=1),
                          torch.cat((context.past_actions[:, 1:], action[:, None]), dim=1))
    assert_frozen(adapter)
    synchronize(adapter)
    return torch.stack(output, dim=1), {'first_support_fingerprint': first_support,
        'positions': positions, 'retrieval_calls': search_calls, 'timing_seconds': timing,
        'total_seconds': time.perf_counter()-started, 'optimizer_steps': 0,
        'support_presentations': sum(p['support_size'] for p in positions),
        'mean_effective_support_size': sum(p['effective_support_size'] for p in positions)/len(positions),
        'cached_prediction_lookups': spec['horizon']*spec['support_size'] if case['control'] == 'residual-transfer' else 0,
        'checks': {'all_model_parameters_and_buffers_frozen': True, 'no_query_targets_in_prediction': True,
                   'memory_only': True, 'predicted_context_feedback': True, 'support_constraints': True}}

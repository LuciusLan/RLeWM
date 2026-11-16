"""Detached imagined contexts and explicitly balanced retrieval-query allocation."""

import random

import torch

from ..model import Context
from ..planning.temporal.state import advance


def balanced_pairs(parents, depths, limit, seed):
    """Balance normalized parent/depth exposure without relying on stream order."""
    pairs = [(p, h) for p in range(parents) for h in depths]
    random.Random(seed).shuffle(pairs)
    ranks = {pair: i for i, pair in enumerate(pairs)}
    parent_counts, depth_counts, selected = [0]*parents, {h: 0 for h in depths}, []
    while pairs and len(selected) < limit:
        pair = min(pairs, key=lambda item: (
            parent_counts[item[0]]*parents + depth_counts[item[1]]*len(depths),
            depth_counts[item[1]], parent_counts[item[0]], ranks[item]))
        pairs.remove(pair)
        selected.append(pair)
        parent_counts[pair[0]] += 1
        depth_counts[pair[1]] += 1
    return selected


def select_parents(candidates, elite_indices, count, source, seed):
    if source == "elite":
        order = elite_indices.cpu().tolist()
    else:
        generator = torch.Generator(device="cpu").manual_seed(seed)
        order = torch.randperm(len(candidates), generator=generator).tolist()
    # Keep distinct sequences where possible, then explicitly disclose duplicates.
    selected, duplicate = [], []
    for index in order:
        if any(torch.equal(candidates[index], candidates[old]) for old in selected):
            duplicate.append(index)
        else:
            selected.append(index)
        if len(selected) == count:
            break
    if len(selected) < count:
        selected += duplicate[:count-len(selected)]
    return candidates[selected].detach().clone(), selected


@torch.no_grad()
def rollout_references(adapter, initial, actions):
    context = Context(initial.latents.expand(len(actions), -1, -1).detach(),
                      initial.past_actions.expand(len(actions), -1, -1).detach())
    contexts, predictions = [], []
    for block in actions.unbind(1):
        contexts.append(context)
        prediction = adapter.predict(context, block).detach()
        predictions.append(prediction)
        context = advance(context, block, prediction)
    return contexts, predictions


def pair_tensors(contexts, actions, pairs):
    return (Context(torch.cat([contexts[h].latents[p:p+1] for p, h in pairs]).detach(),
                    torch.cat([contexts[h].past_actions[p:p+1] for p, h in pairs]).detach()),
            torch.cat([actions[p:p+1, h] for p, h in pairs]).detach())

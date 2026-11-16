"""One global support budget over checkpoint query pairs and optional source replay."""

from collections import OrderedDict
from itertools import islice
import random

import torch

from ..model import Context
from ..retrieval import Support, diverse_prefix
from ..planning.shared_ttt import merge_streams


def directions(settings, method):
    alpha = settings["alpha"]
    values = ({"forward": 1.} if method == "forward" else {"reverse": 1.}
              if method == "reverse" else {"forward": 1-alpha, "reverse": alpha})
    return {key: mass for key, mass in values.items() if mass > 0}


def select_support(engine, contexts, actions, method, seed, scan_limit, forward_cache=None):
    retriever, budget = engine.retriever, engine.settings["support_size"]
    if method == "uniform":
        generator = torch.Generator(device="cpu").manual_seed(seed)
        order = torch.randperm(len(engine.memory), generator=generator).numpy()
        ids = diverse_prefix(order, budget, retriever.episode_positions, retriever.episode_cap,
                             retriever.step_positions, retriever.min_step_gap)
        support = Support(ids, [1/len(ids)]*len(ids) if ids else [], {i: ["uniform"] for i in ids}, {})
        return support, {}, {"forward": 0, "reverse": 0, "successor_predictions": 0}
    channels = directions(engine.settings, method)
    if len(actions)*len(channels) > budget:
        raise ValueError("Subsample balanced query pairs before merging; every stream needs a quota")
    predicted = engine.adapter.predict(contexts, actions).detach() if "reverse" in channels else None
    cache = {} if forward_cache is None else forward_cache
    streams, calls = [], {"forward": 0, "reverse": 0,
                          "successor_predictions": len(actions) if predicted is not None else 0}
    for position in range(len(actions)):
        context = Context(contexts.latents[position:position+1], contexts.past_actions[position:position+1])
        for channel, mass in channels.items():
            if channel == "forward":
                if position not in cache:
                    cache[position] = retriever.forward(context, actions[position:position+1])
                    calls["forward"] += 1
                ranking = cache[position]
            else:
                ranking = retriever.reverse(context, predicted[position:position+1])
                calls["reverse"] += 1
            streams.append((f"pair-{position}/{channel}", islice(ranking.order, min(scan_limit, len(ranking))),
                            mass/len(actions)))
    support = merge_streams(streams, budget, retriever)
    support.details["scan_limit_per_stream"] = scan_limit
    return support, cache, calls


def support_change(before, after):
    left = {} if before is None else dict(zip(before.indices, before.weights))
    right = dict(zip(after.indices, after.weights))
    return {"added": len(set(right)-set(left)), "removed": len(set(left)-set(right)),
            "replacement_fraction": len(set(right)-set(left))/len(right) if right and before is not None else None,
            "weight_total_variation": .5*sum(abs(left.get(i, 0)-right.get(i, 0)) for i in left.keys() | right.keys())
            if before is not None else None}


class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity, self.rows = capacity, OrderedDict()

    def insert(self, support, checkpoint):
        for i, weight in zip(support.indices, support.weights):
            self.rows.pop(i, None)
            self.rows[i] = {"last_selected_checkpoint": checkpoint, "weight": weight,
                            "channels": support.channels[i]}
        while len(self.rows) > self.capacity:
            self.rows.popitem(last=False)

    def mix(self, current, retriever, budget, fraction, checkpoint, seed):
        older = [i for i, row in self.rows.items()
                 if row["last_selected_checkpoint"] < checkpoint and i not in current.indices]
        random.Random(seed).shuffle(older)
        old_quota = int(budget*fraction)
        selected, current_ids, old_ids, episodes, starts = [], [], [], {}, {}

        def take(order, target, quota):
            for i in order:
                if len(target) >= quota or len(selected) >= budget:
                    break
                if i in selected:
                    continue
                episode, step = int(retriever.episode_positions[i]), int(retriever.step_positions[i])
                if retriever.episode_cap and episodes.get(episode, 0) >= retriever.episode_cap:
                    continue
                if any(abs(step-old) < retriever.min_step_gap for old in starts.get(episode, [])):
                    continue
                selected.append(i)
                target.append(i)
                episodes[episode] = episodes.get(episode, 0)+1
                starts.setdefault(episode, []).append(step)

        take(current.indices, current_ids, budget-old_quota)
        take(older, old_ids, old_quota)
        take(current.indices, current_ids, budget)
        take(older, old_ids, budget)
        count = len(selected)
        weights = dict(zip(current.indices, current.weights))
        mass = sum(weights[i] for i in current_ids)
        result = [(len(current_ids)/count)*weights[i]/mass if i in current_ids else 1/count for i in selected]
        channels = {i: current.channels[i] if i in current_ids else ["replay"] for i in selected}
        return Support(selected, result, channels, {
            "policy": "current-plus-replay", "current_count": len(current_ids), "replay_count": len(old_ids),
            "current_loss_mass": len(current_ids)/count if count else 0,
            "replay_loss_mass": len(old_ids)/count if count else 0,
            "shortfall": budget-count, "older_eligible": len(older),
            "current_selection": current.details})

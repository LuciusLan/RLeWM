"""Episode-safe recorded suffixes, queried at updated imagined contexts."""

from pathlib import Path

import torch

from ..model import Context
from ..planning.actions import stream_seed
from ..proposals.context_search import ContextSearch, exact_prefix
from ..proposals.fragments import FragmentProvider, FragmentStore
from ..proposals.goal_search import GoalContextSearch
from ..proposals.injection import Injection
from ..proposals.providers import proposal_bundle
from .references import balanced_pairs


class ContinuationProviders:
    def __init__(self, resources, config, coordinates):
        self.config, self.coordinates = config, coordinates
        settings, proposals = config["interleave"], config["proposals"]
        self.search = ContextSearch(resources.engine.retriever, proposals["context_chunk_rows"])
        self.providers, self.initial = {}, None
        goal_directory = Path(config["experiment_dir"])/"retrieval"
        if settings["injection_slots"] and any(arm in settings["arms"] for arm in ("proposals", "combined")):
            for depth in settings["continuation_depths"]:
                horizon = config["planning"]["horizon"]-depth
                store = FragmentStore(resources.memory, horizon, coordinates)
                local = {**proposals, "fragments": settings["fragments_per_pair"],
                         "goal_weight": settings["goal_weight"]}
                goal = GoalContextSearch(self.search, store, local, goal_directory) if settings["goal_weight"] else None
                self.providers[horizon] = FragmentProvider(store, self.search, local, goal)
        if settings["initial_source"] != "gaussian" and proposals["injection_slots"]:
            store = FragmentStore(resources.memory, config["planning"]["horizon"], coordinates)
            goal = (GoalContextSearch(self.search, store, proposals, goal_directory)
                    if settings["initial_source"] == "goal-context-fragments" else None)
            self.initial = FragmentProvider(store, self.search, proposals, goal)

    def initial_bundle(self, adapter, context, metadata, seed, goal):
        if self.initial is None:
            return None, {"source": "gaussian", "enabled": False, "injected_slots": 0}
        return proposal_bundle(self.initial, adapter, context, metadata,
                               self.config["interleave"]["initial_source"], seed, goal)

    def description(self):
        return {"context": self.search.description(), "suffix_horizons": {
            str(h): {"eligible_windows": int(provider.store.eligible.sum()),
                     "goal": provider.goal_search.description() if provider.goal_search else {"enabled": False}}
            for h, provider in self.providers.items()},
            "initial_goal": self.initial.goal_search.description()
            if self.initial is not None and self.initial.goal_search else {"enabled": False}}


def splice_suffix(parent, depth, suffix):
    if not 0 <= depth < len(parent) or suffix.shape != parent[depth:].shape:
        raise ValueError("Suffix must contain exactly H-depth complete action blocks")
    return torch.cat((parent[:depth].detach(), suffix.detach()), 0)


def suffix_injection(seeds, requested, noise_std, seed, template):
    """Nominal missing sources retain Gaussian slots; exact suffixes precede jitter."""
    generator = torch.Generator(device="cpu").manual_seed(seed)
    noise = torch.randn(requested, *template.shape, generator=generator).to(template)
    slots, actions, provenance = [], [], []
    count = len(seeds)
    if count:
        for nominal in range(requested):
            source, copy = nominal % count, nominal // count
            item = seeds[source]
            if item is None:
                continue
            value = item["actions"].clone()
            if copy:
                value[item["depth"]:] += noise_std*noise[nominal, item["depth"]:]
            slots.append(nominal+1)
            actions.append(value)
            provenance.append({key: value for key, value in item.items() if key != "actions"} | {
                "slot": nominal+1, "kind": "exact" if copy == 0 else "perturbed"})
    values = torch.stack(actions) if actions else template.new_empty((0, *template.shape))
    return Injection(slots, values, provenance, requested)


@torch.no_grad()
def retrieve_continuations(providers, initial, contexts, parents, parent_ids, metadata, goal, seed, checkpoint):
    settings = providers.config["interleave"]
    requested, per_pair = settings["injection_slots"], settings["fragments_per_pair"]
    pair_seed = stream_seed(seed, metadata["transition_id"], f"continuation-pairs-{checkpoint}")
    pairs = balanced_pairs(len(parents), settings["continuation_depths"],
                           len(parents)*len(settings["continuation_depths"]), pair_seed)
    seeds, audit, episodes, starts, used = [], [], {}, {}, set()
    calls = 0
    # Allocate nominal slots before selection, preventing scan/parent ordering bias.
    for parent, depth in pairs:
        nominal_start = len(seeds)
        if nominal_start >= requested:
            break
        horizon = parents.shape[1]-depth
        provider = providers.providers[horizon]
        context = (initial if settings["continuation_context"] == "observed" else
                   Context(contexts[depth].latents[parent:parent+1], contexts[depth].past_actions[parent:parent+1]))
        distances = (provider.goal_search.distances(context, goal) if provider.goal_search else
                     provider.search.distances(context))
        calls += 1
        store, p = provider.store, provider.settings
        eligible = store.eligible_indices(metadata["source_episode_position"])
        order = exact_prefix(distances, eligible, min(p["max_scan_candidates"], len(eligible)))
        rejected = {"global_episode_cap": 0, "global_start_spacing": 0, "duplicate_anchor": 0}

        def admissible():
            # Recheck global reservations for each requested fragment.
            for index in order:
                episode, step = int(store.episodes[index]), int(store.steps[index])
                if index in used:
                    rejected["duplicate_anchor"] += 1
                elif episodes.get(episode, 0) >= p["max_fragments_per_episode"]:
                    rejected["global_episode_cap"] += 1
                elif any(abs(step-old) < p["min_start_step_gap"] for old in starts.get(episode, [])):
                    rejected["global_start_spacing"] += 1
                else:
                    yield index

        count = min(per_pair, requested-nominal_start)
        pair_rows, selection = [], []
        for _ in range(count):
            # Reserve each accepted anchor immediately. This enforces the
            # global cap even when a pair requests more than one fragment.
            fragments = store.select(admissible(), 1, metadata["source_episode_position"],
                                     p["max_fragments_per_episode"], p["min_start_step_gap"])
            selection.append(fragments.diagnostics)
            if not fragments.sources:
                break
            suffix, source = fragments.actions[0], fragments.sources[0]
            index = source["memory_index"]
            episode, step = int(store.episodes[index]), int(store.steps[index])
            used.add(index)
            episodes[episode] = episodes.get(episode, 0)+1
            starts.setdefault(episode, []).append(step)
            evidence = {**source, "context_distance": float(provider.search.selected_distances(context, [index])[0]),
                        "retrieval_distance": float(distances[index])}
            if provider.goal_search:
                evidence["goal_endpoint_distance"] = float(provider.goal_search.endpoint_distances(goal, [index])[0])
            candidate = splice_suffix(parents[parent], depth, suffix)
            if not torch.equal(candidate[:depth], parents[parent, :depth]):
                raise AssertionError("Continuation changed its parent prefix")
            item = {"actions": candidate, "parent_candidate_id": int(parent_ids[parent]),
                    "parent_reference_index": parent, "depth": depth, "source": evidence}
            seeds.append(item)
            pair_rows.append(evidence)
        seeds.extend([None]*(count-len(pair_rows)))
        audit.append({"parent_candidate_id": int(parent_ids[parent]), "depth": depth, "horizon": horizon,
                      "query_context": settings["continuation_context"], "fragments": pair_rows,
                      "selection": selection, "fragment_shortfall": count-len(pair_rows),
                      "global_rejections": rejected})
    noise_seed = stream_seed(seed, metadata["transition_id"], f"continuation-noise-{checkpoint}")
    injection = suffix_injection(seeds, requested, settings["noise_std"], noise_seed, parents[0])
    return injection, {"pairs": audit, "requested_slots": requested, "realized_slots": len(injection.slots),
                       "nominal_sources": len(seeds), "selected_sources": sum(item is not None for item in seeds),
                       "slot_provenance": injection.provenance, "noise_seed": noise_seed,
                       "injection_fingerprint": injection.fingerprint, "retrieval_calls": calls,
                       "missing_fragment_fallback": "ordinary Gaussian population slots"}

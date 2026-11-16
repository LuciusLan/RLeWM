"""Reuse shared TTT, exact retrieval, proposal fragments and CEM for a real call."""

import time

from ..planning.actions import provisional_actions, stream_seed
from ..planning.cem import score_sequences
from ..planning.shared_ttt import adapt_shared
from ..proposals.cem import solve_guided_cem
from ..proposals.providers import make_provider, proposal_bundle
from ..planning.pipeline import action_coordinates
from ..rollout.inference import fixed_predictor
from ..rollout.support_updates import synchronize


class Controller:
    def __init__(self, resources, config):
        self.resources, self.config = resources, config
        self.coordinates = action_coordinates(resources, config)
        self.provider = make_provider(resources, config, self.coordinates)

    def plan(self, context, goal, metadata, method, source, seed, warm_start=None):
        adapter, settings = self.resources.adapter, self.config["planning"]
        synchronize(adapter)
        started = time.perf_counter()
        # Proposals have no access to the supplied goal for the two pilot columns.
        injection, proposal = proposal_bundle(self.provider, adapter, context, metadata, source, seed)
        provisional_seed = stream_seed(seed, metadata["transition_id"], "provisional")
        provisional = provisional_actions(self.coordinates, settings["provisional_actions"], settings["initial_std"], provisional_seed)
        adaptation = adapt_shared(self.resources.engine, context, metadata, provisional,
                                  method, seed, settings["support_scan_limit"])
        cem_seed = stream_seed(seed, metadata["transition_id"], "cem")
        with fixed_predictor(adapter):
            guided = solve_guided_cem(lambda actions: score_sequences(adapter, context, goal, actions,
                                     settings["scoring_batch_size"]), settings, self.coordinates, cem_seed,
                                     injection, initial_mean=warm_start)
        synchronize(adapter)
        raw = self.coordinates.to_raw(guided.plan.actions).detach().cpu().numpy()
        return guided.plan.actions.detach(), raw, {
            "method": method, "proposal_source": source, "cem_seed": cem_seed,
            "provisional_seed": provisional_seed, "model_reset_seed": adaptation["reset_seed"],
            "updates": adaptation["updates"], "unique_support": adaptation["unique_support"],
            "support_presentations": adaptation["support_presentations"],
            "parameter_displacement_mse": adaptation["parameter_displacement_mse"],
            "support_fingerprint": adaptation["support_fingerprint"],
            "fragment_fingerprint": proposal["fragment_fingerprint"],
            "injection_fingerprint": proposal["injection_fingerprint"], "injected_slots": proposal["injected_slots"],
            "fragment_shortfall": proposal["diagnostics"].get("fragment_shortfall", 0),
            "base_population_fingerprint": guided.base_first_population_fingerprint,
            "warm_start_used": warm_start is not None, "model_goal_sse": guided.plan.cost,
            "candidate_evaluations": guided.plan.candidate_evaluations,
            "planning_seconds": time.perf_counter()-started,
            "selected_raw_actions": raw.reshape(-1, raw.shape[-1]).tolist()}

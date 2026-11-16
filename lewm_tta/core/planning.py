"""Reuse the interleave solver and exact first-call budget snapshots."""

from collections import deque
from copy import deepcopy
import time

from ..interleave.anytime import diagnostic_row
from ..interleave.cem import solve_interleaved_cem
from ..interleave.session import CheckpointSession
from ..interleave.snapshots import SnapshotRecorder
from ..planning.actions import stream_seed
from ..rollout.record_io import compact_record
from ..rollout.support_updates import synchronize
from ..simulation.history import ObservedHistory


class LiveHistory(ObservedHistory):
    """Initialize from the verified cached past; append only actually observed RGB."""
    def __init__(self, adapter, coordinates, context):
        from utils import get_img_preprocessor
        self.adapter, self.coordinates = adapter, coordinates
        self.transform = get_img_preprocessor("pixels", "pixels", adapter.image_size)
        self.latents = deque(context.latents.unbind(1), maxlen=adapter.history)
        self.actions = deque(context.past_actions.unbind(1), maxlen=adapter.history-1)
        if len(self.latents) != adapter.history or len(self.actions) != adapter.history-1:
            raise ValueError("Initial live history must contain the complete observed past")


def search(resources, config, providers, coordinates, context, metadata, goal, arm, seed, budgets, accounting, warm=None):
    synchronize(resources.adapter)
    started = time.perf_counter()
    session = CheckpointSession(resources.engine, context, metadata, goal, coordinates, providers, config, arm, seed)
    initial_started = time.perf_counter()
    initial, initial_audit = providers.initial_bundle(resources.adapter, context, metadata, seed, goal)
    synchronize(resources.adapter)
    session.timings["initial_proposals"] = time.perf_counter()-initial_started
    session.counters["initial_action_retrievals"] = int(initial is not None)
    recorder = SnapshotRecorder(session, started)
    solve_interleaved_cem(session.score, config["planning"], coordinates,
        stream_seed(seed, metadata["transition_id"], "cem"), config["interleave"]["checkpoints"], session,
        initial, initial_mean=warm, snapshot_iterations=budgets, on_snapshot=recorder)
    synchronize(resources.adapter)
    accounting["instrumented_search_seconds"] += time.perf_counter()-started
    accounting["search_calls"] += 1
    accounting["actual_cem_sequence_evaluations"] += config["planning"]["population"]*max(budgets)+len(budgets)
    accounting["extra_terminal_mean_probes"] += recorder.extra_terminal_probes
    accounting["measurement_overhead_seconds"] += recorder.overhead_seconds
    for name, value in session.counters.items():
        accounting["checkpoint_counters"][name] = accounting["checkpoint_counters"].get(name, 0)+value
    return recorder, initial, initial_audit


def first_plan_row(recorder, snapshot, task, bank, index, arm, seed, initial, audit, accounting):
    row = diagnostic_row(recorder, snapshot, task, bank, index, arm, seed, initial, audit)
    accounting["diagnostic_sequence_evaluations"] += 3
    accounting["diagnostic_seconds"] += row["diagnostic_seconds_with_checks"]
    return compact_record(row)


def call_row(snapshot, arm, seed, metadata):
    """A real replan has no recorded future against which to score diagnostics."""
    return compact_record({"arm": arm, "seed": seed, "query": metadata,
        "adaptation": deepcopy(snapshot.adaptation), "checkpoints": snapshot.checkpoints,
        "counters": snapshot.counters, "timing_seconds": snapshot.timing_seconds,
        "cem_trace": snapshot.plan.trace, "selected_model_goal_sse": snapshot.plan.cost,
        "selected_actions_normalized": snapshot.plan.actions.cpu().tolist(),
        "cem_candidate_evaluations": snapshot.plan.candidate_evaluations,
        "cem_transition_predictions": snapshot.plan.candidate_evaluations*len(snapshot.plan.actions),
        "primary_seconds_with_checks": snapshot.primary_seconds})

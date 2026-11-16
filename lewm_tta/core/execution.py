"""Execute a budget's own trajectory; real replans never share divergent futures."""

import time

from ..interleave.anytime import stopped_config
from ..outcomes.metrics import PlanOutcome
from ..outcomes.pipeline import compute_record
from ..rollout.support_updates import synchronize
from .planning import LiveHistory, call_row, search


def add_compute(target, values):
    for name, value in values.items():
        target[name] = target.get(name, 0)+value


def execute(replay, case, task, goal, resources, config, providers, coordinates, arm, seed,
            snapshot, source_row, accounting, append_call, condition):
    started = time.perf_counter()
    observation, prefix = replay.reset(case)
    physical_goal = replay.goal(case)
    outcome = PlanOutcome(replay.metrics(observation, physical_goal))
    controls = config["core"]["control_budget"]
    early_stop = controls == 50
    count, planning_seconds, encoding_seconds, compute = 0, 0., 0., {}
    actions_executed = []
    history = LiveHistory(resources.adapter, coordinates, task.context) if early_stop else None
    local = stopped_config(config, snapshot.iterations, config["experiment_dir"])
    plan, pending = snapshot.plan, source_row
    while outcome.steps < controls and not (early_stop and outcome.ever_success):
        if count:
            metadata = {**task.metadata, "transition_id": f"{task.metadata['transition_id']}/real-replan-{count}",
                        "simulator_raw_steps": outcome.steps}
            # The full horizon was executed; shifting it leaves an empty warm
            # start, equivalently a fresh normalized zero mean, as upstream K=H.
            recorder, _, _ = search(resources, local, providers, coordinates, history.context(), metadata,
                goal, arm, seed, [snapshot.iterations], accounting, warm=plan.actions[len(plan.actions):])
            current = recorder.snapshots[snapshot.iterations]
            pending = call_row(current, arm, seed, metadata)
            plan = current.plan
            pending["selected_raw_actions"] = coordinates.to_raw(plan.actions).flatten(0, 1).cpu().tolist()
        latency = pending["primary_seconds_with_checks"]
        planning_seconds += latency
        add_compute(compute, compute_record(pending))
        call = {"case_id": case["case_id"], "iterations": snapshot.iterations, "condition": condition,
                "seed": seed, "replan": count, "raw_steps_before": outcome.steps,
                "first_call_snapshot": count == 0, **pending}
        count += 1
        raw = coordinates.to_raw(plan.actions).detach().cpu().numpy()
        executed = []
        for block in raw:
            for action in block:
                observation = replay.step(action)
                outcome.append(replay.metrics(observation, physical_goal))
                executed.append(action.tolist())
                actions_executed.append(action.tolist())
                if outcome.steps == controls or (early_stop and outcome.ever_success):
                    break
            if outcome.steps == controls or (early_stop and outcome.ever_success):
                break
            if early_stop:
                synchronize(resources.adapter)
                encoding_started = time.perf_counter()
                history.append(observation.rgb, block)
                synchronize(resources.adapter)
                encoding_seconds += time.perf_counter()-encoding_started
        append_call({**call, "executed_raw_actions": executed})
    return {**outcome.record(), "prefix_reconstruction": prefix, "planning_calls": count,
        "planning_seconds": planning_seconds+encoding_seconds, "search_seconds": planning_seconds,
        "context_encoding_seconds": encoding_seconds, "compute": compute,
        "executed_raw_actions": actions_executed, "wall_seconds": time.perf_counter()-started,
        "wall_seconds_scope": "Reset/physics and later real replans; shared first search and diagnostics are charged in the report accounting"}

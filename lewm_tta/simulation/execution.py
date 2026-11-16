"""Environment-independent fixed-horizon execution and real receding-horizon MPC."""

import hashlib
import time
import numpy as np

from ..runtime import object_hash
from ..rollout.support_updates import synchronize
from .history import ObservedHistory, bootstrap


class Outcome:
    def __init__(self, observation, directions):
        self.initial, self.final = observation, observation
        self.best = {key: float(value) for key, value in observation.metrics.items()}
        self.directions, self.steps, self.actions = directions, 0, []
        self.ever_success = bool(observation.metrics["success"])
        self.first_success_raw_step = 0 if self.ever_success else None
        self.ever_out_of_frame = bool(observation.metrics.get("out_of_frame", False))

    def append(self, observation, action):
        self.steps += 1
        self.actions.append(np.asarray(action).tolist())
        self.final = observation
        self.ever_success |= bool(observation.metrics["success"])
        if observation.metrics["success"] and self.first_success_raw_step is None:
            self.first_success_raw_step = self.steps
        self.ever_out_of_frame |= bool(observation.metrics.get("out_of_frame", False))
        for key, value in observation.metrics.items():
            self.best[key] = (max if self.directions[key] > 0 else min)(self.best[key], float(value))

    def record(self):
        return {"initial_metrics": self.initial.metrics, "final_metrics": self.final.metrics,
                "best_metrics": self.best, "ever_success": self.ever_success,
                "initial_success": bool(self.initial.metrics["success"]), "raw_steps": self.steps,
                "first_success_raw_step": self.first_success_raw_step, "ever_out_of_frame": self.ever_out_of_frame,
                "initial_state": self.initial.state.tolist(), "final_state": self.final.state.tolist(),
                "executed_raw_actions": self.actions}


def replay(environment, case, seed, raw_actions, *, recorded=False):
    started = time.perf_counter()
    initial = environment.reset(case, seed, history_start=False)
    outcome = Outcome(initial, environment.description()["metric_directions"])
    for action in raw_actions:
        observation = environment.step(action, recorded=recorded)
        outcome.append(observation, action)
        if observation.truncated:
            raise ValueError("Fixed-horizon replay truncated; do not silently change its endpoint")
        # Ignore success termination intentionally: every saved plan gets H*K steps.
    return {**outcome.record(), "action_origin": "recorded" if recorded else "planner",
            "wall_seconds": time.perf_counter()-started,
            "planning_calls": 0, "planning_seconds": 0., "context_encoding_seconds": 0.}


def history_fingerprint(images, blocks):
    digest = hashlib.sha256()
    for value in [*images, *blocks]:
        value = np.ascontiguousarray(value)
        digest.update(str((value.shape, str(value.dtype))).encode())
        digest.update(value.tobytes())
    return digest.hexdigest()


def mpc(environment, dataset, case, seed, method, source, controller, manifest, config, calls):
    adapter = controller.resources.adapter
    started = time.perf_counter()
    images, blocks, observation = bootstrap(environment, dataset, case, seed, manifest["history"], manifest["frameskip"])
    initial_fingerprint = history_fingerprint(images, blocks)
    outcome = Outcome(observation, environment.description()["metric_directions"])
    reconstruction = environment.reconstruction(observation, case["current_raw_row"])
    synchronize(adapter)
    encoding_started = time.perf_counter()
    history = ObservedHistory(adapter, controller.coordinates, images, blocks)
    synchronize(adapter)
    pending_encoding = time.perf_counter()-encoding_started
    encoding_seconds, planning_seconds, count = pending_encoding, 0., 0
    # This is the supplied task goal, not an observation revealed along a rollout.
    from .green import enabled
    if enabled(config):
        goal_encoding_started = time.perf_counter()
        goal = history.encode(environment.goal_image).detach()
        synchronize(adapter)
        goal_encoding_seconds = time.perf_counter()-goal_encoding_started
        pending_encoding += goal_encoding_seconds
        encoding_seconds += goal_encoding_seconds
    else:
        goal_store = (controller.resources.queries(manifest["split"]).store if config.get("query_dataset")
                      else controller.resources.memory.store)
        goal = goal_store.latents[case["goal_raw_row"]:case["goal_raw_row"]+1].detach()
    goal_fingerprint = object_hash(goal.cpu().tolist())
    warm = None
    max_steps = config["simulation"]["max_raw_steps"]
    while outcome.steps < max_steps and not observation.metrics["success"]:
        metadata = {**case["query"], "transition_id": f"{case['case_id']}/real-replan-{count}",
                    "simulator_raw_steps": outcome.steps}
        # Neither dataset future actions nor any simulator state enters the planner.
        actions, raw, call = controller.plan(history.context(), goal, metadata, method, source, seed, warm)
        call.update(case_id=case["case_id"], query=case["query"], seed=seed, replan=count,
                    simulator_raw_steps_before=outcome.steps, context_encoding_seconds=pending_encoding,
                    goal_fingerprint=goal_fingerprint)
        planning_seconds += call["planning_seconds"]
        count += 1
        executed, complete_blocks, next_encoding = [], 0, 0.
        for block in raw[:config["simulation"].get("execute_blocks", 1)]:
            observed_block = []
            for action in block:
                observation = environment.step(action)
                executed.append(action)
                observed_block.append(action)
                outcome.append(observation, action)
                if observation.metrics["success"] or observation.terminated or observation.truncated or outcome.steps == max_steps:
                    break
            if observation.metrics["success"] or observation.terminated or observation.truncated or outcome.steps == max_steps:
                break
            if len(observed_block) != manifest["frameskip"]:
                raise AssertionError("Only complete observed blocks can enter the next model history")
            synchronize(adapter)
            encoding_started = time.perf_counter()
            history.append(observation.rgb, np.asarray(observed_block))
            synchronize(adapter)
            next_encoding += time.perf_counter()-encoding_started
            complete_blocks += 1
        call.update(executed_raw_actions=np.asarray(executed).tolist(), realized_metrics=observation.metrics,
                    planning_plus_encoding_seconds=call["planning_seconds"]+pending_encoding)
        calls.append(call)
        if observation.metrics["success"] or observation.terminated or observation.truncated or outcome.steps == max_steps:
            break
        pending_encoding = next_encoding
        encoding_seconds += pending_encoding
        warm = actions[complete_blocks:]  # Shift only executed blocks; model/optimizer still reset next call.
    return {**outcome.record(), "planning_calls": count, "planning_seconds": planning_seconds,
            "context_encoding_seconds": encoding_seconds, "wall_seconds": time.perf_counter()-started,
            "warmup_raw_steps": (manifest["history"]-1)*manifest["frameskip"],
            "bootstrap_fingerprint": initial_fingerprint, "goal_fingerprint": goal_fingerprint,
            "mpc_current_reconstruction": reconstruction,
            **({"goal_contract": environment.goal_contract} if enabled(config) else {})}

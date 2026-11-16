"""Execute saved open-loop plans from verified fresh reset/past receipts.

No world model is loaded. Future recorded controls are used only in the
separate reconstruction control, after planning, never to initialize a plan.
"""

import time

import numpy as np

from ..collection.manifest import checked
from ..simulation.dataset import SourceDataset
from ..simulation.environments import make_collection_environment
from .metrics import PlanOutcome, goal_metrics


def comparable_environment(description):
    value = dict(description)
    if "installed_sources" in value:
        # Installation roots can differ between the model and physics venvs.
        value["installed_sources"] = sorted(value["installed_sources"].values())
    return value


class FreshReplay:
    def __init__(self, query, config, settings):
        from ..cache_loading import cache_only
        if cache_only(config):
            from ..collection.queries import verify_query_payloads
            verify_query_payloads(query)
        if query.get("appearance"):
            raise ValueError("Appearance-derived query renders need their own replay adapter")
        self.query, self.config, self.settings = query, config, settings
        manifest = checked(query["collection_manifest_path"])
        if manifest["fingerprint"] != query["collection_manifest"]:
            raise ValueError("Original collection receipt changed")
        self.collection = manifest["config"]
        self.backend = self.collection["backend"]
        self.dataset = SourceDataset(query["source"]["path"])
        try:
            self.environment = make_collection_environment(self.collection, query["regime"])
            self.description = self.environment.description()
            self.fields = list(self.environment.collection_schema().get("fields", {}))
        except BaseException:
            if hasattr(self, "environment"):
                self.environment.close()
            self.dataset.close()
            raise

    def close(self):
        try:
            self.environment.close()
        finally:
            self.dataset.close()

    def goal(self, case):
        row = case["goal_raw_row"]
        return {key: self.dataset.rows(key, row) for key in ("state64", *self.fields)}

    def metrics(self, observation, goal):
        coverage = None
        if self.backend == "pusht":
            from shapely.geometry import Polygon
            from shapely.ops import unary_union
            import pymunk
            body = self.environment.env.unwrapped.block
            target = pymunk.Body(body.mass, body.moment)
            target.center_of_gravity = body.center_of_gravity
            target.angle = float(goal["state64"][4])
            target.position = goal["state64"][2:4].tolist()
            def polygon(pose):
                return unary_union([Polygon([tuple(pose.local_to_world(v)) for v in shape.get_vertices()])
                                    for shape in body.shapes])
            desired = polygon(target)
            coverage = float(np.clip(polygon(body).intersection(desired).area/desired.area, 0., 1.))
        return goal_metrics(self.backend, observation, goal, pusht_success=self.settings["pusht_success"],
                            coverage_threshold=self.settings["coverage_threshold"], coverage=coverage)

    def reconstruction(self, observation, row):
        expected = self.dataset.rows("state64", row)
        if observation.state.shape != expected.shape or not np.isfinite(observation.state).all():
            raise ValueError("Replay state shape/nonfiniteness differs from original collection")
        state_error = float(np.max(np.abs(observation.state-expected)))
        rgb = self.dataset.rows("pixels", row)
        if observation.rgb.shape != rgb.shape:
            raise ValueError("Replay RGB shape differs from original collection")
        rgb_error = float(np.abs(observation.rgb.astype(float)-rgb.astype(float)).mean())
        if state_error > self.settings["state_atol"] or rgb_error > self.settings["rgb_mae_atol"]:
            raise ValueError(f"Fresh reset/prefix replay drift at row {row}: state max={state_error:.6g}, RGB MAE={rgb_error:.6g}; preserve the original simulator/physics environment")
        return {"state_max_abs_error": state_error, "rgb_mae_0_255": rgb_error}

    def recorded_step(self, row):
        # Collection changes hidden behavior waypoints before stepping. Those
        # target markers never drive the supplied-goal success metric. A next
        # frame's marker restores only this target, never physical qpos/qvel.
        native = self.environment.env.unwrapped
        if self.backend == "cube":
            native.set_target_pos(0, self.dataset.rows("mocap_pos", row+1)[0],
                                  self.dataset.rows("mocap_quat", row+1)[0])
        elif self.backend == "tworooms":
            native._set_goal_state(self.dataset.rows("target_pos", row+1))
        return self.step(self.dataset.rows("action", row))

    def step(self, action):
        native = self.environment.env.unwrapped
        if getattr(native, "_reset_next_step", False):
            raise ValueError("Native simulator requested an implicit reset inside a plan")
        observation = self.environment.step(action)
        if observation.truncated:
            raise ValueError("Native episode truncated before the fixed P3 execution horizon")
        return observation

    def reset(self, case):
        self.dataset.validate_case(case, self.config["p1"]["data"]["frameskip"],
                                   self.config["p1"]["data"]["history"])
        position = case["episode_position"]
        receipt = self.query["episode_files"][position]
        if comparable_environment(receipt["environment"]) != comparable_environment(self.description):
            raise ValueError("Installed simulator code/packages/settings differ from the original fresh collection receipt")
        reserved = self.query["cases"][position]
        if reserved["episode_id"] != case["query"]["episode_id"]:
            raise ValueError("Reset receipt belongs to another episode")
        observation = self.environment.reset(reserved["reset_seed"])
        skip = self.config["p1"]["data"]["frameskip"]
        checkpoints = {case["episode_start_raw_row"], *range(case["start_raw_row"], case["current_raw_row"]+1, skip)}
        errors = []
        if case["episode_start_raw_row"] in checkpoints:
            errors.append(self.reconstruction(observation, case["episode_start_raw_row"]))
        for row in range(case["episode_start_raw_row"], case["current_raw_row"]):
            observation = self.recorded_step(row)
            if row+1 in checkpoints:
                errors.append(self.reconstruction(observation, row+1))
        return observation, {key: max(error[key] for error in errors) for key in errors[0]}

    def execute(self, case, actions=None):
        started = time.perf_counter()
        observation, prefix = self.reset(case)
        goal = self.goal(case)
        outcome = PlanOutcome(self.metrics(observation, goal))
        errors = []
        steps = case["goal_raw_row"]-case["current_raw_row"]
        if actions is not None and len(actions) != steps:
            raise ValueError("Executed action count differs from the fixed P3 horizon")
        for index in range(steps):
            row = case["current_raw_row"]+index
            observation = self.recorded_step(row) if actions is None else self.step(actions[index])
            if actions is None:
                errors.append(self.reconstruction(observation, row+1))
            outcome.append(self.metrics(observation, goal))
        result = {**outcome.record(), "prefix_reconstruction": prefix,
                  "physical_execution_seconds": time.perf_counter()-started}
        if errors:
            result["recorded_future_reconstruction"] = {key: max(error[key] for error in errors) for key in errors[0]}
            if not result["final_success"]:
                raise ValueError("Recorded-action reconstruction did not satisfy its own supplied future goal")
        return result

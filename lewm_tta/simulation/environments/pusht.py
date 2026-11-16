"""SWM PushT: seven-coordinate dataset states, relative controls and physical goals."""

import importlib.metadata
import numpy as np

from ...runtime import file_hash
from .base import Observation


def goal_metrics(state, goal):
    state, goal = np.asarray(state), np.asarray(goal)
    angle = float(abs((state[4]-goal[4]+np.pi) % (2*np.pi)-np.pi))
    position = float(np.linalg.norm(state[:4]-goal[:4]))
    return {"position_error": position, "block_position_error": float(np.linalg.norm(state[2:4]-goal[2:4])),
            "angle_error": angle, "success": bool(position < 20 and angle < np.pi/9)}


class PushTEnvironment:
    @staticmethod
    def validate_config(settings, planning):
        if settings["image_size"] != 224:
            raise ValueError("The released PushT source/checkpoint uses 224x224 RGB")
        if planning["raw_action_low"] != [-1, -1] or planning["raw_action_high"] != [1, 1]:
            raise ValueError("SWM PushT requires the relative [-1,1]^2 raw-action contract")

    def __init__(self, settings, dataset):
        import gymnasium as gym
        import stable_worldmodel  # noqa: F401 -- register swm/*.
        from pathlib import Path
        from stable_worldmodel.envs.pusht import env as module
        self.dataset, self.settings = dataset, settings
        self.env = gym.make("swm/PushT-v1", render_mode="rgb_array", relative=True,
                            resolution=settings["image_size"], with_target=True)
        self.source_sha256 = file_hash(Path(module.__file__))
        self.goal = None
        if (self.env.action_space.shape != (2,) or not np.all(self.env.action_space.low == -1)
                or not np.all(self.env.action_space.high == 1)):
            raise ValueError("Unexpected SWM PushT raw action contract")

    def reset(self, case, seed, *, history_start=False):
        row = case["start_raw_row"] if history_start else case["current_raw_row"]
        state = self.dataset.rows("state", row).astype(np.float64)
        self.goal = self.dataset.rows("state", case["goal_raw_row"]).astype(np.float64)
        if state.shape != (7,) or self.goal.shape != (7,) or not np.isfinite([state, self.goal]).all():
            raise ValueError("PushT needs finite seven-coordinate recorded states")
        obs, _ = self.env.reset(seed=int(seed), options={"state": state.copy(), "goal_state": self.goal.copy()})
        return self._observation(obs["state"])

    def _coverage(self):
        import pymunk
        from shapely.geometry import Polygon
        from shapely.ops import unary_union
        block = self.env.unwrapped.block
        # The T body has a nonzero center of gravity. Use Pymunk transforms;
        # rotating local vertices around the recorded position is insufficient.
        target_body = pymunk.Body(block.mass, block.moment)
        target_body.center_of_gravity = block.center_of_gravity
        target_body.angle = float(self.goal[4])
        target_body.position = self.goal[2:4].tolist()
        def polygon(body):
            return unary_union([Polygon([tuple(body.local_to_world(v)) for v in shape.get_vertices()])
                                for shape in block.shapes])
        current, target = polygon(block), polygon(target_body)
        return float(np.clip(current.intersection(target).area / target.area, 0, 1))

    def _observation(self, state, terminated=False, truncated=False):
        state = np.asarray(state, dtype=np.float64).copy()
        metrics = goal_metrics(state, self.goal)
        metrics["coverage"] = self._coverage()
        rgb = self.env.render()
        if rgb.shape != (self.settings["image_size"], self.settings["image_size"], 3) or rgb.dtype != np.uint8:
            raise ValueError("Unexpected PushT RGB rendering contract")
        if not np.isfinite(state).all() or not all(np.isfinite(float(v)) for v in metrics.values()):
            raise ValueError("Nonfinite simulator observation/metric")
        return Observation(rgb.copy(), state, metrics, bool(terminated), bool(truncated))

    def step(self, raw_action, *, recorded=False):
        raw_action = np.asarray(raw_action, dtype=np.float32)
        if raw_action.shape != (2,) or not np.isfinite(raw_action).all():
            raise ValueError("PushT expects a finite two-component relative raw action")
        if recorded:
            # The source expert corpus exceeds Gym's nominal Box in some rows.
            # Native PushT.step applies these commands without saturation; preserve
            # them for reconstruction and use the same values in action history.
            applied = raw_action.copy()
        else:
            if (abs(raw_action) > 1+1e-6).any():
                raise ValueError("Planned PushT actions must lie in [-1, 1]; recorded replay requires recorded=True")
            applied = np.clip(raw_action, -1, 1)  # Round-off at mapped CEM bounds only.
        obs, _, terminated, truncated, _ = self.env.step(applied)
        return self._observation(obs["state"], terminated, truncated)

    def reconstruction(self, observation, row):
        state = self.dataset.rows("state", row)
        return {"position_error": float(np.linalg.norm(observation.state[:4]-state[:4])),
                "angle_error": float(abs((observation.state[4]-state[4]+np.pi) % (2*np.pi)-np.pi)),
                "rgb_mae_0_255": float(np.abs(observation.rgb.astype(float)-self.dataset.rows("pixels", row)).mean())}

    def description(self):
        return {"backend": "pusht", "environment_id": "swm/PushT-v1", "source_sha256": self.source_sha256,
                "metric_directions": {"position_error": -1, "block_position_error": -1, "angle_error": -1,
                                      "success": 1, "coverage": 1},
                "packages": {name: importlib.metadata.version(name) for name in
                             ("stable-worldmodel", "gymnasium", "pymunk", "pygame", "shapely", "opencv-python-headless")},
                "raw_actions": "relative controls; planned bounds [-1,1]^2; SWM controller scale 100",
                "recorded_action_policy": "finite two-component source commands executed unchanged, including values outside the nominal Gym Box; no clipping or rescaling",
                "image_size": self.settings["image_size"],
                "controller": {"action_scale": self.env.unwrapped.action_scale,
                               "control_hz": self.env.unwrapped.control_hz, "physics_dt": self.env.unwrapped.dt,
                               "k_p": self.env.unwrapped.k_p, "k_v": self.env.unwrapped.k_v},
                "primary_success": "combined agent/block position error <20 pixels AND wrapped block-angle error <pi/9",
                "coverage": "T-block overlap / target T-block area; secondary metric, not primary success",
                "reset_limit": "recorded state omits block velocity/contact state; native setter advances one physics substep",
                "goal": "fixed recorded physical goal; RGB goal uses recorded image; stock visual overlay unchanged"}

    def close(self):
        self.env.close()

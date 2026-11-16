"""Native SWM scenes and explicit shifts, shared by collection and fresh MPC.

The original recorded-start adapter is intentionally unchanged. Shape changes
alter contact geometry; recoloring changes observations, not physical dynamics.
"""

import importlib.metadata
from pathlib import Path

import numpy as np

from ...runtime import file_hash, object_hash
from .base import Observation

SHAPES = ("o", "L", "T", "Z", "square", "I", "small_tee", "+")


def validate_shift(shift):
    if not isinstance(shift, dict) or set(shift)-{"geometry", "visual", "dynamics"}:
        raise ValueError("A shift exposes geometry, visual and dynamics mappings only")
    geometry, visual, dynamics = (shift.get(key, {}) for key in ("geometry", "visual", "dynamics"))
    if not all(isinstance(part, dict) for part in (geometry, visual, dynamics)):
        raise ValueError("Shift components must be mappings")
    if set(geometry)-{"block_shape"} or geometry.get("block_shape", "T") not in SHAPES[1:]:
        raise ValueError(f"block_shape must be one of {SHAPES[1:]}")
    if set(visual)-{"agent_color", "block_color", "goal_color", "background_color"}:
        raise ValueError("Unknown visual shift; use explicit RGB color overrides")
    for color in visual.values():
        if not isinstance(color, list) or len(color) != 3 or any(type(c) is not int or not 0 <= c <= 255 for c in color):
            raise ValueError("Colors must be three integer RGB components in [0,255]")
    if set(dynamics)-{"block_mass_scale", "block_friction_scale", "agent_kp_scale", "agent_kv_scale", "space_damping"}:
        raise ValueError("Unsupported dynamics override; add it to the task adapter rather than accepting arbitrary kwargs")
    for key, number in dynamics.items():
        if isinstance(number, bool) or not isinstance(number, (int, float)) or not np.isfinite(number):
            raise ValueError("Dynamics overrides must be finite numbers")
        if (key == "space_damping" and not 0 <= number <= 1) or (key != "space_damping" and number <= 0):
            raise ValueError("Scales must be positive; Pymunk space_damping lies in [0,1]")


class FreshPushT:
    """No dataset access. Every reset starts a new native simulator episode."""

    validate_shift = staticmethod(validate_shift)

    @staticmethod
    def validate_collection_config(config):
        if (config["history"], config["frameskip"], config["image_size"], config["goal_offset_steps"]) != (3, 5, 224, 25):
            raise ValueError("LeWM PushT collection requires H3, frameskip 5, RGB224 and a 25-raw-step goal offset")

    @staticmethod
    def validate_collection_policy(config):
        settings = config["policy"]
        if settings.get("name") in ("contact-exploration", "random"):
            if set(settings) != {"name"}:
                raise ValueError("Legacy collection policies accept only name")
        elif settings.get("name") == "bounded-contact":
            if set(settings) != {"name", "max_action_norm", "push_distance", "safety_margin"}:
                raise ValueError("bounded-contact requires max_action_norm, push_distance and safety_margin")
            for key in ("max_action_norm", "push_distance", "safety_margin"):
                number = settings[key]
                if isinstance(number, bool) or not isinstance(number, (int, float)) or not np.isfinite(number) or number <= 0:
                    raise ValueError(f"Invalid bounded-contact.{key}")
            if settings["max_action_norm"] > 1 or settings["push_distance"] > 40 or settings["safety_margin"] >= 128:
                raise ValueError("Use max_action_norm <= 1, push_distance <= 40 and safety_margin < 128")
            if "quality" not in config or settings["safety_margin"] <= config["quality"]["visibility_margin"]:
                raise ValueError("bounded-contact requires quality gates and safety_margin > visibility_margin")
        else:
            raise ValueError("Unknown collection policy")
        if config.get("quality", {}).get("visibility_margin", 0) >= 128:
            raise ValueError("PushT visibility_margin must be below 128 native pixels")
        if config.get("quality") and config["quality"]["coordinate_units"] != "native_scene_pixels":
            raise ValueError("PushT quality distances use native_scene_pixels (512 scene, rendered at RGB224)")

    @staticmethod
    def collection_schema():
        return {"state_width": 7, "action_width": 2, "proprio_indices": [0, 1, 5, 6]}

    @staticmethod
    def goal_diagnostics(state, goal, shift):
        position = float(np.linalg.norm(state[:4]-goal[:4]))
        shape = shift.get("geometry", {}).get("block_shape", "T")
        period = {"square": np.pi/2, "+": np.pi/2, "I": np.pi, "Z": np.pi}.get(shape, 2*np.pi)
        difference = float(abs(state[4]-goal[4]) % period)
        angle = float(min(difference, period-difference))
        distance = float(np.linalg.norm(state[2:4]-goal[2:4]))
        return {"block_distance": distance, "position_distance": position, "angle_distance": angle,
                "block_movement_over_5_pixels": distance > 5,
                "initial_success": bool(position < 20 and angle < np.pi/9)}

    def __init__(self, image_size=224, shift=None):
        import gymnasium as gym
        import stable_worldmodel  # noqa: F401
        from stable_worldmodel.envs.pusht import env as module
        self.shift = shift or {}
        validate_shift(self.shift)
        self.env = gym.make("swm/PushT-v1", render_mode="rgb_array", relative=True,
                            resolution=image_size, with_target=True)
        self.image_size = image_size
        self.source_sha256 = file_hash(Path(module.__file__))
        self.goal = None
        self._base_kp, self._base_kv = self.env.unwrapped.k_p, self.env.unwrapped.k_v

    def description(self):
        return {"backend": "pusht", "environment_id": "swm/PushT-v1", "native_source_sha256": self.source_sha256,
                "packages": {name: importlib.metadata.version(name) for name in
                             ("stable-worldmodel", "gymnasium", "pymunk", "pygame", "shapely")},
                "image_size": self.image_size, "relative": True, "action_scale": 100,
                "collection_schema": self.collection_schema(),
                "control_hz": self.env.unwrapped.control_hz, "physics_dt": self.env.unwrapped.dt,
                "shift": self.shift, "shift_fingerprint": object_hash(self.shift),
                "shift_classification": {"geometry": "contact/geometry shift", "visual": "appearance shift",
                                         "dynamics": "physical/controller parameter shift"},
                "metric_directions": {"position_error": -1, "block_position_error": -1,
                                      "angle_error": -1, "success": 1, "coverage": 1},
                "goal_convention": "separate observed future goal image/state; the decorative marker stays native, as in LeWM"}

    def reset(self, seed):
        values = {}
        shape = self.shift.get("geometry", {}).get("block_shape", "T")
        values["block.shape"] = SHAPES.index(shape)
        for key, color in self.shift.get("visual", {}).items():
            obj = key.removesuffix("_color")
            values[f"{obj}.color"] = np.asarray(color, dtype=np.uint8)
        obs, _ = self.env.reset(seed=int(seed), options={"variation_values": values})
        native = self.env.unwrapped
        dynamics = self.shift.get("dynamics", {})
        if "block_mass_scale" in dynamics:
            native.block.mass *= dynamics["block_mass_scale"]
            native.block.moment *= dynamics["block_mass_scale"]
        if "block_friction_scale" in dynamics:
            for item in native.block.shapes:
                item.friction *= dynamics["block_friction_scale"]
        native.k_p = self._base_kp*dynamics.get("agent_kp_scale", 1.)
        native.k_v = self._base_kv*dynamics.get("agent_kv_scale", 1.)
        if "space_damping" in dynamics:
            native.space.damping = dynamics["space_damping"]
        self.goal = None
        self._collection_contacts = None
        self._collection_policy = None
        return self._observe(obs["state"])

    def begin_collection(self, policy):
        """Attach measurement-only contacts after reset rebuilt the native space."""
        from .pusht_collection import BoundedContactPolicy, agent_block_pair
        native = self.env.unwrapped
        if any(shape.collision_type != 0 for body in (native.agent, native.block) for shape in body.shapes):
            raise ValueError("Native contact types changed; update the collection contact adapter")
        self._collection_contacts = 0
        def post_solve(arbiter, space, data):
            native._handle_collision(arbiter, space, data)
            if agent_block_pair(arbiter, native):
                self._collection_contacts += len(arbiter.contact_point_set.points)
        native.space.on_collision(0, 0, post_solve=post_solve)
        if policy["name"] == "bounded-contact":
            self._collection_policy = BoundedContactPolicy(native, policy)

    def collection_measurement(self):
        from .pusht_collection import body_bounds
        native = self.env.unwrapped
        return {"bounds": body_bounds(native), "contacts": int(self._collection_contacts or 0),
                "extent": float(native.window_size)}

    def set_goal(self, state):
        self.goal = np.asarray(state, dtype=np.float64).copy()
        self.env.unwrapped._set_goal_state(self.goal.copy())

    def _observe(self, state, terminated=False, truncated=False):
        state = np.asarray(state, dtype=np.float64).copy()
        rgb = self.env.render()
        if rgb.shape != (self.image_size, self.image_size, 3) or rgb.dtype != np.uint8 or not np.isfinite(state).all():
            raise ValueError("Invalid native simulator RGB/state")
        metrics = {} if self.goal is None else self._metrics(state)
        return Observation(rgb.copy(), state, metrics, bool(terminated), bool(truncated))

    def _metrics(self, state):
        from shapely.geometry import Polygon
        from shapely.ops import unary_union
        import pymunk
        native = self.env.unwrapped
        shape = self.shift.get("geometry", {}).get("block_shape", "T")
        period = {"square": np.pi/2, "+": np.pi/2, "I": np.pi, "Z": np.pi}.get(shape, 2*np.pi)
        difference = abs(state[4]-self.goal[4]) % period
        angle = float(min(difference, period-difference))
        position = float(np.linalg.norm(state[:4]-self.goal[:4]))
        body = native.block
        target = pymunk.Body(body.mass, body.moment)
        target.center_of_gravity = body.center_of_gravity
        target.angle, target.position = float(self.goal[4]), self.goal[2:4].tolist()
        def polygon(pose):
            return unary_union([Polygon([tuple(pose.local_to_world(v)) for v in item.get_vertices()])
                                for item in body.shapes])
        actual, desired = polygon(body), polygon(target)
        return {"position_error": position, "block_position_error": float(np.linalg.norm(state[2:4]-self.goal[2:4])),
                "angle_error": angle, "success": bool(position < 20 and angle < np.pi/9),
                "coverage": float(np.clip(actual.intersection(desired).area/desired.area, 0, 1))}

    def step(self, action):
        action = np.asarray(action, dtype=np.float32)
        if action.shape != (2,) or not np.isfinite(action).all() or (np.abs(action) > 1+1e-6).any():
            raise ValueError("Fresh planned/collected controls must be finite relative actions in [-1,1]^2")
        if getattr(self, "_collection_contacts", None) is not None:
            self._collection_contacts = 0
        obs, _, terminated, truncated, _ = self.env.step(np.clip(action, -1, 1))
        # The collection has no task goal; ignore the native random goal's termination.
        return self._observe(obs["state"], terminated if self.goal is not None else False, truncated)

    def collection_action(self, observation, rng, step, settings):
        """A declared contact-biased exploratory collector, not a human/expert policy."""
        if settings["name"] == "bounded-contact":
            if self._collection_policy is None:
                raise ValueError("Call begin_collection after resetting a bounded collector")
            return self._collection_policy.action(observation, rng)
        if settings["name"] == "random":
            if step % 5 == 0:
                self._random_action = rng.uniform(-.5, .5, size=2)
            return np.asarray(self._random_action, dtype=np.float32)
        agent, block = observation.state[:2], observation.state[2:4]
        if step % 30 == 0 or not hasattr(self, "_push_direction"):
            angle = rng.uniform(0, 2*np.pi)
            self._push_direction = np.array([np.cos(angle), np.sin(angle)])
        # Approach the object from behind, then continue through it. A physical
        # state policy is used for collection only; these labels never enter WM inputs.
        behind = block-self._push_direction*65
        target = behind if np.linalg.norm(agent-behind) > 30 and step % 30 < 20 else block+self._push_direction*70
        target = np.clip(target, 45, 465)
        command = (target-agent)/100 + rng.normal(0, .025, size=2)
        return np.clip(command, -.8, .8).astype(np.float32)

    def close(self):
        self.env.close()

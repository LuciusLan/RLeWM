"""Native qpos-match Reacher with declared scripted or optional SAC behavior."""

from pathlib import Path
import numpy as np

from ...runtime import file_hash
from .fresh_native import NativeCollection, validate_policy


class FreshReacher(NativeCollection):
    backend, gym_id = "reacher", "swm/ReacherDMControl-v0"
    packages = ("stable-worldmodel", "gymnasium", "dm-control", "mujoco")
    settings = {"task": "qpos_match"}

    @staticmethod
    def validate_rendering(settings):
        from .reacher_rendering import rendering_settings
        return rendering_settings(settings)

    @staticmethod
    def collection_schema():
        return {"state_width": 4, "action_width": 2, "proprio_indices": [0, 1],
                "fields": {"qpos": [2], "qvel": [2], "observation": [6],
                           "finger_pos": [2], "target_pos": [2], "physics_time": [], "control": [2]}}

    @staticmethod
    def reference_state(source):
        return np.concatenate([source["qpos"][:], source["qvel"][:]], axis=1)

    @staticmethod
    def goal_contract():
        return {"reset": "set_state(qpos, qvel)", "goal": "set_target_qpos(goal_qpos)",
                "success": "all(abs(qpos - goal_qpos) < 0.05), native unwrapped joint errors",
                "goal_fields": ["qpos"]}

    @staticmethod
    def goal_diagnostics(state, goal, shift):
        errors = np.abs(state[:2]-goal[:2])
        return {"initial_success": bool((errors < 0.05).all()), "max_joint_error_radians": float(errors.max())}

    @staticmethod
    def validate_collection_policy(config):
        settings = config["policy"]
        if settings.get("name") == "sac":
            validate_policy(settings, "sac", ("checkpoint", "normalizer", "noise_std"), nonnegative=("noise_std",))
            for key in ("checkpoint", "normalizer"):
                if not Path(settings[key]).is_absolute() or not Path(settings[key]).is_file():
                    raise ValueError(f"SAC {key} must be an existing absolute local path")
        else:
            validate_policy(settings, "joint-waypoints", ("kp", "kd", "action_noise", "waypoint_steps"),
                            positive=("kp", "waypoint_steps"), nonnegative=("kd", "action_noise"))
            if type(settings["waypoint_steps"]) is not int:
                raise ValueError("waypoint_steps must be an integer")

    def __init__(self, image_size, shift, *, rendering=None):
        super().__init__(image_size, shift)
        from .reacher_rendering import configure_floor, rendering_settings
        self.rendering = rendering_settings(rendering)
        import gymnasium as gym
        import stable_worldmodel  # noqa: F401
        self.env = gym.make(self.gym_id, render_mode="rgb_array", **self.settings)
        self.native = self.env.unwrapped
        configure_floor(self.native, self.rendering)
        self.control_hz = 1/(self.native.dmc_env.control_timestep()*self.native.action_repeat)

    def behavior_classes(self):
        from stable_worldmodel.envs.dmcontrol.custom_tasks.reacher import ReacherQPosMatchTask
        from stable_worldmodel.envs.dmcontrol.dmcontrol import DMControlWrapper
        from stable_worldmodel.envs.dmcontrol.expert_policy import ExpertPolicy
        return [ReacherQPosMatchTask, DMControlWrapper, ExpertPolicy]

    def description(self):
        value = {**super().description(), "native_action_repeat": self.native.action_repeat,
                "behavior_distribution": "joint-waypoints is a declared alternative to the paper's SAC reference behavior; optional user-supplied SAC weights supported"}
        if self.rendering["floor"] != "native":
            value["rendering"] = self.rendering
        return value

    def capture(self, obs, info, terminated=False, truncated=False):
        self.obs = np.asarray(obs).copy()
        physics = self.native.dmc_env.physics
        fields = {key: info[key] for key in ("qpos", "qvel", "finger_pos", "target_pos")}
        fields.update(observation=obs, physics_time=physics.data.time, control=physics.data.ctrl)
        return self.observation(self.native.render(width=self.image_size, height=self.image_size),
                                np.r_[info["qpos"], info["qvel"]], fields, terminated, truncated)

    def reset(self, seed):
        from .reacher_rendering import reset_options
        self.policy = None
        self.native.dmc_env.task.target_qpos = None
        options = reset_options(self.rendering)
        obs, info = self.env.reset(seed=seed, **({"options": options} if options is not None else {}))
        return self.capture(obs, info)

    def collection_action(self, observation, rng, step, settings):
        if settings["name"] == "sac":
            if self.policy is None:
                from stable_worldmodel.envs.dmcontrol.expert_policy import ExpertPolicy
                self.policy = ExpertPolicy(settings["checkpoint"], settings["normalizer"],
                                           noise_std=settings["noise_std"], seed=int(rng.integers(2**32)), device="cpu")
                self.assets = {key: {"path": settings[key], "sha256": file_hash(Path(settings[key]))}
                               for key in ("checkpoint", "normalizer")}
            return self.action(self.policy.get_action({"observation": self.obs}).reshape(-1))
        if step % settings["waypoint_steps"] == 0:
            physics = self.native.dmc_env.physics
            self.waypoint = rng.uniform(-np.pi, np.pi, size=2)
            self.limited = np.asarray(physics.model.jnt_limited, dtype=bool).copy()
            for joint in np.flatnonzero(self.limited):
                self.waypoint[joint] = rng.uniform(*physics.model.jnt_range[joint])
        error = (self.waypoint-observation.state[:2]+np.pi) % (2*np.pi)-np.pi
        error[self.limited] = (self.waypoint-observation.state[:2])[self.limited]
        raw = settings["kp"]*error-settings["kd"]*observation.state[2:]+rng.normal(0, settings["action_noise"], 2)
        return self.action(np.clip(raw, -1, 1))

    def step(self, action):
        obs, _, terminated, truncated, info = self.env.step(self.action(action))
        return self.capture(obs, info, terminated, truncated)

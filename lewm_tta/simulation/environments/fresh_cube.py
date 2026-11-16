"""Single-cube collection with explicit native OGBench behavior recipes."""

from importlib.metadata import version
import numpy as np

from .fresh_native import NativeCollection, numpy_stream, validate_policy


class FreshCube(NativeCollection):
    backend, gym_id = "cube", "swm/OGBCube-v0"
    packages = ("stable-worldmodel", "gymnasium", "ogbench", "mujoco")
    control_hz = 20
    settings = {"env_type": "single", "ob_type": "states", "multiview": False,
                "width": 224, "height": 224, "mode": "data_collection", "terminate_at_goal": False,
                "visualize_info": False}

    @staticmethod
    def validate_collection_config(config):
        NativeCollection.validate_collection_config(config)
        expected = config.get("expected_physics")
        if expected is not None and (not isinstance(expected, dict)
                or set(expected) != {"mujoco", "dm-control"}
                or any(not isinstance(value, str) or not value.strip() for value in expected.values())):
            raise ValueError("Cube expected_physics must declare mujoco and dm-control versions")
        if (config["policy"].get("name") == "native-plan-oracle"
                and expected != {"mujoco": "3.6.0", "dm-control": "1.0.38"}):
            raise ValueError("The calibrated Cube plan recipe requires mujoco==3.6.0 and dm-control==1.0.38")

    @staticmethod
    def collection_options(config):
        return {"policy": config["policy"], "expected_physics": config.get("expected_physics")}

    @staticmethod
    def collection_schema():
        return {"state_width": 41, "action_width": 5, "proprio_indices": list(range(6)),
                "fields": {"qpos": [21], "qvel": [20], "observation": [28],
                           "privileged_block_0_pos": [3], "privileged_block_0_quat": [4], "control": [7],
                           "mocap_pos": [1, 3], "mocap_quat": [1, 4], "physics_time": []}}

    @staticmethod
    def reference_state(source):
        return np.concatenate([source["qpos"][:], source["qvel"][:]], axis=1)

    @staticmethod
    def goal_contract():
        return {"reset": "set_state(qpos, qvel); seed and replay prefix for controller state",
                "goal": "set_target_pos(0, goal_privileged_block_0_pos, goal_privileged_block_0_quat)",
                "success": "norm(block_pos - goal_block_pos) <= 0.04 metres; orientation not thresholded by native task",
                "goal_fields": ["privileged_block_0_pos", "privileged_block_0_quat"]}

    @staticmethod
    def goal_diagnostics_rows(stream, current, goal):
        distance = float(np.linalg.norm(stream["privileged_block_0_pos"][current]-stream["privileged_block_0_pos"][goal]))
        return {"initial_success": distance <= 0.04, "block_position_error_metres": distance}

    @staticmethod
    def validate_collection_policy(config):
        if config["policy"].get("name") == "native-plan-oracle":
            validate_policy(config["policy"], "native-plan-oracle", ("noise", "noise_smoothing", "segment_dt"),
                            positive=("noise_smoothing", "segment_dt"), nonnegative=("noise",))
        else:
            validate_policy(config["policy"], "native-markov-oracle", ("action_noise", "min_norm"),
                            positive=("min_norm",), nonnegative=("action_noise",))

    def __init__(self, image_size, shift, *, policy=None, expected_physics=None):
        super().__init__(image_size, shift)
        self.policy_settings = dict(policy) if policy is not None else {
            "name": "native-markov-oracle", "action_noise": 0.1, "min_norm": 0.4}
        self.validate_collection_policy({"policy": self.policy_settings})
        self.expected_physics = dict(expected_physics or {})
        # Fail before constructing a simulator or committing an episode.
        for package, required in self.expected_physics.items():
            installed = version(package)
            if installed != required:
                raise ValueError(f"Cube collection requires {package}=={required}; found {installed}. "
                                 "Use .venvs/physics-mj36/bin/python")
        if self.policy_settings["name"] == "native-plan-oracle":
            self.packages = (*self.packages, "dm-control", "scipy")
        import gymnasium as gym
        import stable_worldmodel  # noqa: F401
        self.env = gym.make(self.gym_id, **self.settings)
        self.native = self.env.unwrapped

    def behavior_classes(self):
        if self.policy_settings["name"] == "native-plan-oracle":
            from ogbench.manipspace.oracles.plan.cube_plan import CubePlanOracle
            from ogbench.manipspace.oracles.plan.plan_oracle import PlanOracle
            return [CubePlanOracle, PlanOracle]
        from ogbench.manipspace.oracles.markov.cube_markov import CubeMarkovOracle
        from ogbench.manipspace.oracles.markov.markov_oracle import MarkovOracle
        return [CubeMarkovOracle, MarkovOracle]

    def description(self):
        result = super().description()
        if self.policy_settings["name"] == "native-plan-oracle":
            result.update(collection_policy=self.policy_settings, expected_physics=self.expected_physics,
                          plan_recipe={"noise_domain": "temporally smoothed interpolated plan",
                                       "extra_action_noise": False,
                                       "retarget": "before next control when native oracle.done; p_stack=0",
                                       "filtering": "none; retain every reserved episode"})
        return result

    def sample_variations(self, rng, names):
        return {f"cube.{name}": rng.uniform(self.native.variation_space["cube"][name].low,
                                            self.native.variation_space["cube"][name].high) for name in names}

    def reset(self, seed):
        self.policy = None
        self.assets = {}
        # Installed CubeEnv.reset drops seed before reaching Gym's reset. Sample
        # every default random variation explicitly and seed its physical RNG.
        self.native.np_random = np.random.default_rng(seed)
        values = self.sample_variations(np.random.default_rng(seed), ("start_position", "start_yaw", "goal_position", "goal_yaw"))
        if self.policy_settings["name"] == "native-plan-oracle":
            # Match the calibrated reset contract without consuming process-global RNG.
            with numpy_stream(np.random.RandomState(seed)):
                obs, self.info = self.env.reset(seed=seed, options={"variation": [], "variation_values": values})
            self.assets = {"reset_seed": int(seed), "reset_variations": {
                key: np.asarray(value).tolist() for key, value in values.items()}, "target_events": []}
        else:
            obs, self.info = self.env.reset(seed=seed, options={"variation": [], "variation_values": values})
        return self.capture(obs, self.info)

    def capture(self, obs, info, terminated=False, truncated=False):
        data = self.native._data
        fields = {"qpos": info["qpos"], "qvel": info["qvel"], "observation": obs,
                  "privileged_block_0_pos": info["privileged/block_0_pos"], "privileged_block_0_quat": info["privileged/block_0_quat"],
                  "control": data.ctrl, "mocap_pos": data.mocap_pos, "mocap_quat": data.mocap_quat,
                  "physics_time": data.time}
        return self.observation(self.native.render(), np.r_[info["qpos"], info["qvel"]], fields, terminated, truncated)

    def collection_action(self, observation, rng, step, settings):
        if settings["name"] == "native-plan-oracle":
            if settings != self.policy_settings:
                raise ValueError("Cube plan policy differs from its recorded environment configuration")
            return self.plan_action(rng, step, settings)
        return self.markov_action(rng, settings)

    def plan_action(self, rng, step, settings):
        """Native interpolated plans; never add Markov's per-control noise."""
        if self.policy is None:
            oracle_seed = int(rng.integers(2**32))
            self.stream = np.random.RandomState(oracle_seed)
            options = {key: value for key, value in settings.items() if key != "name"}
            self.policy = self.behavior_classes()[0](env=self.native, **options)
            self.assets.update(oracle_seed=oracle_seed, extra_action_noise=False)
            self.reset_plan(step)
        elif self.policy.done:
            # Only the hidden behavior target changes; physical state is not reset.
            values = self.sample_variations(rng, ("goal_position", "goal_yaw"))
            self.native.variation_space.set_value(values)
            _, self.info = self.native.set_new_target(p_stack=0)
            self.reset_plan(step)
        with numpy_stream(self.stream):
            action = np.asarray(self.policy.select_action(None, self.info), dtype=np.float64)
        if action.shape != (5,) or not np.isfinite(action).all():
            raise ValueError("Cube plan oracle returned invalid controls")
        return self.action(np.clip(action, -1, 1))

    def reset_plan(self, step):
        with numpy_stream(self.stream):
            self.policy.reset(None, self.info)
        events = self.assets["target_events"]
        events.append({"raw_step": int(step), "target_sequence_index": len(events),
                       "position": np.asarray(self.info["privileged/target_block_pos"]).tolist(),
                       "yaw": np.asarray(self.info["privileged/target_block_yaw"]).tolist()})

    def markov_action(self, rng, settings):
        """Preserve the existing Markov recipe and its original RNG draw order."""
        if self.policy is None:
            self.stream = np.random.RandomState(int(rng.integers(2**32)))
            self.policy = self.behavior_classes()[0](env=self.native, min_norm=settings["min_norm"])
            self.noise = rng.uniform(0, settings["action_noise"])*np.array([1, 1, 1, 3, 10])
            with numpy_stream(self.stream):
                self.policy.reset(None, self.info)
        elif self.policy.done:
            values = self.sample_variations(rng, ("goal_position", "goal_yaw"))
            self.native.variation_space.set_value(values)
            _, self.info = self.native.set_new_target(p_stack=0)
            with numpy_stream(self.stream):
                self.policy.reset(None, self.info)
        with numpy_stream(self.stream):
            action = self.policy.select_action(None, self.info)
        return self.action(np.clip(action+rng.normal(size=5)*self.noise, -1, 1))

    def step(self, action):
        obs, _, terminated, truncated, self.info = self.env.step(self.action(action))
        return self.capture(obs, self.info, terminated, truncated)

    def close(self):
        renderer = getattr(self.native, "_renderer", None)
        if renderer is not None:
            renderer.close()
            self.native._renderer = None
        self.env.close()

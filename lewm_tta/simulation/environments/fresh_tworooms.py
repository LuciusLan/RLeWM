"""Fresh native TwoRoom rollouts from its released doorway-following expert."""

import numpy as np

from .fresh_native import NativeCollection, validate_policy


class FreshTwoRooms(NativeCollection):
    backend, gym_id = "tworooms", "swm/TwoRoom-v1"
    packages = ("stable-worldmodel", "gymnasium")
    settings = {"render_target": False}
    control_hz = 10

    @staticmethod
    def collection_schema():
        return {"state_width": 2, "action_width": 2, "proprio_indices": [0, 1],
                "fields": {"observation": [10], "target_pos": [2]}}

    @staticmethod
    def reference_state(source):
        return source["proprio"][:]

    @staticmethod
    def goal_contract():
        return {"reset": "_set_state(proprio)", "goal": "_set_goal_state(goal_proprio)",
                "success": "agent-to-observed-goal distance < 16 native scene pixels", "goal_fields": ["proprio"]}

    @staticmethod
    def goal_diagnostics(state, goal, shift):
        distance = float(np.linalg.norm(state-goal))
        return {"initial_success": distance < 16, "position_error_native_pixels": distance}

    @staticmethod
    def validate_collection_policy(config):
        validate_policy(config["policy"], "native-doorway", ("action_noise",), nonnegative=("action_noise",))

    def __init__(self, image_size, shift):
        super().__init__(image_size, shift)
        import gymnasium as gym
        import stable_worldmodel  # noqa: F401 -- gym registry
        self.env = gym.make(self.gym_id, render_mode="rgb_array", **self.settings)
        self.native = self.env.unwrapped

    def behavior_classes(self):
        from stable_worldmodel.envs.two_room.expert_policy import ExpertPolicy
        return [ExpertPolicy]

    def capture(self, obs, info, terminated=False, truncated=False):
        return self.observation(self.native.render(), info["state"],
                                {"observation": obs, "target_pos": info["goal_state"]}, terminated, truncated)

    def reset(self, seed):
        self.policy = None
        obs, self.info = self.env.reset(seed=seed)
        return self.capture(obs, self.info)

    def collection_action(self, observation, rng, step, settings):
        if self.policy is None:
            cls = self.behavior_classes()[0]
            self.policy = cls(action_noise=settings["action_noise"], seed=int(rng.integers(2**32)))
            self.policy.set_env(self.env)
        # Continue physical trajectories after a waypoint succeeds. Only the
        # hidden behavior target changes; future image goals are actual frames.
        if step == 0 or np.linalg.norm(np.asarray(self.info["state"])-np.asarray(self.info["goal_state"])) < 16:
            axis = 0 if self.native.wall_axis == 1 else 1
            space = self.native.variation_space["agent"]["position"]
            for _ in range(1000):
                target = rng.uniform(space.low, space.high).astype(np.float32)
                if ((target[axis] > self.native.wall_pos) != (self.info["state"][axis] > self.native.wall_pos)
                        and self.native._constrain_agent_not_in_wall(target)):
                    self.native._set_goal_state(target)
                    self.info = dict(self.info, goal_state=target)
                    break
            else:
                raise ValueError("Could not sample a valid opposite-room waypoint")
        return self.action(self.policy.get_action(self.info))

    def step(self, action):
        obs, _, terminated, truncated, self.info = self.env.step(self.action(action))
        return self.capture(obs, self.info, terminated, truncated)

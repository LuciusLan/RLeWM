"""Reconstruct a fresh case from its original reset and executed past prefix.

Replaying the entire prefix restores unrecorded block velocities/contact state;
it avoids resetting midway from the incomplete seven-coordinate source state.
"""

import numpy as np

from .fresh_pusht import FreshPushT


class FreshRecordedPushT(FreshPushT):
    def __init__(self, settings, dataset):
        super().__init__(settings["image_size"], dataset.fresh_manifest["shift"])
        self.dataset = dataset

    def reset(self, case, seed, *, history_start=False):
        # Simulator reset seed belongs to the collected case. The requested
        # evaluation seed changes CEM/TTT randomness, not the physical problem.
        observation = super().reset(case["episode_reset_seed"])
        self.set_goal(self.dataset.rows("state64", case["goal_raw_row"]))
        end = case["start_raw_row"] if history_start else case["current_raw_row"]
        for action in self.dataset.rows("action", case["episode_start_raw_row"], end):
            observation = super().step(action)
        return self._observe(observation.state)

    def step(self, raw_action, *, recorded=False):
        return super().step(raw_action)

    def reconstruction(self, observation, row):
        state = self.dataset.rows("state64", row)
        difference = abs(observation.state[4]-state[4]) % (2*np.pi)
        return {"position_error": float(np.linalg.norm(observation.state[:4]-state[:4])),
                "angle_error": float(min(difference, 2*np.pi-difference)),
                "rgb_mae_0_255": float(np.abs(observation.rgb.astype(float)-self.dataset.rows("pixels", row)).mean())}

    def description(self):
        return {**super().description(),
                "reset_protocol": "original fresh reset seed followed by the complete recorded past prefix; no future actions",
                "regime": self.dataset.fresh_manifest["regime"],
                "collection_manifest": self.dataset.fresh_manifest["collection_manifest"]}

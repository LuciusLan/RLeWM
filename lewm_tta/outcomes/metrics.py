"""Physical goal criteria, pinned independently of learned latent distances."""

import numpy as np


def goal_metrics(backend, observation, goal, *, pusht_success="native-goal", coverage_threshold=.95, coverage=None):
    state = np.asarray(observation.state, dtype=np.float64)
    target = np.asarray(goal["state64"], dtype=np.float64)
    if not np.isfinite(state).all() or not np.isfinite(target).all():
        raise ValueError("Physical goal states must be finite")
    if backend == "tworooms":
        distance = float(np.linalg.norm(state-target))
        return {"success": distance < 16., "goal_distance": distance, "planning_score": -distance}
    if backend == "reacher":
        errors = np.abs(state[:2]-np.asarray(goal["qpos"]))
        # Match native qpos_match; do not silently wrap joint differences.
        distance = float(errors.max())
        return {"success": bool((errors < .05).all()), "goal_distance": distance,
                "joint_mse": float(np.mean(errors**2)), "planning_score": -distance}
    if backend == "cube":
        position = np.asarray(observation.fields["privileged_block_0_pos"])
        target_position = np.asarray(goal["privileged_block_0_pos"])
        distance = float(np.linalg.norm(position-target_position))
        quaternion = np.asarray(observation.fields["privileged_block_0_quat"], dtype=float)
        target_quaternion = np.asarray(goal["privileged_block_0_quat"], dtype=float)
        denominator = np.linalg.norm(quaternion)*np.linalg.norm(target_quaternion)
        if not np.isfinite(denominator) or denominator <= 0:
            raise ValueError("Cube orientations must be finite nonzero quaternions")
        cosine = np.clip(abs(np.dot(quaternion, target_quaternion))/denominator, 0., 1.)
        return {"success": distance <= .04, "goal_distance": distance, "planning_score": -distance,
                "orientation_error_radians": float(2*np.arccos(cosine))}
    if backend == "pusht":
        if pusht_success not in ("native-goal", "coverage") or not 0 < coverage_threshold <= 1:
            raise ValueError("Choose native-goal or coverage and a coverage threshold in (0,1]")
        if coverage is None or not np.isfinite(coverage) or not 0 <= coverage <= 1:
            raise ValueError("PushT needs geometric coverage measured against the supplied block goal")
        angle = float(abs(state[4]-target[4]) % (2*np.pi))
        angle = min(angle, 2*np.pi-angle)
        position = float(np.linalg.norm(state[:4]-target[:4]))
        native = bool(position < 20. and angle < np.pi/9)
        covered = bool(coverage >= coverage_threshold)
        return {"success": native if pusht_success == "native-goal" else covered,
                "native_goal_success": native, "coverage_success": covered, "coverage": float(coverage),
                "goal_distance": float(np.linalg.norm(state[2:4]-target[2:4])),
                "position_error_agent_and_block": position, "angle_error_radians": angle,
                "planning_score": float(coverage)}
    raise ValueError(f"No physical goal metrics for {backend!r}")


def metric_contract(backend, pusht_success, coverage_threshold):
    contracts = {
        "tworooms": {"success": "Euclidean agent-to-goal position distance < 16 native pixels",
                     "goal_distance_units": "native scene pixels", "planning_score": "negative agent-to-goal distance"},
        "reacher": {"success": "all absolute unwrapped qpos errors < 0.05 radians",
                    "goal_distance_units": "maximum absolute joint error, radians", "planning_score": "negative maximum joint error"},
        "cube": {"success": "Euclidean block-to-goal position distance <= 0.04 metres; orientation unconstrained",
                 "goal_distance_units": "metres", "planning_score": "negative block-to-goal distance"},
        "pusht": {"success": ("combined agent/block position error < 20 native pixels and block angle error < pi/9"
                               if pusht_success == "native-goal" else f"block/goal overlap coverage >= {coverage_threshold}"),
                  "goal_distance_units": "block position distance, native pixels", "planning_score": "geometric block/goal coverage",
                  "additional_success": "both native image-goal success and coverage success are retained",
                  "coverage_threshold": coverage_threshold}}
    return {"backend": backend, **contracts[backend],
            "goal": "same observed future image/physical state supplied to the source planner",
            "primary_endpoint": "final success after exactly H*frameskip controls",
            "secondary_endpoint": "ever success, including the initial state, reported separately",
            "pooling": "distance and planning-score scales are task-specific; do not pool across tasks"}


class PlanOutcome:
    def __init__(self, metrics):
        self.initial, self.final = dict(metrics), dict(metrics)
        self.best_score = float(metrics["planning_score"])
        self.best_distance = float(metrics["goal_distance"])
        self.ever_success = bool(metrics["success"])
        self.first_success_raw_step = 0 if self.ever_success else None
        self.steps = 0

    def append(self, metrics):
        self.steps += 1
        self.final = dict(metrics)
        self.best_score = max(self.best_score, float(metrics["planning_score"]))
        self.best_distance = min(self.best_distance, float(metrics["goal_distance"]))
        self.ever_success |= bool(metrics["success"])
        if metrics["success"] and self.first_success_raw_step is None:
            self.first_success_raw_step = self.steps

    def record(self):
        return {"initial_metrics": self.initial, "final_metrics": self.final,
                "initial_success": bool(self.initial["success"]), "final_success": bool(self.final["success"]),
                "ever_success": self.ever_success, "first_success_raw_step": self.first_success_raw_step,
                "raw_steps": self.steps, "best_planning_score": self.best_score, "best_goal_distance": self.best_distance,
                "planning_score_improvement": float(self.final["planning_score"]-self.initial["planning_score"]),
                "goal_distance_reduction": float(self.initial["goal_distance"]-self.final["goal_distance"])}

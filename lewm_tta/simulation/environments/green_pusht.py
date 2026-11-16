"""Native green-marker geometry, RGB goal and block-only PushT scoring."""

import hashlib

import numpy as np

from .base import Observation
from .fresh_pusht import FreshPushT


def polygon(body, shapes):
    from shapely.geometry import Polygon
    from shapely.ops import unary_union
    return unary_union([Polygon([tuple(body.local_to_world(v)) for v in shape.get_vertices()]) for shape in shapes])


def align_to_marker(body, marker):
    """Match the marker transform, including nonzero block center of gravity."""
    body.angle = marker.angle
    origin = np.asarray(body.local_to_world((0, 0)))
    desired = np.asarray(marker.local_to_world((0, 0)))
    body.position = (np.asarray(body.position)+desired-origin).tolist()


class GreenTargetPushT(FreshPushT):
    def __init__(self, settings, dataset=None):
        self.settings = settings
        super().__init__(settings["image_size"], settings["shift"])
        # Goal rendering never mutates the physical evaluation environment.
        self.goal_renderer = FreshPushT(settings["image_size"], settings["shift"])
        self.target_polygon = None

    def reset(self, case, seed, *, history_start=False):
        observation = super().reset(case["episode_reset_seed"])
        native = self.env.unwrapped
        marker = native._get_goal_pose_body(native.goal_pose)
        self.target_polygon = polygon(marker, native.block.shapes)
        self.goal_renderer.reset(case["episode_reset_seed"])
        renderer = self.goal_renderer.env.unwrapped
        if not np.array_equal(renderer.goal_pose, native.goal_pose):
            raise ValueError("Goal-renderer marker differs from the physical task")
        align_to_marker(renderer.block, renderer._get_goal_pose_body(renderer.goal_pose))
        renderer.block.velocity, renderer.block.angular_velocity = (0., 0.), 0.
        renderer.agent.position = self.settings["goal_agent_position"]
        renderer.agent.velocity = (0., 0.)
        renderer.space.reindex_shapes_for_body(renderer.block)
        renderer.space.reindex_shapes_for_body(renderer.agent)
        self.goal = np.asarray(renderer._get_obs(), dtype=np.float64)
        self.goal_image = self.goal_renderer.env.render().copy()
        desired = polygon(renderer.block, renderer.block.shapes)
        mismatch = float(desired.symmetric_difference(self.target_polygon).area)
        pusher = next(iter(renderer.agent.shapes))
        from shapely.geometry import Point
        pusher_center = renderer.agent.local_to_world(pusher.offset)
        if desired.intersects(Point(tuple(pusher_center)).buffer(pusher.radius)):
            raise ValueError("Canonical goal pusher overlaps the target block; choose another goal_agent_position")
        self.goal_contract = {"marker_pose": native.goal_pose.tolist(), "goal_state": self.goal.tolist(),
            "goal_image_sha256": hashlib.sha256(self.goal_image.tobytes()).hexdigest(),
            "marker_symmetric_difference_area": mismatch,
            "rendered_goal_coverage": float(desired.intersection(self.target_polygon).area/self.target_polygon.area),
            "canonical_pusher_position": self.settings["goal_agent_position"],
            "physical_pusher_constraint": "none; success depends only on block coverage"}
        if mismatch > 1e-7 or self.goal_image.shape != (self.image_size, self.image_size, 3):
            raise ValueError("Rendered goal block does not align with the actual green marker")
        observation = self._observe(observation.state)
        if not history_start:
            for action in case["warmup_raw_actions"]:
                observation = self.step(action, recorded=True)
        return observation

    def _metrics(self, state):
        from .pusht_collection import body_bounds
        from ...collection.quality import geometry_visible
        native = self.env.unwrapped
        actual = polygon(native.block, native.block.shapes)
        coverage = float(np.clip(actual.intersection(self.target_polygon).area/self.target_polygon.area, 0, 1))
        metric = self.goal_diagnostics(state, self.goal, self.shift)
        return {"coverage": coverage, "success": coverage >= self.settings["success_coverage"],
                "block_position_error": metric["block_distance"], "angle_error": metric["angle_distance"],
                "out_of_frame": not geometry_visible(body_bounds(native), native.window_size, 0)}

    def step(self, action, *, recorded=False):
        observation = super().step(action)
        # Stock termination tests a separate random state and a pusher position.
        # It is deliberately replaced by the declared marker-overlap criterion.
        return Observation(observation.rgb, observation.state, observation.metrics,
                           bool(observation.metrics["success"]), observation.truncated)

    def reconstruction(self, observation, row):
        return {}  # Native seeded resets; no recorded state to reconstruct.

    def description(self):
        return {**super().description(), "task": "green-target",
                "metric_directions": {"coverage": 1, "success": 1, "block_position_error": -1,
                                      "angle_error": -1, "out_of_frame": -1},
                "goal_convention": "RGB target uses the exact native green-marker transform and a canonical pusher",
                "goal_agent_position": self.settings["goal_agent_position"],
                "success_coverage": self.settings["success_coverage"],
                "coverage_definition": "intersection(block, rendered marker) / area(rendered marker)",
                "success": "block coverage only; pusher pose unconstrained; stock state-goal termination ignored",
                "reset_protocol": "prespecified native seed and common zero-action history; no trajectory data or outcome filtering"}

    def close(self):
        self.goal_renderer.close()
        super().close()

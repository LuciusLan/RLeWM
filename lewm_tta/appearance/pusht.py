"""Render recorded PushT poses using native geometry/drawing, without stepping."""

from pathlib import Path

import numpy as np

from ..runtime import file_hash
from .settings import background, pattern_rgb


class PushTRenderer:
    def __init__(self, size, settings):
        from stable_worldmodel.envs.pusht import env as module
        self.module, self.settings, self.size = module, settings, size
        self.native = module.PushT(resolution=size, with_target=True, relative=True)
        values = {"block.shape": self.native.shapes.index("T")}
        for option, name in (("object_color", "block"), ("agent_color", "agent"), ("goal_color", "goal")):
            if settings[option] is not None:
                values[f"{name}.color"] = np.asarray(settings[option], dtype=np.uint8)
        # Initial reset constructs native geometry. No recorded action is replayed.
        self.native.reset(seed=0, options={"variation_values": values})
        primary, secondary = background(settings, [255, 255, 255], [220, 220, 220])
        pattern = settings["background_pattern"]
        self.floor = pattern_rgb(self.native.window_size, "solid" if pattern == "preserve" else pattern,
                                 primary, secondary, settings["pattern_size"])
        self.description = {"environment": "swm/PushT-v1", "native_source_sha256": file_hash(Path(module.__file__)),
                            "scene_size": self.native.window_size, "image_size": size,
                            "pose_restore": "direct body transforms, native _set_state is NOT called",
                            "goal_marker": "native fixed decorative marker, separate from observed future RGB goal"}

    def render(self, source, row):
        import cv2
        import pygame
        import pymunk
        native = self.native
        state = np.asarray(source["state64" if "state64" in source else "state"][row], dtype=np.float64)
        if state.shape != (7,) or not np.isfinite(state).all():
            raise ValueError("PushT re-rendering requires finite recorded seven-coordinate states")
        # Assignment order follows native _set_state, including the T centre of gravity.
        native.agent.position = state[:2].tolist()
        native.agent.velocity = state[5:7].tolist()
        native.block.angle = float(state[4])
        native.block.position = state[2:4].tolist()
        native.block.velocity = (0, 0)
        native.block.angular_velocity = 0
        native.space.reindex_shapes_for_body(native.agent)
        native.space.reindex_shapes_for_body(native.block)
        difference = native._get_obs()-state
        difference[4] = (difference[4]+np.pi) % (2*np.pi)-np.pi
        if np.abs(difference).max() > 1e-10:
            raise ValueError("PushT pose changed while restoring a recorded frame")
        canvas = pygame.surfarray.make_surface(self.floor.transpose(1, 0, 2).copy())
        draw = self.module.DrawOptions(canvas)
        # Same layering and DrawOptions as the installed native RGB renderer.
        if native.with_target and native.variation_space["rendering"]["render_goal"].value:
            goal_body = native._get_goal_pose_body(native.goal_pose)
            for shape in native.block.shapes:
                if isinstance(shape, pymunk.Circle):
                    center = pymunk.pygame_util.to_pygame(goal_body.local_to_world(shape.offset), canvas)
                    pygame.draw.circle(canvas, native.variation_space["goal"]["color"].value,
                                       tuple(map(int, center)), int(shape.radius))
                else:
                    points = [pymunk.pygame_util.to_pygame(goal_body.local_to_world(v), canvas)
                              for v in shape.get_vertices()]
                    pygame.draw.polygon(canvas, native.variation_space["goal"]["color"].value, points + points[:1])
        for name in ("agent", "block"):
            native._set_body_color(getattr(native, name), native.variation_space[name]["color"].value.tolist())
        native.space.debug_draw(draw)
        pixels = np.asarray(pygame.surfarray.array3d(canvas)).transpose(1, 0, 2)
        return cv2.resize(pixels, (self.size, self.size))

    def close(self):
        self.native.close()

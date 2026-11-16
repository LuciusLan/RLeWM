"""Reacher appearance changes in MJCF, with recorded qpos/qvel and camera."""

from pathlib import Path
import tempfile

import numpy as np

from ..runtime import file_hash
from ..simulation.environments.reacher_rendering import CHECKERBOARD, configure_floor, reset_options
from .settings import background, pattern_rgb


class ReacherRenderer:
    def __init__(self, size, settings):
        import gymnasium as gym
        import stable_worldmodel  # noqa: F401 -- register native environments
        from stable_worldmodel.envs.dmcontrol import reacher as module
        from PIL import Image
        self.env = gym.make("swm/ReacherDMControl-v0", render_mode="rgb_array", task="qpos_match")
        self.native, self.size = self.env.unwrapped, size
        self.assets = tempfile.TemporaryDirectory(prefix="lewm-appearance-")
        # Reference is the source-compatible checkerboard used by the active fresh bank.
        # Never change the physics, camera, lighting or target visibility.
        baseline = {**CHECKERBOARD}
        primary, secondary = background(settings, [25.5, 51, 76.5], [51, 76.5, 102])
        baseline["rgb1"], baseline["rgb2"] = np.asarray(primary)/255, np.asarray(secondary)/255
        baseline["rgb1"], baseline["rgb2"] = baseline["rgb1"].tolist(), baseline["rgb2"].tolist()
        configure_floor(self.native, baseline)
        options = reset_options(baseline)
        if settings["object_color"] is not None:
            options["variation_values"]["agent.color"] = np.asarray(settings["object_color"], dtype=float)/255
        self.env.reset(seed=0, options=options)
        pattern = settings["background_pattern"]
        if pattern != "preserve":
            # A texture asset changes only the floor. Geometry supplies occlusion,
            # arm shading and shadows, rather than RGB masks or overlays.
            texture_path = Path(self.assets.name)/"floor.png"
            Image.fromarray(pattern_rgb(512, pattern, primary, secondary, settings["pattern_size"])).save(texture_path)
            model = self.native._mjcf_model
            texture, material = model.find("texture", "grid"), model.find("material", "grid")
            texture.builtin, texture.mark = None, None
            texture.file = str(texture_path)
            material.texrepeat, material.texuniform = [1, 1], False
            self.native.mark_dirty()
            self.native.compile_model(seed=0)
        self.description = {"environment": "swm/ReacherDMControl-v0", "task": "qpos_match",
                            "native_source_sha256": file_hash(Path(module.__file__)), "image_size": size,
                            "reference_floor": CHECKERBOARD, "camera_id": self.native.camera_id,
                            "pose_restore": "native set_state(qpos,qvel), forward kinematics only",
                            "pattern_scale": "pattern-size units across the 512-unit texture; native floor UV mapping"}

    def render(self, source, row):
        qpos, qvel = np.asarray(source["qpos"][row]), np.asarray(source["qvel"][row])
        if qpos.shape != (2,) or qvel.shape != (2,) or not np.isfinite([qpos, qvel]).all():
            raise ValueError("Reacher requires finite two-dimensional qpos/qvel")
        self.native.set_state(qpos, qvel)
        physics = self.native.dmc_env.physics
        if "target_pos" in source:
            physics.named.model.geom_pos["target", :2] = source["target_pos"][row]
            physics.forward()
        if not np.array_equal(physics.data.qpos, qpos) or not np.array_equal(physics.data.qvel, qvel):
            raise ValueError("Reacher pose changed while restoring a recorded frame")
        return self.native.render(width=self.size, height=self.size)

    def close(self):
        self.env.close()
        self.assets.cleanup()

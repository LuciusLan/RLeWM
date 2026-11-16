"""Shared collection utilities; task physics and behavior live in task adapters."""

from contextlib import contextmanager
from importlib.metadata import version
from pathlib import Path
import inspect

import numpy as np

from ...runtime import file_hash
from .base import Observation


class NativeCollection:
    @staticmethod
    def validate_collection_config(config):
        if (config["image_size"] != 224 or config["history"] != 3 or config["frameskip"] != 5
                or config["goal_offset_steps"] != 25):
            raise ValueError("Native fresh offline profiles require RGB224, H3, frameskip5 and primary +25-control goals")
        if config.get("quality"):
            raise ValueError("PushT contact/visibility quality rules cannot be used for another task")
        if config["backend"] == "reacher" and config["raw_steps"] > 499:
            raise ValueError("Reacher collection must end before the native 20-second time limit")

    @staticmethod
    def validate_shift(shift):
        if shift:
            raise ValueError("These task adapters currently support source dynamics only; shifts need a task-specific implementation")

    def __init__(self, image_size, shift):
        self.validate_shift(shift)
        self.image_size = image_size
        self.policy = None
        self.assets = {}

    def behavior_assets(self):
        return self.assets

    def close(self):
        self.env.close()

    def action(self, value):
        result = np.asarray(value, dtype=np.float32)
        if (result.shape != self.env.action_space.shape or not np.isfinite(result).all()
                or (np.abs(result) > 1+1e-6).any()):
            raise ValueError("Native policy returned an invalid raw action")
        return result

    def description(self):
        # Capture installed code as well as project code. Never import optional
        # simulators while merely resolving/validating a collection profile.
        modules = [type(self.native)] + self.behavior_classes()
        sources = {str(Path(inspect.getfile(cls)).resolve()): file_hash(Path(inspect.getfile(cls))) for cls in modules}
        return {"backend": self.backend, "gym_id": self.gym_id, "image_size": self.image_size,
                "control_hz": self.control_hz, "settings": self.settings,
                "packages": {name: version(name) for name in self.packages}, "installed_sources": sources,
                "goal_contract": self.goal_contract(),
                "raw_control": "one outer environment.step; no additional frameskip during collection"}

    def observation(self, rgb, state, fields, terminated=False, truncated=False):
        return Observation(np.asarray(rgb, dtype=np.uint8).copy(), np.asarray(state, dtype=np.float64).copy(),
                           {}, bool(terminated), bool(truncated),
                           {name: np.asarray(value, dtype=np.float64).copy() for name, value in fields.items()})


def validate_policy(settings, name, keys, positive=(), nonnegative=()):
    if settings.get("name") != name or set(settings)-({"name"} | set(keys)):
        raise ValueError(f"Expected policy {name!r} with options {list(keys)}")
    if set(keys)-set(settings):
        raise ValueError(f"Missing {name} policy options: {set(keys)-set(settings)}")
    for key in positive+nonnegative:
        value = settings[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value) or value < 0 or (key in positive and value == 0):
            raise ValueError(f"Invalid policy parameter: {key}")


@contextmanager
def numpy_stream(stream):
    """Isolate legacy OGBench oracle calls that consume numpy's global RNG."""
    previous = np.random.get_state()
    np.random.set_state(stream.get_state())
    try:
        yield
    finally:
        stream.set_state(np.random.get_state())
        np.random.set_state(previous)

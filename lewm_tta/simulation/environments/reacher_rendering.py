"""Explicit Reacher floor appearance, isolated from policy and physics settings."""

from copy import deepcopy
import math

import numpy as np


CHECKERBOARD = {"floor": "checkerboard", "rgb1": [0.1, 0.2, 0.3],
                "rgb2": [0.2, 0.3, 0.4], "texrepeat": [1.0, 1.0], "texuniform": False}


def rendering_settings(value=None):
    """Absent settings preserve the historical native-rendering path."""
    if value is None:
        return {"floor": "native"}
    if not isinstance(value, dict) or set(value) - set(CHECKERBOARD):
        raise ValueError(f"Reacher rendering accepts only {list(CHECKERBOARD)}")
    mode = value.get("floor", "native")
    if mode not in ("native", "checkerboard"):
        raise ValueError("Reacher rendering.floor must be native or checkerboard")
    if mode == "native":
        if set(value) - {"floor"}:
            raise ValueError("Floor overrides require rendering.floor=checkerboard")
        return {"floor": "native"}
    settings = {**deepcopy(CHECKERBOARD), **deepcopy(value)}
    for key, width in (("rgb1", 3), ("rgb2", 3), ("texrepeat", 2)):
        numbers = settings[key]
        if (not isinstance(numbers, (list, tuple)) or len(numbers) != width
                or any(isinstance(n, bool) or not isinstance(n, (int, float)) or not math.isfinite(n) for n in numbers)):
            raise ValueError(f"Reacher rendering.{key} requires {width} finite numbers")
        if key.startswith("rgb") and any(n < 0 or n > 1 for n in numbers):
            raise ValueError(f"Reacher rendering.{key} must lie in [0,1]")
        if key == "texrepeat" and any(n <= 0 for n in numbers):
            raise ValueError("Reacher rendering.texrepeat must be positive")
    if type(settings["texuniform"]) is not bool:
        raise ValueError("Reacher rendering.texuniform must be a boolean")
    return settings


def configure_floor(native, settings):
    """Configure MJCF before reset/recompilation; never replace rendered pixels."""
    if settings["floor"] == "native":
        return
    model = native._mjcf_model
    texture, material, ground = (model.find(kind, name) for kind, name in
                                  (("texture", "grid"), ("material", "grid"), ("geom", "ground")))
    if any(element is None for element in (texture, material, ground)):
        raise ValueError("Reacher MJCF must expose grid texture/material and ground geom")
    texture.file = None
    texture.type = "2d"
    texture.builtin = "checker"
    texture.width, texture.height = 300, 300
    texture.rgb1, texture.rgb2 = settings["rgb1"], settings["rgb2"]
    texture.mark, texture.markrgb = "edge", settings["rgb2"]
    material.texture = texture
    material.texrepeat, material.texuniform = settings["texrepeat"], settings["texuniform"]
    ground.material = material
    native.mark_dirty()


def reset_options(settings):
    if settings["floor"] == "native":
        return None
    # The native reset re-applies variation-space colors. Supply the same values
    # so a configured texture is not silently recolored at the next episode.
    return {"variation_values": {"floor.color": np.asarray([settings["rgb1"], settings["rgb2"]], dtype=np.float64)}}


def floor_description(native):
    """Describe effective MJCF mapping after a reset for preview diagnostics."""
    model = native._mjcf_model
    texture, material = model.find("texture", "grid"), model.find("material", "grid")
    return {"builtin": str(texture.builtin), "rgb1": texture.rgb1.tolist(),
            "rgb2": texture.rgb2.tolist(), "texrepeat": material.texrepeat.tolist(),
            "texuniform": str(material.texuniform).lower() == "true"}

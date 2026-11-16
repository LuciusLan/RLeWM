"""Shared RGB and deterministic floor-pattern specifications."""

import numpy as np


PATTERNS = ("preserve", "solid", "checkerboard", "stripes")


def settings_from_args(args):
    settings = {name: getattr(args, name) for name in
                ("object_color", "agent_color", "goal_color", "background_color",
                 "background_color2", "background_pattern", "pattern_size")}
    for key, value in settings.items():
        if key.endswith("color") or key == "background_color2":
            if value is not None and (len(value) != 3 or any(type(v) is not int or not 0 <= v <= 255 for v in value)):
                raise ValueError(f"{key} requires three integer RGB values in [0,255]")
    if not 1 <= settings["pattern_size"] <= 512:
        raise ValueError("--pattern-size must lie in [1,512]")
    if args.dataset == "reacher" and settings["agent_color"] is not None:
        raise ValueError("Reacher's arm uses --object-color; --agent-color is for PushT")
    if args.dataset == "reacher" and settings["goal_color"] is not None:
        raise ValueError("--goal-color controls PushT's decorative goal; Reacher's qpos-match target stays hidden")
    if settings["background_pattern"] == "solid" and settings["background_color2"] is not None:
        raise ValueError("A solid background has one color; omit --background-color2")
    if args.dataset == "pusht" and settings["background_pattern"] == "preserve" and settings["background_color2"] is not None:
        raise ValueError("PushT's preserved background is solid; choose checkerboard/stripes for --background-color2")
    return settings


def background(settings, default_primary, default_secondary):
    primary = settings["background_color"] or default_primary
    secondary = settings["background_color2"] or default_secondary
    return primary, secondary


def pattern_rgb(size, pattern, primary, secondary, tile):
    """One world-aligned, stationary texture; tile is in a 512-unit scene."""
    y, x = np.indices((size, size))
    blocks_x = np.floor(x * 512 / size / tile).astype(np.int32)
    blocks_y = np.floor(y * 512 / size / tile).astype(np.int32)
    mask = (blocks_x + blocks_y) % 2 if pattern == "checkerboard" else blocks_x % 2
    if pattern == "solid":
        mask[:] = 0
    return np.where(mask[..., None], np.asarray(secondary, dtype=np.uint8),
                    np.asarray(primary, dtype=np.uint8)).astype(np.uint8)

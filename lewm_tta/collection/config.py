"""Collection settings are independent of predictor, retrieval and CEM settings."""

from pathlib import Path

from ..config import ROOT
from ..runtime import read_json


def load_config(path):
    value = read_json(path)
    for key in ("output_dir", "reference_dataset"):
        value[key] = str((ROOT / value[key]).resolve())
    validate_config(value)
    return value


def validate_config(value):
    from ..simulation.environments import fresh_environment_class
    adapter = fresh_environment_class(value["backend"])
    adapter.validate_collection_config(value)
    if "rendering" in value:
        if not hasattr(adapter, "validate_rendering"):
            raise ValueError(f"Explicit rendering settings are not supported for {value['backend']}")
        adapter.validate_rendering(value["rendering"])
    from .goals import available_starts, validate_protocols
    validate_protocols(value)
    for key in ("raw_steps", "windows_per_episode", "seed"):
        if type(value[key]) is not int or value[key] < (0 if key == "seed" else 1):
            raise ValueError(f"{key} has an invalid integer value")
    if available_starts(value) < 1:
        raise ValueError("Collected episodes must include the complete history and every declared future goal")
    if value["windows_per_episode"] > available_starts(value):
        raise ValueError("Not enough horizon-safe windows per episode")
    if set(value["splits"]) != {"validation", "test"} or any(type(n) is not int or n < 1 for n in value["splits"].values()):
        raise ValueError("Provide positive validation/test episode counts")
    from .quality import validate_quality
    validate_quality(value)
    adapter.validate_collection_policy(value)
    if value["compression"] not in ("gzip", "lzf"):
        raise ValueError("Use lossless gzip or lzf image compression")
    if not value["regimes"] or "source" not in value["regimes"] or value["regimes"]["source"] != {}:
        raise ValueError("Include an unmodified source regime named 'source'")
    for name, shift in value["regimes"].items():
        if not name or Path(name).name != name or name in (".", ".."):
            raise ValueError("Regime names must be simple directory names")
        adapter.validate_shift(shift)


def headless():
    import os
    for key, value in {"SDL_VIDEODRIVER": "dummy", "SDL_AUDIODRIVER": "dummy", "PYGAME_HIDE_SUPPORT_PROMPT": "1"}.items():
        os.environ.setdefault(key, value)
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

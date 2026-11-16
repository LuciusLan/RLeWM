"""Model-free eligibility and deterministic sampling of recorded query windows.

Bounds/contact measurements belong to the environment adapter. This module
only handles time alignment, declared thresholds and reproducible selection.
"""

import numpy as np

from ..runtime import object_hash


def validate_quality(config):
    settings = config.get("quality")
    if settings is None:
        return
    keys = {"coordinate_units", "visibility_margin", "min_contact_steps", "min_block_translation",
            "min_block_rotation_degrees", "exclude_initial_success", "max_attempts"}
    if not isinstance(settings, dict) or set(settings) != keys:
        raise ValueError(f"quality must specify exactly {sorted(keys)}")
    if not isinstance(settings["coordinate_units"], str) or not settings["coordinate_units"]:
        raise ValueError("Declare quality.coordinate_units for the task adapter's physical measurements")
    for key in ("visibility_margin", "min_block_translation", "min_block_rotation_degrees"):
        number = settings[key]
        if isinstance(number, bool) or not isinstance(number, (int, float)) or not np.isfinite(number) or number < 0:
            raise ValueError(f"quality.{key} must be finite and nonnegative")
    for key in ("min_contact_steps", "max_attempts"):
        if type(settings[key]) is not int or settings[key] < 1:
            raise ValueError(f"quality.{key} must be a positive integer")
    if settings["min_contact_steps"] > config["goal_offset_steps"]:
        raise ValueError("min_contact_steps exceeds the future continuation length")
    if type(settings["exclude_initial_success"]) is not bool:
        raise ValueError("exclude_initial_success must be a boolean")
    if settings["min_block_translation"] == 0 or not 0 < settings["min_block_rotation_degrees"] <= 180:
        raise ValueError("Use positive motion thresholds and a rotation threshold at most 180 degrees")


def geometry_visible(bounds, extent, margin):
    """Full world-space shapes, not body anchors; axes are [..., body, xyxy]."""
    bounds = np.asarray(bounds)
    return bool(np.isfinite(bounds).all() and (bounds[..., :2] >= margin).all()
                and (bounds[..., 2:] <= extent-margin).all()
                and (bounds[..., :2] <= bounds[..., 2:]).all())


def select_starts(case, candidates, count):
    candidates = np.asarray(candidates, dtype=np.int64)
    if len(candidates) < count:
        raise ValueError(f"Only {len(candidates)} eligible windows; {count} required")
    seed = int(object_hash([case["case_id"], "windows"])[:16], 16)
    return np.sort(np.random.default_rng(seed).choice(candidates, count, replace=False)).tolist()


def assess_episode(config, states, bounds, contacts, extent, diagnostics):
    """Contact at row r describes action[r-1]; history and future stay in-frame."""
    settings = config["quality"]
    states, bounds, contacts = map(np.asarray, (states, bounds, contacts))
    rows = len(states)
    if (states.ndim != 2 or rows == 0 or bounds.shape != (rows, 2, 4) or contacts.shape != (rows,)
            or not np.isfinite(states).all() or not np.isfinite(contacts).all() or (contacts < 0).any()):
        raise ValueError("Invalid episode quality measurements")
    failures = {"visibility": 0, "no_contact": 0, "low_motion": 0, "initial_success": 0}
    eligible = []
    history_span = (config["history"]-1)*config["frameskip"]
    future = config["goal_offset_steps"]
    for start in range(max(0, rows-history_span-future)):
        current, goal = start+history_span, start+history_span+future
        metric = diagnostics(states[current], states[goal])
        checks = {
            "visibility": geometry_visible(bounds[start:goal+1], extent, settings["visibility_margin"]),
            "no_contact": np.count_nonzero(contacts[current+1:goal+1]) >= settings["min_contact_steps"],
            "low_motion": (metric["block_distance"] >= settings["min_block_translation"]
                           or metric["angle_distance"] >= np.deg2rad(settings["min_block_rotation_degrees"])),
            "initial_success": not (settings["exclude_initial_success"] and metric["initial_success"]),
        }
        for name, passed in checks.items():
            failures[name] += int(not passed)
        if all(checks.values()):
            eligible.append(start)
    visible = geometry_visible(bounds, extent, settings["visibility_margin"])
    complete = rows == config["raw_steps"]+1
    accepted = complete and visible and len(eligible) >= config["windows_per_episode"]
    reasons = ([] if complete else ["incomplete_horizon"]) + ([] if visible else ["episode_visibility"])
    if len(eligible) < config["windows_per_episode"]:
        reasons.append("insufficient_eligible_windows")
    return {"accepted": bool(accepted), "reasons": reasons, "observations": rows,
            "eligible_start_steps": eligible, "candidate_windows": max(0, rows-history_span-future),
            "window_failure_counts": failures, "agent_block_contact_steps": int(np.count_nonzero(contacts[1:])),
            "minimum_clearance": float(min(bounds[..., :2].min(), (extent-bounds[..., 2:]).min())),
            "extent": float(extent), "thresholds": settings,
            "note": "Reference behavior eligibility only; no learned model, planner outcome or green-target success is consulted"}

"""Active stage variants and their existing numerical backends."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

STAGES = {
    "p1": {"default": ("p1", "p1/default.json"),
           "tworooms": ("p1", "p1/tworooms.json"),
           "reacher": ("p1", "p1/reacher.json"),
           "cube": ("p1", "p1/cube.json")},
    "p2": {"default": ("p2_followup", "p2/default.json"),
           "fresh": ("p2_followup", "p2/fresh.json"),
           "fresh-temporal": ("p2_followup", "p2/fresh-temporal.json"),
           "initial": ("p25", "p2/initial.json"),
           "followup": ("p2_followup", "p2/followup.json")},
    "p3": {"default": ("p3c", "p3/default.json"),
           "fresh": ("p3c", "p3/fresh.json"),
           "fresh-proposals": ("p3b", "p3/fresh-proposals.json"),
           "fresh-temporal": ("p3_temporal", "p3/fresh-temporal.json"),
           "interleave": ("p3_interleave", "p3/interleave.json"),
           "interleave-lewm-budget": ("p3_interleave", "p3/interleave-lewm-budget.json"),
           "core": ("p3_core", "p3/core.json"),
           "shared": ("p3a", "p3/shared.json"),
           "proposals": ("p3b", "p3/proposals.json"),
           "temporal": ("p3_temporal", "p3/temporal.json")},
    "coverage": {"default": ("rank_coverage", "coverage/default.json")},
    "sim": {"default": ("sim", "sim/default.json"), "fresh": ("sim", "sim/fresh.json"),
            "green": ("sim", "sim/green.json")},
    "collect": {"default": ("collect", "collect/default.json"),
                "shifts": ("collect", "collect/shifts.json"),
                "tworooms": ("collect", "collect/tworooms.json"),
                "reacher": ("collect", "collect/reacher.json"),
                "reacher-checkerboard": ("collect", "collect/reacher-checkerboard.json"),
                "cube": ("collect", "collect/cube.json"),
                "cube-plan": ("collect", "collect/cube-plan.json")},
}

BACKENDS = {
    "p1": ("experiments.backends.p1", ()),
    "p2": ("lewm_tta.rollout.cli", ("p2",)),
    "p25": ("lewm_tta.rollout.cli", ("p25",)),
    "p2_followup": ("lewm_tta.rollout.cli", ("p2_followup",)),
    "p3a": ("lewm_tta.planning.cli", ()),
    "p3b": ("lewm_tta.proposals.cli", ()),
    "p3c": ("lewm_tta.proposals.cli", ("p3c",)),
    "p3_temporal": ("lewm_tta.planning.temporal.cli", ()),
    "p3_interleave": ("lewm_tta.interleave.cli", ()),
    "p3_core": ("lewm_tta.core.cli", ()),
    "p3_success": ("lewm_tta.outcomes.cli", ()),
    "rank_coverage": ("lewm_tta.rank_coverage.cli", ()),
    "sim": ("lewm_tta.simulation.cli", ()),
    "collect": ("lewm_tta.collection.cli", ()),
}

REPORT_STAGES = {
    "P2-recorded-rollout": "p2", "P2.5-scope-comparison": "p25",
    "P2-followup-iteration": "p2_followup", "P3a-offline-cem": "p3a",
    "P3b-offline-proposals": "p3b", "P3c-offline-factorial": "p3c",
    "rank-coverage": "rank_coverage",
    "P3-temporal-cem": "p3_temporal",
    "P3-interleaved-cem": "p3_interleave",
    "P3-core-planning": "p3_core",
    "P3-executed-plan-outcomes": "p3_success",
    "simulator-evaluation": "sim",
    "fresh-collection": "collect",
}


def profile(stage, variant="default"):
    try:
        backend, name = STAGES[stage][variant]
    except KeyError:
        raise ValueError(f"Unknown stage/variant: {stage}/{variant}") from None
    return backend, ROOT / "codes/configs" / name


def infer_backend(stage, config):
    if stage == "p2":
        return "p2_followup" if "schedules" in config else "p25" if "scopes" in config else "p2"
    if stage == "p3":
        if "core" in config:
            return "p3_core"
        if config.get("comparison") == "interleaved-planning":
            return "p3_interleave"
        if config.get("comparison") == "temporal-planning":
            return "p3_temporal"
        return {"factorial": "p3c", "proposal-only": "p3b"}.get(config.get("comparison"), "p3a")
    return profile(stage)[0]


def stage_for_backend(backend):
    if backend in ("p2", "p25", "p2_followup"):
        return "p2"
    if backend in ("p3a", "p3b", "p3c", "p3_temporal", "p3_interleave", "p3_success", "p3_core"):
        return "p3"
    return "coverage" if backend == "rank_coverage" else backend

"""Compatibility imports; shared implementation lives in lewm_tta.rollout."""

from ..rollout.compat import expose

expose(__name__, {
    "config": ("config", {}),
    "prepared": ("resources", {}),
    "trajectories": ("cohorts", {"make_cohort": ("cohorts", "filter_cohort")}),
    "inference": ("inference", {}),
    "checks": ("checks", {"run_checks": ("checks", "run_rollout_checks")}),
    "reporting": ("reporting", {
        "summarize": ("reporting", "summarize_methods"),
        "format_summary": ("reporting", "format_method_summary"),
    }),
    "pipeline": ("pipeline", {}),
})

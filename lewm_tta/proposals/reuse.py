"""Verified reuse of temporal Gaussian plans and Frozen proposal cells.

Source reports and records stay immutable. Shared-depth and candidate-temporal
are deliberately ineligible: their scorers differ from the factorial's fixed WM.
"""

from copy import deepcopy
from pathlib import Path

import numpy as np

from ..config import ROOT
from ..runtime import file_hash, object_hash, read_json
from ..rollout.record_io import read_saved_records


def validate_reuse(config):
    settings = config.get("reuse", {})
    if not isinstance(settings, dict) or set(settings) - {"validation", "test"}:
        raise ValueError("reuse must map validation/test to lists of pinned reports")
    for split, entries in settings.items():
        if not isinstance(entries, list):
            raise ValueError(f"reuse.{split} must be a list")
        for entry in entries:
            if (not isinstance(entry, dict) or set(entry) != {"report", "sha256"}
                    or not isinstance(entry["report"], str) or not entry["report"]
                    or not isinstance(entry["sha256"], str) or len(entry["sha256"]) != 64
                    or any(c not in "0123456789abcdef" for c in entry["sha256"])):
                raise ValueError("Each reused report requires a path and lowercase SHA256")


def pinned_report(entry):
    path = Path(entry["report"])
    path = path if path.is_absolute() else ROOT / path
    if file_hash(path) != entry["sha256"]:
        raise ValueError(f"Reused report changed: {path}")
    return path.resolve(), read_json(path)


def normalize_reuse(config):
    validate_reuse(config)
    for entries in config.get("reuse", {}).values():
        for entry in entries:
            entry["report"] = str((ROOT / entry["report"]).resolve())


def check_protocol(source, config, split, manifest):
    if (not source.get("completed") or source.get("limited_run")
            or source.get("stage") != "P3-temporal-cem" or source.get("split") != split):
        raise ValueError("Reuse requires a complete temporal report from the same split")
    previous = source["config"]
    for field in ("planning", "scopes", "query_selection"):
        if previous[field] != config[field]:
            raise ValueError(f"Reused protocol differs: {field}")
    for field in ("paths", "data", "retrieval"):
        if previous["p1"][field] != config["p1"][field]:
            raise ValueError(f"Reused memory/checkpoint/metric differs: p1.{field}")
    for field, value in config["adaptation"].items():
        if field != "methods" and previous["adaptation"].get(field) != value:
            raise ValueError(f"Reused adaptation differs: {field}")
    if source["p1_prepared_fingerprint"] != manifest["p1_prepared_fingerprint"]:
        raise ValueError("Reused observations, memory or normalization changed")
    if source["cohort"] != manifest["splits"][split]:
        raise ValueError("Reused query cohort/episode split differs")
    if (source["summary"]["queries"] != source["cohort"]["queries"]
            or source["summary"]["episodes"] != source["cohort"]["episodes"]):
        raise ValueError("Reused report does not cover its full declared cohort")


def check_numerical_sources(source, path):
    from experiments.provenance import archive_entry, backend_identity

    # A future locked replay can itself live under artifacts/code-archives/.
    # Locate the source workspace from the pinned report, not the replay's ROOT.
    workspace = next((parent for parent in path.parents
                      if (parent / "artifacts/code-archives").is_dir()), ROOT)
    if source["code"] != backend_identity("p3_temporal") and archive_entry("p3_temporal", source["code"], root=workspace) is None:
        raise ValueError("Reused execution source has no verified archive")
    # Presentation/reporting changes are allowed. The shared computation is not.
    names = ("model.py", "adaptation.py", "retrieval.py", "cuvs_retrieval.py", "indexed_keys.py",
             "frame_bank.py", "observation_cache.py", "rollout/scopes.py", "rollout/support_updates.py",
             "rollout/inference.py", "planning/actions.py", "planning/cem.py", "planning/shared_ttt.py",
             "planning/temporal/state.py", "planning/temporal/rollouts.py")
    for name in names:
        relative = f"codes/lewm_tta/{name}"
        if source["code"].get(relative) != file_hash(ROOT / relative):
            raise ValueError(f"Reused numerical implementation changed: {relative}")


class ReusedPlans:
    def __init__(self, config, split, manifest):
        validate_reuse(config)
        self.config, self.rows, self.origins, self.sources = config, {}, {}, []
        self.proposal_rows, self.proposal_origins = {}, {}
        methods = config["adaptation"]["methods"]
        seeds = config["p1"]["evaluation"]["seeds"]
        for entry in config.get("reuse", {}).get(split, []):
            path, source = pinned_report(entry)
            from .reuse_frozen import PROPOSAL_STAGES
            if source.get("stage") in PROPOSAL_STAGES:
                self._load_frozen(path, source, entry, config, split, manifest)
                continue
            check_protocol(source, config, split, manifest)
            check_numerical_sources(source, path)
            rows = read_saved_records(source)
            from ..planning.temporal.reporting import summarize
            if summarize(rows, source["config"]) != source["summary"] or not all(source["checks"].values()):
                raise ValueError("Reused records do not reproduce their checked summary")
            mapping = {"frozen": "frozen", "shared-initial": source["config"]["temporal"]["selector"]}
            origin = {"report": str(path), "sha256": entry["sha256"],
                      "records_file": source["records_file"], "records_sha256": source["records_sha256"]}
            count = 0
            for mode, method in mapping.items():
                if method not in methods:
                    continue
                if not set(seeds) <= set(source["summary"]["seeds_by_mode"].get(mode, [])):
                    raise ValueError(f"Reused {mode} does not cover every requested seed")
                for row in rows:
                    if row["mode"] != mode or row["seed"] not in seeds:
                        continue
                    key = row["query"]["transition_id"], row["seed"], method
                    if key in self.rows or (*key, "gaussian") in self.proposal_rows:
                        raise ValueError("Overlapping reuse sources; choose one origin per Gaussian cell")
                    self.rows[key] = row
                    self.origins[key] = {**origin, "mode": mode, "query": key[0], "seed": key[1]}
                    count += 1
            self.sources.append({**origin, "available_rows": count, "model": source["model"],
                                 "action_coordinates": source["action_coordinates"],
                                 "scope_parameters": source["scope_parameters"]})
        self.reused_count = 0

    def _load_frozen(self, path, source, entry, config, split, manifest):
        from .reuse_frozen import load_frozen_cells
        cells = load_frozen_cells(path, source, config, split, manifest)
        origin = {"report": str(path), "sha256": entry["sha256"],
                  "records_file": source["records_file"], "records_sha256": source["records_sha256"]}
        for key, row in cells.items():
            if key in self.proposal_rows or (key[3] == "gaussian" and key[:3] in self.rows):
                raise ValueError("Overlapping reuse sources for a Frozen proposal cell")
            self.proposal_rows[key] = row
            self.proposal_origins[key] = {**origin, "proposal_source": key[3], "query": key[0], "seed": key[1]}
        self.sources.append({**origin, "available_rows": len(cells), "model": source["model"],
                             "action_coordinates": source["action_coordinates"],
                             "scope_parameters": source["scope_parameters"], "source_stage": source["stage"]})

    def check_resources(self, resources, coordinates):
        for source in self.sources:
            if (source["model"] != resources.adapter.identity
                    or source["action_coordinates"] != coordinates.description()
                    or source["scope_parameters"] != resources.adapter.scope_description()):
                raise ValueError("Reused runtime model, scope, packages or action coordinates changed")

    def reference(self, task, seed, method, source="gaussian"):
        key = task.metadata["transition_id"], seed, method
        return self.proposal_rows.get((*key, source), self.rows.get(key) if source == "gaussian" else None)

    def origin(self, task, seed, method, source="gaussian"):
        key = task.metadata["transition_id"], seed, method
        return self.proposal_origins.get((*key, source), self.origins.get(key) if source == "gaussian" else None)

    def verify_adaptation(self, task, seed, method, provisional, adapted, adapter, coordinates):
        row = self.reference(task, seed, method)
        if row is None:
            return
        if row["query"] != task.metadata or row["goal_raw_row"] != task.goal_row:
            raise ValueError("Reused query/goal alignment differs")
        if "proposal_source" in row:
            if method != "frozen" or row["provisional_fingerprint"] != object_hash(provisional.cpu().tolist()):
                raise ValueError("Reused Frozen reset/provisional-action protocol differs")
            for field in ("support_fingerprint", "reset_seed", "updates", "unique_support", "support_presentations",
                          "parameter_displacement_mse", "skip_reason"):
                if row["adaptation"][field] != adapted[field]:
                    raise ValueError(f"Reused Frozen adaptation differs: {field}")
            return
        import torch
        from ..planning.temporal.rollouts import reference_actions
        from ..planning.temporal.state import ParameterSnapshot

        references = reference_actions(coordinates, self.config, task.metadata, seed)
        torch.testing.assert_close(references[:, 0], provisional, rtol=0, atol=0)
        if object_hash(references.cpu().tolist()) != row["reference_fingerprint"]:
            raise ValueError("Reused provisional action RNG differs")
        previous = row["setup_adaptation"]
        for field in ("support_fingerprint", "reset_seed", "updates", "unique_support", "support_presentations",
                      "parameter_displacement_mse", "skip_reason"):
            if previous[field] != adapted[field]:
                raise ValueError(f"Reused shared adaptation differs: {field}")
        fingerprint = ParameterSnapshot(adapter).fingerprint
        if previous["model_fingerprints"] != [fingerprint] * self.config["planning"]["horizon"]:
            raise ValueError("Reused Gaussian scorer parameters differ from the new guided scorer")

    def verify_proposal(self, task, seed, method, source, proposal):
        row = self.reference(task, seed, method, source)
        if row is None or "proposal_source" not in row:
            return
        for field in ("source", "fragment_fingerprint", "injection_fingerprint", "injected_slots",
                      "requested_slots", "slot_provenance"):
            if row["proposal"][field] != proposal[field]:
                raise ValueError(f"Reused proposal/RNG differs: {field}")

    def guided_plan(self, task, seed, method, adapter, source="gaussian"):
        row = self.reference(task, seed, method, source)
        if row is None:
            return None
        import torch
        from ..planning.cem import Plan
        from .cem import GuidedPlan

        planning = self.config["planning"]
        plan = Plan(torch.tensor(row["selected_actions_normalized"], device=adapter.device),
                    row["selected_model_goal_sse"], deepcopy(row["cem_trace"]),
                    planning["population"] * planning["iterations"] + 1,
                    row["cem_first_population_fingerprint"])
        self.reused_count += 1
        if "proposal_source" in row:
            return GuidedPlan(plan, row["base_first_population_fingerprint"], row["proposal"]["injection_fingerprint"],
                              row["proposal"]["injected_slots"], row["direct_elite_source_fraction"])
        return GuidedPlan(plan, plan.first_population_fingerprint, None, 0, 0.)

    @staticmethod
    def forecast(current, reference):
        if reference is None:
            return current
        if not np.allclose(current, reference["recorded_action_mse_by_block"], rtol=1e-6, atol=1e-8):
            raise ValueError("Reused recorded-action forecast failed numerical parity")
        return list(reference["recorded_action_mse_by_block"])

    @staticmethod
    def retain_scores(row, reference):
        for field in ("selected_model_goal_sse", "selected_plan_frozen_model_goal_sse", "frozen_plan_this_model_goal_sse"):
            if not np.isclose(row[field], reference[field], rtol=1e-5, atol=1e-6):
                raise ValueError(f"Reused selected-plan metric failed numerical parity: {field}")
            row[field] = reference[field]

    def description(self, records):
        return {"source_reports": [{k: v for k, v in item.items()
                                    if k not in ("model", "action_coordinates", "scope_parameters")}
                                   for item in self.sources],
                "reused_cem_calls": self.reused_count,
                "new_cem_calls": len(records) - self.reused_count,
                "verification": "Exact support/model/proposal/RNG parity; cheap saved-plan diagnostic checks; no reused CEM search rerun",
                "timing": "Reused primary/CEM latencies retain their source-run overhead; diagnostic parity work is new and separately timed. Mixed historical timings are not a matched-wall-clock comparison."}


def verify_dependencies(report):
    """Lock/reanalysis must still be able to verify every imported source."""
    for entry in report.get("reuse", {}).get("source_reports", []):
        _, source = pinned_report({"report": entry["report"], "sha256": entry["sha256"]})
        if (source["records_file"] != entry["records_file"]
                or source["records_sha256"] != entry["records_sha256"]
                or file_hash(Path(entry["records_file"])) != entry["records_sha256"]):
            raise ValueError("A reused source record file changed")

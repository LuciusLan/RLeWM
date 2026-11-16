"""CPU simulator contracts and explicit reconstruction diagnostics, not model tests."""

from pathlib import Path
import numpy as np

from experiments.progress import progress
from experiments.provenance import backend_identity
from ..runtime import object_hash, read_json, write_json
from .cohorts import load_manifest, selected_cases
from .dataset import source_dataset
from .environments import make_environment
from .history import bootstrap


def check(config):
    from .green import enabled
    if enabled(config):
        from .reset_checks import check as check_resets
        return check_resets(config)
    manifest = load_manifest(config)
    cases = selected_cases(config, manifest)
    dataset = source_dataset(config)
    environment = make_environment(config["simulation"], dataset)
    rows = []
    seed = config["simulation"]["seeds"][0]
    try:
        for case in progress(cases, config=config, desc="Simulator preflight", unit="episode"):
            dataset.validate_case(case, manifest["frameskip"], manifest["history"])
            first = environment.reset(case, seed, history_start=False)
            second = environment.reset(case, seed, history_start=False)
            if not np.array_equal(first.state, second.state) or not np.array_equal(first.rgb, second.rgb):
                raise AssertionError("Repeated simulator resets are not deterministic")
            initial = environment.reconstruction(first, case["current_raw_row"])
            raw = dataset.rows("action", case["current_raw_row"], case["goal_raw_row"])
            for action in raw:
                second = environment.step(action, recorded=True)
                if second.truncated:
                    raise ValueError("Fixed-horizon recorded-action check truncated unexpectedly")
            endpoint = second
            environment.reset(case, seed, history_start=False)
            for action in raw:
                second = environment.step(action, recorded=True)
            if not np.array_equal(endpoint.state, second.state) or not np.array_equal(endpoint.rgb, second.rgb):
                raise AssertionError("Repeated recorded-action continuations are not deterministic")
            _, _, warm = bootstrap(environment, dataset, case, seed, manifest["history"], manifest["frameskip"])
            past = dataset.rows("action", case["start_raw_row"], case["current_raw_row"])
            low, high = (np.asarray(config["planning"][key]) for key in ("raw_action_low", "raw_action_high"))
            def action_diagnostics(actions):
                outside = ((actions < low-1e-6) | (actions > high+1e-6)).any(axis=1)
                return {"raw_steps": len(actions), "outside_nominal_planner_bounds_steps": int(outside.sum()),
                        "maximum_absolute_coordinate": float(np.max(np.abs(actions), initial=0)),
                        "executed_unchanged": True}
            rows.append({"case_id": case["case_id"], "reset_reconstruction": initial,
                         "recorded_action_diagnostics": {"warmup": action_diagnostics(past),
                                                         "continuation": action_diagnostics(raw)},
                         "recorded_action_endpoint_reconstruction": environment.reconstruction(endpoint, case["goal_raw_row"]),
                         "recorded_action_endpoint_metrics": endpoint.metrics,
                         "mpc_warmup_current_reconstruction": environment.reconstruction(warm, case["current_raw_row"]),
                         "deterministic_reset_and_continuation": True})
        diagnostic_names = ("reset_reconstruction", "recorded_action_endpoint_reconstruction", "mpc_warmup_current_reconstruction")
        reconstruction_summary = {name: {key: {"mean": float(np.mean([row[name][key] for row in rows])),
                                               "maximum": float(max(row[name][key] for row in rows))}
                                        for key in rows[0][name]} for name in diagnostic_names}
        result = {"passed": True, "manifest_fingerprint": manifest["fingerprint"],
                  "environment": environment.description(), "seed": seed,
                  "case_ids": [case["case_id"] for case in cases], "code": backend_identity("sim"),
                  "checks": rows, "reconstruction_summary": reconstruction_summary,
                  "scope": "deterministic CPU physics/rendering, read-only data, finite shape-checked recorded commands executed unchanged; planned-action bounds remain enforced; no learned-model execution",
                  "reconstruction_note": "Reconstruction errors are diagnostics, not hidden acceptance thresholds. Resets omit unrecorded physics state; inspect these values before interpreting source-dynamics fidelity."}
        if config.get("query_dataset"):
            result["reconstruction_note"] = "Fresh episodes reset with the original seed and replay the complete recorded past prefix; block velocities/contact state are restored through execution. Inspect exact state/RGB parity diagnostics."
        result["fingerprint"] = object_hash(result)
        path = Path(config["experiment_dir"])/"checks"/f"preflight-{result['fingerprint'][:16]}.json"
        write_json(path, result)
        print(f"Simulator contracts passed for {len(cases)} episodes. Reconstruction diagnostics: {path}")
        for name, metrics in reconstruction_summary.items():
            print(name + ": " + "; ".join(f"{key} mean={value['mean']:.4g}, max={value['maximum']:.4g}" for key, value in metrics.items()))
        return result
    finally:
        environment.close()
        dataset.close()


def require_check(config, manifest, cases, environment):
    desired = {case["case_id"] for case in cases}
    for path in sorted((Path(config["experiment_dir"])/"checks").glob("preflight-*.json")):
        result = read_json(path)
        if (result.get("passed") and result.get("manifest_fingerprint") == manifest["fingerprint"]
                and result.get("environment") == environment.description()
                and result.get("code") == backend_identity("sim")
                and result.get("seed") == config["simulation"]["seeds"][0]
                and desired <= set(result.get("case_ids", []))
                and result.get("fingerprint") == object_hash({k:v for k,v in result.items() if k != "fingerprint"})):
            return {"path": str(path.resolve()), "fingerprint": result["fingerprint"],
                    "reconstruction_note": result["reconstruction_note"]}
    raise ValueError("Run 'sim check' for this cohort/code/environment before evaluation; a limited check covers only its cases")

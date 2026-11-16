"""CPU preflight for native reset cases and independently rendered target RGB."""

from pathlib import Path

import numpy as np

from experiments.progress import progress
from experiments.provenance import backend_identity
from ..runtime import object_hash, write_json
from .cohorts import load_manifest, selected_cases
from .dataset import source_dataset
from .environments import make_environment
from .history import bootstrap


def check(config):
    manifest = load_manifest(config)
    cases = selected_cases(config, manifest)
    dataset = source_dataset(config)
    environment = make_environment(config["simulation"], dataset)
    rows = []
    seed = config["simulation"]["seeds"][0]
    try:
        for case in progress(cases, config=config, desc="Green target preflight", unit="episode"):
            dataset.validate_case(case, manifest["frameskip"], manifest["history"])
            first = environment.reset(case, seed, history_start=True)
            second = environment.reset(case, seed, history_start=True)
            if not np.array_equal(first.state, second.state) or not np.array_equal(first.rgb, second.rgb):
                raise AssertionError("Native reset is not deterministic")
            images, blocks, current = bootstrap(environment, dataset, case, seed, manifest["history"], manifest["frameskip"])
            goal = environment.goal_image.copy()
            contract = dict(environment.goal_contract)
            next_state = environment.step(np.array([.1, 0], dtype=np.float32))
            repeated_images, repeated_blocks, repeated = bootstrap(environment, dataset, case, seed+1, manifest["history"], manifest["frameskip"])
            if (not np.array_equal(current.state, repeated.state) or not np.array_equal(current.rgb, repeated.rgb)
                    or not np.array_equal(goal, environment.goal_image)
                    or any(not np.array_equal(a, b) for a, b in zip(images+blocks, repeated_images+repeated_blocks))):
                raise AssertionError("Warm-up/goal changed across paired planner seeds")
            repeated_next = environment.step(np.array([.1, 0], dtype=np.float32))
            if not np.array_equal(next_state.state, repeated_next.state) or not np.array_equal(next_state.rgb, repeated_next.rgb):
                raise AssertionError("Native continuation is not deterministic")
            if (goal.shape != current.rgb.shape or goal.dtype != np.uint8
                    or not np.isfinite(current.state).all()
                    or not all(np.isfinite(float(v)) for v in current.metrics.values())
                    or not 0 <= current.metrics["coverage"] <= 1
                    or abs(contract["rendered_goal_coverage"]-1) > 1e-9
                    or contract["marker_symmetric_difference_area"] > 1e-7):
                raise AssertionError("Green goal image/marker geometry or metrics failed validation")
            rows.append({"case_id": case["case_id"], "goal_contract": contract,
                         "initial_state": current.state.tolist(), "initial_metrics": current.metrics,
                         "deterministic_reset_history_goal_and_continuation": True})
        result = {"passed": True, "task": "green-target", "manifest_fingerprint": manifest["fingerprint"],
                  "environment": environment.description(), "seed": seed, "case_ids": [case["case_id"] for case in cases],
                  "code": backend_identity("sim"), "checks": rows,
                  "initially_successful_cases": sum(row["initial_metrics"]["success"] for row in rows),
                  "initially_out_of_frame_cases": sum(row["initial_metrics"]["out_of_frame"] for row in rows),
                  "scope": "CPU native reset/warm-up/continuation determinism and target geometry; no model execution",
                  "reconstruction_note": "Independent native seeds and fixed zero-action history; no recorded-trajectory reconstruction. All reserved cases retained."}
        result["fingerprint"] = object_hash(result)
        path = Path(config["experiment_dir"])/"checks"/f"preflight-{result['fingerprint'][:16]}.json"
        write_json(path, result)
        print(f"Green contracts passed for {len(cases)} episodes: {path}")
        return result
    finally:
        environment.close()
        dataset.close()

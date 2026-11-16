"""Render saved green-target MPC controls; no encoder, CEM or TTT is run."""

from pathlib import Path
import math

import numpy as np

from ..collection.manifest import checked
from ..config import ROOT
from ..rollout.record_io import read_saved_records
from ..runtime import file_hash, read_json, write_json
from .episodes import output_directory
from .media import Panel, index_page, save_gallery


def selections(report, manifest, rows, episodes, methods, proposals, seed):
    """Episode indices refer to the full reserved split, including limited runs."""
    if (report.get("stage") != "simulator-evaluation" or report.get("task") != "green-target"
            or report.get("mode") != "mpc" or manifest.get("task") != "green-target"
            or manifest["fingerprint"] != report["manifest_fingerprint"]):
        raise ValueError("A matching green-target MPC report and manifest are required")
    for name, values in (("episodes", episodes), ("methods", methods), ("proposals", proposals)):
        if not values or len(values) != len(set(values)):
            raise ValueError(f"Choose distinct nonempty {name}")
    seeds = report["config"]["simulation"]["seeds"]
    seed = seeds[0] if seed is None else seed
    if seed not in seeds:
        raise ValueError("Requested seed was not evaluated in this report")
    if set(methods)-set(report["config"]["adaptation"]["methods"]) or set(proposals)-set(report["config"]["proposals"]["sources"]):
        raise ValueError("Requested method/proposal was not evaluated in this report")
    cases = manifest["splits"][report["split"]]
    lookup = {}
    for row in rows:
        key = (row["case_id"], row["seed"], row["method"], row["proposal_source"])
        if key in lookup:
            raise ValueError("Duplicate saved simulator outcome")
        lookup[key] = row
    selected = []
    for episode in episodes:
        if type(episode) is not int or not 0 <= episode < len(cases):
            raise ValueError("Episode index is outside the report's reserved split")
        case = cases[episode]
        for method in methods:
            for proposal in proposals:
                key = (case["case_id"], seed, method, proposal)
                if case["case_id"] not in report["case_ids"] or key not in lookup:
                    raise ValueError(f"This case/treatment has no saved outcome: {key}")
                row = lookup[key]
                if row["query"] != case["query"]:
                    raise ValueError("Saved outcome belongs to a different reserved case")
                selected.append((episode, case, row))
    return selected


def verify_replay_source(report):
    """Bind physical replay, allowing unrelated predictor/runner edits later."""
    dependencies = {"codes/lewm_tta/simulation/history.py", "codes/lewm_tta/simulation/execution.py",
                    "codes/lewm_tta/simulation/dataset.py", "codes/lewm_tta/collection/quality.py"}
    code = {name: digest for name, digest in report["code"].items()
            if name in dependencies or name.startswith("codes/lewm_tta/simulation/environments/")}
    if not dependencies <= set(code) or "codes/lewm_tta/simulation/environments/green_pusht.py" not in code:
        raise ValueError("Report lacks the green simulator replay source identity")
    if any(not (ROOT/name).is_file() or file_hash(ROOT/name) != digest for name, digest in code.items()):
        raise ValueError("Physical replay implementation differs from this report; restore its recorded simulator sources")
    return code


def replay_saved(environment, dataset, case, row, manifest, config):
    """Execute exactly the applied controls, checking the original outcome."""
    from ..simulation.execution import Outcome, history_fingerprint
    from ..simulation.history import bootstrap
    dataset.validate_case(case, manifest["frameskip"], manifest["history"])
    images, blocks, initial = bootstrap(environment, dataset, case, row["seed"], manifest["history"], manifest["frameskip"])
    if history_fingerprint(images, blocks) != row["bootstrap_fingerprint"]:
        raise ValueError("Replayed initial history differs from the evaluated history")
    if environment.goal_contract != row["goal_contract"]:
        raise ValueError("Replayed goal image/geometry differs from the evaluated task")
    steps = row["raw_steps"]
    actions = np.asarray(row["executed_raw_actions"], dtype=np.float32).reshape(-1, 2)
    chunk = config["simulation"]["execute_blocks"]*manifest["frameskip"]
    if (type(steps) is not int or steps < 0 or len(actions) != steps
            or steps > config["simulation"]["max_raw_steps"] or not np.isfinite(actions).all()
            or row["planning_calls"] != math.ceil(steps/chunk)):
        raise ValueError("Saved action count/planning-call boundaries disagree with the MPC protocol")
    outcome = Outcome(initial, environment.description()["metric_directions"])
    frames, metrics, states = [initial.rgb.copy()], [dict(initial.metrics)], [initial.state.tolist()]
    for action in actions:
        observation = environment.step(action)
        outcome.append(observation, action)
        frames.append(observation.rgb.copy())
        metrics.append(dict(observation.metrics))
        states.append(observation.state.tolist())
        if observation.truncated:
            raise ValueError("Saved rollout replay truncated before reproducing its endpoint")
    reproduced = outcome.record()
    differences = {}
    for key in ("initial_state", "final_state"):
        first, second = np.asarray(reproduced[key]), np.asarray(row[key])
        if first.shape != second.shape or not np.allclose(first, second, rtol=0, atol=1e-8):
            raise ValueError(f"Saved rollout {key} was not reproduced")
        differences[key+"_max_abs_error"] = float(np.max(np.abs(first-second)))
    for key in ("initial_metrics", "final_metrics", "best_metrics"):
        if set(reproduced[key]) != set(row[key]) or any(not math.isclose(float(value), float(row[key][name]), rel_tol=0, abs_tol=1e-8)
                                                      for name, value in reproduced[key].items()):
            raise ValueError(f"Saved rollout {key} was not reproduced")
    for key in ("ever_success", "initial_success", "raw_steps", "first_success_raw_step", "ever_out_of_frame"):
        if reproduced[key] != row[key]:
            raise ValueError(f"Saved rollout {key} was not reproduced")
    return {"frames": frames, "metrics": metrics, "states": states,
            "goal_image": environment.goal_image.copy(), "actions": actions.tolist(),
            "replan_start_steps": list(range(0, steps, chunk)),
            "verification": {"bootstrap_matches": True, "goal_contract_matches": True,
                             "outcome_metrics_match": True, **differences}}


def preview(args):
    report_path = args.report.resolve()
    report = read_json(report_path)
    if report.get("task") != "green-target":
        raise ValueError("Use a green-target MPC report; recorded-goal plans use 'preview.py cem'")
    manifest = checked(Path(report["config"]["experiment_dir"])/"manifest.json")
    rows = read_saved_records(report)  # Verify complete records checksum/count before any simulator work.
    selected = selections(report, manifest, rows, args.episodes, args.method, args.proposal, args.seed)
    code = verify_replay_source(report)
    from ..simulation.dataset import ResetCases
    from ..simulation.environments import make_environment
    dataset = ResetCases()
    environment = make_environment(report["config"]["simulation"], dataset)
    try:
        if environment.description() != report["environment"]:
            raise ValueError("Native simulator/packages/task differ from the saved evaluation")
        directory = output_directory(args.output, "green-trajectories")
        records, galleries = [], []
        for episode, case, row in selected:
            trace = replay_saved(environment, dataset, case, row, manifest, report["config"])
            name = f"{report['split']}-{episode:05d}-{row['method']}-{row['proposal_source']}-s{row['seed']}"
            chunk = report["config"]["simulation"]["execute_blocks"]*manifest["frameskip"]
            details = [f"coverage={metric['coverage']:.3f} success={int(metric['success'])} "
                       f"call={'start' if step == 0 else (step-1)//chunk}"
                       for step, metric in enumerate(trace["metrics"])]
            media = save_gallery(directory, name, [
                Panel("MPC: actual executed trajectory", trace["frames"], list(range(len(trace["frames"]))), details=details),
                Panel("Fixed green-target goal", [trace["goal_image"]], [0], fixed=True)],
                fps=environment.description()["control_hz"], every=args.every, scale=args.scale)
            record = {"case": case, "episode": episode, "seed": row["seed"], "method": row["method"],
                      "proposal_source": row["proposal_source"], "source_outcome": row,
                      "observed_states": trace["states"], "observed_metrics": trace["metrics"],
                      "replan_start_steps": trace["replan_start_steps"], "verification": trace["verification"],
                      "media": media, "benchmark_result": False}
            write_json(directory/(name+".json"), record)
            records.append(record)
            galleries.append((name, media))
        description = ("Replay of the exact controls actually executed by saved closed-loop MPC. "
                       "The left panel contains simulator observations and starts after history warm-up; the right goal stays fixed. "
                       "Coverage, success, initial/final states and goal identity are checked against saved outcomes. "
                       "Unexecuted CEM plan tails and decoded latent imagination are not shown. No model, retrieval, TTT or CEM is run.")
        index_page(directory, "Green-target planned trajectory outcomes", galleries, description)
        write_json(directory/"preview.json", {"mode": "green-mpc-saved-execution", "report": str(report_path),
            "report_sha256": file_hash(report_path), "records_sha256": report["records_sha256"],
            "manifest_fingerprint": manifest["fingerprint"], "replay_source": code,
            "description": description, "records": records, "benchmark_result": False})
        print(f"Green MPC trajectory previews: {directory/'index.html'}")
        return directory
    finally:
        environment.close()
        dataset.close()

"""User-run frozen-MPC failure diagnosis, using saved validation contexts.

No benchmark settings change. No retrieval index or adaptation is loaded.
"""

import json
from pathlib import Path

import numpy as np
import torch

from ..collection.manifest import checked
from ..config import ROOT
from ..rollout.record_io import open_records, read_saved_records
from ..runtime import file_hash, object_hash, read_json, write_json
from .episodes import output_directory
from .green import replay_saved, selections, verify_replay_source
from .media import Panel, index_page, save_gallery
from .plans import load_planner
from .prediction_audit import compare_prediction, control_probes, execute_probe, native_rollout_parity


class RestoreContext:
    """Rebuild hidden physics by replaying the saved prefix, never state-setting."""
    def __init__(self, environment, dataset, case, row, manifest, prefix, expected_history):
        self.environment, self.dataset, self.case = environment, dataset, case
        self.row, self.manifest, self.prefix = row, manifest, prefix
        self.expected_history = expected_history

    def __call__(self):
        from ..simulation.execution import history_fingerprint
        from ..simulation.history import bootstrap
        frames, blocks, observation = bootstrap(self.environment, self.dataset, self.case, self.row["seed"],
                                               self.manifest["history"], self.manifest["frameskip"])
        if (history_fingerprint(frames, blocks) != self.row["bootstrap_fingerprint"]
                or self.environment.goal_contract != self.row["goal_contract"]):
            raise ValueError("Diagnostic reset differs from the saved history/goal")
        self.images, self.blocks = list(frames), list(blocks)
        block = []
        for action in self.prefix:
            observation = self.environment.step(action)
            if observation.truncated:
                raise ValueError("Saved prefix truncated unexpectedly")
            block.append(action)
            if len(block) == self.manifest["frameskip"]:
                self.images.append(observation.rgb)
                self.blocks.append(np.asarray(block, dtype=np.float32))
                block = []
        if block:
            raise ValueError("Replan context must start at a complete observation block")
        self.images = self.images[-self.manifest["history"]:]
        self.blocks = self.blocks[-(self.manifest["history"]-1):]
        if not np.allclose(observation.state, self.expected_history, rtol=0, atol=1e-8):
            raise ValueError("Replayed replan state differs from verified saved execution")
        return observation


def goal_variants(environment, observation):
    """Change only the goal-image pusher; physical goal and dynamics stay fixed."""
    native = environment.goal_renderer.env.unwrapped
    original = tuple(native.agent.position)
    variants = {"canonical": environment.goal_image.copy()}
    try:
        # Diagnostic counterfactuals, not alternative benchmark success criteria.
        positions = {"pusher-at-current": observation.state[:2], "pusher-opposite-corner": (448., 448.)}
        from shapely.geometry import Point
        from ..simulation.environments.green_pusht import polygon
        block = polygon(native.block, native.block.shapes)
        radius = next(iter(native.agent.shapes)).radius
        for name, position in positions.items():
            if block.intersects(Point(tuple(position)).buffer(radius)):
                continue  # Avoid occluding the goal block in a nuisance comparison.
            native.agent.position = tuple(position)
            native.space.reindex_shapes_for_body(native.agent)
            variants[name] = environment.goal_renderer.env.render().copy()
    finally:
        native.agent.position = original
        native.space.reindex_shapes_for_body(native.agent)
    return variants


def audit(args):
    from ..planning.cem import score_sequences, solve_cem
    from ..rollout.inference import fixed_predictor
    from ..simulation.dataset import ResetCases
    from ..simulation.environments import make_environment
    from ..simulation.history import ObservedHistory, preprocess_frame
    report_path = args.report.resolve()
    report = read_json(report_path)
    if report.get("split") != "validation" or not report.get("completed"):
        raise ValueError("Use a completed validation MPC report for failure diagnosis")
    manifest = checked(Path(report["config"]["experiment_dir"])/"manifest.json")
    rows = read_saved_records(report)
    selected = selections(report, manifest, rows, args.episodes, ["frozen"], ["gaussian"], args.seed)
    source = verify_replay_source(report)
    # The frozen prediction/action conversion being diagnosed must match the run.
    for name in ("codes/lewm_tta/planning/actions.py", "codes/lewm_tta/planning/cem.py",
                 "codes/lewm_tta/rollout/scopes.py"):
        if report["code"].get(name) != file_hash(ROOT/name):
            raise ValueError(f"Recorded inference implementation differs: {name}")
    if not report.get("calls_file"):
        raise ValueError("Diagnosis needs saved compact/full call records")
    path = Path(report["calls_file"])
    if not path.is_file() or file_hash(path) != report["calls_sha256"]:
        raise ValueError("Missing or changed MPC call records")
    with open_records(path) as stream:
        calls = [json.loads(line) for line in stream if line.strip()]
    if len(calls) != report["planning_calls"]:
        raise ValueError("Saved call count differs from report")
    config, frameskip = report["config"], manifest["frameskip"]
    if args.refine_population < config["planning"]["elites"]:
        raise ValueError("The larger-budget population must contain at least the configured elite count")
    adapter, coordinates, _ = load_planner(config, args.device, args.cpu_threads, "frozen", "gaussian")
    if adapter.identity != report["model"]:
        raise ValueError("Model or numerical packages differ from the saved evaluation")
    directory = output_directory(args.output, "green-audit")
    environment, dataset = make_environment(config["simulation"], ResetCases()), ResetCases()
    if environment.description() != report["environment"]:
        environment.close()
        raise ValueError("Simulator differs from the recorded evaluation")
    records, galleries = [], []
    try:
        for episode, case, row in selected:
            # Verify the entire original outcome before branching off its prefix.
            verified = replay_saved(environment, dataset, case, row, manifest, config)
            case_calls = sorted([call for call in calls if call["case_id"] == row["case_id"]
                                 and call["seed"] == row["seed"] and call["method"] == "frozen"
                                 and call["proposal_source"] == "gaussian"], key=lambda value: value["replan"])
            if [call["replan"] for call in case_calls] != list(range(row["planning_calls"])):
                raise ValueError("Saved frozen/Gaussian calls are incomplete or duplicated")
            if max(args.replans) >= len(case_calls):
                raise ValueError(f"Case {episode} has no saved replan {max(args.replans)}")
            previous_plan = None
            # Replay prior frozen planning calls too: reconstructing a normalized
            # warm start from rounded raw JSON can change near-tied CEM elites.
            for replan in range(max(args.replans)+1):
                call = case_calls[replan]
                start = call["simulator_raw_steps_before"]
                chunk = frameskip*config["simulation"]["execute_blocks"]
                if start != replan*chunk:
                    raise ValueError("Saved replan boundary differs from the MPC execution protocol")
                restore = RestoreContext(environment, dataset, case, row, manifest,
                                         row["executed_raw_actions"][:start], verified["states"][start])
                observed = restore()
                history = ObservedHistory(adapter, coordinates, restore.images, restore.blocks)
                context = history.context()
                images = (goal_variants(environment, observed) if replan in args.replans else
                          {"canonical": environment.goal_image.copy()})
                goals = {name: history.encode(image) for name, image in images.items()}
                if object_hash(goals["canonical"].cpu().tolist()) != call["goal_fingerprint"]:
                    raise ValueError("Re-encoded goal differs from the saved evaluation")
                adapter.reset(call["model_reset_seed"])
                controls = control_probes(call["selected_raw_actions"])
                warm = (None if previous_plan is None else
                        previous_plan[config["simulation"]["execute_blocks"]:])
                settings = config["planning"]
                with fixed_predictor(adapter):
                    scorer = lambda actions: score_sequences(adapter, context, goals["canonical"], actions, settings["scoring_batch_size"])
                    rerun = solve_cem(scorer, settings, coordinates, call["cem_seed"], warm)
                    saved = torch.as_tensor(controls["saved-cem"], device=adapter.device)
                    action_error = float((coordinates.to_raw(rerun.actions).flatten(0, 1)-saved).abs().max())
                    if (action_error > 1e-5 or abs(rerun.cost-call["model_goal_sse"]) > 1e-3
                            or rerun.first_population_fingerprint != call["base_population_fingerprint"]):
                        raise ValueError("Frozen CEM did not reproduce the saved plan/cost; inspect numerical environment first")
                    previous_plan = rerun.actions.detach()
                    if replan not in args.replans:
                        continue
                    goal_media = save_gallery(directory, f"validation-{episode:05d}-replan-{replan:03d}-goal-variants",
                                              [Panel(name, [image], [0], fixed=True) for name, image in images.items()],
                                              fps=environment.description()["control_hz"], scale=args.scale)
                    galleries.append((f"case {episode}, replan {replan}: goal-pusher counterfactuals", goal_media))
                    refined = solve_cem(scorer, {**settings, "iterations": args.refine_iterations,
                                              "population": args.refine_population}, coordinates, call["cem_seed"], warm)
                    controls["larger-budget-cem"] = coordinates.to_raw(refined.actions).flatten(0, 1).cpu().numpy()
                    native_error = native_rollout_parity(adapter, coordinates, context, restore.images,
                                                        history.transform, controls["saved-cem"])
                    comparisons = []
                    for name, actions in controls.items():
                        trace = execute_probe(environment, restore, actions, frameskip)
                        actual = adapter.encode(torch.stack([
                            # Same RGB preprocessing as the real history/cache.
                            preprocess_frame(image, history.transform)
                            for image in trace["keyframes"]])[None])
                        comparison = compare_prediction(adapter, coordinates, context, actions, actual, goals)
                        comparison.update({"control": name, "actions": actions.tolist(),
                                           "initial_metrics": trace["metrics"][0], "final_metrics": trace["metrics"][-1],
                                           "ever_out_of_frame": any(metric["out_of_frame"] for metric in trace["metrics"]),
                                           "block_center_displacement": float(np.linalg.norm(np.asarray(trace["states"][-1][2:4])-trace["states"][0][2:4])),
                                           "initial_state": trace["states"][0], "final_state": trace["states"][-1]})
                        stem = f"validation-{episode:05d}-replan-{replan:03d}-{name}"
                        media = save_gallery(directory, stem, [Panel(name+": actual outcome", trace["frames"],
                                             list(range(start, start+len(actions)+1))),
                                             Panel("Fixed green-target goal", [images["canonical"]], [0], fixed=True)],
                                             fps=environment.description()["control_hz"], every=args.every, scale=args.scale)
                        comparison["media"] = media
                        galleries.append((stem, media))
                        comparisons.append(comparison)
                record = {"episode": episode, "case_id": case["case_id"], "seed": row["seed"], "replan": replan,
                          "saved_outcome_verified": verified["verification"], "saved_raw_action_max_abs_error": action_error,
                          "native_rollout_max_abs_error": native_error, "source_call": call,
                          "reproduced_cem_trace": rerun.trace, "larger_budget_cem": {
                              "population": args.refine_population, "iterations": args.refine_iterations,
                              "candidate_evaluations": refined.candidate_evaluations, "trace": refined.trace},
                          "goal_pusher_only_change_sse": {name: float((goal-goals["canonical"]).square().sum()) for name, goal in goals.items()},
                          "goal_variants_media": goal_media,
                          "controls": comparisons}
                records.append(record)
                for value in comparisons:
                    costs = value["goal_sse"]["canonical"]
                    print(f"case={episode} replan={replan} {value['control']}: predicted/observed goal SSE="
                          f"{costs['predicted_terminal']:.2f}/{costs['observed_terminal']:.2f}, "
                          f"block displacement={value['block_center_displacement']:.2f}, out-of-frame={value['ever_out_of_frame']}")
    finally:
        environment.close()
        dataset.close()
    description = ("Frozen/Gaussian MPC diagnosis from identical saved validation contexts. "
                   "Compares saved and larger-budget CEM plans with raw-zero and signed-axis controls; each plan is "
                   "executed independently for the entire horizon, without replanning or early success stopping. "
                   "Future RGB/physical metrics are diagnostic labels only. Goal-pusher variants are counterfactual "
                   "cost measurements, never used to select plans. No TTT or retrieval is run. Not benchmark results.")
    index_page(directory, "Green MPC prediction/physics diagnostic", galleries, description)
    summary = ["# Frozen green MPC diagnostic", "", description, ""]
    for record in records:
        summary += [f"## Case {record['episode']}, replan {record['replan']}", "",
                    f"Native/adapter max absolute rollout difference: {record['native_rollout_max_abs_error']:.6g}.",
                    f"Saved-plan raw action reproduction error: {record['saved_raw_action_max_abs_error']:.6g}.", "",
                    "| Control | Predicted goal SSE | Observed goal SSE | First-block prediction MSE | Final-block rollout MSE | Final coverage | Leaves view |",
                    "|---|---:|---:|---:|---:|---:|---|"]
        for value in record["controls"]:
            cost = value["goal_sse"]["canonical"]
            summary.append(f"| {value['control']} | {cost['predicted_terminal']:.3f} | {cost['observed_terminal']:.3f} | "
                           f"{value['autoregressive_mse_by_block'][0]:.4f} | {value['autoregressive_mse_by_block'][-1]:.4f} | "
                           f"{value['final_metrics']['coverage']:.4f} | {value['ever_out_of_frame']} |")
        summary += ["", "Teacher-forced and persistence errors by depth, alternative goal costs, physical endpoints and CEM traces are in `audit.json`.", "",
                    "Goal latent SSE caused by changing only the pusher position:", ""]
        summary += [f"- {name}: {cost:.4f}" for name, cost in record["goal_pusher_only_change_sse"].items()]
        summary.append("")
    (directory/"summary.md").write_text("\n".join(summary))
    write_json(directory/"audit.json", {"report": str(report_path), "report_sha256": file_hash(report_path),
               "calls_sha256": report["calls_sha256"], "records_sha256": report["records_sha256"],
               "replay_source": source, "model": adapter.identity, "description": description,
               "audit_source": {str(path.relative_to(ROOT)): file_hash(path) for path in
                                (Path(__file__), Path(__file__).with_name("prediction_audit.py"))},
               "records": records, "benchmark_result": False})
    print(f"Green MPC diagnosis: {directory/'audit.json'}; gallery: {directory/'index.html'}")
    return directory

"""Small validation-only Frozen/Gaussian MPC previews with explicit settings.

Physics, scoring, observed history and execution reuse the simulator backend;
this diagnostic never writes benchmark cohorts, checks, reports or test locks.
"""

from pathlib import Path

from experiments.provenance import backend_identity
from ..config import ROOT
from ..runtime import file_hash, object_hash, write_json
from .episodes import output_directory, resolved_config
from .frozen_planning import FrozenGaussianController
from .green import replay_saved
from .media import Panel, index_page, save_gallery


def preview(args):
    from ..simulation.config import load_config, validate_config
    from ..simulation.dataset import ResetCases
    from ..simulation.environments import make_environment
    from ..simulation.execution import mpc
    from ..simulation.green import enabled, manifest_value
    config = resolved_config(args.config, load_config)
    if not enabled(config) or config["simulation"].get("split", "validation") != "validation":
        raise ValueError("Frozen green previews require a validation green-target profile")
    if args.goal_agent_position is not None:
        config["simulation"]["goal_agent_position"] = list(args.goal_agent_position)
    if args.execute_blocks is not None:
        config["simulation"]["execute_blocks"] = args.execute_blocks
    if args.max_raw_steps is not None:
        config["simulation"]["max_raw_steps"] = args.max_raw_steps
    validate_config(config)
    manifest = manifest_value(config)  # Read-only metadata; no preparation required.
    cases = manifest["splits"]["validation"]
    if (not args.episodes or len(set(args.episodes)) != len(args.episodes)
            or any(type(index) is not int or not 0 <= index < len(cases) for index in args.episodes)):
        raise ValueError("Choose distinct episode indices from the reserved validation suite")
    seed = config["simulation"]["seeds"][0] if args.seed is None else args.seed
    controller = FrozenGaussianController(config, args.device, args.cpu_threads, args.context_frames)
    directory = output_directory(args.output, "green-frozen-mpc")
    dataset, environment = ResetCases(), None
    records, galleries = [], []
    try:
        environment = make_environment(config["simulation"], dataset)
        for episode in args.episodes:
            case, calls = cases[episode], []
            result = mpc(environment, dataset, case, seed, "frozen", "gaussian", controller, manifest, config, calls)
            row = {"case_id": case["case_id"], "query": case["query"], "seed": seed,
                   "method": "frozen", "proposal_source": "gaussian", **result}
            trace = replay_saved(environment, dataset, case, row, manifest, config)
            stem = f"validation-{episode:05d}-frozen-gaussian-s{seed}"
            media = save_gallery(directory, stem, [
                Panel("Frozen/Gaussian: actual MPC", trace["frames"], list(range(len(trace["frames"]))),
                      details=[f"coverage={value['coverage']:.3f} success={int(value['success'])}" for value in trace["metrics"]]),
                Panel("Fixed green-target goal", [trace["goal_image"]], [0], fixed=True)],
                fps=environment.description()["control_hz"], every=args.every, scale=args.scale)
            record = {"episode": episode, "case": case, "seed": seed, "source_outcome": row, "calls": calls,
                      "observed_states": trace["states"], "observed_metrics": trace["metrics"],
                      "verification": trace["verification"], "replan_start_steps": trace["replan_start_steps"],
                      "media": media, "benchmark_result": False}
            write_json(directory/(stem+".json"), record)
            records.append(record)
            galleries.append((stem, media))
            print(f"case={episode} success={result['ever_success']} best coverage={result['best_metrics']['coverage']:.4f} "
                  f"out-of-frame={result['ever_out_of_frame']} steps={result['raw_steps']} replans={result['planning_calls']}", flush=True)
        native_description = environment.description()
    finally:
        if environment is not None:
            environment.close()
        dataset.close()
    adapter = controller.resources.adapter
    description = ("Fresh execution of Frozen/Gaussian MPC on prespecified validation resets. "
                   "No TTT, retrieval index, new data collection or benchmark preparation. "
                   "Controls use the selected planning budget/context and fixed rendered goal; actual outcome is verified by replay. "
                   "This green-marker task differs from LeWM's observed-future-goal benchmark. Preview only; not paper results.")
    index_page(directory, "Frozen/Gaussian green-target validation preview", galleries, description)
    write_json(directory/"preview.json", {"mode": "green-frozen-mpc-diagnostic", "config": config,
        "model": adapter.identity, "prediction_contract": adapter.prediction_contract(),
        "manifest_fingerprint": manifest["fingerprint"], "native_environment": native_description,
        "code": backend_identity("sim"), "preview_code": {str(path.resolve().relative_to(ROOT)): file_hash(path)
            for path in (Path(__file__), Path(__file__).with_name("frozen_planning.py"), Path(__file__).with_name("plans.py"),
                         Path(__file__).with_name("green.py"), Path(__file__).with_name("media.py"))},
        "resolved_settings_fingerprint": object_hash([config, adapter.prediction_contract()]),
        "records": records, "description": description, "benchmark_result": False})
    summary = ["# Frozen/Gaussian green-target diagnostic", "", description, "",
               f"CEM: N{config['planning']['population']}/J{config['planning']['iterations']}/elites{config['planning']['elites']}; "
               f"execute {config['simulation']['execute_blocks']} blocks per call; initial observed context {args.context_frames} frame(s).",
               f"Goal pusher: {config['simulation']['goal_agent_position']} in native scene coordinates; "
               f"max raw controls: {config['simulation']['max_raw_steps']}.", "",
               "| Case | Success | Peak coverage | Final coverage | Ever out of frame | Raw controls | CEM calls |",
               "|---|---|---:|---:|---|---:|---:|"]
    for record in records:
        value = record["source_outcome"]
        summary.append(f"| {record['episode']} | {value['ever_success']} | {value['best_metrics']['coverage']:.4f} | "
                       f"{value['final_metrics']['coverage']:.4f} | {value['ever_out_of_frame']} | {value['raw_steps']} | {value['planning_calls']} |")
    (directory/"summary.md").write_text("\n".join(summary)+"\n")
    print(f"Frozen MPC preview: {directory/'index.html'}; {directory/'summary.md'}")
    return directory

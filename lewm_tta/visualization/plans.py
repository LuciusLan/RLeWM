"""Render a selected CEM action sequence by executing it in the real simulator.

No decoder is supplied with this project: images are observed simulator
outcomes, never synthesized depictions of the model's imagined latent states.
"""

from copy import deepcopy
from pathlib import Path
import time

import numpy as np

from ..runtime import file_hash, object_hash, write_json
from .episodes import load_episode, output_directory, resolved_config
from .media import Panel, index_page, save_gallery


def physical_case(item):
    case_id = f"{item['case']['case_id']}:{item['start']}"
    episode_id = item["case"]["episode_id"]
    execution = item["metadata"].get("execution", item["case"])
    return {"case_id": case_id, "episode_reset_seed": execution["reset_seed"],
            "episode_start_raw_row": 0, "start_raw_row": item["start"],
            "current_raw_row": item["current"], "goal_raw_row": item["goal"],
            "query": {"transition_id": case_id+"/real-replan-0", "episode_id": episode_id,
                      "source_episode_position": -episode_id-1, "start_in_episode": item["start"],
                      "source": str(item["path"]), "dynamics_regime_id": item["regime"]}}


def environment_for(item):
    from ..simulation.dataset import SourceDataset
    from ..simulation.environments import make_environment
    dataset = SourceDataset(item["path"])
    dataset.fresh_manifest = {"shift": item["config"]["regimes"][item["regime"]],
                              "regime": item["regime"], "collection_manifest": item["manifest"]["fingerprint"]}
    try:
        environment = make_environment({"backend": item["config"]["backend"],
                                        "image_size": item["config"]["image_size"]}, dataset)
        current = environment.description()
        if any(current.get(key) != value for key, value in item["metadata"]["environment"].items()):
            environment.close()
            raise ValueError("Native environment/packages/shift differ from the collected episode; refusing to label a new environment as its reconstruction")
        return dataset, environment
    except BaseException:
        dataset.close()
        raise


def load_planner(config, device, cpu_threads, method, proposal):
    import torch
    from ..planning.actions import ActionCoordinates
    from ..rollout.resources import load_resources, source_metadata
    from ..rollout.scopes import ScopedLeWMAdapter
    torch.set_num_threads(cpu_threads)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(config["p1"]["data"]["split_seed"])
    # This single-episode diagnostic never encodes the full fresh query corpus.
    # The chosen episode and goal are read from --collection-config instead.
    memory_config = deepcopy(config)
    memory_config.pop("query_dataset", None)
    memory_config["proposals"]["sources"] = [source for source in config["proposals"]["sources"]
                                              if source in ("gaussian", "context-fragments")]
    if method == "frozen" and proposal == "gaussian":
        _, prepared = source_metadata(memory_config)
        p1 = config["p1"]
        adapter = ScopedLeWMAdapter(Path(p1["paths"]["repo"]), Path(p1["paths"]["checkpoint"]),
                                   device, config["scopes"][0])
        if adapter.fingerprint != prepared["model_fingerprint"]:
            raise ValueError("Checkpoint differs from the original memory-normalization assets")
        coordinates = ActionCoordinates(prepared["normalization"], config["planning"],
                                        p1["data"]["frameskip"], adapter.device)
        return adapter, coordinates, None
    from ..simulation.controller import Controller
    resources = load_resources(memory_config, device, cpu_threads)
    controller = Controller(resources, memory_config)
    return resources.adapter, controller.coordinates, controller


def plan_once(adapter, coordinates, controller, config, context, goal, metadata, method, proposal, seed,
              warm_start=None):
    from ..planning.actions import stream_seed
    from ..planning.cem import score_sequences, solve_cem
    from ..rollout.inference import fixed_predictor
    from ..rollout.support_updates import synchronize
    if controller is not None:
        return controller.plan(context, goal, metadata, method, proposal, seed, warm_start=warm_start)
    synchronize(adapter)
    started = time.perf_counter()
    reset_seed = stream_seed(seed, metadata["transition_id"], "model-reset")
    cem_seed = stream_seed(seed, metadata["transition_id"], "cem")
    adapter.reset(reset_seed)
    with fixed_predictor(adapter):
        plan = solve_cem(lambda actions: score_sequences(adapter, context, goal, actions,
                          config["planning"]["scoring_batch_size"]), config["planning"], coordinates, cem_seed,
                          initial_mean=warm_start)
    synchronize(adapter)
    raw = coordinates.to_raw(plan.actions).detach().cpu().numpy()
    return plan.actions.detach(), raw, {"method": "frozen", "proposal_source": "gaussian", "updates": 0,
        "model_reset_seed": reset_seed, "cem_seed": cem_seed, "model_goal_sse": plan.cost,
        "candidate_evaluations": plan.candidate_evaluations, "planning_seconds": time.perf_counter()-started,
        "first_population_fingerprint": plan.first_population_fingerprint,
        "selected_raw_actions": raw.reshape(-1, raw.shape[-1]).tolist(),
        "memory_index_loaded": False}


def metric_label(metrics):
    return f"success={int(metrics['success'])}  coverage={metrics['coverage']:.3f}"


def preview(args):
    from ..proposals.config import load_config
    from ..simulation.history import ObservedHistory, bootstrap
    config = resolved_config(args.config, load_config)
    if args.method not in config["adaptation"]["methods"] or args.proposal not in config["proposals"]["sources"]:
        raise ValueError("Requested method/proposal must be present in the planner profile")
    if not args.episodes or len(set(args.episodes)) != len(args.episodes) or min(args.episodes) < 0:
        raise ValueError("Choose distinct nonnegative episode indices")
    items = [load_episode(args.collection_config, args.split, index, args.regime,
                          args.start_step, args.window_index) for index in args.episodes]
    if any(item["config"]["backend"] != "pusht" for item in items):
        raise ValueError("CEM simulator preview currently supports PushT only; use collection previews to inspect the other tasks' stored frames")
    for item in items:
        collection = item["config"]
        if (collection["history"] != config["p1"]["data"]["history"]
                or collection["frameskip"] != config["p1"]["data"]["frameskip"]
                or config["planning"]["horizon"]*collection["frameskip"] != collection["goal_offset_steps"]):
            raise ValueError("Planner history/frame skip/horizon disagree with the selected collected goal")
    adapter, coordinates, controller = load_planner(config, args.device, args.cpu_threads, args.method, args.proposal)
    directory = output_directory(args.output, "cem")
    galleries, records = [], []
    for item in items:
        dataset, environment = environment_for(item)
        try:
            case = physical_case(item)
            history_size, frameskip = item["config"]["history"], item["config"]["frameskip"]
            images, blocks, observed = bootstrap(environment, dataset, case, args.seed, history_size, frameskip)
            reconstruction = environment.reconstruction(observed, case["current_raw_row"])
            history = ObservedHistory(adapter, coordinates, images, blocks)
            # Only past observations/actions and the explicitly supplied goal
            # reach CEM. Recorded future actions are never read here.
            goal_rgb = dataset.rows("pixels", case["goal_raw_row"])
            goal = history.encode(goal_rgb).detach()
            _, raw, call = plan_once(adapter, coordinates, controller, config, history.context(), goal,
                                      case["query"], args.method, args.proposal, args.seed)
            actions = raw.reshape(-1, raw.shape[-1])
            frames, metrics, states = [observed.rgb], [observed.metrics], [observed.state.tolist()]
            for action in actions:
                observed = environment.step(action)
                frames.append(observed.rgb)
                metrics.append(observed.metrics)
                states.append(observed.state.tolist())
                if observed.truncated:
                    raise ValueError("Simulator truncated during the selected full-horizon plan")
            # Evaluation-only/reference frames become visible after the plan.
            reference = list(dataset.rows("pixels", item["current"], item["goal"]+1))
            current, goal_step = item["current"], item["goal"]
            name = f"{args.split}-{item['case']['index']:05d}-{args.method}-{args.proposal}-s{args.seed}"
            media = save_gallery(directory, name, [
                Panel("CEM actions: actual outcome", frames, list(range(current, current+len(frames))),
                      details=[metric_label(value) for value in metrics]),
                Panel("Collected reference", reference, list(range(current, goal_step+1))),
                Panel("Supplied future goal", [goal_rgb], [goal_step], fixed=True)],
                fps=item["fps"], every=args.every, scale=args.scale)
            record = {"case": case, "collection_manifest": item["manifest"]["fingerprint"],
                "collection_episode": str(item["path"]), "episode_sha256": item["metadata"]["sha256"],
                "regime": item["regime"], "shift": item["config"]["regimes"][item["regime"]],
                "seed": args.seed, "method": args.method, "proposal": args.proposal,
                "planner_profile": str(Path(args.config).resolve()), "planning": config["planning"],
                "model": adapter.identity, "model_fingerprint": adapter.fingerprint,
                "goal_fingerprint": object_hash(goal.cpu().tolist()), "native_environment": environment.description(),
                "current_reconstruction": reconstruction, "call": call, "observed_metrics": metrics,
                "observed_states": states, "selected_raw_actions": actions.tolist(), "media": media,
                "benchmark_result": False, "visualization": "selected CEM plan executed in the simulator; no decoder or latent imagination images",
                "execution": "one complete H5 plan; no MPC replan, no success-based early stop; initially successful cases are labeled and still shown"}
            write_json(directory/(name+".json"), record)
            records.append(record)
            galleries.append((name, media))
        finally:
            environment.close()
            dataset.close()
    description = ("One selected CEM action sequence per episode, physically executed from the reconstructed current state. "
                   "The collected reference uses different actions and is not the target trajectory for CEM. "
                   "The separate goal image stays fixed. These previews are not benchmark results or decoded WM imagination.")
    index_page(directory, "CEM selected-plan validity previews", galleries, description)
    write_json(directory/"preview.json", {"mode": "cem-plan-physical-execution", "records": records,
                                          "description": description, "benchmark_result": False})
    print(f"CEM previews: {directory/'index.html'}")
    return directory

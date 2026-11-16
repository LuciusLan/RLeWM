"""Run the installed upstream evaluator on the released future-goal protocol.

The native mode bypasses our simulator adapter and CEM. The project mode
substitutes our bounded CEM/scorer in that same upstream evaluation loop.
Both bypass retrieval, H3 physical bootstrap and benchmark locks. Numerical
comparisons precede physics; source data and experiment assets are unchanged.
"""

import argparse
from pathlib import Path
import time

import numpy as np
import torch

from ..config import ROOT
from ..runtime import configure_logging, file_hash, object_hash, write_json


def select_rows(episode_ids, steps, count, seed, goal_offset=25):
    """Vectorized equivalent of le-wm/eval.py, including its last-row omission."""
    episodes, inverse = np.unique(episode_ids, return_inverse=True)
    lengths = np.zeros(len(episodes), dtype=np.int64)
    np.maximum.at(lengths, inverse, steps + 1)
    valid = np.flatnonzero(steps <= lengths[inverse] - goal_offset - 1)
    if len(valid) - 1 < count:
        raise ValueError("Not enough source rows for the requested reference cohort")
    chosen = np.random.default_rng(seed).choice(len(valid) - 1, count, replace=False)
    return np.sort(valid[chosen]), len(valid)


def image_transform():
    import stable_pretraining as spt
    from torchvision.transforms import v2
    # Exact transform in le-wm/eval.py, rather than a new image pipeline.
    return v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True),
                       v2.Normalize(**spt.data.dataset_stats.ImageNet), v2.Resize(224)])


def fit_processors(dataset):
    from sklearn.preprocessing import StandardScaler
    processors, statistics = {}, {}
    for name in ("action", "proprio", "state"):
        values = dataset.get_col_data(name)
        values = values[~np.isnan(values).any(axis=1)]
        scaler = StandardScaler().fit(values)
        processors[name] = scaler
        if name != "action":
            processors["goal_" + name] = scaler
        statistics[name] = {"mean": scaler.mean_.tolist(), "scale": scaler.scale_.tolist(),
                            "variance": scaler.var_.tolist(), "rows": int(scaler.n_samples_seen_)}
    return processors, statistics


def comparison(actual, expected, *, rtol=2e-4, atol=2e-4):
    difference = (actual - expected).abs()
    finite = bool(torch.isfinite(difference).all())
    return {"passed": finite and bool(torch.allclose(actual, expected, rtol=rtol, atol=atol)),
            "finite": finite, "maximum_absolute_error": float(difference.max()) if finite else None,
            "mean_absolute_error": float(difference.mean()) if finite else None, "rtol": rtol, "atol": atol}


@torch.no_grad()
def parity_checks(dataset, rows, episode_ids, steps, processors, policy, native_model, adapter, seed):
    """Compare native installed/repo pixel rollouts with our cached latent path.

    Same RGBs, normalization and candidate actions in every branch; no CEM,
    retrieval or simulator is used here. Check both one- and three-frame starts.
    """
    from utils import get_img_preprocessor
    from ..model import Context
    from .frozen_planning import FrozenRolloutView
    transform = get_img_preprocessor("pixels", "pixels", 224)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    results = []
    for history in (1, 3):
        eligible = [int(row) for row in rows if steps[row] >= (history - 1) * 5]
        if not eligible:
            raise ValueError(f"No reference case has enough past observations for H{history} parity")
        for row in eligible[:2]:
            current, episode = int(steps[row]), int(episode_ids[row])
            beginning = current - (history - 1) * 5
            clip = dataset.load_chunk(np.array([episode]), np.array([beginning]),
                                      np.array([current + 26]))[0]
            pixels = clip["pixels"][torch.arange(0, history * 5, 5)]
            goal_rgb = clip["pixels"][-1:]
            raw = {"pixels": pixels.permute(0, 2, 3, 1).numpy()[None],
                   "goal": goal_rgb.permute(0, 2, 3, 1).numpy()[None],
                   "action": np.zeros((1, history, 2), dtype=np.float32)}
            prepared = policy._prepare_info(raw)
            project_pixels = transform({"pixels": pixels.clone()})["pixels"][None]
            project_goal = transform({"pixels": goal_rgb.clone()})["pixels"][None]
            image_error = comparison(project_pixels, prepared["pixels"], rtol=1e-6, atol=1e-6)
            goal_error = comparison(project_goal, prepared["goal"], rtol=1e-6, atol=1e-6)
            actions = processors["action"].transform(clip["action"][:-1].numpy())
            blocks = torch.from_numpy(actions).float().reshape(1, history - 1 + 5, 10).to(adapter.device)
            past, future = blocks[:, :history - 1], blocks[:, history - 1:]
            candidates = torch.randn(1, 16, 5, 10, generator=generator).to(adapter.device)
            candidates[:, 0] = 0
            candidates[:, 1] = future  # Known recorded continuation is just a parity probe.
            complete = torch.cat((past[:, None].expand(-1, 16, -1, -1), candidates), dim=2)

            def pixel_rollout(model):
                info = {key: value.to(adapter.device)[:, None].expand(-1, 16, *value.shape[1:]).clone()
                        for key, value in prepared.items()}
                costs = model.get_cost(info, complete)
                return costs[0], info["predicted_emb"][0, :, history:]

            installed_cost, installed_rollout = pixel_rollout(native_model)
            repo_cost, repo_rollout = pixel_rollout(adapter.model)
            context = Context(adapter.encode(project_pixels), past)
            repeated = Context(context.latents.expand(16, -1, -1), past.expand(16, -1, -1))
            predicted = FrozenRolloutView(adapter, history).rollout(repeated, candidates[0])
            goal = adapter.encode(project_goal)[:, -1]
            cached_cost = (predicted[:, -1] - goal).square().sum(-1)
            checks = {"pixel_preprocessing": image_error, "goal_preprocessing": goal_error,
                      "installed_vs_repo_rollout": comparison(installed_rollout, repo_rollout),
                      "installed_vs_repo_cost": comparison(installed_cost, repo_cost),
                      "repo_vs_cached_rollout": comparison(repo_rollout, predicted),
                      "repo_vs_cached_cost": comparison(repo_cost, cached_cost)}
            results.append({"source_row": row, "episode": episode, "current_step": current,
                            "observed_history": history, "candidates": 16, "checks": checks,
                            "installed_ranking": installed_cost.argsort().cpu().tolist(),
                            "cached_ranking": cached_cost.argsort().cpu().tolist(),
                            "passed": all(value["passed"] for value in checks.values())})
    return {"passed": all(value["passed"] for value in results), "cases": results,
            "description": "Exact same actions/goals; recorded future actions are parity probes only, never planner inputs"}


def project_solver(adapter, statistics, seed):
    """Our bounded scorer/CEM inside the same upstream policy and physics loop."""
    from stable_worldmodel.solver import CEMSolver
    from ..model import Context
    from ..planning.actions import ActionCoordinates, stream_seed
    from ..planning.cem import score_sequences, solve_cem
    from .frozen_planning import FrozenRolloutView

    class ProjectSolver(CEMSolver):
        def __init__(self):
            super().__init__(adapter.model, device=str(adapter.device), seed=seed)
            settings = {"horizon": 5, "population": 300, "iterations": 30, "elites": 30,
                        "initial_std": 1., "minimum_std": 0.,
                        "raw_action_low": [-1, -1], "raw_action_high": [1, 1]}
            self.settings = settings
            self.coordinates = ActionCoordinates({"mean": statistics["action"]["mean"],
                                                   "std": statistics["action"]["scale"]},
                                                  settings, 5, adapter.device)
            self.calls = []

        @torch.no_grad()
        def solve(self, info_dict, init_action=None):
            actions, costs = [], []
            call = len(self.calls)
            view = FrozenRolloutView(adapter, 1)
            summaries = []
            for index in range(len(info_dict["pixels"])):
                pixels, goal_pixels = info_dict["pixels"][index:index+1], info_dict["goal"][index:index+1]
                context = Context(adapter.encode(pixels), torch.empty(1, 0, 10, device=adapter.device))
                goal = adapter.encode(goal_pixels)[:, -1]
                # Keep our per-call CPU stream and bound policy explicit; do not
                # imply equal random populations to native persistent CUDA RNG.
                plan_seed = stream_seed(seed, f"reference-call-{call}-case-{index}", "cem")
                plan = solve_cem(lambda candidates: score_sequences(view, context, goal, candidates, 64),
                                 self.settings, self.coordinates, plan_seed,
                                 initial_mean=None if init_action is None else init_action[index])
                actions.append(plan.actions.detach().cpu())
                costs.append(plan.cost)
                summaries.append({"active_case_index": index, "cem_seed": plan_seed,
                                  "returned_mean_sse": plan.cost, "trace": plan.trace})
            self.calls.append({"call": call, "cases": summaries})
            return {"actions": torch.stack(actions), "costs": costs}

    return ProjectSolver()


def parser():
    command = argparse.ArgumentParser(description=__doc__)
    command.add_argument("--samples", type=int, default=5, help="Run this prefix of the fixed reference cohort")
    command.add_argument("--cohort-size", type=int, default=50, help="Source eval.py uses 50; keep 50 for a comparable cohort")
    command.add_argument("--seed", type=int, default=42)
    command.add_argument("--dataset", type=Path,
                         default=Path("/opt/dlami/nvme/ubuntu/lewm_pusht/pusht_expert_train.h5"))
    command.add_argument("--checkpoint", type=Path, default=ROOT/"pretrained/lewm-pusht")
    command.add_argument("--repo", type=Path, default=ROOT/"le-wm")
    command.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    command.add_argument("--planner", choices=("native", "project"), default="native",
                         help="Project runs our bounded CEM/scorer in the same native future-goal protocol")
    command.add_argument("--cpu-threads", type=int, default=8)
    command.add_argument("--output", type=Path, required=True, help="New directory; never overwrite existing outputs")
    command.add_argument("--video", action="store_true", help="Save native actual/reference/goal panel MP4s")
    command.add_argument("--parity-only", action="store_true", help="Compare model paths without simulator execution")
    command.add_argument("--verbose", action="store_true")
    return command


def run(args):
    import inspect
    from importlib.metadata import version
    from ..collection.config import headless
    headless()
    import stable_worldmodel as swm
    from stable_worldmodel.solver import CEMSolver
    from ..model import LeWMAdapter
    from .episodes import output_directory
    if not 1 <= args.samples <= args.cohort_size or args.seed < 0 or args.cpu_threads < 1:
        raise ValueError("Require 1 <= samples <= cohort-size, nonnegative seed and positive cpu-threads")
    if args.output.exists():
        raise FileExistsError(f"Choose an unused output directory: {args.output}")
    for path in (args.dataset, args.checkpoint/"weights.pt", args.repo/"eval.py"):
        if not path.is_file():
            raise FileNotFoundError(path)
    torch.set_num_threads(args.cpu_threads)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(args.seed)
    dataset = swm.data.HDF5Dataset(path=args.dataset, keys_to_cache=["action", "proprio", "state"])
    world = None
    directory = output_directory(args.output, "lewm-reference")
    try:
        column = "episode_idx" if "episode_idx" in dataset.column_names else "ep_idx"
        episode_ids, steps = dataset.get_col_data(column), dataset.get_col_data("step_idx")
        if not np.array_equal(np.unique(episode_ids), np.arange(len(dataset.lengths))):
            raise ValueError("Native eval.py assumes contiguous zero-based episode IDs; dataset differs")
        cohort, eligible_count = select_rows(episode_ids, steps, args.cohort_size, args.seed)
        rows = cohort[:args.samples]
        processors, statistics = fit_processors(dataset)
        print("Loading original checkpoint targets and project model for H1/H3 numerical parity...", flush=True)
        native_model = swm.wm.utils.load_pretrained(str(args.checkpoint.resolve())).to(args.device).eval()
        native_model.requires_grad_(False)
        native_model.interpolate_pos_encoding = True
        adapter = LeWMAdapter(args.repo, args.checkpoint, args.device)
        adapter.model.requires_grad_(False)
        plan_config = swm.PlanConfig(horizon=5, receding_horizon=5, action_block=5)
        solver = CEMSolver(native_model, batch_size=1, num_samples=300, var_scale=1.,
                           n_steps=30, topk=30, device=args.device, seed=args.seed)
        policy = swm.policy.WorldModelPolicy(solver=solver, config=plan_config, process=processors,
                                            transform={"pixels": image_transform(), "goal": image_transform()})
        if plan_config.history_len != 1:
            raise ValueError("Installed native policy history default changed; re-audit before reference execution")
        from stable_worldmodel.world.env_pool import EnvPool
        from stable_worldmodel.wrapper import MegaWrapper
        sources = [Path(inspect.getfile(item)) for item in (type(native_model), type(native_model.predictor),
                   type(policy), type(solver), swm.World, EnvPool, MegaWrapper, swm.data.HDF5Dataset)]
        from stable_worldmodel.envs.pusht.env import PushT
        sources += [Path(inspect.getfile(PushT)), args.repo/"eval.py",
                    args.repo/"config/eval/pusht.yaml", args.repo/"config/eval/solver/cem.yaml",
                    Path(__file__), ROOT/"codes/lewm_reproduce.py",
                    Path(__file__).with_name("frozen_planning.py"),
                    ROOT/"codes/lewm_tta/planning/cem.py", ROOT/"codes/lewm_tta/planning/actions.py"]
        stat = args.dataset.stat()
        report = {"mode": "lewm-future-goal-reference", "benchmark_result": False,
                  "source": {"path": str(args.dataset.resolve()), "bytes": stat.st_size,
                             "mtime_ns": stat.st_mtime_ns, "full_dataset_hashed": False},
                  "model": adapter.identity, "code": {str(path): file_hash(path) for path in sources},
                  "simulation_packages": {name: version(name) for name in ("gymnasium", "pymunk", "pygame", "scikit-learn")},
                  "seed": args.seed, "planner": args.planner, "cohort_size": args.cohort_size, "eligible_rows": eligible_count,
                  "cohort_rows": cohort.tolist(), "evaluated_rows": rows.tolist(),
                  "cases": [{"row": int(row), "episode": int(episode_ids[row]), "step": int(steps[row])}
                            for row in rows], "normalization": statistics,
                  "protocol": {"goal_offset": 25, "control_budget": 50, "horizon": 5, "action_block": 5,
                               "receding_horizon": 5, "initial_observed_history": 1,
                               "population": 300, "iterations": 30, "elites": 30, "initial_std": 1.,
                               "action_clipping": args.planner == "project", "intermediate_subgoals": False,
                               "cem_rng": "native persistent device generator" if args.planner == "native"
                                          else "project per-call CPU generator and stream_seed",
                               "normalization_source": "all source rows, sklearn StandardScaler",
                               "success": "native eval_state: joint pusher/block position norm <20 and angle error <pi/9",
                               "initial_observation": "recorded source RGB, exactly as native World.evaluate",
                               "model_precision": "float32, TF32 disabled", "device": args.device},
                  "interpretation": "Source-corpus engineering reproduction; not held-out green-target performance. "
                                    "A prefix run changes the later native RNG stream relative to all 50 cases."}
        write_json(directory/"reference.json", report)
        parity = parity_checks(dataset, cohort, episode_ids, steps, processors, policy, native_model, adapter, args.seed)
        write_json(directory/"parity.json", parity)
        if not parity["passed"]:
            raise RuntimeError(f"Numerical parity failed; inspect {directory/'parity.json'} before running physics")
        print("H1/H3 preprocessing, native/repo rollouts and cached costs passed numerical parity.", flush=True)
        if not args.parity_only:
            if args.planner == "project":
                solver = project_solver(adapter, statistics, args.seed)
                policy.solver = solver
            world = swm.World("swm/PushT-v1", num_envs=args.samples, max_episode_steps=100, image_shape=(224, 224))
            world.set_policy(policy)
            print(f"Running {args.planner} CEM on native future-goal evaluation: {args.samples} cases, "
                  "N300/J30, at most 50 controls...", flush=True)
            started = time.perf_counter()
            metrics = world.evaluate(dataset=dataset, start_steps=steps[rows].tolist(), goal_offset=25,
                                     eval_budget=50, episodes_idx=episode_ids[rows].tolist(),
                                     callables=[{"method": "_set_state", "args": {"state": {"value": "state"}}},
                                                {"method": "_set_goal_state", "args": {"goal_state": {"value": "goal_state"}}}],
                                     video=directory/"videos" if args.video else None)
            report["metrics"] = {key: value.tolist() if isinstance(value, np.ndarray) else value
                                 for key, value in metrics.items()}
            report["evaluation_seconds"] = time.perf_counter() - started
            if args.planner == "project":
                write_json(directory/"cem-calls.json", solver.calls)
            print(f"{args.planner} future-goal success: {metrics['success_rate']:.1f}%", flush=True)
        report["parity_passed"] = True
        report["settings_fingerprint"] = object_hash([report["protocol"], report["cohort_rows"], statistics])
        write_json(directory/"reference.json", report)
        (directory/"summary.md").write_text("# LeWM native future-goal reference\n\n"
            + report["interpretation"] + "\n\nH1/H3 numerical parity: passed.\n\n"
            + ("Physics skipped (--parity-only).\n" if args.parity_only else
               f"{args.planner} success: {report['metrics']['success_rate']:.1f}% on {args.samples} cases.\n")
            + "\nSee reference.json for exact cases/settings and parity.json for numerical errors.\n")
        print(f"Reference report: {directory/'reference.json'}", flush=True)
        return directory
    finally:
        if world is not None:
            world.close()
        if dataset.h5_file is not None:
            dataset.h5_file.close()


def main():
    args = parser().parse_args()
    configure_logging(args.verbose)
    run(args)

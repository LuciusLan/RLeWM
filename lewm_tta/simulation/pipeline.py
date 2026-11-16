"""Small paired simulator replay and MPC experiments; no dataset-specific physics."""

from datetime import datetime, timezone
from pathlib import Path
import time

from experiments.progress import progress
from experiments.provenance import backend_identity
from ..runtime import file_hash, object_hash, read_json, write_json
from ..rollout.record_io import read_saved_records
from .checks import require_check
from .cohorts import load_manifest, selected_cases, saved_plans
from .dataset import source_dataset
from .environments import make_environment
from .execution import replay, mpc
from .recording import Writer
from .reporting import summarize, format_summary


def evaluate(config, mode, device, cpu_threads, run_name=None, validation_lock=None):
    if mode not in ("replay", "mpc"):
        raise ValueError("Simulator mode must be replay or mpc")
    fresh = bool(config.get("query_dataset"))
    from .green import enabled, split_name, locked_test_config
    green = enabled(config)
    split = split_name(config) if green else "validation"
    if green and split == "test":
        if validation_lock is None:
            raise ValueError("Green test evaluation requires --locked-config from completed validation")
        expected, verified = locked_test_config(validation_lock["path"])
        if expected != config or verified != validation_lock:
            raise ValueError("Green test protocol differs from its validation lock")
    if (fresh or green) and mode == "replay":
        raise ValueError("Fresh simulator evaluation uses MPC; there are no old saved source plans to replay")
    manifest = load_manifest(config)
    cases = selected_cases(config, manifest)
    seeds = config["simulation"]["seeds"]
    name = run_name or f"{mode}-{split}-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')}"
    if Path(name).name != name or name in (".", ".."):
        raise ValueError("run_name must be a simple directory name")
    output = Path(config["experiment_dir"])/"runs"/name
    if output.exists():
        raise FileExistsError("Run exists; use a new run name")
    dataset = source_dataset(config)
    environment = None
    try:
        environment = make_environment(config["simulation"], dataset)
        preflight = require_check(config, manifest, cases, environment)
        original_report = read_json(Path(manifest["source_report"])) if not (fresh or green) else None
        plans = saved_plans(config, manifest, cases, seeds) if mode == "replay" else None
        output.mkdir(parents=True, exist_ok=False)
        write_json(output/"config.json", config)
        write_json(output/"status.json", {"status": "loading", "mode": mode})
        controller = resources = None
        loading_started = time.perf_counter()
        if mode == "mpc":
            from ..rollout.resources import load_resources
            from .controller import Controller
            resources = load_resources(config, device, cpu_threads)
            controller = Controller(resources, config)
            if fresh:
                resources.queries("validation")  # Encode once during loading, outside paired episode timings.
        loading_seconds = time.perf_counter()-loading_started
        records, call_count, completed_cases = [], 0, []
        writer = Writer(output, "records", config["evaluation"]["records"])
        calls = Writer(output, "calls", config["evaluation"]["call_records"] if mode == "mpc" else "none")
        started = time.perf_counter()
        budget = config["simulation"]["runtime_budget_hours"]
        budget_exhausted = False
        with writer, calls, progress(cases, config=config, desc=f"Simulator {mode}", unit="episode") as bar:
            for case in bar:
                paired_bootstrap = {}
                for seed in seeds:
                    conditions = [(method, proposal) for method in config["adaptation"]["methods"]
                                  for proposal in config["proposals"]["sources"]]
                    if mode == "replay":
                        conditions += [("recorded-control", "recorded-actions")]
                    for method, proposal in conditions:
                        bar.set_postfix(seed=seed, model=method, proposal=proposal, refresh=False)
                        base = {"case_id": case["case_id"], "query": case["query"], "seed": seed,
                                "method": method, "proposal_source": proposal,
                                "current_raw_row": case["current_raw_row"], "goal_raw_row": case["goal_raw_row"]}
                        if mode == "replay":
                            source = None if method == "recorded-control" else plans[case["case_id"], seed, method, proposal]
                            actions = (dataset.rows("action", case["current_raw_row"], case["goal_raw_row"])
                                       if source is None else source["selected_raw_actions"])
                            result = replay(environment, case, seed, actions, recorded=source is None)
                            base.update(source_plan_fingerprint=None if source is None else object_hash(actions),
                                        source_model_goal_sse=None if source is None else source["selected_model_goal_sse"],
                                        source_frozen_goal_sse=None if source is None else source["selected_plan_frozen_model_goal_sse"])
                        else:
                            counter = _Calls(calls)
                            result = mpc(environment, dataset, case, seed, method, proposal, controller,
                                         manifest, config, counter)
                            call_count += counter.count
                            fingerprint = (result["bootstrap_fingerprint"], result["goal_fingerprint"])
                            if seed in paired_bootstrap and paired_bootstrap[seed] != fingerprint:
                                raise AssertionError("Paired MPC treatments received different observed starting histories or goals")
                            paired_bootstrap[seed] = fingerprint
                        row = {**base, **result}
                        records.append(row)
                        writer.append(row)
                    writer.flush()
                    calls.flush()
                completed_cases.append(case["case_id"])
                write_json(output/"status.json", {"status": "running", "mode": mode,
                    "completed_episodes": len(completed_cases), "requested_episodes": len(cases),
                    "records": len(records), "planning_calls": call_count})
                # Stop only between complete paired episodes, never mid-treatment.
                if budget is not None and time.perf_counter()-started >= budget*3600 and len(completed_cases) < len(cases):
                    budget_exhausted = True
                    break
        report = {"stage": "simulator-evaluation", "mode": mode, "split": split, "config": config,
                  "completed": not budget_exhausted, "budget_exhausted": budget_exhausted,
                  "limited_run": config["evaluation"]["query_limit"] is not None,
                  "manifest_fingerprint": manifest["fingerprint"], "case_ids": completed_cases,
                  "requested_cases": len(cases), "environment": environment.description(),
                  "preflight": preflight, "code": backend_identity("sim"),
                  "source_report": manifest.get("source_report"), "source_report_sha256": manifest.get("source_report_sha256"),
                  "source_records_sha256": manifest.get("source_records_sha256"),
                  "source_adaptation": original_report["config"]["adaptation"] if original_report else config["adaptation"],
                  "source_scopes": original_report["config"]["scopes"] if original_report else config["scopes"],
                  "query_source_fingerprint": manifest.get("query_source_fingerprint"),
                  "p1_prepared_fingerprint": manifest["p1_prepared_fingerprint"],
                  "model": resources.adapter.identity if resources else None,
                  "scope_parameters": resources.adapter.scope_description() if resources else None,
                  "records_file": str(writer.path.resolve()) if writer.path else None,
                  "records_sha256": file_hash(writer.path) if writer.path else None, "record_count": len(records),
                  "calls_file": str(calls.path.resolve()) if calls.path else None,
                  "calls_sha256": file_hash(calls.path) if calls.path else None, "planning_calls": call_count,
                  "record_storage": {"episode_records": config["evaluation"]["records"],
                                     "call_records": config["evaluation"]["call_records"] if mode == "mpc" else "none",
                                     "bytes": sum(path.stat().st_size for path in (writer.path, calls.path) if path)},
                  "resource_loading_seconds": loading_seconds, "evaluation_seconds": time.perf_counter()-started,
                  "protocol": {"replay": "execute every saved H*K raw action, including after early success; recorded-action control included",
                               "recorded_controls": "past warm-up and recorded-action control retain source commands unchanged, including values outside the nominal planner bounds; all planned actions retain configured bounds",
                               "mpc": f"simulate common recorded past for observed context; reset pretrained WM and fresh optimizer every real call; execute up to {config['simulation']['execute_blocks']} blocks, stop on success or raw-step budget",
                               "goal": "same supplied recorded goal throughout the episode; not a moving recorded-future goal",
                               "warm_start": "shift previous action plan and zero-pad in normalized coordinates; no parameter carry",
                               "pairing": "same initial episode/goal, simulator seed and per-call RNG schedule; later trajectories, supports and warm starts may diverge",
                               "bootstrap": "past recorded actions only; warm-up steps excluded from control budget and success scoring",
                               "timing": "planning seconds include reset/retrieval/TTT/CEM; context encoding separately charged; wall includes simulator/rendering and warm-up",
                               "incomplete_budget": "stop only between complete six-cell episodes; retain all completed episodes and label incomplete"},
                  "limits": ["Validation-only pilot conditional on one released checkpoint; not a fresh held-out test",
                             "Reconstructed physics state omits unrecorded velocities/contact state; inspect preflight recorded-action drift",
                             "Base SWM tasks, not an exact reproduction of JEPA-TTT custom tasks or shifts",
                             "Equal CEM candidate budgets, not matched wall-clock compute",
                             "No videos, per-support losses, per-candidate logs or latent tensors are stored"]}
        if fresh:
            report["limits"][0] = "Fresh validation simulator pilot, conditional on one released checkpoint; test episodes remain reserved"
            report["limits"][1] = "Complete original reset/past-prefix reconstruction; inspect CPU preflight state/RGB drift before interpretation"
            report["protocol"]["collection"] = "new simulator episodes; original external memory unchanged; goal frame sampled 25 raw actions ahead under a declared exploratory policy"
            report["protocol"]["goal_marker"] = "native decorative marker retained; task target is the separate observed future goal image/state, following LeWM"
        if green:
            report["task"] = "green-target"
            report["validation_lock"] = validation_lock
            report["protocol"].update({
                "goal": "fixed native green marker; RGB goal rendered in an independent environment with matching geometry and canonical pusher",
                "scoring": "actual block/marker intersection divided by marker area; coverage threshold defines success; pusher position unconstrained",
                "mpc": f"restore pretrained WM/fresh optimizer each call; execute {config['simulation']['execute_blocks']} blocks; stop for block-overlap success or {config['simulation']['max_raw_steps']} raw actions",
                "bootstrap": "common zero-relative controls generate actual history; warm-up excluded from score/control budget",
                "recorded_controls": "no reference trajectories, future recorded controls or collected query cache",
                "case_retention": "every reserved reset retained; initially successful and out-of-frame cases labeled, never replaced",
                "goal_image_bias": "canonical pusher pose affects latent CEM cost although physical success constrains only the block"})
            report["limits"][:2] = [
                "Conditional on one released checkpoint and original support memory; native reset suite is independent of recorded trajectories, not a proof against unknown pretraining exposure",
                "Option1 image-goal planning introduces a canonical-pusher preference; not a JEPA-TTT reward-head reproduction"]
        report["summary"] = summarize(records, config)
        write_json(output/"report.json", report)
        (output/"summary.md").write_text(format_summary(report))
        write_json(output/"status.json", {"status": "budget-stopped" if budget_exhausted else "completed",
                                         "episodes": len(completed_cases), "records": len(records)})
        print(format_summary(report), flush=True)
        print(f"Report: {output/'report.json'}", flush=True)
        return report
    except BaseException as error:
        if output.exists():
            write_json(output/"status.json", {"status": "failed", "error": f"{type(error).__name__}: {error}"})
        raise
    finally:
        if environment is not None:
            environment.close()
        dataset.close()


class _Calls:
    def __init__(self, writer):
        self.writer, self.count = writer, 0

    def append(self, row):
        self.writer.append(row)
        self.count += 1


def analyze(path, output=None):
    report = read_json(path)
    if report.get("stage") != "simulator-evaluation":
        raise ValueError("Not a simulator report")
    rows = read_saved_records(report)
    summary = summarize(rows, report["config"])
    if summary != report["summary"]:
        raise ValueError("Saved simulator records do not reproduce the report")
    text = format_summary(report)
    if output:
        Path(output).write_text(text)
    print(text)
    return summary

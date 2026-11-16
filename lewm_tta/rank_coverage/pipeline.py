"""User-run frozen prediction/retrieval; never calls adaptation or a planner."""

from __future__ import annotations

from datetime import datetime, timezone
import gzip
from itertools import islice
import json
from pathlib import Path
import time

import numpy as np

from experiments.progress import progress

from ..runtime import file_hash, object_hash, read_json, write_json
from .config import checked_json, code_identity, cohort_spec, source_metadata, validate_config
from .metrics import pair_record, validate_ledger
from .statistics import summarize, write_tables


def support_ledger(support, pairs, retriever, alpha: float, shared: bool) -> list[dict]:
    """Reconstruct exact original channel loss contributions, including overlap."""
    labels = {name for channels in support.channels.values() for name in channels}
    counts = {name: sum(name in channels for channels in support.channels.values()) for name in labels}
    if shared:
        masses = {name: support.details["streams"][name]["effective_loss_mass"] for name in labels}
    else:
        masses = {"forward": 1 - alpha, "reverse": alpha}
    result = []
    for index, weight in zip(support.indices, support.weights):
        origins = []
        for label in support.channels[index]:
            action, direction = (int(label.split("/")[0].split("-")[1]), label.split("/")[1]) if shared else (0, label)
            forward, reverse = pairs[action]
            origins.append({"action": action, "direction": direction,
                            "forward_rank": forward.rank_of(index), "reverse_rank": reverse.rank_of(index),
                            "loss_contribution": masses[label] / counts[label]})
        result.append({"index": int(index), "source_episode_position": int(retriever.episode_positions[index]),
                       "start_in_episode": int(retriever.step_positions[index]), "weight": float(weight), "origins": origins})
    return result


def measure_group(resources, query, mode: str, seed: int, value: dict, parity_check: bool) -> tuple[dict, dict]:
    import torch
    from ..model import Context
    from ..planning.actions import ActionCoordinates, provisional_actions, stream_seed
    from ..planning.shared_ttt import merge_streams, select_shared_support
    from ..retrieval import select_support
    from ..rollout.support_updates import synchronize, validate_support

    adapter, retriever = resources.adapter, resources.engine.retriever
    synchronize(adapter)
    started = time.perf_counter()
    shared = mode == "provisional-actions"
    reset_seed = stream_seed(seed, query.metadata["transition_id"], "model-reset") if shared else int(
        object_hash([seed, query.metadata["transition_id"]])[:8], 16)
    adapter.reset(reset_seed)
    coordinates = None
    if shared:
        coordinates = ActionCoordinates(resources.prepared["normalization"], value["provisional"],
                                        resources.manifest["data_config"]["frameskip"], adapter.device)
        actions = provisional_actions(coordinates, value["provisional"]["provisional_actions"],
            value["provisional"]["initial_std"], stream_seed(seed, query.metadata["transition_id"], "provisional"))
    else:
        actions = query.action
    repeated = Context(query.context.latents.expand(len(actions), -1, -1),
                       query.context.past_actions.expand(len(actions), -1, -1))
    with torch.no_grad():
        predictions = adapter.predict(repeated, actions).detach()
    synchronize(adapter)
    prediction_seconds = time.perf_counter() - started
    rank_started = time.perf_counter()
    pairs, measurements = [], []
    generator = torch.Generator(device="cpu").manual_seed(reset_seed)
    budget = value["support_size"]
    for position in range(len(actions)):
        forward = retriever.forward(query.context, actions[position:position + 1])
        reverse = retriever.reverse(query.context, predictions[position:position + 1])
        pairs.append((forward, reverse))
        consensus = select_support("rank-consensus", budget, retriever.settings["pool_size"], value["alpha"],
            forward, reverse, generator, episode_positions=retriever.episode_positions,
            max_per_episode=retriever.episode_cap, max_support=retriever.max_support,
            step_positions=retriever.step_positions, min_step_gap=retriever.min_step_gap)
        validate_support(resources.memory, retriever, consensus, budget)
        measurements.append(pair_record(forward.ranks, reverse.ranks,
            forward.indices[:max(value["prefix_sizes"])], reverse.indices[:max(value["prefix_sizes"])],
            consensus.indices, value["percentages"]))
        measurements[-1]["consensus_sources"] = [
            {"index": int(i), "source_episode_position": int(retriever.episode_positions[i]),
             "start_in_episode": int(retriever.step_positions[i])} for i in consensus.indices]
    synchronize(adapter)
    rank_seconds = time.perf_counter() - rank_started
    selection_started = time.perf_counter()
    if shared:
        streams = []
        for position, pair in enumerate(pairs):
            for direction, ranking, mass in (("forward", pair[0], 1 - value["alpha"]), ("reverse", pair[1], value["alpha"])):
                streams.append((f"action-{position}/{direction}",
                    islice(ranking.order, min(value["provisional"]["support_scan_limit"], len(ranking))), mass / len(actions)))
        support = merge_streams(streams, budget, retriever)
        support.details["scan_limit_per_stream"] = value["provisional"]["support_scan_limit"]
    else:
        forward, reverse = pairs[0]
        support = select_support("fr-fixed", budget, retriever.settings["pool_size"], value["alpha"], forward, reverse,
            generator, episode_positions=retriever.episode_positions, max_per_episode=retriever.episode_cap,
            max_support=retriever.max_support, step_positions=retriever.step_positions, min_step_gap=retriever.min_step_gap)
    validate_support(resources.memory, retriever, support, budget)
    ledger = support_ledger(support, pairs, retriever, value["alpha"], shared)
    validate_ledger(ledger, len(resources.memory))
    checks = {"existing_fr_selector_used": True, "query_episode_excluded": True,
              "existing_consensus_selector_used": True, "global_exact_ranks": True,
              "p3_shared_fr_parity": None}
    if shared and parity_check:
        reference, _, _ = select_shared_support(resources.engine, query.context, actions, "fr-fixed",
            stream_seed(seed, query.metadata["transition_id"], "support"), value["provisional"]["support_scan_limit"])
        if support.indices != reference.indices or support.channels != reference.channels or not np.allclose(
                support.weights, reference.weights, rtol=0, atol=1e-12):
            raise AssertionError("Pilot shared FR support differs from existing P3 selection")
        checks["p3_shared_fr_parity"] = True
    adapter.check_fixed_state()
    if float(adapter.anchor_loss().detach()) != 0 or any(parameter.grad is not None for parameter in adapter.parameters):
        raise AssertionError("Coverage pilot changed model parameters or created gradients")
    synchronize(adapter)
    record = {"query": query.metadata, "mode": mode, "seed": seed, "reset_seed": reset_seed,
              "pairs": measurements, "fr_fixed_support": ledger, "fr_selection": support.details,
              "actions_normalized": actions.detach().cpu().tolist(),
              "action_fingerprint": object_hash(actions.detach().cpu().tolist()),
              "timing_seconds": {"reset_and_frozen_prediction": prediction_seconds,
                                 "full_ranking_consensus_and_coverage": rank_seconds,
                                 "fr_selection_ledger_and_checks": time.perf_counter() - selection_started,
                                 "total": time.perf_counter() - started}}
    if coordinates is not None:
        record["action_coordinates"] = coordinates.description()
    return record, checks


def protocol(value: dict, manifest: dict) -> dict:
    return {"cohort": manifest["fingerprint"], "percentages": value["percentages"],
            "prefix_sizes": value["prefix_sizes"], "query_modes": value["query_modes"], "seeds": value["seeds"],
            "support_size": value["support_size"], "alpha": value["alpha"], "provisional": value["provisional"],
            "p1_retrieval": value["p1"]["retrieval"], "evaluation": value["evaluation"]}


def evaluate(value: dict, device: str, split: str, run_name: str | None, limit: int | None,
             validation_report: Path | None = None) -> dict:
    from ..frame_bank import FrameLatentBank
    from ..rollout.resources import load_resources

    validate_config(value)
    source, prepared = source_metadata(value)
    manifest = checked_json(Path(value["experiment_dir"]) / "manifest.json")
    if manifest["specification"] != cohort_spec(value, source, prepared):
        raise ValueError("Coverage cohort differs from source/config; run prepare in a new experiment directory")
    implementation = code_identity()
    if split == "test":
        if validation_report is None:
            raise ValueError("Test coverage requires --validation-report from the full matching validation pilot")
        validation = read_json(validation_report)
        if (not validation.get("completed") or validation["split"] != "validation" or validation.get("limited") or
                validation["protocol"] != protocol(value, manifest) or validation["code"] != implementation):
            raise ValueError("Validation report must cover the full identical protocol and implementation")
    name = run_name or f"{split}-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')}"
    if Path(name).name != name or name in (".", ".."):
        raise ValueError("Run name must be a simple directory name")
    output = Path(value["experiment_dir"]) / "runs" / name
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "config.json", value)
    write_json(output / "status.json", {"status": "loading", "split": split})
    print("Rank coverage: loading checked existing model, cache and retrieval assets...", flush=True)
    loading_started = time.perf_counter()
    resources = load_resources({"p1": value["p1"], "scopes": ["last-block"],
        "adaptation": {"support_size": value["support_size"], "alpha": value["alpha"]}, "evaluation": value["evaluation"]},
        device, value["evaluation"]["cpu_threads"])
    bank = resources.queries(split)
    starts = np.asarray(manifest["splits"][split]["raw_start_rows"], dtype=np.int64)
    if limit is not None:
        # Integration limits cover different source episodes before revisiting one.
        metadata = source["episode_metadata"]
        groups = {ep: [int(row) for row in starts if metadata["offsets"][ep] <= row < metadata["offsets"][ep] + metadata["lengths"][ep]]
                  for ep in source["episodes"][split]}
        starts = np.asarray([rows[position] for position in range(value["query_selection"]["windows_per_episode"])
                             for rows in groups.values() if position < len(rows)][:limit], dtype=np.int64)
    queries = FrameLatentBank(bank.store, bank.manifest, split, starts, bank.identity)
    episodes = sorted(set(int(ep) for ep in queries.episode_positions))
    write_json(output / "query-episodes.json", {"split": split, "query_episode_positions": episodes,
        "manifest_fingerprint": manifest["fingerprint"]})
    loading_seconds = time.perf_counter() - loading_started
    print(f"Rank coverage: {len(queries)} contexts / {len(episodes)} episodes; modes={value['query_modes']}; "
          "computing exact full-memory ranks, without adaptation or CEM", flush=True)
    records, checks = [], []
    records_path = output / "records.jsonl.gz"
    started = time.perf_counter()
    try:
        with gzip.open(records_path, "wt", encoding="utf-8") as stream, progress(
                range(len(queries)), config=value, desc=f"Rank coverage {split}") as query_progress:
            for index in query_progress:
                query = queries.query(index)
                if query.metadata["source_episode_position"] in resources.engine.retriever.episode_set:
                    raise ValueError("Query episode is present in retrieval memory")
                for mode in value["query_modes"]:
                    seeds = value["seeds"][:1] if mode == "recorded-action" else value["seeds"]
                    for seed in seeds:
                        record, check = measure_group(resources, query, mode, seed, value, parity_check=index == 0)
                        stream.write(json.dumps(record, allow_nan=False, separators=(",", ":")) + "\n")
                        records.append(record)
                        checks.append(check)
                done = index + 1
                if done == len(queries) or (value["evaluation"]["log_every"] and done % value["evaluation"]["log_every"] == 0):
                    stream.flush()
                query_progress.set_postfix(groups=len(records), refresh=False)
        expected = len(queries) * sum(1 if mode == "recorded-action" else len(value["seeds"]) for mode in value["query_modes"])
        if len(records) != expected:
            raise AssertionError("Coverage record matrix is incomplete")
        evaluation_seconds = time.perf_counter() - started
        from .analysis import validate_matrix
        matrix = validate_matrix(records, value, expected_queries=len(queries))
        summary = summarize(records, value)
        parity = [row["p3_shared_fr_parity"] for row in checks if row["p3_shared_fr_parity"] is not None]
        acceptance = {"all_global_ranks_checked": all(row["global_exact_ranks"] for row in checks),
                      "all_query_episodes_excluded": all(row["query_episode_excluded"] for row in checks),
                      "p3_shared_fr_parity": all(parity) if parity else None,
                      "p3_shared_fr_parity_groups": len(parity),
                      "model_parameters_unchanged": True, "adaptation_updates": 0, "cem_calls": 0,
                      "query_targets_scored": False, "complete_record_matrix": matrix["complete_record_matrix"],
                      "full_100_percent_baselines_verified": True}
        report = {"completed": True, "stage": "rank-coverage", "split": split, "limited": limit is not None,
                  "config": value, "protocol": protocol(value, manifest), "code": implementation,
                  "manifest_fingerprint": manifest["fingerprint"], "prepared_fingerprint": prepared["fingerprint"],
                  "model": resources.adapter.identity,
                  "retrieval": {"backend": value["p1"]["retrieval"]["backend"],
                                "calibration": prepared["calibration"],
                                "assets_fingerprint": prepared.get("retrieval_assets", {}).get("fingerprint"),
                                "memory_episodes": len(resources.engine.retriever.episode_set)},
                  "memory_windows": len(resources.memory), "query_episode_positions": episodes,
                  "records_file": str(records_path), "records_sha256": file_hash(records_path), "record_count": len(records),
                  "summary": summary, "checks": acceptance, "loading_seconds": loading_seconds,
                  "evaluation_seconds": evaluation_seconds,
                  "limits": ["Frozen, initial-state retrieval only; no temporal adaptation or FR-refresh trajectory is evaluated",
                             "FR gate statistics mask original supports without refill, weight renormalization or training",
                             "P3 pair consensus diagnostics do not define a shared multi-action consensus policy",
                             "Full-memory rank sorting/statistics time is not ordinary top-k planner latency",
                             "Pointwise exploratory intervals; windows, actions, seeds and corpus rows are not independent episode replicates"]}
        write_tables(summary, value, output)
        write_json(output / "acceptance-checks.json", acceptance)
        write_json(output / "report.json", report)
        write_json(output / "status.json", {"status": "complete", "split": split})
    except Exception as error:
        write_json(output / "status.json", {"status": "failed", "error": f"{type(error).__name__}: {error}"})
        raise
    print(f"Coverage report: {output / 'report.json'}", flush=True)
    return report

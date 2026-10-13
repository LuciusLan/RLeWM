"""User-run P0: a small frozen prediction baseline or a smoke check.

One window per sampled episode; compare LeWM with holding the last latent fixed.
No optimizer, adaptation, simulator, or CEM is launched.
Uses the native HDF5 loader, local LeWM classes, and local image preprocessor.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import inspect
import json
import logging
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
import warnings


ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def aggregate_results(records, bootstrap_samples, seed):
    """Paired statistics; each record comes from a distinct sampled episode."""
    import numpy as np

    episodes = [record["source"]["source_episode_position"] for record in records]
    if len(set(episodes)) != len(episodes):
        raise ValueError("P0 baseline expects one window per distinct episode")
    count = len(records)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, count, size=(bootstrap_samples, count)) if bootstrap_samples and count > 1 else None

    def statistics(values):
        values = np.asarray(values, dtype=np.float64)
        if not np.isfinite(values).all():
            raise ValueError("Nonfinite prediction metrics cannot be summarized")
        interval = None
        if draws is not None:
            interval = np.quantile(values[draws].mean(axis=1), [0.025, 0.975]).tolist()
        return {"mean": float(values.mean()), "median": float(np.median(values)),
                "std": float(values.std(ddof=1)) if count > 1 else 0.0,
                "p90": float(np.quantile(values, 0.90)), "mean_ci95": interval}

    def compare(lewm, persistence):
        lewm, persistence = np.asarray(lewm), np.asarray(persistence)
        return {"lewm": statistics(lewm), "hold_last_latent": statistics(persistence),
                "paired_delta_lewm_minus_hold_last": statistics(lewm - persistence),
                "lewm_win_fraction": float((lewm < persistence).mean())}

    one_step = compare([record["one_step_last_token_mse"] for record in records],
                       [record["one_step_hold_last_latent_mse"] for record in records])
    horizons = len(records[0]["autoregressive_mse_by_future_block"])
    rollouts = [{"horizon_blocks": horizon + 1, **compare(
        [record["autoregressive_mse_by_future_block"][horizon] for record in records],
        [record["hold_last_latent_mse_by_future_block"][horizon] for record in records],
    )} for horizon in range(horizons)]
    return {"metric": "mean squared projected-latent coordinate error",
            "windows": count, "episodes": len(set(episodes)),
            "confidence": {"method": "paired episode bootstrap of the mean",
                           "resamples": bootstrap_samples, "seed": seed},
            "one_step_last_token": one_step, "autoregressive": rollouts}


def format_summary(report, output):
    summary = report["summary"]
    label = "frozen prediction baseline" if report["mode"] == "baseline" else "smoke check"
    status = "completed" if report["mode"] == "baseline" else "passed"
    lines = [f"P0 {label} {status} | {report['samples']} windows / {summary['episodes']} episodes | "
             f"{report['device']} | seed {report['seed']}",
             "Metric           LeWM mean     Median        Mean 95% CI              Hold-last mean  LeWM wins"]
    metrics = [("One step", summary["one_step_last_token"])]
    metrics.extend((f"AR {row['horizon_blocks']} blocks", row) for row in summary["autoregressive"])
    for name, row in metrics:
        stats = row["lewm"]
        interval = stats["mean_ci95"]
        ci = f"[{interval[0]:.6g}, {interval[1]:.6g}]" if interval else "--"
        lines.append(f"{name:<16} {stats['mean']:<13.6g} {stats['median']:<13.6g} {ci:<24} "
                     f"{row['hold_last_latent']['mean']:<15.6g} {row['lewm_win_fraction']:>8.1%}")
    lines.extend(["Checks: checkpoint, alignment, finite outputs, repeat/reset, frozen state passed.",
                  f"Report: {output}"])
    return "\n".join(lines) + "\n"


def load_model(repo, checkpoint):
    from lewm_tta.model import load_native_model
    return load_native_model(repo, checkpoint)


def run(args):
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("WANDB_MODE", "disabled")
    os.environ["LOGURU_LEVEL"] = "INFO" if args.verbose else "WARNING"
    os.environ["TRANSFORMERS_VERBOSITY"] = "info" if args.verbose else "warning"
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING)
    if not args.verbose:
        warnings.filterwarnings("ignore", message=r".*torch\.jit\.script.*deprecated.*", category=FutureWarning)
    import h5py
    import hdf5plugin  # noqa: F401
    import numpy as np
    import stable_pretraining as spt
    import stable_worldmodel as swm
    import torch

    repo = args.repo.resolve(strict=True)
    checkpoint = args.checkpoint.resolve(strict=True)
    dataset_path = args.dataset.resolve(strict=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.set_num_threads(args.cpu_threads)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false; use --device cpu")
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available()
                          else "cpu" if args.device == "auto" else args.device)
    model, checkpoint_state, cfg = load_model(repo, checkpoint)
    from utils import get_column_normalizer, get_img_preprocessor

    model.to(device)
    before = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    history = cfg["predictor"]["num_frames"]
    image_size = cfg["encoder"]["image_size"]
    latent_dim = cfg["projector"]["output_dim"]
    frameskip = args.frameskip
    rollout_steps = args.rollout_steps
    num_steps = history + rollout_steps
    span = num_steps * frameskip
    with h5py.File(dataset_path, "r", swmr=True) as raw:
        required = {"pixels", "action", "ep_len", "ep_offset"}
        if not required.issubset(raw.keys()):
            raise ValueError(f"Expected native HDF5 columns {required}; found {list(raw.keys())}")
        schema = {}
        def describe(name, obj):
            if isinstance(obj, h5py.Dataset):
                schema[name] = {"shape": list(obj.shape), "dtype": str(obj.dtype),
                                "chunks": obj.chunks, "compression": obj.compression}
        raw.visititems(describe)
        if args.verbose:
            print("Dataset schema:\n" + json.dumps(schema, indent=2), flush=True)
        rows = raw["pixels"].shape[0]
        lengths = np.asarray(raw["ep_len"][:], dtype=np.int64).reshape(-1)
        offsets = np.asarray(raw["ep_offset"][:], dtype=np.int64).reshape(-1)
        if len(lengths) != len(offsets) or np.any(lengths <= 0) or np.any(offsets < 0):
            raise ValueError("Invalid episode lengths/offsets")
        if np.any(offsets + lengths > rows) or np.any(offsets[1:] < offsets[:-1] + lengths[:-1]):
            raise ValueError("Episode boundaries overlap or exceed stored rows")
        if raw["action"].ndim != 2 or raw["action"].shape[0] != rows:
            raise ValueError("Expected one action vector per image row")
        if frameskip * raw["action"].shape[1] != cfg["action_encoder"]["input_dim"]:
            raise ValueError("Raw action dimension times frameskip does not match checkpoint")
        eligible = np.flatnonzero(lengths >= span)
        if len(eligible) < args.samples:
            raise ValueError(f"Only {len(eligible)} episodes have {span} complete raw rows")
        rng = np.random.default_rng(args.seed)
        selected = np.sort(rng.choice(eligible, size=args.samples, replace=False))
        windows = []
        for ep in selected:
            local_start = int(rng.integers(0, lengths[ep] - span + 1))
            windows.append({"source_episode_position": int(ep), "source_episode_length": int(lengths[ep]),
                            "episode_local_start": local_start, "source_raw_start": int(offsets[ep]) + local_start,
                            "raw_rows": span})
        dataset_stat = dataset_path.stat()
        manifest = {"purpose": "P0 inspected episodes; exclude these from future untouched test manifests",
                    "source": {"path": str(dataset_path), "bytes": dataset_stat.st_size, "mtime_ns": dataset_stat.st_mtime_ns},
                    "source_schema": schema, "source_episodes": len(lengths), "source_rows": rows,
                    "episode_lengths": {"min": int(lengths.min()), "median": float(np.median(lengths)), "max": int(lengths.max())},
                    "frameskip": frameskip, "history": history, "rollout_steps": rollout_steps, "windows": windows}
    print(f"P0 {args.mode}: {args.samples} distinct episodes, {rollout_steps} rollout blocks, "
          f"device={device}, seed={args.seed}", flush=True)

    loader = getattr(swm.data, "HDF5Dataset", None)
    if loader is None or "path" not in inspect.signature(loader).parameters:
        raise RuntimeError("Install codes/p0-requirements.txt for the audited native HDF5Dataset(path=...) API")
    dataset = loader(path=dataset_path, frameskip=frameskip, num_steps=num_steps,
                     keys_to_load=["pixels", "action"], keys_to_cache=["action"])
    requested = {(window["source_episode_position"], window["episode_local_start"]) for window in windows}
    index_lookup = {clip: index for index, clip in enumerate(dataset.clip_indices) if clip in requested}
    if set(index_lookup) != requested:
        raise AssertionError("Native loader did not expose every selected episode-safe window")
    indices = [index_lookup[(window["source_episode_position"], window["episode_local_start"])] for window in windows]

    # Audit raw sampling and action grouping before applying any transforms.
    # Match the native loader's SWMR read flag when both handles are open.
    with h5py.File(dataset_path, "r", swmr=True) as raw:
        for index in range(args.samples):
            clip = dataset[indices[index]]
            start = windows[index]["source_raw_start"]
            rows = np.arange(num_steps) * frameskip + start
            pixels = raw["pixels"][rows]
            if pixels.ndim == 4 and pixels.shape[-1] in (1, 3):
                pixels = pixels.transpose(0, 3, 1, 2)
            expected_actions = raw["action"][start:start + num_steps * frameskip].reshape(num_steps, -1)
            torch.testing.assert_close(clip["pixels"], torch.from_numpy(pixels.copy()), rtol=0, atol=0)
            torch.testing.assert_close(clip["action"], torch.from_numpy(expected_actions), rtol=0, atol=0, equal_nan=True)
            for key in ("episode_idx", "ep_idx"):
                if key in raw and len(np.unique(raw[key][start:start + num_steps * frameskip])) != 1:
                    raise AssertionError(f"{key} crosses an episode boundary")
            if "step_idx" in raw:
                steps = np.asarray(raw["step_idx"][start:start + num_steps * frameskip]).reshape(-1)
                if not np.all(np.diff(steps) == 1):
                    raise AssertionError("Raw step indices are not consecutive")
            if not torch.isfinite(clip["action"]).all():
                raise AssertionError("Selected window contains nonfinite recorded actions")

    original_actions = torch.from_numpy(np.array(dataset.get_col_data("action")))
    valid = ~torch.isnan(original_actions).any(dim=1)
    original_actions = original_actions[valid]
    if len(original_actions) < 2 or not torch.isfinite(original_actions).all():
        raise AssertionError("Full-source action statistics are not finite")
    mean, std = original_actions.mean(0), original_actions.std(0)
    if not torch.isfinite(std).all() or not (std > 0).all():
        raise AssertionError("Full-source action standard deviation is invalid")
    manifest["action_normalization"] = {"mean": mean.tolist(), "std": std.tolist(),
                                         "std_correction": 1, "valid_rows": len(original_actions),
                                         "nan_rows": int((~valid).sum()),
                                         "source": "native get_column_normalizer on all released source actions"}
    dataset.transform = spt.data.transforms.Compose(
        get_img_preprocessor(source="pixels", target="pixels", img_size=image_size),
        get_column_normalizer(dataset, source="action", target="action"),
    )
    records = []
    first_prediction = None
    started = time.monotonic()
    try:
        with torch.inference_mode():
            for index in range(args.samples):
                sample = dataset[indices[index]]
                pixels = sample["pixels"].unsqueeze(0).to(device)
                actions = torch.nan_to_num(sample["action"], nan=0.0).unsqueeze(0).to(device)
                if pixels.shape != (1, num_steps, 3, image_size, image_size):
                    raise AssertionError(f"Unexpected preprocessed pixels: {pixels.shape}")
                if not torch.isfinite(pixels).all() or not torch.isfinite(actions).all():
                    raise AssertionError("Nonfinite preprocessed inputs")
                encoded = model.encode({"pixels": pixels, "action": actions})
                emb, act_emb = encoded["emb"], encoded["act_emb"]
                if emb.shape != (1, num_steps, latent_dim) or not torch.isfinite(emb).all():
                    raise AssertionError(f"Projected latent shape/finiteness failure: {emb.shape}")
                prediction = model.predict(emb[:, :history], act_emb[:, :history])
                repeat = model.predict(emb[:, :history], act_emb[:, :history])
                torch.testing.assert_close(prediction, repeat, rtol=1e-5, atol=1e-6)
                target = emb[:, 1:history + 1]
                if prediction.shape != target.shape or not torch.isfinite(prediction).all():
                    raise AssertionError("One-step prediction shape/finiteness failure")
                one_step_mse = (prediction - target).square().mean().item()
                last_token_mse = (prediction[:, -1] - target[:, -1]).square().mean().item()
                action_sequence = actions[:, :num_steps - 1].unsqueeze(1)
                rollout = model.rollout({"pixels": pixels[:, :history].unsqueeze(1)},
                                        action_sequence, history_size=history)["predicted_emb"]
                if rollout.shape != (1, 1, num_steps, latent_dim) or not torch.isfinite(rollout).all():
                    raise AssertionError("Autoregressive rollout shape/finiteness failure")
                torch.testing.assert_close(rollout[:, 0, history], prediction[:, -1], rtol=1e-5, atol=1e-6)
                rollout_mse = (rollout[:, 0, history:] - emb[:, history:]).square().mean(-1)[0].tolist()
                hold_last = emb[:, history - 1:history]
                persistence_mse = (hold_last - emb[:, history:]).square().mean(-1)[0].tolist()
                candidates = action_sequence.expand(-1, 2, -1, -1).clone()
                cost = model.get_cost({
                    "pixels": pixels[:, :history].unsqueeze(1).expand(-1, 2, -1, -1, -1, -1),
                    "goal": pixels[:, -1:].unsqueeze(1).expand(-1, 2, -1, -1, -1, -1),
                    "action": candidates,
                }, candidates)
                if cost.shape != (1, 2) or not torch.isfinite(cost).all():
                    raise AssertionError("Goal-cost API shape/finiteness failure")
                torch.testing.assert_close(cost[:, 0], cost[:, 1], rtol=1e-5, atol=1e-6)
                torch.testing.assert_close(cost[0, 0], torch.tensor(rollout_mse[-1] * latent_dim, device=device),
                                           rtol=1e-4, atol=1e-5)
                if index == 0:
                    first_prediction = prediction.detach().cpu().clone()
                record = {"sample": index, "source": manifest["windows"][index],
                          "source_observation_rows": [manifest["windows"][index]["source_raw_start"] + t * frameskip for t in range(num_steps)],
                          "pixel_shape": list(pixels.shape), "action_shape": list(actions.shape),
                          "latent_shape": list(emb.shape), "one_step_all_tokens_mse": one_step_mse,
                          "one_step_last_token_mse": last_token_mse,
                          "one_step_hold_last_latent_mse": persistence_mse[0],
                          "hold_last_latent_mse_by_future_block": persistence_mse,
                          "autoregressive_mse_by_future_block": rollout_mse, "goal_cost": cost[0].tolist()}
                records.append(record)
                if args.verbose:
                    print(json.dumps(record), flush=True)
                elif args.log_every and ((index + 1) % args.log_every == 0 or index + 1 == args.samples):
                    print(f"  evaluated {index + 1}/{args.samples} windows", flush=True)

            # Check before reset so reset cannot conceal changed running statistics.
            for name, value in model.state_dict().items():
                torch.testing.assert_close(value.cpu(), before[name], rtol=0, atol=0)
            # Restore checkpoint and repeat the first query after other samples.
            model.load_state_dict(checkpoint_state, strict=True)
            sample = dataset[indices[0]]
            encoded = model.encode({"pixels": sample["pixels"].unsqueeze(0).to(device),
                                    "action": torch.nan_to_num(sample["action"], nan=0.0).unsqueeze(0).to(device)})
            reset_prediction = model.predict(encoded["emb"][:, :history], encoded["act_emb"][:, :history])
            torch.testing.assert_close(reset_prediction.cpu(), first_prediction, rtol=1e-5, atol=1e-6)
        for name, value in model.state_dict().items():
            torch.testing.assert_close(value.cpu(), before[name], rtol=0, atol=0)
        if any(module.training for module in model.modules()) or any(p.grad is not None for p in model.parameters()):
            raise AssertionError("Model left evaluation-only behavior")
    finally:
        if getattr(dataset, "h5_file", None) is not None:
            dataset.h5_file.close()

    package_versions = {}
    for name in ("torch", "torchvision", "transformers", "stable-worldmodel", "stable-pretraining", "h5py", "numpy"):
        package_versions[name] = importlib.metadata.version(name)
    commit = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"], check=True,
                            capture_output=True, text=True).stdout.strip()
    summary = aggregate_results(records, args.bootstrap_samples, args.seed + 1)
    report = {"status": "passed", "mode": args.mode,
              "scope": "Small P0 frozen prediction baseline" if args.mode == "baseline" else "P0 small-sample sanity check",
              "methods": ["frozen_lewm", "hold_last_observed_projected_latent"],
              "pretraining_holdout_verified": False,
              "config": {"frameskip": frameskip, "history": history, "rollout_steps": rollout_steps,
                         "samples": args.samples, "bootstrap_samples": args.bootstrap_samples},
              "seed": args.seed, "device": str(device), "samples": args.samples,
              "elapsed_seconds": time.monotonic() - started, "python": sys.executable,
              "packages": package_versions, "lewm_commit": commit,
              "checkpoint": str(checkpoint), "checkpoint_sha256": sha256(checkpoint / "weights.pt"),
              "config_sha256": sha256(checkpoint / "config.json"),
              "code_sha256": {name: sha256(repo / name) for name in ("jepa.py", "module.py", "utils.py")},
              "evaluator_sha256": sha256(Path(__file__).resolve()),
              "dataset": str(dataset_path), "dataset_manifest": manifest,
              "p1_adaptable_module": f"predictor.transformer.layers.{len(model.predictor.transformer.layers) - 1}",
              "checks": ["strict checkpoint load", "raw/native temporal and action alignment", "episode boundaries",
                         "finite one-step and autoregressive outputs", "native goal-cost API",
                         "repeat determinism", "checkpoint-reset repeat", "unchanged parameters and buffers"],
              "summary": summary, "records": records}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    text_summary = format_summary(report, args.output)
    summary_path = args.output.with_name(args.output.stem + "-summary.txt")
    summary_path.write_text(text_summary)
    print(text_summary, end="", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("baseline", "smoke"), default="baseline")
    parser.add_argument("--repo", type=Path, default=ROOT / "le-wm")
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "pretrained/lewm-pusht")
    parser.add_argument("--dataset", type=Path, default=Path("/opt/dlami/nvme/ubuntu/lewm_pusht/pusht_expert_train.h5"))
    parser.add_argument("--output", type=Path, help="JSON report; defaults to artifacts/p0/<mode>-report.json")
    parser.add_argument("--samples", type=int, help="Distinct episodes; defaults to 30 for baseline or 3 for smoke")
    parser.add_argument("--frameskip", type=int, default=5)
    parser.add_argument("--rollout-steps", type=int, default=3)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--bootstrap-samples", type=int, help="Episode bootstrap resamples; baseline defaults to 1000, smoke to 0")
    parser.add_argument("--log-every", type=int, default=10, help="Progress interval in windows; 0 disables progress")
    parser.add_argument("--verbose", action="store_true", help="Print schema, per-window records, library info, and full errors")
    args = parser.parse_args()
    if args.samples is None:
        args.samples = 30 if args.mode == "baseline" else 3
    if args.bootstrap_samples is None:
        args.bootstrap_samples = 1000 if args.mode == "baseline" else 0
    if args.output is None:
        args.output = ROOT / f"artifacts/p0/{args.mode}-report.json"
    if min(args.samples, args.cpu_threads, args.frameskip, args.rollout_steps) < 1:
        parser.error("Sample counts, threads, and temporal lengths must be positive")
    if min(args.bootstrap_samples, args.log_every, args.seed) < 0:
        parser.error("Bootstrap samples, progress interval, and seed must be nonnegative")
    try:
        run(args)
    except KeyboardInterrupt:
        print("P0 interrupted.", file=sys.stderr)
        raise SystemExit(130)
    except Exception as error:
        if args.verbose:
            traceback.print_exc()
        else:
            print(f"P0 failed: {type(error).__name__}: {error}", file=sys.stderr)
            print("Rerun with --verbose for the full traceback.", file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()

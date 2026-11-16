"""CPU-only resumable synthesis, preview and exact recorded-label audits."""

from contextlib import contextmanager
from copy import deepcopy
import fcntl
from pathlib import Path

import h5py
import numpy as np
from tqdm.auto import tqdm

from ..collection.manifest import checked, sealed
from ..runtime import file_hash, write_json
from .source import load_parent, shard_paths
from .storage import array_hash, committed_episode, export


def make_renderer(backend, settings):
    if backend == "pusht":
        from .pusht import PushTRenderer
        return PushTRenderer(224, settings)
    from .reacher import ReacherRenderer
    return ReacherRenderer(224, settings)


@contextmanager
def exclusive(directory):
    directory.mkdir(parents=True, exist_ok=True)
    with (directory/"synthesis.lock").open("a") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("Another appearance job owns this output directory") from error
        yield


def prepare(directory, recipe):
    path = directory/"manifest.json"
    if path.exists():
        if checked(path) != recipe:
            raise ValueError("Appearance settings, parent, selection or implementation changed; use a new output directory")
    else:
        if any(path.name != "synthesis.lock" for path in directory.iterdir()):
            raise ValueError("New appearance output directory must be empty")
        write_json(path, recipe)
    rows = sum(ep["rows"] for ep in recipe["episodes"])
    print(f"{len(recipe['episodes']):,} episodes / {rows:,} RGB224 frames; "
          f"{rows*224*224*3/1024**3:.2f} GiB uncompressed RGB; compressed size depends on texture.")
    print(f"Recipe: {path}")


def synthesize(directory, recipe, parent, compression, min_free_gib, workers=1, batch_size=16):
    from .compatibility import record_execution, validate_execution
    from .source import implementation
    from .workers import run
    with exclusive(directory):
        prepare(directory, recipe)
        execution = record_execution(directory, recipe, implementation(), workers, batch_size)
        remaining, done_frames, done_episodes, seen_executions = [], 0, 0, set()
        for ep in tqdm(recipe["episodes"], desc="Verify committed episodes", unit="episode"):
            receipt = committed_episode(directory, recipe, ep)
            if receipt is None:
                remaining.append(ep)
                continue
            identity = receipt.get("execution")
            if identity not in seen_executions:
                validate_execution(directory, recipe, receipt)
                seen_executions.add(identity)
            done_frames += ep["rows"]
            done_episodes += 1
        total_frames = sum(ep["rows"] for ep in recipe["episodes"])
        print(f"Committed: {done_episodes:,} episodes / {done_frames:,} frames; "
              f"remaining: {total_frames-done_frames:,} frames. Workers: {workers}; write batch: {batch_size}.")
        with tqdm(total=total_frames, initial=done_frames, desc=f"Render {recipe['backend']}",
                  unit="frame", unit_scale=True) as bar:
            def completed(ep):
                nonlocal done_episodes
                done_episodes += 1
                bar.update(ep["rows"])
                bar.set_postfix(episodes=f"{done_episodes}/{len(recipe['episodes'])}", refresh=False)
            if remaining:
                run(remaining, min(workers, len(remaining)),
                    (directory, recipe, parent["source"]["path"], compression, batch_size, execution),
                    min_free_gib, completed)
        _, path = export(directory, recipe, parent)
        print(f"Exported: {path}")
    check(directory)


def verify_export(directory, value, recipe):
    if value["appearance"]["fingerprint"] != recipe["fingerprint"]:
        raise ValueError("Appearance export belongs to another recipe")
    target = directory/"dataset.h5"
    if str(target.resolve()) != value["source"]["path"] or file_hash(target) != value["dataset_sha256"]:
        raise ValueError("Appearance dataset view changed")
    stat = target.stat()
    if stat.st_size != value["source"]["bytes"] or stat.st_mtime_ns != value["source"]["mtime_ns"]:
        raise ValueError("Appearance dataset stat changed")
    for ep in value["episode_files"]:
        if file_hash(Path(ep["path"])) != ep["sha256"]:
            raise ValueError("Appearance episode payload changed")


def validate_query_lineage(value):
    """Used before the existing P2/P3 query-bank loader accepts an appearance bank."""
    appearance = value["appearance"]
    recipe_path = Path(appearance["recipe"])
    recipe = checked(recipe_path)
    parent, _, dependencies = load_parent(appearance["parent_manifest"], recipe["backend"])
    if (not parent.get("external_queries") or parent["fingerprint"] != appearance["parent_fingerprint"]
            or dependencies != recipe["dependencies"]):
        raise ValueError("Shifted queries must inherit an unchanged, independently collected query cohort")
    verify_export(recipe_path.parent, value, recipe)
    audit = checked(recipe_path.parent/"audit.json")
    parent_audit = checked(Path(parent["source"]["path"]).with_name("audit.json"))
    if (not parent_audit["passed"] or parent_audit["query_manifest"] != parent["fingerprint"]
            or audit.get("parent_audit_fingerprint") != parent_audit["fingerprint"]
            or audit.get("appearance_recipe") != recipe["fingerprint"]):
        raise ValueError("Shifted query labels require a valid parent audit; run appearance check")


def check(directory):
    from .compatibility import validate_execution
    recipe = checked(directory/"manifest.json")
    parent, columns, dependencies = load_parent(recipe["parent_manifest"], recipe["backend"])
    if parent["fingerprint"] != recipe["parent_fingerprint"] or dependencies != recipe["dependencies"] or columns != recipe["columns"]:
        raise ValueError("Appearance parent identity changed")
    query_path = directory/("queries.json" if parent.get("external_queries") else "episodes.json")
    value = checked(query_path)
    verify_export(directory, value, recipe)
    rows = 0
    seen_executions = set()
    with h5py.File(parent["source"]["path"], "r") as original, h5py.File(directory/"dataset.h5", "r") as view:
        # New exports materialize small label columns. Avoid reopening their HDF5
        # dataset objects for every episode; retain legacy VDS behavior on old files.
        label_views = {}
        for key in columns:
            if key != "pixels":
                dataset = view[key]
                label_views[key] = None if dataset.is_virtual else dataset
        for ep, offset in tqdm(list(zip(recipe["episodes"], value["episode_metadata"]["offsets"])),
                               desc="Audit unchanged labels", unit="episode"):
            payload, receipt_path = shard_paths(directory, ep)
            receipt = checked(receipt_path)
            if receipt["recipe"] != recipe["fingerprint"] or receipt["episode"] != ep or file_hash(payload) != receipt["sha256"]:
                raise ValueError("Appearance episode receipt differs")
            identity = receipt.get("execution")
            if identity not in seen_executions:
                validate_execution(directory, recipe, receipt)
                seen_executions.add(identity)
            left, count = ep["parent_raw_start"], ep["rows"]
            with h5py.File(payload, "r") as shifted:
                for key in columns:
                    if key == "pixels":
                        continue
                    a, b = original[key][left:left+count], shifted[key][:]
                    view_column = label_views[key]
                    if view_column is None:
                        view_column = view[key]
                    if (a.dtype != b.dtype or array_hash(a) != array_hash(b)
                            or array_hash(b) != receipt["labels_sha256"][key]
                            or array_hash(view_column[offset:offset+count]) != array_hash(b)):
                        raise ValueError(f"Recorded {key} changed in parent episode {ep['episode_id']}")
                if shifted["pixels"].shape != (count, 224, 224, 3) or shifted["pixels"].dtype != np.uint8:
                    raise ValueError("Shifted pixels have an invalid shape/type")
            rows += count
    audit = {"version": 1, "passed": True, "appearance_recipe": recipe["fingerprint"],
             "parent_manifest": recipe["parent_manifest"], "parent_fingerprint": parent["fingerprint"],
             "episodes": len(recipe["episodes"]), "rows": rows, "exact_labels_preserved": True,
             "split_inheritance": True, "recorded_controls_executed": 0,
             "note": "Appearance/label audit; no new behavioral, dynamics or success claim"}
    if parent.get("external_queries"):
        parent_audit = checked(Path(parent["source"]["path"]).with_name("audit.json"))
        if not parent_audit["passed"] or parent_audit["query_manifest"] != parent["fingerprint"]:
            raise ValueError("Run collect check on the original fresh cohort first")
        audit.update({key: parent_audit[key] for key in ("source_dataset", "source_size", "source_mtime_ns")})
        audit.update({"parent_audit_fingerprint": parent_audit["fingerprint"], "query_manifest": value["fingerprint"]})
    write_json(directory/"audit.json", sealed(audit))
    print(f"Passed: {rows:,} frames, all recorded row-aligned non-image columns preserved exactly.")


def preview(directory, recipe, parent, count):
    from PIL import Image, ImageDraw
    if count < 1:
        raise ValueError("--preview-episodes must be positive")
    if directory.exists() and any(directory.iterdir()):
        raise ValueError("Choose an empty preview output directory")
    directory.mkdir(parents=True, exist_ok=True)
    baseline_settings = deepcopy(recipe["settings"])
    baseline_settings.update({key: None for key in baseline_settings if "color" in key})
    baseline_settings["background_pattern"] = "preserve"
    baseline = make_renderer(recipe["backend"], baseline_settings)
    shifted = None
    comparisons, files = [], []
    try:
        shifted = make_renderer(recipe["backend"], recipe["settings"])
        with h5py.File(parent["source"]["path"], "r") as source:
            for ep in recipe["episodes"][:count]:
                panels = Image.new("RGB", (3*224, 3*248), "white")
                text = ImageDraw.Draw(panels)
                samples = sorted(set([0, ep["rows"]//2, ep["rows"]-1]))
                for index, step in enumerate(samples):
                    row = ep["parent_raw_start"]+step
                    original, native, changed = source["pixels"][row], baseline.render(source, row), shifted.render(source, row)
                    for col, (name, image) in enumerate(zip(("Recorded", "Reference re-render", "Shifted"), (original, native, changed))):
                        text.text((col*224+4, index*248+4), f"{name}, step {step}", fill="black")
                        panels.paste(Image.fromarray(image), (col*224, index*248+24))
                    comparisons.append({"episode_id": ep["episode_id"], "parent_raw_row": row,
                                        "reference_rgb_mse_0_to_1": float(np.mean(((native.astype(float)-original)/255)**2)),
                                        "shift_rgb_mse_0_to_1": float(np.mean(((changed.astype(float)-original)/255)**2))})
                filename = f"episode-{ep['parent_episode_position']:06d}.png"
                panels.save(directory/filename)
                files.append(filename)
    finally:
        baseline.close()
        if shifted is not None:
            shifted.close()
    write_json(directory/"report.json", {"recipe": recipe, "comparisons": comparisons,
               "note": "Inspect reference re-render against recorded RGB before full synthesis; camera/version/pose precision may affect parity. No WM calls or recorded controls executed."})
    (directory/"index.html").write_text('<!doctype html><meta charset="utf-8"><title>Appearance shift preview</title>'
        '<style>body{font:16px sans-serif;margin:2rem}img{max-width:100%;display:block;margin-bottom:2rem}</style>'
        '<h1>Recorded / reference re-render / shifted</h1><p>Identical recorded physical poses and labels. '
        'Inspect the middle column for renderer mismatch before accepting the shift.</p>'
        + ''.join(f'<img src="{name}" alt="Matched-pose comparison">' for name in files)
        + '<a href="report.json">Settings and RGB discrepancies</a>')
    print(f"Preview: {directory/'index.html'}")

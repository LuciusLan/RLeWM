"""Matched-pose source/native/configured RGB preview; no world model or actions."""

from copy import deepcopy
from html import escape
from pathlib import Path

import h5py
import hdf5plugin  # noqa: F401 -- register source pixel compression.
import numpy as np

from ..runtime import write_json


def preview(config, directory, source_rows):
    if config["backend"] != "reacher":
        raise ValueError("preview-rendering currently supports only Reacher")
    if not source_rows or any(type(row) is not int or row < 0 for row in source_rows):
        raise ValueError("Choose nonnegative --source-rows for the released source")
    directory = Path(directory).resolve()
    if directory.exists() and any(directory.iterdir()):
        raise ValueError("Preview output already contains files; choose a new --output-dir")
    from ..simulation.environments import make_collection_environment
    from ..simulation.environments.reacher_rendering import floor_description
    configs = {"native": {**deepcopy(config), "rendering": {"floor": "native"}},
               "configured": deepcopy(config)}
    frames, records, descriptions = [], [], {}
    environments = {}
    try:
        for name, settings in configs.items():
            environments[name] = make_collection_environment(settings, "source")
        with h5py.File(config["reference_dataset"], "r") as source:
            for row in source_rows:
                if row >= len(source["pixels"]):
                    raise ValueError(f"Source row {row} exceeds the released dataset")
                original = source["pixels"][row]
                qpos, qvel = source["qpos"][row], source["qvel"][row]
                panels, details = [original], {}
                for name, environment in environments.items():
                    environment.reset(0)
                    environment.native.set_state(qpos, qvel)
                    physics = environment.native.dmc_env.physics
                    if (not np.array_equal(physics.data.qpos, qpos)
                            or not np.array_equal(physics.data.qvel, qvel)):
                        raise ValueError("Preview failed to restore the recorded source pose/velocity")
                    image = environment.native.render(width=config["image_size"], height=config["image_size"])
                    if image.shape != original.shape:
                        raise ValueError("Source and preview image sizes differ")
                    panels.append(image)
                    details[name] = {"rgb_mse_0_to_1": float(np.mean(((image.astype(float)-original)/255)**2)),
                                     "effective_floor_mjcf": floor_description(environment.native)}
                    descriptions[name] = environment.description()
                records.append({"source_row": row, "qpos": qpos.tolist(), "qvel": qvel.tolist(), **details})
                frames.append(panels)
    finally:
        for environment in environments.values():
            environment.close()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figure, axes = plt.subplots(len(frames), 3, figsize=(10, 3.4*len(frames)), squeeze=False,
                                layout="constrained")
    for index, panels in enumerate(frames):
        for col, (label, pixels) in enumerate(zip(("Released RGB", "Native renderer", "Configured renderer"), panels)):
            axes[index, col].imshow(pixels)
            axes[index, col].set_title(f"{label}: source row {source_rows[index]}", fontsize=10)
            axes[index, col].axis("off")
    figure.suptitle("Reacher: identical recorded qpos/qvel; no controls executed")
    directory.mkdir(parents=True, exist_ok=True)
    figure.savefig(directory/"comparison.png", dpi=160)
    plt.close(figure)
    report = {"source_dataset": config["reference_dataset"], "rendering": config.get("rendering"),
              "environments": descriptions, "cases": records, "world_model_calls": 0, "controls_executed": 0,
              "note": "Engineering render check on released-source poses, not evaluation data. RGB MSE includes possible lighting/camera/version differences; no automatic claim of source parity."}
    write_json(directory/"report.json", report)
    (directory/"index.html").write_text(
        '<!doctype html><meta charset="utf-8"><title>Reacher background comparison</title>'
        '<style>body{font:16px sans-serif;margin:2rem}img{max-width:100%}</style>'
        '<h1>Reacher background comparison</h1><p>Same recorded arm poses; no WM or controls.</p>'
        f'<p>Requested rendering: <code>{escape(str(config.get("rendering")))}</code></p>'
        '<img src="comparison.png" alt="Released, native and configured rendering">'
        '<p><a href="report.json">Rendering settings and RGB errors</a></p>')
    print(f"Rendering comparison: {directory/'index.html'}")

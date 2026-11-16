"""Inspect reset/history and the fixed green-goal render without loading a model."""

from ..runtime import write_json
from ..visualization.episodes import output_directory
from ..visualization.media import Panel, index_page, save_gallery
from .cohorts import load_manifest, selected_cases
from .dataset import source_dataset
from .environments import make_environment
from .green import enabled, split_name
from .history import bootstrap


def preview(config, output=None):
    if not enabled(config):
        raise ValueError("sim preview currently supports --variant green; recorded cases use codes/preview.py")
    manifest = load_manifest(config)
    # A preview never defaults to producing the entire validation/test suite.
    cases = selected_cases(config, manifest)
    if config["evaluation"]["query_limit"] is None:
        cases = cases[:4]
    directory = output_directory(output, "green-goals")
    dataset = source_dataset(config)
    environment = make_environment(config["simulation"], dataset)
    rows, galleries = [], []
    try:
        for index, case in enumerate(cases):
            images, blocks, current = bootstrap(environment, dataset, case, config["simulation"]["seeds"][0],
                                               manifest["history"], manifest["frameskip"])
            gallery = save_gallery(directory, f"{split_name(config)}-{index:05d}", [
                Panel("Actual zero-control history", images, [i*manifest["frameskip"] for i in range(len(images))]),
                Panel("Fixed green-target goal", [environment.goal_image], [0], fixed=True)],
                fps=environment.description()["control_hz"]/manifest["frameskip"], scale=2)
            row = {"case_id": case["case_id"], "initial_state": current.state.tolist(), "initial_metrics": current.metrics,
                   "goal_contract": environment.goal_contract, "gallery": gallery}
            rows.append(row)
            galleries.append((case["case_id"], gallery))
        description = ("Native reset cases and common warm-up; target block aligns with the actual green marker. "
                       "The goal pusher pose affects image-latent planning cost, but physical success uses block overlap only. "
                       "No planner or learned model is run.")
        index_page(directory, "Green target goal checks", galleries, description)
        write_json(directory/"preview.json", {"task": "green-target", "split": split_name(config),
            "manifest_fingerprint": manifest["fingerprint"], "environment": environment.description(),
            "cases": rows, "benchmark_result": False})
        print(f"Green target previews: {directory/'index.html'}")
        return directory
    finally:
        environment.close()
        dataset.close()

"""Read-only raw data access; state interpretation belongs to task adapters."""

from pathlib import Path
import numpy as np


class SourceDataset:
    def __init__(self, path):
        import h5py
        import hdf5plugin  # noqa: F401 -- registers source compression filters.
        self.path = str(Path(path).resolve())
        self.file = h5py.File(self.path, "r", swmr=True)

    def rows(self, key, start, stop=None):
        column = self.file[key]
        if stop is not None and int(start) == int(stop):
            # A history window can start at its episode's first frame: its
            # reset then has no recorded prefix actions. Empty selections in
            # virtual HDF5 datasets can fail in HDF5's selection-bounds reader.
            # Match the normal slice shape/dtype without issuing that read.
            return np.empty((0, *column.shape[1:]), dtype=column.dtype)
        value = column[int(start)] if stop is None else column[int(start):int(stop)]
        return np.asarray(value).copy()

    def validate_case(self, case, frameskip, history):
        start, current, goal = (case[key] for key in ("start_raw_row", "current_raw_row", "goal_raw_row"))
        if current != start + (history-1)*frameskip or not start <= current < goal:
            raise ValueError("Invalid history/current/goal alignment")
        episodes = self.rows("episode_idx", start, goal+1)
        steps = self.rows("step_idx", start, goal+1)
        if (not np.all(episodes == case["query"]["episode_id"])
                or not np.array_equal(np.diff(steps), np.ones(goal-start, dtype=steps.dtype))):
            raise ValueError("Simulator case crosses an episode or step boundary")

    def close(self):
        self.file.close()


def source_dataset(config):
    from .green import enabled
    if enabled(config):
        return ResetCases()
    if config.get("query_dataset"):
        from ..collection.queries import query_source
        from ..rollout.resources import source_metadata
        manifest = query_source(config, source_metadata(config)[0])
        result = SourceDataset(manifest["source"]["path"])
        result.fresh_manifest = manifest
        return result
    return SourceDataset(config["p1"]["paths"]["dataset"])


class ResetCases:
    """No dataset dependency: only explicit common warm-up controls are allowed."""

    def rows(self, *args):
        raise ValueError("Green reset cases have no recorded observations or future actions")

    def validate_case(self, case, frameskip, history):
        actions = np.asarray(case["warmup_raw_actions"])
        if (case["start_raw_row"] != 0 or case["current_raw_row"] != (history-1)*frameskip
                or case["goal_raw_row"] is not None or actions.shape != ((history-1)*frameskip, 2)
                or not np.isfinite(actions).all() or np.any(actions != 0)):
            raise ValueError("Green cases require a fixed zero-action history and no recorded-future goal row")

    def close(self):
        pass

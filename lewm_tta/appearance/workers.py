"""Bounded process workers; each owns its native renderer and HDF5 handle."""

from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
import multiprocessing
from multiprocessing.util import Finalize
import shutil

import h5py

from .storage import write_episode


_STATE = None


class RenderWorker:
    def __init__(self, directory, recipe, parent_path, compression, batch_size, execution):
        from .pipeline import make_renderer
        import cv2
        cv2.setNumThreads(1)  # Avoid nested OpenCV thread pools across worker processes.
        self.directory, self.recipe = directory, recipe
        self.compression, self.batch_size, self.execution = compression, batch_size, execution
        self.source = h5py.File(parent_path, "r")
        self.renderer = None
        try:
            self.renderer = make_renderer(recipe["backend"], recipe["settings"])
        except BaseException:
            self.source.close()
            raise

    def render(self, episode):
        return write_episode(self.directory, self.recipe, episode, self.source, self.renderer,
                             self.compression, batch_size=self.batch_size, execution=self.execution)

    def close(self):
        try:
            if self.renderer is not None:
                self.renderer.close()
        finally:
            self.source.close()


def initialize(*arguments):
    global _STATE
    _STATE = RenderWorker(*arguments)
    Finalize(None, _STATE.close, exitpriority=10)


def render_episode(episode):
    _STATE.render(episode)
    return episode["rows"]  # Return only progress, never RGB/large receipts over IPC.


def require_space(directory, episode, minimum_gib, simultaneous=1):
    required = minimum_gib*1024**3 + simultaneous*episode["rows"]*224*224*3*1.05
    if shutil.disk_usage(directory).free < required:
        raise ValueError("Insufficient free space for in-flight episodes plus --min-free-gib reserve; free space and resume the same command")


def run(episodes, workers, arguments, minimum_gib, completed):
    directory = arguments[0]
    if workers == 1:
        worker = RenderWorker(*arguments)
        try:
            for episode in episodes:
                require_space(directory, episode, minimum_gib)
                worker.render(episode)
                completed(episode)
        finally:
            worker.close()
        return
    # Spawn avoids inheriting live HDF5/MuJoCo/SDL handles. At most `workers`
    # episodes are in flight; the full corpus is not queued or held in memory.
    iterator = iter(episodes)
    largest = max(episode["rows"] for episode in episodes)
    with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn"),
                             initializer=initialize, initargs=arguments) as pool:
        pending = {}
        try:
            while True:
                while len(pending) < workers:
                    episode = next(iterator, None)
                    if episode is None:
                        break
                    require_space(directory, {"rows": largest}, minimum_gib, workers)
                    pending[pool.submit(render_episode, episode)] = episode
                if not pending:
                    break
                finished, _ = wait(pending, return_when=FIRST_COMPLETED)
                for future in finished:
                    episode = pending.pop(future)
                    future.result()
                    completed(episode)
        except BaseException:
            for future in pending:
                future.cancel()
            raise

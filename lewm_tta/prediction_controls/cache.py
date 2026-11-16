"""Resumable frozen predictions indexed by the original memory-window order."""

from contextlib import contextmanager
import fcntl
import os
from pathlib import Path
import time

import numpy as np
import torch

from ablations.saved_runs import checked, require
from experiments.progress import progress
from ..model import Context
from ..runtime import file_hash, object_hash, read_json, write_json


@contextmanager
def exclusive(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a') as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ValueError(f'Another process owns {path}') from error
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def freeze(adapter, seed):
    adapter.reset(seed)
    adapter.model.requires_grad_(False)
    adapter.model.eval()


def assert_frozen(adapter):
    adapter.check_fixed_state()
    require(float(adapter.anchor_loss()) == 0, 'The selected predictor block changed')
    require(all(not p.requires_grad and p.grad is None for p in adapter.model.parameters()),
            'Frozen control has trainable parameters or gradients')


def specification(resources):
    return {'version': 1, 'split': 'memory', 'memory': resources.memory.identity,
            'model': resources.adapter.identity, 'normalization': resources.prepared['normalization'],
            'shape': [len(resources.memory), resources.adapter.latent_dim], 'dtype': 'float32',
            'generator_sha256': file_hash(Path(__file__))}


def location(resources, spec):
    identity = specification(resources)
    return Path(spec['cache_dir'])/f"frozen-predictions-{object_hash(identity)[:20]}", identity


def load(resources, spec):
    directory, expected = location(resources, spec)
    metadata = checked(directory/'complete.json')
    require(metadata['specification'] == expected, 'Memory-prediction cache identity differs')
    path = directory/'predictions.npy'
    require(file_hash(path) == metadata['sha256'], 'Memory-prediction cache checksum differs')
    values = np.load(path, mmap_mode='r', allow_pickle=False)
    require(list(values.shape) == expected['shape'] and values.dtype == np.float32,
            'Invalid frozen-prediction cache shape/dtype')
    return values, metadata


def prepare(resources, spec):
    require(resources.memory.split == 'memory', 'Prediction cache must contain memory only')
    directory, identity = location(resources, spec)
    with exclusive(directory/'build.lock'):
        if (directory/'complete.json').is_file():
            _, metadata = load(resources, spec)
            print(f'Using frozen memory predictions: {directory}', flush=True)
            return metadata
        progress_path = directory/'progress.json'
        path = directory/'predictions.npy'
        if progress_path.exists():
            state = read_json(progress_path)
            require(state['specification'] == identity and path.is_file(), 'Incomplete cache specification changed')
            values = np.load(path, mmap_mode='r+', allow_pickle=False)
            require(list(values.shape) == identity['shape'] and values.dtype == np.float32, 'Incomplete cache shape changed')
        else:
            require(not path.exists(), f'Orphan cache file: {path}; choose a new cache directory')
            values = np.lib.format.open_memmap(path, mode='w+', dtype=np.float32, shape=tuple(identity['shape']))
            state = {'specification': identity, 'completed_rows': 0, 'construction_seconds': 0.0}
            write_json(progress_path, state)
        start = state['completed_rows']
        require(type(start) is int and 0 <= start <= len(values), 'Invalid resume offset')
        freeze(resources.adapter, spec['seeds'][0])
        batch = spec['cache']['batch_rows']
        checkpoint = spec['cache']['checkpoint_rows']
        committed, started = start, time.perf_counter()
        with progress(total=len(values), initial=start, config=spec, desc='Frozen memory predictions', unit='window') as bar:
            for left in range(start, len(values), batch):
                right = min(left+batch, len(values))
                memory = resources.memory
                context = Context(memory.contexts[left:right], memory.past_actions[left:right])
                prediction = resources.adapter.predict(context, memory.actions[left:right])
                require(bool(torch.isfinite(prediction).all()), 'Nonfinite frozen memory prediction')
                values[left:right] = prediction.detach().cpu().numpy()
                bar.update(right-left)
                if right-committed >= checkpoint or right == len(values):
                    values.flush()
                    with path.open('r+b') as stream:
                        os.fsync(stream.fileno())
                    state.update(completed_rows=right, construction_seconds=state['construction_seconds']+time.perf_counter()-started)
                    write_json(progress_path, state)
                    committed, started = right, time.perf_counter()
        assert_frozen(resources.adapter)
        metadata = {'specification': identity, 'sha256': file_hash(path), 'bytes': path.stat().st_size,
                    'construction_seconds': state['construction_seconds'], 'batch_rows': batch,
                    'path': str(path), 'complete': True}
        metadata['fingerprint'] = object_hash(metadata)
        write_json(directory/'complete.json', metadata)
        print(f'Frozen memory predictions ready: {directory}', flush=True)
        return metadata

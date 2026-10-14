"""cuVS exact flat search with lazy sorting and complete-corpus cross-ranks.

Portable key tables are persisted rather than opaque version-dependent native
indexes. GPU flat handles borrow the uploaded tables, avoiding a second copy.
"""

from __future__ import annotations

from importlib.metadata import version
import os
from pathlib import Path

import numpy as np
import torch

from .indexed_keys import DIRECTIONS, FIELDS, KeyLayout, key_cache
from .rank_diagnostics import exact_distance_rank
from .retrieval import ExactRetriever


def trace_native(message: str) -> None:
    if os.environ.get("LEWM_CUVS_TRACE") == "1":
        print(f"[cuVS] {message}", flush=True)


class TorchCudaArray:
    """Owning CUDA-array-interface view; cuVS writes preallocated Torch buffers.

    cuvs-cu13 does not require CuPy. Retaining the tensor also retains the
    storage borrowed by a brute-force index. Calls use Torch's current stream.
    """
    def __init__(self, tensor: torch.Tensor):
        if tensor.device.type != "cuda" or not tensor.is_contiguous():
            raise ValueError("cuVS requires contiguous CUDA buffers")
        self.tensor = tensor
        dtype = {torch.float32: np.dtype("float32"), torch.int64: np.dtype("int64")}[tensor.dtype]
        self.__cuda_array_interface__ = {"shape": tuple(tensor.shape), "strides": None,
            "typestr": dtype.str, "data": (tensor.data_ptr(), False), "version": 3,
            "stream": torch.cuda.current_stream(tensor.device).cuda_stream or 1}


class CuVSRuntime:
    def __init__(self, device):
        self.device = torch.device(device)
        trace_native("checking Torch CUDA availability")
        if self.device.type != "cuda" or not torch.cuda.is_available():
            raise ValueError("The cuVS backend requires --device cuda; use backend 'torch' for CPU")
        try:
            trace_native("importing cuVS native modules")
            from cuvs.common import Resources
            from cuvs.distance import pairwise_distance
            from cuvs.neighbors import brute_force
        except ImportError as error:
            raise ImportError("cuVS is unavailable in the launcher's Python; install cuvs-cu13 there") from error
        self.Resources, self.pairwise_distance, self.brute_force = Resources, pairwise_distance, brute_force
        # In the installed cuVS 26.10 wheel, Resources(stream=int_handle)
        # casts the Python object pointer to cudaStream_t instead of converting
        # its integer value. Do not pass external stream handles through it.
        trace_native("creating cuVS-managed resources (no external stream)")
        with torch.cuda.device(self.device):
            self.handle = self.Resources()
        trace_native("cuVS-managed resources ready")

    def resources(self):
        # Input storage belongs to Torch; wait for all producer streams before
        # cuVS reads it. Every caller syncs cuVS before Torch consumes outputs.
        torch.cuda.synchronize(self.device)
        return self.handle

    def build(self, keys):
        trace_native(f"build: keys={tuple(keys.shape)}, dtype={keys.dtype}, device={keys.device}")
        with torch.cuda.device(self.device):
            resources = self.resources()
            index = self.brute_force.build(TorchCudaArray(keys), metric="sqeuclidean", resources=resources)
            resources.sync()
        trace_native("build completed")
        return index

    def search(self, index, query, k):
        trace_native(f"search: queries={tuple(query.shape)}, k={k}")
        neighbors = torch.empty((len(query), k), dtype=torch.int64, device=self.device)
        distances = torch.empty((len(query), k), dtype=torch.float32, device=self.device)
        with torch.cuda.device(self.device):
            resources = self.resources()
            self.brute_force.search(index, TorchCudaArray(query), k,
                neighbors=TorchCudaArray(neighbors), distances=TorchCudaArray(distances), resources=resources)
            resources.sync()
        trace_native("search completed")
        return distances, neighbors

    def distances(self, keys, query):
        trace_native(f"pairwise distance: keys={tuple(keys.shape)}, queries={tuple(query.shape)}")
        output = torch.empty((len(keys), len(query)), device=self.device, dtype=torch.float32)
        with torch.cuda.device(self.device):
            resources = self.resources()
            self.pairwise_distance(TorchCudaArray(keys), TorchCudaArray(query),
                out=TorchCudaArray(output), metric="sqeuclidean", resources=resources)
            resources.sync()
        trace_native("pairwise distance completed")
        if not torch.isfinite(output).all():
            raise ValueError("Nonfinite cuVS retrieval distances")
        # Expanded L2 can round tiny distances below zero.
        return output.clamp_min_(0).flatten()


class CuVSOrder:
    """Sequence whose iteration expands exact prefixes only as needed."""
    def __init__(self, ranking):
        self.ranking = ranking

    def __len__(self):
        return len(self.ranking)

    def __getitem__(self, index):
        if isinstance(index, slice):
            start, stop, step = index.indices(len(self))
            if step < 0:
                return self.ranking.indices[index]
            return self.ranking.prefix(stop)[start:stop:step]
        index = int(index)
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        return int(self.ranking.prefix(index+1)[index])

    def __iter__(self):
        position = 0
        while position < len(self):
            self.ranking.prefix(position+1)
            prefix = self.ranking._prefix
            while position < len(prefix):
                yield int(prefix[position])
                position += 1


class CuVSRanking:
    def __init__(self, owner, direction, query, components):
        self.owner, self.direction, self.query, self.query_components = owner, direction, query, components
        self._distances, self._indices, self._ranks = None, None, None
        self._prefix = np.empty(0, dtype=np.int64)
        self._rank_cache = {}
        self.order = CuVSOrder(self)

    def __len__(self):
        return len(self.owner.memory)

    @property
    def distances(self):
        if self._distances is None:
            self._distances = self.owner.runtime.distances(self.owner.keys[self.direction], self.query)
        return self._distances

    @property
    def indices(self):
        # Full sorting is required by the complete-memory consensus control,
        # not by ordinary forward/reverse support selection or cross-ranks.
        if self._indices is None:
            self._indices = torch.argsort(self.distances, stable=True).cpu().numpy()
            self._prefix = self._indices
        return self._indices

    @property
    def ranks(self):
        if self._ranks is None:
            self._ranks = np.empty(len(self), dtype=np.int64)
            self._ranks[self.indices] = np.arange(1, len(self)+1, dtype=np.int64)
        return self._ranks

    def rank_of(self, index):
        if self._ranks is not None:
            return int(self._ranks[index])
        if index not in self._rank_cache:
            self._rank_cache[index] = exact_distance_rank(self.distances, index)
        return self._rank_cache[index]

    def prefix(self, count):
        count = min(count, len(self))
        if len(self._prefix) >= count:
            return self._prefix[:count]
        requested = min(max(count, self.owner.settings.get("initial_search_k", 256),
                            len(self._prefix)*2), len(self))
        if requested >= len(self) or requested > self.owner.settings.get("max_search_k", 2048):
            return self.indices[:count]
        _, neighbors = self.owner.runtime.search(self.owner.indexes[self.direction], self.query, requested)
        selected = neighbors[0]
        if torch.any(selected < 0) or torch.any(selected >= len(self)) or len(torch.unique(selected)) != requested:
            raise ValueError("cuVS returned invalid or repeated memory indices")
        # Use one canonical full-corpus score vector for selection and ranks.
        # Including ALL scores through the candidate boundary resolves ties
        # and any roundoff difference between index and pairwise kernels.
        # This contains the true first requested candidates even if cuVS's
        # internal score differs slightly: requested valid rows lie below it.
        threshold = self.distances[selected].max()
        candidates = torch.nonzero(self.distances <= threshold, as_tuple=False).flatten()
        order = torch.argsort(self.distances[candidates], stable=True)
        self._prefix = candidates[order][:requested].cpu().numpy()
        return self._prefix[:count]


def check_cuvs_backend(device) -> dict:
    """User-run synthetic API/score/tie/spacing checks; no world-model call."""
    from .cache import LatentBank
    from .model import Context
    from .retrieval import diverse_prefix, select_support
    runtime = CuVSRuntime(device)
    keys = torch.tensor([[0., 1.], [0., 1.], [2., 0.], [-1., 0.]], device=runtime.device)
    index = runtime.build(keys)
    query = torch.tensor([[.1, .2]], device=runtime.device)
    distances, neighbors = runtime.search(index, query, 3)
    full = runtime.distances(keys, query)
    reference = (keys-query).square().sum(1)
    torch.testing.assert_close(full, reference, rtol=2e-5, atol=2e-6)
    torch.testing.assert_close(distances[0], full[neighbors[0]], rtol=2e-5, atol=2e-6)
    assert exact_distance_rank(full, 0) == 1 and exact_distance_rank(full, 1) == 2
    # A small complete memory lets every rank be compared to the Torch
    # reference, including winners outside the opposite top-k and spacing.
    count = 12
    memory = LatentBank("memory", torch.zeros(count, 1, 1, device=runtime.device),
        torch.zeros(count, 0, 1, device=runtime.device),
        torch.arange(count, device=runtime.device).float().unsqueeze(1),
        torch.arange(count-1, -1, -1, device=runtime.device).float().unsqueeze(1),
        [{"transition_id": str(i), "source_episode_position": i//3, "start_in_episode": (i%3)*20}
         for i in range(count)], {})
    settings = {"chunk_size": 4, "pool_size": 2, "beta_action": 1., "beta_successor": 1.,
                "initial_search_k": 2, "max_search_k": 4, "max_windows_per_episode": 2,
                "min_start_step_gap": 20}
    calibration = {"scales": dict.fromkeys(FIELDS, 1.)}
    layout = KeyLayout.from_memory(memory, calibration, settings)
    tables = {direction: layout.transform(direction, {name: getattr(memory, FIELDS[name])
              for name in DIRECTIONS[direction]}) for direction in DIRECTIONS}
    actual = CuVSRetriever(memory, calibration, settings, None, tables=tables)
    expected = ExactRetriever(memory, calibration, settings)
    context = Context(memory.contexts[:1], memory.past_actions[:1])
    f, r = actual.forward(context, memory.actions[:1]), actual.reverse(context, torch.zeros(1, 1, device=runtime.device))
    ef, er = expected.forward(context, memory.actions[:1]), expected.reverse(context, torch.zeros(1, 1, device=runtime.device))
    assert f.order[:2].tolist() == [0, 1] and r.order[:2].tolist() == [11, 10]
    assert f._indices is None and r._indices is None
    diagnostic = actual.diagnostics(f, r)
    assert diagnostic["forward_top1_reverse_rank"] == diagnostic["reverse_top1_forward_rank"] == count
    assert f._indices is None and r._indices is None
    assert f.indices.tolist() == ef.indices.tolist() and r.indices.tolist() == er.indices.tolist()
    torch.testing.assert_close(f.distances, ef.distances, rtol=2e-5, atol=2e-6)
    for method in ("forward", "reverse", "fr-fixed", "rank-consensus"):
        args = (memory.episode_positions, 2, count, memory.step_positions, 20)
        a = select_support(method, 4, 2, .5, f, r, torch.Generator().manual_seed(42), None, *args)
        b = select_support(method, 4, 2, .5, ef, er, torch.Generator().manual_seed(42), None, *args)
        assert a.indices == b.indices and a.weights == b.weights
    assert diverse_prefix(f.order, 4, memory.episode_positions, 2, memory.step_positions, 20) == [0, 1, 3, 4]
    generator = torch.Generator(device="cpu").manual_seed(718)
    weighted = LatentBank("memory", torch.randn(12, 2, 3, generator=generator).to(runtime.device),
        torch.randn(12, 1, 4, generator=generator).to(runtime.device),
        torch.randn(12, 4, generator=generator).to(runtime.device),
        torch.randn(12, 3, generator=generator).to(runtime.device), memory.items, {})
    weighted_settings = settings | {"beta_action": .3, "beta_successor": 1.7}
    weighted_calibration = {"scales": {"context": .7, "past_actions": 1.3, "action": 2.1, "successor": .2}}
    weighted_layout = KeyLayout.from_memory(weighted, weighted_calibration, weighted_settings)
    weighted_tables = {direction: weighted_layout.transform(direction,
        {name: getattr(weighted, FIELDS[name]) for name in DIRECTIONS[direction]}) for direction in DIRECTIONS}
    actual_weighted = CuVSRetriever(weighted, weighted_calibration, weighted_settings, None, tables=weighted_tables)
    expected_weighted = ExactRetriever(weighted, weighted_calibration, weighted_settings)
    weighted_context = Context(weighted.contexts[:1] + .1, weighted.past_actions[:1] - .2)
    wf = actual_weighted.forward(weighted_context, weighted.actions[:1] + .3)
    wr = actual_weighted.reverse(weighted_context, weighted.successors[:1] - .4)
    ef = expected_weighted.forward(weighted_context, weighted.actions[:1] + .3)
    er = expected_weighted.reverse(weighted_context, weighted.successors[:1] - .4)
    for indexed, reference_ranking in ((wf, ef), (wr, er)):
        torch.testing.assert_close(indexed.distances, reference_ranking.distances, rtol=2e-5, atol=2e-6)
        assert indexed.indices.tolist() == reference_ranking.indices.tolist()
    return {"backend": "cuvs", "version": version("cuvs-cu13"), "synthetic_scores": True,
            "stable_ties": True, "opposite_ranks_outside_topk": True, "lazy_directional_prefixes": True,
            "support_spacing_and_consensus_parity": True, "weighted_multiblock_mse_parity": True}


def diagnose_cuvs(device) -> dict:
    """User-run backend probe without dataset access or world-model loading."""
    import json
    print(f"Retrieval diagnostic | Torch {torch.__version__} | compiled CUDA {torch.version.cuda}", flush=True)
    print("LD_LIBRARY_PATH=" + os.environ.get("LD_LIBRARY_PATH", "<unset>"), flush=True)
    print("Checking loaded cuDNN version", flush=True)
    cudnn_version = torch.backends.cudnn.version()
    print(f"Loaded cuDNN version: {cudnn_version}", flush=True)
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    result = check_cuvs_backend(device)
    # Only library paths are collected, never process environment/core memory.
    markers = ("libcuda.so", "libcudart.so", "libcudnn", "libcublas", "libcuvs", "libraft", "librmm")
    maps_path = Path("/proc/self/maps")
    loaded = set()
    if maps_path.exists():
        for line in maps_path.read_text().splitlines():
            fields = line.split(maxsplit=5)
            if len(fields) == 6 and any(marker in fields[5] for marker in markers):
                loaded.add(fields[5])
    result.update(torch_version=torch.__version__, torch_compiled_cuda=torch.version.cuda,
                  loaded_cudnn_version=cudnn_version, loaded_libraries=sorted(loaded),
                  stream_policy="cuVS-managed stream; Torch synchronize before calls; cuVS sync after calls")
    print(json.dumps(result, indent=2), flush=True)
    return result


def prepare_cuvs(memory, calibration, settings, directory, device):
    runtime = CuVSRuntime(device)
    record = key_cache(memory, calibration, settings, Path(directory), build=True)
    if not len(memory):
        return record
    # Validate each full-size table sequentially so preparation does not need
    # both tables resident while the encoder/cache are still in memory.
    checks = {}
    for direction, entry in record["arrays"].items():
        torch.cuda.empty_cache()
        free, _ = torch.cuda.mem_get_info(runtime.device)
        required = entry["shape"][0] * entry["shape"][1] * 4
        reserve = int(settings.get("device_reserve_gib", 2) * 2**30)
        if required + reserve > free:
            raise MemoryError(f"{direction} cuVS preparation needs {required/2**30:.2f} GiB plus "
                              f"{reserve/2**30:.2f} GiB reserve; {free/2**30:.2f} GiB is free")
        keys = torch.from_numpy(np.load(entry["path"], mmap_mode="c", allow_pickle=False)).to(runtime.device)
        index = runtime.build(keys)
        distances, neighbors = runtime.search(index, keys[:1], min(8, len(keys)))
        if not torch.isfinite(distances).all() or torch.any(neighbors < 0) or torch.any(neighbors >= len(keys)):
            raise ValueError("Invalid full-size cuVS preparation probe")
        checks[direction] = {"rows": len(keys), "width": keys.shape[1], "probe_k": neighbors.shape[1]}
        del index, keys
        torch.cuda.empty_cache()
    print(f"cuVS index keys prepared | {len(memory):,} memory windows | "
          f"{record['numeric_payload_bytes']/2**30:.2f} GiB | version {version('cuvs-cu13')}", flush=True)
    return record | {"cuvs_version": version("cuvs-cu13"), "full_index_build_checks": checks}


class CuVSRetriever(ExactRetriever):
    def __init__(self, memory, calibration, settings, directory, *, tables=None):
        super().__init__(memory, calibration, settings)
        self.runtime = CuVSRuntime(memory.contexts.device)
        self.layout = KeyLayout.from_memory(memory, calibration, settings)
        self.assets = None
        if tables is None and len(memory):
            self.assets = key_cache(memory, calibration, settings, Path(directory), build=False)
            required = self.assets["numeric_payload_bytes"]
            torch.cuda.empty_cache()
            free, _ = torch.cuda.mem_get_info(self.runtime.device)
            reserve = int(settings.get("device_reserve_gib", 2) * 2**30)
            if required + reserve > free:
                raise MemoryError(f"cuVS keys need {required/2**30:.2f} GiB plus {reserve/2**30:.2f} GiB "
                                  f"reserve; {free/2**30:.2f} GiB is free")
            tables = {direction: torch.from_numpy(np.load(entry["path"], mmap_mode="c", allow_pickle=False))
                      .to(self.runtime.device) for direction, entry in self.assets["arrays"].items()}
        self.keys = tables or {}
        self.indexes = {direction: self.runtime.build(keys) for direction, keys in self.keys.items()}

    @property
    def backend_info(self):
        return {"backend": "cuvs", "version": version("cuvs-cu13"), "index": "brute_force",
                "metric": "sqeuclidean", "precision": "float32", "rank_ties": "memory row",
                "global_rank_scores": "cuVS pairwise distances over all memory rows, clamped at zero",
                "state_control_backend": "torch", "memory_windows": len(self.memory)}

    def _query(self, direction, context, extra):
        if context.latents.shape[0] != 1:
            raise ValueError("Independent-query retrieval requires batch size one")
        components = {"context": context.latents, "past_actions": context.past_actions,
                      "action" if direction == "forward" else "successor": extra}
        query = self.layout.transform(direction, components)
        return CuVSRanking(self, direction, query, components)

    @torch.no_grad()
    def forward(self, context, action):
        return self._query("forward", context, action) if len(self.memory) else super().forward(context, action)

    @torch.no_grad()
    def reverse(self, context, prediction):
        return self._query("reverse", context, prediction) if len(self.memory) else super().reverse(context, prediction)

    def describe(self, index, forward, reverse):
        if not isinstance(forward, CuVSRanking):
            return super().describe(index, forward, reverse)
        components = forward.query_components | reverse.query_components
        raw = {}
        for name, query in components.items():
            value = getattr(self.memory, FIELDS[name])[index:index+1]
            difference = (value-query).reshape(1, self.layout.dimensions[name])
            raw[name] = float(difference.square().mean()) if difference.numel() else 0.0
        return {"forward_distance": float(forward.distances[index]),
                "reverse_distance": float(reverse.distances[index]), "raw_component_mse": raw}

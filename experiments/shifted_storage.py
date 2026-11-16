"""Model-free, filesystem-aware reservations for new shifted encodings/keys."""

from pathlib import Path
import shutil

from lewm_tta.collection.manifest import checked, sealed
from lewm_tta.preprocessing.inspection import cache_bytes
from lewm_tta.preprocessing.storage import existing_parent
from lewm_tta.runtime import file_hash, read_json, write_json


GIB = 2**30


def storage_requests(ctx, memory_kinds):
    paths, base = ctx["paths"], ctx["source_p1"]
    checkpoint = read_json(Path(base["paths"]["checkpoint"])/"config.json")
    latent = checkpoint["projector"]["output_dim"]
    width = ctx["source_manifest"]["source"]["schema"]["action"]["shape"][-1]
    requests = []
    library = Path(__file__).resolve().parents[1]/"lewm_tta"
    builders = {name: file_hash(library/name) for name in
                ("data.py", "model.py", "observation_cache.py", "frame_bank.py", "indexed_keys.py",
                 "retrieval.py", "cache.py", "preprocessing/banks.py")}
    if "shifted" in memory_kinds:
        export_path = Path(paths["memory"])/"episodes.json"
        if export_path.exists():
            memory = checked(export_path)
            rows = memory["source"]["rows"]
            span = (base["data"]["history"]+1)*base["data"]["frameskip"]
            windows = sum(max(0, length-span+1) for length in memory["episode_metadata"]["lengths"])
            source = memory["fingerprint"]
        else:
            manifest = ctx["source_manifest"]
            rows = sum(manifest["episode_metadata"]["lengths"][i] for i in manifest["episodes"]["memory"])
            windows, source = ctx["source_prepared"]["cache_counts"]["memory"], "pending-synthesis"
        size = cache_bytes(rows, windows, base["data"]["history"], latent, width, base["data"]["frameskip"])["total_bytes"]
        requests.append({"path": paths["memory_cache"], "bytes": size, "purpose": "shifted memory latents/actions/flat keys",
                         "source": source})
    query_path = Path(paths["queries"])/"queries.json"
    query = checked(query_path if query_path.exists() else Path(paths["query_parent"]))
    requests.append({"path": paths["query_cache"], "bytes": query["source"]["rows"]*(latent+width)*4,
                     "purpose": "shared shifted query latents/actions", "source": query["fingerprint"] if query_path.exists() else "pending-synthesis"})
    for item in requests:
        item["identity"] = sealed({"version": 1, "source": item["source"], "purpose": item["purpose"],
                                   "model": ctx["source_prepared"]["model_fingerprint"],
                                   "normalization": ctx["source_prepared"]["normalization"],
                                   "calibration": ctx["source_prepared"]["calibration"],
                                   "builders": builders,
                                   "bytes": item["bytes"]})
    return requests


def check_space(contexts, memory_kinds, keep_free_gib, *, reserve=False):
    volumes, requests = {}, []
    for ctx in contexts:
        requests += storage_requests(ctx, memory_kinds)
        requests.append({"path": ctx["paths"]["output"], "bytes": 64*1024**2, "purpose": "shifted stage metadata"})
    for item in requests:
        path = Path(item["path"])
        parent = existing_parent(path)
        credit = 0
        marker = path/".shifted-cache.json"
        if "identity" in item and marker.exists():
            if checked(marker) != item["identity"]:
                raise ValueError(f"Shifted cache reservation differs from this source/model: {marker}")
            credit = min(item["bytes"], sum(p.stat().st_blocks*512 for p in path.rglob("*.npy")))
        additional = max(0, item["bytes"]-credit)
        print(f"{item['purpose']}: {item['bytes']/GIB:.2f} GiB total, {additional/GIB:.2f} GiB additional | {path}")
        group = volumes.setdefault(parent.stat().st_dev, {"path": str(parent), "additional": 0,
                                                          "free": shutil.disk_usage(parent).free})
        group["additional"] += additional
    sufficient = True
    for group in volumes.values():
        needed = group["additional"]+keep_free_gib*GIB
        fits = group["free"] >= needed
        sufficient &= fits
        print(f"Volume {group['path']}: {group['free']/GIB:.2f} GiB free; need {needed/GIB:.2f} GiB with reserve: {'fits' if fits else 'insufficient'}")
    if reserve:
        if not sufficient:
            raise ValueError("Not enough space for shifted caches. Use --cache-root /opt/dlami/nvme/ubuntu/lewm_appearance_cache or expand persistent storage.")
        for item in requests:
            if "identity" in item:
                marker = Path(item["path"])/".shifted-cache.json"
                if not marker.exists():
                    write_json(marker, item["identity"])
    return sufficient

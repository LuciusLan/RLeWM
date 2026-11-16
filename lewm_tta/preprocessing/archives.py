"""Stream one native HDF5 from a tar.zst without extracting arbitrary paths."""

from contextlib import contextmanager
import fcntl
import hashlib
import os
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import tarfile

from ..runtime import file_hash, read_json, write_json
from .hdf5 import raw_dataset_path


def file_signature(path):
    path = Path(path).resolve(strict=True)
    stat = path.stat()
    return {"path": str(path), "bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}


@contextmanager
def tar_stream(path):
    if shutil.which("zstd") is None:
        raise FileNotFoundError("zstd is required to read the released .tar.zst archives")
    process = subprocess.Popen(["zstd", "--decompress", "--stdout", "--", str(path)],
                               stdout=subprocess.PIPE)
    try:
        with tarfile.open(fileobj=process.stdout, mode="r|") as archive:
            yield archive, process
    finally:
        process.stdout.close()
        if process.poll() is None:
            process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()


def checked_member(member, expected):
    path = PurePosixPath(member.name)
    if path.is_absolute() or ".." in path.parts or not member.isfile() or str(path) != expected:
        raise ValueError(f"Expected a regular archive member {expected!r}, found {member.name!r}")
    return {"member": member.name, "bytes": member.size, "mtime_ns": int(member.mtime * 10**9)}


def archive_header(path, member):
    """Only inspect the first regular header; never scan a 100-GB payload."""
    signature = file_signature(path)
    with tar_stream(path) as (archive, _):
        for entry in archive:
            if entry.isdir():
                continue
            return {"archive": signature, **checked_member(entry, member)}
    raise ValueError(f"Archive has no HDF5 member: {path}")


def persist_receipt(config, receipt):
    """Keep the streamed HDF5 checksum even when the NVMe volume is erased."""
    path = Path(config["paths"]["experiment_dir"]) / "source-extraction.json"
    if path.exists():
        previous = read_json(path)
        if previous.get("sha256") and receipt.get("sha256") != previous["sha256"]:
            raise ValueError(f"Restored HDF5 payload differs from the persistent checksum: {path}")
    write_json(path, receipt)


def extract_hdf5(config, header):
    """Atomic extraction; interrupted owned partial files restart on rerun.

    The supplied tar archives contain one HDF5. Preserve the member mtime so
    restoring an erased NVMe file retains the source-cache identity.
    """
    from experiments.progress import progress

    target = raw_dataset_path(config)
    target.parent.mkdir(parents=True, exist_ok=True)
    receipt_path = target.with_name(target.name + ".extraction.json")
    partial = target.with_name(target.name + ".partial")
    with target.with_name(target.name + ".extract.lock").open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(f"Another process is extracting {target}") from error
        if target.exists():
            if target.stat().st_size != header["bytes"]:
                raise ValueError(f"Existing HDF5 size differs from the archive: {target}")
            if receipt_path.exists():
                receipt = read_json(receipt_path)
                if receipt.get("header") != header:
                    raise ValueError(f"Extraction receipt differs: {receipt_path}")
                if receipt.get("status") == "extracting" and target.stat().st_mtime_ns == header["mtime_ns"]:
                    # Atomic rename succeeded before the final receipt write.
                    # Revalidate the native schema before sealing that receipt.
                    from ..data import inspect_source
                    inspect_source(target)
                    receipt = {"status": "complete", "header": header, "output": file_signature(target),
                               "sha256": file_hash(target), "recovered_after_atomic_rename": True}
                    write_json(receipt_path, receipt)
                if receipt.get("status") != "complete":
                    raise ValueError(f"Incomplete extraction receipt: {receipt_path}")
                if receipt["output"] != file_signature(target):
                    raise ValueError(f"Extracted HDF5 changed: {target}")
                persist_receipt(config, receipt)
            print(f"Reuse extracted HDF5: {target}", flush=True)
            return
        if partial.exists():
            if not receipt_path.exists() or read_json(receipt_path).get("header") != header:
                raise ValueError(f"Unrecognized partial file; inspect it before removing: {partial}")
            print(f"Restart interrupted extraction: {target.name}", flush=True)
        write_json(receipt_path, {"status": "extracting", "header": header})
        archive_path = Path(config["preprocessing"]["archive"])
        digest, written, found = hashlib.sha256(), 0, False
        with tar_stream(archive_path) as (archive, process), partial.open("wb") as destination, progress(
                total=header["bytes"], config=config, desc=f"Extract {config['data']['dataset_id']}",
                unit="B", unit_scale=True) as bar:
            for entry in archive:
                if entry.isdir():
                    continue
                if found:
                    raise ValueError("Expected exactly one regular HDF5 file in the source archive")
                actual = checked_member(entry, config["preprocessing"]["member"])
                if actual != {key: header[key] for key in actual}:
                    raise ValueError("Archive header changed after storage preflight")
                found = True
                with archive.extractfile(entry) as source:
                    for block in iter(lambda: source.read(8 * 1024**2), b""):
                        destination.write(block)
                        digest.update(block)
                        written += len(block)
                        bar.update(len(block))
            # tar's end marker can precede the zstd checksum. Drain the stream
            # and require a successful decompressor exit, including that checksum.
            while process.stdout.read(1024**2):
                pass
            if process.wait() != 0 or not found or written != header["bytes"]:
                raise ValueError(f"Incomplete or corrupt archive: {archive_path}")
            destination.flush()
            os.fsync(destination.fileno())
        os.utime(partial, ns=(header["mtime_ns"], header["mtime_ns"]))
        partial.replace(target)
        receipt = {"status": "complete", "header": header,
                   "output": file_signature(target), "sha256": digest.hexdigest()}
        persist_receipt(config, receipt)
        write_json(receipt_path, receipt)
        print(f"Extracted: {target}", flush=True)

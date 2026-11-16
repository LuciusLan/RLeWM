#!/usr/bin/env bash
# A small child venv: local physics pins, inherited parent packages.
set -euo pipefail
TASK_DIAGNOSTIC_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
TASK_PROJECT_ROOT="$(cd -- "$TASK_DIAGNOSTIC_DIR/../.." && pwd)"
TASK_BASE_PYTHON="${BASE_PYTHON:-/home/ubuntu/.venv/bin/python}"
TASK_PHYSICS_VENV="${PHYSICS_VENV:-$TASK_PROJECT_ROOT/.venvs/physics-mj36}"
TASK_DRY_RUN=0
if [[ $# -eq 1 && "$1" == "--dry-run" ]]; then
    TASK_DRY_RUN=1
elif [[ $# -ne 0 ]]; then
    printf 'Usage: bash codes/diagnostics/setup-physics-venv.sh [--dry-run]\nOverrides: BASE_PYTHON, PHYSICS_VENV\n' >&2
    exit 2
fi

# -I prevents a previous PYTHONPATH overlay from affecting setup/resolution.
"$TASK_BASE_PYTHON" -I - "$TASK_PHYSICS_VENV" "$TASK_DRY_RUN" <<'PY'
import json
import site
import subprocess
import sys
import venv
from importlib.metadata import PackageNotFoundError, distribution
from pathlib import Path

target = Path(sys.argv[1]).expanduser().absolute()
parent = Path(sys.prefix).resolve()
pins = {"mujoco": "3.6.0", "dm-control": "1.0.38"}
if sys.prefix == sys.base_prefix:
    raise SystemExit("BASE_PYTHON must select the existing parent venv's interpreter.")
if target.resolve().is_relative_to(parent):
    raise SystemExit("PHYSICS_VENV must be outside the parent environment.")

# --system-site-packages inherits the system installation, not this parent
# venv. Add its local package directories explicitly, after the child's own.
parent_sites = [str(Path(path).resolve()) for path in site.getsitepackages()
                if Path(path).is_dir() and Path(path).resolve().is_relative_to(parent)]
parent_sites = list(dict.fromkeys(parent_sites))
if not parent_sites:
    raise SystemExit("No local parent site-packages directory found.")

def package_info(name):
    try:
        item = distribution(name)
    except PackageNotFoundError:
        return None
    return {"version": item.version, "location": str(Path(item.locate_file("")).resolve())}

receipt = {
    "complete": False,
    "parent_python": sys.executable,
    "parent_prefix": str(parent),
    "python_version": sys.version,
    "parent_site_packages": parent_sites,
    "parent_packages": {name: package_info(name) for name in
                        [*pins, "torch", "stable-worldmodel", "stable-pretraining"]},
    "pins": pins,
    "venv": str(target),
}
print(f"Parent interpreter: {sys.executable}", flush=True)
print(f"Child venv: {target}", flush=True)
print(f"Inherited parent package directories: {parent_sites}", flush=True)
print(f"Install locally, without dependencies: {pins}", flush=True)
if sys.argv[2] == "1":
    raise SystemExit(0)
if target.exists() or target.is_symlink():
    raise SystemExit(f"Preserved existing/partial venv: {target}\nChoose a new PHYSICS_VENV.")

# Create at the final path: moving a venv afterward breaks script shebangs.
target.mkdir(parents=True, exist_ok=False)
try:
    venv.EnvBuilder(system_site_packages=True, with_pip=True).create(target)
    python = str(target / "bin/python")
    purelib = Path(subprocess.check_output(
        [python, "-I", "-c", "import sysconfig; print(sysconfig.get_path('purelib'))"],
        text=True).strip())
    if not purelib.resolve().is_relative_to(target.resolve()):
        raise RuntimeError(f"Child site-packages resolved outside the child: {purelib}")
    (purelib / "lewm-parent.pth").write_text("\n".join(parent_sites) + "\n")
    receipt_path = target / "physics-venv.json"
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
    subprocess.run([
        python, "-I", "-m", "pip", "--isolated", "install",
        "--ignore-installed", "--no-deps", "--only-binary=:all:",
        "--report", str(target / "pip-report.json"),
        *[f"{name}=={version}" for name, version in pins.items()],
    ], check=True)
    # Metadata only: importing native bindings and running simulators are left
    # to the user's environment smoke check. Verify local overrides and reuse.
    subprocess.run([python, "-I", "-c", '''
import json, sys
from importlib.metadata import distribution
from pathlib import Path
path = Path(sys.argv[1])
receipt = json.loads(path.read_text())
root = Path(sys.prefix).resolve()
resolved = {}
for name, expected in receipt["pins"].items():
    item = distribution(name)
    location = Path(item.locate_file("")).resolve()
    if item.version != expected or not location.is_relative_to(root):
        raise SystemExit(f"Child override failed: {name} {item.version} at {location}")
    resolved[name] = {"version": item.version, "location": str(location)}
for name, expected in receipt["parent_packages"].items():
    if name in receipt["pins"] or expected is None:
        continue
    item = distribution(name)
    actual = {"version": item.version, "location": str(Path(item.locate_file("")).resolve())}
    if actual != expected:
        raise SystemExit(f"Parent package reuse failed: {name}: {actual} != {expected}")
    resolved[name] = actual
receipt.update(complete=True, resolved_packages=resolved)
path.write_text(json.dumps(receipt, indent=2) + "\\n")
''', str(receipt_path)], check=True)
except Exception:
    print(f"Setup incomplete; preserved partial venv at {target}. "
          "Choose a new PHYSICS_VENV for a retry.", file=sys.stderr)
    raise
print(f"Ready: {target / 'bin/python'}", flush=True)
print("Run the documented environment smoke check before collection.", flush=True)
PY

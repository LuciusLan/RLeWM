"""Resume unchanged render recipes across recorded I/O-only optimizations."""

from ..collection.manifest import checked, sealed
from ..runtime import write_json


# These modules schedule/write/audit frames. Renderer, color, parent inspection,
# native simulator, and package identities must still match exactly.
IO_MODULES = {f"appearance/{name}.py" for name in
              ("cli", "pipeline", "storage", "workers", "compatibility")}


def validate_implementation(previous, current):
    if any(previous[key] != current[key] for key in ("native_code", "packages")):
        raise ValueError("Native rendering sources or package versions changed; use a new output directory")
    old, new = previous["code"], current["code"]
    changed = [key for key in old.keys() | new.keys() if old.get(key) != new.get(key) and key not in IO_MODULES]
    if changed:
        raise ValueError(f"Appearance renderer/inputs changed ({', '.join(sorted(changed))}); use a new output directory")


def resume_recipe(directory, current):
    path = directory/"manifest.json"
    if not path.exists():
        return current
    saved = checked(path)
    if saved == current:
        return saved
    specification = lambda value: {key: item for key, item in value.items() if key not in ("fingerprint", "implementation")}
    if specification(saved) != specification(current):
        raise ValueError("Appearance settings, parent, selection or compression changed; use a new output directory")
    validate_implementation(saved["implementation"], current["implementation"])
    print("Resume original recipe: renderer/labels/settings unchanged; updated I/O execution recorded separately.")
    return saved


def record_execution(directory, recipe, implementation, workers, batch_size):
    validate_implementation(recipe["implementation"], implementation)
    record = sealed({"version": 1, "recipe": recipe["fingerprint"], "implementation": implementation,
                     "workers": workers, "write_batch_size": batch_size,
                     "compatibility": "same renderer, settings, native sources, packages and input selection; I/O scheduling may differ"})
    path = directory/"executions"/(record["fingerprint"] + ".json")
    if path.exists():
        if checked(path) != record:
            raise ValueError("Appearance execution metadata changed")
    else:
        write_json(path, record)
    return record["fingerprint"]


def validate_execution(directory, recipe, receipt):
    identity = receipt.get("execution")
    if identity is None:
        return  # Original receipts bind directly to recipe.implementation.
    if not isinstance(identity, str) or len(identity) != 64 or any(c not in "0123456789abcdef" for c in identity):
        raise ValueError("Invalid appearance execution identity")
    record = checked(directory/"executions"/(identity + ".json"))
    if record["fingerprint"] != identity or record["recipe"] != recipe["fingerprint"]:
        raise ValueError("Appearance execution belongs to another recipe")
    validate_implementation(recipe["implementation"], record["implementation"])

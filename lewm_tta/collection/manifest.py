"""Reserve split membership and random streams before observing any outcomes."""

from pathlib import Path

from ..runtime import object_hash, read_json, write_json


def sealed(value):
    return {**value, "fingerprint": object_hash(value)}


def checked(path):
    value = read_json(Path(path))
    if value.get("fingerprint") != object_hash({k: v for k, v in value.items() if k != "fingerprint"}):
        raise ValueError(f"Manifest checksum mismatch: {path}")
    return value


def cases(config):
    records = []
    for split, count in config["splits"].items():
        for index in range(count):
            # Counterfactual regimes share case IDs/reset/action RNGs. Validation
            # and test are independently reserved before any collection.
            case_id = f"fresh-{config['backend']}:{config['seed']}:{split}:{index}"
            seed = int(object_hash([case_id, "reset"])[:8], 16)
            action_seed = int(object_hash([case_id, "actions"])[:16], 16)
            records.append({"case_id": case_id, "split": split, "index": index,
                            "episode_id": 1_000_000_000+len(records), "reset_seed": seed,
                            "action_seed": action_seed})
    if len({row["reset_seed"] for row in records}) != len(records):
        raise ValueError("Reset seed collision; choose another master seed")
    return records


def attempt_case(case, attempt):
    """A declared finite candidate stream; attempt zero preserves the reserved seed."""
    if attempt == 0:
        return dict(case)
    return {**case, "reset_seed": int(object_hash([case["case_id"], "attempt", attempt, "reset"])[:8], 16),
            "action_seed": int(object_hash([case["case_id"], "attempt", attempt, "actions"])[:16], 16)}


def prepare(config):
    reserved = cases(config)
    if config.get("quality"):
        candidates = [attempt_case(case, index)["reset_seed"] for case in reserved
                      for index in range(config["quality"]["max_attempts"])]
        if len(candidates) != len(set(candidates)):
            raise ValueError("Candidate reset seeds collide across cases/attempts; choose another master seed")
    value = sealed({"version": 2 if config.get("quality") else 1, "stage": "fresh-collection", "config": config, "cases": reserved,
        "selection": ("bounded deterministic candidate-seed stream per reserved case; geometry/contact/motion eligibility; "
                      "seeded q windows among eligible starts; no evaluated-model or green-target success filtering"
                      if config.get("quality") else "all prespecified episodes retained; no model or outcome-based rejection"),
        "pretraining_provenance": "new simulator executions, no released trajectory resets; duplicate audit against the released source still required",
        "memory_policy": "all collected validation/test observations and targets remain evaluation-only"})
    path = Path(config["output_dir"])/"manifest.json"
    if path.exists() and checked(path) != value:
        raise ValueError("Collection specification changed; choose another output directory")
    if not path.exists():
        write_json(path, value)
    print(f"Reserved {len(value['cases'])} fresh episodes per regime: {path}")
    return value


def load_manifest(config):
    value = checked(Path(config["output_dir"])/"manifest.json")
    if value["config"] != config:
        raise ValueError("Collection config changed; preserve it or choose a new output directory")
    return value

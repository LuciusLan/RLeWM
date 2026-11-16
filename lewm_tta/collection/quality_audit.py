"""Verify accepted quality windows and the complete candidate-attempt ledger."""

from pathlib import Path

import numpy as np

from ..runtime import file_hash, object_hash
from .manifest import attempt_case
from .quality import assess_episode, select_starts


def audit_episode(config, record, stream, adapter, shift):
    diagnose = lambda state, goal: adapter.goal_diagnostics(state, goal, shift)
    computed = assess_episode(config, stream["state64"][:], stream["quality_bounds"][:],
                             stream["quality_contacts"][:], record["quality"]["extent"], diagnose)
    computed["selected_start_steps"] = select_starts(record["case"], computed["eligible_start_steps"], config["windows_per_episode"])
    if not computed["accepted"] or computed != record["quality"] or stream["quality_contacts"][0] != 0:
        raise ValueError("Accepted trajectory fails its declared geometry/contact/motion eligibility")
    attempts = record["attempts"]
    if len(attempts) != record["accepted_attempt"]+1 or len(attempts) > config["quality"]["max_attempts"]:
        raise ValueError("Incomplete candidate attempt ledger")
    rejected = {}
    for index, attempt in enumerate(attempts):
        path = Path(attempt["measurements"])
        if (attempt["attempt"] != index or attempt["execution"] != attempt_case(record["case"], index)
                or attempt["collection_manifest"] != record["collection_manifest"]
                or attempt["code_fingerprint"] != object_hash(record["code"])
                or attempt["environment_fingerprint"] != object_hash(record["environment"])
                or attempt["regime"] != record["regime"] or file_hash(path) != attempt["measurements_sha256"]):
            raise ValueError("Candidate attempt identities or physical traces changed")
        with np.load(path, allow_pickle=False) as trace:
            result = assess_episode(config, trace["states"], trace["bounds"], trace["contacts"],
                                    attempt["quality"]["extent"], diagnose)
            if "native_truncation" in attempt["quality"]["reasons"]:
                result["accepted"] = False
                result["reasons"].append("native_truncation")
            if result["accepted"]:
                result["selected_start_steps"] = select_starts(record["case"], result["eligible_start_steps"], config["windows_per_episode"])
                if (index != len(attempts)-1 or attempt["execution"] != record["execution"]
                        or attempt["episode_sha256"] != record["sha256"]
                        or not np.array_equal(trace["states"], stream["state64"][:])
                        or not np.array_equal(trace["actions"], stream["action"][:-1])):
                    raise ValueError("Exported episode is not the first accepted attempt")
            elif index == len(attempts)-1:
                raise ValueError("Episode refers to a rejected final attempt")
            if result != attempt["quality"]:
                raise ValueError("Candidate rejection/eligibility cannot be reproduced from its trace")
        for reason in result["reasons"]:
            rejected[reason] = rejected.get(reason, 0)+1
    return {"attempts": len(attempts), "rejection_counts": rejected,
            "candidate_windows": computed["candidate_windows"],
            "eligible_windows": len(computed["eligible_start_steps"]),
            "minimum_clearance": computed["minimum_clearance"],
            "window_failure_counts": computed["window_failure_counts"]}

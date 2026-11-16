"""Depth-wise checkpoint training; final weights score every rollout depth."""

import torch

from ..model import Context
from ..planning.temporal.state import advance
from ..runtime import object_hash
from .references import balanced_pairs
from .supports import directions


def adapt_sequential(session, checkpoint, references, reference_ids, record):
    settings, engine = session.config["interleave"], session.engine
    channels = 1 if settings["support_method"] == "uniform" else len(directions(engine.settings, settings["support_method"]))
    limit = settings["support_pairs"] or max(1, engine.settings["support_size"]//2)
    limit = min(limit, engine.settings["support_size"]//channels)
    for depth in settings["adaptation_depths"]:
        timing_before = dict(session.timings)
        before = session.version
        # Every prefix starts at the actual observation under the current
        # predictor. Earlier imagined states are never reused after an update.
        if depth:
            contexts, predictions = session.roll(references[:, :depth])
            current = advance(contexts[-1], references[:, depth-1], predictions[-1])
        else:
            current = Context(session.context.latents.expand(len(references), -1, -1).detach(),
                              session.context.past_actions.expand(len(references), -1, -1).detach())
        pairs = balanced_pairs(len(references), [depth], limit,
                               session.rng(f"adaptation-pairs-depth-{depth}", checkpoint))
        contexts = Context(torch.cat([current.latents[p:p+1] for p, _ in pairs]).detach(),
                           torch.cat([current.past_actions[p:p+1] for p, _ in pairs]).detach())
        actions = torch.cat([references[p:p+1, depth] for p, _ in pairs]).detach()
        stage = {"depth": depth, "model_version_before": before,
                 "reference_prefix_model_version": before, "reference_prefix_blocks": depth,
                 "requested_updates": engine.settings["updates"], "actual_updates": 0,
                 "support_presentations": 0, "rounds": [],
                 "adaptation_pairs": [{"parent_candidate_id": int(reference_ids[p]), "depth": h}
                                      for p, h in pairs],
                 "adaptation_query_fingerprint": object_hash({
                     "latents": contexts.latents.cpu().tolist(),
                     "past_actions": contexts.past_actions.cpu().tolist(), "actions": actions.cpu().tolist()})}
        # One optimizer is owned by the planning call. FR-fixed reuses this
        # depth's batch only; the next depth always selects fresh support.
        session.adapt_pairs(checkpoint, contexts, actions, stage, depth=depth)
        rounds = stage.pop("rounds")
        stage.update(model_version_after=session.version,
                     update_indices=[row["update"] for row in rounds],
                     timing_seconds={name: value-timing_before[name] for name, value in session.timings.items()},
                     skip_reason="no_eligible_support" if stage["actual_updates"] < stage["requested_updates"] else None)
        record["depths"].append(stage)
        record["adaptation_pairs"].extend(stage["adaptation_pairs"])
        record["rounds"].extend(rounds)
        record["actual_updates"] += stage["actual_updates"]
        record["support_presentations"] += stage["support_presentations"]
    record["adaptation_query_fingerprint"] = object_hash([
        {"depth": stage["depth"], "model_version": stage["reference_prefix_model_version"],
         "query": stage["adaptation_query_fingerprint"]} for stage in record["depths"]])

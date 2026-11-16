"""Exact context-only retrieval using the existing calibrated cuVS keys.

Process contiguous chunks of the context/past-action prefix. This adds bounded
scratch space, not another full resident index. Future-action/successor columns
are never included in the metric, even when they exist in the source table.
"""

import torch


def exact_prefix(distances, eligible, limit):
    """Canonical ties by memory row, including every tie at the top-k boundary."""
    if limit < 1:
        return []
    ids = torch.as_tensor(eligible, dtype=torch.long, device=distances.device)
    if not len(ids):
        return []
    values = distances[ids]
    count = min(limit, len(ids))
    boundary = torch.topk(values, count, largest=False).values.max()
    candidates = ids[values <= boundary]
    # eligible is sorted in memory-row order; stable sorting resolves equal costs.
    order = torch.argsort(distances[candidates], stable=True)[:count]
    return candidates[order].cpu().tolist()


class ContextSearch:
    def __init__(self, retriever, chunk_rows=32768):
        self.base, self.memory = retriever, retriever.memory
        self.chunk_rows = chunk_rows
        self.cuvs = hasattr(retriever, "runtime") and "forward" in getattr(retriever, "keys", {})
        if self.memory.split != "memory":
            raise ValueError("Action retrieval can index only memory episodes")

    @torch.no_grad()
    def distances(self, context):
        if context.latents.shape[0] != 1:
            raise ValueError("Context retrieval requires one query context")
        device = self.memory.contexts.device
        result = torch.empty(len(self.memory), device=device, dtype=torch.float32)
        if self.cuvs:
            layout = self.base.layout
            width = layout.dimensions["context"]+layout.dimensions["past_actions"]
            query = torch.cat([
                context.latents.reshape(1, -1)*layout.factor("context"),
                context.past_actions.reshape(1, -1)*layout.factor("past_actions")], 1).to(dtype=torch.float32).contiguous()
            for start in range(0, len(self.memory), self.chunk_rows):
                end = min(start+self.chunk_rows, len(self.memory))
                keys = self.base.keys["forward"][start:end, :width].contiguous()
                result[start:end] = self.base.runtime.distances(keys, query)
        else:
            for start in range(0, len(self.memory), self.chunk_rows):
                end = min(start+self.chunk_rows, len(self.memory))
                delta = (self.memory.contexts[start:end]-context.latents).flatten(1)
                value = delta.square().mean(1)/self.base.scales["context"]
                past = (self.memory.past_actions[start:end]-context.past_actions).flatten(1)
                if past.shape[1]:
                    value += past.square().mean(1)/self.base.scales["past_actions"]
                result[start:end] = value
        if not torch.isfinite(result).all():
            raise ValueError("Nonfinite context-only distances")
        return result

    @torch.no_grad()
    def selected_distances(self, context, indices):
        """Small direct calculation for uniform-control proximity diagnostics."""
        if not indices:
            return []
        rows = torch.as_tensor(indices, device=self.memory.contexts.device, dtype=torch.long)
        value = (self.memory.contexts[rows]-context.latents).flatten(1).square().mean(1)/self.base.scales["context"]
        past = (self.memory.past_actions[rows]-context.past_actions).flatten(1)
        if past.shape[1]:
            value += past.square().mean(1)/self.base.scales["past_actions"]
        return value.cpu().tolist()

    def description(self):
        width = self.memory.contexts.shape[1]*self.memory.contexts.shape[2] + self.memory.past_actions.shape[1]*self.memory.past_actions.shape[2]
        return {"backend": "cuvs_exact_chunked_context_prefix" if self.cuvs else "torch_exact_chunked_context",
                "metric": "context_MSE / context_scale + past_action_MSE / past_action_scale",
                "ties": "memory row", "query_fields": ["context_latents", "past_observed_actions"],
                "chunk_rows": self.chunk_rows, "context_key_width": width,
                "additional_persistent_key_bytes": 0,
                "key_chunk_bytes_upper_bound": min(len(self.memory), self.chunk_rows)*width*4,
                "distance_vector_bytes": len(self.memory)*4,
                "note": "Scratch estimates exclude native workspace and top-k/tie-selection buffers"}

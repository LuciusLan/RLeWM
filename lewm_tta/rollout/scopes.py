"""Selectable predictor/action scopes with the legacy last-block path intact."""

from __future__ import annotations

import torch

from ..adaptation import AdaptationEngine
from ..model import Context, LeWMAdapter
from .config import SCOPES


class ScopedLeWMAdapter(LeWMAdapter):
    def __init__(self, repo, checkpoint, device="cuda", scope="last-block"):
        super().__init__(repo, checkpoint, device)
        # Keep the original model/encoder fingerprint for existing cached data.
        # Scope selection belongs to the experiment identity, not the encoding.
        self._legacy_names = tuple(self.parameter_names)
        self._legacy_module = self.adaptable_module
        self._anchors = dict(zip(self.parameter_names, self.anchor))
        self.scope = None
        self.set_scope(scope)

    def set_scope(self, scope: str) -> None:
        if scope not in SCOPES:
            raise ValueError(f"Unknown adaptation scope: {scope}")
        if self.scope == scope:
            return
        roots = ((self._legacy_module,) if scope == "last-block" else
                 ("predictor", "pred_proj") if scope == "predictor" else
                 ("predictor", "pred_proj", "action_encoder"))
        self.model.requires_grad_(False)
        self.model.zero_grad(set_to_none=True)
        selected = [(name, parameter) for name, parameter in self.model.named_parameters()
                    if any(name.startswith(root+".") for root in roots)]
        if not selected or (scope == "last-block" and tuple(name for name, _ in selected) != self._legacy_names):
            raise AssertionError("Adaptation parameter selection differs from the expected modules")
        self.scope = scope
        self.adaptable_module = "+".join(roots)
        self.parameter_names = [name for name, _ in selected]
        self.parameters = [parameter for _, parameter in selected]
        for name, parameter in selected:
            parameter.requires_grad_(True)
            if name not in self._anchors:
                self._anchors[name] = self.checkpoint[name].to(self.device)
        self.anchor = [self._anchors[name] for name in self.parameter_names]
        self.parameter_count = sum(parameter.numel() for parameter in self.parameters)
        self.model.eval()

    def predict(self, context: Context, action: torch.Tensor, gradients: bool = False):
        if self.scope == "last-block":
            # Exactly the original path, including detached action embeddings.
            return super().predict(context, action, gradients)
        if context.latents.shape[1:] != (self.history, self.latent_dim):
            raise ValueError("Context latent shape does not match the checkpoint")
        if context.past_actions.shape[1:] != (self.history-1, self.action_dim):
            raise ValueError("Past-action history shape does not match the checkpoint")
        actions = torch.cat((context.past_actions, action.unsqueeze(1)), dim=1)
        # The joint scope must retain this graph. Merely adding action-encoder
        # parameters to AdamW would fail while using the legacy no_grad block.
        with torch.enable_grad() if gradients else torch.no_grad():
            action_latents = self.model.action_encoder(actions)
            return self.model.predict(context.latents, action_latents)[:, -1]

    def scope_description(self) -> dict:
        groups = {}
        for name, parameter in self.model.named_parameters():
            if name in self.parameter_names:
                root = name.split(".", 1)[0]
                groups[root] = groups.get(root, 0) + parameter.numel()
        return {"scope": self.scope, "trainable_module_roots": list(groups),
                "trainable_parameters": self.parameter_count, "parameters_by_module": groups,
                "parameter_names": self.parameter_names,
                "visual_encoder_and_latent_projector_frozen": True,
                "normalization_statistics_fixed": True}

    @torch.no_grad()
    def module_metrics(self) -> dict:
        selected = set(self.parameter_names)
        metrics = {}
        for root in ("predictor", "pred_proj", "action_encoder"):
            values = [(name, parameter) for name, parameter in self.model.named_parameters()
                      if name.startswith(root+".") and name in selected]
            if not values:
                continue
            count = sum(parameter.numel() for _, parameter in values)
            gradient = torch.zeros((), device=self.device)
            distance = torch.zeros((), device=self.device)
            tensors_with_grad = 0
            for name, parameter in values:
                distance += (parameter.detach()-self._anchors[name]).square().sum()
                if parameter.grad is not None:
                    gradient += parameter.grad.detach().square().sum()
                    tensors_with_grad += 1
            if not torch.isfinite(gradient) or not torch.isfinite(distance):
                raise ValueError("Nonfinite per-module adaptation metrics")
            metrics[root] = {"trainable_parameters": count, "parameter_tensors_with_gradient": tensors_with_grad,
                "last_update_gradient_l2_after_clipping": float(gradient.sqrt()),
                "displacement_from_checkpoint_mse": float(distance/count),
                "displacement_from_checkpoint_l2": float(distance.sqrt())}
        return metrics


class ScopeAdaptationEngine(AdaptationEngine):
    def run(self, query, method: str, seed: int) -> dict:
        record = super().run(query, method, seed)
        record["module_updates"] = self.adapter.module_metrics()
        if record["rounds"] and any(not row["parameter_tensors_with_gradient"]
                                    for row in record["module_updates"].values()):
            raise AssertionError("An intended trainable module received no adaptation gradient")
        return record

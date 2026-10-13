"""Native LeWM loading and a latent-only interface reusable for later stages."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import random
import subprocess
import sys

import numpy as np
import torch

from .runtime import file_hash, object_hash, package_versions


@dataclass(frozen=True)
class Context:
    latents: torch.Tensor             # (batch, history, latent_dim)
    past_actions: torch.Tensor        # (batch, history-1, ordered_block_dim)


def load_native_model(repo: Path, checkpoint: Path):
    import stable_pretraining as spt
    repo, checkpoint = Path(repo).resolve(), Path(checkpoint).resolve()
    sys.path.insert(0, str(repo))
    from jepa import JEPA
    from module import ARPredictor, Embedder, MLP
    config = json.loads((checkpoint / "config.json").read_text())
    encoder_cfg = {key: value for key, value in config["encoder"].items() if not key.startswith("_")}
    if encoder_cfg.get("pretrained", False):
        raise ValueError("Expected a local weights-only checkpoint with pretrained=False")
    encoder = spt.backbone.utils.vit_hf(**encoder_cfg)
    def mlp(key):
        cfg = config[key]
        return MLP(input_dim=cfg["input_dim"], output_dim=cfg["output_dim"],
                   hidden_dim=cfg["hidden_dim"], norm_fn=torch.nn.BatchNorm1d)
    model = JEPA(encoder=encoder,
        predictor=ARPredictor(**{k: v for k, v in config["predictor"].items() if not k.startswith("_")}),
        action_encoder=Embedder(**{k: v for k, v in config["action_encoder"].items() if not k.startswith("_")}),
        projector=mlp("projector"), pred_proj=mlp("pred_proj"))
    state = torch.load(checkpoint / "weights.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    model.eval().requires_grad_(False)
    return model, state, config


class LeWMAdapter:
    def __init__(self, repo: Path, checkpoint: Path, device: str = "cuda"):
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable; check the launcher or use --device cpu")
        self.device = torch.device(device)
        self.model, state, self.config = load_native_model(repo, checkpoint)
        self.checkpoint = {name: tensor.detach().clone() for name, tensor in state.items()}
        self.buffer_checkpoint = {name: value.detach().cpu().clone()
                                  for name, value in self.model.named_buffers()}
        self.model.to(self.device)
        self.history = self.config["predictor"]["num_frames"]
        self.image_size = self.config["encoder"]["image_size"]
        self.latent_dim = self.config["projector"]["output_dim"]
        self.action_dim = self.config["action_encoder"]["input_dim"]
        block = len(self.model.predictor.transformer.layers) - 1
        self.adaptable_module = f"predictor.transformer.layers.{block}"
        self.parameters = list(self.model.predictor.transformer.layers[block].parameters())
        self.parameter_names = [name for name, _ in self.model.named_parameters()
                                if name.startswith(self.adaptable_module + ".")]
        self.model.predictor.transformer.layers[block].requires_grad_(True)
        self.anchor = [self.checkpoint[name].to(self.device) for name in self.parameter_names]
        self.parameter_count = sum(parameter.numel() for parameter in self.parameters)
        commit = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"], check=True,
                                capture_output=True, text=True).stdout.strip()
        self.identity = {"checkpoint": str(checkpoint), "weights_sha256": file_hash(Path(checkpoint) / "weights.pt"),
            "config_sha256": file_hash(Path(checkpoint) / "config.json"), "lewm_commit": commit,
            "native_code": {name: file_hash(Path(repo) / name) for name in ("jepa.py", "module.py", "utils.py")},
            "adapter_sha256": file_hash(Path(__file__)),
            "packages": package_versions(), "precision": "float32, TF32 disabled in P1"}
        self.fingerprint = object_hash(self.identity)

    def reset(self, seed: int) -> None:
        self.model.load_state_dict(self.checkpoint, strict=True)
        with torch.no_grad():
            for name, value in self.model.named_buffers():
                value.copy_(self.buffer_checkpoint[name])
        self.model.eval()
        self.model.zero_grad(set_to_none=True)
        random.seed(seed)
        np.random.seed(seed % (2**32))
        torch.manual_seed(seed)
        if self.device.type == "cuda":
            torch.cuda.manual_seed_all(seed)

    @torch.no_grad()
    def encode(self, pixels: torch.Tensor) -> torch.Tensor:
        # Normal no_grad tensors can be saved for backward during predictor TTA.
        # Do not create adaptation inputs inside torch.inference_mode().
        latents = self.model.encode({"pixels": pixels.to(self.device)})["emb"]
        if not torch.isfinite(latents).all():
            raise ValueError("Nonfinite encoder/projector output")
        return latents.detach()

    def predict(self, context: Context, action: torch.Tensor, gradients: bool = False) -> torch.Tensor:
        if context.latents.shape[1:] != (self.history, self.latent_dim):
            raise ValueError("Context latent shape does not match the checkpoint")
        if context.past_actions.shape[1:] != (self.history - 1, self.action_dim):
            raise ValueError("Past-action history shape does not match the checkpoint")
        actions = torch.cat((context.past_actions, action.unsqueeze(1)), dim=1)
        with torch.no_grad():
            action_latents = self.model.action_encoder(actions)
        with torch.enable_grad() if gradients else torch.no_grad():
            return self.model.predict(context.latents, action_latents)[:, -1]

    @torch.no_grad()
    def rollout(self, context: Context, future_actions: torch.Tensor) -> torch.Tensor:
        """Freeze one predictor through a rollout; future_actions has (B, horizon, A)."""
        predictions = []
        for action in future_actions.unbind(1):
            prediction = self.predict(context, action)
            predictions.append(prediction)
            context = Context(torch.cat((context.latents[:, 1:], prediction.unsqueeze(1)), dim=1),
                              torch.cat((context.past_actions[:, 1:], action.unsqueeze(1)), dim=1))
        return torch.stack(predictions, dim=1)

    def anchor_loss(self) -> torch.Tensor:
        return sum((parameter - reference).square().sum()
                   for parameter, reference in zip(self.parameters, self.anchor)) / self.parameter_count

    def check_fixed_state(self, check_parameters: bool = True) -> None:
        if any(module.training for module in self.model.modules()):
            raise AssertionError("Dropout/BatchNorm must remain in evaluation mode")
        for name, value in self.model.named_buffers():
            torch.testing.assert_close(value.cpu(), self.buffer_checkpoint[name], rtol=0, atol=0)
        if check_parameters:
            for name, value in self.model.named_parameters():
                if name not in self.parameter_names:
                    torch.testing.assert_close(value.detach().cpu(), self.checkpoint[name], rtol=0, atol=0)

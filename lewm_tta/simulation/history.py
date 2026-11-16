"""Causal observed history, independent of simulator state and task metrics."""

from collections import deque
import numpy as np
import torch

from ..model import Context


def preprocess_frame(image, transform):
    """Match HDF5Dataset: uint8 HWC -> Torch CHW *before* native transforms."""
    image = np.asarray(image)
    if image.ndim != 3 or image.shape[-1] != 3 or image.dtype != np.uint8:
        raise ValueError("Expected one uint8 HWC RGB simulator frame")
    pixels = torch.from_numpy(image.copy()).permute(2, 0, 1).contiguous()
    return transform({"pixels": pixels})["pixels"]


def bootstrap(environment, dataset, case, seed, history, frameskip):
    """Common observed history from recorded past or explicit reset-case controls."""
    observation = environment.reset(case, seed, history_start=True)
    images, blocks = [observation.rgb], []
    start, current = case["start_raw_row"], case["current_raw_row"]
    raw = (np.asarray(case["warmup_raw_actions"], dtype=np.float32) if "warmup_raw_actions" in case
           else dataset.rows("action", start, current))
    if len(raw) != (history-1)*frameskip:
        raise ValueError("Warm-up actions do not span the complete past context")
    for index in range(history-1):
        block = raw[index*frameskip:(index+1)*frameskip]
        for action in block:
            observation = environment.step(action, recorded=True)
            if observation.truncated:
                raise ValueError("Simulator truncated during common history warm-up")
        images.append(observation.rgb)
        blocks.append(block)
    return images, blocks, observation


class ObservedHistory:
    def __init__(self, adapter, coordinates, images, blocks):
        from utils import get_img_preprocessor
        self.adapter, self.coordinates = adapter, coordinates
        self.transform = get_img_preprocessor("pixels", "pixels", adapter.image_size)
        self.latents = deque(maxlen=adapter.history)
        self.actions = deque(maxlen=adapter.history-1)
        if len(images) != adapter.history or len(blocks) != adapter.history-1:
            raise ValueError("Observed history has the wrong number of frames/actions")
        self.latents.extend(self.encode(image) for image in images)
        self.actions.extend(self.normalize(block) for block in blocks)

    def encode(self, image):
        # Exactly the native cache transform; only simulator RGB enters here.
        pixels = preprocess_frame(image, self.transform)
        return self.adapter.encode(pixels[None, None])[:, 0]

    def normalize(self, raw_block):
        raw = torch.as_tensor(np.asarray(raw_block), dtype=torch.float32, device=self.adapter.device)
        if raw.shape != (self.coordinates.frameskip, len(self.coordinates.mean)):
            raise ValueError("A history update must contain one complete ordered raw action block")
        return self.coordinates.from_raw(raw).unsqueeze(0)

    def append(self, image, raw_block):
        self.latents.append(self.encode(image))
        self.actions.append(self.normalize(raw_block))

    def context(self):
        actions = (torch.stack(list(self.actions), dim=1) if self.actions else
                   self.latents[0].new_empty((1, 0, self.adapter.action_dim)))
        return Context(torch.stack(list(self.latents), dim=1), actions)

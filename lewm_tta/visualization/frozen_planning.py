"""Frozen-only preview controller, independent of task geometry and retrieval.

The short-history view follows the native LeWM rollout: start with the chosen
number of observations, then grow the imagined context up to the trained limit.
It does not alter the shared adapter or any cached observation embeddings.
"""

from types import SimpleNamespace

import torch

from ..model import Context
from .plans import load_planner, plan_once


class FrozenRolloutView:
    def __init__(self, adapter, observed_context):
        if type(observed_context) is not int or not 1 <= observed_context <= adapter.history:
            raise ValueError("Observed planning context must lie within the trained model history")
        self.adapter, self.observed_context = adapter, observed_context

    def __getattr__(self, name):
        return getattr(self.adapter, name)

    @torch.no_grad()
    def rollout(self, context, future_actions):
        if self.observed_context == self.adapter.history:
            return self.adapter.rollout(context, future_actions)
        count = self.observed_context
        context = Context(context.latents[:, -count:], context.past_actions[:, -(count-1):]
                          if count > 1 else context.past_actions[:, :0])
        predictions = []
        for action in future_actions.unbind(1):
            actions = torch.cat((context.past_actions, action[:, None]), 1)
            prediction = self.model.predict(context.latents, self.model.action_encoder(actions))[:, -1]
            predictions.append(prediction)
            latents = torch.cat((context.latents, prediction[:, None]), 1)[:, -self.adapter.history:]
            past = actions[:, -(latents.shape[1]-1):] if latents.shape[1] > 1 else actions[:, :0]
            context = Context(latents, past)
        return torch.stack(predictions, 1)

    def prediction_contract(self):
        return {"initial_observed_context_frames": self.observed_context,
                "maximum_predictor_history": self.adapter.history,
                "imagined_history": "grow from initial observed count to the trained maximum, following native LeWM rollout",
                "encoding_and_physical_bootstrap": "existing H3/f5 protocol retained; unused older observations never enter prediction"}


class FrozenGaussianController:
    def __init__(self, config, device, cpu_threads, observed_context):
        adapter, self.coordinates, _ = load_planner(config, device, cpu_threads, "frozen", "gaussian")
        self.resources = SimpleNamespace(adapter=FrozenRolloutView(adapter, observed_context))
        self.config = config

    def plan(self, context, goal, metadata, method, source, seed, warm_start=None):
        if method != "frozen" or source != "gaussian":
            raise ValueError("This preview controller supports only Frozen/Gaussian")
        return plan_once(self.resources.adapter, self.coordinates, None, self.config, context, goal,
                         metadata, method, source, seed, warm_start=warm_start)

"""User-run short-history contracts; no learned checkpoint or native simulator.

PYTHONPATH=codes python -m unittest discover -s codes/tests -p test_frozen_planning.py
"""

from types import SimpleNamespace
from contextlib import nullcontext
import unittest
from unittest.mock import MagicMock, patch, sentinel

import numpy as np
import torch

from lewm_tta.model import Context
from lewm_tta.visualization.frozen_planning import FrozenGaussianController, FrozenRolloutView
from lewm_tta.visualization.plans import plan_once


class ToyModel:
    def __init__(self):
        self.calls = []

    def action_encoder(self, actions):
        return actions

    def predict(self, latents, actions):
        self.calls.append((latents.clone(), actions.clone()))
        return latents+actions


class FrozenPlanningContracts(unittest.TestCase):
    def test_one_observed_frame_grows_imagined_context_like_native_rollout(self):
        model = ToyModel()
        adapter = SimpleNamespace(history=3, model=model)
        view = FrozenRolloutView(adapter, 1)
        # Very different older observations/actions must not contaminate H1.
        context = Context(torch.tensor([[[999., 999.], [500., 500.], [2., 2.]]]),
                          torch.full((1, 2, 2), 1000.))
        future = torch.ones(1, 5, 2)
        predicted = view.rollout(context, future)
        torch.testing.assert_close(predicted[0, :, 0], torch.arange(3., 8.))
        self.assertEqual([latents.shape[1] for latents, actions in model.calls], [1, 2, 3, 3, 3])
        self.assertTrue(all(torch.equal(actions, torch.ones_like(actions)) for latents, actions in model.calls))
        self.assertEqual(view.history, 3)  # Physical bootstrap / encoding contract stays unchanged.

    def test_full_history_delegates_to_existing_adapter_without_new_math(self):
        expected = torch.ones(1, 2, 2)
        seen = []
        def existing(context, actions):
            seen.append((context, actions))
            return expected
        context, actions = object(), torch.zeros(1, 2, 2)
        view = FrozenRolloutView(SimpleNamespace(history=3, rollout=existing), 3)
        self.assertIs(view.rollout(context, actions), expected)
        self.assertEqual(seen, [(context, actions)])

    def test_invalid_context_is_rejected(self):
        for count in (0, 4, True):
            with self.assertRaises(ValueError):
                FrozenRolloutView(SimpleNamespace(history=3), count)


class FrozenControllerCallContracts(unittest.TestCase):
    def test_controller_matches_helper_signature_and_preserves_warm_start(self):
        # Bypass checkpoint loading. Autospec enforces the real helper API.
        controller = FrozenGaussianController.__new__(FrozenGaussianController)
        controller.resources = SimpleNamespace(adapter=sentinel.adapter)
        controller.coordinates, controller.config = sentinel.coordinates, sentinel.config
        for warm in (None, sentinel.warm):
            with self.subTest(warm=warm), patch("lewm_tta.visualization.frozen_planning.plan_once",
                                               autospec=True, return_value=sentinel.result) as helper:
                result = controller.plan(sentinel.context, sentinel.goal, sentinel.metadata,
                                         "frozen", "gaussian", 42, warm)
                self.assertIs(result, sentinel.result)
                helper.assert_called_once_with(sentinel.adapter, sentinel.coordinates, None, sentinel.config,
                                               sentinel.context, sentinel.goal, sentinel.metadata,
                                               "frozen", "gaussian", 42, warm_start=warm)

    def test_single_plan_helper_forwards_warm_start_to_cem(self):
        adapter, coordinates = MagicMock(), MagicMock()
        coordinates.to_raw.return_value.detach.return_value.cpu.return_value.numpy.return_value = np.zeros((1, 5, 2))
        plan = SimpleNamespace(actions=MagicMock(), cost=1., candidate_evaluations=1,
                               first_population_fingerprint="population")
        config = {"planning": {"scoring_batch_size": 64}}
        for warm in (None, sentinel.warm):
            with self.subTest(warm=warm), \
                    patch("lewm_tta.rollout.support_updates.synchronize"), \
                    patch("lewm_tta.planning.actions.stream_seed", return_value=7), \
                    patch("lewm_tta.rollout.inference.fixed_predictor", return_value=nullcontext()), \
                    patch("lewm_tta.planning.cem.solve_cem", return_value=plan) as solver:
                # Omitting the optional argument exercises the original 10-argument call.
                options = {} if warm is None else {"warm_start": warm}
                plan_once(adapter, coordinates, None, config, sentinel.context, sentinel.goal,
                          {"transition_id": "query"}, "frozen", "gaussian", 42, **options)
                self.assertIs(solver.call_args.kwargs["initial_mean"], warm)

    def test_delegated_controller_receives_warm_start(self):
        controller = MagicMock()
        plan_once(sentinel.adapter, sentinel.coordinates, controller, sentinel.config,
                  sentinel.context, sentinel.goal, sentinel.metadata, "frozen", "gaussian", 42,
                  warm_start=sentinel.warm)
        controller.plan.assert_called_once_with(sentinel.context, sentinel.goal, sentinel.metadata,
                                                "frozen", "gaussian", 42, warm_start=sentinel.warm)


if __name__ == "__main__":
    unittest.main()

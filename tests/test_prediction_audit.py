"""User-run synthetic diagnostics; no learned model or native simulator.

PYTHONPATH=codes python -m unittest discover -s codes/tests -p test_prediction_audit.py
"""

from types import SimpleNamespace
import unittest

import numpy as np
import torch

from lewm_tta.model import Context
from lewm_tta.planning.actions import ActionCoordinates
from lewm_tta.visualization.prediction_audit import compare_prediction, control_probes, execute_probe


class LinearAdapter:
    device = torch.device("cpu")
    latent_dim = 2

    def predict(self, context, action):
        # Read raw controls after undoing the fixture's nontrivial normalization.
        raw = action.reshape(-1, 2, 2)*torch.tensor([.2, .4])+torch.tensor([.1, -.2])
        return context.latents[:, -1]+raw.sum(1)

    def rollout(self, context, actions):
        predicted = []
        for action in actions.unbind(1):
            next_latent = self.predict(context, action)
            predicted.append(next_latent)
            context = Context(torch.cat((context.latents[:, 1:], next_latent[:, None]), 1),
                              torch.cat((context.past_actions[:, 1:], action[:, None]), 1))
        return torch.stack(predicted, 1)


class PredictionAuditContracts(unittest.TestCase):
    def setUp(self):
        self.coordinates = ActionCoordinates({"mean": [.1, -.2], "std": [.2, .4]},
                                            {"raw_action_low": [-1, -1], "raw_action_high": [1, 1]}, 2, "cpu")
        self.context = Context(torch.zeros(1, 3, 2), torch.zeros(1, 2, 4))

    def test_raw_zero_is_preserved_with_nonzero_normalization_mean(self):
        controls = control_probes(np.full((4, 2), .3, dtype=np.float32))
        self.assertEqual(len(controls), 6)
        np.testing.assert_array_equal(controls["raw-zero"], np.zeros((4, 2)))
        self.assertAlmostEqual(float(controls["minus-y"][0, 1]), -.2)
        value = compare_prediction(LinearAdapter(), self.coordinates, self.context,
                                   controls["raw-zero"], torch.zeros(1, 2, 2), {"goal": torch.ones(1, 2)})
        self.assertLess(value["raw_roundtrip_max_abs_error"], 1e-7)
        self.assertEqual(value["autoregressive_mse_by_block"], [0., 0.])
        self.assertEqual(value["goal_sse"]["goal"]["predicted_terminal"], 2.)

    def test_teacher_forcing_uses_observed_successors_instead_of_own_predictions(self):
        actual = torch.tensor([[[1., 2.], [3., 4.]]])
        result = compare_prediction(LinearAdapter(), self.coordinates, self.context,
                                    np.zeros((4, 2)), actual, {"goal": actual[:, -1]})
        self.assertEqual(result["autoregressive_mse_by_block"], [2.5, 12.5])
        self.assertEqual(result["teacher_forced_mse_by_block"], [2.5, 4.])
        self.assertEqual(result["goal_sse"]["goal"]["observed_terminal"], 0.)

    def test_every_physical_probe_restores_before_branching(self):
        class Environment:
            def observe(self):
                return SimpleNamespace(rgb=np.full((2, 2, 3), self.position, dtype=np.uint8),
                                       state=np.array([self.position]), metrics={"position": self.position}, truncated=False)

            def reset(self):
                self.position = 10
                return self.observe()

            def step(self, action):
                self.position += 1
                return self.observe()
        environment = Environment()
        for _ in range(2):
            trace = execute_probe(environment, environment.reset, np.zeros((4, 2)), 2)
            self.assertEqual(trace["states"], [[10], [11], [12], [13], [14]])
            self.assertEqual(len(trace["keyframes"]), 2)
            self.assertEqual([int(image[0, 0, 0]) for image in trace["keyframes"]], [12, 14])


if __name__ == "__main__":
    unittest.main()

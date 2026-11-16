"""User-run CEM/source-sampling parity, without weights or a simulator.

PYTHONPATH=codes python -m unittest discover -s codes/tests -p test_lewm_reference.py
"""

from types import SimpleNamespace
import unittest

import numpy as np
import torch
from gymnasium.spaces import Box
from stable_worldmodel import PlanConfig
from stable_worldmodel.solver import CEMSolver

from lewm_tta.planning.cem import solve_cem
from lewm_tta.visualization.lewm_reference import select_rows


class QuadraticCost(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dummy = torch.nn.Parameter(torch.zeros(()))
        self.populations = []

    def get_cost(self, info, candidates):
        self.populations.append(candidates.detach().clone())
        weights = torch.linspace(1., 3., candidates.shape[-1])
        return ((candidates - .37).square() * weights).sum(dim=(-2, -1))


class ReferenceContracts(unittest.TestCase):
    def test_cohort_matches_original_eval_loop_including_last_row_omission(self):
        episodes = np.repeat(np.arange(4), [41, 62, 55, 39])
        steps = np.concatenate([np.arange(count) for count in [41, 62, 55, 39]])
        lengths = {int(ep): max(steps[episodes == ep]) + 1 for ep in np.unique(episodes)}
        valid = np.nonzero(steps <= np.array([lengths[int(ep)] - 26 for ep in episodes]))[0]
        expected = np.sort(valid[np.random.default_rng(42).choice(len(valid) - 1, 50, replace=False)])
        actual, count = select_rows(episodes, steps, 50, 42)
        np.testing.assert_array_equal(actual, expected)
        self.assertEqual(count, len(valid))

    def test_project_and_native_cem_match_when_rng_and_action_bounds_are_matched(self):
        settings = {"horizon": 5, "population": 64, "iterations": 5, "elites": 8,
                    "initial_std": 1., "minimum_std": 0.}
        coordinates = SimpleNamespace(low=torch.zeros(10), width=10, clip=lambda value: value)
        for warm in (None, torch.full((2, 10), .2)):
            with self.subTest(warm_start=warm is not None):
                native_cost = QuadraticCost()
                native = CEMSolver(native_cost, batch_size=1, num_samples=64, n_steps=5,
                                   topk=8, var_scale=1., device="cpu", seed=42)
                native.configure(action_space=Box(-1., 1., shape=(1, 2)), n_envs=1,
                                 config=PlanConfig(horizon=5, receding_horizon=5, action_block=5))
                expected = native({"dummy": torch.zeros(1, 1)},
                                  init_action=None if warm is None else warm[None])["actions"][0]
                project_cost = QuadraticCost()
                actual = solve_cem(lambda candidates: project_cost.get_cost({}, candidates[None])[0],
                                   settings, coordinates, 42, initial_mean=warm)
                for upstream, project in zip(native_cost.populations, project_cost.populations):
                    torch.testing.assert_close(upstream, project, rtol=2e-6, atol=2e-6)
                torch.testing.assert_close(actual.actions, expected, rtol=2e-6, atol=2e-6)
                # Our extra evaluation scores the returned mean without changing it.
                self.assertEqual(len(project_cost.populations), len(native_cost.populations) + 1)


if __name__ == "__main__":
    unittest.main()

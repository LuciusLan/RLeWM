"""Synthetic user-run checks; no simulator, weights or project dataset access."""

from pathlib import Path
import tempfile
import unittest

import h5py
import numpy as np

from .collection import numpy_stream, write_episode
from .protocol import array_digest, seed_for


class Contracts(unittest.TestCase):
    def test_oracle_rng_is_isolated_and_continues_between_calls(self):
        original = np.random.get_state()
        first, expected = np.random.RandomState(42), np.random.RandomState(42)
        with numpy_stream(first):
            np.testing.assert_array_equal(np.random.uniform(size=5), expected.uniform(size=5))
        # Another policy can consume a different number of random draws.
        with numpy_stream(np.random.RandomState(17)):
            np.random.uniform(size=100)
        with numpy_stream(first):
            np.testing.assert_array_equal(np.random.uniform(size=5), expected.uniform(size=5))
        restored = np.random.get_state()
        for left, right in zip(original, restored):
            np.testing.assert_equal(left, right)

    def test_synthetic_episode_preserves_raw_action_and_terminal_mask(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'episode.h5'
            frames = [{'pixels': np.zeros((224, 224, 3), dtype=np.uint8), 'qpos': np.zeros(21)} for _ in range(3)]
            actions = [np.array([.1, .2, -.3, .4, -.5], dtype=np.float32), np.ones(5, dtype=np.float32)]
            case = {'episode_id': 1_500_000_000}
            write_episode(path, frames, actions, case)
            with h5py.File(path, 'r', swmr=True) as stream:
                np.testing.assert_array_equal(stream['action'][:-1], actions)
                np.testing.assert_array_equal(stream['action_valid'][:], [1, 1, 0])
                np.testing.assert_array_equal(stream['step_idx'][:], [0, 1, 2])
                np.testing.assert_array_equal(stream['episode_idx'][:], [case['episode_id']]*3)
                self.assertEqual(stream['ep_len'][0], 3)
                self.assertEqual(stream['pixels'].shape[0], 3)
            self.assertFalse(path.with_suffix('.partial.h5').exists())

    def test_case_stream_seeds_and_physics_receipts(self):
        seeds = [seed_for(20261113, case, stream) for case in range(16) for stream in ('environment', 'oracle', 'noise', 'targets')]
        self.assertEqual(len(seeds), len(set(seeds)))
        self.assertEqual(seed_for(20261113, 0, 'environment'), seeds[0])
        first = {'qpos': np.array([1., 2.]), 'qvel': np.zeros(2)}
        self.assertEqual(array_digest(first), array_digest(dict(reversed(list(first.items())))))
        second = {key: value.copy() for key, value in first.items()}
        second['qpos'][1] += 1e-12
        self.assertNotEqual(array_digest(first), array_digest(second))


if __name__ == '__main__':
    unittest.main()

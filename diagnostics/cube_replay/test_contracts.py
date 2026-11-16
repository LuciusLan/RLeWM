"""Small synthetic contract tests. Never opens project datasets or simulators.

User-run: PYTHONPATH=codes python -m unittest diagnostics.cube_replay.test_contracts
"""

from pathlib import Path
import tempfile
import unittest

import h5py
import numpy as np

from .data import RecordedSource
from .report import compare, summarize


class Contracts(unittest.TestCase):
    def test_terminal_action_is_never_executed_or_hashed_as_input(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'episode.h5'
            with h5py.File(path, 'w', libver='latest') as stream:
                qpos = np.zeros((3, 21))
                qpos[:, -4] = 1
                stream['qpos'], stream['qvel'] = qpos, np.zeros((3, 20))
                stream['action'] = np.vstack([np.zeros((2, 5)), np.full((1, 5), np.nan)])
                stream['time'] = np.arange(3).reshape(-1, 1)*.05
                stream['prev_qpos'] = qpos
            source = RecordedSource(path, render=False)
            try:
                case = source.select(1, [0], 2, 42)[0]
                values, pixels, receipt = source.load(case, [])
                self.assertEqual(values['action'].shape, (2, 5))
                self.assertEqual(values['qpos'].shape[0], 3)
                self.assertTrue(np.isfinite(values['action']).all())
                self.assertEqual(pixels, {})
                self.assertTrue(receipt['stored_prev_qpos_matches_preceding_row'])
                with self.assertRaises(ValueError):
                    source.select(1, [0], 3, 42)
                source.assert_unchanged()
            finally:
                source.close()

    def test_missing_virtual_dependency_rejected_instead_of_silent_fill(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'missing.h5'
            with h5py.File(path, 'w', libver='latest') as stream:
                for key, shape in [('qpos', (2, 21)), ('qvel', (2, 20)), ('action', (2, 5)), ('time', (2,))]:
                    layout = h5py.VirtualLayout(shape, dtype='float64')
                    layout[:] = h5py.VirtualSource(str(Path(directory)/'not-here.h5'), key, shape=shape)
                    stream.create_virtual_dataset(key, layout)
            with self.assertRaises(FileNotFoundError):
                RecordedSource(path, render=False)

    def test_sign_equivalent_quaternions_and_separate_rgb_decision(self):
        qpos = np.zeros(21)
        qpos[-4] = 1
        values = {'qpos': np.stack([qpos, qpos]), 'qvel': np.zeros((2, 20)),
                  'control': np.zeros((2, 7)), 'time': np.array([0., .05])}
        actual = {key: value[0].copy() for key, value in values.items()}
        actual['qpos'][-4:] *= -1
        limits = {'qpos': 1e-5, 'qvel': 1e-4, 'control': 1e-5, 'time': 1e-8, 'rgb_rmse': 2.}
        black, white = np.zeros((2, 2, 3), dtype=np.uint8), np.full((2, 2, 3), 255, dtype=np.uint8)
        first = compare(values, actual, 0, limits, black, white)
        self.assertTrue(first['state_passed'])
        self.assertFalse(first['rgb_passed'])
        self.assertEqual(first['cube_orientation_error_radians'], 0.)
        self.assertEqual(first['rgb_rmse_255'], 255.)
        actual['control'][0] = .1
        actual['time'] = np.asarray(.05)
        second = compare(values, actual, 1, limits)
        self.assertFalse(second['state_passed'])
        summary = summarize([first, second])
        self.assertEqual(summary['first_state_mismatch_step'], 1)
        self.assertEqual(summary['first_rgb_mismatch_step'], 0)
        self.assertFalse(summary['passed'])


if __name__ == '__main__':
    unittest.main()

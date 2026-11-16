"""User-run virtual-HDF5 read regression; no learned model or simulator.

PYTHONPATH=codes python -m unittest discover -s codes/tests -p test_simulation_dataset.py
"""

from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock

import h5py
import numpy as np

from lewm_tta.simulation.dataset import SourceDataset


class SimulationDatasetContracts(unittest.TestCase):
    def test_virtual_episode_boundary_allows_empty_past_prefix(self):
        # Like the fresh export, concatenate separate episode files virtually.
        # The second episode begins at a nonzero global row, as in the failure.
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            reference = np.arange(12, dtype=np.float32).reshape(6, 2)
            for episode in range(2):
                with h5py.File(root/f"episode-{episode}.h5", "w", libver="latest") as source:
                    source.create_dataset("action", data=reference[episode*3:(episode+1)*3])
                    source.create_dataset("pixels", data=np.full((3, 2, 2, 3), episode, dtype=np.uint8))
            path = root/"dataset.h5"
            with h5py.File(path, "w", libver="latest") as output:
                for key, shape, dtype in (("action", (3, 2), "f4"), ("pixels", (3, 2, 2, 3), "u1")):
                    layout = h5py.VirtualLayout((6, *shape[1:]), dtype=dtype)
                    for episode in range(2):
                        layout[episode*3:(episode+1)*3] = h5py.VirtualSource(
                            str(root/f"episode-{episode}.h5"), key, shape=shape)
                    output.create_virtual_dataset(key, layout)
            dataset = SourceDataset(path)
            try:
                self.assertTrue(dataset.file["action"].is_virtual)
                for boundary in (0, 3, 6):
                    actions = dataset.rows("action", boundary, boundary)
                    self.assertEqual(actions.shape, (0, 2))
                    self.assertEqual(actions.dtype, np.dtype("float32"))
                    self.assertEqual(list(actions), [])
                    pixels = dataset.rows("pixels", boundary, boundary)
                    self.assertEqual(pixels.shape, (0, 2, 2, 3))
                    self.assertEqual(pixels.dtype, np.dtype("uint8"))
                # Nonempty reads still replay the actual prefix, including
                # slices crossing virtual source-file boundaries.
                np.testing.assert_array_equal(dataset.rows("action", 2, 5), reference[2:5])
                np.testing.assert_array_equal(dataset.rows("action", 3), reference[3])
                copied = dataset.rows("action", 3, 5)
                copied[:] = -1
                np.testing.assert_array_equal(dataset.rows("action", 3, 5), reference[3:5])
            finally:
                dataset.close()

    def test_empty_read_does_not_call_hdf5_and_other_read_errors_propagate(self):
        dataset = SourceDataset.__new__(SourceDataset)
        column = Mock()
        column.shape, column.dtype = (32964, 2), np.dtype("float32")
        # Deliberately model the native failure for any attempted payload read.
        column.__getitem__ = Mock(side_effect=OSError("unable to get selection bounds"))
        dataset.file = {"action": column}
        self.assertEqual(dataset.rows("action", 7035, 7035).shape, (0, 2))
        column.__getitem__.assert_not_called()
        with self.assertRaises(OSError):
            dataset.rows("action", 7035, 7045)
        with self.assertRaises(KeyError):
            dataset.rows("missing", 7035, 7035)


if __name__ == "__main__":
    unittest.main()

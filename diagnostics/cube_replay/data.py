"""Read-only raw/canonical/fresh HDF5 access with explicit row/action alignment."""

import hashlib
import json
from pathlib import Path

import h5py
import hdf5plugin  # noqa: F401 -- source compression filters
import numpy as np

from diagnostics.common import checked, identity, require, sha


FIELDS = {
    'qpos': ('qpos',), 'qvel': ('qvel',), 'action': ('action',), 'control': ('control',),
    'time': ('physics_time', 'time'), 'pixels': ('pixels',), 'observation': ('observation',),
    'mocap_pos': ('mocap_pos',), 'mocap_quat': ('mocap_quat',),
    'block_pos': ('privileged_block_0_pos', 'privileged/block_0_pos'),
    'block_quat': ('privileged_block_0_quat', 'privileged/block_0_quat'),
    'target_pos': ('privileged_target_block_pos', 'privileged/target_block_pos'),
    'target_yaw': ('privileged_target_block_yaw', 'privileged/target_block_yaw'),
    'target_block': ('privileged_target_block', 'privileged/target_block'),
    'effector_pos': ('proprio_effector_pos', 'proprio/effector_pos'),
    'gripper_opening': ('proprio_gripper_opening', 'proprio/gripper_opening'),
    'prev_qpos': ('prev_qpos',), 'prev_qvel': ('prev_qvel',),
    'act': ('act',), 'qacc_warmstart': ('qacc_warmstart',),
}


def array_hash(values):
    digest = hashlib.sha256()
    for key, value in sorted(values.items()):
        value = np.ascontiguousarray(value)
        digest.update(json.dumps([key, value.dtype.str, list(value.shape)]).encode())
        digest.update(value.tobytes())
    return digest.hexdigest()


def file_stat(path):
    info = path.stat()
    return {'bytes': info.st_size, 'mtime_ns': info.st_mtime_ns}


class RecordedSource:
    def __init__(self, path, render):
        self.path = Path(path).expanduser().resolve(strict=True)
        self.before = {str(self.path): file_stat(self.path)}
        self.file = h5py.File(self.path, 'r', swmr=True)
        try:
            self._inspect(render)
        except BaseException:
            self.file.close()
            raise

    def _inspect(self, render):
        # A restored raw file may have a different mtime. Read it directly instead
        # of rebinding a historical canonical view to it without verification.
        if 'lewm_source_view' in self.file.attrs:
            view = json.loads(self.file.attrs['lewm_source_view'])
            require(view['fingerprint'] == identity(view['identity']), 'Canonical-view checksum differs')
            raw = view['identity']['raw_source']
            require(file_stat(Path(raw['path'])) == {k: raw[k] for k in ('bytes', 'mtime_ns')},
                    'Canonical view has a changed raw dependency. Use --source RAW_HDF5; no view rebuild is needed here.')
        self.fields = {key: next((name for name in names if name in self.file), None)
                       for key, names in FIELDS.items()}
        required = ['qpos', 'qvel', 'action', 'time'] + (['pixels'] if render else [])
        require(all(self.fields[key] for key in required), f'Source requires fields {required}')
        rows = self.file[self.fields['qpos']].shape[0]
        require(self.file[self.fields['qpos']].shape == (rows, 21)
                and self.file[self.fields['qvel']].shape == (rows, 20)
                and self.file[self.fields['action']].shape == (rows, 5),
                'Expected single-Cube qpos[21], qvel[20], action[5]')
        for name in filter(None, self.fields.values()):
            column = self.file[name]
            require(column.shape[0] == rows, f'Inconsistent row count: {name}')
            self._dependency(Path(column.file.filename))
            if column.is_virtual:
                for mapping in column.virtual_sources():
                    filename = mapping.file_name
                    filename = filename.decode() if isinstance(filename, bytes) else filename
                    if filename != '.':
                        self._dependency(Path(filename), Path(column.file.filename).parent)
        if 'ep_len' in self.file or 'ep_offset' in self.file:
            require('ep_len' in self.file and 'ep_offset' in self.file, 'Incomplete episode metadata')
            self.lengths = np.asarray(self.file['ep_len'][:], dtype=np.int64)
            self.offsets = np.asarray(self.file['ep_offset'][:], dtype=np.int64)
        else:
            self.lengths, self.offsets = np.array([rows]), np.array([0])
        require(len(self.lengths) > 0 and self.offsets.shape == self.lengths.shape
                and np.all(self.lengths > 0)
                and np.array_equal(self.offsets, np.cumsum(np.r_[0, self.lengths[:-1]]))
                and int(self.lengths.sum()) == rows, 'Episodes must partition all rows')
        self.episode_column = next((name for name in ('episode_idx', 'ep_idx') if name in self.file), None)
        self.image_shape = None
        if render:
            pixels = self.file[self.fields['pixels']]
            require(len(pixels.shape) == 4 and pixels.shape[-1] == 3 and pixels.dtype == np.uint8,
                    'Replay expects stored uint8 RGB [row,height,width,3], without resize or normalization')
            self.image_shape = list(pixels.shape[1:])
        self.schema = {key: {'column': name, 'shape': list(self.file[name].shape),
                             'dtype': str(self.file[name].dtype)}
                       for key, name in self.fields.items() if name}

    def _dependency(self, path, parent=None):
        path = (parent/path if parent is not None and not path.is_absolute() else path).resolve(strict=True)
        self.before.setdefault(str(path), file_stat(path))

    def select(self, samples, episodes, steps, seed):
        eligible = np.flatnonzero(self.lengths >= steps+1)
        if episodes is None:
            require(len(eligible) >= samples, f'Only {len(eligible)} episodes have {steps} controls')
            episodes = sorted(np.random.default_rng(seed).choice(eligible, samples, replace=False).tolist())
        require(len(set(episodes)) == len(episodes) and episodes and set(episodes).issubset(eligible.tolist()),
                'Episode positions must be unique, in range, and long enough for --steps')
        return [{'episode_position': int(ep), 'start_raw_row': int(self.offsets[ep]),
                 'stored_frames': int(self.lengths[ep]), 'replay_controls': steps,
                 'episode_id': int(self.file[self.episode_column][int(self.offsets[ep])])
                 if self.episode_column else None} for ep in episodes]

    def load(self, case, render_steps):
        first, stop = case['start_raw_row'], case['start_raw_row']+case['replay_controls']+1
        values = {}
        for key, name in self.fields.items():
            if name and key != 'pixels':
                end = stop-1 if key == 'action' else stop
                values[key] = np.asarray(self.file[name][first:end]).copy()
                require(np.issubdtype(values[key].dtype, np.number) and np.isfinite(values[key]).all(),
                        f'Nonfinite/non-numeric selected data: {key}')
        values['time'] = values['time'].reshape(-1)
        require(np.all(np.abs(values['action']) <= 1+1e-6),
                'Recorded action lies outside native [-1,1]; no scaling/clipping is implicit')
        require(np.allclose(np.linalg.norm(values['qpos'][:, -4:], axis=1), 1., atol=1e-3),
                'Invalid cube quaternion; check missing virtual/external source files')
        dt = np.diff(values['time'])
        require(len(dt) > 0 and dt[0] > 0 and np.allclose(dt, dt[0], rtol=0, atol=1e-8),
                'Stored timestamps must have a positive, constant raw-control interval')
        if self.episode_column:
            require(np.all(self.file[self.episode_column][first:stop] == case['episode_id']), 'Episode boundary crossed')
        if 'step_idx' in self.file:
            require(np.array_equal(self.file['step_idx'][first:stop], np.arange(stop-first)),
                    'Replay must start at stored episode step zero')
        if 'action_valid' in self.file:
            require(np.all(self.file['action_valid'][first:stop-1]), 'Attempted to execute a terminal placeholder')
        pixels = {step: np.asarray(self.file[self.fields['pixels']][first+step]).copy() for step in render_steps}
        receipt = {'numeric_sha256': array_hash(values), 'rgb_sha256': array_hash({str(k): v for k, v in pixels.items()}),
                   'source_control_interval_seconds': float(dt[0]), 'full_source_payload_hashed': False}
        # Optional fresh-export provenance; verify selected backing files only.
        query_file = self.path.with_name('queries.json')
        if query_file.is_file():
            query = checked(query_file)
            require(Path(query['source']['path']).resolve() == self.path and sha(self.path) == query['dataset_sha256'],
                    'Fresh export manifest/container checksum differs')
            record = query['episode_files'][case['episode_position']]
            require(sha(Path(record['path'])) == record['sha256'], 'Selected fresh episode checksum differs')
            receipt['fresh_episode'] = {'path': record['path'], 'sha256': record['sha256']}
        if 'prev_qpos' in values:
            receipt['stored_prev_qpos_matches_preceding_row'] = bool(
                np.allclose(values['prev_qpos'][1:], values['qpos'][:-1], rtol=0, atol=1e-10))
        return values, pixels, receipt

    def assert_unchanged(self):
        require(all(file_stat(Path(path)) == before for path, before in self.before.items()),
                'Source/dependency changed while reading; wait for extraction to finish and use a new output')

    def description(self):
        return {'path': str(self.path), **self.before[str(self.path)], 'episodes': len(self.lengths),
                'schema': self.schema, 'image_shape': self.image_shape,
                'dependency_files_checked': len(self.before),
                'checksum_scope': 'selected numeric/RGB payloads; optional selected fresh episode receipts; not full raw source'}

    def close(self):
        self.file.close()

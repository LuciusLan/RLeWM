"""Only this module knows SWM/OGBench/MuJoCo internals; imported for user-run replay."""

from importlib import metadata
import inspect
from pathlib import Path

import numpy as np

from diagnostics.common import headless, require, sha


class CubeReplay:
    def __init__(self, steps, image_shape, camera):
        headless()
        import gymnasium as gym
        import mujoco
        import stable_worldmodel  # noqa: F401 -- registers native environments
        self.mujoco, self.camera = mujoco, camera
        height, width = image_shape[:2] if image_shape else (224, 224)
        self.settings = {'env_type': 'single', 'ob_type': 'states', 'multiview': False,
                         'height': height, 'width': width, 'mode': 'data_collection',
                         'terminate_at_goal': False, 'visualize_info': False}
        self.env = gym.make('swm/OGBCube-v0', max_episode_steps=steps+2, **self.settings)
        self.native = self.env.unwrapped

    def reset(self, values, seed):
        env = self.native
        # CubeEnv currently drops reset(seed), so seed its physical RNG directly.
        env.np_random = np.random.default_rng(seed)
        self.env.reset(seed=seed, options={'variation': []})
        model, data = env._model, env._data
        require((model.nq, model.nv, model.nu) == (21, 20, 7), 'Unexpected single-Cube model topology')
        require(abs(env._control_timestep-np.diff(values['time'])[0]) <= 1e-8,
                'Simulator control interval differs from recorded timestamps; no frameskip is inserted')
        fallback_pos, fallback_quat = data.mocap_pos.copy(), data.mocap_quat.copy()
        self.mujoco.mj_resetData(model, data)
        data.mocap_pos[:], data.mocap_quat[:] = fallback_pos, fallback_quat
        data.qpos[:], data.qvel[:], data.time = values['qpos'][0], values['qvel'][0], float(values['time'][0])
        restored = ['qpos', 'qvel', 'time']
        for key in ('control', 'mocap_pos', 'mocap_quat', 'act'):
            if key in values:
                target = data.ctrl if key == 'control' else getattr(data, key)
                require(target.shape == values[key][0].shape, f'Restore shape differs: {key}')
                target[:] = values[key][0]
                restored.append(key)
        self.apply_goal(values, 0)
        self.mujoco.mj_forward(model, data)
        self.mujoco.mj_rnePostConstraint(model, data)
        if 'qacc_warmstart' in values:
            require(data.qacc_warmstart.shape == values['qacc_warmstart'][0].shape, 'Warmstart shape differs')
            data.qacc_warmstart[:] = values['qacc_warmstart'][0]
            restored.append('qacc_warmstart')
        env._prev_qpos = values.get('prev_qpos', values['qpos'])[0].copy()
        env._prev_qvel = values.get('prev_qvel', values['qvel'])[0].copy()
        env._reset_next_step = False
        env.post_step()
        return {'restored_fields': restored,
                'goal': 'recorded target mocap/position/yaw' if any(k in values for k in ('mocap_pos', 'target_pos')) else 'reset default; target unrecorded',
                'unrecorded_state': [k for k in ('qacc_warmstart', 'act')
                                     if k not in values and (k != 'act' or model.na > 0)],
                'full_mujoco_state_restored': False,
                'note': 'Missing internal state is recomputed/reset; physical state is restored once, at episode start.'}

    def apply_goal(self, values, step):
        """Restore target labels, never robot/cube qpos during the replay."""
        env, data = self.native, self.native._data
        ids = np.asarray(env._cube_target_mocap_ids, dtype=int)
        require(len(ids) == 1, 'Only a single cube target is supported')
        if 'target_block' in values:
            require(int(np.asarray(values['target_block'][step]).item()) == 0, 'Unexpected target cube')
        env._target_block = 0
        if 'mocap_pos' in values:
            require(values['mocap_pos'][step].shape == data.mocap_pos.shape, 'Mocap position shape differs')
            data.mocap_pos[ids] = values['mocap_pos'][step][ids]
        elif 'target_pos' in values:
            data.mocap_pos[ids[0]] = values['target_pos'][step]
        if 'mocap_quat' in values:
            require(values['mocap_quat'][step].shape == data.mocap_quat.shape, 'Mocap quaternion shape differs')
            data.mocap_quat[ids] = values['mocap_quat'][step][ids]
        elif 'target_yaw' in values:
            yaw = float(np.asarray(values['target_yaw'][step]).item())
            data.mocap_quat[ids[0]] = [np.cos(yaw/2), 0., 0., np.sin(yaw/2)]

    def step(self, action):
        require(action.shape == (5,) and np.isfinite(action).all() and np.all(np.abs(action) <= 1+1e-6),
                'Invalid raw action; no normalization/clipping/repetition is applied')
        require(not self.native._reset_next_step, 'Unexpected implicit reset')
        _, _, terminated, truncated, _ = self.env.step(action.copy())
        require(not terminated and not truncated, 'Replay ended early; refusing to auto-reset')

    def snapshot(self):
        data, info = self.native._data, self.native.compute_ob_info()
        result = {'qpos': data.qpos.copy(), 'qvel': data.qvel.copy(), 'control': data.ctrl.copy(),
                  'time': np.asarray(data.time), 'observation': np.asarray(self.native.compute_observation()).copy()}
        for key, native in (('block_pos', 'privileged/block_0_pos'), ('block_quat', 'privileged/block_0_quat'),
                            ('effector_pos', 'proprio/effector_pos'), ('gripper_opening', 'proprio/gripper_opening')):
            result[key] = np.asarray(info[native]).copy()
        return result

    def render(self):
        return self.native.render(camera=self.camera).copy()

    def description(self):
        env, model = self.native, self.native._model
        files = set()
        for cls in [*type(env).__mro__[:-1], type(env._ik)]:
            try:
                path = Path(inspect.getfile(cls)).resolve()
                if path.is_file():
                    files.add(path)
            except TypeError:
                pass
        return {'gym_id': 'swm/OGBCube-v0', 'settings': self.settings, 'camera': self.camera,
                'packages': {name: metadata.version(name) for name in
                             ('mujoco', 'dm-control', 'ogbench', 'stable-worldmodel', 'gymnasium', 'numpy')},
                'mujoco_library': self.mujoco.mj_versionString(),
                'mujoco_module': self.mujoco.__file__, 'installed_sources': {str(p): sha(p) for p in sorted(files)},
                'physics_timestep': float(model.opt.timestep), 'control_timestep': float(env._control_timestep),
                'physics_substeps_per_control': int(env._n_steps),
                'native_physical_action_low': env.action_low.tolist(),
                'native_physical_action_high': env.action_high.tolist(),
                'dimensions': {'qpos': model.nq, 'qvel': model.nv, 'ctrl': model.nu,
                               'act': model.na, 'mocap': model.nmocap},
                'gravity': model.opt.gravity.tolist(), 'integrator': int(model.opt.integrator)}

    def close(self):
        renderer = getattr(self.native, '_renderer', None)
        if renderer is not None:
            renderer.close()
            self.native._renderer = None
        self.env.close()

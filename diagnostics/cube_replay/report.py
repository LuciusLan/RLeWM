"""Pure comparison metrics and compact artifacts; no simulator/model imports."""

import csv

import numpy as np

from diagnostics.common import require, sha


def aligned_qpos(reference, actual):
    """Single-Cube free-joint quaternion: q and -q denote the same pose."""
    reference = np.asarray(reference, dtype=np.float64).copy()
    if np.dot(reference[-4:], actual[-4:]) < 0:
        reference[-4:] *= -1
    return reference


def compare(values, actual, step, tolerances, source_rgb=None, replay_rgb=None):
    row = {'raw_step': step, 'source_time_seconds': float(values['time'][step])}
    for key in ('qpos', 'qvel', 'control', 'time', 'observation', 'effector_pos', 'gripper_opening'):
        if key not in values:
            continue
        reference, current = np.asarray(values[key][step]), np.asarray(actual[key])
        require(reference.shape == current.shape and np.isfinite(current).all(),
                f'Replay shape/nonfinite value at step {step}: {key}')
        if key == 'qpos':
            reference = aligned_qpos(reference, current)
        error = np.asarray(current, dtype=np.float64)-reference
        row[key+'_max_abs'] = float(np.max(np.abs(error)))
        row[key+'_rmse'] = float(np.sqrt(np.mean(error**2)))
    # Free cube joint is last; first six coordinates are the arm joints.
    row['arm_qpos_max_abs'] = float(np.max(np.abs(actual['qpos'][:6]-values['qpos'][step, :6])))
    row['cube_position_l2_metres'] = float(np.linalg.norm(actual['qpos'][-7:-4]-values['qpos'][step, -7:-4]))
    first, second = values['qpos'][step, -4:], actual['qpos'][-4:]
    require(np.linalg.norm(second) > 0, 'Invalid replay quaternion')
    dot = abs(float(np.dot(first, second)/(np.linalg.norm(first)*np.linalg.norm(second))))
    row['cube_orientation_error_radians'] = float(2*np.arccos(np.clip(dot, 0, 1)))
    row['state_passed'] = all(row[key+'_max_abs'] <= limit for key, limit in tolerances.items()
                              if key != 'rgb_rmse' and key+'_max_abs' in row)
    if source_rgb is not None:
        require(replay_rgb is not None and source_rgb.shape == replay_rgb.shape
                and replay_rgb.dtype == np.uint8, 'Replay RGB shape/dtype differs; resizing is not implicit')
        error = replay_rgb.astype(np.float64)-source_rgb
        row['rgb_rmse_255'] = float(np.sqrt(np.mean(error**2)))
        row['rgb_mae_255'] = float(np.mean(np.abs(error)))
        row['rgb_changed_pixel_fraction'] = float(np.mean(np.any(error != 0, axis=-1)))
        row['rgb_passed'] = row['rgb_rmse_255'] <= tolerances['rgb_rmse']
    return row


def summarize(rows):
    rgb = [row for row in rows if 'rgb_passed' in row]
    bad_state = [row['raw_step'] for row in rows if not row['state_passed']]
    bad_rgb = [row['raw_step'] for row in rgb if not row['rgb_passed']]
    keys = [key for key in rows[0] if key not in ('raw_step', 'source_time_seconds', 'state_passed')]
    keys += [key for key in (rgb[0] if rgb else {}) if key.startswith('rgb_') and key != 'rgb_passed']
    return {'state_passed': not bad_state, 'rgb_passed': not bad_rgb if rgb else None,
            'passed': not bad_state and not bad_rgb, 'first_state_mismatch_step': next(iter(bad_state), None),
            'first_rgb_mismatch_step': next(iter(bad_rgb), None),
            'state_frames_compared': len(rows), 'rgb_frames_compared': len(rgb),
            'initial': rows[0], 'first_transition': rows[1], 'endpoint': rows[-1],
            'maximum_errors': {key: max(row[key] for row in rows if key in row) for key in sorted(set(keys))
                               if key not in ('rgb_passed',)},
            'note': 'RGB is checked only at sampled frames. Optional observation/effector errors are descriptive; '
                    'the state decision uses qpos, qvel, time and control (when stored).'}


def save_trace(directory, values, actual, rows):
    """Store numeric arrays only. No RGB payload, object arrays, pickle or JSONL."""
    directory.mkdir()
    path = directory/'trace.npz'
    arrays = {'source_'+key: value for key, value in values.items()}
    arrays.update({'replay_'+key: np.stack([row[key] for row in actual]) for key in actual[0]})
    np.savez_compressed(path, **arrays)
    with (directory/'errors.csv').open('w', newline='') as stream:
        fields = list(dict.fromkeys(key for row in rows for key in row))
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return {'trace': str(path.name), 'trace_sha256': sha(path), 'errors': 'errors.csv'}


def save_preview(directory, source, actual, dt):
    from PIL import Image, ImageDraw
    frames, steps = [], sorted(source)
    for step in steps:
        first, second = source[step], actual[step]
        difference = np.clip(np.abs(first.astype(np.int16)-second.astype(np.int16))*4, 0, 255).astype(np.uint8)
        height, width = first.shape[:2]
        frame = Image.new('RGB', (width*3, height+28), 'white')
        draw = ImageDraw.Draw(frame)
        for index, (value, title) in enumerate(((first, 'Recorded'), (second, 'Replayed'), (difference, '|Difference| x4'))):
            frame.paste(Image.fromarray(value), (index*width, 28))
            draw.text((index*width+5, 7), f'{title} / raw t={step}', fill='black')
        frames.append(frame)
    durations = [max(20, round((b-a)*dt*1000)) for a, b in zip(steps, steps[1:])]
    durations.append(durations[-1] if durations else 100)
    frames[0].save(directory/'comparison.gif', save_all=True, append_images=frames[1:],
                   duration=durations, loop=0)
    chosen = sorted(set([0, min(1, len(frames)-1), len(frames)//2, len(frames)-1]))
    sheet = Image.new('RGB', (frames[0].width, frames[0].height*len(chosen)), 'white')
    for row, index in enumerate(chosen):
        sheet.paste(frames[index], (0, row*frames[0].height))
    sheet.save(directory/'comparison.png')
    return {'gif': 'comparison.gif', 'contact_sheet': 'comparison.png'}


def write_summary(directory, report):
    cases = report['cases']
    lines = ['# Recorded Cube replay', '',
             f"Completed: **{report['completed']}**. Cases finished: **{len(cases)}/{len(report['selection'])}**.", '',
             'The simulator receives each recorded raw action once, from the restored episode start. '
             'No expert, model, CEM or intermediate physical-state correction is used.', '',
             '| Episode | State pass | RGB pass | First state mismatch | Max cube position error (m) | Max RGB RMSE (0–255) |',
             '|---|---|---|---|---|---|']
    for case in cases:
        row = case['comparison']
        maximum = row['maximum_errors']
        rgb = maximum.get('rgb_rmse_255')
        rgb_text = f'{rgb:.6g}' if rgb is not None else 'unmeasured'
        lines.append(f"| {case['episode_position']} | {row['state_passed']} | {row['rgb_passed']} | "
                     f"{row['first_state_mismatch_step']} | {maximum['cube_position_l2_metres']:.6g} | {rgb_text} |")
    lines += ['', 'Read `report.json` for thresholds, initial/one-step/endpoint errors, input hashes, '
              'restored/missing state and installed package identities. Each case has `errors.csv` '
              'and numeric `trace.npz`; previewed cases also have `comparison.png` and `comparison.gif`.', '',
              '- A step-zero RGB gap with matching qpos/qvel points to camera/appearance/model geometry or renderer differences.',
              '- Control disagreement at step one points to a controller/IK/derived-state difference before accumulated physics drift.',
              '- Matching controls followed by state drift points toward physics/model/internal-state differences. '
              'Later control divergence can be a consequence of earlier state divergence.',
              '- Passing replay supports this state/action/render contract. It does not establish that source and fresh behavior policies match.',
              '', 'Full MuJoCo integration state was not recorded. Missing warmstart/controller/model state prevents '
              'an unconditional claim of exact reproduction. These are physics/RGB errors, not latent MSE or task success.', '']
    if 'error' in report:
        lines += [f"Runtime failure: `{report['error']['type']}: {report['error']['message']}`", '']
    (directory/'summary.md').write_text('\n'.join(lines))

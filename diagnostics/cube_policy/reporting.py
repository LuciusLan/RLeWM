"""Compact paired behavior summaries/previews. No simulator or model imports."""

import csv

import h5py
import numpy as np

from diagnostics.common import write
from .protocol import episode_paths, episode_receipt


def behavior_report(root, manifest):
    summaries, rows = {}, []
    for policy in ('markov', 'plan'):
        actions, changes, lifts = [], [], []
        for case in manifest['cases']:
            receipt = episode_receipt(root, policy, case, manifest)
            with h5py.File(episode_paths(root, policy, case)[0], 'r', swmr=True) as stream:
                action = stream['action'][:-1].astype(np.float64)
                xyz = stream['privileged_block_0_pos'][:]
                lift = float((xyz[:, 2]-xyz[0, 2]).max())
                row = {'policy': policy, 'case': case['index'], 'raw_controls': len(action),
                       'gripper_abs_mean': float(np.abs(action[:, -1]).mean()),
                       'gripper_saturation_fraction': float((np.abs(action[:, -1]) >= .99).mean()),
                       'gripper_abs_step_change_mean': float(np.abs(np.diff(action[:, -1])).mean()),
                       'peak_cube_rise_metres': lift,
                       'peak_cube_displacement_metres': float(np.linalg.norm(xyz-xyz[0], axis=1).max()),
                       'initial_collection_goal_reached': bool(stream['collection_goal_success'][0]),
                       'ever_collection_goal_reached': bool(stream['collection_goal_success'][:].any()),
                       'retargets': len(receipt['target_events'])-1}
            rows.append(row)
            actions.append(action)
            changes.append(np.abs(np.diff(action, axis=0)))
            lifts.append(lift)
        action, change = np.concatenate(actions), np.concatenate(changes)
        summaries[policy] = {'episodes': len(actions), 'valid_raw_controls': len(action),
            'gripper_abs_mean': float(np.abs(action[:, -1]).mean()),
            'gripper_saturation_fraction': float((np.abs(action[:, -1]) >= .99).mean()),
            'gripper_abs_step_change_mean': float(change[:, -1].mean()),
            'action_abs_mean_by_axis': np.abs(action).mean(0).tolist(),
            'action_saturation_fraction_by_axis': (np.abs(action) >= .99).mean(0).tolist(),
            'median_peak_cube_rise_metres': float(np.median(lifts)),
            'episodes_initially_at_collection_goal': sum(row['initial_collection_goal_reached'] for row in rows if row['policy'] == policy),
            'episodes_reaching_a_collection_goal': sum(row['ever_collection_goal_reached'] for row in rows if row['policy'] == policy)}
    with (root/'behavior-by-episode.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    report = {'manifest': manifest['fingerprint'], 'completed': True, 'summary': summaries,
              'saturation_threshold': .99, 'success_scope': 'native collection target within 4cm, not MPC/planning success'}
    write(root/'behavior-report.json', report)
    lines = ['# Cube behavior calibration', '',
        '| Policy | Episodes | Mean abs gripper | Gripper saturation | Mean abs gripper change | Median peak rise (m) |',
        '|---|---:|---:|---:|---:|---:|']
    for name, row in summaries.items():
        lines.append(f"| {name} | {row['episodes']} | {row['gripper_abs_mean']:.6f} | "
                     f"{row['gripper_saturation_fraction']:.2%} | {row['gripper_abs_step_change_mean']:.6f} | "
                     f"{row['median_peak_cube_rise_metres']:.6f} |")
    lines += ['', 'Pairs have identical initial physics and a common ordered collection-target sequence. '
              'Retarget times depend on each oracle. All cases and the fixed prediction windows are retained.', '',
              'This is a calibration dataset, not a final test. Native collection-goal contact/progress is not '
              'CEM success. Plan-oracle historical provenance remains unverified.', '',
              'Run `cube-policy evaluate` on this root for Frozen C3 H1/H5 prediction metrics. '
              'Inspect `previews/` and `behavior-by-episode.csv` without a model.', '']
    (root/'collection-summary.md').write_text('\n'.join(lines))


def previews(root, manifest, count):
    if not count:
        return
    from PIL import Image, ImageDraw
    directory = root/'previews'
    directory.mkdir(exist_ok=True)
    links = []
    times = sorted({0, manifest['config']['raw_steps'], *range(0, manifest['config']['raw_steps']+1, 5)})
    for case in manifest['cases'][:count]:
        frames = []
        with h5py.File(episode_paths(root, 'markov', case)[0], 'r', swmr=True) as left, \
             h5py.File(episode_paths(root, 'plan', case)[0], 'r', swmr=True) as right:
            for step in times:
                frame = Image.new('RGB', (448, 248), 'white')
                draw = ImageDraw.Draw(frame)
                for index, (stream, name) in enumerate(((left, 'Markov'), (right, 'Plan'))):
                    frame.paste(Image.fromarray(stream['pixels'][step]), (index*224, 24))
                    draw.text((index*224+5, 6), f'{name} / raw t={step}', fill='black')
                frames.append(frame)
        stem = f"pair-{case['index']:03d}"
        frames[0].save(directory/f'{stem}.gif', save_all=True, append_images=frames[1:], duration=250, loop=0)
        chosen = sorted(set([0, len(frames)//4, len(frames)//2, 3*len(frames)//4, len(frames)-1]))
        sheet = Image.new('RGB', (448, 248*len(chosen)), 'white')
        for row, index in enumerate(chosen):
            sheet.paste(frames[index], (0, row*248))
        sheet.save(directory/f'{stem}.png')
        links.append(f"<h2>Case {case['index']}</h2><img src='{stem}.gif'><p><a href='{stem}.png'>Contact sheet</a></p>")
    (directory/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>Cube policy pilot</title>'
        '<h1>Matched initial cases: Markov vs plan oracle</h1><p>Actual simulator observations; '
        'goals are consumed at policy-dependent times. These are collection trajectories.</p>'+''.join(links))


def evaluation_summary(directory, report):
    lines = ['# Frozen Cube behavior-pilot prediction', '',
        f"Numerical native/project parity: **{report['passed']}**.", '',
        '| Policy | Episodes | Queries | H1 endpoint MSE | H5 endpoint MSE | H1–H5 path mean |',
        '|---|---:|---:|---:|---:|---:|']
    for name, row in report['summary'].items():
        lines.append(f"| {name} | {row['episodes']} | {row['queries']} | {row['mean_mse_by_block'][0]:.6f} | "
                     f"{row['mean_mse_by_block'][-1]:.6f} | {row['path_mse']:.6f} |")
    paired = report['paired_initial_case_comparison']
    lines += ['', f"Paired-case mean H5 difference (Markov − Plan): **{paired['mean_h5_markov_minus_plan']:.6f}**; "
              f"exploratory episode-bootstrap95% interval: {paired['bootstrap95_h5_markov_minus_plan']}.", '',
              'Observed history C3; five autoregressive blocks of five ordered raw controls. '
              'H1/H5 are endpoints at +5/+25 controls, not teacher-forced prediction. '
              'One unchanged checkpoint, encoder/projector and source-fitted action statistics serve both policies.', '',
              'Queries share initial-case seeds across policies but have different generated contexts/actions/targets. '
              'The comparison measures behavior-distribution predictability, not a model improvement on identical queries. '
              'Four windows within one episode are not four independent episodes; bootstrap resamples matched episode pairs.', '',
              'Every prespecified case/window is included. There is no TTT, retrieval, CEM, learned decoder, '
              'WM-based collection filtering or final test. Lower MSE does not prove historical source-policy identity.', '']
    (directory/'summary.md').write_text('\n'.join(lines))

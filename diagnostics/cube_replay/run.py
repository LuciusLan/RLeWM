"""User-invoked replay orchestration. Isolated from P1/P2/P3 and their artifacts."""

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys

import numpy as np

from diagnostics.common import ROOT, output, require, sha, write
from .data import RecordedSource
from .report import compare, save_preview, save_trace, summarize, write_summary


def validate(args):
    require(args.steps >= 1 and args.seed >= 0 and args.render_every >= 1 and args.previews >= 0,
            'Use positive steps/render interval, nonnegative seed/preview count')
    samples = args.samples if args.samples is not None else 4
    require(samples > 0, '--samples must be positive')
    tolerances = {'qpos': args.state_atol, 'qvel': args.velocity_atol, 'control': args.control_atol,
                  'time': args.time_atol, 'rgb_rmse': args.rgb_rmse_atol}
    require(all(np.isfinite(value) and value >= 0 for value in tolerances.values()),
            'Tolerances must be finite and nonnegative')
    if args.expected_source_sha256:
        value = args.expected_source_sha256.lower()
        require(len(value) == 64 and all(c in '0123456789abcdef' for c in value), 'Expected SHA256 must be 64 hex digits')
    render_steps = [] if args.no_render else sorted({0, 1, args.steps, *range(0, args.steps+1, args.render_every)})
    return samples, tolerances, render_steps


def code_identity():
    files = sorted(Path(__file__).parent.glob('*.py'))
    files += [ROOT/'codes/diagnostics'/name for name in ('cli.py', 'common.py', 'run.sh')]
    return {str(path.relative_to(ROOT)): sha(path) for path in files}


def run(args):
    samples, tolerances, render_steps = validate(args)
    source = RecordedSource(args.source, render=not args.no_render)
    environment, progress = None, None
    directory, report = None, None
    try:
        selection = source.select(samples, args.episodes, args.steps, args.seed)
        directory = (args.output or args.diagnostic_root/f'{source.path.stem}-n{len(selection)}-t{args.steps}-s{args.seed}').expanduser().resolve()
        plan = {'source': source.description(), 'selection': selection, 'output': str(directory),
                'render_steps': render_steps, 'tolerances': tolerances,
                'source_checksum_requested': bool(args.expected_source_sha256),
                'simulator_execution': not args.dry_run, 'model_calls': False}
        if args.dry_run:
            print(json.dumps(plan, indent=2, allow_nan=False))
            return 0
        directory = output(directory)
        report = {'version': 1, 'diagnostic': 'recorded-cube-replay',
                  'started_utc': datetime.now(timezone.utc).isoformat(), 'completed': False, **plan,
                  'interpreter': sys.executable, 'command': sys.argv,
                  'seed': args.seed, 'camera': args.camera, 'code_identity': code_identity(),
                  'cases': [], 'contract': {'state_restore': 'once at episode start',
                      'action_alignment': 'action[t] takes state[t] to state[t+1]; terminal action is never read',
                      'action_transform': 'none; native normalized 5D action passed once per raw control',
                      'goal_handling': 'recorded target labels/mocap only; never robot/cube state during rollout',
                      'full_physics_state_available': False, 'has_model_or_expert_or_retrieval': False}}
        write(directory/'report.json', report)
        if args.expected_source_sha256:
            actual_sha = sha(source.path)
            require(actual_sha == args.expected_source_sha256.lower(), 'Full source CONTAINER checksum differs')
            report['verified_container_sha256'] = actual_sha
        # Simulator dependencies are imported only after dry-run and schema validation.
        from .cube import CubeReplay
        from tqdm.auto import tqdm
        environment = CubeReplay(args.steps, source.image_shape, args.camera)
        progress = tqdm(total=len(selection)*args.steps, desc='Cube recorded replay', unit='control',
                        disable=args.quiet or os.environ.get('TQDM_DISABLE', '').lower() in ('1', 'true'))
        for index, case in enumerate(selection):
            for key in ('active_case_inputs', 'active_case_restoration'):
                report.pop(key, None)
            report['active_episode_position'] = case['episode_position']
            write(directory/'report.json', report)
            values, recorded_rgb, receipt = source.load(case, render_steps)
            restoration = environment.reset(values, args.seed+case['episode_position'])
            if 'environment' not in report:
                report['environment'] = environment.description()
            report['active_case_inputs'] = receipt
            report['active_case_restoration'] = restoration
            write(directory/'report.json', report)
            actual, rows, rendered_rgb = [], [], {}
            progress.set_postfix(episode=case['episode_position'])
            for step in range(args.steps+1):
                if step:
                    # Goal changes do not alter the physical state or chosen actions.
                    environment.apply_goal(values, step-1)
                    environment.step(values['action'][step-1])
                    environment.apply_goal(values, step)
                    progress.update()
                snapshot = environment.snapshot()
                image = environment.render() if step in recorded_rgb else None
                rows.append(compare(values, snapshot, step, tolerances, recorded_rgb.get(step), image))
                actual.append(snapshot)
                if image is not None and index < args.previews:
                    rendered_rgb[step] = image
            case_dir = directory/f"episode-{case['episode_position']:05d}"
            files = save_trace(case_dir, values, actual, rows)
            if rendered_rgb:
                files.update(save_preview(case_dir, recorded_rgb, rendered_rgb, receipt['source_control_interval_seconds']))
            report['cases'].append({**case, 'inputs': receipt, 'restoration': restoration,
                                    'directory': case_dir.name, 'artifacts': files, 'comparison': summarize(rows)})
            write(directory/'report.json', report)
        source.assert_unchanged()
        for key in ('active_episode_position', 'active_case_inputs', 'active_case_restoration'):
            report.pop(key, None)
        report.update(completed=True, passed=all(row['comparison']['passed'] for row in report['cases']),
                      state_passed=all(row['comparison']['state_passed'] for row in report['cases']),
                      rgb_passed=all(row['comparison']['rgb_passed'] for row in report['cases']) if render_steps else None,
                      source_unchanged=True, finished_utc=datetime.now(timezone.utc).isoformat())
        write(directory/'report.json', report)
        write_summary(directory, report)
        rgb_status = report['rgb_passed'] if render_steps else 'unmeasured'
        print(f"Cube replay: {'PASS' if report['passed'] else 'MISMATCH'} "
              f"(state={report['state_passed']}, RGB={rgb_status}); report: {directory/'summary.md'}")
        return 0 if report['passed'] else 2
    except BaseException as error:
        if report is not None:
            report.update(completed=False, error={'type': type(error).__name__, 'message': str(error)})
            write(directory/'report.json', report)
            write_summary(directory, report)
        raise
    finally:
        if progress is not None:
            progress.close()
        try:
            if environment is not None:
                environment.close()
        finally:
            source.close()

"""Small user-run diagnostics; model/simulator work is never launched by the agent."""

import argparse
import importlib
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from diagnostics.common import ROOT, read, require, sha, write


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='command', required=True)
    f = sub.add_parser('frozen', help='Same-checkpoint native/pixel vs project/latent rollout; no CEM/retrieval')
    f.add_argument('--dataset', choices=('pusht', 'cube', 'reacher', 'tworooms'), default='pusht')
    f.add_argument('--profiles', type=Path, default=ROOT/'codes/configs/diagnostics/frozen.json')
    f.add_argument('--cohorts', nargs='+', choices=('source', 'fresh'), default=['source', 'fresh'])
    f.add_argument('--split', choices=('validation', 'test'), default='validation')
    f.add_argument('--samples', type=int, default=16, help='Windows per cohort, sampled from distinct episodes')
    f.add_argument('--seed', type=int, default=42)
    f.add_argument('--horizon', type=int, default=5)
    f.add_argument('--histories', nargs='+', type=int, choices=(1, 3), default=None)
    f.add_argument('--device', choices=('cuda', 'cpu'), default='cuda')
    f.add_argument('--cpu-threads', type=int, default=8)
    f.add_argument('--skip-cache-check', action='store_true', help='Live native/project parity only')
    f.add_argument('--verify-cache-sha256', action='store_true', help='Also hash complete large latent files')
    f.add_argument('--target-replay', type=Path, help='AdaJEPA SWM replay.npz; PushT one-frame H5 only')
    f.add_argument('--output', type=Path)
    f.add_argument('--dry-run', action='store_true', help='Show metadata plan without imports/inference/writes')
    t = sub.add_parser('targets', help='Inspect numeric pickle or generate an explicit SWM replay')
    t.add_argument('mode', choices=('inspect', 'replay'))
    t.add_argument('--file', type=Path, default=ROOT/'datasets/plan_targets.pkl')
    t.add_argument('--samples', type=int, default=4)
    t.add_argument('--seed', type=int, default=42)
    t.add_argument('--output', type=Path)
    e = sub.add_parser('environment', help='Installed metadata and optional simulator smoke subprocesses')
    e.add_argument('--output', type=Path)
    e.add_argument('--smoke', nargs='+', choices=('cube', 'reacher', 'pusht'))
    e.add_argument('--worker', choices=('cube', 'reacher', 'pusht'), help=argparse.SUPPRESS)
    from diagnostics.cube_replay import add_parser
    add_parser(sub)
    from diagnostics.cube_policy import add_parser as add_policy_parser
    add_policy_parser(sub)
    from diagnostics.retrieval_stats import add_parser as add_retrieval_parser
    add_retrieval_parser(sub)
    return p


def main():
    args = parser().parse_args()
    if args.command == 'cube-replay':
        from diagnostics.cube_replay.run import run
        raise SystemExit(run(args))
    if args.command == 'cube-policy':
        from diagnostics.cube_policy.run import run
        raise SystemExit(run(args))
    if args.command == 'retrieval-stats':
        from diagnostics.retrieval_stats.run import run
        raise SystemExit(run(args))
    if args.command == 'environment':
        from diagnostics import environment
        if args.worker:
            print('DIAGNOSTIC_JSON='+json.dumps(environment.worker(args.worker), allow_nan=False))
            return
        result = environment.snapshot()
        if args.smoke:
            result['simulator_smoke'] = environment.smoke(args.smoke)
        if args.output:
            require(not args.output.exists(), 'Choose a new metadata output file')
            write(args.output, result)
            for name, row in result['installed'].items():
                print(f"{name}: {row.get('version', 'not installed')}")
            blockers = result['problems_if_only_mujoco_is_set_to_3_6_0']
            print(f'MuJoCo-only downgrade dependency conflicts: {len(blockers)}')
            for row in blockers:
                print(f"  {row['package']} requires {row['requires']}; candidate resolves {row['resolved']}")
            for row in result.get('simulator_smoke', []):
                print(f"Smoke {row.get('task', '?')}: {'passed' if row['passed'] else 'FAILED'}")
            print(f'Environment report: {args.output.resolve()}')
        else:
            print(json.dumps(result, indent=2, allow_nan=False))
        if args.smoke and not all(r['passed'] for r in result['simulator_smoke']):
            raise SystemExit(2)
        return
    if args.command == 'targets':
        from diagnostics import targets
        if args.mode == 'inspect':
            result = targets.inspect(args.file)
            if args.output:
                require(not args.output.exists(), 'Choose a new target-inspection output file')
                write(args.output, result)
            print(json.dumps(result, indent=2, allow_nan=False))
        else:
            args.output = args.output or ROOT/'artifacts/diagnostics/adajepa-target-replay'
            targets.replay(args)
        return
    require(args.samples > 0 and args.seed >= 0 and args.cpu_threads > 0 and 1 <= args.horizon <= 10,
            'Use positive samples/threads, nonnegative seed and horizon 1–10')
    require(len(set(args.cohorts)) == len(args.cohorts), 'Duplicate cohorts')
    args.histories = args.histories or ([1] if args.target_replay else [3, 1])
    require(len(set(args.histories)) == len(args.histories), 'Duplicate histories')
    require(not args.target_replay or args.dataset == 'pusht' and args.histories == [1] and args.horizon <= 5,
            'AdaJEPA segments have only one initial observation and 25 controls; use PushT/history1/horizon<=5')
    entry = read(args.profiles)[args.dataset]
    report_path = ROOT/entry['report']
    require(sha(report_path) == entry['sha256'], 'Pinned reference report changed')
    reference = read(report_path)
    require(reference['completed'] and not reference['limited_run'] and reference['split'] == 'validation',
            'Require the completed fresh validation reference')
    if args.dry_run:
        print(json.dumps({'dataset': args.dataset, 'reference': entry, 'cohorts': args.cohorts if not args.target_replay else ['adajepa-swm-replay'],
            'samples_per_cohort': args.samples, 'histories': args.histories, 'horizon': args.horizon,
            'cache_check': not args.skip_cache_check and not args.target_replay,
            'no_retrieval_or_cem': True, 'simulator_execution': False, 'model_calls': False}, indent=2))
        return
    # Use exactly the numerical core that generated the saved P2 reference. No
    # retrieval assets/resources are loaded, and nothing in the archive is edited.
    from ablations.saved_runs import executor
    prefix = executor(reference, 'p2_followup')
    core_codes = Path(prefix[prefix.index('-u')+1]).parent
    sys.path.insert(0, str(core_codes))
    importlib.import_module('lewm_tta')
    from diagnostics.frozen import run
    run(args, reference, entry, core_codes)


if __name__ == '__main__':
    try:
        main()
    except (ValueError, OSError, KeyError) as error:
        print(f'Diagnostic failed: {error}', file=sys.stderr)
        raise SystemExit(1)

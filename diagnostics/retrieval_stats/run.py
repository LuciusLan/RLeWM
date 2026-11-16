"""Read-only reference selection, isolated workers and resumable metric receipts."""

from contextlib import contextmanager
import fcntl
import gzip
import json
import math
import os
from pathlib import Path
import subprocess
import sys

from diagnostics.common import ROOT, checked, identity, read, require, sha, write


def code():
    paths = [Path(__file__).parent/name for name in ('run.py', 'measure.py', 'metrics.py')]
    return {str(p.relative_to(ROOT)): sha(p) for p in paths}


def reference(args, dataset):
    entry = read(args.profiles)[dataset]
    path = (ROOT/entry['report']).resolve()
    require(sha(path) == entry['sha256'], f'Pinned {dataset} validation report changed: {path}')
    report = read(path)
    require(report.get('completed') and not report.get('limited_run') and report['split'] == 'validation'
            and report['stage'] == 'P2-followup-iteration', 'Require completed full fresh P2 validation')
    config = report['config']
    require(config.get('query_dataset') and config['p1']['data'].get('dataset_id', 'pusht') == dataset,
            'Reference must use the named fresh task, not a source-corpus split')
    schedule = config['schedules']['temporal-u1']
    require(schedule['placement'] == 'each-step' and schedule['updates'] == 1
            and schedule['persistence'] == 'trajectory', 'Reference temporal-u1 semantics changed')
    require('last-block' in config['scopes'], 'Reference must support the last-block scope')
    manifest = checked(Path(config['experiment_dir'])/'manifest.json')
    query = checked(Path(config['query_dataset']['manifest']))
    require(manifest['fingerprint'] == report['manifest_fingerprint']
            and manifest.get('query_source_fingerprint') == query['fingerprint']
            and query.get('external_queries') and not query['episodes']['memory'],
            'Fresh report/cohort/query identities disagree')
    if 'collection_policy' in entry:
        require(query['collection_policy']['name'] == entry['collection_policy'], 'Wrong Cube behavior collection')
    starts = manifest['splits']['validation']['raw_start_rows']
    seeds = args.seeds if args.seeds is not None else config['p1']['evaluation']['seeds']
    require(seeds and len(seeds) == len(set(seeds)) and all(type(s) is int and s >= 0 for s in seeds), 'Invalid seeds')
    request = {'version': 1, 'dataset': dataset, 'reference': {'path': str(path), 'sha256': sha(path)},
        'validation_manifest': manifest['fingerprint'], 'fresh_queries': query['fingerprint'],
        'collection': query['source']['path'], 'collection_policy': query.get('collection_policy'),
        'split': 'validation', 'queries': len(starts[:args.limit]) if args.limit else len(starts),
        'full_queries': len(starts), 'limit': args.limit, 'seeds': seeds,
        'methods': args.methods, 'learning_rates': args.learning_rates, 'ks': args.ks,
        'scope': args.scope, 'schedule': 'temporal-u1', 'horizon': max(config['horizons']),
        'device': args.device, 'cpu_threads': args.cpu_threads,
        'support_size': config['adaptation'].get('support_size', config['p1']['adaptation']['support_size']),
        'code': code()}
    return report, request


@contextmanager
def lock(directory):
    directory.mkdir(parents=True, exist_ok=True)
    with (directory/'.writer.lock').open('a') as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ValueError(f'Another writer owns {directory}') from error
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def seal(value):
    return {**value, 'fingerprint': identity(value)}


def records_per_query(request):
    h, k = request['horizon'], len(request['ks'])
    per_session = h*(6*k+4) + (h-1)*2*k
    return per_session*len(request['methods'])*len(request['seeds'])*len(request['learning_rates'])


def save_query(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name+f'.tmp.{os.getpid()}')
    with gzip.open(temporary, 'wt', encoding='utf-8', compresslevel=6) as stream:
        json.dump(seal(value), stream, separators=(',', ':'), allow_nan=False)
    temporary.replace(path)


def load_query(path, fingerprint, index):
    with gzip.open(path, 'rt', encoding='utf-8') as stream:
        value = json.load(stream)
    require(value.pop('fingerprint') == identity(value) and value['request'] == fingerprint
            and value['query_index'] == index, f'Changed query receipt: {path}')
    return value


def output_directory(args):
    suffix = '-predictor' if args.scope == 'predictor' else ''
    return (args.output or ROOT/f'artifacts/diagnostics/retrieval-stats-reverse{suffix}').expanduser().resolve()


def run(args):
    require(args.cpu_threads > 0 and args.bootstrap_samples >= 0
            and (args.limit is None or args.limit > 0), 'Invalid threads/bootstrap/limit')
    for values in (args.datasets, args.methods, args.ks, args.learning_rates):
        require(values and len(values) == len(set(values)), 'Choose nonempty, distinct option values')
    require(all(k > 0 for k in args.ks) and all(math.isfinite(lr) and lr > 0 for lr in args.learning_rates),
            'Use positive k and finite positive LRs')
    args.output = output_directory(args)
    if args.compare_to is not None:
        args.compare_to = args.compare_to.expanduser().resolve()
        require(args.scope == 'predictor' and args.compare_to != args.output,
                '--compare-to requires --scope predictor and a separate last-block root')
    if args.action == 'analyze':
        from .analysis import analyze
        if args.dry_run:
            print(json.dumps({'analyze': args.datasets, 'output': str(args.output),
                              'compare_to': str(args.compare_to) if args.compare_to else None, 'model_calls': False}))
        else:
            analyze(args)
        return 0
    plans = {dataset: reference(args, dataset) for dataset in args.datasets}
    if args.compare_to is not None:
        from .comparison import validate_pair
        for dataset, (_, request) in plans.items():
            old = checked(args.compare_to/dataset/'manifest.json')
            validate_pair(request, old)
            completed = read(args.compare_to/dataset/'measurement.json')
            require(completed.get('completed') and completed['request'] == old['fingerprint'],
                    f'Complete the last-block baseline first: {dataset}')
    if args.dry_run:
        print(json.dumps({'plans': {name: request for name, (_, request) in plans.items()},
                          'compare_to': str(args.compare_to) if args.compare_to else None,
                          'simulator': False, 'model_calls': False, 'writes': False}, indent=2))
        return 0
    if args.worker:
        require(len(args.datasets) == 1, 'A worker handles one isolated core/dataset')
        dataset = args.datasets[0]
        report, request = plans[dataset]
        from ablations.saved_runs import executor
        prefix = executor(report, 'p2_followup')
        require('lewm_tta' not in sys.modules and 'experiments' not in sys.modules,
                'WM core imported before reference selection; use a fresh worker')
        core = Path(prefix[prefix.index('-u')+1]).parent
        sys.path.insert(0, str(core))
        from .measure import measure
        with lock(args.output/dataset):
            measure(args, report, request, core)
        return 0
    for dataset in args.datasets:
        command = [sys.executable, '-X', 'faulthandler', '-u', str(ROOT/'codes/diagnostics/cli.py'),
                   'retrieval-stats', 'measure', '--worker', '--datasets', dataset,
                   '--profiles', str(args.profiles.resolve()), '--output', str(args.output),
                   '--scope', args.scope,
                   '--learning-rates', *map(str, args.learning_rates), '--ks', *map(str, args.ks),
                   '--methods', *args.methods, '--device', args.device, '--cpu-threads', str(args.cpu_threads)]
        if args.seeds is not None:
            command += ['--seeds', *map(str, args.seeds)]
        if args.limit is not None:
            command += ['--limit', str(args.limit)]
        if args.quiet:
            command += ['--quiet']
        result = subprocess.run(command, cwd=ROOT, check=False)
        if result.returncode:
            return result.returncode
    if args.action == 'all':
        from .analysis import analyze
        analyze(args)
    return 0

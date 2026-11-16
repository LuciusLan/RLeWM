"""User-run phases for frozen-memory prediction ablations."""

import argparse
import json
from pathlib import Path

from ablations.saved_runs import checked, require
from .configuration import cases, implementation, load_lock, resolve, save_plan


def main(root, config_path, validation, core_codes):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('prepare', 'check', 'validation', 'test', 'analyze'))
    parser.add_argument('--config', type=Path, default=config_path)
    parser.add_argument('--device', choices=('cuda', 'cpu'), default='cuda')
    parser.add_argument('--cpu-threads', type=int, default=8)
    parser.add_argument('--output-dir', type=Path, help='Separate immutable experiment root (also for smoke runs)')
    parser.add_argument('--run-name', help='Run subdirectory; default validation/test or validation-smoke-qN')
    parser.add_argument('--limit', type=int, help='Validation-only smoke query count; never creates a test lock')
    parser.add_argument('--split', choices=('validation', 'test'), default='validation', help='Saved split for analyze')
    parser.add_argument('--dry-run', action='store_true', help='Show metadata/plan; no model calls or file writes')
    args = parser.parse_args()
    spec, base = resolve(root, args.config.resolve(), validation, args.output_dir)
    code = implementation(root, validation)
    require(args.cpu_threads > 0, '--cpu-threads must be positive')
    require(args.limit is None or (args.limit > 0 and (args.phase == 'validation'
            or args.phase == 'analyze' and args.split == 'validation')), '--limit is for validation smoke only')
    if args.phase == 'test' or args.phase == 'analyze' and args.split == 'test':
        lock = load_lock(spec, code)  # Before reading any test outcomes or loading a model.
        chosen = lock['selected_gains']
    else:
        chosen = None
    if args.dry_run:
        q = checked(base['query_dataset']['manifest'])
        selected = cases(spec, chosen)
        split = args.phase if args.phase in ('validation', 'test') else args.split
        count = len(q['evaluation_windows'][split]['raw_start_rows'])
        if args.limit is not None:
            count = min(count, args.limit)
        print(json.dumps({'phase': args.phase, 'split': split, 'queries': count,
            'seeds': spec['seeds'], 'new_cases': selected, 'new_query_seed_case_rows': count*len(spec['seeds'])*len(selected),
            'reuse': ['frozen', *[f'ttt/{s}' for s in spec['selectors']]],
            'experiment_dir': spec['experiment_dir'], 'cache_dir': spec['cache_dir'],
            'numerical_core': str(core_codes), 'model_calls': False, 'writes': False}, indent=2))
        return
    save_plan(spec, base, code)
    from .cache import exclusive
    from .evaluation import evaluate
    with exclusive(Path(spec['experiment_dir'])/'execution.lock'):
        if args.phase in ('validation', 'test', 'analyze'):
            evaluate(root, spec, base, code, validation, args, analyze=args.phase == 'analyze')
            return
        from . import cache, references
        from .evaluation import check, resources_for
        resources, manifest = resources_for(base, spec, validation, args)
        if args.phase == 'prepare':
            cache.prepare(resources, spec)
        else:
            values, metadata = cache.load(resources, spec)
            _, rows = references.load(root, spec, 'validation', validation)
            references.verify_runtime(resources, manifest, validation, 'validation')
            check(resources, manifest, spec, values, metadata, rows)

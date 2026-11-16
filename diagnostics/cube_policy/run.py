"""One existing diagnostic CLI, with separate CPU collection and WM evaluation."""

import json

from .protocol import ensure_manifest, resolve, writer_lock


def run(args):
    config, reference, cases, root, manifest = resolve(args)
    if args.dry_run:
        print(json.dumps({'action': args.action, 'output': str(root), 'config': config, 'cases': cases,
            'protocol': manifest['protocol'], 'queries_per_policy': len(cases)*len(config['window_starts']),
            'simulator_execution': False, 'model_calls': False, 'writes': False}, indent=2, allow_nan=False))
        return 0
    with writer_lock(root):
        ensure_manifest(root, manifest, create=args.action in ('collect', 'all'))
        if args.action in ('collect', 'all'):
            from .collection import collect
            collect(root, manifest, args)
        if args.action in ('evaluate', 'all'):
            from .evaluation import evaluate
            return evaluate(root, manifest, reference, args)
    return 0

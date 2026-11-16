"""Validation-only retrieval diagnostics; no model imports during CLI discovery."""

import argparse
from pathlib import Path

from diagnostics.common import ROOT


def add_parser(subparsers):
    parser = subparsers.add_parser('retrieval-stats', help='Forward/reverse overlap and within-position reverse refresh')
    parser.add_argument('action', choices=('measure', 'analyze', 'all'), default='all', nargs='?')
    parser.add_argument('--datasets', nargs='+', choices=('pusht', 'tworooms', 'reacher', 'cube'),
                        default=['pusht', 'tworooms', 'reacher', 'cube'])
    parser.add_argument('--profiles', type=Path, default=ROOT/'codes/configs/diagnostics/retrieval-stats.json')
    parser.add_argument('--scope', choices=('last-block', 'predictor'), default='last-block',
                        help='Trainable scope; predictor includes predictor + pred_proj, with action/visual encoders frozen')
    parser.add_argument('--output', type=Path, help='Default: retrieval-stats-reverse[-predictor] under artifacts/diagnostics')
    parser.add_argument('--compare-to', type=Path,
                        help='Completed last-block statistics root; read-only paired comparison after predictor measurement')
    parser.add_argument('--learning-rates', nargs='+', type=float, default=[1e-4, 1e-5, 1e-6])
    parser.add_argument('--ks', nargs='+', type=int, default=[32], help='Diagnostic prefixes; does not change TTT support size')
    parser.add_argument('--methods', nargs='+', choices=('fr-fixed', 'forward', 'reverse', 'uniform'), default=['reverse'],
                        help='Method supplying the actual u1 update; default: reverse-only support')
    parser.add_argument('--seeds', nargs='+', type=int, help='Default: seeds in each saved validation reference')
    parser.add_argument('--limit', type=int, help='Smoke only: first N reserved validation queries')
    parser.add_argument('--device', choices=('cuda', 'cpu'), default='cuda')
    parser.add_argument('--cpu-threads', type=int, default=8)
    parser.add_argument('--bootstrap-samples', type=int, default=1000)
    parser.add_argument('--quiet', action='store_true')
    parser.add_argument('--dry-run', action='store_true', help='Metadata plan only; no model imports, cache access or writes')
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    return parser

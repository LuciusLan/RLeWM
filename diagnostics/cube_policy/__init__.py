"""Paired Cube behavior pilot; runtime modules are imported lazily."""

from pathlib import Path

from diagnostics.common import ROOT


def add_parser(subparsers):
    parser = subparsers.add_parser('cube-policy', help='Isolated paired Markov/plan-oracle calibration pilot')
    parser.add_argument('action', choices=('collect', 'evaluate', 'all'))
    parser.add_argument('--config', type=Path, default=ROOT/'codes/configs/diagnostics/cube-policy.json')
    parser.add_argument('--output', type=Path, help='Pilot root; default artifacts/diagnostics/cube-policy/paired-eN-sSEED')
    parser.add_argument('--episodes', type=int, help='Override calibration pairs; default 8')
    parser.add_argument('--seed', type=int, help='Override independent calibration seed namespace')
    parser.add_argument('--previews', type=int, default=2, help='Paired episode GIF/contact sheets; 0 to skip')
    parser.add_argument('--device', choices=('cuda', 'cpu'), default='cuda')
    parser.add_argument('--cpu-threads', type=int, default=8)
    parser.add_argument('--run-name', default='frozen-c3-h5', help='Named evaluation under the pilot root')
    parser.add_argument('--quiet', action='store_true')
    parser.add_argument('--dry-run', action='store_true', help='Inspect protocol/reference; no simulator, inference or writes')
    return parser

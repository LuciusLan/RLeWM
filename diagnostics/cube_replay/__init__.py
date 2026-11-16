"""Recorded Cube physics replay. No imports from the experiment/model packages."""

from pathlib import Path

from diagnostics.common import ROOT


def add_parser(subparsers):
    parser = subparsers.add_parser(
        'cube-replay', help='Replay recorded Cube actions and compare physical states/RGB; no WM')
    parser.add_argument('--source', type=Path,
                        default=Path('/opt/dlami/nvme/ubuntu/lewm_cube/cube_single_expert.h5'))
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument('--samples', type=int, default=None, help='Distinct episodes; default 4')
    selection.add_argument('--episodes', type=int, nargs='+', help='Explicit zero-based episode positions')
    parser.add_argument('--steps', type=int, default=35, help='Raw controls from episode start; no frameskip')
    parser.add_argument('--seed', type=int, default=42, help='Episode selection and reset scaffolding')
    parser.add_argument('--output', type=Path, help='New directory under artifacts/diagnostics by default')
    parser.add_argument('--render-every', type=int, default=5, help='Compare RGB every N controls, plus steps 0, 1 and final')
    parser.add_argument('--previews', type=int, default=2, help='Number of side-by-side GIF/contact-sheet previews')
    parser.add_argument('--no-render', action='store_true', help='Physics only; no EGL context or RGB decoding')
    parser.add_argument('--camera', choices=('front_pixels', 'side_pixels', 'front'), default='front_pixels')
    parser.add_argument('--state-atol', type=float, default=1e-5, help='Max qpos error, quaternion sign aligned')
    parser.add_argument('--velocity-atol', type=float, default=1e-4)
    parser.add_argument('--control-atol', type=float, default=1e-5, help='Stored actuator ctrl, native units')
    parser.add_argument('--time-atol', type=float, default=1e-8, help='Seconds')
    parser.add_argument('--rgb-rmse-atol', type=float, default=2., help='RGB RMSE on the 0–255 scale')
    parser.add_argument('--expected-source-sha256', help='Optional full CONTAINER-file checksum to verify')
    parser.add_argument('--quiet', action='store_true', help='Disable tqdm')
    parser.add_argument('--dry-run', action='store_true', help='Inspect schema/selection; no simulator imports or writes')
    parser.set_defaults(diagnostic_root=ROOT/'artifacts/diagnostics/cube-replay')
    return parser

"""Load the pinned numerical core plus the new, isolated prediction controls."""

import argparse
import importlib
from pathlib import Path
import sys

sys.dont_write_bytecode = True

from saved_runs import ROOT, executor, read, require, sha


def main():
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument('--config', type=Path, default=ROOT/'codes/configs/ablations/memory-prediction-pusht.json')
    known, _ = pre.parse_known_args()
    spec = read(known.config)
    entry = spec['references']['validation']
    report_path = ROOT/entry['report']
    require(sha(report_path) == entry['sha256'], 'Pinned validation reference changed')
    reference = read(report_path)
    prefix = executor(reference, 'p2_followup')
    core_codes = Path(prefix[prefix.index('-u')+1]).parent
    # Import existing numerical modules from the verified archive. Extend only
    # the package search path to expose our new subpackage, never alter it on disk.
    sys.path.insert(0, str(ROOT/'codes'))
    sys.path.insert(0, str(core_codes))
    core = importlib.import_module('lewm_tta')
    core.__path__.append(str(ROOT/'codes/lewm_tta'))
    from lewm_tta.prediction_controls.cli import main as run
    run(ROOT, known.config.resolve(), reference, core_codes)


if __name__ == '__main__':
    try:
        main()
    except (ValueError, KeyError, OSError) as error:
        print(f'Memory-prediction controls failed: {error}', file=sys.stderr)
        raise SystemExit(1)

"""Model-free paths, provenance and output helpers."""

import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4*1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def identity(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def checked(path):
    result = read(path)
    require(result.get('fingerprint') == identity({k: v for k, v in result.items() if k != 'fingerprint'}),
            f'Metadata checksum differs: {path}')
    return result


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f'{path.name}.tmp.{os.getpid()}')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    temp.replace(path)


def output(path):
    directory = Path(path).expanduser().resolve()
    require(not directory.exists(), f'Choose a new output directory; preserved existing {directory}')
    directory.mkdir(parents=True)
    return directory


def code_identity():
    return {str(p.relative_to(ROOT)): sha(p) for p in sorted(Path(__file__).parent.iterdir())
            if p.suffix in ('.py', '.sh')}


def headless():
    for name, value in {'MUJOCO_GL': 'egl', 'PYOPENGL_PLATFORM': 'egl',
                        'SDL_VIDEODRIVER': 'dummy', 'SDL_AUDIODRIVER': 'dummy',
                        'PYGAME_HIDE_SUPPORT_PROMPT': '1'}.items():
        os.environ.setdefault(name, value)

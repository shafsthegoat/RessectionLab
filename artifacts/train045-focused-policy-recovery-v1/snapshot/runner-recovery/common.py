"""Fixed runner contract; no task/model/array imports."""
import errno
import hashlib
import json
import os
from pathlib import Path
import stat
import sysconfig

HERE = Path(__file__).resolve().parent
ROOT = next(p for p in HERE.parents if (p/'src/resectionlab').is_dir())
TRAIN = ('ReMIND-002', 'ReMIND-015', 'ReMIND-018', 'ReMIND-045')
IL = 'build/post-exposure-learning-v1/IL64/attempt-01'
RL = 'build/post-exposure-learning-rl8-recovery-v1/RL8/attempt-01'
OLD_INDEX = 'artifacts/post-exposure-rl8-result-v1/audit-input-index.json'
PUBLIC = 'build/remind-partial-domain-public-preparation-v1/manifests-v2/public-index.json'
PUBLIC_SHA = 'c5138559e45fd67b44b047e8ea6d800b56b8e63ae3371a5ff5c9d83e60b25fe8'
COHORT = 'manifests/experiments/remind-component-cohort-v1.json'
COHORT_SHA = '326b4ebb4a6e439e47fb166d8fcfeec5ff65798294820d5aac0ed240b21fdd05'
HELPER = 'build/native-cut-residual-diagnostic-v1/monitor-exit-repair-v1/owned_child_exit.py'
HELPER_SHA = 'aff4b197ea033aef8b160f43eabdd5a03394ae4024b6fc2f59f5241f2fcbfe14'
SAMPLER = 'build/goal-conditioned-policy-v1/darwin_fast_sampler.py'
SAMPLER_SHA = '906a70502e4f0a48fff925cca9fc1fe45882818558cc7ac95e9034e3d6910d2c'


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def semantic(value):
    return 'sha256:'+hashlib.sha256(canonical(value)).hexdigest()


def read_bytes(path, expected=None, limit=8*1024**2):
    path = Path(path)
    fd = os.open(path, os.O_RDONLY | os.O_NONBLOCK | getattr(os, 'O_NOFOLLOW', 0))
    try:
        before = os.fstat(fd)
        if not stat.S_ISREG(before.st_mode) or not 0 < before.st_size <= limit:
            raise ValueError('Bounded regular input required: '+str(path))
        with os.fdopen(fd, 'rb', closefd=False) as stream: raw = stream.read(limit+1)
        after = os.fstat(fd)
        if ((before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns)
                != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns)
                or len(raw) != before.st_size):
            raise ValueError('Input changed during bounded read')
    finally: os.close(fd)
    if expected is not None and hashlib.sha256(raw).hexdigest() != expected:
        raise ValueError('Pinned bytes changed: '+str(path))
    return raw


def sha(path): return hashlib.sha256(read_bytes(path)).hexdigest()
def read_json(path, expected=None): return json.loads(read_bytes(path, expected))


def runtime_roots():
    """Only interpreter-selected dependency directories, never dataset/build roots."""
    roots=sorted({str(Path(sysconfig.get_paths()[key]).resolve()) for key in ('purelib','platlib')})
    if any(not Path(root).is_relative_to((ROOT/'.venv').resolve()) or Path(root).name!='site-packages' for root in roots):
        raise ValueError('Existing repository venv dependency roots required')
    return roots


def write_json(path, value, limit=2*1024**2):
    raw = canonical(value)+b'\n'
    if len(raw) > limit: raise ValueError('Declared JSON output cap')
    temporary = path.with_name(path.name+'.tmp')
    with temporary.open('xb') as stream:
        stream.write(raw); stream.flush(); os.fsync(stream.fileno())
    temporary.replace(path)


def tree_bytes(root, live=False, live_output=None):
    total = 0
    for directory, dirs, files in os.walk(root, followlinks=False):
        for name in dirs+files:
            path = Path(directory)/name
            try: value = path.lstat()
            except FileNotFoundError as error:
                if (live and error.errno == errno.ENOENT and Path(directory) == Path(live_output or root)
                        and name in files and name in {'report.json.tmp', 'progress.json.tmp'}): continue
                raise
            if stat.S_ISLNK(value.st_mode): raise ValueError('Linked stage entry refused')
            if stat.S_ISREG(value.st_mode): total += value.st_size
            elif not stat.S_ISDIR(value.st_mode): raise ValueError('Nonregular stage entry')
    return total


def guard(root, index):
    for name, digest in index['files'].items(): read_bytes(root/name, digest)


PAIR_SUBJECT="ReMIND-045"

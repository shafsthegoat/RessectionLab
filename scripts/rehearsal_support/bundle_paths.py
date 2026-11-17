"""Locations only: historical source, scene and protection identities stay fixed."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def data_root(value=None, *, must_exist=True):
    """A checkout-shaped data root, separate from the executing source checkout.

    Module defaults defer existence checks so an explicit CLI root can override
    an unavailable environment default before any historical input is read.
    """
    root = Path(value or os.environ.get('RESSECTIONLAB_REHEARSAL_BUNDLE_ROOT') or ROOT).resolve()
    if must_exist and not root.is_dir():
        raise ValueError('Rehearsal bundle root does not exist: ' + str(root))
    return root


def require_data_file(path):
    """Fail with restoration guidance for optional historical inputs."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            'Optional historical rehearsal data is missing: ' + str(path)
            + '. Restore the rehearsal archive into a new data root and select it with '
            '--bundle-root or RESSECTIONLAB_REHEARSAL_BUNDLE_ROOT. '
            'See scripts/rehearsal_support/README.md for restoration instructions.')
    return path

"""Locations only: historical source, scene and protection identities stay fixed."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def data_root(value=None):
    """A checkout-shaped data root, separate from the executing source checkout."""
    root = Path(value or os.environ.get('RESSECTIONLAB_REHEARSAL_BUNDLE_ROOT') or ROOT).resolve()
    if not root.is_dir():
        raise ValueError('Rehearsal bundle root does not exist: ' + str(root))
    return root

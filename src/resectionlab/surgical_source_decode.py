"""Exact original NIfTI → native import binding, before task admission.

The image import contract is float32 decoded voxels in the original affine.
Supplied categorical annotations preserve every integer class, without sampling,
thresholding or edited-label substitution. Model estimates remain estimates.
"""
from __future__ import annotations

import math
import nibabel as nib
import numpy as np

from .patient_anatomy import MAX_ARRAY


def _require(value, message):
    if not value: raise ValueError('TASK_ADMISSION: original/native decode binding: '+message)


def verify_original_decodes(collection, adapter, *, check):
    manifest = collection.document; series = {r['id']: r for r in manifest['series']}
    def image(original, bytes_per_cell):
        check()
        try: value = nib.load(collection.path(original['path']))
        except (OSError, nib.filebasedimages.ImageFileError) as error:
            raise ValueError('TASK_ADMISSION: original/native decode binding: source NIfTI is unavailable or invalid') from error
        _require(type(value) in (nib.Nifti1Image, nib.Nifti2Image),
                 'single-file NIfTI required; paired sidecar data needs its own source-member binding')
        _require(len(value.shape) == 3 and math.prod(value.shape)*bytes_per_cell <= MAX_ARRAY,
                 'bounded three-dimensional source NIfTI required')
        return value
    for row in manifest['series']:
        original = image(row['original'], 4)
        _require(tuple(original.shape) == tuple(row['data']['shape']), 'imaging shape differs')
        _require(original.header.get_xyzt_units()[0] == 'mm' and row.get('units_status') == 'mm', 'imaging units differ')
        _require(np.array_equal(original.affine, row['affine']), 'imaging affine differs')
        _require(np.allclose(nib.affines.voxel_sizes(original.affine), row['grid_spacing_mm'], rtol=0., atol=1e-12),
                 'imaging spacing differs')
        decoded = original.get_fdata(dtype=np.float32); stored = collection.array(row['data'])
        _require(stored.dtype == np.float32 and np.isfinite(decoded).all() and np.array_equal(decoded, stored),
                 'imaging voxels differ from the float32 source import')
        check()
    layers = {r['id']: r for r in manifest['layers']}
    for selected in adapter['sourceLayers'].values():
        if selected.get('provenance', {}).get('kind') != 'observed_source_annotation': continue
        row = layers[selected['layerId']]
        _require(not row.get('edits') and not row.get('original_data') and 'original_label_value' not in row,
                 'edited supplied annotations require a separately qualified derivation')
        for key in ('annotation_value', 'source_label_value'):
            declared = row['provenance'].get(key, row.get('label_value'))
            _require(type(declared) is int and declared == row.get('label_value'),
                     'selected source label contradicts the annotation declaration')
        original = image(row['provenance']['source'], 8)
        _require(tuple(original.shape) == tuple(row['data']['shape']), 'categorical annotation shape differs')
        _require(np.array_equal(original.affine, row['affine']), 'categorical annotation affine differs')
        units = original.header.get_xyzt_units()[0]; parent = series[row['source_series']]
        inherited = (units == 'unknown' and parent.get('units_status') == 'mm'
                     and np.array_equal(original.affine, parent['affine'])
                     and tuple(original.shape) == tuple(parent['data']['shape']))
        _require(units == 'mm' or inherited, 'categorical annotation units are not bound to the verified source grid')
        # float64 preserves noninteger scaled values that a float32 cast could
        # round into an invented categorical label. All uint16 codes are exact.
        decoded = original.get_fdata(dtype=np.float64)
        _require(np.isfinite(decoded).all() and np.all(decoded >= 0) and np.all(decoded <= 65535)
                 and np.array_equal(decoded, np.floor(decoded)), 'supplied annotation is not exact categorical data')
        stored = collection.array(row['data'])
        # Native importers have used both lossless unsigned formats even when
        # all class values fit uint8. Storage width never changes membership.
        _require(stored.dtype in (np.dtype('uint8'), np.dtype('uint16'))
                 and np.array_equal(decoded, stored), 'categorical annotation voxels/classes differ')
        _require(type(row.get('label_value')) is int and row['label_value'] > 0 and np.any(decoded == row['label_value']),
                 'selected source label is absent')
        check()

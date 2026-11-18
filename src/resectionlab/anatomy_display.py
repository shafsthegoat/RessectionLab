"""Display derivatives. These never authorize quantitative patient-frame operations."""
from copy import deepcopy
import nibabel as nib
import numpy as np
from scipy import ndimage as ndi
from .patient_anatomy import affine, file_hash, nifti_geometry, resample


def display_affine(collection, row, *, preview=False):
    sid = row.get('source_series', row['id'])
    transform = collection.document['transforms'].get(sid)
    if transform and transform['status'] in {'reference', 'reviewed', 'source-provided'}:
        try:
            return collection.patient_affine(row), False
        except ValueError:
            if not preview:
                raise
    if preview and transform and transform['status'] in {'reviewed', 'source-provided', 'unreviewed'}:
        source = collection.series(sid)
        reference = collection.series(collection.document['reference_series'])
        if (source['timepoint'] != reference['timepoint'] or
                transform.get('source_sha256') != source['original']['sha256'] or
                transform.get('reference_sha256') != reference['original']['sha256']):
            raise ValueError('Alignment preview does not match these source scans')
        if sid != row['id']:
            collection.require_native_geometry(row)
        matrix = collection.registration_matrix(sid, require_geometry=True)
        if (not np.allclose(matrix[:3, :3].T @ matrix[:3, :3], np.eye(3), atol=1e-5)
                or np.linalg.det(matrix[:3, :3]) < 0):
            raise ValueError('Alignment preview requires a rigid transform')
        return matrix @ affine(row['affine']), True
    raise ValueError(f'No usable alignment for {sid}')


def source_lattice_annotation_preview(collection, row, selected):
    """Admit an undeclared-unit annotation by index on its own acquired source."""
    provenance = row.get("provenance", {})
    original = provenance.get("source")
    if (row.get("units_status") != "unknown" or selected.get("units_status") != "mm"
            or row.get("source_series") != selected["id"]
            or provenance.get("kind") not in {"supplied_annotation", "observed_source_annotation"}
            or not isinstance(original, dict) or not isinstance(original.get("path"), str)
            or row["data"]["shape"] != selected["data"]["shape"]
            or not np.array_equal(row["affine"], selected["affine"])):
        return False
    try:
        collection.require_native_geometry(selected)
        path = collection.path(original["path"])
        label = nib.load(path)
        geometry = nifti_geometry(label)
        # With no units declared, compare every active form against the known
        # source lattice. Matching only the selected form can conceal conflict.
        return (geometry["spatial_units"] == "unknown" and geometry["frame_status"] == "coded"
                and tuple(row["data"]["shape"]) == label.shape
                and all(np.allclose(geometry[key], selected["affine"], atol=1e-5, rtol=0)
                        for key in ("qform", "sform") if geometry[key + "_code"])
                and file_hash(path) == original.get("sha256"))
    except (OSError, ValueError, nib.filebasedimages.ImageFileError):
        return False


def brain_vessel_display(collection, row):
    """Intersect a vessel estimate with a source-bound brain envelope for display.

    A filled brain mask includes internal holes. No dilation, joining, component
    deletion, or anatomy revision is performed. The full source stays available.
    """
    brains = [b for b in collection.document['layers'] if b['kind'] == 'brain']
    brains.sort(key=lambda b: b['source_series'] != row['source_series'])
    for brain in brains:
        if collection.series(brain['source_series'])['timepoint'] != collection.series(row['source_series'])['timepoint']:
            continue
        try:
            if brain['source_series'] == row['source_series']:
                collection.require_native_geometry(brain)
                collection.require_native_geometry(row)
                source, target = affine(brain['affine']), affine(row['affine'])
            else:
                source, target = collection.patient_affine(brain), collection.patient_affine(row)
        except ValueError:
            continue
        mask = collection.mask(row)
        envelope = ndi.binary_fill_holes(collection.mask(brain))
        envelope = resample(envelope.astype(np.uint8), source, mask.shape, target, order=0) > 0
        filtered = mask & envelope
        result = deepcopy(row)
        result['data'] = collection.put_array(filtered.astype(np.uint8))
        result['label_value'] = 1
        result['edits'] = [{'display_only': 'brain-envelope', 'brain': brain['data']['sha256']}]
        result.pop('probability', None)
        return result, {'scope': 'brain', 'hiddenVoxels': int(mask.sum() - filtered.sum()),
                        'brainLayer': brain['id'], 'sourcePreserved': True,
                        'reason': f'{int(mask.sum() - filtered.sum()):,} source voxels hidden by the brain-only filter.'}
    return row, {'scope': 'all', 'reason': 'No aligned brain mask; showing the full vessel estimate', 'sourcePreserved': True}

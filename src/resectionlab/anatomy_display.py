"""Display derivatives. These never authorize quantitative patient-frame operations."""
from copy import deepcopy
from itertools import product
import re
import nibabel as nib
import numpy as np
from scipy import ndimage as ndi
from .patient_anatomy import affine, file_hash, nifti_geometry, projection_or_localizer, resample, world


def display_affine(collection, row, *, preview=False):
    sid = row.get('source_series', row['id'])
    source = collection.series(sid)
    reference = collection.series(collection.document["reference_series"])
    if projection_or_localizer(source) or projection_or_localizer(reference):
        raise ValueError("Projection or localizer supports independent source review only")
    transform = collection.document['transforms'].get(sid)
    if transform and transform['status'] in {'reference', 'reviewed', 'source-provided'}:
        try:
            return collection.patient_affine(row), False
        except ValueError:
            if not preview:
                raise
    if preview and transform and transform['status'] in {'reviewed', 'source-provided', 'unreviewed'}:
        if (source['timepoint'] != reference['timepoint'] or
                transform.get('source_sha256') != source['original']['sha256'] or
                transform.get('reference_sha256') != reference['original']['sha256']):
            raise ValueError('Alignment preview does not match these source scans')
        if sid != row['id']:
            collection.require_native_geometry(row)
        # A valid child receipt cannot make an ineligible ancestor displayable.
        matrix = collection.registration_matrix(sid, require_geometry=True)
        parent = transform.get("parent_series")
        while parent:
            ancestor = collection.document["transforms"][parent]
            if ancestor["status"] not in {"reference", "reviewed", "source-provided", "unreviewed"}:
                raise ValueError(f"No usable alignment for {parent}")
            parent = ancestor.get("parent_series")
        if (not np.allclose(matrix[:3, :3].T @ matrix[:3, :3], np.eye(3), atol=1e-5)
                or np.linalg.det(matrix[:3, :3]) < 0):
            raise ValueError('Alignment preview requires a rigid transform')
        return matrix @ affine(row['affine']), True
    raise ValueError(f'No usable alignment for {sid}')


def source_display_grid(scan, displayed):
    """Choose the displayed lattice without changing the retained native source."""
    native = affine(scan["affine"])
    data = scan["data"]
    preview = next((item for item in scan.get("preview_pyramid", [])
                    if item.get("factor") == 2), None)
    if preview and np.prod(data["shape"]) > 16_000_000:
        sampled = affine(preview["affine"])
        return preview["data"], sampled, displayed @ np.linalg.solve(native, sampled), preview
    return data, native, displayed, None


def voxel_edge_corners(shape, frame):
    return world(np.asarray(frame), list(product(*[(-.5, n - .5) for n in shape])))


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


def add_cta_arterial_coverage(collection, series_id, focus_ids, wider_id):
    """Prepare an exact display union of complementary native CTA annotations."""
    from .anatomy_model_registry import source_volume

    scan = collection.series(series_id)
    if scan['modality'] != 'CTA' or not source_volume(scan):
        raise ValueError('Arterial coverage requires an acquired volumetric CTA')
    collection.require_native_geometry(scan)
    if not focus_ids or len(set([*focus_ids, wider_id])) != len(focus_ids) + 1:
        raise ValueError('Select distinct focused and wider annotation layers')
    rows = [collection.layer(layer_id) for layer_id in [*focus_ids, wider_id]]
    focused = np.zeros(scan['data']['shape'], dtype=bool)
    previous_path = None
    for row in rows:
        provenance = row.get('provenance', {})
        original = provenance.get('source')
        if (row['kind'] != 'artery' or row['source_series'] != series_id or row.get('edits')
                or provenance.get('kind') not in {'supplied_annotation', 'observed_source_annotation'}
                or row['data']['shape'] != scan['data']['shape'] or row['affine'] != scan['affine']
                or not isinstance(original, dict) or not isinstance(original.get('path'), str)):
            raise ValueError(f'{row["id"]} must be a supplied artery on the selected CTA lattice')
        if not source_lattice_annotation_preview(collection, row, scan):
            collection.require_native_geometry(row)
            if file_hash(collection.path(original['path'])) != original.get('sha256'):
                raise ValueError(f'{row["id"]} no longer matches its supplied annotation')
        mask = collection.mask(row)
        if not mask.any():
            raise ValueError(f'{row["id"]} has no supplied arterial positives')
        if original['path'] != previous_path:
            source_values = np.asanyarray(nib.load(collection.path(original['path'])).dataobj)
            previous_path = original['path']
        grouping = provenance.get('source_label_values')
        if grouping is None:
            if row.get('publisher_surface') and 'publisher_class_id' not in provenance:
                source_mask = source_values > 0
            else:
                source_mask = source_values == row['label_value']
        else:
            if (not isinstance(grouping, list) or not grouping
                    or any(type(value) is not int or value <= 0 for value in grouping)):
                raise ValueError(f'{row["id"]} has an invalid source label grouping')
            source_mask = np.isin(source_values, grouping)
        if not np.array_equal(mask, source_mask):
            raise ValueError(f'{row["id"]} differs from its supplied arterial positive labels')
        if row['id'] == wider_id:
            wider = mask
        else:
            focused |= mask
    if not np.any(focused & wider) or not np.any(wider & ~focused):
        raise ValueError('The wider annotation must overlap and extend the focused arterial field')
    display = collection.document.setdefault('display', {})
    initial = display.get('initial_layers', [])
    if not all(layer_id in initial for layer_id in focus_ids) or wider_id in initial:
        raise ValueError('The initial display must select the focus layers as its arterial view')
    inputs = [{'id': row['id'], 'data_sha256': row['data']['sha256'], 'revision': row['revision'],
               'label_value': row['label_value']} for row in rows]
    union = collection.add_layer(
        (focused | wider).astype(np.uint8), scan['affine'], layer_id=f'{series_id}-supplied-arterial-coverage',
        source_series=series_id, kind='artery', role='context', color='#aa352b', label='Combined supplied arterial coverage',
        provenance={'kind': 'derived_annotation_union', 'method': 'Boolean union on the acquired CTA grid',
                    'source_scan_sha256': scan['original']['sha256'], 'input_layers': inputs, 'evidence_use': 'context_only'},
    )
    union['coverage'] = 'Union of supplied arterial annotations; unlabeled vessels remain unassessed.'
    display['cta_arterial_coverage'] = {'series_id': series_id, 'layer_id': union['id'], 'focus_layers': list(focus_ids)}
    focus_set = set(focus_ids)
    display['initial_layers'] = [name for name in initial if name not in focus_set] + [union['id']]
    return union, {'focus_voxels': int(focused.sum()), 'wider_voxels': int(wider.sum()),
                   'focus_only_voxels': int(np.count_nonzero(focused & ~wider)),
                   'wider_only_voxels': int(np.count_nonzero(wider & ~focused))}


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


def patient_display_context(collection, *, preview, check=lambda: None):
    """Resolve shared RAS bounds and acquisition provenance without reading voxel arrays."""
    sources, corners = [], []
    for scan in collection.document["series"]:
        check()
        entry = {"seriesId": scan["id"], "modality": scan["modality"],
                 "sourceSha256": scan["original"]["sha256"], "inPatientBounds": False}
        sources.append(entry)
        try:
            frame, estimated = display_affine(collection, scan, preview=preview)
        except ValueError as exc:
            entry.update(alignment="unavailable", reason=str(exc))
            continue
        entry.update(alignment="estimated" if estimated else collection.document["transforms"][scan["id"]]["status"],
                     voxelToPatientRasMm=frame.tolist(), nativeShape=scan["data"]["shape"])
        if scan.get("synthetic") or re.search(
                r"diagnostic|segmentation only|analytic signal|generated image signal",
                scan["modality"], re.I):
            entry["reason"] = "Acquisition excluded from shared cutaway bounds"
            continue
        entry["inPatientBounds"] = True
        corners.extend(voxel_edge_corners(scan["data"]["shape"], frame))
    context = {"sources": sources}
    if not corners:
        return context
    low, high = np.min(corners, axis=0), np.max(corners, axis=0)
    # Retain the former display envelope so previously saved RAS cutaways,
    # including planes at its padded edges, stay valid on the same revision.
    spacing = max(1., float(np.max((high - low) / 192)))
    cells = np.clip(np.ceil((high - low) / spacing), 2, 192)
    context["boundsRasMm"] = [low.tolist(), (low + cells * spacing).tolist()]
    return context

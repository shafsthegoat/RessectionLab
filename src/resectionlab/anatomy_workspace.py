"""Source-bound view preferences and explicit, reversible component review."""
import hashlib
import json
import os
import tempfile
import numpy as np
from scipy import ndimage as ndi
from .patient_anatomy import affine, world


def checked_view(collection, state):
    if not isinstance(state, dict) or state.get('schema') != 'patient-anatomy-view-v1':
        raise ValueError('Unsupported patient view state')
    if state.get('patientId') != collection.document['patient_id'] or state.get('snapshotId') != collection.identity():
        raise ValueError('View state belongs to a different anatomy revision')
    if state.get('vesselScope', 'brain') not in {'brain', 'all'}:
        raise ValueError('Invalid vessel display scope')
    series = {s['id'] for s in collection.document['series']}
    layers = {s['id'] for s in collection.document['layers']}
    if state.get('seriesId') not in series or not isinstance(state.get('states'), dict) or not set(state['states']) <= series:
        raise ValueError('View references an unknown scan')
    # Bound nesting/size and reject non-finite JSON before admitting display preferences.
    encoded = json.dumps(state, allow_nan=False)
    if len(encoded) > 256 * 1024:
        raise ValueError('View state exceeds the size budget')
    for series_id, view in state['states'].items():
        if not isinstance(view, dict):
            raise ValueError('Invalid scan view')
        cursor = view.get('cursor')
        if cursor is not None and (np.asarray(cursor).shape != (3,) or not np.isfinite(cursor).all()):
            raise ValueError('Invalid physical cursor')
        visible = view.get('visible', {})
        if not isinstance(visible, dict) or not set(visible) <= layers or any(type(v) is not bool for v in visible.values()):
            raise ValueError('Invalid anatomy visibility')
        if not isinstance(view.get('settings'), dict):
            raise ValueError('Invalid display settings')
        from types import SimpleNamespace
        from .workspace_bundle import _viewer_state
        source=collection.series(series_id)
        transform=collection.document['transforms'].get(series_id)
        a=collection.patient_affine(source) if transform and transform['status'] in {'reference','reviewed','source-provided'} else affine(source['affine'])
        proxy=SimpleNamespace(compartments={name:None for name in layers|{'__fragment_review','brain','tumor','vessels','bone','scalp-envelope'}},brain_mask=None,frame='RAS+',affine=a,mri=SimpleNamespace(shape=source['data']['shape']))
        _viewer_state(proxy,view['settings'],{})
    return state


def read_view(collection):
    try:
        return checked_view(collection, json.loads((collection.root / 'workspace-view.json').read_text()))
    except (OSError, ValueError, TypeError, RecursionError):
        return None


def save_view(collection, state):
    checked_view(collection, state)
    with tempfile.NamedTemporaryFile(dir=collection.root, mode='w', delete=False) as out:
        json.dump(state, out, allow_nan=False, indent=2)
        filename = out.name
    os.replace(filename, collection.root / 'workspace-view.json')


def component_review(collection, layer_id, mask=None):
    row = collection.layer(layer_id)
    data = collection.mask(row) if mask is None else np.asarray(mask, dtype=bool)
    labels, count = ndi.label(data, structure=np.ones((3, 3, 3)))
    sizes = np.bincount(labels.ravel(), minlength=count + 1)
    a = collection.patient_affine(row)
    # Accumulate sparse foreground coordinates, avoiding three full float64 grids.
    coordinates=np.nonzero(data)
    ids=labels[coordinates]
    centers=np.zeros((count,3),dtype=float)
    for axis,values in enumerate(coordinates):
        totals=np.bincount(ids,weights=values,minlength=count+1)
        if count:centers[:,axis]=totals[1:]/sizes[1:]
    del coordinates, ids
    identity = hashlib.sha256(collection.identity().encode() + layer_id.encode() + data.tobytes()).hexdigest()
    scale = abs(np.linalg.det(affine(row['affine'])[:3, :3]))
    rows = [{'id': i, 'voxels': int(sizes[i]), 'volumeMm3': float(sizes[i] * scale), 'center': world(a, centers[i-1]).tolist()} for i in range(1, count + 1)]
    rows.sort(key=lambda r: (r['volumeMm3'], r['id']))
    return {'reviewId': identity, 'layerId': layer_id, 'components': rows}, labels

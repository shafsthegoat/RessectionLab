"""Source-bound view preferences and explicit, reversible component review."""
import hashlib
import json
import os
import tempfile
from types import SimpleNamespace
import numpy as np
from scipy import ndimage as ndi
from .patient_anatomy import affine, world
from .anatomy_planning import is_native_target
from .anatomy_display import display_affine, patient_display_grid, source_display_grid, voxel_edge_corners


def checked_view(collection, state, *, retired_clipping=None):
    if not isinstance(state, dict) or state.get('schema') != 'patient-anatomy-view-v1':
        raise ValueError('Unsupported patient view state')
    if state.get('patientId') != collection.document['patient_id'] or state.get('snapshotId') != collection.identity():
        raise ValueError('View state belongs to a different anatomy revision')
    if state.get('vesselScope', 'all') not in {'brain', 'all', 'intracranial', 'excluded'}:
        raise ValueError('Invalid vessel display scope')
    if 'vesselDisplayVersion' in state and (type(state['vesselDisplayVersion']) is not int or state['vesselDisplayVersion'] != 2):
        raise ValueError('Unsupported vessel display version')
    series = {s['id'] for s in collection.document['series']}
    layers = {s['id'] for s in collection.document['layers']} | {s['id'] for s in collection.document.get('tractography', [])}
    for key in ('selectedApproachId', 'selectedStructure'):
        if state.get(key) is not None and (not isinstance(state[key], str) or len(state[key]) > 256):
            raise ValueError('Invalid selected planning display object')
    targets = {row['id'] for row in collection.document['layers'] if is_native_target(row)}
    active_target = state.get('selectedPlanningTargetId')
    if active_target is not None and (not isinstance(active_target, str) or active_target not in targets):
        raise ValueError('Invalid selected planning target')
    selected = state.get('selectedApproachIds', {})
    # Per-target display preferences can outlive a retired evidence layer.
    if (not isinstance(selected, dict) or len(selected) > 513 or
            any(not isinstance(key, str) or len(key) > 256 or
                (value is not None and (not isinstance(value, str) or len(value) > 256))
                for key, value in selected.items())):
        raise ValueError('Invalid saved target approach selections')
    if state.get('seriesId') not in series or not isinstance(state.get('states'), dict) or not set(state['states']) <= series:
        raise ValueError('View references an unknown scan')
    # Bound nesting/size and reject non-finite JSON before admitting display preferences.
    encoded = json.dumps(state, allow_nan=False)
    if len(encoded) > 256 * 1024:
        raise ValueError('View state exceeds the size budget')
    from .workspace_bundle import ClippingBoundsError, _viewer_cutaway, _viewer_state
    scoped = 'patientCutaway' in state
    if scoped:
        cutaway = state['patientCutaway']
        if not isinstance(cutaway, dict) or set(cutaway) - {'clipping', 'clipBox', 'obliqueClip'}:
            raise ValueError('Invalid shared patient cutaway settings')
        if any(value is not None for value in cutaway.values()):
            context, _ = patient_display_grid(collection, preview=True)
            if len(series) < 2 or 'shape' not in context:
                if retired_clipping is None:
                    raise ValueError('Shared patient cutaway requires a patient display grid')
                retired_clipping.append({'seriesId': None, 'cutaway': cutaway})
                state['patientCutaway'] = {}
            else:
                corners = voxel_edge_corners(context['shape'], context['affine'])
                try:
                    _viewer_cutaway(cutaway, corners=corners, frame_name='shared patient context')
                except ClippingBoundsError:
                    if retired_clipping is None:
                        raise
                    retired_clipping.append({'seriesId': None, 'cutaway': {'clipping': cutaway.pop('clipping')}})
    for series_id, view in state['states'].items():
        if not isinstance(view, dict):
            raise ValueError('Invalid scan view')
        if 'cranialView' in view and view['cranialView'] not in ('planning', 'inspection'):
            raise ValueError('Invalid cranial view')
        cursor = view.get('cursor')
        if cursor is not None and (np.asarray(cursor).shape != (3,) or not np.isfinite(cursor).all()):
            raise ValueError('Invalid physical cursor')
        visible = view.get('visible', {})
        if not isinstance(visible, dict) or not set(visible) <= layers or any(type(v) is not bool for v in visible.values()):
            raise ValueError('Invalid anatomy visibility')
        if not isinstance(view.get('settings'), dict):
            raise ValueError('Invalid display settings')
        source = collection.series(series_id)
        try:
            a = display_affine(collection, source, preview=True)[0] if scoped else collection.patient_affine(source)
        except ValueError:
            a = affine(source['affine'])  # Independent review preferences use the original grid.
        data = source['data']
        if scoped:
            data, _, a, _ = source_display_grid(source, a)
        proxy=SimpleNamespace(compartments={name:None for name in layers|{'__fragment_review','brain','tumor','vessels','bone','scalp-envelope'}},brain_mask=None,frame='RAS+',affine=a,mri=SimpleNamespace(shape=data['shape']))
        try:
            _viewer_state(proxy,view['settings'],{})
        except ClippingBoundsError:
            if retired_clipping is None:
                raise
            retired_clipping.append({'seriesId': series_id, 'cutaway': {'clipping': view['settings'].pop('clipping')}})
            if view['settings'].get('skullMode') == 'cutaway' and not any(
                    (view['settings'].get(key) or {}).get('enabled') for key in ('clipBox', 'obliqueClip')):
                opacity = view['settings'].get('layerOpacity', {})
                transparent = any(layer['kind'] in {'bone', 'skull'} and opacity.get(layer['id'], 1) < 1
                                  for layer in collection.document['layers'])
                view['settings']['skullMode'] = 'transparent' if transparent else 'exterior'
            _viewer_state(proxy,view['settings'],{})
    return state


def read_view(collection):
    try:
        return checked_view(collection, json.loads((collection.root / 'workspace-view.json').read_text()))
    except (OSError, ValueError, TypeError, RecursionError):
        return None


def save_view(collection, state, *, after_registration=False):
    # Registration may replace the grid behind an already validated saved view.
    retired = [] if after_registration else None
    checked_view(collection, state, retired_clipping=retired)
    with tempfile.NamedTemporaryFile(dir=collection.root, mode='w', delete=False) as out:
        json.dump(state, out, allow_nan=False, indent=2)
        filename = out.name
    os.replace(filename, collection.root / 'workspace-view.json')
    return retired


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

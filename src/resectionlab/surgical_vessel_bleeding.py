"""Opt-in bleeding from the vessel that was cut, on any vessel-labelled scene.

``vesselBleedingV1`` answers "where will bleeding come from" for a scene whose
vessel regions came from patient imaging. It is a research surrogate, not
patient haemodynamics. Every number is a property parameter with a recorded
source; most are labelled assumed-research.

* Segments are the 26-connected components of each vessel region, with a
  kind (artery, vein, sinus, uncertain) from the region's declared kind or
  name, stored as run-length cell lists.
* A wound opens on a segment when one of its cells is removed, when a face
  between two of its cells is transected or cut, or (when the compliant V2
  wound model is superseded) when contact tears a face.
* Arterial flow scales from the middle cerebral artery by Murray's law,
  Q proportional to d^3, times the fraction of the lumen the wound opens, and
  pulses at the heart rate. Venous, sinus and uncertain wounds well steadily
  at the venous-to-arterial pressure ratio, changed by the hydrostatic head of
  the declared patient position.
* Newly exposed tumour faces ooze per square millimetre, decaying with time,
  until bipolar or a placed haemostatic dressing stops them.
* Blood enters the ordinary per-cell fluid inventory, so suction, irrigation,
  overflow and gravity pooling keep working. Blocked outlets hold blood in a
  per-wound reservoir until an outlet opens, so mass is conserved.
* Bipolar seals a wound only when its electrodes touch the wound and enough
  activation-seconds are delivered; vessels above a calibre limit seal only
  partly and slowly, sinuses not at all. A wet field cools the tips, raising
  the energy needed and cleaning char; dry coagulation builds char sticking and
  pulling a stuck bipolar away re-opens the wound once.

Scenes without the property never import state from this module.
"""
from __future__ import annotations

import copy
import math
import re
from collections import OrderedDict

import numpy as np
from scipy import ndimage

PROPERTY = 'vesselBleedingV1'
SCHEMA = 'resectionlab.vessel-bleeding.v1'
KINDS = ('artery', 'vein', 'sinus', 'uncertain')
VENOUS_KINDS = ('vein', 'sinus', 'uncertain')

# 1 mL = 1000 mm3 and 1 min = 60 s, so 1 mL/min = 1000/60 mm3/s.
ML_PER_MIN_TO_MM3_PER_S = 1000. / 60.

ASSUMED = 'assumed-research'
ZARRINKOOB = ('Zarrinkoob L, Ambarki K, Wahlin A, Birgander R, Eklund A, Malm J. Blood flow distribution in '
              'cerebral arteries. J Cereb Blood Flow Metab 2015;35:648-654')
PARAMETERS = {
    # Total cerebral blood flow and the MCA share of it.
    'totalCerebralBloodFlowMlPerMin': (717., ZARRINKOOB),
    'middleCerebralArteryFlowFraction': (.21, ZARRINKOOB),
    # Reference M1 lumen diameter for the Murray scaling.
    'referenceM1DiameterMm': (2.8, ASSUMED),
    # Murray CD. The physiological principle of minimum work. Proc Natl Acad Sci USA 1926;12:207-214
    # gives Q proportional to d^3; the exponent is fixed in code, not a parameter.
    'heartRatePerMin': (70., ASSUMED),
    # Gosling-King pulsatility index (Vmax - Vmin) / Vmean; the value is assumed.
    'arterialPulsatilityIndex': (.9, ASSUMED),
    'meanArterialPressureMmHg': (90., ASSUMED),
    'corticalVenousPressureMmHg': (10., ASSUMED),
    'bipolarFullSealCalibreMm': (1.5, ASSUMED),
    'sealDoseActivationSeconds': (2., ASSUMED),
    'irrigationCoolingFactor': (1.5, ASSUMED),
    'wetFluidFractionOfVoxel': (.25, ASSUMED),
    'dryStickingPerActivationSecond': (.25, ASSUMED),
    'wetCleaningPerActivationSecond': (.5, ASSUMED),
    'stuckPullOffMm': (.5, ASSUMED),
    'tumourOozeMm3PerSecondPerMm2': (.004, ASSUMED),
    'tumourOozeDecaySeconds': (1800., ASSUMED),
    'oozeSealDoseActivationSeconds': (.5, ASSUMED),
    'dressingVenousTamponadeFraction': (.7, ASSUMED),
    'outletSearchVoxels': (2, ASSUMED),
    'calibreWindowVoxels': (10, ASSUMED),
}
INTEGER_PARAMETERS = ('outletSearchVoxels', 'calibreWindowVoxels')
CALIBRE_ESTIMATOR = ('Local thickness (Hildebrand T, Ruegsegger P. A new method for the model-independent assessment '
                     'of thickness in three-dimensional images. J Microsc 1997;185:67-75): the largest inscribed '
                     'sphere of the segment containing the cell, from the Euclidean distance transform on the scene '
                     'grid. Diameter = 2R - s with s the smallest voxel spacing, floored at s. A one-voxel vessel '
                     'reports s; true calibres below the grid spacing are unknown and the floor overstates them. '
                     'Even-width vessels are under-read by up to one voxel.')
LIMITS = (
    'No ischaemic territory: cutting a vessel does not starve tissue downstream in this model.',
    'Calibres below the grid spacing are unknown; the estimate is floored at one voxel.',
    'No coagulopathy, systemic blood volume, blood pressure response or haemorrhagic shock.',
    'Arterial wound flow is the normal segment flow by Murray scaling, not a pressure-driven orifice law.',
    'No distal back-bleeding or collateral supply; one wound per local breach of a segment.',
    'The fluid inventory mixes blood and irrigation; a wet field cannot tell which is present.',
    'Bipolar sealing uses delivered activation-seconds; tissue temperature does not drive it.',
    'Tumour ooze is a per-area rate with exponential decay; brain parenchyma does not ooze.',
)
OOZE_FIELDS = ('faceBits', 'activeAreaMm2', 'outletCell', 'stoppedBy', 'doseActivationSeconds', 'lossMm3')
STOPPED = {0: None, 1: 'bipolar', 2: 'haemostatic-dressing'}
_DIRECTIONS = tuple((axis, sign) for axis in range(3) for sign in (-1, 1))
_INDEX_CACHE = OrderedDict()
_CALIBRE_CACHE = OrderedDict()


def default_parameters():
    return {name: value for name, (value, _) in PARAMETERS.items()}


def parameter_sources():
    return {name: source for name, (_, source) in PARAMETERS.items()}


def _spacing(affine):
    return np.linalg.norm(np.asarray(affine, float)[:3, :3], axis=0)


def _runs(cells):
    cells = np.asarray(cells, np.int64)
    if not len(cells):
        return []
    breaks = np.flatnonzero(np.diff(cells) != 1) + 1
    starts = np.concatenate([[0], breaks]); stops = np.concatenate([breaks, [len(cells)]])
    return [[int(cells[a]), int(b - a)] for a, b in zip(starts, stops)]


def cells_from_runs(runs):
    if not runs:
        return np.zeros(0, np.int64)
    return np.concatenate([np.arange(start, start + length, dtype=np.int64) for start, length in runs])


def region_kind(region, kinds=None):
    """Kind from an explicit declaration, else from the region name; never guessed from imaging."""
    if kinds and region.get('name') in kinds:
        kind = kinds[region['name']]
        if kind not in KINDS:
            raise ValueError('VESSEL_BLEEDING_UNKNOWN_KIND:' + str(kind))
        return kind, 'declared'
    declared = region.get('vesselKind')
    if declared is not None:
        if declared not in KINDS:
            raise ValueError('VESSEL_BLEEDING_UNKNOWN_KIND:' + str(declared))
        return declared, 'region vesselKind'
    name = str(region.get('name', '')).lower()
    if 'sinus' in name:
        return 'sinus', 'region name'
    if 'vein' in name or 'venous' in name:
        return 'vein', 'region name'
    if 'arter' in name:
        return 'artery', 'region name'
    return 'uncertain', 'no artery, vein or sinus label'


def _slug(name):
    return re.sub(r'[^a-z0-9]+', '-', str(name).lower()).strip('-') or 'vessel'


def _segments_from_mask(mask, affine, name, kind, basis, region_id):
    spacing = _spacing(affine); floor = float(spacing.min())
    components, count = ndimage.label(mask, structure=np.ones((3, 3, 3), bool))
    distance = ndimage.distance_transform_edt(np.pad(mask, 1), sampling=spacing)[1:-1, 1:-1, 1:-1]
    ridge = mask & (distance >= ndimage.maximum_filter(distance, size=3))
    diameter = np.maximum(floor, 2 * distance - floor)
    segments = []
    for identifier, window in enumerate(ndimage.find_objects(components), 1):
        if window is None:
            continue
        local = components[window] == identifier
        offset = np.array([part.start for part in window])
        ijk = np.argwhere(local) + offset
        cells = np.sort(np.ravel_multi_index(ijk.T, mask.shape))
        values = diameter[window][local & ridge[window]]
        segments.append({
            'id': f'{_slug(name)}-{len(segments) + 1}', 'regionId': region_id, 'name': name,
            'kind': kind, 'kindBasis': basis, 'runs': _runs(cells), 'cellCount': int(len(cells)),
            'calibreMm': {'maximum': float(diameter[window][local].max()),
                          'medialMedian': float(np.median(values)) if len(values) else floor,
                          'floorMm': floor,
                          'resolvedAtGrid': bool(diameter[window][local].max() >= 2 * floor - 1e-9)}})
    return segments


def _property(segments, provenance):
    return {'schema': SCHEMA, 'provenance': {'kind': ASSUMED, 'description': provenance},
            'parameters': default_parameters(), 'sources': parameter_sources(), 'segments': segments}


def declare(labels, regions, affine, *, kinds=None, include=None):
    """Build ``vesselBleedingV1`` for every vessel-material region of a labelled grid.

    ``kinds`` maps region names to a declared kind; ``include`` limits the regions.
    """
    labels = np.asarray(labels)
    vessel = [r for r in sorted(regions, key=lambda r: r['id'])
              if r['material'] == 'vessel' and (include is None or r['name'] in include)]
    segments = []
    for region in vessel:
        mask = labels == region['id']
        if not mask.any():
            continue
        kind, basis = region_kind(region, kinds)
        segments.extend(_segments_from_mask(mask, affine, region['name'], kind, basis, int(region['id'])))
    if not segments:
        raise ValueError('VESSEL_BLEEDING_REQUIRES_VESSEL_CELLS')
    seen = set()
    for segment in segments:
        while segment['id'] in seen:
            segment['id'] += '-b'
        seen.add(segment['id'])
    return _property(segments, 'Segments are connected vessel-region components of the scene labels. Flow, sealing, '
                     'ooze and sticking laws use the listed parameters and sources; most are assumed-research.')


def declare_mask(mask, affine, *, name='vessel', kind='uncertain'):
    """Declare segments from a bare vessel mask on the scene grid (e.g. a local angiography wall)."""
    if kind not in KINDS:
        raise ValueError('VESSEL_BLEEDING_UNKNOWN_KIND:' + str(kind))
    mask = np.asarray(mask, bool)
    if not mask.any():
        raise ValueError('VESSEL_BLEEDING_REQUIRES_VESSEL_CELLS')
    segments = _segments_from_mask(mask, affine, name, kind, 'declared for an unlabelled vessel mask', None)
    return _property(segments, 'Segments are connected components of a vessel mask without artery/vein labels; '
                     'they bleed with the venous law and stay labelled uncertain.')


def _number(value, name, integer=False):
    if (type(value) not in ((int,) if integer else (int, float)) or not math.isfinite(value) or value < 0):
        raise ValueError('INVALID_VESSEL_BLEEDING_PARAMETER:' + name)
    return value


def validate(scene):
    p = scene.properties.get(PROPERTY)
    if p is None:
        return None
    if 'tissuePhysicsV2' not in scene.properties:
        # The compiled resident engine does not run this model; V2 keeps the Python path.
        raise ValueError('VESSEL_BLEEDING_REQUIRES_EXPLICIT_V2')
    if not isinstance(p, dict) or set(p) != {'schema', 'provenance', 'parameters', 'sources', 'segments'} or p['schema'] != SCHEMA:
        raise ValueError('COMPLETE_EXPLICIT_VESSEL_BLEEDING_V1_REQUIRED')
    provenance = p['provenance']
    if (not isinstance(provenance, dict) or set(provenance) != {'kind', 'description'} or provenance['kind'] != ASSUMED
            or not isinstance(provenance['description'], str) or not provenance['description'].strip()):
        raise ValueError('VESSEL_BLEEDING_RESEARCH_PROVENANCE_REQUIRED')
    if not isinstance(p['parameters'], dict) or set(p['parameters']) != set(PARAMETERS):
        raise ValueError('COMPLETE_VESSEL_BLEEDING_PARAMETERS_REQUIRED')
    for name, value in p['parameters'].items():
        _number(value, name, name in INTEGER_PARAMETERS)
    q = p['parameters']
    for name in ('referenceM1DiameterMm', 'meanArterialPressureMmHg', 'corticalVenousPressureMmHg',
                 'bipolarFullSealCalibreMm', 'sealDoseActivationSeconds', 'irrigationCoolingFactor',
                 'tumourOozeDecaySeconds', 'oozeSealDoseActivationSeconds', 'calibreWindowVoxels'):
        if q[name] == 0:
            raise ValueError('INVALID_VESSEL_BLEEDING_PARAMETER:' + name)
    if (q['arterialPulsatilityIndex'] > 2 or q['middleCerebralArteryFlowFraction'] > 1
            or q['dressingVenousTamponadeFraction'] > 1 or q['irrigationCoolingFactor'] < 1
            or q['calibreWindowVoxels'] > 32 or q['outletSearchVoxels'] > 8):
        raise ValueError('VESSEL_BLEEDING_PARAMETER_OUTSIDE_DOMAIN')
    sources = p['sources']
    if (not isinstance(sources, dict) or set(sources) != set(PARAMETERS)
            or any(not isinstance(v, str) or not v.strip() for v in sources.values())):
        raise ValueError('VESSEL_BLEEDING_PARAMETER_SOURCES_REQUIRED')
    _index(scene)
    return p


def _index(scene):
    """Sorted segment cells and owners; validated once per immutable scene identity."""
    key = scene.identity
    if key in _INDEX_CACHE:
        _INDEX_CACHE.move_to_end(key)
        return _INDEX_CACHE[key]
    p = scene.properties[PROPERTY]
    segments = p['segments']
    if not isinstance(segments, list) or not 1 <= len(segments) <= 65536:
        raise ValueError('VESSEL_BLEEDING_SEGMENTS_REQUIRED')
    required = {'id', 'regionId', 'name', 'kind', 'kindBasis', 'runs', 'cellCount', 'calibreMm'}
    vessel = {r['id'] for r in scene.regions if r['material'] == 'vessel'}
    labels = scene.labels.ravel(); size = labels.size
    owners, cells, ids = [], [], set()
    for number, segment in enumerate(segments):
        if not isinstance(segment, dict) or set(segment) != required or segment['kind'] not in KINDS:
            raise ValueError('COMPLETE_VESSEL_BLEEDING_SEGMENT_REQUIRED')
        if not isinstance(segment['id'], str) or not segment['id'] or segment['id'] in ids:
            raise ValueError('VESSEL_BLEEDING_DISTINCT_SEGMENT_ID_REQUIRED')
        ids.add(segment['id'])
        runs = segment['runs']
        if (not isinstance(runs, list) or not runs or any(
                not isinstance(r, list) or len(r) != 2 or any(type(v) is not int for v in r)
                or r[0] < 0 or r[1] < 1 or r[0] + r[1] > size for r in runs)):
            raise ValueError('VESSEL_BLEEDING_SEGMENT_RUNS_INVALID')
        members = cells_from_runs(runs)
        if len(members) != segment['cellCount'] or np.any(np.diff(members) <= 0):
            raise ValueError('VESSEL_BLEEDING_SEGMENT_RUNS_INVALID')
        member_labels = labels[members]
        if not np.isin(member_labels, list(vessel)).all():
            raise ValueError('VESSEL_BLEEDING_SEGMENT_CELL_NOT_VESSEL')
        if segment['regionId'] is not None and not np.all(member_labels == segment['regionId']):
            raise ValueError('VESSEL_BLEEDING_SEGMENT_REGION_MISMATCH')
        cells.append(members); owners.append(np.full(len(members), number, np.int64))
    cells = np.concatenate(cells); owners = np.concatenate(owners)
    order = np.argsort(cells, kind='stable'); cells, owners = cells[order], owners[order]
    if np.any(np.diff(cells) == 0):
        raise ValueError('VESSEL_BLEEDING_SEGMENTS_OVERLAP')
    tumour = np.isin(labels, [r['id'] for r in scene.regions if r['material'] == 'tumor'])
    result = (cells, owners, tumour)
    _INDEX_CACHE[key] = result
    while len(_INDEX_CACHE) > 8:
        _INDEX_CACHE.popitem(last=False)
    return result


def segment_of(scene, cells):
    """Segment ordinal for each cell, -1 where the cell belongs to no segment."""
    index, owners, _ = _index(scene)
    cells = np.asarray(cells, np.int64).reshape(-1)
    if not len(index):
        return np.full(len(cells), -1, np.int64)
    position = np.clip(np.searchsorted(index, cells), 0, len(index) - 1)
    return np.where(index[position] == cells, owners[position], -1)


def initialize(engine):
    if PROPERTY not in engine.scene.properties:
        return None
    if PROPERTY in engine.state:
        return engine.state[PROPERTY]
    validate(engine.scene)
    engine.state[PROPERTY] = {
        'schema': SCHEMA, 'elapsedSeconds': 0., 'scanned': {'removed': 0, 'oozeRemoved': 0, 'transectedEdges': 0, 'cutEdges': 0, 'tornFaces': 0},
        'wounds': {}, 'nextWound': 0, 'segmentLossMm3': {}, 'lossByKindMm3': {kind: 0. for kind in KINDS},
        'tumourOozeMm3': 0., 'bloodLossMm3': 0., 'depositedMm3': 0., 'reservoirMm3': {},
        'oozeCells': {}, 'bipolar': {}, 'venousHeadFactor': 1., 'rateMm3PerSecond': 0.}
    return engine.state[PROPERTY]


def _parameters(engine):
    return engine.scene.properties[PROPERTY]['parameters']


def _ijk(engine, cells):
    return np.column_stack(np.unravel_index(np.asarray(cells, np.int64), engine.scene.shape))


def _face_area(engine, a, b):
    """Physical area of the face shared by two 6-neighbour cells."""
    axis = int(np.argmax(np.abs(_ijk(engine, [a])[0] - _ijk(engine, [b])[0])))
    basis = engine.scene.affine[:3, :3]; other = [i for i in range(3) if i != axis]
    return float(np.linalg.norm(np.cross(basis[:, other[0]], basis[:, other[1]])))


def _axis_areas(engine):
    basis = engine.scene.affine[:3, :3]
    return [float(np.linalg.norm(np.cross(basis[:, (a + 1) % 3], basis[:, (a + 2) % 3]))) for a in range(3)]


def local_calibre(engine, cell, segment):
    """Local-thickness diameter (mm) at ``cell`` within its own segment; see CALIBRE_ESTIMATOR."""
    key = (engine.scene.identity, int(cell), int(segment))
    if key not in _CALIBRE_CACHE:
        _CALIBRE_CACHE[key] = _local_calibre(engine, cell, segment)
        while len(_CALIBRE_CACHE) > 4096:
            _CALIBRE_CACHE.popitem(last=False)
    return _CALIBRE_CACHE[key]


def _local_calibre(engine, cell, segment):
    scene = engine.scene; shape = np.array(scene.shape)
    spacing = np.asarray(scene.spacing, float); floor = float(spacing.min())
    half = int(_parameters(engine)['calibreWindowVoxels'])
    centre = _ijk(engine, [cell])[0]
    lo = np.maximum(centre - half, 0); hi = np.minimum(centre + half + 1, shape)
    grid = np.indices(hi - lo).reshape(3, -1).T + lo
    window = segment_of(scene, np.ravel_multi_index(grid.T, scene.shape)) == segment
    mask = window.reshape(hi - lo)
    if mask.all():
        return float(2 * half * floor), 'window-filled'
    pad = [(int(lo[a] == 0), int(hi[a] == shape[a])) for a in range(3)]
    padded = np.pad(mask, pad)
    distance = ndimage.distance_transform_edt(padded, sampling=spacing)
    distance = distance[tuple(slice(b, b + n) for (b, _), n in zip(pad, mask.shape))].reshape(-1)
    offsets = (grid - centre) @ scene.affine[:3, :3].T
    reach = np.linalg.norm(offsets, axis=1)
    inscribed = window & (reach < distance - 1e-9)
    radius = float(distance[inscribed].max()) if inscribed.any() else float(distance[np.flatnonzero(
        (grid == centre).all(axis=1))[0]])
    diameter = 2 * radius - floor
    if diameter <= floor + 1e-9:
        return floor, 'grid-floor'
    return float(diameter), 'local-thickness'


def mean_flow_mm3_per_second(parameters, kind, calibre_mm, opening, venous_head=1.):
    """Unsealed interval-mean wound flow in mm3/s before pulsation, sealing and tamponade."""
    q = parameters
    # Reference MCA flow: mL/min -> mm3/s by x1000/60.
    reference = q['totalCerebralBloodFlowMlPerMin'] * q['middleCerebralArteryFlowFraction'] * ML_PER_MIN_TO_MM3_PER_S
    # Murray (1926): segment flow proportional to diameter cubed.
    full = reference * (calibre_mm / q['referenceM1DiameterMm']) ** 3 * opening
    if kind == 'artery':
        return full
    # Low-pressure welling: same lumen, venous driving pressure under a linear resistance.
    return full * q['corticalVenousPressureMmHg'] / q['meanArterialPressureMmHg'] * venous_head


def pulse_integral(parameters, t0, t1):
    """Integral of 1 + (PI/2) sin(2 pi f t) over [t0, t1]; mean 1, peak-to-trough PI."""
    omega = 2 * math.pi * parameters['heartRatePerMin'] / 60.
    amplitude = parameters['arterialPulsatilityIndex'] / 2
    if omega == 0:
        return t1 - t0
    return (t1 - t0) + amplitude * (math.cos(omega * t0) - math.cos(omega * t1)) / omega


def _venous_head(engine):
    """Site venous pressure / supine reference: (P_cv + p_site - p_heart) / P_cv, floored at 0."""
    if 'patientPosition' not in engine.state:
        return 1.
    from .surgical_position import PA_PER_MMHG, describe_position
    hydrostatic = describe_position(engine.state['patientPosition'])['hydrostaticSiteMinusHeartPa']
    reference = _parameters(engine)['corticalVenousPressureMmHg'] * PA_PER_MMHG
    return max(0., (reference + hydrostatic) / reference)


def _wound_contact_cells(engine, wound):
    """Retained cells an electrode can touch: the wound cells, or the segment rim of removed ones."""
    removed = set(engine.state.get('removed', []))
    cells = {c for c in wound['cells'] if c not in removed}
    gone = [c for c in wound['cells'] if c in removed]
    if gone:
        shape = np.array(engine.scene.shape)
        for ijk in _ijk(engine, gone):
            for axis, sign in _DIRECTIONS:
                n = ijk.copy(); n[axis] += sign
                if np.all(n >= 0) and np.all(n < shape):
                    cell = int(np.ravel_multi_index(n, engine.scene.shape))
                    if cell not in removed and segment_of(engine.scene, [cell])[0] == wound['segmentIndex']:
                        cells.add(cell)
    return cells


def _update_wound_geometry(engine, wound):
    q = _parameters(engine)
    calibres = [local_calibre(engine, c, wound['segmentIndex']) for c in wound['cells'][:8]]
    wound['calibreMm'], wound['calibreBasis'] = max(calibres)
    lumen = math.pi * wound['calibreMm'] ** 2 / 4
    wound['openingFraction'] = min(1., wound['breachAreaMm2'] / lumen)
    full = q['bipolarFullSealCalibreMm']
    wound['bipolarSealCap'] = 0. if wound['kind'] == 'sinus' else (
        1. if wound['calibreMm'] <= full else (full / wound['calibreMm']) ** 2)


def _open_sites(engine, s, sites):
    """Join each new breach to an open wound on the same segment within the search reach, else open one."""
    segments = engine.scene.properties[PROPERTY]['segments']
    reach = int(_parameters(engine)['outletSearchVoxels'])
    for cells, faces, mechanism, area in sorted(sites, key=lambda site: (min(site[0]), site[2])):
        owner = int(segment_of(engine.scene, [min(cells)])[0])
        segment = segments[owner]
        target = None
        for wound in (s['wounds'][k] for k in sorted(s['wounds'], key=lambda k: int(k.split(':')[1]))):
            if wound['segmentIndex'] != owner or wound['sealFraction'] >= 1:
                continue
            gap = np.abs(_ijk(engine, cells)[:, None, :] - _ijk(engine, wound['cells'])[None, :, :]).max(axis=2).min()
            if gap <= reach:
                target = wound
                break
        if target is None:
            identifier = 'vessel-bleeding:' + str(s['nextWound']); s['nextWound'] += 1
            target = s['wounds'][identifier] = {
                'id': identifier, 'segment': segment['id'], 'segmentIndex': owner, 'kind': segment['kind'],
                'kindBasis': segment['kindBasis'], 'mechanisms': [], 'cells': [], 'faces': [],
                'breachAreaMm2': 0., 'openedAtSeconds': s['elapsedSeconds'], 'sealFraction': 0.,
                'coagulationDoseActivationSeconds': 0., 'charReopened': False, 'tamponadeFraction': 0.,
                'rateMm3PerSecond': 0., 'lossMm3': 0.}
        target['cells'] = sorted(set(target['cells']) | set(cells))
        target['faces'] = sorted({tuple(f) for f in target['faces']} | {tuple(f) for f in faces})
        target['faces'] = [list(f) for f in target['faces']]
        target['mechanisms'] = sorted(set(target['mechanisms']) | {mechanism})
        target['breachAreaMm2'] += area
        _update_wound_geometry(engine, target)


def _scan_vessel_injury(engine, s):
    scanned = s['scanned']; sites = []
    removed = engine.state.get('removed', [])
    if len(removed) != scanned['removed']:
        scanned['removed'] = len(removed)
        known_cells = {c for w in s['wounds'].values() for c in w['cells']}
        cells = np.asarray(removed, np.int64)
        owners = segment_of(engine.scene, cells) if len(cells) else np.zeros(0, np.int64)
        area = min(_axis_areas(engine))
        for cell in cells[owners >= 0].tolist():
            if cell not in known_cells:
                sites.append(([int(cell)], [], 'removal', area))
    torn = []
    vessels = engine.state.get('vesselsV2')
    if superseded(engine) and vessels:
        torn = [list(map(int, key.split(':'))) for key, face in vessels['faces'].items() if face['damage'] >= 1.]
    known_faces = None
    for field, mechanism, edges in (('transectedEdges', 'transection', engine.state.get('transectedEdges', [])),
                                    ('cutEdges', 'cut', engine.state.get('cutEdges', [])),
                                    ('tornFaces', 'contact-tear', torn)):
        if len(edges) == scanned[field]:
            continue
        scanned[field] = len(edges)
        if not edges:
            continue
        if known_faces is None:
            known_faces = {tuple(f) for w in s['wounds'].values() for f in w['faces']}
        pairs = np.asarray(edges, np.int64).reshape(-1, 2)
        owners = segment_of(engine.scene, pairs.ravel()).reshape(-1, 2)
        for (a, b), (oa, ob) in zip(pairs.tolist(), owners.tolist()):
            face = tuple(sorted((a, b)))
            if oa < 0 or ob < 0 or face in known_faces:
                continue
            known_faces.add(face)
            sites.append(([face[0], face[1]], [list(face)], mechanism, _face_area(engine, a, b)))
    if sites:
        _open_sites(engine, s, sites)


def _scan_ooze(engine, s):
    """Record tumour faces exposed by removal; drop entries whose own cell was removed."""
    _, _, tumour = _index(engine.scene)
    ooze = s['oozeCells']
    removed = np.asarray(engine.state.get('removed', []), np.int64)
    if not len(removed) or not tumour.any():
        return
    removed_mask = np.zeros(engine.scene.labels.size, bool); removed_mask[removed] = True
    for key in [k for k in ooze if removed_mask[int(k)]]:
        del ooze[key]
    shape = np.array(engine.scene.shape); ijk = _ijk(engine, removed)
    areas = _axis_areas(engine)
    exposed, bits, outlets = [], [], []
    for number, (axis, sign) in enumerate(_DIRECTIONS):
        neighbour = ijk.copy(); neighbour[:, axis] += sign
        valid = np.all((neighbour >= 0) & (neighbour < shape), axis=1)
        cells = np.ravel_multi_index(neighbour[valid].T, engine.scene.shape)
        keep = tumour[cells] & ~removed_mask[cells]
        # The tumour cell's exposed face points back toward the removed cell.
        exposed.append(cells[keep]); outlets.append(removed[valid][keep])
        bits.append(np.full(int(keep.sum()), 1 << (number ^ 1), np.int64))
    exposed = np.concatenate(exposed); bits = np.concatenate(bits); outlets = np.concatenate(outlets)
    if not len(exposed):
        return
    order = np.lexsort((outlets, exposed)); exposed, bits, outlets = exposed[order], bits[order], outlets[order]
    starts = np.flatnonzero(np.r_[True, exposed[1:] != exposed[:-1]])
    masks = np.bitwise_or.reduceat(bits, starts)
    for cell, mask, outlet in zip(exposed[starts].tolist(), masks.tolist(), outlets[starts].tolist()):
        key = str(cell); entry = ooze.get(key)
        new = mask if entry is None else mask & ~entry[0]
        if not new:
            continue
        area = sum(areas[number // 2] for number in range(6) if new >> number & 1)
        if entry is None:
            ooze[key] = [int(mask), area, int(outlet), 0, 0., 0.]
        else:
            entry[0] |= new
            if not entry[3]:
                entry[1] += area


def _outlet(engine, occupancy, cells):
    """Nearest free cell: a wound cell, its face neighbours, then a bounded Chebyshev search."""
    from .surgical_effects import _free_outlet
    for cell in sorted(cells):
        if not occupancy[cell]:
            return cell
    for cell in sorted(cells):
        found = _free_outlet(engine, cell)
        if found is not None:
            return found
    reach = int(_parameters(engine)['outletSearchVoxels']); shape = np.array(engine.scene.shape)
    best = None
    for ijk in _ijk(engine, sorted(cells)):
        lo = np.maximum(ijk - reach, 0); hi = np.minimum(ijk + reach + 1, shape)
        grid = np.indices(hi - lo).reshape(3, -1).T + lo
        ids = np.ravel_multi_index(grid.T, engine.scene.shape)
        free = occupancy[ids] == 0
        if not free.any():
            continue
        distance = np.linalg.norm((grid[free] - ijk) @ engine.scene.affine[:3, :3].T, axis=1)
        for d, cell in zip(distance.tolist(), ids[free].tolist()):
            if best is None or (d, cell) < best:
                best = (d, cell)
    return None if best is None else int(best[1])


def _deposit(engine, s, key, cells, amount, occupancy, flow):
    """Put blood in the per-cell inventory, or hold it at the wound until an outlet opens."""
    held = s['reservoirMm3'].pop(key, 0.)
    total = amount + held
    if total <= 0:
        return
    destination = _outlet(engine, occupancy, cells)
    if destination is None:
        s['reservoirMm3'][key] = total
        return
    k = str(destination)
    engine.state['fluid'][k] = engine.state['fluid'].get(k, 0.) + total
    engine.state['fluidAddedMm3'] += total
    s['depositedMm3'] += total
    if flow is not None:
        flow.record('deposit', 'fluid', 'wound', None, destination, total)


def _dressing_cover(engine):
    """Centres and radii of placed, released haemostatic dressings."""
    objects = engine.state.get('operativeExposureV1', {}).get('objects', [])
    result = []
    for obj in objects:
        if obj.get('kind') != 'hemostat' or obj.get('heldBy') is not None:
            continue
        if obj.get('status') not in ('placed', 'intended-retained'):
            continue
        centre = np.asarray(obj['positionMm'], float)
        result.append((centre, float(obj['radiusMm']) + .5 * float(np.linalg.norm(engine.scene.spacing)),
                       obj.get('sourceCell')))
    return result


def _covered(engine, cover, cells):
    if not cover or not cells:
        return np.zeros(len(cells), bool)
    cells = list(cells); points = engine.scene.points(cells)
    hit = np.zeros(len(cells), bool)
    for centre, radius, source in cover:
        hit |= np.linalg.norm(points - centre, axis=1) <= radius
        if source is not None:
            hit |= np.asarray(cells) == source
    return hit


def _release_stuck(engine, s):
    q = _parameters(engine)
    for key in sorted(s['bipolar']):
        row = s['bipolar'][key]
        if not row['stuck']:
            continue
        slot = engine.state.get('slots', {}).get(key)
        pulled = slot is None or slot['instrument'] != 'bipolar_forceps' or not slot.get('inserted')
        if not pulled:
            pulled = float(np.linalg.norm(np.subtract(slot['position'], row['stuckAtMm']))) > q['stuckPullOffMm']
        if not pulled and hasattr(engine, 'parameters'):
            pulled = engine.parameters(slot)['jawOpeningMm'] > .5
        if not pulled:
            continue
        reopened = []
        for identifier in row['stuckWounds']:
            wound = s['wounds'].get(identifier)
            if wound is not None and not wound['charReopened'] and wound['sealFraction'] > 0:
                wound.update(sealFraction=0., coagulationDoseActivationSeconds=0., charReopened=True,
                             reopenedAtSeconds=s['elapsedSeconds'])
                reopened.append(identifier)
        row.update(stuck=False, charSticking=0., stuckAtMm=None, stuckWounds=[],
                   pullOffs=row['pullOffs'] + 1, lastReopenedWounds=reopened)


def tick(engine, dt):
    """Open wounds from this substep's injuries and emit their blood for ``dt`` seconds.

    Called from surgical_effects.tick_fields before overflow and gravity relaxation.
    """
    if PROPERTY not in engine.scene.properties or not dt > 0:
        return
    s = initialize(engine); q = _parameters(engine)
    _scan_vessel_injury(engine, s)
    if len(engine.state.get('removed', [])) != s['scanned']['oozeRemoved']:
        s['scanned']['oozeRemoved'] = len(engine.state.get('removed', []))
        _scan_ooze(engine, s)
    _release_stuck(engine, s)
    t0 = s['elapsedSeconds']; t1 = t0 + dt
    s['venousHeadFactor'] = head = _venous_head(engine)
    cover = _dressing_cover(engine)
    occupancy = engine.occupancy()
    flow = getattr(engine, '_flow_display_trace', None)
    if flow is not None and not flow.active:
        flow = None
    rate = 0.
    for identifier in sorted(s['wounds'], key=lambda k: int(k.split(':')[1])):
        wound = s['wounds'][identifier]
        contact = sorted(_wound_contact_cells(engine, wound)) or wound['cells']
        wound['tamponadeFraction'] = (q['dressingVenousTamponadeFraction'] if wound['kind'] in VENOUS_KINDS
                                      and _covered(engine, cover, contact).any() else 0.)
        mean = mean_flow_mm3_per_second(q, wound['kind'], wound['calibreMm'], wound['openingFraction'], head)
        factor = (1 - wound['sealFraction']) * (1 - wound['tamponadeFraction'])
        duration = pulse_integral(q, t0, t1) if wound['kind'] == 'artery' else dt
        amount = max(0., mean * factor * duration)
        wound['rateMm3PerSecond'] = amount / dt
        wound['lossMm3'] += amount; rate += amount / dt
        s['segmentLossMm3'][wound['segment']] = s['segmentLossMm3'].get(wound['segment'], 0.) + amount
        s['lossByKindMm3'][wound['kind']] += amount
        s['bloodLossMm3'] += amount
        _deposit(engine, s, identifier, wound['cells'], amount, occupancy, flow)
    ooze = s['oozeCells']
    if ooze:
        tau = q['tumourOozeDecaySeconds']; decay = math.exp(-dt / tau)
        keys = sorted(ooze, key=int)
        covered = _covered(engine, cover, [int(k) for k in keys]) if cover else None
        per_outlet = {}
        for number, key in enumerate(keys):
            entry = ooze[key]
            if covered is not None and covered[number] and not entry[3]:
                entry[1] = 0.; entry[3] = 2
            if entry[3] or entry[1] <= 0:
                continue
            # Exact integral of q0 A exp(-(t - t0)/tau) over the interval.
            amount = q['tumourOozeMm3PerSecondPerMm2'] * entry[1] * tau * (1 - decay)
            entry[1] *= decay; entry[5] += amount
            per_outlet.setdefault((entry[2], int(key)), 0.)
            per_outlet[(entry[2], int(key))] += amount
            s['tumourOozeMm3'] += amount; s['bloodLossMm3'] += amount; rate += amount / dt
        for (outlet, cell), amount in sorted(per_outlet.items()):
            if not occupancy[outlet]:
                k = str(outlet)
                engine.state['fluid'][k] = engine.state['fluid'].get(k, 0.) + amount
                engine.state['fluidAddedMm3'] += amount; s['depositedMm3'] += amount
                if flow is not None:
                    flow.record('deposit', 'fluid', 'wound', None, outlet, amount)
            else:
                _deposit(engine, s, 'ooze:' + str(cell), [cell], amount, occupancy, flow)
    s['rateMm3PerSecond'] = rate
    s['elapsedSeconds'] = t1


def _wet(engine, position):
    """Free fluid within one voxel of the tip, from the mixed blood and irrigation inventory."""
    q = _parameters(engine)
    if not hasattr(engine, 'cells'):
        return False
    reach = float(np.max(engine.scene.spacing))
    fluid = engine.state['fluid']
    present = sum(fluid.get(str(int(c)), 0.) for c in engine.cells(position, position, reach))
    return present >= q['wetFluidFractionOfVoxel'] * engine.scene.voxel_mm3


def coagulate(engine, key, cells, position, power, dt):
    """Bipolar sealing of contacted wounds and ooze, with irrigation cooling and char sticking."""
    if PROPERTY not in engine.scene.properties:
        return []
    s = initialize(engine); q = _parameters(engine)
    dose = float(power) * float(dt)
    if dose <= 0 or not cells:
        return []
    wet = _wet(engine, np.asarray(position, float))
    effective = dose / q['irrigationCoolingFactor'] if wet else dose
    contacts = set(map(int, cells))
    events = []; sealing = []
    for identifier in sorted(s['wounds'], key=lambda k: int(k.split(':')[1])):
        wound = s['wounds'][identifier]
        if not contacts.intersection(_wound_contact_cells(engine, wound)):
            continue
        if wound['kind'] == 'sinus':
            events.append({'kind': 'bipolar_ineffective', 'woundId': identifier, 'vesselKind': 'sinus',
                           'reason': 'sinus wall cannot be sealed by bipolar in this model'})
            continue
        wound['coagulationDoseActivationSeconds'] += effective
        full = q['bipolarFullSealCalibreMm']
        required = q['sealDoseActivationSeconds'] * max(1., (wound['calibreMm'] / full) ** 2)
        wound['sealFraction'] = max(wound['sealFraction'], min(wound['bipolarSealCap'],
                                    wound['coagulationDoseActivationSeconds'] / required))
        sealing.append(identifier)
        events.append({'kind': 'vessel_coagulation', 'woundId': identifier, 'vesselKind': wound['kind'],
                       'calibreMm': wound['calibreMm'], 'sealFraction': wound['sealFraction'],
                       'sealCap': wound['bipolarSealCap'], 'wet': wet})
    near = set(contacts); shape = np.array(engine.scene.shape)
    for ijk in _ijk(engine, sorted(contacts)):
        for axis, sign in _DIRECTIONS:
            n = ijk.copy(); n[axis] += sign
            if np.all(n >= 0) and np.all(n < shape):
                near.add(int(np.ravel_multi_index(n, engine.scene.shape)))
    stopped = 0
    for cell in sorted(near):
        entry = s['oozeCells'].get(str(cell))
        if entry is None or entry[3]:
            continue
        entry[4] += effective
        if entry[4] >= q['oozeSealDoseActivationSeconds']:
            entry[1] = 0.; entry[3] = 1; stopped += 1
    row = s['bipolar'].setdefault(key, {'charSticking': 0., 'stuck': False, 'stuckAtMm': None, 'stuckWounds': [],
                                        'pullOffs': 0, 'lastReopenedWounds': []})
    if wet:
        row['charSticking'] = max(0., row['charSticking'] - dose * q['wetCleaningPerActivationSecond'])
    else:
        row['charSticking'] += dose * q['dryStickingPerActivationSecond']
    if row['charSticking'] >= 1.:
        if not row['stuck']:
            row.update(stuck=True, stuckAtMm=np.asarray(position, float).tolist())
        row['stuckWounds'] = sorted(set(row['stuckWounds']) | set(sealing))
    events.append({'kind': 'bipolar_field', 'wet': wet, 'charSticking': row['charSticking'], 'stuck': row['stuck'],
                   'oozeCellsStopped': stopped})
    return events


def superseded(engine):
    """With this property the single compliant V2 compartment no longer emits blood."""
    return PROPERTY in engine.scene.properties


def metrics(engine):
    s = engine.state.get(PROPERTY)
    if s is None:
        return {}
    uncontrolled = [w for w in s['wounds'].values() if w['sealFraction'] < 1]
    held = sum(s['reservoirMm3'].values())
    return {'bloodLossMm3': s['bloodLossMm3'], 'bloodLossByKindMm3': copy.deepcopy(s['lossByKindMm3']),
            'bloodLossBySegmentMm3': copy.deepcopy(s['segmentLossMm3']), 'tumourOozeMm3': s['tumourOozeMm3'],
            'openVesselWoundsByKind': {kind: sum(w['kind'] == kind for w in uncontrolled) for kind in KINDS},
            'largestUncontrolledVesselCalibreMm': max((w['calibreMm'] for w in uncontrolled), default=None),
            'bleedingRateMm3PerSecond': s['rateMm3PerSecond'],
            'oozingTumourCells': sum(1 for e in s['oozeCells'].values() if not e[3] and e[1] > 0),
            'blockedBleedingReservoirMm3': held, 'depositedBloodMm3': s['depositedMm3'],
            'bloodConservationResidualMm3': s['bloodLossMm3'] - s['depositedMm3'] - held}


def descriptor(engine):
    if PROPERTY not in engine.scene.properties:
        return None
    p = engine.scene.properties[PROPERTY]
    return {'model': SCHEMA, 'parameters': copy.deepcopy(p['parameters']), 'sources': copy.deepcopy(p['sources']),
            'segmentCount': len(p['segments']),
            'segmentsByKind': {kind: sum(seg['kind'] == kind for seg in p['segments']) for kind in KINDS},
            'calibreEstimator': CALIBRE_ESTIMATOR, 'oozeFields': list(OOZE_FIELDS),
            'summary': metrics(engine), 'limits': list(LIMITS)}

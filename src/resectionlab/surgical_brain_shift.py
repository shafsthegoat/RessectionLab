"""Opt-in brain-shift SCENARIO after dural opening; a labelled scenario, not a prediction.

Once the dura is open and CSF drains, the brain sags under gravity, so tissue is
no longer where the preoperative image (and a navigation system registered to
it) places it. A scene that declares ``brainShiftV1`` receives a smooth,
analytic displacement field that starts when operative access first opens and
ramps to a declared peak over a declared settling time. The field is largest
at the exposed cortex under the craniotomy, decays with depth and with lateral
distance from the opening, and tapers to zero within one cell of rigid
material (bone, dura still attached to bone, any non-shifting material), so no
falx, tentorium or skull-base neighbour moves and no tissue lands on rigid
material.

Representation. The field is never written to ``state['deformation']``; state
stores only compact parameters (onset, direction, peak, settling level), so
replay, snapshots and state ids stay small and exact. For each settling level a
destination map is computed once and cached on the scene:

* every shifting source cell has one forward destination (its rounded shifted
  voxel); a rounded destination that would be rigid is pulled back along the
  displacement until it is not, so the map is a function with no rigid overlap;
* several sources landing on one cell keep every source identity as
  memberships, the representative chosen by the existing region precedence
  (later declared regions own overlaps);
* a vacated cell that no source claims is filled from the source the inverse
  field points to, when that source's own destination is adjacent, so rounding
  never leaves a one-cell hole; cells whose pre-image lies above the cortex stay
  empty, which opens the CSF gap under the dura while free space downstream
  closes as tissue moves into it.

Instruments query the same occupancy, so they meet tissue at its shifted
position, while removal and every ledger still name the original source cells.
The protected-displacement guard reads only ``state['deformation']``, so this
scenario displacement is excluded from it by construction.

Every constant below is assumed research unless a source is named. Published
intraoperative series cited in ``SOURCES`` describe the kind and order of
magnitude of shift; no coefficient is fitted to them.
"""
from __future__ import annotations

from collections import OrderedDict
import copy
import json
import math

import numpy as np

from .surgical_occupancy import MaterialOccupancy

PROPERTY = 'brainShiftV1'
SCHEMA = 'resectionlab.brain-shift-scenario.v1'
STATE_SCHEMA = 'resectionlab.brain-shift-state.v1'
AUTO_DIRECTION = 'patient-position-or-opening-normal'
DEEP_TARGET_MARGIN = 'deep-target-margin'
ONSET = 'dural-opening'
SHIFTING_MATERIALS = ('soft_tissue', 'tumor', 'vessel')
_BOUNDS = {'peakSagMm': (0., 15.), 'lateralDecayMm': (1., 200.), 'boundaryTaperMm': (.5, 50.),
           'settlingSeconds': (0., 3600.)}
SOURCES = [
    {'id': 'roberts-1998', 'citation': 'Roberts DW, Hartov A, Kennedy FE, Miga MI, Paulsen KD. Intraoperative brain '
     'shift and deformation: a quantitative analysis of cortical displacement in 28 cases. Neurosurgery. 1998.',
     'scope': 'Cortical surface displacement of several millimetres up to about a centimetre, predominantly along '
              'gravity; context for the default order of magnitude and direction only.'},
    {'id': 'nimsky-2000', 'citation': 'Nimsky C, Ganslandt O, Cerny S, Hastreiter P, Greiner G, Fahlbusch R. '
     'Quantification of, visualization of, and compensation for brain shift using intraoperative magnetic '
     'resonance imaging. Neurosurgery. 2000.',
     'scope': 'Intraoperative MRI series: cortical shift exceeds deep tumour margin shift; qualitative depth decay only.'},
    {'id': 'nabavi-2001', 'url': 'https://pubmed.ncbi.nlm.nih.gov/11322439/',
     'citation': 'Nabavi A, et al. Serial intraoperative magnetic resonance imaging of brain shift. Neurosurgery. 2001.',
     'scope': 'Gravity largely explained surface shift; subsurface shift during resection was more complex.'},
]
LIMITS = [
    'Labelled scenario, not a patient prediction: no CSF volume, brain compliance, osmotic therapy or resection-cavity collapse is modeled.',
    'Analytic field rasterized to whole source voxels; compressed tissue keeps every source identity.',
    'Shift does not change after it settles; closing the dura does not restore the brain.',
    'Rigid material and any non-declared region stay fixed; tissue never enters it, so outward sag through an opening is clamped.',
]


def default_property(scene, **overrides):
    """A complete declaration with assumed-research defaults for ``scene``."""
    result = {
        'schema': SCHEMA, 'scenario': 'gravity-sag-after-dural-opening',
        'gravityRasUnit': AUTO_DIRECTION,
        # assumed-research, order of magnitude from the cortical series in SOURCES.
        'peakSagMm': 5.0,
        # assumed-research: depth at which the sag halves; by default the deepest
        # original target margin, so that margin moves about half the peak.
        'depthHalfLengthMm': DEEP_TARGET_MARGIN,
        # assumed-research: e-folding distance beyond the opening edge.
        'lateralDecayMm': 15.0,
        # assumed-research: distance over which the sag tapers to zero at rigid
        # supports (beyond the one pinned cell); short, so a small craniotomy
        # still shows the full sag at the centre of the exposed cortex.
        'boundaryTaperMm': 3.0,
        # assumed-research: sag settles over a minute after the dura opens, in six steps.
        'settlingSeconds': 60.0, 'settlingSteps': 6,
        'onset': ONSET,
        'regionIds': intracranial_region_ids(scene),
        'provenance': {'evidenceClass': 'assumed-research', 'calibration': 'none',
                       'statement': 'Scenario chosen to show the navigation error a surgeon meets after dural opening; '
                                    'not fitted to, or predictive of, any patient.',
                       'sources': copy.deepcopy(SOURCES)},
        'limits': list(LIMITS)}
    result.update(overrides)
    return result


def intracranial_region_ids(scene, *, bins=96):
    """Soft, tumour and vessel regions whose cells mostly lie inside the cranial shell.

    Seen from the target centroid, a cell is inside when it is nearer than the
    outermost bone or membrane cell in its direction (angular bins of about
    two degrees); directions with no shell inside the image do not vote. A
    region joins when at least 80% of its voting cells are inside, or more
    than half are and it touches that clearly intracranial set. Scalp and
    extracranial tissue lie beyond the shell. Soft connectivity is not used:
    on a real or cropped head it leaks from brain to scalp through the skull
    base or the image border.
    """
    by_id = {r['id']: r for r in scene.regions}
    shifting = [r['id'] for r in scene.regions if r['material'] in SHIFTING_MATERIALS]
    targets = [r['id'] for r in scene.regions if r.get('target')]
    if not targets:
        raise ValueError('BRAIN_SHIFT_REQUIRES_TARGET_REGION')
    flat = scene.labels.ravel()
    center = scene.points(np.flatnonzero(np.isin(flat, targets))).mean(0)

    def polar(cells):
        offset = scene.points(cells) - center
        radius = np.linalg.norm(offset, axis=1)
        theta = np.arccos(np.clip(offset[:, 2] / np.maximum(radius, 1e-12), -1., 1.))
        phi = np.arctan2(offset[:, 1], offset[:, 0])
        index = (np.minimum((theta / math.pi * bins).astype(int), bins - 1) * (2 * bins)
                 + np.minimum(((phi + math.pi) / (2 * math.pi) * 2 * bins).astype(int), 2 * bins - 1))
        return index, radius

    shell_ids = [r['id'] for r in scene.regions if r['material'] in ('bone', 'membrane')]
    index, radius = polar(np.flatnonzero(np.isin(flat, shell_ids)))
    outer = np.zeros(2 * bins * bins)
    np.maximum.at(outer, index, radius)
    candidates = np.flatnonzero(np.isin(flat, shifting))
    index, radius = polar(candidates)
    # A direction with no shell inside the image (a cropped field of view)
    # gives no evidence either way; the vote uses the remaining cells.
    known = outer[index] > 0
    votes = np.bincount(flat[candidates[known]], minlength=int(flat.max()) + 1)
    inside = np.bincount(flat[candidates[known & (radius < outer[index])]], minlength=len(votes))
    share = {r: inside[r] / votes[r] for r in shifting if votes[r]}
    result = [r for r in shifting if share.get(r, 0.) >= .8]
    # A region only partly seen inside the shell (for example cerebellum under
    # a cropped skull base) joins when it touches the clearly intracranial set.
    if result:
        from scipy import ndimage
        near = ndimage.binary_dilation(np.isin(scene.labels, result))
        for r in shifting:
            mask = scene.labels == r
            if .5 < share.get(r, 0.) < .8 and np.count_nonzero(near & mask) >= .01 * np.count_nonzero(mask):
                result.append(r)
    result = sorted(int(r) for r in result)
    if not any(by_id[i].get('target') for i in result):
        raise ValueError('BRAIN_SHIFT_REQUIRES_TARGET_REGION')
    return sorted(result)


def _number(value, name, low, high, *, inclusive_low=False):
    if type(value) not in (int, float) or not math.isfinite(value) or not (
            (low <= value if inclusive_low else low < value) and value <= high):
        raise ValueError('BRAIN_SHIFT_INVALID: ' + name)
    return float(value)


def declaration(scene):
    """Validated declaration, or None when the scene does not opt in."""
    if PROPERTY not in scene.properties:
        return None
    declared = scene.properties[PROPERTY]
    required = {'schema', 'scenario', 'gravityRasUnit', 'peakSagMm', 'depthHalfLengthMm', 'lateralDecayMm',
                'boundaryTaperMm', 'settlingSeconds', 'settlingSteps', 'onset', 'regionIds', 'provenance', 'limits'}
    if not isinstance(declared, dict) or set(declared) != required or declared['schema'] != SCHEMA:
        raise ValueError('BRAIN_SHIFT_INVALID: complete ' + SCHEMA + ' declaration required')
    if not isinstance(declared['scenario'], str) or not declared['scenario'].strip():
        raise ValueError('BRAIN_SHIFT_INVALID: scenario')
    result = {'schema': SCHEMA, 'scenario': declared['scenario']}
    direction = declared['gravityRasUnit']
    if direction != AUTO_DIRECTION:
        if not isinstance(direction, (list, tuple)) or len(direction) != 3:
            raise ValueError('BRAIN_SHIFT_INVALID: gravityRasUnit')
        vector = np.asarray([_number(v, 'gravityRasUnit', -1.5, 1.5, inclusive_low=True) for v in direction])
        if abs(np.linalg.norm(vector) - 1.) > 1e-6:
            raise ValueError('BRAIN_SHIFT_INVALID: gravityRasUnit must be a unit vector')
        direction = vector.tolist()
    result['gravityRasUnit'] = direction
    for name, (low, high) in _BOUNDS.items():
        result[name] = _number(declared[name], name, low, high)
    depth = declared['depthHalfLengthMm']
    result['depthHalfLengthMm'] = depth if depth == DEEP_TARGET_MARGIN else _number(depth, 'depthHalfLengthMm', 1., 200.)
    steps = declared['settlingSteps']
    if type(steps) is not int or not 1 <= steps <= 60:
        raise ValueError('BRAIN_SHIFT_INVALID: settlingSteps')
    result['settlingSteps'] = steps
    if declared['onset'] != ONSET:
        raise ValueError('BRAIN_SHIFT_INVALID: onset must be ' + ONSET)
    result['onset'] = ONSET
    by_id = {r['id']: r for r in scene.regions}
    ids = declared['regionIds']
    if (type(ids) is not list or not ids or ids != sorted(set(ids)) or any(type(i) is not int for i in ids)
            or any(i not in by_id or by_id[i]['material'] not in SHIFTING_MATERIALS for i in ids)):
        raise ValueError('BRAIN_SHIFT_INVALID: regionIds must name sorted soft, tumour or vessel regions')
    result['regionIds'] = list(ids)
    provenance = declared['provenance']
    if (not isinstance(provenance, dict) or provenance.get('evidenceClass') != 'assumed-research'
            or not isinstance(provenance.get('statement'), str) or not provenance['statement'].strip()):
        raise ValueError('BRAIN_SHIFT_INVALID: assumed-research provenance required')
    result['provenance'] = copy.deepcopy(provenance)
    limits = declared['limits']
    if type(limits) is not list or not limits or any(not isinstance(item, str) or not item.strip() for item in limits):
        raise ValueError('BRAIN_SHIFT_INVALID: limits')
    result['limits'] = list(limits)
    if 'operativeExposureV1' not in scene.properties:
        raise ValueError('BRAIN_SHIFT_REQUIRES_OPERATIVE_EXPOSURE: onset is the dural opening')
    return result


def _cache(scene):
    cache = getattr(scene, '_brain_shift_cache', None)
    if cache is None:
        cache = scene._brain_shift_cache = {'geometry': {}, 'maps': OrderedDict()}
    return cache


def _key(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'))


class _Geometry:
    """Level-independent amplitude field over the declared shifting cells."""

    def __init__(self, scene, decl):
        from scipy import ndimage
        from .surgical_exposure_geometry import BODIES
        shape = scene.shape
        labels = scene.labels
        exposure = scene.properties['operativeExposureV1']
        bodies = np.zeros(labels.size, bool)
        for name in BODIES:
            bodies[np.asarray(exposure['bodies'][name]['cells'], dtype=np.int64)] = True
        domain = np.isin(labels, decl['regionIds']).ravel() & ~bodies
        obstacle = (labels.ravel() != 0) & ~domain
        self.shape, self.affine, self.inverse = shape, scene.affine, scene.inverse
        self.domain_flat, self.obstacle_flat = domain, obstacle
        self.cells = np.flatnonzero(domain).astype(np.int64)
        if not len(self.cells):
            raise ValueError('BRAIN_SHIFT_INVALID: no shifting source cells')
        # Distance to rigid support: every non-shifting material except the
        # mobile exposure bodies that open the craniotomy.
        anchors = (obstacle & ~bodies).reshape(shape)
        ijk = np.column_stack(np.unravel_index(self.cells, shape))
        margin = int(math.ceil((decl['boundaryTaperMm'] + 2 * float(scene.spacing.max())) / float(scene.spacing.min()))) + 1
        low = np.maximum(ijk.min(0) - margin, 0)
        high = np.minimum(ijk.max(0) + margin + 1, shape)
        box = tuple(slice(a, b) for a, b in zip(low, high))
        self.box_low, self.box_high = low, high
        self.support = ndimage.distance_transform_edt(~anchors[box], sampling=scene.spacing).astype(np.float32)
        # Opening frame: the dural flap that opens access.
        dura = np.asarray(exposure['bodies']['dura']['cells'], dtype=np.int64)
        points = scene.points(dura)
        outward = np.asarray(exposure['exterior']['normalRas'], float)
        self.inward = -outward / np.linalg.norm(outward)
        self.center = points.mean(0)
        offset = points - self.center
        self.radius = float(np.max(np.linalg.norm(offset - np.outer(offset @ self.inward, self.inward), axis=1)))
        self.cell_mm = float(scene.spacing.max())
        self.taper = decl['boundaryTaperMm']
        self.lateral_decay = decl['lateralDecayMm']
        depth, lateral = self._frame(scene.points(self.cells))
        # The exposed cortical surface: the shallowest shifting cell in the
        # central footprint of the opening.
        central = lateral <= max(self.cell_mm, .25 * self.radius)
        self.surface = float(np.min(depth[central] if np.any(central) else depth))
        targets = [r['id'] for r in scene.regions if r.get('target')]
        target_cells = np.flatnonzero(np.isin(labels, targets).ravel())
        self.target_cells = target_cells
        if decl['depthHalfLengthMm'] == DEEP_TARGET_MARGIN:
            if not len(target_cells):
                raise ValueError('BRAIN_SHIFT_INVALID: deep-target-margin requires target cells')
            self.half_depth = max(float(np.max(self._frame(scene.points(target_cells))[0]) - self.surface), self.cell_mm)
        else:
            self.half_depth = decl['depthHalfLengthMm']
        self.amplitude = self.evaluate(ijk)
        self.ijk = ijk.astype(np.int64)

    def _frame(self, points):
        offset = points - self.center
        depth = offset @ self.inward
        lateral = np.linalg.norm(offset - np.outer(depth, self.inward), axis=1)
        return depth, lateral

    def evaluate(self, ijk):
        """Unit-peak amplitude at integer voxel indices (any cell, shifting or not)."""
        ijk = np.asarray(ijk, dtype=np.int64).reshape(-1, 3)
        points = ijk @ self.affine[:3, :3].T + self.affine[:3, 3]
        depth, lateral = self._frame(points)
        depth = np.maximum(depth - self.surface, 0.)
        local = np.clip(ijk - self.box_low, 0, np.asarray(self.support.shape) - 1)
        inside = np.all((ijk >= self.box_low) & (ijk < self.box_high), axis=1)
        support = np.where(inside, self.support[tuple(local.T)], 0.)
        return (np.exp2(-depth / self.half_depth) * np.exp(-np.maximum(lateral - self.radius, 0.) / self.lateral_decay)
                * np.clip((support - self.cell_mm) / self.taper, 0., 1.))


def geometry(scene, decl):
    cache = _cache(scene)['geometry']
    key = _key(decl)
    if key not in cache:
        cache[key] = _Geometry(scene, decl)
    return cache[key]


class ShiftMap:
    """Immutable destination map for one settling level (sparse over changed cells)."""

    def __init__(self, scene, decl, direction, scale):
        geo = geometry(scene, decl)
        self.geometry, self.scale = geo, float(scale)
        shape = np.asarray(scene.shape)
        size = int(np.prod(shape))
        self.unit_vox = scene.inverse[:3, :3] @ np.asarray(direction, float)
        self.direction = np.asarray(direction, float)
        labels = scene.labels.ravel()
        cells, ijk = geo.cells, geo.ijk
        displacement = np.outer(geo.amplitude * scale, self.unit_vox)
        factor = np.ones(len(cells))
        candidate = np.flatnonzero(np.max(np.abs(displacement), axis=1) >= .5 - 1e-12)
        destination = cells.copy()
        if len(candidate):
            remaining = candidate
            for fraction in (1., .75, .5, .25, 0.):
                if not len(remaining):
                    break
                target = np.rint(ijk[remaining] + fraction * displacement[remaining]).astype(np.int64)
                inside = np.all((target >= 0) & (target < shape), axis=1)
                flat = np.full(len(remaining), -1, dtype=np.int64)
                flat[inside] = np.ravel_multi_index(target[inside].T, scene.shape)
                good = inside.copy()
                good[inside] = ~geo.obstacle_flat[flat[inside]]
                destination[remaining[good]] = flat[good]
                factor[remaining[good]] = fraction
                remaining = remaining[~good]
        self.factor = factor
        moving = np.flatnonzero(destination != cells)
        self.sources = cells[moving]
        self.targets = destination[moving]
        is_moving = np.zeros(size, bool)
        is_moving[self.sources] = True
        precedence = np.zeros(int(labels.max()) + 1, dtype=np.int64)
        for rank, region in enumerate(scene.regions):
            precedence[region['id']] = rank
        # Forward claims, grouped by destination.
        order = np.argsort(self.targets, kind='stable')
        claimed, first, counts = np.unique(self.targets[order], return_index=True, return_counts=True)
        stationary = geo.domain_flat[claimed] & ~is_moving[claimed]
        sorted_sources = self.sources[order]
        single = (counts == 1) & ~stationary
        locations, owners = [claimed[single]], [sorted_sources[first[single]] + 1]
        members, shared_locations, shared_owners = {}, [], []
        for index in np.flatnonzero(~single).tolist():
            location = int(claimed[index])
            claim = sorted_sources[first[index]:first[index] + counts[index]].tolist()
            if stationary[index]:
                claim.append(location)
            claim = sorted(set(claim), key=lambda c: (-int(precedence[labels[c]]), -c))
            shared_locations.append(location)
            shared_owners.append(claim[0] + 1)
            if len(claim) > 1:
                members[location] = tuple(claim)
        locations.append(np.asarray(shared_locations, dtype=np.int64))
        owners.append(np.asarray(shared_owners, dtype=np.int64))
        # Vacated or neighbouring free cells: fill from the inverse field when
        # that source's own destination is adjacent; otherwise leave empty.
        neighbours = [self.sources]
        claimed_ijk = np.column_stack(np.unravel_index(claimed, scene.shape))
        for axis in range(3):
            for sign in (-1, 1):
                step = claimed_ijk.copy()
                step[:, axis] += sign
                ok = (step[:, axis] >= 0) & (step[:, axis] < shape[axis])
                neighbours.append(np.ravel_multi_index(step[ok].T, scene.shape))
        candidates = np.unique(np.concatenate(neighbours))
        claimed_mask = np.zeros(size, bool)
        claimed_mask[claimed] = True
        candidates = candidates[~claimed_mask[candidates] & ~geo.obstacle_flat[candidates]
                                & ~(geo.domain_flat[candidates] & ~is_moving[candidates])]
        fills = np.zeros(len(candidates), dtype=np.int64)
        if len(candidates):
            y = np.column_stack(np.unravel_index(candidates, scene.shape)).astype(float)
            first_guess = np.rint(y - np.outer(geo.evaluate(np.rint(y)) * scale, self.unit_vox))
            guess = np.rint(y - np.outer(geo.evaluate(first_guess) * scale, self.unit_vox)).astype(np.int64)
            inside = np.all((guess >= 0) & (guess < shape), axis=1)
            source = np.full(len(candidates), -1, dtype=np.int64)
            source[inside] = np.ravel_multi_index(guess[inside].T, scene.shape)
            valid = inside.copy()
            valid[inside] = geo.domain_flat[source[inside]]
            lookup = np.searchsorted(self.sources, source[valid])
            lookup = np.minimum(lookup, max(len(self.sources) - 1, 0))
            landed = source[valid].copy()
            if len(self.sources):
                hit = self.sources[lookup] == source[valid]
                landed[hit] = self.targets[lookup[hit]]
            near = np.max(np.abs(np.column_stack(np.unravel_index(landed, scene.shape)) - y[valid]), axis=1) <= 1
            chosen = np.flatnonzero(valid)[near]
            fills[chosen] = source[chosen] + 1
        identity = np.where(labels[candidates] != 0, candidates + 1, 0)
        changed = fills != identity
        locations.append(candidates[changed])
        owners.append(fills[changed])
        self.members = members
        self._publish(np.concatenate(locations), np.concatenate(owners))
        self.filled = int(np.count_nonzero(fills))
        # A vacated cell enclosed on all six faces would let an instrument
        # fall through a one-cell hole: fill it from the face neighbour whose
        # source lies nearest to the inverse-field pre-image.
        empty = candidates[fills == 0]
        empty = empty[self.base(empty, labels) == 0] if len(empty) else empty
        if len(empty):
            ijk_empty = np.column_stack(np.unravel_index(empty, scene.shape))
            faces = []
            for axis in range(3):
                for sign in (-1, 1):
                    step = ijk_empty.copy()
                    step[:, axis] = np.clip(step[:, axis] + sign, 0, shape[axis] - 1)
                    faces.append(self.base(np.ravel_multi_index(step.T, scene.shape), labels))
            faces = np.column_stack(faces)
            enclosed = np.flatnonzero(np.all(faces != 0, axis=1))
            if len(enclosed):
                pre = ijk_empty[enclosed] - np.outer(geo.evaluate(ijk_empty[enclosed]) * scale, self.unit_vox)
                choice = faces[enclosed] - 1
                centres = np.stack(np.unravel_index(choice, scene.shape), axis=-1)
                nearest = np.argmin(np.linalg.norm(centres - pre[:, None, :], axis=2), axis=1)
                self._publish(np.concatenate([self.locations, empty[enclosed]]),
                              np.concatenate([self.owners, choice[np.arange(len(enclosed)), nearest] + 1]))
                self.filled += len(enclosed)

    def _publish(self, locations, owners):
        """Sorted sparse owners; a later entry for one location replaces an earlier one."""
        reverse = slice(None, None, -1)
        unique, index = np.unique(locations[reverse], return_index=True)
        self.locations = unique.astype(np.int64)
        self.owners = owners[reverse][index].astype(np.int64)
        self.member_locations = np.asarray(sorted(self.members), dtype=np.int64)

    def displacement_vox(self, cells):
        """Continuous clamped voxel displacement of source cells (zero off the shifting domain)."""
        cells = np.asarray(cells, dtype=np.int64).ravel()
        geo = self.geometry
        position = np.searchsorted(geo.cells, cells)
        position = np.minimum(position, len(geo.cells) - 1)
        inside = geo.cells[position] == cells
        result = np.zeros((len(cells), 3))
        index = position[inside]
        result[inside] = np.outer(geo.amplitude[index] * self.factor[index] * self.scale, self.unit_vox)
        return result

    def displacement_mm(self, cells):
        return self.displacement_vox(cells) @ self.geometry.affine[:3, :3].T

    def forward(self, cells):
        cells = np.asarray(cells, dtype=np.int64).ravel()
        if not len(self.sources):
            return cells.copy()
        position = np.minimum(np.searchsorted(self.sources, cells), len(self.sources) - 1)
        hit = self.sources[position] == cells
        result = cells.copy()
        result[hit] = self.targets[position[hit]]
        return result

    def base(self, ids, labels):
        """Shifted owner+1 at raster cells, before removal and other overrides."""
        values = np.where(labels[ids] != 0, ids + 1, 0)
        if len(self.locations):
            position = np.minimum(np.searchsorted(self.locations, ids), len(self.locations) - 1)
            hit = self.locations[position] == ids
            values[hit] = self.owners[position[hit]]
        return values

    def sources_at(self, location, labels):
        location = int(location)
        if location in self.members:
            return list(self.members[location])
        value = int(self.base(np.asarray([location]), labels)[0])
        return [value - 1] if value else []


def state(engine):
    return engine.state.get(PROPERTY)


def access_open(engine):
    from .surgical_operative_exposure import _access
    return _access(engine)


def _direction(engine, decl, geo):
    if decl['gravityRasUnit'] != AUTO_DIRECTION:
        return list(map(float, decl['gravityRasUnit'])), 'declared'
    if 'patientPosition' in engine.state:
        from .surgical_position import describe_position
        gravity = np.asarray(describe_position(engine.state['patientPosition'])['gravityRasMPerS2'])
        return (gravity / np.linalg.norm(gravity)).tolist(), 'patientPosition'
    return geo.inward.tolist(), 'opening-inward-normal'


def advance(engine):
    """Record onset when access first opens, then step the settling level."""
    decl = engine._brain_shift
    current = engine.state.get(PROPERTY)
    elapsed = float(engine.state['elapsedSeconds'])
    if current is None:
        if not access_open(engine):
            return []
        direction, source = _direction(engine, decl, geometry(engine.scene, decl))
        engine.state[PROPERTY] = {'schema': STATE_SCHEMA, 'onsetSeconds': elapsed, 'gravityRasUnit': direction,
                                  'directionSource': source, 'peakSagMm': decl['peakSagMm'],
                                  'settlingSeconds': decl['settlingSeconds'], 'settlingSteps': decl['settlingSteps'],
                                  'level': 0}
        current = engine.state[PROPERTY]
        events = [{'kind': 'brain_shift_onset', 'onsetSeconds': elapsed, 'directionSource': source}]
    else:
        events = []
    steps = current['settlingSteps']
    if current['settlingSeconds'] <= 0:
        level = steps
    else:
        level = min(steps, int(math.floor((elapsed - current['onsetSeconds']) / current['settlingSeconds'] * steps + 1e-9)))
    if level != current['level']:
        current['level'] = level
        events.append({'kind': 'brain_shift_settling', 'level': level, 'peakMm': peak_mm(current)})
    return events


def peak_mm(current):
    return current['peakSagMm'] * current['level'] / current['settlingSteps'] if current else 0.


def active_map(engine):
    """The cached destination map of the current level, or None when no shift is active."""
    scene = getattr(engine, 'scene', None)
    current_state = getattr(engine, 'state', None)
    if type(current_state) is not dict or PROPERTY not in current_state or scene is None:
        return None
    current = current_state[PROPERTY]
    if not current.get('level'):
        return None
    decl = getattr(engine, '_brain_shift', None) or declaration(scene)
    key = _key([decl, current['gravityRasUnit'], peak_mm(current)])
    maps = _cache(scene)['maps']
    if key in maps:
        maps.move_to_end(key)
        return maps[key]
    result = ShiftMap(scene, decl, current['gravityRasUnit'], peak_mm(current))
    maps[key] = result
    while len(maps) > 3:
        maps.popitem(last=False)
    return result


def shifted_raster(engine, *, include_rods=True, local_only=False):
    """Composed source->raster pairs, or None when no shift is active.

    Cells with a local lattice displacement compose it with the scenario shift.
    ``local_only`` omits the forward pairs of the remaining shifted cells.
    """
    shift = active_map(engine)
    if shift is None:
        return None
    scene = engine.scene
    removed = set(engine.state['removed'])
    excluded = ()
    if not include_rods:
        from . import surgical_vessel_rods as rods
        if rods.enabled(engine):
            excluded = rods.definitions(engine).index
    local = [(int(key), value) for key, value in engine.state.get('deformation', {}).items()
             if int(key) not in removed and int(key) not in excluded]
    pairs = {}
    if local:
        cells = np.asarray([cell for cell, _ in local], dtype=np.int64)
        ijk = np.column_stack(np.unravel_index(cells, scene.shape))
        moved = ijk + np.asarray([value for _, value in local], float) @ scene.inverse[:3, :3].T + shift.displacement_vox(cells)
        target = np.rint(moved).astype(np.int64)
        if np.any(target < 0) or np.any(target >= scene.shape):
            raise ValueError('DISPLACEMENT_OUTSIDE_GRID')
        pairs.update(zip(cells.tolist(), np.ravel_multi_index(target.T, scene.shape).tolist()))
    if not local_only and len(shift.sources):
        keep = np.ones(len(shift.sources), bool)
        skip = removed | set(pairs) | set(map(int, excluded))
        if skip:
            keep = ~np.isin(shift.sources, np.fromiter(skip, dtype=np.int64, count=len(skip)))
        for source, target in zip(shift.sources[keep].tolist(), shift.targets[keep].tolist()):
            pairs[source] = target
    return [[cell, pairs[cell]] for cell in sorted(pairs)]


def forward_pairs(engine, cells):
    """Scenario destinations of the given source cells, excluding removed ones."""
    shift = active_map(engine)
    if shift is None:
        return []
    removed = set(engine.state['removed'])
    cells = np.asarray([int(c) for c in cells if int(c) not in removed], dtype=np.int64)
    return list(zip(cells.tolist(), shift.forward(cells).tolist()))


class ShiftedOccupancy(MaterialOccupancy):
    """Occupancy over the shifted base map; ``cleared`` erases SOURCE identities.

    Without a shift the base is the identity, where clearing a source and
    clearing its raster cell coincide; this generalizes that rule.
    """

    def __init__(self, labels, cleared, moved, memberships, shift, precedence):
        self.labels = labels.ravel()
        self.cleared = frozenset(cleared)
        self.moved = dict(moved)
        self._overrides = dict(memberships or {})
        self._shift = shift
        self._precedence = precedence
        self._cleared_array = np.asarray(sorted(self.cleared), dtype=np.int64)
        self._merged = None
        self._integer_cleared = None
        self._sorted_integer_cleared = None
        # The override mapping is built once by occupancy() and never edited.
        self._moved_keys = np.array(sorted(self.moved), dtype=np.int64)
        self._moved_owners = np.array([self.moved[int(k)] for k in self._moved_keys], dtype=np.int64)

    @property
    def memberships(self):
        if self._merged is None:
            merged = {}
            for location, owners in self._shift.members.items():
                if location in self.moved or location in self._overrides:
                    continue
                kept = [c for c in owners if c not in self.cleared]
                if kept:
                    merged[location] = kept
            merged.update(self._overrides)
            self._merged = merged
        return self._merged

    @memberships.setter
    def memberships(self, value):
        self._overrides = dict(value or {})
        self._merged = None

    def _is_cleared(self, sources):
        if not len(self._cleared_array):
            return np.zeros(np.shape(sources), bool)
        position = np.minimum(np.searchsorted(self._cleared_array, sources), len(self._cleared_array) - 1)
        return self._cleared_array[position] == sources

    def _values(self, ids):
        flat = np.asarray(ids, dtype=np.int64).ravel()
        values = self._shift.base(flat, self.labels)
        if len(self._cleared_array):
            occupied = values != 0
            erased = np.zeros(len(flat), bool)
            erased[occupied] = self._is_cleared(values[occupied] - 1)
            values[erased] = 0
            shared = self._shift.member_locations
            if len(shared) and np.any(erased):
                for index in np.flatnonzero(erased).tolist():
                    owners = self._shift.members.get(int(flat[index]))
                    if owners:
                        kept = next((c for c in owners if c not in self.cleared), None)
                        if kept is not None:
                            values[index] = kept + 1
        if len(self._moved_keys):
            keys = self._moved_keys
            position = np.minimum(np.searchsorted(keys, flat), len(keys) - 1)
            matched = keys[position] == flat
            values[matched] = self._moved_owners[position[matched]]
        return values.reshape(np.shape(ids))

    def __getitem__(self, indices):
        if np.isscalar(indices):
            return int(self._values(np.asarray([int(indices)]))[0])
        return self._values(np.asarray(indices, dtype=np.int64))

    def material_ids(self, indices):
        ids = np.unique(np.asarray(indices, dtype=np.int64).ravel())
        values = self._values(ids)
        result = set()
        members = self._shift.members
        for cell, value in zip(ids.tolist(), values.tolist()):
            if cell in self._overrides:
                result.update(self._overrides[cell])
            elif cell in self.moved:
                result.add(value - 1)
            elif cell in members:
                result.update(c for c in members[cell] if c not in self.cleared)
            elif value:
                result.add(value - 1)
        return sorted(result)

    def __array__(self, dtype=None, copy=None):
        result = self._values(np.arange(len(self.labels), dtype=np.int64))
        return result.astype(dtype, copy=False) if dtype is not None else result


def occupancy(engine):
    """Shared occupancy with the scenario shift as its base; mirrors SurgicalEngine.occupancy."""
    from . import surgical_vessel_rods as rods
    from . import surgical_operative_exposure as exposure
    shift = active_map(engine)
    scene = engine.scene
    labels = scene.labels.ravel()
    removed = set(engine.state['removed'])
    cleared = set(removed)
    if rods.enabled(engine):
        cleared.update(rods.definitions(engine).index)
    precedence = {r['id']: rank for rank, r in enumerate(scene.regions)}
    moves = []
    held_ids = set()
    for hold in [*engine.state['holds'].values(), *engine.state.get('detachedFragments', [])]:
        if 'target' in hold:
            continue
        retained = np.array([i for i in hold['cells'] if i not in removed], dtype=np.int64)
        if not len(retained):
            continue
        if held_ids.intersection(retained.tolist()):
            raise ValueError('MATERIAL_ALREADY_HELD')
        held_ids.update(retained.tolist())
        indices = np.column_stack(np.unravel_index(retained, scene.shape)) + hold['offset']
        if np.any(indices < 0) or np.any(indices >= scene.shape):
            raise ValueError('DISPLACEMENT_OUTSIDE_GRID')
        cleared.update(retained.tolist())
        moves.append((retained, np.ravel_multi_index(indices.T, scene.shape)))
    operative_pairs = exposure.moved(engine)
    cleared.update(original for original, _ in operative_pairs)
    soft_pairs = shifted_raster(engine, include_rods=False, local_only=True)
    cleared.update(original for original, _ in soft_pairs)

    # Base occupants of every queried raster cell, resolved in one vectorized pass.
    queried = np.unique(np.concatenate([np.asarray([t for _, t in soft_pairs], dtype=np.int64),
                                        np.asarray([t for _, t in operative_pairs], dtype=np.int64),
                                        *[destination for _, destination in moves]]))
    values = shift.base(queried, labels) if len(queried) else queried
    present_by_cell = {}
    for location, value in zip(queried.tolist(), values.tolist()):
        owners = shift.members.get(location, (value - 1,) if value else ())
        present_by_cell[location] = [c for c in owners if c not in cleared]

    def occupants(target):
        return present_by_cell[int(target)]

    destinations = {}
    memberships = {}
    for retained, destination in moves:
        for original, target in zip(retained.tolist(), destination.tolist()):
            if target in destinations or occupants(target):
                raise ValueError('DISPLACEMENT_COLLISION')
            destinations[int(target)] = int(original) + 1
    for original, target in soft_pairs:
        if target in destinations:
            raise ValueError('DISPLACEMENT_COLLISION')
        present = occupants(target)
        for cell in present:
            if engine.by_label[int(labels[cell])]['material'] not in ('soft_tissue', 'tumor'):
                raise ValueError('DISPLACEMENT_COLLISION')
        memberships.setdefault(target, []).extend(present)
        memberships[target].append(original)
    for target, originals in memberships.items():
        memberships[target] = sorted(set(originals))
        destinations[target] = max(memberships[target], key=lambda c: (precedence[int(labels[c])], c)) + 1
    for original, target in operative_pairs:
        previous = destinations.get(target)
        if previous is None:
            present = occupants(target)
            if present:
                previous = max(present, key=lambda c: (precedence[int(labels[c])], c)) + 1
        if previous is not None:
            if engine._operative_body_for_cell.get(previous - 1) != engine._operative_body_for_cell[original]:
                raise ValueError('OPERATIVE_DISPLACEMENT_COLLISION')
            memberships.setdefault(target, [previous - 1]).append(original)
        destinations[target] = original + 1
    return ShiftedOccupancy(scene.labels, cleared, destinations, memberships, shift, precedence)


def _target_centroids(engine, shift):
    geo = geometry(engine.scene, engine._brain_shift)
    cells = geo.target_cells
    if not len(cells):
        return None, None
    points = engine.scene.points(cells)
    planned = points.mean(0)
    shifted = planned if shift is None else (points + shift.displacement_mm(cells)).mean(0)
    return planned, shifted


def metrics(engine):
    current = engine.state.get(PROPERTY)
    shift = active_map(engine)
    planned, shifted = _target_centroids(engine, shift)
    return {'brainShiftPeakMm': peak_mm(current),
            'targetShiftMm': 0. if planned is None else float(np.linalg.norm(shifted - planned))}


def view(engine, *, max_samples=512):
    """Compact descriptor: parameters, current peak, a coarse lattice and target centroids."""
    decl = engine._brain_shift
    current = engine.state.get(PROPERTY)
    geo = geometry(engine.scene, decl)
    shift = active_map(engine)
    planned, shifted = _target_centroids(engine, shift)
    samples = []
    if shift is not None:
        moving = np.flatnonzero(geo.amplitude * shift.scale > .05)
        if len(moving):
            stride = 4
            while True:
                pick = moving[np.all(geo.ijk[moving] % stride == 0, axis=1)]
                if len(pick) <= max_samples:
                    break
                stride *= 2
            cells = geo.cells[pick]
            points = engine.scene.points(cells)
            vectors = shift.displacement_mm(cells)
            samples = [{'positionRasMm': p.tolist(), 'displacementMm': v.tolist()} for p, v in zip(points, vectors)]
    return {'schema': SCHEMA, 'scenario': decl['scenario'], 'evidenceClass': 'assumed-research scenario, not a prediction',
            'declaration': {k: copy.deepcopy(decl[k]) for k in ('gravityRasUnit', 'peakSagMm', 'depthHalfLengthMm',
                            'lateralDecayMm', 'boundaryTaperMm', 'settlingSeconds', 'settlingSteps', 'onset', 'provenance', 'limits')},
            'resolved': {'depthHalfLengthMm': geo.half_depth, 'openingCenterRasMm': geo.center.tolist(),
                         'openingInwardRas': geo.inward.tolist(), 'openingRadiusMm': geo.radius},
            'state': copy.deepcopy(current), 'started': current is not None, 'currentPeakMm': peak_mm(current),
            'latticeSamples': samples,
            'plannedTargetRasMm': None if planned is None else planned.tolist(),
            'shiftedTargetRasMm': None if shifted is None else shifted.tolist(),
            'rasterFilledHoleCells': 0 if shift is None else shift.filled}

"""Source-region exposure from accepted mechanics steps and logged contacts."""
from collections import Counter, defaultdict

import numpy as np

from .surgical_contact_complementarity import FORCE_TOLERANCE_N


PROPERTY = 'mechanicalExposureV1'
SCHEMA = 'resectionlab.mechanical-exposure.v1'
METRICS = {
    'contactCellSeconds': ('cell*s', 'sum'),
    'retractorTouchCellSeconds': ('cell*s', 'sum'),
    'retractorLoadedCellSeconds': ('cell*s', 'sum'),
    'retractorForcePeakN': ('N', 'peak'),
    'retractorForceTimeNs': ('N*s', 'sum'),
    'peakStrainNorm': ('1', 'peak'),
    'strainSquaredVolumeTimeMm3Seconds': ('mm3*s', 'sum'),
    'peakStressNormPa': ('Pa', 'peak'),
    'stressSquaredVolumeTimePa2Mm3Seconds': ('Pa2*mm3*s', 'sum'),
    'peakNeighborDisplacementRatio': ('1', 'peak'),
    'neighborDisplacementRatioSquaredBondSeconds': ('bond*s', 'sum'),
}

_CONTACT = ('retractorTouchCellSeconds', 'retractorLoadedCellSeconds')
_FORCE = ('retractorForcePeakN', 'retractorForceTimeNs')
_CONTINUUM = ('peakStrainNorm', 'strainSquaredVolumeTimeMm3Seconds',
              'peakStressNormPa', 'stressSquaredVolumeTimePa2Mm3Seconds')
_LATTICE = ('peakNeighborDisplacementRatio', 'neighborDisplacementRatioSquaredBondSeconds')


def profile():
    return {'schema': SCHEMA}


def validate(scene):
    if PROPERTY not in scene.properties:
        return
    if scene.properties[PROPERTY] != profile():
        raise ValueError('COMPLETE_MECHANICAL_EXPOSURE_PROFILE_REQUIRED')
    if not scene.properties.get('tissuePhysicsV2') or 'planarBladeManeuverV1' in scene.properties:
        raise ValueError('MECHANICAL_EXPOSURE_REQUIRES_SUPPORTED_V2_MECHANICS')


def enabled(engine):
    return PROPERTY in engine.scene.properties


def _row(engine, label):
    rows = engine.state.setdefault(PROPERTY, {})
    return rows.setdefault(str(label), {})


def _record(engine, label, **values):
    row = _row(engine, label)
    for key, value in values.items():
        before = row.get(key, 0.)
        row[key] = max(before, float(value)) if METRICS[key][1] == 'peak' else before + float(value)


def record_lattice_step(engine, labels, endpoints, lengths, displacement, h):
    """Count each retained bond once, sharing interface bonds between labels."""
    first, second = endpoints.T
    ratios = np.linalg.norm(displacement[first] - displacement[second], axis=1) / lengths
    for label in np.unique(labels[endpoints]):
        halves = (labels[first] == label).astype(int) + (labels[second] == label)
        _record(engine, label, peakNeighborDisplacementRatio=ratios[halves > 0].max(),
                neighborDisplacementRatioSquaredBondSeconds=(halves @ (ratios ** 2)) * (h / 2.))


def record_continuum_peaks(engine, label, values):
    _record(engine, label, peakStrainNorm=values['peakStrainNorm'], peakStressNormPa=values['peakStressNormPa'])


def record_bonded_step(engine, cells, forces, h):
    """Track source attachment duration and magnitude of the regional resultant."""
    by_label = defaultdict(list)
    labels = engine.scene.labels.ravel()
    for cell, force in zip(cells, forces):
        by_label[int(labels[cell])].append(force)
    for label, values in by_label.items():
        loaded = np.count_nonzero(np.linalg.norm(values, axis=1) > FORCE_TOLERANCE_N)
        force = np.linalg.norm(np.sum(values, axis=0))
        _record(engine, label, retractorTouchCellSeconds=len(values) * h, retractorLoadedCellSeconds=loaded * h,
                retractorForcePeakN=force, retractorForceTimeNs=force * h)


def record_surface_step(engine, result, h):
    labels = engine.scene.labels.ravel()
    touched = Counter(int(labels[c]) for c in result['contactSourceIds'])
    loaded = Counter(int(labels[c]) for c in result['loadedSourceIds'])
    candidates = touched.keys() | loaded.keys()
    for label in sorted(candidates):
        _record(engine, label, retractorTouchCellSeconds=touched[label] * h,
                retractorLoadedCellSeconds=loaded[label] * h)
    if not loaded:
        return
    if len(candidates) == 1:
        label, = candidates
        force = np.linalg.norm(result['forceOnTissueN'])
        _record(engine, label, retractorForcePeakN=force, retractorForceTimeNs=force * h)
    else:
        # Rank-deficient contact multipliers can redistribute load between
        # source faces. Retain touch/loaded samples and withhold the force split.
        for label in sorted(candidates):
            _row(engine, label)['forceAttributionUnavailable'] = True


def summarize(engine):
    """Return cumulative peaks and integrals from the declared exposure ledger.

    Subtracting two integrals gives an interval integral. Subtracting peaks
    gives the increase in the cumulative peak. The interval peak is untracked.
    """
    if not enabled(engine):
        return None
    from . import surgical_continuum as continuum
    from . import surgical_retractor_runtime as retractor

    is_continuum = continuum.PROPERTY in engine.scene.properties
    is_surface = retractor.enabled(engine)
    model_metrics = _CONTINUUM if is_continuum else _LATTICE
    model = continuum.SCHEMA if is_continuum else 'kelvin-voigt-lattice'
    boundary = ('sampled-frictionless-face' if is_surface else
                'bonded-centroid-spring' if is_continuum else 'effective-bonded-lattice-spring')
    labels = engine.scene.labels.ravel()
    contact = defaultdict(float)
    for cell, seconds in engine.state['contact'].items():
        contact[str(labels[int(cell)])] += seconds
    continuum_rows = engine.state.get(continuum.PROPERTY, {}).get('exposureBySourceLabel', {})
    saved_rows = engine.state.get(PROPERTY, {})
    regions = {}
    for region in engine.scene.regions:
        label = str(region['id'])
        soft = region['material'] in continuum.SOFT
        saved = saved_rows.get(label, {})
        row = {key: None for key in METRICS}
        row.update(material=region['material'], contactCellSeconds=contact[label])
        if soft:
            for key in (*model_metrics, *_CONTACT):
                row[key] = saved.get(key, continuum_rows.get(label, {}).get(key, 0.))
            if not saved.get('forceAttributionUnavailable'):
                for key in _FORCE:
                    row[key] = saved.get(key, 0.)
        regions[label] = row
    return {'schema': SCHEMA, 'model': model, 'retractorBoundary': boundary, 'regions': regions,
            'contactScope': 'Logged retained holds, elevation/dissection, and sampled retractor touch.',
            'loadScope': ('Source cells with a contact sample above the solver force tolerance. '
                          'Regional resultant requires single-label touch.' if is_surface else
                          'Source attachments with modeled spring force above the numerical tolerance.'),
            'peakScope': 'Cumulative maximum over accepted mechanical substeps.',
            'status': 'modeled_unvalidated'}

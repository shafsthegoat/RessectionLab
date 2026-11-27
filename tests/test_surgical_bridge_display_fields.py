"""Display-only fields the desktop bridge adds to a rehearsal view (generated grids, no patient data)."""
from fractions import Fraction

import numpy as np

from resectionlab import surgical_cusa_v4 as v4
from resectionlab.surgical_bridge import _with_display_fields, cusa_progress_display, injury_display
from resectionlab.surgical_engine import SurgicalEngine
from resectionlab.surgical_injury import PROPERTY as INJURY, default_profile as injury_profile
from resectionlab.surgical_scene import SurgicalScene


def engine_with(properties, material='tumor', name='coupon core'):
    labels = np.zeros((20, 20, 20), np.uint16)
    labels[8:11, 8:11, 8:11] = 1
    regions = [{'id': 1, 'name': name, 'material': material, 'target': material == 'tumor'}]
    return SurgicalEngine(SurgicalScene(labels, np.eye(4), regions, {'kind': 'generated'}, properties))


def test_cusa_v4_progress_uses_the_tissue_kind_threshold():
    engine = engine_with({v4.PROPERTY: {'schema': v4.SCHEMA}}, material='soft_tissue', name='white matter')
    cell = int(np.ravel_multi_index((9, 9, 9), (20, 20, 20)))
    rate = v4.profile(engine)['relativeRate']['white_matter']
    threshold = Fraction(1) / Fraction.from_float(rate)
    worked = threshold / 4
    engine.state[v4.PROPERTY]['progressExact'] = {str(cell): [worked.numerator, worked.denominator]}
    shown = cusa_progress_display(engine)
    assert shown['model'] == 'cusaV4' and shown['cells'] == [cell] and shown['fraction'] == [0.25]


def test_no_pending_work_and_absent_properties_add_nothing():
    engine = engine_with({})
    view = _with_display_fields(engine, {'stateId': 'x'})
    assert view == {'stateId': 'x'}
    assert injury_display(engine) is None


def test_injury_cells_are_passed_through_unchanged():
    engine = engine_with({INJURY: injury_profile()})
    record = engine.state['injuryV1']
    record['contused'] = [5, 3]
    record['lacerated'] = [7]
    record['events'] = [{'kind': 'contusion', 'code': 'SHAFT_COLLISION', 'cells': 2, 'structures': [], 'regions': ['coupon core'],
                         'timeSeconds': 1., 'slot': 'left', 'instrument': 'cusa', 'cellIds': [3, 5]}]
    shown = _with_display_fields(engine, {})['surgicalInjuryV1']
    assert shown['contused'] == [3, 5] and shown['lacerated'] == [7] and shown['eventCount'] == 1
    assert 'cellIds' not in shown['events'][0]
    assert engine.state['injuryV1']['contused'] == [5, 3]

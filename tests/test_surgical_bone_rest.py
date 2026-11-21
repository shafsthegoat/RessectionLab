from types import SimpleNamespace

import numpy as np
import pytest

from resectionlab import surgical_bone_rest as rest


def test_three_point_parking_rest_requires_contact_and_blocks_entry():
    shape = (19, 19, 18)
    local = np.array([[x, y, 3] for x in range(4, 13) for y in range(4, 13)])
    ids = np.ravel_multi_index(local.T, shape).tolist()
    body = {'cells':ids, 'hingeAxis':[1., 0., 0.], 'outward':[0., 0., 1.],
            'parkCenterMm':[9., 8., 12.], 'restCenterMm':local.mean(0).tolist()}
    fixture = rest.rest_declaration(shape, np.eye(4), body)
    scene = SimpleNamespace(shape=shape, affine=np.eye(4), properties={'operativeExposureV1':{
        'boneRest':fixture, 'bodies':{'bone':body}}})
    engine = SimpleNamespace(scene=scene)
    rest.initialize(engine)
    parked = local + body['parkCenterMm'] - np.array(body['restCenterMm'])
    assert np.array(list(rest._gaps(engine, parked))) == pytest.approx([0.]*3, abs=1e-8)
    assert np.all(np.array(list(rest._gaps(engine, parked + [0.,0.,1.]))) > fixture['contactToleranceMm'])
    rest.check_body(engine, 'bone', parked)
    with pytest.raises(ValueError, match='REST_BODY_COLLISION'):
        rest.check_body(engine, 'bone', parked - [0.,0.,.1])
    a, b, _, _ = engine._bone_rest[0]
    with pytest.raises(ValueError, match='REST_TOOL_COLLISION'):
        rest.check_tool(engine, [(a, b, .35, 'shaft')])
    fixture['pads'][0]['bMm'][0] += .5
    with pytest.raises(ValueError, match='BINDING_MISMATCH'):
        rest.initialize(engine)


def test_exterior_setup_cannot_move_full_handle_through_parked_hand():
    from scripts.generated_glioma_walkthrough_case import build_glioma_case
    from resectionlab.surgical_engine import SurgicalEngine

    _, scene, _ = build_glioma_case()
    engine = SurgicalEngine(scene)
    right = engine.state['slots']['right']
    end = list(right['position'])
    end[0] += 30.
    before = engine.state_id
    action = {'slot':'left', 'operation':'place', 'duration':.3,
              'pose':{'position':end, 'quaternion':right['quaternion'][:]}}
    refused = engine.step(action)
    assert refused['status'] == 'rejected' and 'TOOL_COLLISION' in refused['reason']
    assert engine.state_id == before and not engine.actions


def test_specimen_uses_vacated_cells_and_collides_with_reflected_sheet():
    from scripts.generated_glioma_walkthrough_case import build_glioma_case
    from resectionlab import surgical_operative_exposure as exposure
    from resectionlab.surgical_engine import SurgicalEngine

    _, scene, _ = build_glioma_case()
    engine = SurgicalEngine(scene)
    engine.state[exposure.PROPERTY]['bodies']['scalp']['angleRad'] = np.deg2rad(100.)
    pairs = exposure.moved(engine)
    occupied = engine.occupancy()
    vacancy = next(original for original, _ in pairs if not occupied[original])
    folded = next(target for _, target in pairs if target != vacancy)
    source = next(int(i) for i in np.flatnonzero(scene.labels)
                  if engine.by_label[int(scene.labels.flat[i])]['material'] == 'tumor')
    def offset(destination):
        return np.subtract(np.unravel_index(destination, scene.shape),
                           np.unravel_index(source, scene.shape)).tolist()
    fragment = {'cells':[source], 'offset':offset(vacancy)}
    engine.state['detachedFragments'] = [fragment]
    assert engine.occupancy().material_ids([vacancy]) == [source]
    fragment['offset'] = offset(folded)
    with pytest.raises(ValueError, match='DISPLACEMENT_COLLISION'):
        engine.occupancy()

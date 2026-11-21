"""Generated local material coupons. No cellular material parameter is a patient measurement."""

import json

import numpy as np
import pytest

from resectionlab import surgical_cusa_v3 as cusa
from resectionlab import surgical_deformation_v2 as tissue
from resectionlab import surgical_glioma_materials as materials
from resectionlab import surgical_vascular_graph as graph
from resectionlab.surgical_effects import tick_fields
from resectionlab.surgical_engine import SurgicalEngine
from resectionlab.surgical_physics_v2 import default_profile
from resectionlab.surgical_scene import SurgicalScene


def soft_patch(step=1 / 60, shear=None):
    labels = np.ones((11, 11, 11), np.uint16)
    labels[6:, :, :] = 2
    p = materials.research_profile()
    if shear:
        p['equilibriumShearPa'] = {name: [shear, shear] for name in materials.SOFT}
    physics = default_profile()
    physics['brain'].update(radiusMm=2.0, fixedStepSeconds=step)
    return SurgicalEngine(
        SurgicalScene(
            labels,
            np.eye(4),
            [
                dict(id=1, name='Brain', material='soft_tissue'),
                dict(id=2, name='Tumor', material='tumor', target=True),
            ],
            {},
            {'tissuePhysicsV2': physics, materials.PROPERTY: p},
        )
    )


def indentation(step=1 / 60, shear=None, partition=False):
    engine = soft_patch(step, shear)
    cell = int(np.ravel_multi_index((5, 5, 5), engine.scene.shape))
    tissue.begin_hold(engine, 'left', [cell], [5.0, 5.0, 5.0])
    tissue.set_target(engine, 'left', [5.0, 5.0, 5.12])
    tissue.tick(engine, 0.2)
    initial = engine.state['deformationModel']['summedHandleLoadN']
    if partition:
        for dt in (0.21, 0.17, 0.62, 1.0):
            tissue.tick(engine, dt)
    else:
        tissue.tick(engine, 2.0)
    held = engine.state['deformationModel'].copy()
    max_u = np.linalg.norm(engine.state['deformation'][str(cell)])
    tissue.release(engine, 'left')
    tissue.tick(engine, 0.4)
    after = np.linalg.norm(engine.state['deformation'][str(cell)])
    return dict(
        initialLoadN=initial,
        heldLoadN=held['summedHandleLoadN'],
        heldMm=max_u,
        recoveryMm=after,
        minVolumeRatio=held['minVolumeRatio'],
        maxVolumeRatio=held['maxVolumeRatio'],
        maxGradient=held['maxGradientNorm'],
    )


def test_indent_hold_release_timestep_sensitivity_and_property_uncertainty():
    trace = indentation()
    assert 0 < trace['heldLoadN'] < trace['initialLoadN']
    assert 0 < trace['recoveryMm'] < trace['heldMm']
    assert 0.82 <= trace['minVolumeRatio'] < 1 < trace['maxVolumeRatio'] <= 1.18
    assert trace['maxGradient'] <= 0.2
    assert indentation(partition=True) == trace
    assert indentation(1 / 120) == pytest.approx(trace, rel=0.06, abs=1e-6)
    assert indentation(shear=1100)['heldLoadN'] > indentation(shear=350)['heldLoadN']


def test_cut_and_removal_retire_connections_memory_and_collision_identity():
    engine = soft_patch()
    center = int(np.ravel_multi_index((5, 5, 5), engine.scene.shape))
    neighbor = center + 1
    tissue.begin_hold(engine, 'left', [center], [5.0, 5.0, 5.0])
    tissue.set_target(engine, 'left', [5.0, 5.0, 5.1])
    engine.state['cutEdges'] = [[int(c), neighbor] for c, _ in tissue._neighbors(engine, neighbor)]
    tissue.tick(engine, 0.3)
    assert np.linalg.norm(engine.state['deformation'][str(neighbor)]) == pytest.approx(0, abs=1e-12)
    engine.state['removed'] = [center]
    tissue.tick(engine, 0.1)
    assert str(center) not in engine.state['deformationModel']['maxwellMemoryMm']
    assert engine.occupancy()[center] == 0


def test_contact_costs_and_remaining_material_through_public_cusa_cold_replay():
    labels = np.zeros((15, 15, 15), np.uint16)
    labels[7, 7, 7] = 1
    labels[7, 7, 8] = 2
    region = [
        dict(id=1, name='Brain', material='soft_tissue'),
        dict(id=2, name='Tumor', material='tumor', target=True),
    ]
    p = {
        materials.PROPERTY: materials.research_profile(),
        'tissuePhysicsV2': default_profile(),
        'cusaV3': cusa.default_profile(),
    }
    engine = SurgicalEngine(SurgicalScene(labels, np.eye(4), region, {}, p))

    def action(op, z=None, dt=0.1, **extra):
        return {
            'slot': 'left',
            'operation': op,
            'duration': dt,
            **({'pose': {'position': [7.0, 7.0, z], 'quaternion': [0, 0, 0, 1]}} if z is not None else {}),
            **extra,
        }

    for a in [action('exchange', instrument='cusa'), action('place', 2.0)]:
        assert engine.step(a)['status'] == 'accepted'
    stroke = action('apply', 10.0, dt=0.1)
    assert engine.step(stroke)['status'] == 'partial'
    assert not engine.state['removed']
    copy = SurgicalEngine.restore(engine.scene, json.loads(json.dumps(engine.snapshot())))
    for _ in range(16):
        assert engine.step(stroke) == copy.step(stroke)
        if len(engine.state['removed']) == 2:
            break
    assert engine.metrics()['targetRemovedMm3'] == 1.0
    assert engine.metrics()['removedByLabelMm3']['Brain'] == 1.0
    m = cusa.metrics(engine)
    assert m['cusaResistanceCapacityMm3'] > 0
    assert m['cusaInputCapacityMm3'] == pytest.approx(
        sum(
            m[k]
            for k in (
                'cusaAspiratedSolidMm3',
                'cusaPendingCapacityMm3',
                'cusaUnusedCapacityMm3',
                'cusaDiscardedCapacityMm3',
                'cusaResistanceCapacityMm3',
            )
        )
    )
    assert copy.state_id == engine.state_id


def vascular_patch():
    shape = (9, 9, 9)
    labels = np.zeros(shape, np.uint16)
    artery = [(3, 4, 4), (4, 4, 4)]
    vein = [(3, 6, 4), (4, 6, 4)]
    tethers = {(3, 4, 4): (3, 4, 5), (4, 4, 4): (4, 4, 5)}
    for point in artery + vein:
        labels[point] = 2
    for point in tethers.values():
        labels[point] = 1
    p = materials.generated_vascular_properties(shape, artery, vein, tethers, 'generated-traction-coupon')
    return SurgicalEngine(
        SurgicalScene(
            labels,
            np.eye(4),
            [dict(id=1, name='Brain', material='soft_tissue'), dict(id=2, name='Vessels', material='vessel')],
            {},
            {'tissuePhysicsV2': default_profile(), **p},
        )
    )


def test_tether_geometry_traction_pressure_wound_and_source_specific_hemostasis():
    engine = vascular_patch()
    prop = engine.scene.properties[materials.PROPERTY]
    vessel, tissue = prop['vesselAttachments'][1]
    engine.state['deformation'][str(tissue)] = [0.10, 0.0, 0.0]
    materials.move_vessels(engine)
    assert engine.state['deformation'][str(vessel)] == [0.10, 0.0, 0.0]
    graph.refresh_compression(engine)
    edge = engine.state[graph.PROPERTY]['edges']['artery']
    assert edge['tractionN'] == pytest.approx(0.006)
    source = engine.scene.properties[graph.PROPERTY]['edges'][0]
    assert graph._edge_conductance(source, edge) == pytest.approx(source['conductanceMm3PerPaS'] / 1.1**3)
    assert edge['cutFace'] is None and not engine.state['sources']
    engine.state['deformation'][str(tissue)] = [0.56, 0.0, 0.0]
    materials.move_vessels(engine)
    assert engine.occupancy().material_ids([materials.outlet_cell(engine, vessel)]) == [vessel]
    graph.refresh_compression(engine)
    assert edge['cutFace'] is not None and edge['injuryCause'] == 'declared-tether-stretch'
    graph.tick(engine, 0.35)
    tick_fields(engine, 0.35)
    assert graph.metrics(engine)['generatedMm3'] > 0
    assert abs(graph.metrics(engine)['conservationResidualMm3']) < 1e-9
    ports = engine.state[graph.PROPERTY]['ports']
    graph.coagulate(engine, [edge['cutFace'][0]], 2.0, 1.0)
    assert (
        ports['vascular:artery:0']['sealFraction'] == 1.0
        and ports['vascular:artery:1']['sealFraction'] == 0.0
    )
    assert edge['cutFace'] is not None and engine.state[graph.PROPERTY]['edges']['vein']['cutFace'] is None


def test_optional_bindings_remain_disabled_for_earlier_generated_sources():
    p = materials.research_profile()
    del p['scalpSeamCell'], p['biopsyNeckFace']
    scene = SurgicalScene(
        np.zeros((3, 3, 3), np.uint16),
        np.eye(4),
        [],
        {},
        {'tissuePhysicsV2': default_profile(), materials.PROPERTY: p},
    )
    assert SurgicalEngine(scene).snapshot()['sceneId'] == scene.identity


def test_compartmental_biopsy_bleed_requires_actual_cut_and_seals_its_source_only():
    shape = (9, 9, 9)
    labels = np.zeros(shape, np.uint16)
    artery = [(2, 1, 4), (3, 1, 4)]
    vein = [(2, 7, 4), (3, 7, 4)]
    neck = [(4, 4, 4), (4, 5, 4)]
    for v in artery + vein:
        labels[v] = 2
    for t in neck:
        labels[t] = 1
    graph_props = materials.generated_vascular_properties(
        shape, artery, vein, {}, 'local-biopsy-coupon', biopsy_neck_ijk=neck
    )
    scene = SurgicalScene(
        labels,
        np.eye(4),
        [dict(id=1, name='Sample', material='tumor'), dict(id=2, name='Local vessels', material='vessel')],
        {},
        {'tissuePhysicsV2': default_profile(), **graph_props},
    )
    engine = SurgicalEngine(scene)
    description = graph.descriptor(engine)
    assert description['boundSourceCells'] == description['vesselSourceCells'] == 4
    assert description['boundMicrovascularBedSourceCells'] == 2
    assert description['edgeKinds']['microvascular-bed'] == 1

    def step(operation, point=None, duration=0.1, **extra):
        command = {'slot': 'left', 'operation': operation, 'duration': duration, **extra}
        if point is not None:
            command['pose'] = {'position': point, 'quaternion': [0.0, 0.0, 0.0, 1.0]}
        out = engine.step(command)
        assert out['status'] == 'accepted', out

    step('configure', mechanicsRevision=2, clockRevision=1)
    step('exchange', instrument='micro_scissors')
    step('place', [4, 4.5, -3], parameters={'jawOpeningMm': 0.0, 'depthMm': 1.0})
    assert not engine.state[graph.PROPERTY]['ports']
    step('apply', [4, 4.5, 3.78], duration=0.3)
    ports = engine.state[graph.PROPERTY]['ports']
    assert ports['vascular:biopsy-bed:0']['generatedMm3'] > 0
    assert ports['vascular:biopsy-bed:1']['generatedMm3'] == 0
    assert engine.state[graph.PROPERTY]['edges']['artery']['cutFace'] is None
    step('withdraw')
    step('exchange', instrument='bipolar_forceps')
    step('place', [4, 5, -3], parameters={'jawOpeningMm': 0.3, 'power': 0.0, 'forceN': 0.0, 'depthMm': 1.0})
    step('apply', [4, 5, 3.78])
    step('apply', duration=2.1, parameters={'power': 1.0})
    port = engine.state[graph.PROPERTY]['ports']['vascular:biopsy-bed:0']
    assert port['sealFraction'] == 1.0
    before = port['generatedMm3']
    step('deactivate')
    step('withdraw')
    assert engine.state[graph.PROPERTY]['ports']['vascular:biopsy-bed:0']['generatedMm3'] == before
    assert engine.state[graph.PROPERTY]['edges']['biopsy-bed']['cutFace'] == [
        port['cell'],
        int(np.ravel_multi_index(neck[0], shape)),
    ]
    assert abs(graph.metrics(engine)['conservationResidualMm3']) < 1e-9
    reopened = SurgicalEngine.restore(scene, json.loads(json.dumps(engine.snapshot())))
    assert [e['stateAfter'] for e in reopened.history] == [e['stateAfter'] for e in engine.history]


def test_supplied_intravascular_pressure_resists_same_closed_jaw_force():
    engine = vascular_patch()

    def step(op, p=None, **extra):
        action = {'slot': 'left', 'operation': op, 'duration': 0.15, **extra}
        if p is not None:
            action['pose'] = {'position': p, 'quaternion': [0, 0, 0, 1]}
        assert engine.step(action)['status'] == 'accepted'

    step('configure', mechanicsRevision=2, clockRevision=1)
    step('exchange', instrument='bipolar_forceps')
    step(
        'place', [3.5, 4, -4], parameters={'jawOpeningMm': 0.3, 'forceN': 0.02, 'power': 0.0, 'depthMm': 1.0}
    )
    step('apply', [3.5, 4, 3.8])
    s = engine.state[graph.PROPERTY]
    compression = s['edges']['artery']['compressionFraction']
    assert 0 < compression < 0.02 / engine.scene.properties[graph.PROPERTY]['compressionForceN']
    step('apply', parameters={'jawOpeningMm': 2.0})
    assert engine.state[graph.PROPERTY]['edges']['artery']['compressionFraction'] == 0.0
    assert not engine.state[graph.PROPERTY]['ports']

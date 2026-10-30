"""Generated controls for the offline kernel, not patient/device validation."""
from dataclasses import replace
from types import SimpleNamespace
import copy

import numpy as np
import pytest

from resectionlab import surgical_planar_contact as contact
from resectionlab.surgical_scene import SurgicalScene
from scripts.planar_blade_contact_fixtures import generated_fixture


def model(*, partial=True, maxwell=True, **changes):
    fixture = generated_fixture(2, partial=partial, maxwell=maxwell)
    args = dict(contact_faces=fixture.contact_faces, support_faces=fixture.support_faces,
                blade=fixture.blade, quadrature_order=2)
    args.update(changes)
    return contact.build(fixture.scene, fixture.cells, **args)


def test_candidates_are_pure_detached_and_cannot_reenable_writes():
    kernel = model(); old = kernel.initial_state()
    first = kernel.step(old, offset_mm=.002, dt_s=.1)
    second = kernel.step(old, offset_mm=.002, dt_s=.1)
    np.testing.assert_array_equal(first['state'].displacement_mm, second['state'].displacement_mm)
    assert not np.any(old.displacement_mm) and not np.any(old.memory_mm) and old.steps == 0
    assert first['state'].steps == 1
    for a in (first['state'].displacement_mm, first['state'].memory_mm,
              first['normal_forces_n'], kernel.mesh.nodes, kernel.blade.origin_mm,
              kernel.contact.data, kernel.matrices[0].data):
        with pytest.raises(ValueError): a.flags.writeable = True
    first['receipt']['reactionOnBladeN'][0] = 42
    assert second['receipt']['reactionOnBladeN'][0] == 0


@pytest.mark.parametrize('operator', ['body', 'contact'])
def test_replacing_operator_buffer_cannot_reuse_source_bound_model_id(operator):
    kernel = model(); state = kernel.initial_state()
    matrix = kernel.matrices[0] if operator == 'body' else kernel.contact
    matrix.data = matrix.data.copy() * 2
    with pytest.raises(ValueError, match='MODEL_CHANGED_REBUILD_REQUIRED'):
        kernel.step(state, offset_mm=.002, dt_s=.1)
    assert not np.any(state.displacement_mm)


@pytest.mark.parametrize('field', ['nodes', 'node_owners', 'cells'])
def test_mesh_replacement_cannot_reuse_source_bound_model_id(field):
    kernel = model(); old = kernel.initial_state()
    original = getattr(kernel.mesh, field)
    setattr(kernel.mesh, field, np.array(original, copy=True) if field == 'nodes' else list(original))
    with pytest.raises(ValueError, match='MODEL_CHANGED_REBUILD_REQUIRED'):
        kernel.step(old, offset_mm=.002, dt_s=.1)


@pytest.mark.parametrize('field', ['times', 'nodes', 'body_data', 'contact_data'])
def test_same_array_metadata_changes_cannot_reuse_source_bound_model_id(field):
    kernel = model(); old = kernel.initial_state()
    array = {'times': kernel.times, 'nodes': kernel.mesh.nodes,
             'body_data': kernel.matrices[0].data, 'contact_data': kernel.contact.data}[field]
    array.dtype = np.int64
    with pytest.raises(ValueError, match='MODEL_CHANGED_REBUILD_REQUIRED'):
        kernel.step(old, offset_mm=.002, dt_s=.1)


def test_source_or_foreign_history_changes_are_refused():
    kernel = model(); old = kernel.initial_state()
    with pytest.raises(ValueError, match='STATE_OR_SCHEDULE'):
        kernel.step(replace(old, model_id='foreign'), offset_mm=.002, dt_s=.1)
    kernel.scene.properties['brainContinuumV1']['materials']['1']['shearPa'] *= 2
    with pytest.raises(ValueError, match='Source scene changed'):
        kernel.step(old, offset_mm=.002, dt_s=.1)


def test_contact_is_not_rigid_support_for_a_touching_disconnected_body():
    f = generated_fixture(2, partial=True)
    with pytest.raises(ValueError, match='UNSUPPORTED_RIGID_COMPONENT'):
        contact.build(f.scene, [f.cells[0], f.cells[-1]],
            contact_faces=[(f.cells[-1], 2, 1)], support_faces=[(f.cells[0], 2, -1)], blade=f.blade)


def test_in_image_material_interface_is_not_an_exposed_face():
    f = generated_fixture(2)
    with pytest.raises(ValueError, match='FACE_NOT_EXPOSED'):
        contact.build(f.scene, f.cells, contact_faces=[(f.cells[0], 2, 1)],
                      support_faces=f.support_faces, blade=f.blade)


def test_cancellation_during_contact_solve_preserves_input_history():
    kernel = model(); old = kernel.initial_state(); calls = 0
    class Cancelled(Exception): pass
    def cancel():
        nonlocal calls
        calls += 1
        if calls == 4: raise Cancelled
    with pytest.raises(Cancelled): kernel.step(old, offset_mm=.002, dt_s=.1, check=cancel)
    assert calls == 4 and old.steps == 0 and not np.any(old.displacement_mm) and not np.any(old.memory_mm)
    assert kernel.step(old, offset_mm=.002, dt_s=.1)['state'].steps == 1


def test_false_optimizer_success_is_refused_by_physical_kkt(monkeypatch):
    kernel = model(); old = kernel.initial_state()
    monkeypatch.setattr(contact, 'minimize', lambda fun, x, **kw: SimpleNamespace(x=x, nit=0, status=0, nfev=1))
    with pytest.raises(ValueError, match='UNCONVERGED_KKT'):
        kernel.step(old, offset_mm=.002, dt_s=.1)
    assert not np.any(old.displacement_mm) and not np.any(old.memory_mm)


def test_incorrect_body_inverse_is_refused_before_contact(monkeypatch):
    kernel = model(); old = kernel.initial_state()
    monkeypatch.setattr(contact, 'splu', lambda *a, **kw: SimpleNamespace(solve=lambda b: np.zeros_like(b)))
    with pytest.raises(ValueError, match='BODY_OR_COMPLIANCE_RESIDUAL'):
        kernel.step(old, offset_mm=.002, dt_s=.1)
    assert not np.any(old.displacement_mm)


def test_infeasible_old_gap_and_large_gradient_refuse_without_accepting_history():
    kernel = model(); old = kernel.initial_state()
    with pytest.raises(ValueError, match='INITIAL_PENETRATION'):
        kernel.step(replace(old, offset_mm=.002), offset_mm=.003, dt_s=.1)
    with pytest.raises(ValueError, match='SMALL_GRADIENT_LIMIT'):
        kernel.step(old, offset_mm=2., dt_s=.1)
    assert old.offset_mm == 0 and not np.any(old.displacement_mm)


def test_finite_footprint_zero_tangential_force_and_unloaded_separation():
    kernel = model(maxwell=False); old = kernel.initial_state()
    first = kernel.step(old, offset_mm=.002, dt_s=.1)
    assert first['receipt']['footprintAreaMm2'] == pytest.approx(4.)
    assert first['receipt']['normalReactionN'] > 0
    assert first['receipt']['maxTangentialForceN'] == 0
    assert set(first['receipt']['contactSourceIds']) <= set(kernel.mesh.cells)
    last = kernel.step(first['state'], offset_mm=-.02, dt_s=.1)
    assert last['receipt']['normalReactionN'] == 0 and last['receipt']['contactAreaMm2'] == 0
    assert np.all(last['gaps_mm'] > 0) and np.all(last['normal_forces_n'] == 0)


def test_uniform_compression_closed_form_reaction_stored_energy_and_be_work():
    kernel = model(partial=False, maxwell=False)
    result = kernel.step(kernel.initial_state(), offset_mm=.04, dt_s=.1)
    receipt = result['receipt']
    # nu=0: E=2mu, A=16mm², H=4mm, indentation=.04mm.
    force = 2 * 1000 * 1e-6 * 16 * .04 / 4
    energy = .5 * force * .04
    assert receipt['normalReactionN'] == pytest.approx(force, rel=1e-10)
    assert receipt['storedEnergyNmm'] == pytest.approx(energy, rel=1e-10)
    assert receipt['actuatorWorkNmm'] == pytest.approx(force * .04, rel=1e-10)
    assert receipt['numericalDissipationNmm'] == pytest.approx(energy, rel=1e-10)
    assert receipt['viscousDissipationNmm'] == 0


def test_normal_schedule_is_bounded_and_model_exposes_no_tangential_motion_api():
    kernel = model(); old = kernel.initial_state()
    for offset, h in [(float('nan'), .1), (.002, 0), (.002, -.1), (True, .1)]:
        with pytest.raises(ValueError, match='STATE_OR_SCHEDULE'): kernel.step(old, offset_mm=offset, dt_s=h)
    with pytest.raises(ValueError, match='STATE_OR_SCHEDULE'):
        kernel.step(replace(old, steps=120), offset_mm=.002, dt_s=.1)
    with pytest.raises(TypeError): kernel.step(old, offset_mm=.002, dt_s=.1, tangential_mm=.1)


def test_underflowing_time_step_cannot_publish_nonfinite_energy():
    kernel = model(); old = kernel.initial_state()
    with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
        with pytest.raises(ValueError, match='ENERGY_BALANCE'):
            kernel.step(old, offset_mm=.002, dt_s=1e-320)
    assert not np.any(old.displacement_mm) and not np.any(old.memory_mm)


@pytest.mark.parametrize('limit', ['dofs', 'gradient'])
def test_stricter_source_profile_limits_are_preserved(limit):
    f = generated_fixture(2, maxwell=False)
    properties = copy.deepcopy(f.scene.properties)
    properties['brainContinuumV1']['maxDofs' if limit == 'dofs' else 'maxElementGradientNorm'] = 24 if limit == 'dofs' else .001
    scene = SurgicalScene(f.scene.labels, f.scene.affine, f.scene.regions, f.scene.source, properties)
    if limit == 'dofs':
        with pytest.raises(ValueError, match='CONTINUUM_DOF_LIMIT'):
            contact.build(scene, f.cells, contact_faces=f.contact_faces, support_faces=f.support_faces, blade=f.blade)
    else:
        kernel = contact.build(scene, f.cells, contact_faces=f.contact_faces, support_faces=f.support_faces, blade=f.blade)
        with pytest.raises(ValueError, match='SMALL_GRADIENT_LIMIT'):
            kernel.step(kernel.initial_state(), offset_mm=.04, dt_s=.1)


def test_source_changed_inside_callback_cannot_publish_a_candidate():
    kernel = model(); old = kernel.initial_state(); calls = 0
    def changed():
        nonlocal calls
        calls += 1
        if calls == 4: kernel.scene.source['changedDuringSolve'] = True
    with pytest.raises(ValueError, match='Source scene changed'):
        kernel.step(old, offset_mm=.002, dt_s=.1, check=changed)
    assert not np.any(old.displacement_mm) and not np.any(old.memory_mm)

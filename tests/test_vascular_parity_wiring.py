import importlib.util
import json
from copy import deepcopy
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

from resectionlab.patient_anatomy import PatientImagingCollection as Collection
from resectionlab.vascular_cleanup import rooted_vessel_layer
from resectionlab.vascular_parity import floating_components

SCRIPT = Path(__file__).resolve().parents[1] / 'scripts' / 'capture_vascular_parity.py'
spec = importlib.util.spec_from_file_location('capture_vascular_parity', SCRIPT)
parity = importlib.util.module_from_spec(spec)
spec.loader.exec_module(parity)

FRAME = np.diag([.5, .5, .5, 1.])
SHAPE = (40, 24, 24)


def save(path, data, frame=FRAME):
    image = nib.Nifti1Image(data, frame)
    image.header.set_xyzt_units('mm')
    nib.save(image, path)


def labels():
    named = np.zeros(SHAPE, np.uint8)
    named[2:20, 10:14, 10:14] = 4     # R-ICA, a root
    named[20:26, 10:14, 10:14] = 5    # R-M1
    named[33:38, 10:14, 10:14] = 17   # R-M2, more than 3 mm away
    named[2:30, 2:5, 2:5] = 36        # Straight sinus
    return named


def additions():
    score = np.zeros(SHAPE, np.float32)
    score[20:26, 14:20, 10:14] = 0.8  # Touches R-M1: a connected 2 mm addition.
    score[30:34, 18:22, 18:22] = 0.9  # Reaches nothing.
    return score


def ensemble(tmp_path, *, modality='ct', frame=FRAME, name='ensemble-UZH-ARG-KDH', named=None, score=None):
    """A Phase 2 ensemble folder inside a marked data root at tmp_path."""
    (tmp_path / '.ressectionlab-data-root.json').write_text('{}')
    folder = tmp_path / 'phase2' / name
    folder.mkdir(parents=True, exist_ok=True)
    named = labels() if named is None else named
    save(folder / 'named_labels.nii.gz', named, frame)
    save(folder / 'vessel_probability.nii.gz', np.where(named > 0, .9, .1).astype(np.float32), frame)
    save(folder / 'vesselfm_connected_probability.nii.gz', additions() if score is None else score, frame)
    (folder / 'ensemble.json').write_text(json.dumps({'case': 'synthetic', 'modality': modality, 'teams': ['UZH', 'ARG', 'KDH'],
                                                       'labels': list(range(41)), 'vein_labels': list(range(35, 41))}))
    return folder


def study(tmp_path, modality='CTA'):
    tmp_path.mkdir(parents=True, exist_ok=True)
    save(tmp_path / 'scan.nii.gz', np.where((labels() > 0) | (additions() > 0), 400., 40.).astype(np.float32))
    c = Collection.create(tmp_path / 'study', 'wiring')
    c.add_series(tmp_path / 'scan.nii.gz', series_id='scan', modality=modality)
    c.add_layer((labels() == 36).astype(np.uint8), FRAME, layer_id='supplied-veins', source_series='scan', kind='vein',
                role='context', color='#71b7ef', provenance={'kind': 'supplied_annotation'})
    c.add_layer((labels() == 0).astype(np.uint8), FRAME, layer_id='brain', source_series='scan', kind='brain',
                role='context', color='#d09bb6')
    c.document['display'] = {'initial_layers': ['brain', 'supplied-veins']}
    c.save()
    return c


def files(root):
    return {path.relative_to(root) for folder in ('arrays', 'revisions') if (root / folder).is_dir()
            for path in (root / folder).iterdir()}


def test_wiring_makes_the_ensemble_the_default_vessels_once_and_keeps_the_annotation(tmp_path):
    c = study(tmp_path)
    before = deepcopy(c.document['layers']), deepcopy(c.document['series'])
    folder = ensemble(tmp_path)
    receipt = parity.wire_ensemble(c, 'scan', folder)
    ids = receipt['layers']
    assert receipt['added'] == ids and len(ids) == 2 and ids[1] == ids[0] + '-vesselfm' and receipt['changed']
    reopened = Collection(c.root)
    assert reopened.document['layers'][:2] == before[0] and reopened.document['series'] == before[1]
    assert reopened.document['display']['initial_layers'] == ['brain', *ids]
    assert receipt['initial_layers'] == {'before': ['brain', 'supplied-veins'], 'after': ['brain', *ids]}
    assert receipt['replaced_initial_layers'] == ['supplied-veins'] and 'unwire' in receipt['undo']
    assert receipt['manifest_sha256']['before'] != receipt['manifest_sha256']['after']
    tree, extra = reopened.layer(ids[0]), reopened.layer(ids[1])
    assert tree['provenance']['kind'] == 'image_derived_estimate' and tree['provenance']['estimate'] == 'model'
    assert tree['provenance']['source'] == 'phase2/ensemble-UZH-ARG-KDH'  # Relative to the marked data root.
    assert tree['named_labels']['scheme'] == 'topbrain-v1-ct'
    assert np.array_equal(reopened.array(tree['named_labels']['data']), labels())
    assert extra['rooted_parent'] == ids[0] and not (reopened.mask(extra) & reopened.mask(tree)).any()
    again = parity.wire_ensemble(reopened, 'scan', folder)
    assert again['layers'] == ids and not again['added'] and not again['changed']
    assert parity.ensemble_layers(reopened, folder) == ids
    assert parity.displayed_vessels(reopened, {'vessels': ['supplied-veins']}, folder) == ids


def test_a_rewire_retires_the_previous_generation_and_a_revert_must_be_rewired(tmp_path):
    c = study(tmp_path)
    folder = ensemble(tmp_path)
    first = parity.wire_ensemble(c, 'scan', folder)
    old_arrays = parity._array_paths([c.layer(layer) for layer in first['layers']])
    changed = labels(); changed[33:38, 10:14, 10:14] = 0
    save(folder / 'named_labels.nii.gz', changed)
    with pytest.raises(LookupError, match='wire --ensemble ensemble-UZH-ARG-KDH'):
        parity.ensemble_layers(c, folder)
    second = parity.wire_ensemble(c, 'scan', folder)
    assert second['retired'] == first['layers'] and set(second['layers']).isdisjoint(first['layers'])
    reopened = Collection(c.root)
    assert [row['id'] for row in reopened.document['layers'] if row['id'].startswith('ensemble-')] == second['layers']
    assert reopened.document['display']['initial_layers'] == ['brain', *second['layers']]
    shared = parity._array_paths(reopened.document)
    assert all(not reopened.path(path).exists() for path in old_arrays - shared)
    save(folder / 'named_labels.nii.gz', labels())  # Revert the folder.
    with pytest.raises(LookupError, match='changed since it was wired'):
        parity.ensemble_layers(reopened, folder)
    third = parity.wire_ensemble(reopened, 'scan', folder)
    assert third['retired'] == second['layers'] and parity.ensemble_layers(reopened, folder) == third['layers']
    undo = parity.unwire(reopened)
    assert undo['changed'] and Collection(c.root).document['display']['initial_layers'] == ['brain', 'supplied-veins']
    assert not any(row['id'].startswith('ensemble-') for row in Collection(c.root).document['layers'])


def test_wiring_refuses_the_other_modality_another_grid_and_an_empty_map(tmp_path):
    c = study(tmp_path)
    with pytest.raises(ValueError, match='cannot be wired'):
        parity.wire_ensemble(c, 'scan', ensemble(tmp_path, modality='mr'))
    with pytest.raises(ValueError, match='not on the scan'):
        parity.wire_ensemble(c, 'scan', ensemble(tmp_path, frame=np.diag([.6, .5, .5, 1.]), name='shifted'))
    with pytest.raises(ValueError, match='empty named vessel map'):
        parity.wire_ensemble(c, 'scan', ensemble(tmp_path, name='empty', named=np.zeros(SHAPE, np.uint8)))
    mr = study(tmp_path / 'mr', modality='TOF')
    with pytest.raises(ValueError, match='cannot be wired'):
        parity.wire_ensemble(mr, 'scan', ensemble(tmp_path, name='ct-on-mr'))
    assert len(Collection(c.root).document['layers']) == 2


def test_a_failed_wiring_leaves_no_trace(tmp_path, monkeypatch):
    c = study(tmp_path)
    manifest = (c.root / 'patient-anatomy.json').read_bytes()
    stored = files(c.root)
    monkeypatch.setattr(parity, '_publish', lambda *args: (_ for _ in ()).throw(OSError('disk full')))
    with pytest.raises(OSError):
        parity.wire_ensemble(c, 'scan', ensemble(tmp_path))
    assert (c.root / 'patient-anatomy.json').read_bytes() == manifest and files(c.root) == stored
    assert len(c.document['layers']) == 2


def test_wiring_refuses_when_saving_would_change_anything_else(tmp_path, monkeypatch):
    import resectionlab.anatomy_display as display
    c = study(tmp_path)
    manifest = (c.root / 'patient-anatomy.json').read_bytes()
    stored = files(c.root)
    monkeypatch.setattr(display, 'retire_stale_cta_coverage', lambda collection: collection.document['layers'].pop())
    with pytest.raises(ValueError, match='retire stale CTA coverage'):
        parity.wire_ensemble(c, 'scan', ensemble(tmp_path))
    assert (c.root / 'patient-anatomy.json').read_bytes() == manifest and files(c.root) == stored


def test_wiring_a_linked_review_manifest_writes_only_that_document(tmp_path):
    from resectionlab.anatomy_bridge import open_anatomy_manifest
    from resectionlab.data_root import sha256_file
    c = study(tmp_path)
    source = c.root / 'patient-anatomy.json'
    document = deepcopy(c.document)
    document['vascular_review'] = {'schema': 'eden2020-restricted-vascular-companion-v1', 'linked_source': True,
                                   'source_manifest': {'sha256': sha256_file(source), 'snapshot_id': c.identity()}}
    document.pop('snapshot_id')
    document['snapshot_id'] = Collection(c.root, document).identity()
    linked = c.root / 'vascular-review.anatomy.json'
    linked.write_text(json.dumps(document))
    review = open_anatomy_manifest(linked)
    before = source.read_bytes()
    receipt = parity.wire_ensemble(review, 'scan', ensemble(tmp_path), manifest=linked)
    assert source.read_bytes() == before
    reopened = open_anatomy_manifest(linked)
    assert reopened.document['display']['initial_layers'] == ['brain', *receipt['layers']]
    assert reopened.layer(receipt['layers'][0])['named_labels']['scheme'] == 'topbrain-v1-ct'


def test_wired_layers_show_a_rooted_tree_and_the_addition_only_where_it_joins(tmp_path):
    c = study(tmp_path)
    ids = parity.wire_ensemble(c, 'scan', ensemble(tmp_path))['layers']
    display, receipt, _, split = rooted_vessel_layer(c, c.layer(ids[0]))
    assert np.array_equal(c.mask(display), np.isin(labels(), (4, 5)))
    assert np.array_equal(c.mask(split), labels() == 36) and split['kind'] == 'vein'
    shown, child, _, _ = rooted_vessel_layer(c, c.layer(ids[1]))
    near = np.zeros(SHAPE, bool); near[:30] = True
    # The shown branch reaches one voxel into R-M1 (display-only overlap), so its surface joins the tree's.
    overlap = np.zeros(SHAPE, bool); overlap[20:26, 13, 10:14] = True
    assert np.array_equal(c.mask(shown), ((additions() > 0) & near) | overlap)
    assert child['rooting']['display']['overlapVoxels'] == int(overlap.sum())
    # The addition's own receipt: its own wording and counts, not the tree's.
    assert child['reason'].startswith('Attached to the named tree of Model vessels') and '1 detached components' in child['reason']
    assert child['rooting']['input']['voxels'] == int((additions() > 0).sum())
    assert child['rooting']['input']['components'] == 2 and child['rooting']['display']['components'] == 1
    union, receipts = parity.rooted_display(c, ids)
    roots = parity.display_roots(c, ids, SHAPE)
    assert floating_components(union, roots & union, (.5, .5, .5))['floating'] == 0


def test_an_addition_joining_a_vein_is_kept_as_a_vein_part(tmp_path):
    score = np.zeros(SHAPE, np.float32)
    score[10:20, 5:9, 2:5] = 0.8  # Touches only the straight sinus.
    c = study(tmp_path)
    ids = parity.wire_ensemble(c, 'scan', ensemble(tmp_path, score=score))['layers']
    display, receipt, _, split = rooted_vessel_layer(c, c.layer(ids[1]))
    assert not c.mask(display).any() and split['kind'] == 'vein' and split['color'] == '#aa352b'
    overlap = np.zeros(SHAPE, bool); overlap[10:20, 4, 2:5] = True  # Straight-sinus voxels it enters, drawn as vein.
    assert np.array_equal(c.mask(split), (score > 0) | overlap) and receipt['rooting']['classes']['vein'] == int((score > 0).sum())


@pytest.mark.parametrize('depth, branch', [(1, False), (2, False), (3, True)])
def test_an_addition_that_only_thickens_the_tree_wall_is_kept_for_review(tmp_path, depth, branch):
    score = np.zeros(SHAPE, np.float32)
    score[4:14, 14:14 + depth, 10:14] = 0.8  # Lies on the R-ICA wall, depth voxels thick.
    score[20:26, 14:20, 10:14] = 0.8         # A branch off R-M1.
    c = study(tmp_path)
    ids = parity.wire_ensemble(c, 'scan', ensemble(tmp_path, score=score))['layers']
    display, receipt, hidden, split = rooted_vessel_layer(c, c.layer(ids[1]), review=True)
    tree = np.isin(labels(), (4, 5, 36))
    shown, review, own = c.mask(display), c.mask(hidden) if hidden else np.zeros(SHAPE, bool), score > 0
    crust = np.zeros(SHAPE, bool); crust[4:14, 14:14 + depth, 10:14] = True
    overlap = np.zeros(SHAPE, bool); overlap[20:26, 13, 10:14] = True
    if branch:
        overlap[4:14, 13, 10:14] = True
    assert np.array_equal(shown & tree, overlap) and not (review & tree).any() and split is None
    assert np.array_equal(shown & ~tree, own if branch else own & ~crust)
    assert np.array_equal(review, np.zeros(SHAPE, bool) if branch else crust)
    rooting = receipt['rooting']
    wall = rooting['review']['wallThickening']
    assert wall['components'] == (0 if branch else 1) and wall['voxels'] == (0 if branch else int(crust.sum()))
    assert wall['volumeMm3'] == wall['voxels'] * .125 and 'hugs the tree wall' in wall['reason'].lower()
    assert rooting['review']['components'] == wall['components'] and rooting['review']['detachedComponents'] == 0
    # The overlap is display-only: the partition of the addition's own voxels is unchanged.
    assert rooting['display']['overlapVoxels'] == int(overlap.sum())
    assert rooting['display']['sourceVoxels'] + rooting['review']['voxels'] + rooting['lod']['voxels'] == int(own.sum())
    assert receipt['hiddenVoxels'] == int(review.sum()) and receipt['excludedFragments'] == wall['components']
    assert ('1 wall-thickening pieces that hug the tree wall' in receipt['reason']) is not branch


@pytest.mark.parametrize('case', ['missing', 'grid', 'failed', 'empty'])
def test_an_addition_without_a_usable_tree_shows_nothing(tmp_path, monkeypatch, case):
    import resectionlab.vascular_cleanup as cleanup
    c = study(tmp_path)
    ids = parity.wire_ensemble(c, 'scan', ensemble(tmp_path))['layers']
    child = deepcopy(c.layer(ids[1]))
    if case == 'missing':
        child['rooted_parent'] = 'retired-layer'
    elif case == 'grid':
        c.layer(ids[0])['affine'] = np.diag([.6, .5, .5, 1.]).tolist()
    elif case == 'failed':
        c.layer(ids[0])['named_labels']['data'] = c.put_array(np.full(SHAPE, 99, np.uint8))
    else:
        c.layer(ids[0])['data'] = c.put_array(np.zeros(SHAPE, np.uint8))
        c.layer(ids[0])['named_labels']['data'] = c.put_array(np.zeros(SHAPE, np.uint8))
    display, receipt, hidden, split = rooted_vessel_layer(c, child, review=True)
    assert not c.mask(display).any() and split is None
    assert np.array_equal(c.mask(hidden), additions() > 0) and receipt['reason'].startswith('Not shown:')


def test_named_review_measures_major_vessels_against_the_reference(tmp_path):
    c = study(tmp_path)
    ids = parity.wire_ensemble(c, 'scan', ensemble(tmp_path))['layers']
    union, receipts = parity.rooted_display(c, ids)
    none = parity.named_review({'x': {'rooting': {'roots': {'scheme': None}, 'namedLabels': []}}})
    assert none['gate'] == 'not-measured'
    gone = parity.named_review({'x': {'namedLabelsUnavailable': 'The supplied label original is missing',
                                      'rooting': {'roots': {'scheme': None}, 'namedLabels': []}}})
    assert gone['gate'] == 'not-measured' and 'missing' in gone['reason']
    prediction = parity.named_review(receipts)
    assert prediction['basis'] == 'prediction-side status only' and 'not available' in prediction['referenceCheck']
    assert [row['name'] for row in prediction['partlyInReview'] + prediction['lost']] == ['R-M2']  # M2 is major.
    reference = labels().astype(np.int16)
    reference[2:30, 18:20, 18:20] = 6  # A left ICA the model never predicted.
    review = parity.named_review(receipts, display=union, reference=reference, spacing=(.5, .5, .5))
    assert {row['name'] for row in review['lost']} == {'L-ICA', 'R-M2'} and review['gate'] == 'not met'
    assert next(row for row in review['referenceCheck'] if row['name'] == 'R-ICA')['covered'] == 1.0
    assert 'A1-A2' in review['mergedClasses']


def test_the_mesh_gate_counts_surface_pieces_and_can_fail(tmp_path):
    from resectionlab.anatomy_geometry import mesh
    c = study(tmp_path)
    roots = np.zeros(SHAPE, bool); roots[4:8, 10:14, 10:14] = True
    joined = np.zeros(SHAPE, np.uint8); joined[4:12, 10:14, 10:14] = 1
    edge = joined.copy(); edge[12:16, 14:18, 14:18] = 1  # Touches only at a corner.
    rows = {}
    for name, data in (('joined', joined), ('edge', edge)):
        rows[name] = c.add_layer(data, FRAME, layer_id=name, source_series='scan', kind='artery', role='context',
                                 color='#aa352b')
    ok = parity.mesh_floating([('joined', mesh(c, rows['joined']))], roots, SHAPE)
    assert ok == {**ok, 'pieces': 1, 'floating': 0}
    bad = parity.mesh_floating([('edge', mesh(c, rows['edge']))], roots, SHAPE)
    assert bad['pieces'] == 2 and bad['floating'] == 1


def test_the_mesh_gate_joins_a_branch_that_enters_the_tree_but_not_one_that_only_abuts_it(tmp_path):
    from resectionlab.anatomy_geometry import mesh
    c = study(tmp_path)
    roots = np.zeros(SHAPE, bool); roots[4:6, 8:14, 8:14] = True
    tree = np.zeros(SHAPE, np.uint8); tree[4:20, 8:14, 8:14] = 1
    abut = np.zeros(SHAPE, np.uint8); abut[10:14, 14:20, 9:13] = 1
    enter = np.zeros(SHAPE, np.uint8); enter[10:14, 13:20, 9:13] = 1  # One voxel into the tree.
    # Score fields put the abutting walls at different positions, as on the real models.
    rows = {name: c.add_layer(data, FRAME, layer_id=name, source_series='scan', kind='artery', role='context',
                              color='#aa352b', probability=np.where(data > 0, level, 0.).astype(np.float32))
            for name, data, level in (('tree', tree, .9), ('abut', abut, .7))}
    rows['enter'] = c.add_layer(enter, FRAME, layer_id='enter', source_series='scan', kind='artery', role='context',
                                color='#aa352b')
    surfaces = {name: mesh(c, row) for name, row in rows.items()}
    for child, pieces, floating in (('abut', 2, 1), ('enter', 1, 0)):
        gate = parity.mesh_floating([('tree', surfaces['tree']), (child, surfaces[child])], roots, SHAPE,
                                    [c.mask(rows['tree']), c.mask(rows[child])])
        assert gate['pieces'] == pieces and gate['floating'] == floating, child
    # Without the displayed masks the entering branch is a separate piece too.
    alone = parity.mesh_floating([('tree', surfaces['tree']), ('enter', surfaces['enter'])], roots, SHAPE)
    assert alone['floating'] == 1


def test_the_mesh_gate_joins_a_tube_that_enters_another_tube_and_not_one_that_stops_short(tmp_path):
    from resectionlab.anatomy_geometry import mesh
    from resectionlab.vessel_display_tubes import DISPLAY_TUBES, thin_tubes
    c = study(tmp_path)
    roots = np.zeros(SHAPE, bool); roots[4:6, 12, 12] = True
    parent = np.zeros(SHAPE, bool); parent[4:36, 12, 12] = True      # One voxel wide.
    enters = np.zeros(SHAPE, bool); enters[20, 13:22, 12] = True     # Starts beside the parent.
    short = np.zeros(SHAPE, bool); short[20, 17:22, 12] = True       # Starts two millimetres away.
    rows = {}
    for name, data, beside in (('parent', parent, None), ('enters', enters, parent), ('short', short, parent)):
        row = c.add_layer(data.astype(np.uint8), FRAME, layer_id=name, source_series='scan', kind='artery', role='context',
                          color='#aa352b')
        packed, _ = thin_tubes(data, (.5, .5, .5), context=beside)
        rows[name] = {**row, DISPLAY_TUBES: c.put_array(packed)}
    surfaces = {name: mesh(c, row) for name, row in rows.items()}
    assert all(str(surface['surface_field']).startswith('thin-tubes+') for surface in surfaces.values())
    for child, floating in (('enters', 0), ('short', 1)):
        gate = parity.mesh_floating([('parent', surfaces['parent']), (child, surfaces[child])], roots, SHAPE,
                                    [parity.drawn_solid(c, rows['parent']), parity.drawn_solid(c, rows[child])])
        assert gate['floating'] == floating and gate['pieces'] == 1 + floating, child
    # A part without tubes is still judged by its voxels.
    assert parity.drawn_solid(c, c.layer('parent')).dtype == bool


def test_capture_recipes_ask_for_the_rooted_view(tmp_path):
    c = study(tmp_path)
    case = {'series': 'scan', 'anchor': [5., 5., 5.], 'vessels': ['supplied-veins']}
    rows = parity.recipes(c, case, {'settings': {}}, ['supplied-veins'])
    assert {row['scope'] for row in rows} == {'rooted'}
    assert {row['scope'] for row in parity.recipes(c, case, {'settings': {}}, ['supplied-veins'], scope='all')} == {'all'}


def test_bone_framings_are_captured_only_on_ct_in_the_path_traced_mode():
    bone = parity.PRESETS['left-lateral-solid']
    assert bone['skull'] == 'exterior' and parity.PRESETS['left-lateral-sagittal-cut']['clip'][0] == 'x'
    assert parity.preset_applies(bone, 'cinematic', 'CTA') and parity.preset_applies(bone, 'cinematic', 'CT')
    assert not parity.preset_applies(bone, None, 'CTA') and not parity.preset_applies(bone, 'cinematic', 'TOF')
    assert all(parity.preset_applies(parity.PRESETS[name], None, 'TOF') for name in ('anterior', 'skull-base', 'anterior-coronal-cut'))
    assert all(parity.REFERENCE_FRAME[name] in parity.PRESETS for name in parity.PRESETS if parity.PRESETS[name].get('bone'))


def test_recipes_for_named_presets_leave_out_the_timed_rows_and_inapplicable_framings(tmp_path):
    c = study(tmp_path)
    case = {'series': 'scan', 'anchor': [5., 5., 5.], 'vessels': ['supplied-veins']}
    every = [row['name'] for row in parity.recipes(c, case, {'settings': {}}, ['supplied-veins'])]
    assert every[-2:] == ['interaction', 'orbit'] and 'left-lateral-solid' not in every
    only = parity.recipes(c, case, {'settings': {}}, ['supplied-veins'], only=['skull-base', 'oblique-solid'])
    assert [row['name'] for row in only] == ['skull-base']
    assert only[0]['patientCutaway']['clipping']['axis'] == 'z'


def test_requested_presets_refuse_unknown_names_and_ignore_spacing():
    assert parity.requested_presets('oblique-solid, left-lateral-solid') == ['oblique-solid', 'left-lateral-solid']
    with pytest.raises(ValueError, match='Unknown presets'):
        parity.requested_presets('oblique-solid,lateral')
    with pytest.raises(ValueError):
        parity.requested_presets(' , ')


def test_the_mesh_gate_reports_inner_walls_apart_from_floating_pieces(tmp_path):
    from resectionlab.anatomy_geometry import mesh
    c = study(tmp_path)
    hollow = np.zeros(SHAPE, np.uint8); hollow[4:20, 8:16, 8:16] = 1; hollow[8:16, 11:13, 11:13] = 0
    row = c.add_layer(hollow, FRAME, layer_id='hollow', source_series='scan', kind='artery', role='context', color='#aa352b')
    roots = np.zeros(SHAPE, bool); roots[4:6, 8:16, 8:16] = True
    gate = parity.mesh_floating([('hollow', mesh(c, row))], roots, SHAPE)
    assert gate['pieces'] == 2 and gate['innerWalls'] == 1 and gate['floating'] == 0


def test_candidates_and_fine_branches_are_wired_as_a_chain_and_the_major_detail_hides_the_fine_layer(tmp_path):
    from resectionlab.vascular_cleanup import vessel_detail
    folder = ensemble(tmp_path)
    save(folder / 'vesselfm_candidate_probability.nii.gz', additions())  # One connected piece, one that reaches nothing.
    fine = np.zeros(SHAPE, np.float32)
    fine[22:24, 20:23, 11:13] = 0.7  # Continues the connected candidate, three voxels beyond it.
    fine[2:6, 20:23, 20:23] = 0.7    # Reaches nothing.
    save(folder / 'fine_branch_score.nii.gz', fine)
    c = study(tmp_path)
    ids = parity.wire_ensemble(c, 'scan', folder)['layers']
    assert [layer.rsplit('-', 1)[-1] for layer in ids[1:]] == ['vesselfm', 'fine']
    extra, branch = c.layer(ids[1]), c.layer(ids[2])
    assert extra['rooted_parent'] == ids[0] and branch['rooted_parent'] == ids[1]
    assert extra['provenance']['model'] == 'vesselfm-candidate-addition'
    assert branch['provenance']['model'] == 'signal-grown-fine-branches'
    assert c.document['display']['initial_layers'][-3:] == ids

    # The candidates: the connected piece is shown, the other waits in review. Nothing is lost.
    shown, receipt, review, _ = rooted_vessel_layer(c, extra, review=True)
    assert c.mask(shown)[20:26, 15:20, 10:14].all() and not c.mask(shown)[30:34, 18:22, 18:22].any()
    assert c.mask(review)[30:34, 18:22, 18:22].all()

    # CT shows fine branches by default, rooted on the candidates' shown display.
    assert vessel_detail('CTA') == 'fine' and vessel_detail('TOF') == 'major' and vessel_detail('TOF', 'fine') == 'fine'
    shown, receipt, review, _ = rooted_vessel_layer(c, branch, review=True)
    assert receipt['detail'] == 'fine' and receipt['rooting']['rootedOn']['layer'] == ids[1]
    assert c.mask(shown)[22:24, 20:23, 11:13].all() and not c.mask(shown)[2:6, 20:23, 20:23].any()
    assert c.mask(review)[2:6, 20:23, 20:23].all()

    # The major detail hides the whole fine layer and keeps it under review; other layers are unchanged.
    hidden, receipt, review, _ = rooted_vessel_layer(c, branch, review=True, detail='major')
    assert not c.mask(hidden).any() and 'fine vessel detail' in receipt['detailHidden'] and receipt['detail'] == 'major'
    assert int(c.mask(review).sum()) == int((fine > 0).sum())
    major, _, _, _ = rooted_vessel_layer(c, extra, detail='major')
    assert c.mask(major)[20:26, 15:20, 10:14].all()
    with pytest.raises(ValueError, match='detail'):
        rooted_vessel_layer(c, branch, detail='everything')


def test_a_chain_of_additions_that_loops_or_loses_its_tree_shows_nothing(tmp_path):
    folder = ensemble(tmp_path)
    save(folder / 'vesselfm_candidate_probability.nii.gz', additions())
    fine = np.zeros(SHAPE, np.float32)
    fine[22:24, 20:23, 11:13] = 0.7
    save(folder / 'fine_branch_score.nii.gz', fine)
    c = study(tmp_path)
    ids = parity.wire_ensemble(c, 'scan', folder)['layers']
    c.layer(ids[1])['rooted_parent'] = ids[2]  # The candidates and the fine branches now extend each other.
    for layer in ids[1:]:
        display, receipt, _, _ = rooted_vessel_layer(c, c.layer(layer))
        assert not c.mask(display).any() and receipt['parentUnavailable']
    c.layer(ids[1])['rooted_parent'] = 'retired-layer'
    display, receipt, _, _ = rooted_vessel_layer(c, c.layer(ids[2]))
    assert not c.mask(display).any() and 'retired-layer' in receipt['parentUnavailable']


def test_the_major_detail_hides_a_thin_end_branch_that_the_fine_detail_shows(tmp_path):
    score = additions()
    score[22, 16, 14:23] = 0.8  # One voxel (0.5 mm) wide, leaving the connected piece.
    folder = ensemble(tmp_path)
    save(folder / 'vesselfm_candidate_probability.nii.gz', score)
    c = study(tmp_path)
    extra = c.layer(parity.wire_ensemble(c, 'scan', folder)['layers'][1])
    tip = (22, 16, slice(19, 23))
    fine, receipt, _, _ = rooted_vessel_layer(c, extra, detail='fine')
    assert c.mask(fine)[tip].all() and receipt['rooting']['lod']['voxels'] == 0
    assert receipt['rooting']['policy']['always_mm'] == 0.5
    major, receipt, review, _ = rooted_vessel_layer(c, extra, detail='major', review=True)
    assert not c.mask(major)[tip].any() and c.mask(review)[tip].all()
    assert receipt['rooting']['lod']['voxels'] >= 4 and receipt['rooting']['policy']['always_mm'] == 1.0
    assert c.mask(major)[20:26, 15:20, 10:14].all()


def test_a_thin_branch_is_drawn_as_a_tube_in_the_surface_and_the_volume_tags_while_its_mask_stays(tmp_path):
    from resectionlab.anatomy_bridge import _tube_tags, open_anatomy_manifest
    from resectionlab.anatomy_geometry import mesh
    from resectionlab.vessel_display_tubes import DISPLAY_TUBES, unpack
    score = additions()
    score[22, 16, 14:23] = 0.8  # One voxel (0.5 mm) wide, leaving the connected piece.
    folder = ensemble(tmp_path)
    save(folder / 'vesselfm_candidate_probability.nii.gz', score)
    c = study(tmp_path)
    ids = parity.wire_ensemble(c, 'scan', folder)['layers']
    extra = c.layer(ids[1])
    tip = (22, 16, slice(17, 23))
    smooth, receipt, _, _ = rooted_vessel_layer(c, extra)
    plain, plain_receipt, _, _ = rooted_vessel_layer(c, extra, smooth=False)
    # The same voxels either way: tubes change the drawing, not the display mask.
    assert np.array_equal(c.mask(smooth), c.mask(plain)) and c.mask(smooth)[tip].all()
    assert DISPLAY_TUBES not in plain and 'tubes' not in plain_receipt['rooting']
    tubes = receipt['rooting']['tubes']['own']
    assert tubes['replacedVoxels'] >= 6 and tubes['tubeLengthMm'] > 2
    field, replaced = unpack(c.array(smooth[DISPLAY_TUBES]))
    assert replaced[tip].all() and (field[tip] >= .5).all()
    assert not replaced[20:26, 15:19, 10:14].any(), 'the 2 mm piece keeps its voxels'
    # The thick tree alone has nothing thin, so it carries no tube field.
    tree, tree_receipt, _, _ = rooted_vessel_layer(c, c.layer(ids[0]))
    assert DISPLAY_TUBES not in tree and tree_receipt['rooting']['tubes']['own']['replacedVoxels'] == 0
    # Surface: one contour of the tube field joined to the remaining voxels' own field.
    drawn, stepped = mesh(c, smooth), mesh(c, plain)
    assert str(drawn['surface_field']).startswith('thin-tubes+') and str(stepped['surface_field']) == 'categorical'
    assert len(drawn['indices']) and not np.array_equal(drawn['source'], stepped['source'])
    # The raw contour and the faired surface the app shows: a round wall about one voxel out, not a square prism.
    for points in (drawn['source'], drawn['positions']):
        world = points[(points[:, 2] > 18) & (points[:, 2] < 21)]
        radial = np.hypot(world[:, 0] - 22, world[:, 1] - 16)
        assert len(radial) >= 12 and radial.std() < .12 and .8 < radial.mean() < 1.2
    # The thick piece is contoured from its softened mask now; it stays within a voxel of the stepped surface.
    from scipy.spatial import cKDTree
    piece = drawn['positions'][(drawn['positions'][:, 1] < 15.5) & (drawn['positions'][:, 2] < 13.5)]
    assert len(piece) > 50 and cKDTree(stepped['positions']).query(piece)[0].max() < 1.
    # A display that keeps every source voxel still draws its tubes (the mesh takes the source row's field).
    source = {key: value for key, value in smooth.items() if key != DISPLAY_TUBES} | {'edits': []}
    same = mesh(c, smooth, display_source=source)
    assert str(same['surface_field']).startswith('thin-tubes+')
    # A failure while drawing tubes costs only the tubes.
    import resectionlab.vessel_display_tubes as tube_module
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(tube_module, 'thin_tubes', lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError('no skeleton')))
        fresh = open_anatomy_manifest(c.root / 'patient-anatomy.json')
        for stored in (c.root / 'display-cache').glob('rooted-vessel-display-cache-*.json'):
            stored.unlink()
        fallback, failed, _, _ = rooted_vessel_layer(fresh, fresh.layer(ids[1]))
        assert DISPLAY_TUBES not in fallback and np.array_equal(fresh.mask(fallback), c.mask(smooth))
        assert 'no skeleton' in failed['rooting']['tubes']['own']['failed']
        for stored in (c.root / 'display-cache').glob('rooted-vessel-display-cache-*.json'):
            stored.unlink()
    # Volume tags on the displayed grid.
    tags = c.array(_tube_tags(c, smooth, c.mask(smooth), FRAME, SHAPE, FRAME, lambda data, dtype: c.put_array(data.astype(dtype)))['tags'])
    assert tags.dtype == np.uint8 and tags.shape == SHAPE and (tags[tip] >= 128).all() and tags[5, 20, 20] == 0
    assert _tube_tags(c, plain, c.mask(plain), FRAME, SHAPE, FRAME, None) == {}
    # Reopening reads the stored display, tubes included; a missing tube array prepares it again.
    rooted_vessel_layer(c, extra)
    reopened = open_anatomy_manifest(c.root / 'patient-anatomy.json')
    again, same_receipt, _, _ = rooted_vessel_layer(reopened, reopened.layer(ids[1]))
    assert again[DISPLAY_TUBES] == smooth[DISPLAY_TUBES] and same_receipt == receipt
    (c.root / smooth[DISPLAY_TUBES]['path']).unlink()
    rebuilt = open_anatomy_manifest(c.root / 'patient-anatomy.json')
    restored, _, _, _ = rooted_vessel_layer(rebuilt, rebuilt.layer(ids[1]))
    assert restored[DISPLAY_TUBES] == smooth[DISPLAY_TUBES] and (c.root / smooth[DISPLAY_TUBES]['path']).is_file()


def test_a_fourth_layer_in_a_chain_shows_nothing(tmp_path):
    folder = ensemble(tmp_path)
    save(folder / 'vesselfm_candidate_probability.nii.gz', additions())
    fine = np.zeros(SHAPE, np.float32)
    fine[22:24, 20:23, 11:13] = 0.7
    save(folder / 'fine_branch_score.nii.gz', fine)
    c = study(tmp_path)
    ids = parity.wire_ensemble(c, 'scan', folder)['layers']
    more = np.zeros(SHAPE, np.uint8)
    more[22:24, 20:23, 13:17] = 1
    fourth = c.add_layer(more, FRAME, layer_id='fourth', source_series='scan', kind='vessels', role='context', color='#aa352b')
    fourth['rooted_parent'] = ids[2]
    display, receipt, _, _ = rooted_vessel_layer(c, fourth)
    assert not c.mask(display).any() and 'longer than 3 layers' in receipt['parentUnavailable']
    shown, _, _, _ = rooted_vessel_layer(c, c.layer(ids[2]))
    assert c.mask(shown).any()


def test_the_connected_display_is_kept_beside_the_study_and_reused_on_reopen(tmp_path, monkeypatch):
    import resectionlab.vascular_rooting as rooting
    from resectionlab.anatomy_bridge import open_anatomy_manifest
    c = study(tmp_path)
    tree = parity.wire_ensemble(c, 'scan', ensemble(tmp_path))['layers'][0]
    first, receipt, _, _ = rooted_vessel_layer(c, c.layer(tree))
    stored = list((c.root / 'display-cache').glob('rooted-vessel-display-cache-*.json'))
    assert len(stored) == 1
    calls = []
    original = rooting.root_vessels
    monkeypatch.setattr(rooting, 'root_vessels', lambda *args, **kwargs: calls.append(1) or original(*args, **kwargs))
    reopened = open_anatomy_manifest(c.root / 'patient-anatomy.json')
    again, same, _, _ = rooted_vessel_layer(reopened, reopened.layer(tree))
    assert not calls and again['data'] == first['data'] and same == receipt
    # The other detail is its own display; a damaged file is prepared again and replaced.
    rooted_vessel_layer(reopened, reopened.layer(tree), detail='major')
    assert len(calls) == 1 and len(list((c.root / 'display-cache').glob('rooted-vessel-display-cache-*.json'))) == 2
    stored[0].write_text('{"identity": 1')
    fresh = open_anatomy_manifest(c.root / 'patient-anatomy.json')
    repaired, _, _, _ = rooted_vessel_layer(fresh, fresh.layer(tree))
    assert len(calls) == 2 and repaired['data'] == first['data'] and json.loads(stored[0].read_text())['result']


def test_any_prepared_study_becomes_a_capture_case_with_an_anchor_from_its_wired_tree(tmp_path):
    c = study(tmp_path)
    built = ensemble(tmp_path)
    folder = tmp_path / parity.STUDY_STORE / 'study-ct' / built.name
    folder.parent.mkdir(parents=True)
    built.rename(folder)
    manifest = c.root / 'patient-anatomy.json'
    name, case = parity.study_entry(tmp_path, manifest)
    assert name == 'study-ct' and 'anchor' not in case and case['series'] == 'scan' and case['held_out']
    assert case['manifest'] == 'study/patient-anatomy.json' and case['reference'] is None
    assert parity.ensemble_folder(case, built.name) == parity.STUDY_STORE / 'study-ct' / built.name
    with pytest.raises(LookupError, match='No layers wired'):
        parity.study_entry(tmp_path, manifest, ensemble=built.name)
    ids = parity.wire_ensemble(c, 'scan', folder)['layers']
    name, case = parity.study_entry(tmp_path, 'study/patient-anatomy.json', ensemble=built.name)
    # R-ICA voxels 2..19, 10..13, 10..13 at 0.5 mm.
    assert case['anchor'] == pytest.approx(np.array([5.25, 5.75, 5.75]) + parity.ANCHOR_SHIFT, abs=.011)
    assert parity.displayed_vessels(Collection(c.root), case, built.name, tmp_path) == ids
    assert all(row['seriesId'] == 'scan' for row in parity.recipes(Collection(c.root), case, {'settings': {}}, ids, None, 'rooted'))
    with pytest.raises(SystemExit, match='No study document'):
        parity.study_entry(tmp_path, 'elsewhere/patient-anatomy.json')
    assert case['cohort_view'] is False and parity.study_entry(tmp_path, manifest, cohort_view=True)[1]['cohort_view'] is True
    # A study folder named like a fixed case is captured as that case, not as a second one of the same name.
    fixed = tmp_path / 'TopCoW-014'
    fixed.mkdir()
    (fixed / 'patient-anatomy.json').write_text(manifest.read_text())
    with pytest.raises(SystemExit, match='fixed case'):
        parity.study_entry(tmp_path, fixed / 'patient-anatomy.json')


def test_a_study_without_a_named_carotid_is_centred_on_its_whole_tree(tmp_path):
    c = study(tmp_path)
    named = np.where(labels() == 4, 1, labels()).astype(np.uint8)     # the carotid named as the basilar: still a root
    folder = ensemble(tmp_path, named=named)
    wired = parity.wire_ensemble(c, 'scan', folder)['layers'][0]
    tree = Collection(c.root).layer(wired)
    centre = np.argwhere(named > 0).mean(0) * .5
    assert parity.study_anchor(Collection(c.root), 'scan', tree) == pytest.approx(centre, abs=.011)



def test_an_ensemble_made_from_another_file_is_not_wired(tmp_path):
    c = study(tmp_path)
    folder = ensemble(tmp_path)
    original = c.series('scan')['original']['sha256']
    record = folder / 'ensemble.json'
    meta = json.loads(record.read_text())
    record.write_text(json.dumps({**meta, 'sha256': '0' * 64}))
    before = (c.root / 'patient-anatomy.json').read_bytes()
    with pytest.raises(ValueError, match='made from another file'):
        parity.wire_ensemble(c, 'scan', folder)
    assert (c.root / 'patient-anatomy.json').read_bytes() == before
    record.write_text(json.dumps({**meta, 'sha256': original}))
    assert parity.wire_ensemble(Collection(c.root), 'scan', folder)['changed']

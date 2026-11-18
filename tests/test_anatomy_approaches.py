"""Native planning coordinates, whole-envelope coverage and durable revisions."""
import numpy as np
import nibabel as nib
import pytest
from resectionlab.patient_anatomy import PatientImagingCollection as Collection, PatientAnatomySnapshot as Snapshot, world
from resectionlab.anatomy_approaches import NativeRouteGeometry, envelope_coverage, dispatch_plan, _segment_bundle_distances, read_approaches
from resectionlab.anatomy_planning import route_clearance
from resectionlab.anatomy_package import save_package, open_package


@pytest.fixture
def native(tmp_path):
    a=np.eye(4);a[:3,:3]=[[-1,0,0],[0,0,-1.2],[0,1.1,0]];a[:3,3]=[20,0,0]
    image=nib.Nifti1Image(np.ones((32,32,32),np.float32),a);image.header.set_xyzt_units('mm')
    p=tmp_path/'source.nii.gz';nib.save(image,p)
    c=Collection.create(tmp_path/'patient','native-test');c.add_series(p,series_id='t1',modality='T1')
    a=c.patient_affine(c.series('t1'))
    vessel=np.zeros(image.shape,np.uint8);vessel[14,5:28,15]=1
    target=np.zeros(image.shape,np.uint8);target[8:13,9:14,13:18]=1
    for key,data,kind,role in [('brain',np.ones(image.shape,np.uint8),'brain','context'),('artery',vessel,'artery','critical'),('target',target,'synthetic-target','target')]:
        c.add_layer(data,a,layer_id=key,source_series='t1',kind=kind,role=role,color='#abcdef')
    c.save();return c


def test_whole_envelope_and_sheared_inverse_support():
    a=np.eye(4);a[0,1]=.8
    result=envelope_coverage(a,(20,20,20),world(a,[.2,5,5]),world(a,[.2,12,5]),1)
    assert result['centerlineWithinField'] and not result['wholeEnvelopeWithinField']
    assert result['envelopeVoxelBounds'][0][0]==pytest.approx(.2-np.sqrt(1+.8**2))
    assert '0-' in result['outsideFaces']
    assert envelope_coverage(a,(20,20,20),world(a,[5,5,5]),world(a,[5,12,5]),1)['wholeEnvelopeWithinField']


@pytest.fixture
def cortical(native):
    affine = native.patient_affine(native.series('t1'))
    x, y, z = np.indices((32, 32, 32))
    cerebral = (3 <= y) & (y < 28) & (4 <= z) & (z < 28)
    masks = {
        'right-cerebral-hemisphere': cerebral & (4 <= x) & (x < 16),
        'left-cerebral-hemisphere': cerebral & (18 <= x) & (x < 27),
        'cortical-entry-surface': cerebral & ((x == 4) | (x == 26)),
        'supratentorial-compartment': y >= 3,
        'falx-cerebri': cerebral & (x == 17),
        'tentorium-cerebelli': y == 2,
        'posterior-fossa': y < 2,
    }
    for role, mask in masks.items():
        row = native.add_layer(mask.astype(np.uint8), affine, layer_id=role, label='Generated evidence ' + role,
                               source_series='t1', kind='context', role='context', color='#abcdef',
                               provenance={'cortical_access_role': role, 'cortical_access_evidence': 'generated-reference'})
        row['review_status'] = 'generated-engineering'
    native.save()
    return native


def _cortical_route(collection, entry, target=(10, 11, 15), radius=.4):
    affine = collection.patient_affine(collection.series('t1'))
    return {'entry': world(affine, entry).tolist(), 'target': world(affine, target).tolist(),
            'radiusMm': radius, 'targetLayerId': 'target', 'patchRadiusMm': 0}


def test_oblique_lower_entry_supported_and_revision_bound(cortical):
    route = _cortical_route(cortical, [3.5, 6, 16])
    assert route['entry'][2] < route['target'][2]
    result = dispatch_plan(cortical, {'action': 'suggest', **route})
    proposal, = result['approaches']
    access = proposal['measurements']['accessAssessment']
    assert access['status'] == 'supported' and access['hemisphere'] == 'right'
    assert access['scope'] == 'ipsilateral-supratentorial-cortical'
    assert not access['missingEvidence'] and not access['violations']
    assert access['postEntryEnvelopeSupported'] and access['distalCapSupported']
    assert access['surfaceEntry']['point'] == pytest.approx(route['entry'])
    assert access['surfaceEntry']['hemisphereCell'] == [4, 6, 16]
    assert access['surfaceEntry']['surfaceLayerId'] == 'cortical-entry-surface'
    assert access['sourceSnapshotId'] == result['snapshotId'] == cortical.identity()
    assert not access['simulationExecutionAuthorized'] and not result['simulationExecutionAuthorized']
    reopened = Collection(cortical.root)
    assert dispatch_plan(reopened, {'action': 'list'})['approaches'][0] == proposal
    assert dispatch_plan(reopened, {'action': 'list'})['proposalReport'] == result['proposalReport']
    surface = cortical.layer('cortical-entry-surface')
    surface['revision'] += 1
    cortical.save('Revise source cortical surface')
    reopened = dispatch_plan(Collection(cortical.root), {'action': 'list'})
    assert reopened['approaches'][0]['assessmentStatus'] == 'stale'
    assert reopened['proposalReport']['assessmentStatus'] == 'stale'


def test_missing_access_source_keeps_native_geometric_comparison(native):
    route = _cortical_route(native, [3.5, 6, 16])
    result = dispatch_plan(native, {'action': 'suggest', **route})
    proposal, = result['approaches']
    access = proposal['measurements']['accessAssessment']
    assert access['status'] == 'incomplete' and access['surfaceEntry'] is None
    assert {'left-cerebral-hemisphere', 'right-cerebral-hemisphere', 'falx-cerebri', 'tentorium-cerebelli',
            'cortical-entry-surface'}.issubset(access['missingEvidence'])
    comparison = dispatch_plan(native, {'action': 'compareInputs', **route})['inputComparison']
    for arm in comparison['arms'].values():
        nomination, = arm['nominated']
        assert nomination['nominalEligible'] and nomination['accessAssessment'] == access


@pytest.mark.parametrize(('entry', 'reason'), [
    ([9, 1, 15], 'posterior-fossa'),
    ([10, 2.5, 15], 'tentorium-cerebelli'),
    ([26.5, 7, 16], 'contralateral cerebral hemisphere'),
    ([25, 7, 16], 'falx-cerebri'),
    ([9, 15, 16], 'Entry must cross from outside to inside'),
])
def test_source_established_wrong_access_rejected(cortical, entry, reason):
    route = _cortical_route(cortical, entry)
    manual = dispatch_plan(cortical, {'action': 'upsert', 'approach': route})['approaches'][0]['measurements']
    assert manual['status'] == 'known-conflict'
    assert manual['accessAssessment']['status'] == 'rejected'
    assert any(reason in violation for violation in manual['accessAssessment']['violations'])
    receipt = dispatch_plan(cortical, {'action': 'suggest', **route})['proposalReport']
    assert receipt['returned'] == 0 and receipt['candidates'][0]['accessAssessment'] == manual['accessAssessment']


def test_distal_cap_and_partial_boundary_evidence(cortical):
    hemisphere = cortical.layer('right-cerebral-hemisphere')
    mask = cortical.mask(hemisphere).astype(np.uint8)
    # The target point stays in the hemisphere but the distal spherical cap
    # touches its source-labelled boundary.
    mask[11, 11, 15] = 0
    hemisphere['data'] = cortical.put_array(mask)
    hemisphere['revision'] += 1
    cortical.save()
    route = _cortical_route(cortical, [3.5, 6, 16], radius=.6)
    access = dispatch_plan(cortical, {'action': 'upsert', 'approach': route})['approaches'][0]['measurements']['accessAssessment']
    assert access['status'] == 'rejected' and access['surfaceEntry'] is not None
    assert access['distalCapSupported'] is False

    tentorium = cortical.layer('tentorium-cerebelli')
    tentorium['coverage_fields'] = {'supported_axial_indices': [8, 9, 10]}
    cortical.save()
    route = _cortical_route(cortical, [3.5, 6, 16])
    access = dispatch_plan(cortical, {'action': 'upsert', 'approach': route})['approaches'][-1]['measurements']['accessAssessment']
    assert access['status'] == 'incomplete' and 'tentorium-cerebelli-corridor-coverage' in access['missingEvidence']


def test_protection_uses_kind_and_legacy_source_id_not_display_labels(native):
    affine = native.patient_affine(native.series('t1'))
    mask = np.zeros((32, 32, 32), np.uint8)
    mask[10, 18, 15] = 1
    for key, kind, label in (('region-a', 'brainstem', 'Renamed A'), ('head-7', 'brain-region', 'Renamed B'),
                             ('context', 'context', 'Brainstem cerebellum pons')):
        native.add_layer(mask, affine, layer_id=key, source_series='t1', kind=kind, role='context',
                         color='#abcdef', label=label)
    native.save()
    route = _cortical_route(native, [10, 26, 15])
    measured = dispatch_plan(native, {'action': 'upsert', 'approach': route})['approaches'][0]['measurements']
    assert set(measured['detectedConflictLayerIds']) == {'region-a', 'head-7'}
    assert measured['accessAssessment']['status'] == 'rejected'
    for row in measured['structures']:
        if row['layerId'] in {'region-a', 'head-7'}:
            assert row['excluded'] and row['confirmedIntersection']


def test_labels_radius_sensitivity_empty_unknown(native):
    c=native;a=c.patient_affine(c.series('t1'));g=NativeRouteGeometry(Snapshot.capture(c))
    start,end=world(a,[10,26,15]),world(a,[10,11,15])
    clear,bigger=g.measure(start,end,.5),g.measure(start,end,4)
    artery=next(s for s in clear['structures'] if s['layerId']=='artery')
    crossed=next(s for s in bigger['structures'] if s['layerId']=='artery')
    assert artery['available'] and not artery['confirmedIntersection']
    assert crossed['confirmedIntersection'] and bigger['status']=='known-conflict'
    assert crossed['clearanceLowerBoundMm']<artery['clearanceLowerBoundMm']
    assert clear['targets'][0]['targetPointInsideLabel'] and clear['normalTraversal']['available']
    assert clear['clinicalDeficitProbability'] is None and not clear['simulationExecutionAuthorized']
    c.add_layer(np.zeros((32,32,32),np.uint8),a,layer_id='empty-language',source_series='t1',kind='language',role='critical',color='#ffffff');c.save()
    empty=NativeRouteGeometry(Snapshot.capture(c)).measure(start,end,.5)
    row=next(s for s in empty['structures'] if s['layerId']=='empty-language')
    assert not row['available'] and 'clearanceLowerBoundMm' not in row
    assert any('empty-language: empty' in x for x in empty['unknowns'])
    assert route_clearance(Snapshot.capture(c),start,end,.5)['constraints']['language']=={'available':False}


def test_shear_contact_remains_conservative(native):
    c = native
    shear = np.eye(4)
    shear[0, 1] = .4
    a = c.patient_affine(c.series('t1')) @ shear
    mask = np.zeros((20, 20, 20), np.uint8)
    mask[10, 10, 10] = 1
    c.add_layer(mask, a, layer_id='sheared', source_series='t1', kind='vein', role='critical', color='#ff0000')
    c.save()
    measured=NativeRouteGeometry(Snapshot.capture(c)).measure(world(a,[4,10,10]),world(a,[16,10,10]),.2)
    row=next(s for s in measured['structures'] if s['layerId']=='sheared')
    assert row['confirmedIntersection'] and 'shear' in row['intersectionModel'] and row['clearanceLowerBoundMm']<0


def test_exact_orthogonal_cell_margin_avoids_circumsphere_false_refusal(native):
    c = native
    affine = c.patient_affine(c.series('t1'))
    start, end = world(affine, [[13, 22, 15], [13, 12, 15]])
    geometry = NativeRouteGeometry(Snapshot.capture(c))
    # Cell centres are 1 mm away. The vessel's nearest face is 0.5 mm
    # away, while its circumsphere extends beyond the 0.2 mm capsule.
    passed = geometry.measure(start, end, .2, {'minClearanceMm': .25, 'excludedLayerIds': ['artery']})
    assert passed['constraintFailures'] == []
    assert passed['structures'][0]['suppliedClearanceEstablished']
    insufficient = geometry.measure(start, end, .2, {'minClearanceMm': .35})
    assert not insufficient['detectedConflictLayerIds']
    assert 'declared clearance' in insufficient['constraintFailures'][0]
    assert not insufficient['structures'][0]['suppliedClearanceEstablished']
    assert passed['structures'][0]['clearanceLowerBoundMm'] == insufficient['structures'][0]['clearanceLowerBoundMm']
    tangent = geometry.measure(start, end, .5, {'minClearanceMm': 0})
    assert tangent['detectedConflictLayerIds'] == ['artery']
    assert 'detected capsule contact' in tangent['constraintFailures'][0]


def test_declared_clearance_margin_requires_source_field_coverage(native):
    c = native
    affine = c.patient_affine(c.series('t1'))
    mask = np.zeros((16, 32, 32), np.uint8)
    mask[10, 15, 15] = 1
    c.add_layer(mask, affine, layer_id='cropped-artery', source_series='t1', kind='artery', role='critical', color='#ff0000')
    c.save()
    geometry = NativeRouteGeometry(Snapshot.capture(c))
    start, end = world(affine, [[15, 22, 15], [15, 12, 15]])
    unguarded = geometry.measure(start, end, .2)
    guarded = geometry.measure(start, end, .2, {'minClearanceMm': .5, 'excludedLayerIds': ['cropped-artery']})
    assert unguarded['structures'][-1]['coverage']['wholeEnvelopeWithinField']
    row = guarded['structures'][-1]
    assert row['suppliedClearanceEstablished'] and not row['coverage']['wholeEnvelopeWithinField']
    assert not guarded['detectedConflictLayerIds']
    assert any('explicit exclusion is unassessed' in reason for reason in guarded['constraintFailures'])


def test_target_label_outside_acquisition_does_not_erase_traversal(native, tmp_path):
    c = native
    affine = c.patient_affine(c.series('t1'))
    image = nib.Nifti1Image(np.ones((8, 32, 32), np.float32), affine)
    image.header.set_xyzt_units('mm')
    source = tmp_path / 'limited.nii.gz'
    nib.save(image, source)
    c.add_series(source, series_id='limited', modality='FLAIR')
    c.set_registration('limited', np.eye(4), status='source-provided', method='declared fixture')
    c.add_layer(np.ones((32, 32, 32), np.uint8), affine, layer_id='unsupported-target',
                source_series='limited', kind='synthetic-target', role='target', color='#abcdea')
    c.save()
    geometry = NativeRouteGeometry(Snapshot.capture(c))
    start, end = world(affine, [[10, 26, 15], [10, 11, 15]])
    measured = geometry.measure(start, end, .5, target_id='unsupported-target')
    assert not measured['targets'][0]['acquisitionCoverage']['wholeEnvelopeWithinField']
    assert not measured['targets'][0]['targetPointInsideLabel']
    assert 'endpoint is outside or unassessed in the selected target label' in measured['constraintFailures']
    traversal = measured['normalTraversal']
    assert traversal['conservativeVolumeUpperBoundMm3'] > 0
    assert traversal['supportedNonTargetVoxelCenterVolumeMm3'] is None
    assert traversal['unclassifiedBrainVoxelCenterCount'] > 0


def test_paired_report_calls_incomplete_evidence_incomplete(native):
    affine = native.patient_affine(native.series('t1'))
    start, end = world(affine, [[10, 26, 15], [10, 11, 15]])
    result = dispatch_plan(native, {'action': 'compareInputs', 'start': start.tolist(), 'end': end.tolist(),
                                    'radiusMm': .5, 'patchRadiusMm': 0, 'targetLayerId': 'target'})
    for arm in result['inputComparison']['arms'].values():
        selected, = arm['nominated']
        assert selected['status'] == 'incomplete-coverage'
        assert selected['scenarioGuard']['incompleteScenarioIds']
        assert arm['refusalCount'] == 0


def test_streamline_paths_not_bridged_and_segment_math(native):
    c=native;a=c.patient_affine(c.series('t1'))
    points=world(a,np.array([[4,20,15],[6,20,15],[20,20,15],[22,20,15]]))
    c.document['tractography']=[{'id':'tract-A','label':'Supplied bundle A','source_series':'t1','points_ras_mm':c.put_array(points.astype(np.float32)),'offsets':c.put_array(np.array([0,2,4],np.uint32))}];c.save()
    measured=NativeRouteGeometry(Snapshot.capture(c)).measure(world(a,[13,25,15]),world(a,[13,15,15]),.5)
    row=next(s for s in measured['structures'] if s['layerId']=='tract-A')
    assert row['available'] and not row['confirmedIntersection'] and row['clearanceLowerBoundMm']>5
    values=_segment_bundle_distances(np.array([0.,0,0]),np.array([10.,0,0]),np.array([[5.,-3,0],[2.,2,0]]),np.array([[5.,3,0],[2.,2,0]]))
    assert values==pytest.approx([0,2])


def test_save_reopen_edits_and_source_invalidation(native,tmp_path):
    c=native;a=c.patient_affine(c.series('t1'));identity=c.identity()
    value={'entry':world(a,[10,26,15]).tolist(),'target':world(a,[10,11,15]).tolist(),'radiusMm':.5,'name':'Saved corridor'}
    saved=dispatch_plan(c,{'action':'upsert','approach':value})['approaches'][0]
    assert saved['assessmentStatus']=='current' and c.identity()==identity
    assert read_approaches(Collection(c.root))[0]['measurements']==saved['measurements']
    edited=dispatch_plan(c,{'action':'upsert','approach':{**saved,'radiusMm':4},'assess':False})['approaches'][0]
    assert edited['assessmentStatus']=='unassessed' and 'measurements' not in edited
    reassessed=dispatch_plan(c,{'action':'assess','approachId':saved['id']})['approaches'][0]
    assert reassessed['measurements']['status']=='known-conflict'
    archive=tmp_path/'saved.rsworkspace';save_package(c,archive)
    reopened=Collection(open_package(archive,tmp_path/'opened').parent)
    assert read_approaches(reopened)==read_approaches(c)
    c.layer('artery')['revision']+=1;c.save('changed source labels')
    stale=read_approaches(Collection(c.root))[0]
    assert stale['assessmentStatus']=='stale' and stale['sourceSnapshotId']==identity
    assert dispatch_plan(c,{'action':'assess','approachId':saved['id']})['approaches'][0]['assessmentStatus']=='current'


def test_same_snapshot_cached_measurements_revalidate_geometry(native):
    c = native
    reference = c.series('t1')
    a = c.patient_affine(reference)
    value = {'entry': world(a, [10, 26, 15]).tolist(), 'target': world(a, [10, 11, 15]).tolist(),
             'radiusMm': .5, 'name': 'Saved corridor'}
    saved = dispatch_plan(c, {'action': 'upsert', 'approach': value})['approaches'][0]
    assert saved['assessmentStatus'] == 'current'
    path = c.path(reference['original']['path'])
    image = nib.load(path)
    changed = nib.Nifti1Image(np.asarray(image.dataobj), image.affine, image.header.copy())
    changed.header.set_xyzt_units('unknown')
    nib.save(changed, path)
    checked, = read_approaches(c)
    assert checked['assessmentStatus'] == 'stale'
    assert checked['sourceSnapshotId'] == c.identity() == saved['sourceSnapshotId']


def test_saved_vascular_receipts_revalidate_evidence_without_staling_all_v6(native, monkeypatch):
    c = native
    a = c.patient_affine(c.series('t1'))
    entry, end = world(a, [10, 26, 15]).tolist(), world(a, [10, 11, 15]).tolist()
    args = {'action': 'suggest', 'targetLayerId': 'target', 'start': entry, 'end': end, 'radiusMm': .5,
            'entryPatch': {'center': entry, 'normal': [0, 1, 0], 'radiusMm': 0.}}
    generated = dispatch_plan(c, args)
    assert generated['approaches'][0]['assessmentStatus'] == 'current'
    assert generated['proposalReport']['assessmentStatus'] == 'current'
    import json
    from resectionlab.patient_anatomy import digest
    path = c.root/'anatomy-approaches.json'
    previous = json.loads(path.read_text())
    for approach in previous['approaches']:
        approach.pop('metricEvidencePolicy')
    report = previous['comparisons']['target']['proposalReport']
    report['report'].pop('metricEvidencePolicy')
    report['hash'] = digest(report['report'])
    path.write_text(json.dumps(previous))
    checked = dispatch_plan(c, {'action': 'list'})
    assert checked['approaches'][0]['assessmentStatus'] == 'current'
    assert checked['proposalReport']['assessmentStatus'] == 'current'
    monkeypatch.setattr('resectionlab.anatomy_approaches.vascular_evidence_supported',
                        lambda _collection, row: row['kind'] != 'artery')
    checked = dispatch_plan(c, {'action': 'list'})
    assert checked['approaches'][0]['assessmentStatus'] == 'stale'
    assert checked['proposalReport']['assessmentStatus'] == 'stale'


def test_bounded_distinct_suggestions_and_shortfall(native):
    c=native;a=c.patient_affine(c.series('t1'))
    args={'action':'suggest','start':world(a,[10,26,15]).tolist(),'end':world(a,[10,11,15]).tolist(),'radiusMm':.5,'patchRadiusMm':5}
    result=dispatch_plan(c,args);rows=result['approaches'];assert 1<=len(rows)<=3
    for i,row in enumerate(rows):
        assert row['entry'][2]>=row['target'][2] and row['measurements']['coverage']['withinReferenceEnvelope']
        for other in rows[:i]:assert np.linalg.norm(np.array(row['entry'])-other['entry'])>=4
    assert [r['id'] for r in rows]==[r['id'] for r in dispatch_plan(c,args)['approaches']]
    tiny=dispatch_plan(c,{**args,'patchRadiusMm':0})
    assert len(tiny['approaches'])<=1 and tiny['proposalReport']['whyFewer']
    rejected=dispatch_plan(c,{**args,'constraints':{'maxLengthMm':1}})
    assert not rejected['approaches'] and rejected['proposalReport']['reasons']
    with pytest.raises(ValueError,match='supratentorial'):dispatch_plan(c,{**args,'constraints':{'supratentorialOnly':False}})


def test_vectorized_segment_predicate_matches_static_reference():
    from resectionlab.geometry import segment_segment_distance
    rng=np.random.default_rng(42)
    start,end=rng.normal(size=(2,3));first,last=rng.normal(size=(2,200,3))
    first[:3]=[[0,0,0],[1,1,1],[2,2,2]];last[:3]=first[:3]
    actual=_segment_bundle_distances(start,end,first,last)
    expected=[segment_segment_distance(start,end,a,b) for a,b in zip(first,last)]
    assert actual==pytest.approx(expected,abs=1e-10)


def test_bridge_tract_selection_and_plan_persistence(native,tmp_path):
    from resectionlab.desktop_bridge import BridgeSession,_Request
    c=native;a=c.patient_affine(c.series('t1'));points=world(a,np.array([[5,20,15],[6,20,15]]))
    c.document['tractography']=[{'id':'tract-A','label':'Supplied bundle A','source_series':'t1','points_ras_mm':c.put_array(points.astype(np.float32)),'offsets':c.put_array(np.array([0,2],np.uint32))}];c.save()
    b=BridgeSession(tmp_path/'transfers')
    def call(op,args):return b.execute(op,args,_Request('test'),lambda *a:None)
    opened=call('anatomyOpen',{'path':str(c.root/'patient-anatomy.json')})
    binding={'collectionId':opened['collectionId'],'snapshotId':c.identity()}
    view=call('anatomyView',{**binding,'layers':['tract-A'],'seriesId':'t1'})
    assert view['tractography'][0]['id']=='tract-A' and not view['compartments']
    with pytest.raises(ValueError,match='Unknown anatomical'):call('anatomyView',{**binding,'layers':['absent-tract']})
    result=call('anatomyPlan',{**binding,'action':'upsert','approach':{'entry':world(a,[10,26,15]).tolist(),'target':world(a,[10,11,15]).tolist(),'radiusMm':.5}})
    assert len(result['approaches'])==1 and not result['simulationExecutionAuthorized']
    reopened=call('anatomyOpen',{'path':str(c.root/'patient-anatomy.json')});assert b.anatomy_plan_cache is None
    assert call('anatomyPlan',{'collectionId':reopened['collectionId'],'snapshotId':c.identity(),'action':'list'})['approaches']==result['approaches']


def test_source_bound_rigid_transform_for_labels_and_streamlines(native):
    c=native;reference=c.series('t1');a=c.patient_affine(reference)
    c.add_series(c.path(reference['original']['path']),series_id='flair',modality='FLAIR')
    matrix=np.eye(4);matrix[:3,3]=[3,-2,1]
    c.set_registration('flair',matrix,status='source-provided',method='test supplied source transform')
    mask=np.zeros((32,32,32),np.uint8);mask[13,20,15]=1
    c.add_layer(mask,a,layer_id='source-vessel',source_series='flair',kind='vein',role='critical',color='#ff0000')
    points=world(a,np.array([[5,20,15],[20,20,15]]))
    c.document['tractography']=[{'id':'source-tract','label':'Source bundle','source_series':'flair','points_ras_mm':c.put_array(points.astype(np.float32)),'offsets':c.put_array(np.array([0,2],np.uint32))}];c.save()
    start,end=world(matrix,world(a,[13,25,15])),world(matrix,world(a,[13,15,15]))
    m=NativeRouteGeometry(Snapshot.capture(c)).measure(start,end,.5)
    selected=[s for s in m['structures'] if s['layerId'] in {'source-tract','source-vessel'}]
    assert len(selected)==2 and all(s['confirmedIntersection'] for s in selected)
    c.document['transforms']['flair']['status']='unreviewed';c.save()
    unknown=NativeRouteGeometry(Snapshot.capture(c)).measure(start,end,.5)
    assert all(not s['available'] for s in unknown['structures'] if s['layerId'] in {'source-tract','source-vessel'})
    assert not unknown['simulationExecutionAuthorized']


def test_normal_upper_bound_never_subtracts_partly_target_brain_cell(tmp_path):
    # One 1 mm brain cell contains a smaller 0.5 mm target cell. The capsule's
    # spherical end caps protrude from target while staying inside brain.
    image=nib.Nifti1Image(np.ones((4,4,4),np.float32),np.eye(4));image.header.set_xyzt_units('mm')
    p=tmp_path/'tiny.nii.gz';nib.save(image,p)
    c=Collection.create(tmp_path/'tiny-patient','tiny');c.add_series(p,series_id='t1',modality='T1')
    brain=np.zeros((4,4,4),np.uint8);brain[1,1,1]=1
    c.add_layer(brain,np.eye(4),layer_id='brain',source_series='t1',kind='brain',role='context',color='#aaaaaa')
    a=np.diag([.5,.5,.5,1.]);a[:3,3]=1
    target=c.add_layer(np.ones((1,1,1),np.uint8),a,layer_id='target',source_series='t1',kind='tumor',role='target',color='#ff00ff');c.save()
    result=NativeRouteGeometry(Snapshot.capture(c)).measure([1,1,.75],[1,1,1.25],.1)
    upper=result['normalTraversal']['conservativeVolumeUpperBoundMm3']
    assert result['normalTraversal']['available'] and upper==pytest.approx(1)
    assert upper>=4*np.pi*.1**3/3
    # A completely labelled surrounding target box can prove the brain cell is
    # inside target; only then may its contribution be subtracted entirely.
    target['data']=c.put_array(np.ones((4,4,4),np.uint8));target['affine']=np.eye(4).tolist();c.save()
    proved=NativeRouteGeometry(Snapshot.capture(c)).measure([1,1,.75],[1,1,1.25],.1)
    assert proved['normalTraversal']['conservativeVolumeUpperBoundMm3']==0


def test_empty_brain_is_unassessed_and_manual_entry_patch_enforced(native):
    from resectionlab.anatomy_approaches import assess_approach
    c=native;c.layer('brain')['data']=c.put_array(np.zeros((32,32,32),np.uint8));c.save()
    m=NativeRouteGeometry(Snapshot.capture(c)).measure([3,3,3],[1,1,1],.1)
    assert not m['normalTraversal']['available'] and 'empty brain mask' in m['normalTraversal']['reason']
    assert 'conservativeVolumeUpperBoundMm3' not in m['normalTraversal']
    assessed=assess_approach(c,{'entry':[3,3,3],'target':[1,1,1],'radiusMm':.1,
                                 'entryPatch':{'center':[1,1,3],'normal':[0,0,-1],'radiusMm':.1}})
    assert any('outside declared entry-centre disk' in x for x in assessed['measurements']['constraintFailures'])
    assert assessed['measurements']['status']=='known-conflict'


def test_changed_assessment_engine_invalidates_saved_comparison(native):
    import json
    from resectionlab.patient_anatomy import digest
    from resectionlab.anatomy_approaches import FILENAME,ASSESSMENT_VERSION
    c=native;a=c.patient_affine(c.series('t1'))
    dispatch_plan(c,{'action':'upsert','approach':{'entry':world(a,[10,26,15]).tolist(),'target':world(a,[10,11,15]).tolist(),'radiusMm':.5}})
    path=c.root/FILENAME;data=json.loads(path.read_text());saved=data['approaches'][0]
    assert saved['measurements']['assessmentVersion']==ASSESSMENT_VERSION
    saved['measurements']['assessmentVersion']='native-source-capsule-assessment-v1'
    saved['measurementHash']=digest(saved['measurements']);path.write_text(json.dumps(data))
    assert read_approaches(c)[0]['assessmentStatus']=='stale'


def test_edit_and_reassessment_preserve_comparison_order(native):
    c=native;a=c.patient_affine(c.series('t1'))
    for index in range(3):
        dispatch_plan(c,{'action':'upsert','approach':{'id':f'ordered-{index}','name':f'Approach {index}',
            'entry':world(a,[10+index,26,15]).tolist(),'target':world(a,[10,11,15]).tolist(),'radiusMm':.5}})
    before=read_approaches(c);ids=[row['id'] for row in before]
    changed=dispatch_plan(c,{'action':'upsert','approach':{**before[0],'radiusMm':1}})
    assert [row['id'] for row in changed['approaches']]==ids
    assert changed['approaches'][0]['radiusMm']==1
    refreshed=dispatch_plan(c,{'action':'assess','approachId':ids[0]})
    assert [row['id'] for row in refreshed['approaches']]==ids


def test_saved_entry_patch_edits_recompute_failures_and_survive_reopen(native, tmp_path):
    c = native
    a = c.patient_affine(c.series('t1'))
    entry, target = world(a, [10, 26, 15]), world(a, [10, 11, 15])
    value = {'id': 'patch-edited', 'name': 'Editable patch', 'entry': entry.tolist(),
             'target': target.tolist(), 'radiusMm': .5,
             'entryPatch': {'center': (entry + [2, 0, 0]).tolist(), 'normal': [0, 0, -1], 'radiusMm': 4},
             'constraints': {'maxLengthMm': 50, 'minClearanceMm': 0}}
    first = dispatch_plan(c, {'action': 'upsert', 'approach': value})['approaches'][0]
    assert not any('entry-centre disk' in f for f in first['measurements']['constraintFailures'])
    edited = {**first, 'entryPatch': {**first['entryPatch'], 'radiusMm': 1}}
    changed = dispatch_plan(c, {'action': 'upsert', 'approach': edited})['approaches'][0]
    assert changed['parameterHash'] != first['parameterHash']
    assert changed['measurementHash'] != first['measurementHash']
    assert changed['measurements']['lengthMm'] == first['measurements']['lengthMm']
    assert changed['measurements']['status'] == 'known-conflict'
    assert any('entry-centre disk' in f for f in changed['measurements']['constraintFailures'])
    assert changed['sourceSnapshotId'] == first['sourceSnapshotId'] == c.identity()
    assert read_approaches(Collection(c.root))[0] == changed
    archive = tmp_path / 'patch.rsworkspace'
    save_package(c, archive)
    restored = Collection(open_package(archive, tmp_path / 'cold-working').parent)
    assert read_approaches(restored)[0] == changed
    tightened = dispatch_plan(c, {'action': 'upsert', 'approach': {
        **changed, 'constraints': {**changed['constraints'], 'maxLengthMm': 1}}})['approaches'][0]
    assert any('maximum length' in f for f in tightened['measurements']['constraintFailures'])
    assert tightened['parameterHash'] != changed['parameterHash']


def test_native_shared_scenarios_keep_fixed_capsule_and_report_contact(native):
    c = native
    reference = c.series('t1')
    a = c.patient_affine(reference)
    c.add_series(c.path(reference['original']['path']), series_id='flair', modality='FLAIR')
    c.set_registration('flair', np.eye(4), status='source-provided', method='declared fixture')
    mask = np.zeros((32, 32, 32), np.uint8)
    mask[12, 20, 15] = 1
    c.add_layer(mask, a, layer_id='registered-vein', source_series='flair', kind='vein', role='critical', color='#ff0000')
    points = world(a, [[12, 20, 15], [12, 21, 15]])
    c.document['tractography'] = [{
        'id': 'registered-tract', 'label': 'Registered fixture tract', 'source_series': 'flair',
        'points_ras_mm': c.put_array(points.astype(np.float32)),
        'offsets': c.put_array(np.array([0, 2], np.uint32)),
    }]
    c.save()
    args = {'action': 'suggest', 'start': world(a, [10, 26, 15]).tolist(), 'end': world(a, [10, 11, 15]).tolist(),
            'targetLayerId': 'target', 'radiusMm': .6, 'patchRadiusMm': 0,
            'sensitivityConfig': {'translationMm': 1.5, 'boundaryMm': .2}}
    reply = dispatch_plan(c, args)
    assert not reply['approaches']
    report = reply['proposalReport']
    assert report['nominalEligible'] == report['scenarioVetoCount'] == report['rejected'] == 1
    assert report['scenarioGuardEligible'] == report['returned'] == 0 and report['whyFewer']
    rejected, = report['candidates']
    assert rejected['nominalEligible'] and rejected['disposition'] == 'scenario-vetoed'
    assert report['candidateFreeze']['candidates'][0]['id'] == rejected['id']
    assert dispatch_plan(Collection(c.root), {'action': 'list'})['proposalReport'] == report
    edited = dispatch_plan(c, {**args, 'action': 'upsert'})
    approach, = edited['approaches']
    measured = approach['measurements']
    assert measured['capsule']['start'] == args['start']
    assert measured['capsule']['radiusMm'] == .6
    assert measured['simulatedRemovedTargetVolumeMm3'] is None
    result = measured['sensitivity']
    design = reply['proposalReport']['scenarioSet']
    assert result['scenarioSetId'] == design['id']
    assert design['registrationGroups'] == {'flair': ['flair']}
    shifted_id = next(s['id'] for s in design['scenarios']
                      if s['sourceTranslationsMm'].get('flair') == [1.5, 0, 0] and s['boundaryMm'] == 0)
    shifted = next(s for s in result['scenarios'] if s['id'] == shifted_id)
    nominal = result['scenarios'][0]
    assert nominal['status'] == 'partial-coverage'
    assert shifted['status'] == 'constraint-violation'
    assert {'registered-vein', 'registered-tract'} <= set(shifted['conflictLayerIds'])
    assert shifted['sourceReadings']['registered-vein'][1] < nominal['sourceReadings']['registered-vein'][1]
    assert shifted['sourceReadings']['registered-tract'][1] < nominal['sourceReadings']['registered-tract'][1]
    assert shifted['values']['targetReachMm3'] == nominal['values']['targetReachMm3']
    assert shifted['values']['nonTargetTraversalMm3'] == nominal['values']['nonTargetTraversalMm3']
    metric = result['metrics']['vascularClearanceMm']
    assert metric['worst'] <= metric['tailMean'] < metric['nominal']
    assert result['metrics']['otherCriticalClearanceMm']['unknownCount'] == result['scenarioCount']
    frozen = report['finalistOutcomes'][rejected['id']]
    assert shifted_id in frozen['violatedScenarioIds']
    compact = next(s for s in frozen['scenarios'] if s['id'] == shifted_id)
    assert compact['conflictLayerIds'] == shifted['conflictLayerIds']
    assert 'otherCriticalClearanceMm' in compact['missingMetrics']

    paired = dispatch_plan(c, {**args, 'action': 'compareInputs'})['inputComparison']
    for arm in paired['arms'].values():
        assert arm['refusalCount'] == 1 and arm['abstentionCount'] == 2
        nomination, = arm['nominated']
        assert nomination['nominalEligible'] and nomination['status'] == 'refused'
        assert 'nominal' not in nomination['scenarioGuard']['vetoScenarioIds']
        assert shifted_id in nomination['scenarioGuard']['vetoScenarioIds']


def test_restricted_proposals_invariant_to_withheld_geometry_and_veto_without_retry(native):
    c = native
    a = c.patient_affine(c.series('t1'))
    args = {'action': 'compareInputs', 'start': world(a, [10, 26, 15]).tolist(), 'end': world(a, [10, 11, 15]).tolist(),
            'targetLayerId': 'target', 'radiusMm': .5, 'patchRadiusMm': 0}
    first = dispatch_plan(c, args)['inputComparison']
    contact = np.zeros((32, 32, 32), np.uint8)
    contact[10, 20, 15] = 1
    c.layer('artery')['data'] = c.put_array(contact)
    c.save()
    second = dispatch_plan(c, args)['inputComparison']
    rich, restricted = second['arms']['rich'], second['arms']['restricted']
    assert first['candidateBudget'] == second['candidateBudget'] == 1
    assert first['arms']['restricted']['candidates'] == restricted['candidates']
    assert first['arms']['restricted']['selectedCandidateIds'] == restricted['selectedCandidateIds']
    assert restricted['sourceSeriesIds'] == ['t1'] and restricted['tractIds'] == []
    assert 'artery' not in restricted['layerIds']
    assert rich['abstentionCount'] == 3 and rich['refusalCount'] == 0
    assert restricted['abstentionCount'] == 2 and restricted['refusalCount'] == 1
    refused, = restricted['nominated']
    assert refused['status'] == 'refused' and refused['detectedConflictLayerIds'] == ['artery']
    assert not second['simulationExecutionAuthorized']


def test_per_target_comparison_receipts_round_trip_and_go_stale(native, tmp_path):
    c = native
    a = c.patient_affine(c.series('t1'))
    extra = np.zeros((32, 32, 32), np.uint8)
    extra[19:22, 9:14, 13:18] = 1
    c.add_layer(extra, a, layer_id='target-08', source_series='t1', kind='synthetic-target', role='target', color='#abcdef')
    c.save()

    def task(target_id, x, **extra):
        start  = world(a, [x, 26, 15]).tolist()
        return {'action': 'suggest', 'start': start, 'end': world(a, [x, 11, 15]).tolist(),
                'radiusMm': .5, 'targetLayerId': target_id,
                'entryPatch': {'center': start, 'normal': [2., -3., 4.], 'radiusMm': 0.}, **extra}

    first = dispatch_plan(c, task('target', 10))
    paired = dispatch_plan(c, task('target', 10, action='compareInputs'))['inputComparison']
    rejected = dispatch_plan(c, task('target-08', 20, constraints={'maxLengthMm': 1.}))
    assert not rejected['approaches'] and rejected['proposalReport']['whyFewer']
    assert rejected['savedTargetLayerIds'] == ['target', 'target-08']
    assert len(rejected['proposalReport']['candidates']) == 1
    assert rejected['proposalReport']['candidates'][0]['disposition'] == 'rejected'

    pack = tmp_path / 'paired-native.rsworkspace'
    save_package(c, pack)
    reopened = Collection(open_package(pack, tmp_path / 'cold-paired').parent)
    latest = dispatch_plan(reopened, {'action': 'list'})
    assert latest['activeTargetLayerId'] == 'target-08'
    assert latest['proposalReport'] == rejected['proposalReport']
    restored = dispatch_plan(reopened, {'action': 'list', 'targetLayerId': 'target'})
    assert restored['approaches'] == first['approaches']
    assert restored['proposalReport'] == first['proposalReport']
    assert restored['inputComparison'] == paired
    assert restored['approaches'][0]['measurements']['targets'][0]['layerId'] == 'target'
    reopened.save('source revision')
    stale = dispatch_plan(reopened, {'action': 'list', 'targetLayerId': 'target'})
    assert stale['approaches'][0]['assessmentStatus'] == 'stale'
    assert stale['proposalReport']['assessmentStatus'] == stale['inputComparison']['assessmentStatus'] == 'stale'


def test_portable_study_accepts_comparison_metadata_above_legacy_limit(native, tmp_path):
    from resectionlab.anatomy_approaches import _bound_report, save_approaches

    report = _bound_report(native, {'diagnostics': 'A' * (2 * 1024**2)})
    save_approaches(native, [], comparisons={'target': {'proposalReport': report}}, target_id='target')
    pack = tmp_path / 'extended-metadata.rsworkspace'
    save_package(native, pack)
    restored = Collection(open_package(pack, tmp_path / 'extended-metadata-cold').parent)
    assert dispatch_plan(restored, {'action': 'list'})['proposalReport'] == report


def test_explicit_lesion_candidate_selection_preserves_evidence_and_reopens(native, tmp_path):
    from resectionlab.anatomy_planning import native_target_options

    c = native
    a = c.patient_affine(c.series("t1"))
    labels = np.zeros((32, 32, 32), np.uint8)
    labels[20:23, 9:14, 13:18] = 4
    proposal = c.add_layer(
        labels, a, layer_id="enhancing-proposal", source_series="t1",
        kind="enhancing-lesion", role="context", color="#abcdef",
        label="Enhancing lesion", label_value=4,
        provenance={"model": "test-research-estimate", "ontology": "enhancing lesion, class 4"},
    )
    c.add_layer(
        np.zeros_like(labels), a, layer_id="empty-lesion", source_series="t1",
        kind="tumor", role="context", color="#abcdef",
    )
    c.add_series(tmp_path / "source.nii.gz", series_id="flair", modality="FLAIR")
    c.set_registration("flair", np.eye(4), status="unreviewed", method="test estimate")
    c.add_layer(
        labels, a, layer_id="unaligned-lesion", source_series="flair",
        kind="nonenhancing-abnormality", role="context", color="#abcdef", label_value=4,
    )
    snapshot_id = c.save("native target candidates")
    assert dispatch_plan(c, {"action": "list"})["activeTargetLayerId"] is None
    options = {item["layerId"]: item for item in native_target_options(c)}
    option = options["enhancing-proposal"]
    assert option["reviewStatus"] == "unreviewed"
    assert option["origin"] == "Model estimate"
    assert option["labelValue"] == 4
    assert options["empty-lesion"]["reason"] == "Empty target label"
    assert "needs review" in options["unaligned-lesion"]["reason"]
    start, end = world(a, [[21, 26, 15], [21, 11, 15]]).tolist()
    target_id = option["layerId"]
    task = {"action": "suggest", "start": start, "end": end, "radiusMm": .5,
            "patchRadiusMm": 0, "targetLayerId": target_id}
    selection = {"action": "selectTarget", "targetLayerId": target_id,
                 "candidateBinding": option["candidateBinding"]}
    with pytest.raises(ValueError, match="Select this lesion"):
        dispatch_plan(c, task)
    with pytest.raises(ValueError, match="Empty target label"):
        dispatch_plan(c, {**selection, "targetLayerId": "empty-lesion"})
    assert dispatch_plan(c, selection)["activeTargetLayerId"] == target_id
    assert c.identity() == snapshot_id
    assert c.layer(target_id) == proposal
    assert target_id not in Snapshot.capture(c).simulation_grid()["targets"]

    result = dispatch_plan(c, task)
    report = result["proposalReport"]
    assert report["task"]["targetCandidateBinding"] == option["candidateBinding"]
    approach, = result["approaches"]
    assert approach["targetCandidateBinding"] == option["candidateBinding"]
    target, = approach["measurements"]["targets"]
    assert target["layerId"] == target_id
    assert target["sourceIdentity"]["labelValue"] == 4
    assert target["reviewStatus"] == "unreviewed"
    assert target["targetPointInsideLabel"]
    assert not approach["measurements"]["simulationExecutionAuthorized"]

    pack = tmp_path / "explicit-target.rsworkspace"
    save_package(c, pack)
    restored = Collection(open_package(pack, tmp_path / "cold-target").parent)
    reopened = dispatch_plan(restored, {"action": "list", "targetLayerId": None})
    assert reopened["activeTargetLayerId"] == target_id
    assert reopened["approaches"] == result["approaches"]
    assert reopened["proposalReport"] == report

    restored.save("unrelated anatomy revision")
    stale = dispatch_plan(restored, {"action": "list"})
    assert stale["targetSelectionIssue"] is None
    assert stale["activeTargetLayerId"] == target_id
    assert stale["approaches"][0]["assessmentStatus"] == "stale"
    assert stale["proposalReport"]["assessmentStatus"] == "stale"

    corrected = labels == 4
    corrected[21, 11, 15] = False
    restored.commit_edit(
        target_id, corrected, expected=restored.identity(), author="fixture", operations=["erase"]
    )
    assert "changed" in dispatch_plan(restored, {"action": "list"})["targetSelectionIssue"]
    with pytest.raises(ValueError, match="Select this lesion"):
        dispatch_plan(restored, task)
    with pytest.raises(ValueError, match="evidence changed"):
        dispatch_plan(restored, selection)
    option = next(item for item in native_target_options(restored) if item["layerId"] == target_id)
    reselection = dispatch_plan(restored, {**selection, "candidateBinding": option["candidateBinding"]})
    assert reselection["targetSelectionIssue"] is None
    assert reselection["proposalReport"]["assessmentStatus"] == "stale"
    assert restored.layer(target_id)["review_status"] == "edited; anatomical review pending"


def test_routine_mri_vascular_estimate_stays_unassessed_in_native_planning(native, tmp_path):
    c = native
    a = c.patient_affine(c.series("t1"))
    same_field = np.ones((32, 32, 32), np.uint8)
    c.add_layer(
        same_field, a, layer_id="inferred-artery", source_series="t1", kind="artery",
        role="critical", color="#ff0000", provenance={"model": "SuperSynth", "kind": "model_estimate"},
    )
    c.add_series(tmp_path / "source.nii.gz", series_id="cta", modality="CTA")
    c.set_registration("cta", np.eye(4), status="source-provided", method="fixture")
    c.add_layer(
        same_field, a, layer_id="angiographic-vessels", source_series="cta", kind="vessels",
        role="critical", color="#ff0000", provenance={"model": "vesselfm", "kind": "model_estimate"},
    )
    for identity, model, kind in [
        ("unsupported-inference", "unknown-model", "vessel"),
        ("unsupported-venous-class", "vesselfm-base", "vein"),
    ]:
        c.add_layer(same_field, a, layer_id=identity, source_series="cta", kind=kind,
                    role="critical", color="#ff0000", provenance={"model": model})
    c.save()
    start, end = world(a, [[10, 26, 15], [10, 11, 15]])
    geometry = NativeRouteGeometry(Snapshot.capture(c))
    m = geometry.measure(start, end, .5)
    supplied = next(row for row in m["structures"] if row["layerId"] == "artery")
    estimate = next(row for row in m["structures"] if row["layerId"] == "inferred-artery")
    angiographic = next(row for row in m["structures"] if row["layerId"] == "angiographic-vessels")
    assert supplied["available"]
    assert not estimate["available"] and "Inferred vascular context" in estimate["reason"]
    assert "clearanceLowerBoundMm" not in estimate
    assert "inferred-artery" not in m["detectedConflictLayerIds"]
    assert angiographic["available"] and angiographic["confirmedIntersection"]
    assert "angiographic-vessels" in m["detectedConflictLayerIds"]
    for identity in ("unsupported-inference", "unsupported-venous-class"):
        unsupported = next(row for row in m["structures"] if row["layerId"] == identity)
        assert not unsupported["available"] and "clearanceLowerBoundMm" not in unsupported
        assert identity not in m["detectedConflictLayerIds"]
    guarded = geometry.measure(start, end, .5, {"excludedLayerIds": ["inferred-artery"]})
    assert any("explicit exclusion is unassessed" in reason for reason in guarded["constraintFailures"])


def test_cropped_inference_does_not_turn_padding_into_observed_anatomy(native):
    from resectionlab.anatomy_planning import native_target_options

    c = native
    a = c.patient_affine(c.series("t1"))
    crop = [[6, 10, 5], [24, 20, 26]]
    for key, kind, positions in [
        ("cropped-brainstem", "brainstem", [[10, 16, 15], [10, 25, 15]]),
        ("cropped-target", "tumor", [[10, 12, 15], [10, 25, 15]]),
        ("padding-only-target", "tumor", [[10, 25, 15]]),
    ]:
        label = np.zeros((32, 32, 32), np.uint8)
        label[tuple(np.asarray(positions).T)] = 1
        row = c.add_layer(
            label, a, layer_id=key, source_series="t1", kind=kind,
            role="context", color="#abcdef", provenance={"model": "fixture"},
        )
        row["coverage_fields"] = {"inference_bounds_voxel_inclusive": crop}
    c.add_layer(
        np.ones((32, 32, 32), np.uint8), a, layer_id="dura-proxy", source_series="t1",
        kind="dura", role="context", color="#abcdef",
        provenance={"model": "dural-envelope", "evidence_use": "context_only", "supported_axial_indices": [15]},
    )
    c.document["display"] = {"planning_targets": {
        "cropped-target": {"cursor_ras_mm": world(a, [10, 25, 15]).tolist()},
    }}
    c.save()
    options = {row["layerId"]: row for row in native_target_options(c)}
    assert options["cropped-target"]["target"] == world(a, [10, 12, 15]).tolist()
    assert options["padding-only-target"]["target"] is None
    assert "inference" in options["padding-only-target"]["reason"]
    assert "dura-proxy" not in options

    geometry = NativeRouteGeometry(Snapshot.capture(c))
    start, end = world(a, [[10, 26, 15], [10, 12, 15]])
    measured = geometry.measure(start, end, .5, {"excludedLayerIds": ["dura-proxy"]}, target_id="cropped-target")
    brainstem = next(row for row in measured["structures"] if row["layerId"] == "cropped-brainstem")
    assert brainstem["confirmedIntersection"] and brainstem["intersectedCellCount"] == 1
    assert brainstem["acquisitionCoverage"]["wholeEnvelopeWithinField"]
    assert not brainstem["coverage"]["wholeEnvelopeWithinField"]
    assert brainstem["coverage"]["inferenceBoundsVoxelInclusive"] == crop
    assert "cropped-brainstem" in measured["detectedConflictLayerIds"]
    assert "cropped-brainstem: required envelope outside inference field" in measured["unknowns"]
    target, = measured["targets"]
    assert target["targetPointInsideLabel"]
    assert not target["coverage"]["wholeEnvelopeWithinField"]
    assert measured["normalTraversal"]["unclassifiedBrainVoxelCenterCount"] > 0
    dura = next(row for row in measured["structures"] if row["layerId"] == "dura-proxy")
    assert not dura["available"] and not dura["confirmedIntersection"]
    assert "geometric context only" in dura["reason"]
    assert any("dura-proxy: explicit exclusion is unassessed" in failure for failure in measured["constraintFailures"])

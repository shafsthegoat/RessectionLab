"""Independent v3 audit controls on generated native paired-lane programs.

These are software geometry/mechanics controls, never acquired or biological
evidence. Producer execution creates the held proof; the auditor rebuilds its
geometry and replays native prefixes without importing the producer compiler.
"""
import copy
import importlib.util
import json
from pathlib import Path
import tempfile

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from resectionlab import surgical_exposure as producer
from resectionlab.core import semantic_digest
from resectionlab.imaging import save_case
from resectionlab.workspace_bundle import load_workspace
from test_surgical_paired_exposure import paired_source

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('paired_source_exposure_auditor',ROOT/'scripts/verify_source_exposure.py')
audit=importlib.util.module_from_spec(spec);spec.loader.exec_module(audit)


def reseal(proof):
    public=proof['publicPlan'];public['publicSeal']=semantic_digest({k:v for k,v in public.items() if k!='publicSeal'})
    proof['preparationSeal']=semantic_digest({k:v for k,v in proof.items() if k!='preparationSeal'})
    return proof


@pytest.fixture(scope='module')
def evidence(paired_source,tmp_path_factory):
    source,native,ray=paired_source
    root=tmp_path_factory.mktemp('paired-exposure-independent-audit');cleanup=None
    if not root.resolve().is_relative_to(ROOT/'testcases'):
        cleanup=tempfile.TemporaryDirectory(prefix='paired-exposure-independent-audit-',dir=ROOT/'testcases');root=Path(cleanup.name)
    controls={**producer.DEFAULTS,'maxCandidates':1,'maxWallSeconds':90.,'maxNonTargetMm3':200.,'maxActions':128,'maxSeconds':180.}
    case,artifacts,_=load_workspace(native.root.parent/'scene/patient.ressectionlab');rows={}
    before=source.snapshot()
    for width in (0.,.3,3.42):
        selected,proof=producer.prepare(source,{**controls,'lateralSweepMm':width},native_collection=native.root,
            source_ray=ray,family='paired-lanes-v1')
        assert selected is not None and source.snapshot()==before
        assert proof['schema']=='resectionlab.source-exposure-preparation.v3'
        rows[width]=(selected,proof)
    record=copy.deepcopy(artifacts['surgical_toolbox'])
    record.update(branches={'source':source.snapshot(),**{str(w):e.snapshot() for w,(e,_) in rows.items()}},
        branchId='3.42',position=len(rows[3.42][0].actions))
    artifacts=copy.deepcopy(artifacts);artifacts['surgical_toolbox']=record
    path=root/'paired.ressectionlab';save_case(case,path,artifacts=artifacts);counter=0
    def run(proof,expected):
        nonlocal counter
        counter+=1;held=root/f'held-{counter:03d}.json';held.write_text(json.dumps(proof))
        return audit.verify(path,held,expected_snapshot=expected)
    try:yield {'rows':rows,'run':run,'path':path,'root':root,'source':source,'controls':controls,'native':native,'ray':ray}
    finally:
        if cleanup is not None:cleanup.cleanup()


@pytest.mark.parametrize('width,count',[(0.,5),(.3,8),(3.42,10)])
def test_actual_v3_programs_pass_independent_geometry_native_and_capacity_checks(evidence,width,count):
    selected,proof=evidence['rows'][width];before=evidence['path'].read_bytes()
    route=proof['publicPlan']['attempts'][0]['route'];assert len(route['paths'])==count
    result=evidence['run'](proof,selected.snapshot())
    assert result['passed'] and result['sourceInputsUnchanged'] and result['microsteps']>0 and result['boxTests']>0
    assert evidence['path'].read_bytes()==before
    summary=result['results'][0]['summary']
    assert summary['additionalTargetMm3']==0. and 0.<summary['additionalNonTargetMm3']<=200. and summary['withdrawn']
    assert summary['elapsedSeconds']<=180.


@pytest.mark.parametrize('change',['family','missing-family','legacy-schema','missing-path','duplicate-path','lane-axis',
    'local-offset','deeper-endpoint','proximal-endpoint','radius','width','wide-width','central-shadow','source-ray'])
def test_resealed_paired_family_path_offset_depth_radius_and_width_forgeries_refuse(evidence,change):
    selected,original=evidence['rows'][3.42];proof=copy.deepcopy(original);public=proof['publicPlan']
    attempt=public['attempts'][0];route=attempt['route'];axis=np.asarray(route['axisRasUnit'])
    if change=='family':public['family']='unbounded-lanes'
    elif change=='missing-family':public.pop('family')
    elif change=='legacy-schema':
        proof['schema']='resectionlab.source-exposure-preparation.v2';public['schema']='resectionlab.public-source-exposure-preparation.v2'
    elif change=='missing-path':route['paths'].pop()
    elif change=='duplicate-path':route['paths'][5]=copy.deepcopy(route['paths'][0])
    elif change=='lane-axis':route['paths'][5]['outsideMm'][0]+=3.42
    elif change=='local-offset':route['paths'][1]['outsideMm'][0]+=.01
    elif change=='deeper-endpoint':route['paths'][-1]['endpointMm']=(np.asarray(route['paths'][-1]['endpointMm'])+.1*axis).tolist()
    elif change=='proximal-endpoint':route['paths'][-1]['endpointMm']=(np.asarray(route['paths'][-1]['endpointMm'])-.1*axis).tolist()
    elif change=='radius':attempt['actionProgram'][0]['parameters']['radiusMm']=2.
    elif change=='width':public['controls']['lateralSweepMm']=3.
    elif change=='wide-width':public['controls']['lateralSweepMm']=4.00001
    elif change=='central-shadow':route['endpointMm'][2]+=.1
    else:public['sourceRay']['surfaceMm'][0]+=.5
    with pytest.raises(ValueError):evidence['run'](reseal(proof),selected.snapshot())


def test_width_zero_paired_lanes_cannot_be_downgraded_to_legacy_single_path(evidence):
    selected,original=evidence['rows'][0.];proof=copy.deepcopy(original)
    proof['schema']='resectionlab.source-exposure-preparation.v2'
    proof['publicPlan']['schema']='resectionlab.public-source-exposure-preparation.v2';proof['publicPlan'].pop('family')
    with pytest.raises(ValueError,match='path inventory'):evidence['run'](reseal(proof),selected.snapshot())


def test_coordinated_quaternion_roll_and_path_rotation_cannot_change_fixed_frame_y(evidence):
    selected,original=evidence['rows'][3.42];proof=copy.deepcopy(original)
    route=proof['publicPlan']['attempts'][0]['route'];old=Rotation.from_quat(route['quaternion'])
    rolled=old*Rotation.from_rotvec([0.,0.,np.pi/2]);center=np.asarray(route['outsideMm'])
    transform=rolled.as_matrix()@old.as_matrix().T
    route['quaternion']=rolled.as_quat().tolist()
    for path in route['paths']:
        for key in ('outsideMm','endpointMm'):
            path[key]=(center+transform@(np.asarray(path[key])-center)).tolist()
    route['outsideMm']=route['paths'][0]['outsideMm'][:];route['endpointMm']=route['paths'][0]['endpointMm'][:]
    assert np.allclose(rolled.as_matrix()[:,2],route['axisRasUnit'])
    with pytest.raises(ValueError,match='paired source frame roll differs'):
        evidence['run'](reseal(proof),selected.snapshot())


def test_executed_paired_prefix_refusal_has_no_installed_effects(evidence):
    source=evidence['source'];controls={**evidence['controls'],'lateralSweepMm':3.42,'maxActions':6}
    selected,proof=producer.prepare(source,controls,native_collection=evidence['native'].root,
        source_ray=evidence['ray'],family='paired-lanes-v1')
    assert selected is None and proof['publicPlan']['attempts'] and proof['publicPlan']['decision']['selectedIndex'] is None
    result=evidence['run'](proof,source.snapshot())
    assert result['passed'] and result['microsteps']>0 and not result['results'][0]['complete']


def test_tilted_held_program_clips_lateral_passes_before_central_depth(evidence):
    ray={'surfaceMm':[12.,13.,5.],'sourceTargetMm':[13.,13.,9.]}
    controls={**evidence['controls'],'lateralSweepMm':3.42}
    selected,proof=producer.prepare(evidence['source'],controls,native_collection=evidence['native'].root,
        source_ray=ray,family='paired-lanes-v1')
    assert proof['publicPlan']['attempts']
    route=proof['publicPlan']['attempts'][0]['route'];axis=np.asarray(route['axisRasUnit'])
    depths=[float((np.asarray(p['endpointMm'])-p['outsideMm'])@axis) for p in route['paths']]
    assert all(depth<=depths[0]+1e-8 for depth in depths) and any(depth<depths[0]-1e-6 for depth in depths)
    expected=selected.snapshot() if selected is not None else evidence['source'].snapshot()
    assert evidence['run'](proof,expected)['passed']


def test_independent_paired_checker_does_not_import_the_producer_compiler():
    source=(ROOT/'scripts/verify_source_exposure.py').read_text()
    assert 'from resectionlab.surgical_exposure import' not in source
    assert 'import surgical_exposure' not in source

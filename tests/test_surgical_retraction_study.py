"""Native branch/task binding for paired-tool desktop study evidence."""
import copy
import numpy as np
import pytest

from resectionlab.core import semantic_digest
from resectionlab.surgical_engine import SurgicalEngine
from resectionlab.surgical_scene import SurgicalScene
from resectionlab.surgical_physics_v2 import default_profile
from resectionlab.surgical_staged_planning import larger_task, source_zones, assess_task
from resectionlab.surgical_sector_planning import _sector_cells
from resectionlab.surgical_retraction_study import install_retraction_study, validate_retraction_study


def fixture():
    labels=np.zeros((20,20,20),np.uint16);labels[8:12,8:12,8:12]=1
    scene=SurgicalScene(labels,np.eye(4),[{'id':1,'name':'target','material':'tumor','target':True}],{},
        {'tissuePhysicsV2':default_profile()})
    engine=SurgicalEngine(scene)
    engine.step({'slot':'left','operation':'inspect','duration':.1})
    source=engine.fresh()
    route={'surfacePose':{'position':[10.,10.,15.]},'inwardRasUnit':[0.,0.,-1.],
        'sourceSceneId':scene.identity,'pathType':'oblique_mapped_brain','pathLengthMm':3.,
        'accessCompleteness':'unsupported_skull_access_brain_corridor'}
    task=larger_task(scene);zones,_=source_zones(scene);sectors,_=_sector_cells(scene,route)
    baseline={**source.metrics(),'removedCells':[],'nativeActions':0}
    outcome=assess_task(engine,task,zones,{},exact_replay=True,baseline=baseline,
        admitted_strokes=0,legacy_sectors=sectors)
    variant={'id':'bimanual-unassisted','assisted':False,'status':'failed_native_program',
        'failureReason':'generated no-access STOP','accessComplete':False,'workStages':[],'outcome':outcome,
        'nativeActions':1,'nativeSnapshot':engine.snapshot(),'exactReplay':True,
        'finalMetrics':engine.metrics(),'sampledMechanicalPeaks':{'brainMaxDisplacementMm':0},
        'accessEndpointMetrics':baseline,'publicCalls':1}
    record={'branches':{'main':source.snapshot(),variant['id']:engine.snapshot()},
        'timeline':{variant['id']:engine.history},'parents':{variant['id']:{'branchId':'main','position':0}}}
    receipt={'route':route,'task':task,'variants':[variant],'cost':{'physicalCalls':1},'wallSeconds':1.}
    return scene,record,receipt


def test_study_display_install_is_read_only_and_does_not_simulate(monkeypatch):
    scene,record,receipt=fixture();branches=copy.deepcopy(record['branches'])
    monkeypatch.setattr(SurgicalEngine,'step',lambda *_a,**_kw:pytest.fail('Display projection must not simulate'))
    report=install_retraction_study(scene,record,receipt,receipt_sha256='a'*64)
    assert record['branches']==branches
    assert not report['variants'][0]['outcome']['taskGateMet']
    assert validate_retraction_study(scene,record,report)


@pytest.mark.parametrize('change',[
    lambda r:r['variants'][0]['outcome'].update(wholeTargetRemovedMm3=1.),
    lambda r:r['variants'][0]['outcome'].update(taskGateMet=True),
    lambda r:r['variants'][0]['baseline'].update(removedCells=[0]),
    lambda r:r.update(sourceStateId='sha256:'+'0'*64),
    lambda r:r.update(physicalCalls=2),
    lambda r:r['variants'][0].update(assisted=True),
    lambda r:r['variants'][0].update(accessComplete=True),
    lambda r:r['variants'][0].update(status='completed_native_program'),
    lambda r:r['variants'][0]['finalMetrics'].update(brainMaxDisplacementMm=1.),
    lambda r:r['variants'][0]['finalMetrics'].update(contactCellSeconds=1.),
    lambda r:r['variants'][0]['finalMetrics'].update(elapsedSeconds=5.)])
def test_resealed_study_cannot_replace_native_effects_or_lineage(change):
    scene,record,receipt=fixture()
    report=install_retraction_study(scene,record,receipt,receipt_sha256='a'*64)
    change(report);report['comparisonSeal']=semantic_digest({k:v for k,v in report.items() if k!='comparisonSeal'})
    with pytest.raises(ValueError):validate_retraction_study(scene,record,report)

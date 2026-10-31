"""Paired physical-state budgets and non-mutating source-energy exports."""
from pathlib import Path
import runpy
import sys

import numpy as np
import pytest

from resectionlab import surgical_continuum as c,surgical_deformation_v2 as brain
from resectionlab.surgical_continuum_solve import PROPERTY,profile
from resectionlab.surgical_scene import SurgicalScene
from resectionlab.surgical_engine import SurgicalEngine
from resectionlab.surgical_physics_v2 import default_profile

scripts=Path(__file__).resolve().parents[1]/'scripts';sys.path.insert(0,str(scripts))
try:api=runpy.run_path(str(scripts/'qualify_continuum_history.py'))
finally:sys.path.remove(str(scripts))

LIMITS={'displacementMm':1e-9,'forceN':1e-10,'energyNmm':1e-10,'strainExposureMm3S':1e-9,
        'stressExposurePa2Mm3S':.001,'otherFloat':1e-9}


def test_physical_state_units_have_separate_disagreement_budgets():
    maximum={};a={'cornerDisplacementMm':[[.1,0.,0.]],'removedStoredEnergyNmm':.001,
        'exposure':{'strainSquaredVolumeTimeMm3Seconds':1.,'stressSquaredVolumeTimePa2Mm3Seconds':1e5},'removed':[1,2]}
    api['compare_state'](a,a,LIMITS,maximum)
    assert set(maximum)=={'displacementMm','energyNmm','strainExposureMm3S','stressExposurePa2Mm3S'}
    for bad in [dict(a,removed=[1,3]),dict(a,removed=[1]),dict(a,removedStoredEnergyNmm=.0011)]:
        with pytest.raises(ValueError):api['compare_state'](a,bad,LIMITS,{})
    with pytest.raises(ValueError,match='Nonfinite'):api['compare_state'](0.,float('nan'),LIMITS,{})
    with pytest.raises(ValueError,match='type'):api['compare_state'](False,0,LIMITS,{})


def test_observed_owned_removal_is_exported_once_and_tick_is_restored(tmp_path):
    p={'tissuePhysicsV2':default_profile(),c.PROPERTY:c.research_profile({'1':{
        'shearPa':1000.,'poissonRatio':.3,'relaxationShearPa':[500.],'source':'Generated sampler control'}},relaxation_seconds=[.5]),PROPERTY:profile()}
    scene=SurgicalScene(np.ones((13,13,13),np.uint16),np.eye(4),
        [{'id':1,'name':'Generated block','material':'soft_tissue'}],{'kind':'generated numerical control'},p)
    e=SurgicalEngine(scene);center=int(np.ravel_multi_index((6,6,6),scene.shape))
    removed=int(np.ravel_multi_index((5,6,6),scene.shape));events=[];samples=[];tick=c.tick
    with api['observe_history'](tmp_path,'generated',events,samples,12):
        brain.begin_hold(e,'left',[center],[6.,6.,6.]);brain.set_target(e,'left',[6.1,6.,6.]);brain.tick(e,.1)
        e.state['removed']=[removed];brain.tick(e,.1);brain.tick(e,.1)
    assert c.tick is tick and len(events)==1 and events[0]['countDelta']==1 and events[0]['energyDeltaNmm']>0
    assert len(samples)==2
    with np.load(tmp_path/samples[-1]['file']) as data:
        assert removed not in data['cells']
        np.testing.assert_array_equal(data['displacement'][data['elementNodes']],data['cellCorners'])
        np.testing.assert_array_equal(data['memories'][:,data['elementNodes']],data['cellMemories'])

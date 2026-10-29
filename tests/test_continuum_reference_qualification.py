"""Reference experiment completeness and transient native-clock observation."""
import copy
from pathlib import Path
import runpy
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from resectionlab import surgical_continuum as c, surgical_deformation_v2 as b

scripts=Path(__file__).resolve().parents[1]/'scripts'
sys.path.insert(0,str(scripts))
try: api=runpy.run_path(str(scripts/'qualify_continuum_reference.py'))
finally:sys.path.remove(str(scripts))


def experiment():
    limits={name:.05 for name in ['maximumReactionRelativeDifference','maximumFixedProbeStrainSquaredRelativeDifference',
        'maximumFixedProbeStressSquaredRelativeDifference','maximumFixedProbeCumulativeStrainSquaredRelativeDifference',
        'maximumFixedProbeCumulativeStressSquaredRelativeDifference']}
    limits.update(lastTwoRadiiMm=[16.,20.],maximumCenterDisplacementRmsDifferenceMm=.005)
    protocol={'branches':['short','long'],'referenceRadiiMm':[8.,12.,16.,20.],'stability':limits}
    rows=[{'branchId':branch,'radiusMm':radius,'status':'completed','commonSourceIdsHash':'same source',
        'loadHistoryHash':branch,'probeCenterDisplacementMm':[[.1,0.,0.],[.01,0.,0.]],
        'reactionNormN':.001,'fixedProbeStrainSquaredMm3':.1,'fixedProbeStressSquaredPa2Mm3':1.,
        'fixedProbeCumulativeStrainSquaredMm3S':.2,'fixedProbeCumulativeStressSquaredPa2Mm3S':2.}
        for branch in protocol['branches'] for radius in protocol['referenceRadiiMm']]
    return rows,protocol


def test_complete_equal_histories_are_stable_and_missing_worlds_refuse():
    rows,p=experiment(); assert all(row['radiusStable'] for row in api['stability'](rows,p))
    for bad in [rows[:-1],rows+[copy.deepcopy(rows[0])]]:
        with pytest.raises(ValueError,match='Every declared'):api['stability'](bad,p)


@pytest.mark.parametrize('index',range(4))
def test_every_declared_failed_world_stays_negative(index):
    rows,p=experiment(); rows[index]={'branchId':'short','radiusMm':rows[index]['radiusMm'],'status':'refused'}
    result=api['stability'](rows,p)
    assert result[0]=={'branchId':'short','status':'not_estimable','radiusStable':False}
    assert result[1]['radiusStable']


@pytest.mark.parametrize('index',range(4))
def test_invalid_magnitudes_cannot_hide_in_noncompared_worlds(index):
    rows,p=experiment();rows[index]['reactionNormN']=-1.
    with pytest.raises(ValueError,match='nonnegative'):api['stability'](rows,p)


def test_probe_shapes_do_not_broadcast_and_unknown_status_refuses():
    rows,p=experiment();rows[0]['probeCenterDisplacementMm']=[[.1,0.,0.]]
    with pytest.raises(ValueError,match='source-center'):api['stability'](rows,p)
    rows,p=experiment();rows[0]['status']='started'
    with pytest.raises(ValueError,match='terminal'):api['stability'](rows,p)


class ClockFixture:
    def __init__(self,fraction=1.):
        settings=b.profile();settings['fixedStepSeconds']=.03
        self.scene=SimpleNamespace(properties={'tissuePhysicsV2':{'brain':settings}})
        self.state={'removed':[],'deformationModel':{'remainderSeconds':.02},'holds':{
            'left':{'cells':[0],'target':[.1,0.,0.],'origin':[0.,0.,0.]}},'ticks':0}
        self.fraction=fraction;self.state_id='generated clock control'
    def clone(self):return copy.deepcopy(self)
    def snapshot(self):return copy.deepcopy(self.state)
    def step(self,action,check):
        check();c.tick(self,.06)
        return {'status':'accepted','reason':None,'completedFraction':self.fraction}


def fixture_tick(current,dt,diagnostic=None):
    total=current.state['deformationModel']['remainderSeconds']+dt
    steps=int(np.floor((total+1e-12)/.03));current.state['ticks']+=steps
    current.state['deformationModel']['remainderSeconds']=round(total-steps*.03,14)


def test_transient_observer_uses_remainder_and_restores_original_tick(monkeypatch):
    monkeypatch.setattr(c,'tick',fixture_tick);base=ClockFixture();expected=base.clone();expected.step({},lambda:None)
    before=base.snapshot();target,h,_,order=api['capture_targets'](base,[{}],expected.snapshot(),check=lambda:None)
    np.testing.assert_array_equal(target,[[.1,0.,0.],[.1,0.,0.]])
    assert h==.03 and order==(('left',0),) and base.snapshot()==before and c.tick is fixture_tick


def test_fractional_accepted_probe_is_invalid_and_observer_unwinds(monkeypatch):
    monkeypatch.setattr(c,'tick',fixture_tick);base=ClockFixture(.5)
    with pytest.raises(ValueError,match='complete accepted'):
        api['capture_targets'](base,[{}],{},check=lambda:None)
    assert c.tick is fixture_tick

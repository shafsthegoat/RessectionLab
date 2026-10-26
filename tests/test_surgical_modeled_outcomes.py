"""Units, spatial source geometry, and unavailable modeled endpoints."""
import numpy as np
import pytest

from resectionlab.surgical_engine import SurgicalEngine
from resectionlab.surgical_modeled_outcomes import (
    _target_distribution, summarize_engine_outcomes,
)
from resectionlab.surgical_physics_v2 import default_profile
from resectionlab.surgical_scene import SurgicalScene


def _scene(*, physics=False, include_bone=False, include_vessel=False):
    labels=np.zeros((3,3,3),np.uint16)
    labels[1,1,1]=1
    for x in (0,2):
        for y in (0,2):
            for z in (0,2):
                labels[x,y,z]=2
    regions=[{'id':1,'name':'source brain','material':'soft_tissue','target':False},
             {'id':2,'name':'source target','material':'tumor','target':True}]
    if include_bone:
        labels[0,1,1]=3
        regions.append({'id':3,'name':'source bone','material':'bone','target':False})
    if include_vessel:
        labels[2,1,1]=4
        regions.append({'id':4,'name':'source vessel','material':'vessel','target':False})
    affine=np.diag([2.,2.,2.,1.])
    return SurgicalScene(labels,affine,regions,{'caseHash':'source-case'},
        {'tissuePhysicsV2':default_profile()} if physics else {})


def test_target_octants_use_ras_millimeters_and_exact_voxel_volume():
    scene=_scene()
    removed={int(np.ravel_multi_index((2,2,2),scene.shape))}
    summary=_target_distribution(scene,(2,),removed,set())
    assert summary['sourceVolumeMm3']==pytest.approx(64.)
    assert summary['removedMm3']==pytest.approx(8.)
    assert summary['residualMm3']==pytest.approx(56.)
    assert summary['sourceCentroidRasMm']==[2.,2.,2.]
    assert summary['removedCentroidRasMm']==[4.,4.,4.]
    octant=next(row for row in summary['octants'] if row['region']=='R-A-S')
    assert octant['sourceMm3']==pytest.approx(8.)
    assert octant['removedMm3']==pytest.approx(8.)
    assert octant['incrementalRemovedMm3']==pytest.approx(8.)
    assert octant['residualMm3']==pytest.approx(0.)
    assert sum(row['residualMm3'] for row in summary['octants'])==pytest.approx(56.)


def test_unmapped_vessels_skull_and_protection_are_unavailable_not_zero():
    engine=SurgicalEngine(_scene(physics=True))
    result=summarize_engine_outcomes(engine)
    assert result['frame']=='RAS+' and result['voxelVolumeMm3']==pytest.approx(8.)
    assert result['target']['sourceVolumeMm3']==pytest.approx(64.)
    assert result['target']['residualMm3']==pytest.approx(64.)
    assert next(row for row in result['sourceRegions'] if row['target'])['origin']=='unclassified_source_layer'
    assert result['nonTargetByLabel'][0]['sourceMm3']==pytest.approx(8.)
    assert result['contact']['workingToolCellSecondsByMaterial']['soft_tissue']['endpoint']=={
        'status':'measured','value':0.,'unit':'cell*s'}
    assert result['mechanics']['deformation']['brainMaxDisplacementMm']['unit']=='mm'
    assert result['mechanics']['drillWork']['status']=='unavailable'
    assert result['mechanics']['drillWork']['value'] is None
    assert result['localVessels']['status']=='unavailable'
    assert result['localVessels']['value'] is None
    assert result['protection']['status']=='unavailable'
    assert result['globalPerfusion']['status']=='unknown'
    assert result['neurologicalRisk']['status']=='unknown'


def test_mapped_bone_and_vessels_expose_only_local_model_ledgers():
    engine=SurgicalEngine(_scene(physics=True,include_bone=True,include_vessel=True))
    result=summarize_engine_outcomes(engine)
    assert result['mechanics']['drillWork']['status']=='measured_model_only'
    assert result['mechanics']['drillWork']['drillInputEnergyJ']['unit']=='J'
    assert result['localVessels']['status']=='measured_mapped_local_model_only'
    assert result['localVessels']['woundAreaMm2']['unit']=='mm2'
    assert result['globalPerfusion']['value'] is None


def test_empty_configured_protection_still_has_unavailable_coverage(monkeypatch):
    engine=SurgicalEngine(_scene(physics=True))
    # A permissive guard report can say configured while declaring no actual
    # protected source cell; the outcome summary must not mark coverage known.
    monkeypatch.setattr(engine,'guard_report',lambda:{'configured':True,
        'missingRequired':[],'structures':[],'contractId':'empty-contract',
        'unknownStructures':['critical anatomy unprovided']})
    outcome=summarize_engine_outcomes(engine)
    assert outcome['protection']['status']=='unavailable'
    assert outcome['protection']['value'] is None
    assert outcome['protection']['declaredProtectedSourceCells']==0


def test_baseline_must_be_exact_native_action_prefix():
    engine=SurgicalEngine(_scene())
    baseline=engine.fresh()
    assert engine.step({'slot':'left','operation':'inspect','duration':.2})['status']=='accepted'
    result=summarize_engine_outcomes(engine,baseline=baseline)
    assert result['mechanics']['elapsedSeconds']['delta']=={
        'status':'measured','value':.2,'unit':'s'}
    other=engine.fresh()
    assert other.step({'slot':'left','operation':'inspect','duration':.4})['status']=='accepted'
    with pytest.raises(ValueError,match='exact native action prefix'):
        summarize_engine_outcomes(engine,baseline=other)

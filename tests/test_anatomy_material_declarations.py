from dataclasses import replace
import numpy as np
import pytest
from resectionlab.core import CaseData,SourceRef
from resectionlab.surgical_scene import from_case


def fixture(metadata):
    fluid=np.zeros((8,8,8),bool);fluid[2:4,2:4,2:4]=True
    tumor=np.zeros_like(fluid);tumor[5:7,5:7,5:7]=True
    return CaseData('generated-material-control',np.arange(512,dtype=np.float32).reshape((8,)*3),{'csf':fluid,'target':tumor},np.eye(4),
        (SourceRef('generated','synthetic://material-control',provenance='simulated'),),metadata=metadata)


def test_explicit_anatomical_materials_preserve_csf_fluid_and_target_roles():
    case=fixture({'anatomy_layers':{'csf':{'kind':'context'},'target':{'kind':'tumor'}},
        'surgical_adapter':{'materialByRegion':{'csf':'fluid','target':'tumor'}}})
    scene=from_case(case);regions={r['name']:r for r in scene.regions}
    assert regions['csf']['material']=='fluid' and not regions['csf']['target']
    assert regions['target']['material']=='tumor' and regions['target']['target']
    assert np.array_equal(scene.labels==regions['csf']['id'],case.compartments['csf'])


def test_partial_or_unknown_material_declarations_refuse_and_legacy_defaults_remain():
    metadata={'anatomy_layers':{'csf':{'kind':'context'},'target':{'kind':'tumor'}}}
    case=fixture(metadata)
    assert {r['name']:r['material']for r in from_case(case).regions}=={'csf':'unknown','target':'tumor'}
    for declaration in [{'csf':'fluid'},{'csf':'water','target':'tumor'}]:
        with pytest.raises(ValueError,match='every source region'):
            from_case(replace(case,metadata={**metadata,'surgical_adapter':{'materialByRegion':declaration}}))

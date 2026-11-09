"""Exact prior catalog compatibility; new RL capability is explicitly versioned."""
import copy
from types import SimpleNamespace
import pytest
from resectionlab.core import semantic_digest
from resectionlab import surgical_task_admission as admission
from resectionlab.surgical_tools import catalog

OLD16=('cranial_drill','cusa','brain_retractor','scalpel','periosteal_elevator','craniotome','dural_scissors','dissector','frazier_suction','bipolar_forceps','micro_scissors','tumor_forceps','bone_flap_holder','irrigation','cortical_probe','subcortical_probe')


def test_membrane_tool_appends_without_reordering_or_resizing_existing_tools():
    tools=catalog();assert tuple(x['id'] for x in tools[:-1])==OLD16
    membrane=tools[-1];assert membrane['id']=='membrane_forceps' and membrane['materials']==['membrane']
    assert (membrane['radiusMm'],membrane['shaftRadiusMm'],membrane['workingLengthMm'])==(1.5,.5,40.)
    assert next(x for x in tools if x['id']=='cusa')['materials']==['soft_tissue','tumor']
    assert next(x for x in tools if x['id']=='tumor_forceps')['materials']==['tumor']


@pytest.mark.parametrize('exclude',[{'membrane_forceps'},{'membrane_forceps','bone_flap_holder'}])
def test_exact_historical_16_and15_catalog_models_still_inspectable(exclude):
    engine=SimpleNamespace(scene=SimpleNamespace(properties={}))
    old=dict(admission._models(engine),instrumentCatalogHash=semantic_digest([x for x in catalog() if x['id'] not in exclude]))
    assert admission._compatible_models(engine,old)
    for field,value in [('physicalClockRevision',0),('toolMechanicsRevision',1),('physicsHash','other')]:
        assert not admission._compatible_models(engine,dict(old,**{field:value}))


def test_omitting_arbitrary_tool_or_altering_existing_geometry_is_not_historical(monkeypatch):
    engine=SimpleNamespace(scene=SimpleNamespace(properties={}))
    phantom=dict(admission._models(engine),instrumentCatalogHash=semantic_digest([x for x in catalog() if x['id']!='bone_flap_holder']))
    assert not admission._compatible_models(engine,phantom)
    old=dict(admission._models(engine),instrumentCatalogHash=semantic_digest([x for x in catalog() if x['id']!='membrane_forceps']))
    changed=copy.deepcopy(catalog());next(x for x in changed if x['id']=='cusa')['radiusMm']=1.6
    monkeypatch.setattr(admission,'catalog',lambda:changed)
    assert not admission._compatible_models(engine,old)


def test_rl_catalog17_refuses_old_action_contract_instead_of_resizing_vectors():
    pytest.importorskip('gymnasium')
    from test_surgical_rl_contract import env
    e=env();e.reset(seed=1);record=e.snapshot();before=e.engine.snapshot()
    assert record['actionContract']['id']=='catalog-radius-fixed-while-inserted.v2-membrane17'
    record['actionContract']['id']='catalog-radius-fixed-while-inserted.v1'
    record['actionContract']['radiusCapsMm'].pop('membrane_forceps')
    with pytest.raises(ValueError,match='action contract'):e.restore(record)
    assert e.engine.snapshot()==before and not e.attempts

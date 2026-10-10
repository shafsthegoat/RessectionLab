"""Small opt-in contract/leaf-loss controls; no native task or optimizer step."""
import ast
from pathlib import Path
from types import SimpleNamespace as NS
import pytest
import torch
from resectionlab.contact_learning_contract import (freeze_contact_experiment,
    freeze_full_teacher_refit, FULL_TEACHER_VERSION, PROTOCOL, common_initial_policies)
from resectionlab.contact_learning import ContactLearningSession
from resectionlab.contact_checkpoint import encode_contact_checkpoint, decode_contact_checkpoint, initial_lineage
from resectionlab.core import semantic_digest
from resectionlab.spatial_policy import parameter_hash
import resectionlab.contact_learning as learning_source


def states():
    experiment=freeze_contact_experiment(); result=[]
    for i,(layout,goal) in enumerate(experiment.keys('TRAIN')):
        for step in range(1 if i<8 else 2):
            result.append({'layout_id':layout,'goal_id':goal,'step':step,
                'observation_hash':semantic_digest(['unit_observation',i,step]),
                'action_id':'STOP' if i<8 else 'unit-movement-'+str(step),
                'original_binding_hash':semantic_digest(['unit_binding',i]),
                'strategy_seal':semantic_digest(['unit_strategy',i])})
    return result


def test_default_identity_is_exact_and_refit_is_separate():
    baseline=freeze_contact_experiment()
    assert baseline.fingerprint=='sha256:a01e6ba30fbf1e6efabf70f9637d42c0813931624cb50d2d393858029d4fd894'
    assert baseline.protocol is PROTOCOL and 'teacher_states' not in baseline.record()
    refit=freeze_full_teacher_refit(states())
    assert refit.fingerprint != baseline.fingerprint
    assert refit.record()['version']==FULL_TEACHER_VERSION
    assert refit.protocol['batch_size']==40 and refit.protocol['methods']==('IL',)
    assert refit.protocol['updates']==32 and refit.protocol['loss_forward_calls']==1280


def test_refit_keeps_negative_stop_and_refuses_subset_or_role_substitution():
    rows=states()
    with pytest.raises(ValueError):freeze_full_teacher_refit(rows[:-1])
    with pytest.raises(ValueError):freeze_full_teacher_refit(list(reversed(rows)))
    altered=[dict(r) for r in rows];altered[0]['layout_id']=freeze_contact_experiment().keys('SELECT')[0][0]
    with pytest.raises(ValueError):freeze_full_teacher_refit(altered)
    altered=[dict(r) for r in rows];altered[0]['action_id']='unit-movement'
    with pytest.raises(ValueError):freeze_full_teacher_refit(altered)


def test_refit_initialization_matches_but_checkpoint_does_not_alias_pilot():
    refit=freeze_full_teacher_refit(states()); models=common_initial_policies(refit)
    assert set(models)=={'IL'}
    model=models['IL']; baseline=freeze_contact_experiment()
    assert parameter_hash(model)=='sha256:e7215950e221045b8e9612e4482837f9b1ff2c52278454b26efc746598cf1487'
    with pytest.raises(ValueError):ContactLearningSession(refit,'RL',model,parameter_hash(model))
    payload=encode_contact_checkpoint(model,refit,initial_lineage(refit))
    import hashlib
    digest=hashlib.sha256(payload).hexdigest()
    restored,metadata=decode_contact_checkpoint(payload,expected_sha256=digest,experiment=refit,kind='initial')
    assert parameter_hash(restored)==parameter_hash(model)
    assert metadata['lineage']['learning_contract_version']==FULL_TEACHER_VERSION
    with pytest.raises(ValueError):decode_contact_checkpoint(payload,expected_sha256=digest,experiment=baseline,kind='initial')


def test_all40_loss_is_unchanged_mean_ce_and_preserves_stop_gradient():
    # Extract the executing loss, replacing only admission/model with leaf tensors.
    node=next(n for n in ast.parse(Path(learning_source.__file__).read_text()).body
              if isinstance(n,ast.FunctionDef) and n.name=='contact_imitation_loss')
    namespace={'torch':torch,'_session':lambda session,method:session.policy}
    exec(compile(ast.Module(body=[node],type_ignores=[]),'<canonical-loss>','exec'),namespace)
    rows=states(); leaves=torch.zeros((40,2),requires_grad=True)
    samples=[]
    for index,row in enumerate(rows):
        obs=NS(fingerprint=row['observation_hash'],base=NS(base=NS(state_features=[row['step']])),index=index)
        action_index=0 if row['action_id']=='STOP' else 1
        samples.append(NS(binding=NS(layout_id=row['layout_id'],goal_id=row['goal_id'],context=None),
            observation=obs,action_id=row['action_id'],validate=lambda chosen=action_index:chosen))
    session=NS(experiment=NS(protocol={'batch_size':40},teacher_states=rows),
        require_samples=lambda _:None,register_loss=lambda *_:None,
        policy=lambda obs,context:(leaves[obs.index],leaves.new_tensor(0.)))
    loss,receipt=namespace['contact_imitation_loss'](session,samples)
    targets=torch.tensor([0 if r['action_id']=='STOP' else 1 for r in rows])
    expected=torch.nn.functional.cross_entropy(leaves,targets)
    torch.testing.assert_close(loss,expected)
    assert receipt['supervised_actions']==40
    gradient=torch.autograd.grad(loss,leaves)[0]
    assert torch.allclose(gradient,torch.autograd.grad(expected,leaves)[0])
    assert (gradient[:8,0]<0).all() and (gradient[8:,0]>0).all()
    with pytest.raises(ValueError):namespace['contact_imitation_loss'](session,samples[:-1])
    duplicated=list(samples);duplicated[-1]=samples[0]
    with pytest.raises(ValueError):namespace['contact_imitation_loss'](session,duplicated)


def test_default_desktop_source_closure_still_admits_reviewed_sources():
    from resectionlab.contact_family_desktop_bridge import _controller_ready
    assert _controller_ready()  # Hash checks only; no child, model, task or release-result read.

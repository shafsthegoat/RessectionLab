from pathlib import Path
import sys
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from export_sequential_episodes import _frozen_model,verify_recorded_episode,original_inputs
from resectionlab.surgical_sequential_task import generated_task


def test_wrong_checkpoint_bytes_refused_before_policy_loading(tmp_path):
    path=tmp_path/'checkpoint.pt';path.write_bytes(b'wrong checkpoint')
    freeze={'checkpoints':[dict(seed=17,path=str(path),sha256='0'*64,rewardDrivenUpdates=True,selectedUpdates=192)]}
    with pytest.raises(ValueError,match='CHECKPOINT_CHANGED'):
        _frozen_model(freeze,{'method':'learned','seed':17})


def test_learned_label_cannot_substitute_other_recorded_skill_choice():
    task=generated_task(200);first=next(s for s in task.skills if s.kind=='prepare' and s.radius==.4)
    row=dict(parent=200,sourceSceneId=task.scene.identity,configHash=task.config.fingerprint,
        snapshot=task.initial_engine().snapshot(),skillHistory=[dict(skillId=first.id)])
    class WrongCheckpoint:
        def select(self,features,mask,stochastic):return next(i for i,s in enumerate(task.skills) if s.kind=='stop'),{}
    with pytest.raises(ValueError,match='CHOICE_DIFFERS_FROM_CHECKPOINT'):
        verify_recorded_episode(task,row,model=WrongCheckpoint())


def test_edited_original_experiment_cannot_be_resealed_as_original(tmp_path):
    (tmp_path/'protocol.json').write_text('{}')
    with pytest.raises(ValueError,match='ORIGINAL_COMMITTED_EXPERIMENT_BYTES_CHANGED'):
        original_inputs(tmp_path)

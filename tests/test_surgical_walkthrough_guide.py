from types import SimpleNamespace

from resectionlab.surgical_walkthrough_guide import SCHEMA, next_cue


def test_cue_requires_matching_source_branch_and_physical_boundary():
    action = {'operation':'apply', 'slot':'left', 'duration':.1}
    guide = {'schema':SCHEMA, 'caseHash':'case', 'sceneId':'scene', 'branchId':'branch',
             'steps':[{'before':'a','after':'b','action':action,'phase':'scalp','label':'Engage blade'}],
             'endpoint':'b'}
    engine = SimpleNamespace(_operative=True, scene=SimpleNamespace(identity='scene'), actions=[], state_id='a')
    assert next_cue(guide, engine, 'branch', 'case')['action'] == action
    assert next_cue(guide, engine, 'other-branch', 'case') is None
    assert next_cue(guide, engine, 'branch', 'other-case') is None
    engine.state_id = 'changed-before'
    assert next_cue(guide, engine, 'branch', 'case')['status'] == 'modified'
    engine.actions.append(action)
    engine.state_id = 'b'
    assert next_cue(guide, engine, 'branch', 'case')['status'] == 'complete'
    engine.state_id = 'altered-result'
    assert next_cue(guide, engine, 'branch', 'case')['status'] == 'modified'
    engine._operative = False
    assert next_cue(guide, engine, 'branch', 'case') is None

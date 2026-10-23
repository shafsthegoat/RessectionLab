"""Generated admission-barrier checks; no patient load or physical transition."""
from types import SimpleNamespace

import numpy as np
import pytest

from resectionlab.surgical_rl_env import INSTRUMENTS, PARAMETERS, SurgicalRLEnv
from resectionlab.surgical_skills import propose_skill
from resectionlab.surgical_tools import TOOLS


def entry(*, inserted=True):
    slot=dict(instrument='micro_scissors',active=False,inserted=inserted,grip=[])
    return SimpleNamespace(engine=SimpleNamespace(state={'slots':{'left':slot},'holds':{}},
        parameters=lambda _: {'radiusMm':1.5}),
        _radius_caps={key:float(value['radiusMm']) for key,value in TOOLS.items()})


def test_inherited_oversized_scissors_cannot_escape_raw_admission_via_control_or_resize():
    env=entry()
    for operation in ('deactivate','withdraw'):
        with pytest.raises(ValueError,match='INSTRUMENT_RADIUS_EXCEEDS_CATALOG'):
            SurgicalRLEnv._validate_action_radius(env,dict(slot='left',operation=operation))
    with pytest.raises(ValueError,match='WITHDRAW_BEFORE_RADIUS_CHANGE'):
        SurgicalRLEnv._validate_action_radius(env,dict(slot='left',operation='withdraw',parameters={'radiusMm':1.}))


def test_withdrawn_task_setup_can_exchange_to_catalog_tip_without_changing_global_caps():
    env=entry(inserted=False)
    original=env._radius_caps.copy()
    SurgicalRLEnv._validate_action_radius(env,dict(slot='left',operation='exchange',instrument='bipolar_forceps'))
    assert env._radius_caps==original
    assert TOOLS['micro_scissors']['radiusMm']==1.


def test_published_skill_proposal_refuses_inherited_oversize_before_any_execution():
    slots=np.zeros((2,12));slots[:,7]=INSTRUMENTS.index('micro_scissors');slots[0,9]=1
    parameters=np.zeros((2,len(PARAMETERS)));parameters[:,0]=1.5
    observation=dict(local=np.zeros((2,8,9,9,9)),slots=slots,tool_parameters=parameters,
        metrics=np.zeros(14),goal=np.zeros(4),coverage=np.zeros(3),local_spacing_mm=np.ones(1),
        cleanup=np.zeros(3),budgets=np.ones(3))
    with pytest.raises(ValueError,match='Skill radius must fit'):
        propose_skill(observation,'withdraw')

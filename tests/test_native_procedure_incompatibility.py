"""Historical geometric policies cannot claim the procedure harm contract."""
import pytest
from types import SimpleNamespace

from resectionlab.native_spatial_task import (NativeSpatialTask,
    make_native_opening_task, native_spatial_task_from_case)


@pytest.mark.parametrize('policy', [{}, {'policyId': 'declared-supratentorial-task'}, False])
def test_explicit_procedure_task_is_refused_before_loading_or_native_execution(policy):
    for make in (
        lambda: NativeSpatialTask(None, procedure_policy=policy),
        lambda: native_spatial_task_from_case(None, access=None, tools=(), procedure_policy=policy),
        lambda: make_native_opening_task(procedure_policy=policy),
    ):
        with pytest.raises(ValueError, match='PROCEDURE_POLICY_INCOMPATIBLE'):
            make()


def test_historical_generated_native_fixture_remains_available():
    task = make_native_opening_task()
    assert task.supported_procedure_policy is False
    assert task.observation().action_ids


@pytest.mark.parametrize('binding',[None,False,{}])
def test_patient_metadata_binding_cannot_omit_policy_keyword_to_bypass_guard(binding):
    case = SimpleNamespace(metadata={'surgical_properties': {'surgicalProcedurePolicyV1': binding}})
    with pytest.raises(ValueError, match='PROCEDURE_POLICY_INCOMPATIBLE'):
        native_spatial_task_from_case(case, access=None, tools=())

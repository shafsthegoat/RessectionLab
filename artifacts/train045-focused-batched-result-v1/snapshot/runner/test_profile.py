"""Active-budget same-catalog profile and honest diagnostic closure controls."""
import importlib.util,json,os,sys
from pathlib import Path
import pytest
HERE=Path(__file__).resolve().parent;ROOT=next(p for p in HERE.parents if (p/'src/resectionlab').is_dir())
sys.path[:0]=[str(HERE),str(HERE.parent/'model'),str(ROOT/'src')]
from catalog_profile import profile_catalog,timing_gate
from plan_transport import close_diagnostic_prefix
from comparison import plain
from common import write_json
from compact_output import encode_gzip,decode_gzip
from resectionlab.native_resection import NativeResectionEngine
from resectionlab.native_spatial_task import make_native_opening_task
from resectionlab.planning_budget import PlanningBudget
from resectionlab.patient_planning_cohort_io import OutputBudget


def model_fixture():
    path=HERE.parent/'model'/'test_batched_authentication.py'
    spec=importlib.util.spec_from_file_location('_profile_generated_catalog',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def test_active_budget_and_cancellation_callback_preserve_wrapper_and_zero_native_work():
    f=model_fixture();catalog,_,_=f.generated_catalog()
    budget=PlanningBudget(NativeResectionEngine,max_native_previews=1000,seconds=20)
    with budget:
        def checkpoint():budget.check();return False
        for page in (catalog.work,*catalog.pages):page._cancelled=checkpoint
        wrapper=NativeResectionEngine.preview_stroke
        result=profile_catalog(catalog,validate=f.adapter.validate_catalog,guard=checkpoint,
            counters=lambda:{'previews':budget.snapshot()['native_preview_entries']})
        assert NativeResectionEngine.preview_stroke is wrapper
        assert result['integrity_passed'] and len(result['pairs'])==4
        assert budget.snapshot()['native_preview_entries']==0
        for row in result['pairs']:
            for mode in ('reference','batched'):
                assert row['timings'][mode]['calls']['certificate']==len(catalog.identifiers)-1
                assert row['timings'][mode]['calls']['score']==len(catalog.identifiers)-1
        # Genuine terminal history retained to close this control's budget.
        page,_,transition=catalog.commit(0);assert transition.terminated
        write_json(Path(os.environ['TMPDIR'])/'profile-control-terminal.json',plain(page.metrics()['history']))
        budget.complete(history_complete=True)
    assert budget.snapshot()['counting_reliable']


def test_predeclared_two_part_gate_and_profile_only_real_STOP_closure():
    def rows(r,b):return [{'timings':{'reference':{'wall_seconds':r},'batched':{'wall_seconds':b}}} for _ in range(4)]
    assert timing_gate(rows(2.,1.))['material_gate_passed']
    assert not timing_gate(rows(.2,.1))['material_gate_passed']  # relative gain alone insufficient
    assert not timing_gate(rows(4.,3.5))['material_gate_passed']  # absolute gain alone insufficient
    out=Path(os.environ['TMPDIR'])/'profile-negative-closure';out.mkdir();sink=OutputBudget(out,1024**2)
    budget=PlanningBudget(NativeResectionEngine,max_native_previews=1000,seconds=20)
    with budget:
        task=make_native_opening_task(max_steps=3);task.step(next(iter(task._prepare_inventory())))
        def save(history):
            raw,binding=encode_gzip(plain(history));sink.write_bytes(out/'history.json.gz',raw)
            return {'path':'history.json.gz',**binding}
        result=close_diagnostic_prefix(task,expected_steps=1,save_history=save)
        assert task.terminated and task._steps==2 and result['additional_transitions']==1
        assert decode_gzip((out/result['path']).read_bytes(),result)[-1]['action_id']=='STOP'
        budget.complete(history_complete=True)
    assert result['scope'].startswith('Charged diagnostic cleanup')

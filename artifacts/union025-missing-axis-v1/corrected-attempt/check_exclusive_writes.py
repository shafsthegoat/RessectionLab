"""Generated write-flow control using exact nested worker code and real OutputBudget.

Task/geometry/audit doubles test serialization flow only. No native/task engine,
model, optimizer, patient array or checkpoint is constructed or called.
"""
import ast
import copy
import hashlib
import json
from pathlib import Path
import signal
import sys
import tempfile
import time
from types import SimpleNamespace
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(HERE))
from resectionlab.patient_planning_cohort_io import OutputBudget
from axis_contract import canonical,semantic,same_ray,sha

def functions(path):
    tree=ast.parse(path.read_text())
    world=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='run_world')
    examine=next(n for n in world.body if isinstance(n,ast.FunctionDef) and n.name=='examine')
    history=next(n for n in world.body if isinstance(n,ast.FunctionDef) and n.name=='history')
    start=next(i for i,n in enumerate(world.body) if isinstance(n,ast.Assign)
        and any(isinstance(t,ast.Name) and t.id=='first' for t in n.targets))
    flow=ast.FunctionDef(name='flow',args=ast.arguments(posonlyargs=[],args=[],kwonlyargs=[],kw_defaults=[],defaults=[]),
        body=world.body[start:],decorator_list=[])
    return compile(ast.fix_missing_locations(ast.Module(body=[examine,history,flow],type_ignores=[])),str(path),'exec')

class FakeTask:
    calls=0
    def __init__(self,availability):
        self.availability=availability;self.rows=[];self.terminated=False
        self._engine=SimpleNamespace(state_hash='generated-state-0');self.decision_model_hash='generated-model'
    def clone(self):return copy.deepcopy(self)
    def candidate_inventory(self):
        stage='clearance' if not self.rows else 'target'
        action='prep' if stage=='clearance' else 'ray'
        rows=[] if not self.availability[stage] else [{**RAYS[stage],'action_id':action,'feasible':True}]
        return {'emitted':rows,'omitted_count':0}
    def observation(self):return SimpleNamespace(action_ids=('STOP',*[r['action_id'] for r in self.candidate_inventory()['emitted']]))
    def step(self,action):
        if self.terminated:raise AssertionError('Fake flow executed afterSTOP')
        FakeTask.calls+=1
        row={'action_id':action,'reward':0. if action=='STOP' else .25}
        if action!='STOP':row.update(RAYS['clearance' if action=='prep' else 'target'])
        self.rows.append(row);self.terminated=action=='STOP'
        if action!='STOP':self._engine.state_hash='generated-state-'+str(len(self.rows))
        return SimpleNamespace(info=dict(row))
    def metrics(self):return {'history':copy.deepcopy(self.rows),'terminated':self.terminated}

RAYS={'clearance':{'tool_id':'generated-tool','entry_mm':[0.,0.,0.],'tip_mm':[1.,0.,0.]},
      'target':{'tool_id':'generated-tool','entry_mm':[0.,0.,0.],'tip_mm':[2.,0.,0.]}}

def run_flow(path,feasible,availability,*,old_failure=False):
    with tempfile.TemporaryDirectory(prefix='exclusive-generated-',dir=HERE) as tmp:
        output=Path(tmp);sink=OutputBudget(output,2*1024**2);writes=[];diagnostics={}
        def write(p,v):writes.append(str(p.relative_to(output)));sink.write(p,v)
        def diagnostic(task,dest,ray):
            name=dest.name
            record={**ray,'feasible':feasible[name],'reason':'GENERATED_FLOW_ONLY','committed':False}
            write(dest/'fixed-ray-preview.json',record);return record
        def audit(task,**unused):
            return {'accepted':True,'outcomes':{'total_reward':sum(r['reward'] for r in task.rows)},'target_access_success':False}
        base=FakeTask(availability);FakeTask.calls=0;result={'histories':[]}
        scope={'output':output,'write':write,'diagnostic':diagnostic,'diagnostics':diagnostics,
            'same_ray':same_ray,'canonical':canonical,'semantic':semantic,'base':base,
            'context':SimpleNamespace(require_task=lambda task:None,fingerprint='generated-context'),
            'counters':{'phase':'setup'},'guard':sink.check,'source':SimpleNamespace(source_hash='generated-source'),
            'obs':SimpleNamespace(fingerprint='generated-observation'),'evaluate_native_spatial_episode':audit,
            'initial_state':'generated-state-0','result':result,'public':{'full_target_mm3':1.},
            'inv':{'omitted_count':0},'clearance_ray':RAYS['clearance'],'target_ray':RAYS['target']}
        exec(functions(path),scope)
        if old_failure:
            try:scope['flow']()
            except FileExistsError:pass
            else:raise AssertionError('Old double-write defect not reproduced')
            assert writes.count('diagnostics.json')==2 and len(result['histories'])==1
            assert (output/'target/fixed-ray-preview.json').is_file()
            return {'old_second_examine_refused':True,'complete_histories_before_failure':1}
        scope['flow']()
        n=1+int(feasible['clearance'] and availability['clearance'])
        if n==2:n+=int(feasible['target'] and availability['target'])
        assert len(result['histories'])==n and writes.count('diagnostics.json')==1
        assert len(json.loads((output/'diagnostics.json').read_text()))==1+int(n>=2)
        for i in range(n):
            plan=json.loads((output/f'history-{i:02d}/plan.json').read_text())
            replay=json.loads((output/f'history-{i:02d}/replay.json').read_text())
            assert len(plan['plan']['actions'])==i+1 and plan['plan']['actions'][-1]=='STOP'
            assert semantic(plan['plan'])==plan['plan_seal']
            assert plan['plan']['history']==replay['metrics']['history'] and replay['metrics']['terminated']
        assert FakeTask.calls==2*sum(range(1,n+1))
        assert len(writes)==len(set(writes))
        return {'complete_histories':n,'aggregate_writes':1,'unique_writes':len(writes),
            'fake_step_calls':FakeTask.calls,'diagnostic_names':list(diagnostics)}

def main():
    start=time.perf_counter();old=HERE.parent/'union025-missing-axis-v1/axis_worker.py';new=HERE/'axis_worker.py'
    evidence={'old_reproduction':run_flow(old,{'clearance':True,'target':True},{'clearance':True,'target':True},old_failure=True),
        'cases':{}}
    for name,feasible,available in (
        ('both_executable',{'clearance':True,'target':True},{'clearance':True,'target':True}),
        ('clearance_blocked',{'clearance':False,'target':False},{'clearance':False,'target':False}),
        ('target_blocked',{'clearance':True,'target':False},{'clearance':True,'target':False}),
        ('clearance_not_emitted',{'clearance':True,'target':True},{'clearance':False,'target':True}),
        ('target_not_emitted',{'clearance':True,'target':True},{'clearance':True,'target':False})):
        evidence['cases'][name]=run_flow(new,feasible,available)
    worker=(HERE/'axis_worker.py').read_text();parent=(HERE/'run_owned.py').read_text();contract=(HERE/'axis_contract.py').read_text()
    assert "status='complete_owned_union025_missing_axis'" in worker
    assert "worker_final.get('status') != 'complete_owned_union025_missing_axis'" in parent
    assert "result['status']='complete_missing_axis_diagnostic'" in worker
    assert "result.get('status')=='complete_missing_axis_diagnostic'" in contract
    evidence.update(status='PASS',seconds=time.perf_counter()-start,
        old_source_sha256=sha(old),new_source_sha256=sha(new),actual_OutputBudget_source_sha256=sha(ROOT/'src/resectionlab/patient_planning_cohort_io.py'),
        parent_worker_contract_status_literals_match=True,native_engine_calls=0,model_forwards=0,patient_payload_reads=0,
        scope='Exact nested source write flow with generated task/geometry/audit doubles; canonical OutputBudget imported and used unchanged')
    (HERE/'exclusive-write-controls.json').write_text(json.dumps(evidence,indent=2)+'\n');print(json.dumps(evidence,indent=2))

if __name__=='__main__':
    def deadline(*unused):raise TimeoutError('Generated control30s cap')
    signal.signal(signal.SIGALRM,deadline);signal.alarm(30)
    try:main()
    finally:signal.alarm(0)

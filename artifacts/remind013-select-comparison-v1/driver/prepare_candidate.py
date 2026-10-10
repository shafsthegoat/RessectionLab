"""Bind terminal JSON only; never open weights, patient arrays or held037 input.

The candidate remains non-executable until the fixed64 attempt is terminal and
root separately freezes/promotes its complete source and execution release.
"""
import json
from pathlib import Path
import subprocess
import sys
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
from comparison_contract import (ROOT,ARMS,TRAIN,ENDPOINTS,OUTPUT,LIMITS,PARENT_SECONDS,
    PUBLIC_INDEX,PUBLIC_SHA,sha,authenticate_endpoint,comparison_world,COHORT)

def ref(path):
    path=Path(path)
    return {'path':path.as_posix(),'sha256':sha(ROOT/path)}

def endpoint(name,directory,method):
    directory=Path(directory);output=directory/'attempt-01'
    paths={'release':directory/'root-release.json','result':output/'result.json',
        'parent':directory/'attempt-01.supervision/receipt.json',
        'worker_final':directory/'attempt-01.supervision/worker-final.json'}
    if not all((ROOT/p).is_file() for p in paths.values()):
        return {'status':'pending_terminal_TRAIN_evidence','expected_paths':{k:p.as_posix() for k,p in paths.items()},
            'expected_checkpoint':(output/(method+'-final.psckpt')).as_posix(),
            'meaning':'predetermined endpoint; never skip or substitute based on SELECT results'}
    rows={k:ref(p) for k,p in paths.items()}
    result=json.loads((ROOT/paths['result']).read_text())
    rows.update(status='terminal_bound',contexts={s:ref(output/(s+'-context.json')) for s in TRAIN},
        checkpoint={'path':(output/(method+'-final.psckpt')).as_posix(),
                    'sha256':result['checkpoints'][method]['sha256']})
    authenticate_endpoint(name,rows)
    return rows

def main():
    baseline='build/cross-patient-planning-v1/fixed-four-train-eight-updates-v1'
    endpoints={ENDPOINTS[0]:endpoint(ENDPOINTS[0],baseline,'IL'),
        ENDPOINTS[1]:endpoint(ENDPOINTS[1],baseline,'RL'),
        ENDPOINTS[2]:endpoint(ENDPOINTS[2],'build/balanced-teacher-il-v1','IL'),
        ENDPOINTS[3]:endpoint(ENDPOINTS[3],'build/balanced-teacher-il64-v1','IL')}
    head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    source_files={p.relative_to(ROOT).as_posix():sha(p) for p in sorted((ROOT/'src/resectionlab').glob('*.py'))}
    for relative in ('comparison_contract.py','select_worker.py','run_owned.py'):
        path=HERE/relative;source_files[path.relative_to(ROOT).as_posix()]=sha(path)
    helper='build/goal-conditioned-policy-v1/run_contact_owned.py';source_files[helper]=sha(ROOT/helper)
    index={'status':'candidate_requires_root_final_source_freeze','head':head,'source_files':source_files,
        'metadata_files':{PUBLIC_INDEX:PUBLIC_SHA,COHORT:sha(ROOT/COHORT)}}
    index_path=HERE/'source-index-candidate.json';index_path.write_text(json.dumps(index,indent=2)+'\n')
    value={'status':'candidate_not_released','expected_head':head,'output':OUTPUT,'source_index':ref(index_path.relative_to(ROOT)),
        'public_index':{'path':PUBLIC_INDEX,'sha256':PUBLIC_SHA},'arms':list(ARMS),'endpoints':endpoints,
        'limits':LIMITS,'parent_seconds':PARENT_SECONDS,'greedy_seconds':60,'attempts':1,'automatic_retry':False,
        'selection_rule':'All four fixed TRAIN endpoints and both existing planners; no SELECT-result-dependent choice',
        'planned_SELECT_denominator':2,'held_cases':['ReMIND-037'],'EVAL_closed':True,
        'claim':'annotation-conditioned nominal planning only; no physical surgery or clinical claim',
        'blockers':['IL64 must be clean terminal and every final descriptor bound',
            'root must serialize and release exact final source/HEAD; no patient execution during preparation']}
    (HERE/'candidate.json').write_text(json.dumps(value,indent=2)+'\n')
    print(json.dumps({'candidate':'candidate.json','endpoints':{k:v['status'] for k,v in endpoints.items()},
        'model_loads':0,'patient_array_reads':0,'held_input_reads':0}))

if __name__=='__main__':main()

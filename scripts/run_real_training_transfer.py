#!/usr/bin/env python3
"""Frozen PAT05-trained imitation policy versus greedy on remaining TRAIN cases.

Zero updates/adaptation. All five preparation outcomes remain in the denominator.
This is development transfer, not population pretraining or held-out efficacy.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import sys
import time
import traceback

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(ROOT/'scripts'))
import run_real_patient_learning as rollout
from preflight_real_spatial_policy import sha256,write_json,read_declaration,supervise_worker,peak_rss_bytes
from resectionlab.data_policy import historical_only
from resectionlab.real_patient_learning import COHORT_SHA256,read_development_cohort,require_development_role

VERSION='remaining-training-frozen-spatial-comparison-v1'
SUBJECTS=('sub-PAT16','sub-PAT20','sub-PAT22','sub-PAT25','sub-PAT28')
PRIMARY_METHODS=('frozen_policy','greedy_search')
METHODS=(*PRIMARY_METHODS,'random_legal')
COHORT_PATH='manifests/experiments/btc-spatial-development-cohort-v1.json'
ANCHOR=ROOT/'artifacts/pat05-real-visited-imitation-v1'
CHECKPOINT=ANCHOR/'augmented_latest.pt'
POLICY_HASH='sha256:74e90d9487dff6fe9db83c92e3887f15533e0499abaac386a73bbd7d4d566b4e'
ARCHITECTURE_HASH='sha256:a3b6740ab188f01016610c566e2009c57ba422c8f9be24adca014f695ae9c917'
SETTINGS={'max_wall_seconds':180.,'max_rss_bytes':6*1024**3,'torch_threads':1,
          'device':'cpu','greedy_planning_seconds':90.,'max_steps':3,'optimizer_updates':0,
          'random_seed':200011}


def source_inventory():
    paths=[Path(__file__),ROOT/'scripts/prepare_real_training_cases.py',
        ROOT/'scripts/run_real_patient_learning.py',ROOT/'scripts/compare_real_spatial_search.py',
        ROOT/'scripts/preflight_real_spatial_policy.py',*(ROOT/'src/resectionlab').rglob('*.py')]
    return {str(p.relative_to(ROOT)):sha256(p)for p in sorted(paths)}


def declaration():
    import prepare_real_training_cases as prep
    return {'version':VERSION,'subjects':list(SUBJECTS),'methods':list(METHODS),'settings':SETTINGS,
        'declared_at':datetime.now(timezone.utc).isoformat(),'source_sha256':source_inventory(),
        'cohort_path':COHORT_PATH,'cohort_sha256':COHORT_SHA256,
        'checkpoint':{'path':str(CHECKPOINT.relative_to(ROOT)),'sha256':sha256(CHECKPOINT),
            'parameter_hash':POLICY_HASH,'architecture_hash':ARCHITECTURE_HASH,
            'training_patient':'sub-PAT05','training_kind':'imitation_with_one_visited_state_aggregation'},
        'patient_registry':prep.preparation_inputs(),
        'random_scope':'one fixed-seed uniform-legal episode including STOP, after primary methods; n=1 is diagnostic only',
        'scope':'five existing TRAIN development anatomies; frozen transfer only, no adaptation, selection, population or clinical efficacy claim',
        'resource_policy':'serial patients; entire180s envelope includes preparation, frozen rollout, greedy planning/replay and audits; incomplete methods remain unassessed',
        'failure_policy':'all five patients retained, including blocked preparation and partial/timeout methods'}


def validate(record):
    import prepare_real_training_cases as prep
    if (record.get('version')!=VERSION or record.get('subjects')!=list(SUBJECTS)
            or record.get('methods')!=list(METHODS) or record.get('settings')!=SETTINGS):
        raise ValueError('Only the fixed remaining-TRAIN zero-update comparison is allowed')
    if record.get('source_sha256')!=source_inventory():raise ValueError('Numerical sources changed')
    if record.get('cohort_path')!=COHORT_PATH or record.get('cohort_sha256')!=COHORT_SHA256:
        raise ValueError('Original cohort binding changed')
    cohort=read_development_cohort(ROOT/COHORT_PATH)
    for subject in SUBJECTS:require_development_role(cohort,subject,role='TRAIN')
    if record.get('patient_registry')!=prep.preparation_inputs():raise ValueError('Patient/source registry changed')
    checkpoint=record.get('checkpoint',{})
    if (checkpoint.get('path')!=str(CHECKPOINT.relative_to(ROOT))
            or checkpoint.get('parameter_hash')!=POLICY_HASH or checkpoint.get('architecture_hash')!=ARCHITECTURE_HASH
            or checkpoint.get('sha256')!=sha256(CHECKPOINT)):
        raise ValueError('Frozen PAT05-trained checkpoint changed')
    index=json.loads((ANCHOR/'output-sha256.json').read_text())
    if index.get('augmented_latest.pt')!=checkpoint['sha256']:
        raise ValueError('Checkpoint no longer matches its completed training byte inventory')
    try:datetime.fromisoformat(record['declared_at'])
    except (ValueError,KeyError,TypeError):raise ValueError('Fixed declaration time required')
    return cohort


@historical_only("GENERATED_POLICY_INELIGIBLE")
def load_frozen_policy(path,expected_file_hash):
    """Verify before opening any patient; never construct an optimizer."""
    import torch
    from resectionlab.spatial_policy import SpatialPolicy,SpatialPolicyConfig,parameter_hash
    if sha256(path)!=expected_file_hash:raise ValueError('Copied checkpoint bytes changed')
    # The byte-verified local checkpoint includes NumPy scalar architecture
    # metadata; use the same trusted-artifact loader as its training runners.
    checkpoint=torch.load(path,map_location='cpu',weights_only=False)
    policy=SpatialPolicy(SpatialPolicyConfig(critic_candidate_context=True))
    policy.load_state_dict(checkpoint['policy'])
    if (checkpoint.get('additional_updates')!=8 or checkpoint.get('parameter_hash')!=POLICY_HASH
            or parameter_hash(policy)!=POLICY_HASH or policy.architecture_hash!=ARCHITECTURE_HASH
            or checkpoint.get('architecture')!=policy.architecture_record()):
        raise ValueError('Unexpected frozen spatial policy tensors or architecture')
    policy.eval().requires_grad_(False)
    return policy


def blank_receipt(subject):
    if subject not in SUBJECTS:raise ValueError('Only remaining TRAIN subjects are permitted')
    return {'version':VERSION,'subject':subject,'role':'TRAIN','status':'preparing',
        'optimizer_updates':0,'gradient_steps':0,'adaptation':False,'clinical_efficacy_measured':False,
        'methods':{name:{'status':'not_executed','outcomes':None}for name in METHODS},
        'method_order':list(METHODS),'settings':SETTINGS}


def run_patient(subject,record,output):
    """One bounded process; caller persists failures and keeps all five cases."""
    validate(record)
    import torch
    import prepare_real_training_cases as prep
    from resectionlab.spatial_policy import parameter_hash
    from resectionlab.native_spatial_evaluation import evaluate_native_spatial_episode
    started=time.perf_counter();report=blank_receipt(subject)
    torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    checkpoint=output.parent/'frozen-checkpoint.pt'
    policy=load_frozen_policy(checkpoint,record['checkpoint']['sha256'])
    random_generator=torch.Generator().manual_seed(SETTINGS['random_seed'])
    report['initial_parameter_hash']=parameter_hash(policy)

    def cancelled():
        return time.perf_counter()-started>=SETTINGS['max_wall_seconds'] or peak_rss_bytes()>SETTINGS['max_rss_bytes']
    def guard():
        if cancelled():raise TimeoutError('Per-patient runtime or memory envelope exhausted')
    def preserve(enforce=True):
        report.update(elapsed_seconds=time.perf_counter()-started,peak_rss_bytes=peak_rss_bytes())
        write_json(output/'receipt.json',report)
        if enforce:guard()
    def audit(task,*,metrics):
        return evaluate_native_spatial_episode(task,metrics=metrics,minimum_target_cells=1,cancelled=cancelled,
            distance_backend='batch',distance_batch_size=256)

    preserve()
    try:
        prepare_started=time.perf_counter()
        base,preparation=prep.prepare_training_case(subject,declared_at=record['declared_at'],cancelled=cancelled)
        report['preparation_seconds']=time.perf_counter()-prepare_started
        write_json(output/'preparation.json',preparation)
        report['preparation_sha256']=sha256(output/'preparation.json')
        report['preparation_status']=preparation.get('status')
        if base is None:
            report.update(status='preparation_blocked',preparation=preparation,
                          final_parameter_hash=parameter_hash(policy))
            preserve();return report
        if preparation.get('status')!='prepared' or base.max_steps!=3:
            raise ValueError('Preparation did not establish the fixed common task')
        report['status']='evaluating'
        report['decision_model_hash']=base.decision_model_hash
        report['initial_task_metrics']=base.metrics()
        preserve()
        for name in METHODS:
            guard();method_started=time.perf_counter()
            row={'status':'running','outcomes':None,'initial_parameter_hash':parameter_hash(policy)}
            report['methods'][name]=row;preserve()
            try:
                sequence=None
                if name=='greedy_search':
                    remaining=SETTINGS['max_wall_seconds']-(time.perf_counter()-started)
                    planning_cap=min(SETTINGS['greedy_planning_seconds'],remaining)
                    if planning_cap<=0:raise TimeoutError('No remaining patient budget for search')
                    row['effective_planning_cap_seconds']=planning_cap
                    plan_started=time.perf_counter()
                    try:sequence,accounting=base.observed_greedy_search(seconds=planning_cap)
                    except BaseException as error:
                        row.update(planning_seconds=time.perf_counter()-plan_started,
                            partial_search_accounting=getattr(error,'accounting',None),
                            unreplayed_partial_sequence=getattr(error,'best_sequence',None))
                        raise
                    row.update(planning_seconds=time.perf_counter()-plan_started,
                               sequence=list(sequence),search_accounting=accounting)
                    preserve()
                mode={'frozen_policy':'argmax','greedy_search':'sequence','random_legal':'random'}[name]
                _,episode=rollout.run_episode(base,policy,random_generator if name=='random_legal'else None,mode=mode,
                    name=name,output=output,guard=guard,audit=audit,sequence=sequence)
                if episode['independent_evaluation'].get('accepted') is not True:
                    raise RuntimeError('Completed episode lacks an accepted independent audit')
                row.update(status='complete',outcomes=episode['independent_evaluation']['outcomes'],
                    independent_evaluation_accepted=True,
                    target_access_success=episode['independent_evaluation']['target_access_success'],
                    episode_seconds=episode['elapsed_seconds'],independent_audit_seconds=episode['independent_evaluation_seconds'],
                    policy_forward_calls=episode['policy_forward_calls'],invalid_actions=episode['invalid_actions'],
                    actions=[x['action_id']for x in episode['decisions']],
                    actor_decision_seconds=sum(x['decision_seconds']for x in episode['decisions']),
                    native_steps_including_inventory_seconds=sum(x['transition_seconds']for x in episode['decisions']))
            except BaseException as error:
                row.update(status='failed',outcomes=None,failure={'type':type(error).__name__,'message':str(error)})
            row.update(total_method_seconds=time.perf_counter()-method_started,final_parameter_hash=parameter_hash(policy))
            if row['initial_parameter_hash']!=POLICY_HASH or row['final_parameter_hash']!=POLICY_HASH:
                raise RuntimeError('Method changed the frozen policy')
            preserve()
        report['status']='complete'if all(x['status']=='complete'for x in report['methods'].values())else'partial'
        report['final_parameter_hash']=parameter_hash(policy)
        if source_inventory()!=record['source_sha256'] or sha256(checkpoint)!=record['checkpoint']['sha256']:
            raise RuntimeError('Sources or checkpoint changed during patient evaluation')
        preserve();return report
    except BaseException as error:
        report.update(status='failed',failure={'type':type(error).__name__,'message':str(error)},
                      final_parameter_hash=parameter_hash(policy))
        preserve(enforce=False);raise


def runtime_closure(record,checkpoint):
    import prepare_real_training_cases as prep
    return {'sources_unchanged':source_inventory()==record['source_sha256'],
        'preparation_inputs_unchanged':prep.preparation_inputs()==record['patient_registry'],
        'checkpoint_bytes_unchanged':sha256(checkpoint)==record['checkpoint']['sha256'],
        'scope':'parent check after worker exit, including a timeout during optional random'}


def primary_pair_eligible(patient):
    closure=patient.get('closure_check') or {}
    if not all(closure.get(key)is True for key in ('sources_unchanged','preparation_inputs_unchanged','checkpoint_bytes_unchanged')):
        return False
    for name in PRIMARY_METHODS:
        row=patient['methods'].get(name,{})
        if (row.get('status')!='complete' or row.get('outcomes')is None
                or row.get('independent_evaluation_accepted')is not True
                or row.get('initial_parameter_hash')!=POLICY_HASH or row.get('final_parameter_hash')!=POLICY_HASH):
            return False
    return True


def summarize_patients(output,subjects=SUBJECTS):
    """Always retain the prescribed denominator; missing/partial is not zero."""
    patients=[]
    for subject in subjects:
        directory=output/subject
        receipt=json.loads((directory/'receipt.json').read_text())if(directory/'receipt.json').exists()else {**blank_receipt(subject),'status':'not_executed'}
        supervisor=json.loads((directory/'supervisor.json').read_text())if(directory/'supervisor.json').exists()else None
        closure=json.loads((directory/'closure-check.json').read_text())if(directory/'closure-check.json').exists()else None
        patients.append({'subject':subject,'status':receipt.get('status','not_executed'),
            'preparation_status':receipt.get('preparation_status'),'preparation_seconds':receipt.get('preparation_seconds'),
            'methods':receipt['methods'],'initial_parameter_hash':receipt.get('initial_parameter_hash'),
            'final_parameter_hash':receipt.get('final_parameter_hash'),'supervisor':supervisor,'closure_check':closure})
    paired=[x for x in patients if primary_pair_eligible(x)]
    return {'version':VERSION,'patients_prescribed':len(subjects),'patients':patients,'complete_pairs':len(paired),
        'optimizer_updates':0,'adaptation':False,'checkpoint_parameter_hash':POLICY_HASH,
        'scope':'remaining TRAIN development anatomy transfer from a single-patient trained model; not held-out generalization',
        'paired_return_difference':{x['subject']:x['methods']['frozen_policy']['outcomes']['total_reward']-
            x['methods']['greedy_search']['outcomes']['total_reward']for x in paired}}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--declaration',type=Path,required=True);parser.add_argument('--write-declaration',action='store_true')
    parser.add_argument('--output',type=Path);parser.add_argument('--execute',action='store_true')
    parser.add_argument('--worker',action='store_true',help=argparse.SUPPRESS)
    parser.add_argument('--subject',choices=SUBJECTS,help=argparse.SUPPRESS)
    parser.add_argument('--expected-declaration-sha256',help=argparse.SUPPRESS);args=parser.parse_args()
    if args.write_declaration:
        if args.declaration.exists():raise SystemExit('Preserve existing declarations')
        write_json(args.declaration,declaration());return
    record,digest,payload=read_declaration(args.declaration,args.expected_declaration_sha256);validate(record)
    if args.worker:
        if not args.expected_declaration_sha256 or args.subject not in SUBJECTS:raise ValueError('Bound declaration and subject required')
        try:run_patient(args.subject,record,args.output)
        except BaseException as error:
            write_json(args.output/'failure.json',{'type':type(error).__name__,'message':str(error),'traceback':traceback.format_exc()});raise
        return
    if not args.execute:print(json.dumps({'status':'validated_not_executed','declaration_sha256':digest}));return
    if args.output is None or args.output.exists():raise SystemExit('Choose a new output directory')
    args.output.mkdir(parents=True);(args.output/'declaration-input.json').write_bytes(payload)
    shutil.copy2(CHECKPOINT,args.output/'frozen-checkpoint.pt')
    for relative in record['source_sha256']:
        destination=args.output/'source-snapshot'/relative;destination.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/relative,destination)
    for subject in SUBJECTS:
        directory=args.output/subject;directory.mkdir()
        write_json(directory/'receipt.json',{**blank_receipt(subject),'status':'not_executed'})
    write_json(args.output/'summary.json',summarize_patients(args.output))
    for subject in SUBJECTS:
        validate(record)
        directory=args.output/subject
        supervise_worker([sys.executable,str(Path(__file__).resolve()),'--worker','--subject',subject,
            '--declaration',str((args.output/'declaration-input.json').resolve()),'--expected-declaration-sha256',digest,
            '--output',str(directory.resolve())],directory,SETTINGS,digest)
        write_json(directory/'closure-check.json',runtime_closure(record,args.output/'frozen-checkpoint.pt'))
        write_json(args.output/'summary.json',summarize_patients(args.output))
    write_json(args.output/'output-sha256.json',{str(p.relative_to(args.output)):sha256(p)
        for p in sorted(args.output.rglob('*'))if p.is_file()and 'source-snapshot'not in p.parts and p.name!='output-sha256.json'})


if __name__=='__main__':main()

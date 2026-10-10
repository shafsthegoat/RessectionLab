"""Fixed HBE specimen fit: numerical axial confirmation, then prediction freeze.

No CSV/archive reader, fitter, mesher or held-out evaluator is called. A separate
root release and owned-stage launcher are required. Poor axial fit remains an
unchanged result; numerical scale agreement is not physical validation.
"""
from __future__ import annotations
from contextlib import ExitStack
from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path
import time
import xml.etree.ElementTree as ET

from scripts import mechanics_hbe_v5_axial_calibration as authority

need, canonical, sha = authority.need, authority.canonical, authority.sha
local, read_binding, exclusive_json = authority.local, authority.read_binding, authority.exclusive_json
CORE = 'scripts/mechanics_hbe_v5_fitted_confirmation.py'
RUNNER = 'scripts/mechanics_hbe_v5_fitted_confirmation_experiment.py'
DECLARATION = 'manifests/experiments/hbe-01-03-v5-fixed-fit-continuation-v1.json'
DECLARATION_SHA256 = '083b9eac91dcc26353243fbb3c03ed4b3d9e6ddbb373337d55322d747bc8ff68'
OUTPUT = 'outputs/mechanics/hbe-v5-fixed-fit-confirmation-v1/attempt-01'
FIT_SHA = 'b6b23a297f81b9481a6a0c7b584c8d1d84907baf86dd4ecb62b78474caf9666a'
MU, SCALE = 715.361571139082, 0.715361571139082
AXIAL = ('compression', 'tension')
DONE = 'completed_fitted_confirmation_and_prediction_freeze'
CAPS = {'total_seconds':3600,'cleanup_reserve_seconds':10,'preparation_seconds':60,
        'native_seconds':{'compression':2100,'tension':420},'readout_seconds':600,
        'sampled_group_rss_bytes':3*1024**3,'new_output_bytes':2*1024**3,
        'branch_output_bytes':{'compression':1536*1024**2,'tension':512*1024**2},
        'threads':1,'attempts':1,'native_calls':2,'fit_calls':0,'mesher_calls':0,
        'measured_member_reads':0,'held_out_member_reads':0,'automatic_retry':False}
SOURCE_ADDITIONS = {CORE,RUNNER,'scripts/mechanics_hbe_v5_frame.py','scripts/mechanics_hbe_v5_stream.py'}


def check_release(root, binding, *, executing):
    release=read_binding(root,binding,maximum=1024**2)
    need(set(release)=={'schema','execution_released','declaration','output_directory','caps',
         'source_commit','source_bindings','runtime','independent_source_review'},'release schema')
    need(release['schema']=='hbe-v5-fixed-fit-release-v1' and type(release['execution_released']) is bool,
         'explicit fixed-fit release')
    need(not executing or release['execution_released'],'separate root release required')
    need(release['declaration']=={'path':DECLARATION,'sha256':DECLARATION_SHA256}
         and release['output_directory']==OUTPUT and release['caps']==CAPS,'frozen scope/caps')
    declaration=read_binding(root,release['declaration'])
    need(declaration['caps']==CAPS and declaration['fixed_fit']['fit']['sha256']==FIT_SHA,
         'fixed declaration')
    if executing:
        import re
        need(isinstance(release['source_commit'],str) and re.fullmatch('[a-f0-9]{40}',release['source_commit']),
             'committed source required')
        review=read_binding(root,release['independent_source_review'])
        candidate={**release,'execution_released':False,'source_commit':None,'independent_source_review':None}
        need(review.get('schema')=='hbe-v5-fixed-fit-source-review-v1' and review.get('decision')=='GO'
             and review.get('release_candidate_sha256')==sha(canonical(candidate)),'exact independent review')
    return release,declaration


def verify_sources(root,release,declaration,*,committed):
    """Reuse exact source/runtime validation with this phase's source names."""
    expected=declaration['inherited_source_bindings']
    need(set(release['source_bindings'])==set(expected)|SOURCE_ADDITIONS,'source closure')
    need(all(release['source_bindings'][p]==d for p,d in expected.items()),'inherited source drift')
    # Existing implementation accepts its own two names; they are already in
    # inherited closure. Pass all the phase's sources as inherited, unchanged.
    proxy={'inherited_source_bindings':dict(release['source_bindings'])}
    authority.verify_sources(root,release,proxy,committed=committed)


audit_loaded=authority.audit_loaded


def fixed_fit(root,declaration):
    """Authenticate accepted ten-file terminal, not a new estimator invocation."""
    fixed=declaration['fixed_fit'];publication=read_binding(root,fixed['publication'])
    parent=local(root,fixed['publication']['path']).parent
    names={'intent.json','intent-receipt.json','receipt.json','readout-console.txt','state.json',
           'access.jsonl','fit.json','calibration-metrics.json','terminal.json'}
    need(publication.get('accepted') is True and set(publication['outputs'])==names
         and {p.name for p in parent.iterdir()}==names|{'publication.json'},'exact accepted fit publication')
    for name,row in publication['outputs'].items():
        # Empty console is a legitimate bound output; shared launcher reader
        # performs one bounded regular read and authenticates those same bytes.
        from scripts.mechanics_hbe_v5_axial_calibration_experiment import regular_bytes
        raw=regular_bytes(parent/name,16*1024**2,expected=row['sha256'],allow_empty=name=='readout-console.txt')
        need(len(raw)==row['bytes'],'accepted fit size changed')
    need(publication['outputs']['fit.json']['sha256']==FIT_SHA,'fixed fit identity')
    fit=read_binding(root,fixed['fit']);need(fit['mu_Pa']==MU and fit['scale']==SCALE,'fixed scalar changed')
    state=read_binding(root,{'path':str((parent/'state.json').relative_to(root)),
                            'sha256':publication['outputs']['state.json']['sha256']})
    need(state['status']==authority.DONE and state['fit_calls']==1 and state['native_calls']==0
         and state['held_out_member_reads']==0 and state['freeze_saved'] is False,'fixed fit terminal')
    for key in ('independent_audit','independent_report'):read_binding(root,fixed[key],decode=False)
    return fit,{'path':str((parent/'access.jsonl').relative_to(root)),
                'sha256':publication['outputs']['access.jsonl']['sha256']}


def admit_evidence(root,declaration):
    axial=read_binding(root,{'path':authority.DECLARATION,'sha256':authority.DECLARATION_SHA256})
    records,_=authority.admit_evidence(root,axial)
    fit,ledger=fixed_fit(root,declaration)
    need(records['protocol']['roles']==declaration['evidence']['roles'],'unchanged specimen roles')
    return records,fit,ledger


def fitted_deck(reference):
    from scripts.mechanics_hbe_experiment import check_fitted_deck
    tree=ET.fromstring(reference)
    for path,value in [('Material/material/c1',2*MU),('Material/material/k',149*MU/3),
                       ('Control/solver/min_residual',(1e-10*MU*.004**2)**2)]:
        matches=tree.findall(path);need(len(matches)==1,'exact scalar node');matches[0].text=repr(value)
    text=ET.tostring(tree,encoding='unicode',xml_declaration=True)+'\n'
    check_fitted_deck(reference,text,MU)
    return text.encode()


def fitted_contract(reference, deck_sha):
    need(reference['mu_Pa']==1000. and reference['steps']==120
         and reference['native_domain']=='lower_half_reconstructed','exact v5 axial reference')
    branch=reference['branch'];need(branch in AXIAL,'axial branch only')
    endpoint={'compression':-0.00073726,'tension':0.0007360099999999}[branch]
    need(reference['full_coordinates_m'][-1].hex()==endpoint.hex()
         and tuple(reference['times'])==tuple(i/120 for i in range(121)), 'frozen 120-step endpoint')
    return dict(reference,run_id=reference['run_id'].replace(':reference',':fitted'),mu_Pa=MU,
                adapted_deck_sha256=deck_sha,fixed_fit_sha256=FIT_SHA)


def primitive_caps(branch):
    need(branch in AXIAL,'axial branch')
    return {'nodes':(768 if branch=='compression' else 256)*1024**2,
            'elements':(512 if branch=='compression' else 256)*1024**2}


def paired_streams(reference, fitted, *, deadline):
    """Consume two evaluated generators in lockstep; retain no primitive history."""
    from scripts.mechanics_hbe_branch_calibration_v3_readout import ScaleReduction
    reductions={key:ScaleReduction(MU) for key in ('native','reconstructed_full')};reports=[]
    try:
        for index in range(121):
            need(time.monotonic()<deadline,'paired stream deadline')
            pair=[]
            for stream in (reference,fitted):
                try:pair.append(next(stream))
                except StopIteration as error:raise ValueError('incomplete paired state') from error
            (a,pa),(b,pb)=pair
            need(pa.native_mesh.fingerprint==pb.native_mesh.fingerprint
                 and pa.full_mesh.fingerprint==pb.full_mesh.fingerprint
                 and pa.mapping_sha256==pb.mapping_sha256,'same paired native/full geometry')
            reductions['reconstructed_full'].update(a,b,pa.full_mesh.rest_nodes_m)
            na=[dict(row,current=row['native_current'],raw=row['native_raw'],torque=row['native_torque']) for row in (a,b)]
            reductions['native'].update(*na,pa.native_mesh.rest_nodes_m)
        for stream in (reference,fitted):
            try:next(stream)
            except StopIteration as complete:reports.append(complete.value)
            else:raise ValueError('extra paired state')
    finally:
        reference.close();fitted.close()
    metrics={key:r.finish() for key,r in reductions.items()}
    passed=all(r['numerical_passed'] for r in reports) and all(
        item['actual']<=item['limit'] for group in metrics.values() for item in group.values())
    return reports,metrics,passed


def load_reference(root,declaration,branch):
    from scripts import mechanics_hbe_v5_frame as frame
    from scripts import mechanics_hbe_v5_stream as stream
    q=declaration['references'][branch];bindings=q['primitive_bindings']
    old=read_binding(root,bindings['source_deck'],maximum=32*1024**2,decode=False)
    adapted=read_binding(root,bindings['adapted_deck'],maximum=32*1024**2,decode=False)
    order=read_binding(root,q['work_order'])
    study=read_binding(root,declaration['evidence']['v5'],decode=False)
    prior_raw=read_binding(root,declaration['evidence']['v4'],decode=False)
    # Work order is the exact already accepted reader invocation.
    contract=frame.verified_schedule(study,prior_raw,q['reference_run_id'],old,adapted,order['adapter_receipt'])
    mesh=read_binding(root,bindings['mesh'],maximum=32*1024**2)
    reconstruction=stream._bound_reconstruction(root,json.loads(prior_raw),
        {'N':q['N']},bindings['mesh'],mesh)
    return contract,mesh,reconstruction,adapted


def prepare(root,declaration,out):
    started=time.monotonic();admit_evidence(root,declaration)
    rows={}
    for branch in AXIAL:
        q=declaration['references'][branch]
        contract,_,_,adapted=load_reference(root,declaration,branch)
        target=out/branch;target.mkdir(exist_ok=False)
        raw=fitted_deck(adapted);path=target/'specimen.feb'
        fd=os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
        with os.fdopen(fd,'wb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
        rows[branch]={'contract':fitted_contract(contract,sha(raw)),
            'deck':{'path':str(path.relative_to(root)),'sha256':sha(raw)},'reference':q['readout']}
    need(time.monotonic()-started<CAPS['preparation_seconds'],'combined preparation budget')
    return rows


def bound_hash(root,binding,maximum):
    from scripts.mechanics_hbe_v5_n8_one_shot import file_hash
    need(file_hash(local(root,binding['path']),maximum=maximum)==binding['sha256'],'primitive digest changed')


def check_execution(root,declaration,branch,prepared,execution):
    from scripts import mechanics_hbe_backend as backend
    q=declaration['references'][branch];supervision=execution['supervision']
    stage=supervision['receipt']['native_stage'];cleanup=supervision['cleanup']
    command=[str(root/backend.PREFIX/'install/bin/febio4'),'-noconfig','-no_title','-i','specimen.feb','-o','solver.log']
    need(execution['run_id']==q['fitted_run_id'] and execution['fixed_fit_sha256']==FIT_SHA
         and execution['deck']==prepared['deck'] and execution['command']==stage['command']==command
         and execution['runtime_identity']==declaration['runtime_identity'],'fitted execution identity')
    need(stage['status']=='completed_within_caps' and stage['exit_code']==0 and stage['kill_reason'] is None
         and 0 <= stage['elapsed_seconds'] < stage['wall_cap_seconds'] <= CAPS['native_seconds'][branch]
         and stage['sampled_process_group_rss_cap_bytes']==CAPS['sampled_group_rss_bytes']
         and stage['active_output_cap_bytes']==CAPS['branch_output_bytes'][branch]
         and cleanup['contained'] and cleanup['direct_child_reaped'] and cleanup['exit_code']==0
         and not cleanup['fallback_used'] and cleanup['remaining_members']==[] and cleanup['errors']==[],
         'successful owned fitted invocation required')
    need(set(execution['primitive_bindings'])=={'nodes','elements','solver'},'exact fitted primitive keys')
    for key,binding in execution['primitive_bindings'].items():
        need(binding['path']==OUTPUT+'/'+branch+'/'+key+'.log','fitted primitive path')
    read_binding(root,prepared['deck'],maximum=32*1024**2,decode=False)


def read_pairs(root,declaration,prepared,executions,*,deadline):
    from scripts import mechanics_hbe_v5_stream as stream
    rows={}
    for branch in AXIAL:
        q=declaration['references'][branch];ref=q['primitive_bindings'];new=executions[branch]['primitive_bindings']
        check_execution(root,declaration,branch,prepared[branch],executions[branch])
        contract,mesh,reconstruction,_=load_reference(root,declaration,branch)
        expected=fitted_contract(contract,prepared[branch]['deck']['sha256'])
        need(canonical(prepared[branch]['contract'])==canonical(expected),'prepared fitted contract changed')
        caps={**primitive_caps(branch),'solver':16*1024**2}
        for bindings in (ref,new):
            for key,cap in caps.items():bound_hash(root,bindings[key],cap)
        with ExitStack() as stack:
            generators=[]
            for bindings,c,fitted in ((ref,contract,False),(new,expected,True)):
                opened={key:stack.enter_context(local(root,bindings[key]['path']).open(encoding='utf-8',errors='strict')) for key in caps}
                generators.append(stream.evaluated_frames(c,mesh,opened['nodes'],opened['elements'],opened['solver'],
                    reconstruction=reconstruction,fitted=fitted,primitive_caps=primitive_caps(branch)))
            reports,metrics,passed=paired_streams(*generators,deadline=deadline)
        saved=read_binding(root,q['readout'])
        need({k:v for k,v in reports[0].items() if k!='provenance'}==
             {k:v for k,v in saved.items() if k!='provenance'},'reference stream differs from accepted readout')
        for bindings in (ref,new):
            for key,cap in caps.items():bound_hash(root,bindings[key],cap)
        reports[1]['provenance'].update(native_output_observed=True,generated_fixture_only=False,
            source_binding_checked=True,output_origin='owned_fixed_fit_native',
            fixed_fit_sha256=FIT_SHA,primitive_bindings=new)
        rows[branch]={'execution':executions[branch],'reference':q['readout'],'readout':reports[1],
                      'scale':metrics,'passed':passed}
    return {'schema':'hbe-v5-fixed-fit-paired-evidence-v1','mu_Pa':MU,'fixed_fit_sha256':FIT_SHA,
        'branches':rows,'passed':all(row['passed'] for row in rows.values()),'states_per_native_run':121,
        'physical_validation_pass':None,'full_native_fine_equivalence_claim':False}


def require_confirmations(numerical):
    need(numerical.get('passed') is True and numerical.get('mu_Pa')==MU
         and numerical.get('fixed_fit_sha256')==FIT_SHA
         and set(numerical.get('branches',{}))==set(AXIAL)
         and all(row.get('passed') is True for row in numerical['branches'].values()),'both fitted checks must pass')


def checked_predictions(references,numerical):
    from scripts.mechanics_hbe_branch_calibration_v3 import predictions, checked_reference, TORSION
    require_confirmations(numerical)
    need(set(references)==set(TORSION),'two original torsion predictions')
    for branch,row in references.items():checked_reference(branch,row)
    return predictions(references,SCALE)


def qualify_torsion(root,study,registry):
    """Original v3 torsion predicates, excluding its superseded axial entrypoint."""
    import tarfile
    from scripts.mechanics_hbe_branch_calibration_v3 import access, checked_reference, TORSION
    references={}
    legacy = registry.bound(study['legacy']['aggregate'])
    outcome = registry.bound(study['legacy']['outcome'])
    if not outcome['failed_aggregate_criteria'] or outcome['actual_solver_calls'] != 18:
        raise ValueError('Original failed aggregate must remain preserved')
    legacy_release = registry.bound(study['legacy']['release'])
    if legacy_release['execution']['backend_profile'] != study['backend_profile']:
        raise ValueError('Original torsion backend differs')
    original_sources = legacy_release['execution']['source_bindings']
    original_archive = legacy_release['execution']['source_archive']
    registry.bound(original_archive,json_value=False)
    with tarfile.open(access.local_path(root,original_archive['path'])) as bundle:
        if bundle.pax_headers.get('comment') != legacy_release['source_commit']:
            raise ValueError('Original torsion archive commit differs')
        for key,binding in original_sources.items():
            report_key = 'run_readout' if key == 'readout' else key
            if report_key in legacy['source_bindings'] and legacy['source_bindings'][report_key] != binding:
                raise ValueError('Original torsion report/source origin differs')
            registry.bound(binding,json_value=False)
            members=[m for m in bundle.getmembers() if m.name=='scripts/'+Path(binding['path']).name]
            if len(members)!=1 or not members[0].isfile() or members[0].size>1024**2:
                raise ValueError('Original torsion source member differs')
            if hashlib.sha256(bundle.extractfile(members[0]).read()).hexdigest()!=binding['sha256']:
                raise ValueError('Original torsion archived source differs')
    for branch in TORSION:
        for group in ('mesh','step'):
            for metric in legacy[group][branch].values():
                access._metric_record(metric)
        q = study['torsion_references'][branch]
        registry.primitives(branch,q['primitives'])
        row = registry.bound(q['readout']); checked_reference(branch,row)
        if legacy['runs'][f'{branch}:N12:S120:reference'] != row:
            raise ValueError('Original torsion aggregate/readout identity differs')
        access.verify_run_execution(root, f'{branch}:N12:S120:reference', row,
                                   study['protocol']['sha256'], expected_backend_profile=study['backend_profile'])
        registry.bound(q['execution']); references[branch] = row
    for branch in ('compression','torsion_pos'):
        for metric in legacy['scale'][branch].values():
            access._metric_record(metric)
    return references


def finish(root,declaration,release_binding,out,numerical,executions):
    from scripts import mechanics_hbe_branch_calibration_v3 as v3
    require_confirmations(numerical)
    fit,old_ledger=fixed_fit(root,declaration)
    study=read_binding(root,{'path':v3.DECLARATION_PATH,'sha256':v3.DECLARATION_SHA256})
    registry=v3.Registry(root)
    references=qualify_torsion(root,study,registry)
    prediction=checked_predictions(references,numerical)
    prediction_sha=exclusive_json(out/'predictions.json',prediction)
    numeric_sha=sha(canonical(numerical))
    read_binding(root,{'path':str((out/'numerical.json').relative_to(root)),'sha256':numeric_sha},maximum=4*1024**2)
    registry.verify_all()
    freeze={'schema':'hbe-v5-fixed-parameter-prediction-freeze-v1','release':release_binding,
        'declaration':{'path':DECLARATION,'sha256':DECLARATION_SHA256},
        'fixed_fit':declaration['fixed_fit'],'original_fit_ledger':old_ledger,
        'roles':declaration['evidence']['roles'],'protocol':declaration['evidence']['protocol'],
        'mu_Pa':MU,'scale':SCALE,'refit_calls':0,'native_calls':2,'mesher_calls':0,
        'predictions':{'path':str((out/'predictions.json').relative_to(root)),'sha256':prediction_sha},
        'numerical':{'path':str((out/'numerical.json').relative_to(root)),'sha256':numeric_sha},
        'fitted_executions':executions,'original_torsion_bindings':registry.snapshot(),
        'measured_member_reads':0,'held_out_member_reads':0,'physical_validation_pass':None,
        'calibration_quality':'unchanged_poor_within_specimen_fit_not_physical_validation',
        'patient_tool_mechanics_admitted':False,'held_out_access_released':False}
    freeze_sha=exclusive_json(out/'freeze.json',freeze,maximum=4*1024**2)
    event={'phase':'freeze_saved','freeze':{'path':str((out/'freeze.json').relative_to(root)),
           'sha256':freeze_sha},'previous_fit_ledger':old_ledger,'measured_member_reads':0,'refit_calls':0}
    exclusive_json(out/'continuation-ledger.json',event)
    return freeze,event


def recheck_fixed_inputs(root,declaration,freeze):
    """Final fixity of admitted specimen evidence; never reparse response members."""
    admit_evidence(root,declaration)
    for branch,q in declaration['references'].items():
        caps={**primitive_caps(branch),'solver':16*1024**2,'source_deck':32*1024**2,
              'adapted_deck':32*1024**2,'mesh':32*1024**2}
        for key,cap in caps.items():bound_hash(root,q['primitive_bindings'][key],cap)
        for key in ('readout','work_order'):read_binding(root,q[key],decode=False)
    for item in freeze['original_torsion_bindings']:
        bound_hash(root,{k:item[k] for k in ('path','sha256')},item['maximum_bytes'])

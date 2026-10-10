"""One descriptive torsion comparison after an independently accepted v5 freeze.

This adapter never fits, creates predictions, resumes old studies or runs native
mechanics. A distinct ledger admits an immutable predecessor; old ledgers stay
unchanged. Import and false-release checks do not open experimental responses.
"""
from __future__ import annotations
from pathlib import Path
import math
import re
import time

from scripts import mechanics_hbe_v5_axial_calibration as authority

need, canonical, sha = authority.need, authority.canonical, authority.sha
local, read_binding = authority.local, authority.read_binding
exclusive_json, audit_loaded = authority.exclusive_json, authority.audit_loaded
CORE = 'scripts/mechanics_hbe_v5_torsion_evaluation.py'
RUNNER = 'scripts/mechanics_hbe_v5_torsion_evaluation_experiment.py'
DECLARATION = 'manifests/experiments/hbe-01-03-v5-torsion-evaluation-v1.json'
DECLARATION_SHA256 = 'fd5f04054cb153b0e69bfd2921ff584c1c4da4f664e539fd39c405e5a153f072'
OUTPUT = 'build/hbe-v5-torsion-evaluation-v1/attempt-01'
DONE = 'completed_descriptive_frozen_torsion_comparison'
FAILED = 'failed_torsion_evaluation_attempt'
TORSION = ('torsion_neg', 'torsion_pos')
MU, SCALE = 715.361571139082, 0.715361571139082
FIT_SHA = 'b6b23a297f81b9481a6a0c7b584c8d1d84907baf86dd4ecb62b78474caf9666a'
CAPS = {'worker_seconds':60, 'cleanup_seconds':10, 'total_seconds':70,
    'sampled_group_rss_bytes':512*1024**2, 'new_output_bytes':16*1024**2,
    'threads':1, 'attempts':1, 'fit_calls':0, 'native_calls':0,
    'mesher_calls':0, 'held_out_member_reads':2, 'automatic_retry':False}
UPSTREAM_NAMES = ('publication','terminal','freeze','predictions','numerical','continuation_ledger')


def check_release(root, binding, *, executing):
    release=read_binding(root,binding,maximum=1024**2)
    need(set(release)=={'schema','execution_released','declaration','output_directory','caps',
        'source_commit','source_bindings','runtime','independent_source_review','upstream'},'release schema')
    need(release['schema']=='hbe-v5-torsion-evaluation-release-v1'
         and type(release['execution_released']) is bool,'explicit release')
    need(not executing or release['execution_released'],'separate root release required')
    need(release['declaration']=={'path':DECLARATION,'sha256':DECLARATION_SHA256}
         and release['output_directory']==OUTPUT and release['caps']==CAPS,'frozen scope/caps')
    declaration=read_binding(root,release['declaration'])
    need(declaration['caps']==CAPS and declaration['fixed_fit_sha256']==FIT_SHA,'declaration')
    if executing:
        need(isinstance(release['source_commit'],str) and re.fullmatch('[a-f0-9]{40}',release['source_commit']),
             'committed source required')
        review=read_binding(root,release['independent_source_review'])
        candidate=dict(release,execution_released=False,source_commit=None,
                       independent_source_review=None,upstream=None)
        need(review.get('schema')=='hbe-v5-torsion-source-review-v1' and review.get('decision')=='GO'
             and review.get('release_candidate_sha256')==sha(canonical(candidate)),'exact source review')
        check_upstream_shape(root,release['upstream'])
    else:
        need(release['upstream'] is None or release['execution_released'],'unreleased dynamic inputs')
    return release,declaration


def verify_sources(root,release,declaration,*,committed):
    base=read_binding(root,declaration['source_template'])
    need(set(release['source_bindings'])==set(base['source_bindings'])|{CORE,RUNNER},'source closure')
    need(all(release['source_bindings'][p]==d for p,d in base['source_bindings'].items()),'source drift')
    need(release['runtime']==base['runtime'],'existing runtime closure only')
    authority.verify_sources(root,release,{'inherited_source_bindings':dict(release['source_bindings'])},
                             committed=committed)


def check_upstream_shape(root,upstream):
    need(type(upstream) is dict and set(upstream)==set(UPSTREAM_NAMES)|{'independent_review'},
         'all exact upstream bindings required')
    for binding in upstream.values():
        need(type(binding) is dict and set(binding)=={'path','sha256'}
             and type(binding['sha256']) is str and re.fullmatch('[a-f0-9]{64}',binding['sha256']),
             'exact upstream path/hash')
        local(root,binding['path'])
    parent=Path(upstream['publication']['path']).parent
    names={'publication':'publication.json','terminal':'terminal.json','freeze':'freeze.json',
           'predictions':'predictions.json','numerical':'numerical.json',
           'continuation_ledger':'continuation-ledger.json'}
    need(str(parent)=='outputs/mechanics/hbe-v5-fixed-fit-confirmation-v1/attempt-01',
         'exact predecessor attempt')
    for key,name in names.items():need(upstream[key]['path']==str(parent/name),'upstream location')


def admit_upstream(root,release,declaration):
    """Small saved metadata only. No primitive replay, archive, fit-body or member IO."""
    from scripts import mechanics_hbe_evaluation as evaluation
    upstream=release['upstream'];check_upstream_shape(root,upstream)
    records={k:read_binding(root,upstream[k],maximum=4*1024**2) for k in upstream}
    review=records['independent_review'];public=records['publication'];freeze=records['freeze']
    expected={k:upstream[k] for k in UPSTREAM_NAMES}
    need(review.get('schema')=='hbe-v5-fixed-fit-heldout-admission-v1' and review.get('decision')=='GO'
         and {k:review.get(k) for k in UPSTREAM_NAMES}==expected
         and review.get('fixed_fit_sha256')==FIT_SHA
         and review.get('mu_Pa')==MU and review.get('scale')==SCALE
         and review.get('fitted_checks')=={'compression':'PASS','tension':'PASS'}
         and review.get('states_per_native_run')==121 and review.get('native_calls')==2
         and review.get('refit_calls')==0 and review.get('held_out_member_reads')==0
         and review.get('measured_member_reads')==0 and review.get('physical_validation_pass') is None
         and review.get('held_out_access_released') is False
         and review.get('scope')=='accepted saved fixed-fit confirmation and pre-reveal prediction freeze evidence only'
         and review.get('release')==public.get('release'),
         'independently accepted exact predecessor evidence, not access permission, required')
    parent=local(root,upstream['publication']['path']).parent
    need(not (parent/'finalization-failure.json').exists()
         and not (parent/'finalization-failure.json').is_symlink(),'failed upstream publication')
    need(public.get('schema')=='hbe-v5-fixed-fit-publication-v1' and public.get('accepted') is True
         and public.get('physical_validation_pass') is None,'accepted upstream publication')
    for key in UPSTREAM_NAMES[1:]:
        name=Path(upstream[key]['path']).name;row=public['outputs'].get(name)
        raw=read_binding(root,upstream[key],maximum=4*1024**2,decode=False)
        need(row=={'bytes':len(raw),'sha256':sha(raw)},'published small output differs')
    terminal=records['terminal']
    need(terminal.get('status')=='completed_fitted_confirmation_and_prediction_freeze'
         and terminal.get('native_calls')==2 and terminal.get('freeze')==upstream['freeze']
         and terminal.get('release')==public['release']==freeze.get('release'), 'completed predecessor')
    stages=terminal.get('stages',{})
    need(set(stages)=={'prepare','compression','tension','readout'},'four completed upstream stages')
    for stage in stages.values():
        cleanup=stage['cleanup']
        need(cleanup.get('contained') is True and cleanup.get('direct_child_reaped') is True
             and cleanup.get('remaining_members')==[] and cleanup.get('fallback_used') is False
             and cleanup.get('exit_code')==0 and cleanup.get('errors')==[], 'unclean upstream child')
    need(freeze.get('schema')=='hbe-v5-fixed-parameter-prediction-freeze-v1'
         and freeze.get('roles')==declaration['roles'] and freeze.get('protocol')==declaration['protocol']
         and freeze.get('fixed_fit',{}).get('fit',{}).get('sha256')==FIT_SHA
         and freeze.get('mu_Pa')==MU and freeze.get('scale')==SCALE
         and freeze.get('refit_calls')==0 and freeze.get('native_calls')==2
         and freeze.get('mesher_calls')==0 and freeze.get('measured_member_reads')==0
         and freeze.get('held_out_member_reads')==0 and freeze.get('held_out_access_released') is False
         and freeze.get('patient_tool_mechanics_admitted') is False
         and freeze.get('physical_validation_pass') is None
         and freeze.get('predictions')==upstream['predictions']
         and freeze.get('numerical')==upstream['numerical'],'fixed freeze semantics')
    from scripts.mechanics_hbe_v5_fitted_confirmation import require_confirmations
    require_confirmations(records['numerical'])
    ledger=records['continuation_ledger']
    need(ledger=={'phase':'freeze_saved','freeze':upstream['freeze'],
        'previous_fit_ledger':freeze['original_fit_ledger'],'measured_member_reads':0,'refit_calls':0},
        'predecessor freeze chronology')
    read_binding(root,freeze['original_fit_ledger'],decode=False)  # Prior ledger, never fit body.
    predictions=records['predictions']
    need(predictions.get('schema')=='hbe-torsion-prediction-v1'
         and set(predictions)=={'schema','reference','fitted'}
         and set(predictions['reference'])==set(TORSION)==set(predictions['fitted']),'frozen pair')
    curves={}
    for branch in TORSION:
        ref=predictions['reference'][branch];fitted=predictions['fitted'][branch]
        need(set(ref)=={'coordinate','response'} and set(fitted)=={'coordinate','response','response_unit'}
             and fitted['response_unit']=='Nm' and len(ref['coordinate'])==121,'exact prediction schema')
        reference=evaluation.curve(branch,ref['coordinate'],ref['response'])
        curve=evaluation.curve(branch,fitted['coordinate'],fitted['response'])
        need(curve.coordinate==reference.coordinate
             and curve.response==tuple(v*SCALE for v in reference.response),'unchanged frozen scaling')
        endpoint=(-1 if branch=='torsion_neg' else 1)*.15*.00489159/.004
        # The unchanged reference schedule predicate admits serialization error;
        # these coordinates are never replaced, snapped or extrapolated.
        need(all(math.isclose(x,endpoint*i/120,rel_tol=2e-14,abs_tol=1e-18)
                 for i,x in enumerate(curve.coordinate)),'unchanged frozen schedule')
        curves[branch]=curve
    return records,curves


def make_study(root,rb,release,declaration,*,recheck):
    from scripts import mechanics_hbe_access as access
    class TorsionStudy(access.ReleasedStudy):
        def _release(self):
            recheck()
            current,decl=check_release(root,rb,executing=True)
            need(current==release and decl==declaration,'authority changed')
            return self.protocol,self.roles,compatible

        def _events(self):
            if not self.ledger_path.exists():return []
            from scripts.mechanics_hbe_v5_axial_calibration_experiment import regular_bytes,unique_json
            rows=[unique_json(line) for line in regular_bytes(self.ledger_path,1024**2).splitlines()]
            phases=['upstream_freeze_admitted','held_out_attempt','held_out_completed']
            need(1<=len(rows)<=3,'single evaluation ledger')
            names=[m['path'] for m in self.roles['held_out_validation']['members']]
            for i,row in enumerate(rows):
                need(row.get('sequence')==i and row.get('phase')==phases[i]
                     and row.get('protocol_sha256')==self.protocol_binding['sha256']
                     and row.get('release_sha256')==rb['sha256']
                     and row.get('members')==([] if i==0 else names),'ledger chronology/identity')
            need(rows[0].get('upstream')==release['upstream'],'predecessor identity')
            return rows

        def _read_selected(self,roles,members,branches,schemas,phase):
            self._release()
            need(roles==self.roles and members==self.roles['held_out_validation']['members']
                 and tuple(branches)==TORSION and phase=='held_out','two torsion members only')
            need(len(self._events())==1,'attempt already consumed including partial exposure')
            self._check_schema_release(schemas,TORSION,compatible)
            local(root,self.roles['source']['archive_path'])
            return super()._read_selected(roles,members,branches,schemas,phase)

        def read_calibration(self,*args,**kwargs):raise ValueError('No calibration authority')
        def freeze_predictions(self,*args,**kwargs):raise ValueError('No new prediction/freeze authority')
        def evaluate_held_out(self,*args,**kwargs):raise ValueError('Use exact successor admission')

    compatible={**release,'execution':{'csv_schemas':declaration['csv_schemas']}}
    study=TorsionStudy(root,declaration['protocol'],rb,ledger_path=OUTPUT+'/access.jsonl')
    need(not study._events(),'exclusive evaluation attempt')
    return study


def validate_terminal_ledger(root,events,rb,release,declaration,state):
    roles=read_binding(root,declaration['roles'])
    names=[m['path'] for m in roles['held_out_validation']['members']]
    need([e.get('phase') for e in events]==['upstream_freeze_admitted','held_out_attempt','held_out_completed'],
         'one completed evaluation sequence')
    for i,event in enumerate(events):
        need(event.get('sequence')==i and event.get('protocol_sha256')==declaration['protocol']['sha256']
             and event.get('release_sha256')==rb['sha256']
             and event.get('members')==([] if i==0 else names),'terminal ledger identity')
    need(events[0].get('upstream')==release['upstream']
         and events[2].get('member_sha256')==state['member_sha256']
         and set(state['member_sha256'])==set(TORSION),'upstream/member digests')
    for digest in state['member_sha256'].values():
        need(type(digest) is str and re.fullmatch('[a-f0-9]{64}',digest),'member SHA required')


def run_evaluation(root,rb,release,declaration,*,deadline):
    from scripts import mechanics_hbe_evaluation as evaluation
    from scripts import mechanics_hbe_v5_n8_one_shot as io
    out=local(root,OUTPUT)
    need(out.is_dir() and not any((out/name).exists() for name in
         ('state.json','access.jsonl','held-out-metrics.json','observed-torsion.json')),
         'worker output already consumed')
    state={'schema':'hbe-v5-torsion-state-v1','status':'preflight','fit_calls':0,'native_calls':0,
        'mesher_calls':0,'held_out_member_reads':0,'held_out_access_attempted':False,
        'held_out_responses_accessed':False,'physical_validation_pass':None,'empirical_tolerance':None,
        'patient_tool_mechanics_admitted':False,'independent_donor_validation':False,
        'automatic_retry':False,'new_freeze_saved':False}
    def recheck():
        need(time.monotonic()<deadline,'worker deadline')
        current,decl=check_release(root,rb,executing=True)
        need(current==release and decl==declaration,'root authority changed')
        verify_sources(root,release,declaration,committed=False);audit_loaded(root,release)
        admit_upstream(root,release,declaration)
        need(time.monotonic()<deadline,'worker deadline')
    io.durable_json(out/'state.json',state)
    try:
        recheck();records,predicted=admit_upstream(root,release,declaration)
        study=make_study(root,rb,release,declaration,recheck=recheck)
        study._audit('upstream_freeze_admitted',[],upstream=release['upstream'])
        state.update(status='reading_exact_held_out_torsion',held_out_access_attempted=True,
                     held_out_responses_accessed=None,held_out_member_reads=None)
        io.durable_json(out/'state.json',state)
        observed=study._read_selected(study.roles,study.roles['held_out_validation']['members'],
            TORSION,declaration['csv_schemas'],'held_out')
        state.update(held_out_responses_accessed=True,held_out_member_reads=2)
        need(time.monotonic()<deadline,'worker deadline')
        observed_record={'schema':'hbe-held-out-torsion-curves-v1','release':rb,
            'curves':{b:{'coordinate_rad':list(observed[b].coordinate),
                         'torque_Nm':list(observed[b].response),'source_sha256':observed[b].source_sha256}
                      for b in TORSION}}
        observed_sha=exclusive_json(out/'observed-torsion.json',observed_record,maximum=1024**2)
        metrics=evaluation.paired_metrics(observed,predicted,characteristic_response_scale=MU*.004**3)
        need(metrics['physical_validation_pass'] is None and metrics['empirical_tolerance'] is None,
             'descriptive comparison only')
        metrics_sha=exclusive_json(out/'held-out-metrics.json',metrics,maximum=4*1024**2)
        recheck()
        state.update(status=DONE,release=rb,declaration=release['declaration'],upstream=release['upstream'],
            held_out_metrics={'path':OUTPUT+'/held-out-metrics.json','sha256':metrics_sha},
            observed_torsion={'path':OUTPUT+'/observed-torsion.json','sha256':observed_sha},
            member_sha256={b:observed[b].source_sha256 for b in TORSION},
            fixed_fit_sha256=FIT_SHA,mu_Pa=MU,scale=SCALE,
            calibration_quality='unchanged_poor_within_specimen_fit_not_physical_validation')
    except BaseException as error:
        state.update(status=FAILED,error={'type':type(error).__name__,'message':str(error)[:2048]})
        raise
    finally:io.durable_json(out/'state.json',state)
    return state

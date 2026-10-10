"""One axial calibration terminal; existing specimen roles and estimator unchanged.

No archive is opened on import or metadata preflight. This successor consumes
the exact saved numerical qualification, not a caller's numerical-pass flag.
The one-shot root release is separate from this source. Torsion stays closed.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat
import subprocess
import sys
import time

DECLARATION = 'manifests/experiments/hbe-01-03-v5-axial-calibration-v1.json'
DECLARATION_SHA256 = '31b4c3b85e4fdab0010be28300c8e6fda2930c48659ee78792dc6f225c2cf794'
CORE = 'scripts/mechanics_hbe_v5_axial_calibration.py'
RUNNER = 'scripts/mechanics_hbe_v5_axial_calibration_experiment.py'
OUTPUT = 'build/hbe-v5-axial-calibration-v1/attempt-01'
AXIAL = ('compression', 'tension')
CAPS = {'worker_seconds':60, 'cleanup_seconds':10, 'total_seconds':70,
        'sampled_group_rss_bytes':512*1024**2, 'new_output_bytes':16*1024**2,
        'threads':1, 'attempts':1, 'fit_calls':1, 'native_calls':0,
        'mesher_calls':0, 'held_out_member_reads':0, 'automatic_retry':False}
DONE = 'completed_axial_calibration_pending_fitted_confirmation'
HEX64 = re.compile(r'[0-9a-f]{64}\Z')


def need(condition, reason):
    if not condition:
        raise ValueError(reason)


def canonical(value):
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False)+'\n').encode()


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def local(root, relative):
    need(isinstance(relative,str) and relative, 'relative path required')
    p=Path(relative)
    need(not p.is_absolute() and '..' not in p.parts, 'path escapes root')
    root=Path(root).resolve(); result=root/p
    need(not any(x.is_symlink() for x in (result,*result.parents) if x.is_relative_to(root)),
         'linked input/output forbidden')
    return result


def read_binding(root, binding, *, maximum=16*1024**2, decode=True):
    need(isinstance(binding,dict) and set(binding)=={'path','sha256'}
         and isinstance(binding['sha256'],str) and HEX64.fullmatch(binding['sha256']),
         'exact SHA binding required')
    path=local(root,binding['path'])
    fd=os.open(path,os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK)
    try:
        before=os.fstat(fd)
        need(stat.S_ISREG(before.st_mode) and 0<before.st_size<=maximum,'bounded regular input required')
        with os.fdopen(fd,'rb',buffering=0,closefd=False) as stream:
            raw=stream.read(maximum+1)
        after=os.fstat(fd)
        identity=lambda s:(s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns)
        need(identity(before)==identity(after)==identity(path.lstat()) and len(raw)==before.st_size
             and sha(raw)==binding['sha256'],'input changed or digest differs')
    finally:
        os.close(fd)
    if not decode:
        return raw
    def pairs(rows):
        out={}
        for k,v in rows:
            need(k not in out,'duplicate JSON key');out[k]=v
        return out
    return json.loads(raw,object_pairs_hook=pairs,
        parse_constant=lambda _: (_ for _ in ()).throw(ValueError('nonfinite JSON')))


def file_binding(root,path):
    path=Path(path);raw=path.read_bytes()
    return {'path':str(path.relative_to(root)), 'sha256':sha(raw)}


def exclusive_json(path,value,*,maximum=1024**2):
    raw=canonical(value);need(len(raw)<=maximum,'output cap')
    fd=os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
    with os.fdopen(fd,'wb') as stream:
        stream.write(raw);stream.flush();os.fsync(stream.fileno())
    fd=os.open(Path(path).parent,os.O_RDONLY)
    try:os.fsync(fd)
    finally:os.close(fd)
    return sha(raw)


def check_release(root,binding,*,executing):
    release=read_binding(root,binding,maximum=1024**2)
    keys={'schema','execution_released','declaration','output_directory','caps',
          'source_commit','source_bindings','runtime','independent_source_review'}
    need(set(release)==keys and release['schema']=='hbe-v5-axial-fit-release-v1','release schema')
    need(type(release['execution_released']) is bool,'explicit release flag')
    need(not executing or release['execution_released'] is True,'root release required')
    need(release['declaration']=={'path':DECLARATION,'sha256':DECLARATION_SHA256}
         and release['output_directory']==OUTPUT and release['caps']==CAPS,'exact declaration/output/caps')
    declaration=read_binding(root,release['declaration'])
    need(declaration['schema']=='hbe-v5-axial-fit-declaration-v1'
         and declaration['caps']==CAPS,'declaration scope')
    need(declaration['authorized_held_out_members']==[]
         and declaration['native_calls']==0 and declaration['fit_calls']==1,'fit-only declaration')
    if executing:
        need(isinstance(release['source_commit'],str)
             and re.fullmatch(r'[0-9a-f]{40}',release['source_commit']),'committed source required')
        review=read_binding(root,release['independent_source_review'],maximum=1024**2)
        candidate={**release,'execution_released':False,'source_commit':None,'independent_source_review':None}
        need(review.get('schema')=='hbe-v5-axial-fit-source-review-v1'
             and review.get('decision')=='GO'
             and review.get('release_candidate_sha256')==sha(canonical(candidate)),
             'independent source review must bind exact candidate')
    return release,declaration


def verify_sources(root,release,declaration,*,committed):
    expected=dict(declaration['inherited_source_bindings'])
    need(set(release['source_bindings'])==set(expected)|{CORE,RUNNER},'exact executing source inventory')
    for p,d in expected.items():
        need(release['source_bindings'][p]==d,'inherited source drift')
    for p,d in release['source_bindings'].items():
        need(p.startswith(('scripts/','launchers/')) and p.endswith('.py'),'source path scope')
        raw=read_binding(root,{'path':p,'sha256':d},maximum=1024**2,decode=False)
        if committed:
            env={k:v for k,v in os.environ.items() if not k.startswith('GIT_')}
            env.update(GIT_CONFIG_NOSYSTEM='1',GIT_CONFIG_GLOBAL='/dev/null',GIT_CONFIG_SYSTEM='/dev/null')
            # Small exact historical Git blobs; unrelated later HEAD commits do not invalidate a release.
            blob=subprocess.run(['/usr/bin/git','cat-file','blob',release['source_commit']+':'+p],
                cwd=root,env=env,capture_output=True,check=True,timeout=2).stdout
            need(blob==raw,'committed source bytes differ')
    runtime=release['runtime']
    need(set(runtime)=={'executable','sha256','site_packages','files'},'runtime schema')
    executable=Path(sys.executable).resolve()
    need(executable==Path(runtime['executable']).resolve() and sha(executable.read_bytes())==runtime['sha256'],
         'interpreter differs')
    for p,d in runtime['files'].items():
        path=Path(p)
        need(path.is_absolute() and path.is_file() and not path.is_symlink(),'runtime regular file')
        need(sha(path.read_bytes())==d,'runtime dependency changed')


def audit_loaded(root,release):
    for module in tuple(sys.modules.values()):
        origin=getattr(module,'__file__',None)
        if not origin:continue
        path=Path(origin).resolve()
        if path.is_relative_to(root) and path.parts[len(root.parts)] in ('scripts','launchers'):
            relative=str(path.relative_to(root));expected=release['source_bindings'].get(relative)
            need(expected is not None and sha(path.read_bytes())==expected,'unbound loaded project source')
        elif 'site-packages' in path.parts:
            need(release['runtime']['files'].get(str(path))==sha(path.read_bytes()),'unbound loaded dependency')


def admit_evidence(root,declaration):
    """Consume hash-pinned saved qualification and legacy failure; no raw logs or archives."""
    inputs=declaration['evidence']; records={k:read_binding(root,b) for k,b in inputs.items()
                                         if k not in ('independent_report','v3_access_ledger')}
    read_binding(root,inputs['independent_report'],decode=False)
    ledger=read_binding(root,inputs['v3_access_ledger'],decode=False)
    events=[json.loads(line) for line in ledger.splitlines()]
    need([e['phase'] for e in events]==['calibration_attempt','calibration_completed'],
         'preserved v3 exposure chronology differs')
    roles,protocol=records['roles'],records['protocol']
    need(protocol['roles']==inputs['roles'],'original role binding')
    need([m['path'] for m in roles['calibration']['members']]==declaration['permitted_members'],
         'exact two calibration members required')
    need(records['v3_state']['status']=='failed_or_incomplete'
         and records['v3_state']['native_calls']==0
         and records['v3_state']['held_out_responses_accessed'] is False
         and not any(k in records['v3_state'] for k in ('fit','freeze','predictions')),
         'v3 terminal history changed')
    need(records['v3_publication']['accepted'] is False,'v3 failure publication changed')
    comparison,manifest,receipt,summary,audit=(records[k] for k in
        ('comparison','manifest','comparison_receipt','independent_summary','independent_audit'))
    need(comparison['schema']=='hbe-v5-native-numerical-comparison-v1'
         and comparison['comparison_manifest']==inputs['manifest']
         and comparison['numerical_qualification_passed'] is True
         and comparison['native_output_admitted'] is True
         and comparison['physical_validation_pass'] is None
         and comparison['calibration_released'] is False
         and comparison['measured_response_accessed'] is False,'accepted numerical result required')
    need(receipt['status']=='completed_numerical_comparison'
         and receipt['comparison_sha256']==inputs['comparison']['sha256']
         and receipt['cleanup']['contained'] is True and receipt['cleanup']['direct_child_reaped'] is True
         and receipt['cleanup']['remaining_members']==[] and receipt['cleanup']['errors']==[],
         'completed numerical supervision required')
    need(summary['comparison_sha256']==inputs['comparison']['sha256']
         and summary['manifest_sha256']==inputs['manifest']['sha256']
         and summary['receipt_sha256']==inputs['comparison_receipt']['sha256']
         and summary['numerical_qualification_passed'] is True and summary['physical_validation_pass'] is None
         and audit['status']=='PASS' and isinstance(audit['files_pre_post_identical'],dict)
         and audit['files_pre_post_identical'][inputs['comparison']['path']]['sha256']==inputs['comparison']['sha256']
         and audit['files_pre_post_identical'][inputs['manifest']['path']]['sha256']==inputs['manifest']['sha256']
         and audit['numerical_qualification_passed'] is True,'independent saved verdict required')
    from scripts import mechanics_hbe_v5_comparison as numeric
    from scripts import mechanics_hbe_evaluation as evaluation
    from scripts import mechanics_hbe_branch_calibration_v4 as coordinates
    references={}
    for branch in AXIAL:
        spec=declaration['references'][branch];run_id=spec['run_id']
        need(manifest['rows'][run_id]['readout']==spec['readout'],'qualified reference origin')
        row=read_binding(root,spec['readout'])
        view,passed=numeric._numerical_view(records['v5'],records['v4'],run_id,row)
        need(passed and view['steps']==120 and view['frame_count']==121 and view['mu_Pa']==1000.,
             'complete 120-increment reference required')
        need(view['mesh_N']==(36 if branch=='compression' else 24),'frozen branch mesh')
        coordinates.strict_coverage(records['v4'],branch,coordinates.frozen_coordinates(records['v4'],branch),
            view['load_coordinate_m'],member_sha256=spec['member']['sha256'])
        references[branch]=evaluation.curve(branch,view['load_coordinate_m'],view['applied_force_N'],
            source_sha256=spec['readout']['sha256'])
    return records,references


def make_study(root,release_binding,release,declaration,*,recheck):
    from scripts import mechanics_hbe_access as access
    class AxialStudy(access.ReleasedStudy):
        def _release(self):
            recheck()
            checked,decl=check_release(root,release_binding,executing=True)
            need(checked==release and decl==declaration,'release changed')
            return self.protocol,self.roles,compatible

        def _read_selected(self,roles,members,branches,schemas,phase):
            self._release()
            need(phase=='calibration' and tuple(branches)==AXIAL
                 and members==self.roles['calibration']['members'],'only two axial members admitted')
            local(root,self.roles['source']['archive_path'])
            return super()._read_selected(roles,members,branches,schemas,phase)

        def read_calibration(self,schemas):
            need(not self._events(),'attempt already consumed, including partial exposure')
            return super().read_calibration(schemas)

        def freeze_predictions(self,*args,**kwargs):
            raise ValueError('No torque prediction freeze in axial-only phase')

        def evaluate_held_out(self,*args,**kwargs):
            raise ValueError('No held-out authority in axial-only phase')
    # Existing reader expects this exact CSV field; no other execution permission is inferred.
    compatible={**release,'execution':{'csv_schemas':declaration['csv_schemas']}}
    study=AxialStudy(root,declaration['evidence']['protocol'],release_binding,
        ledger_path=OUTPUT+'/access.jsonl')
    return study


def fit_terminal(calibration,references,declaration,*,on_fit=lambda:None):
    """Pure unchanged estimator after exact member/coordinate checks; no IO permission."""
    from scripts import mechanics_hbe_evaluation as evaluation
    from scripts import mechanics_hbe_branch_calibration_v4 as coordinates
    need(set(calibration)==set(AXIAL) and set(references)==set(AXIAL),'two branches only')
    for branch in AXIAL:
        curve=evaluation.checked_curve(calibration[branch]);spec=declaration['references'][branch]
        need(curve.branch==branch and len(curve.coordinate)==30
             and curve.source_sha256==spec['member']['sha256'],'exact measured member and row count')
        need(coordinates.coordinate_hash(curve.coordinate)==spec['coordinate_sha256'],
             'frozen measured coordinate identity')
    # Coverage is checked for both branches before the sole fit invocation.
    for branch in AXIAL:evaluation.interpolate(references[branch],calibration[branch].coordinate)
    on_fit()
    fit=evaluation.fit_scale(calibration,references,reference_mu_Pa=1000.,radius_m=.004)
    predicted={b:evaluation.scale_prediction(references[b],fit['scale']) for b in AXIAL}
    metrics=evaluation.paired_metrics(calibration,predicted,characteristic_response_scale=fit['mu_Pa']*.004**2)
    return fit,metrics


def run_fit(root,release_binding,release,declaration,*,deadline):
    """One worker body. Parent must already reserve and supervise this output directory."""
    root=Path(root).resolve();out=local(root,OUTPUT)
    checked,decl=check_release(root,release_binding,executing=True)
    need(checked==release and decl==declaration,'exact root authority required')
    need(out.is_dir() and not any((out/name).exists() or (out/name).is_symlink()
        for name in ('state.json','access.jsonl','fit.json','calibration-metrics.json')),
        'worker output already consumed')
    state={'schema':'hbe-v5-axial-fit-state-v1','status':'preflight','fit_calls':0,
           'native_calls':0,'mesher_calls':0,'held_out_member_reads':0,
           'calibration_access_attempted':False,'calibration_responses_accessed':False,
           'physical_validation_pass':None,'automatic_retry':False,'freeze_saved':False}
    def check_clock():need(time.monotonic()<deadline,'worker deadline')
    def recheck():
        check_clock()
        for binding in declaration['evidence'].values():read_binding(root,binding,decode=False)
        for row in declaration['references'].values():read_binding(root,row['readout'],decode=False)
        verify_sources(root,release,declaration,committed=False)
        audit_loaded(root,release);check_clock()
    from scripts import mechanics_hbe_v5_n8_one_shot as io
    io.durable_json(out/'state.json',state)
    try:
        check_clock();_,references=admit_evidence(root,declaration);recheck()
        study=make_study(root,release_binding,release,declaration,recheck=recheck)
        state.update(status='reading_exact_axial_members',calibration_access_attempted=True,
                     calibration_responses_accessed=None)
        io.durable_json(out/'state.json',state)
        calibration=study.read_calibration(declaration['csv_schemas'])
        state['calibration_responses_accessed']=True
        check_clock()
        def mark_fit():
            need(state['fit_calls']==0,'single estimator invocation')
            state.update(status='fitting_once',fit_calls=1)
            io.durable_json(out/'state.json',state)
        fit,metrics=fit_terminal(calibration,references,declaration,on_fit=mark_fit)
        fit_sha=exclusive_json(out/'fit.json',fit);metric_sha=exclusive_json(out/'calibration-metrics.json',metrics)
        recheck()
        # Exact archive fixity was checked before/after selected reads by the inherited reader.
        # No repeat measured read, native solve, torque prediction or freeze_saved event.
        state.update(status=DONE,fit={'path':OUTPUT+'/fit.json','sha256':fit_sha},
            calibration_metrics={'path':OUTPUT+'/calibration-metrics.json','sha256':metric_sha},
            declaration=release['declaration'],release=release_binding,
            reference_readouts={b:declaration['references'][b]['readout'] for b in AXIAL},
            calibration_member_sha256={b:calibration[b].source_sha256 for b in AXIAL},
            qualification=declaration['evidence']['comparison'],
            held_out_access_released=False,patient_tool_mechanics_admitted=False)
        check_clock()
    except BaseException as error:
        state.update(status='failed_axial_calibration_attempt',error={'type':type(error).__name__,
                     'message':str(error)[:2048]})
        raise
    finally:
        io.durable_json(out/'state.json',state)
    return state

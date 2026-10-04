#!/usr/bin/env python3
"""Eight existing numerical controls; only the linear-solver subtree changes.

Preparation never invokes FEBio. Execution requires a separate exact-source
release and an existing, verified repaired runtime identity. No data readers,
acquisition/build operations, parameter overrides or retry path are exposed.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
VERSION='mechanics-accelerate-controls-v1'
BUNDLE='artifacts/'+VERSION
DECLARATION=BUNDLE+'/declaration.json'
HEX_CASES=('zero','translation','finite_stretch','shear','shear_double_stiffness')
TET_CASES=('tet10_affine','mpc_translation','mpc_nonrigid')
CASES=HEX_CASES+TET_CASES
CAPS={'aggregate_seconds':60,'process_group_rss_bytes':3*1024**3,
      'numerical_threads':1,'maximum_cases':8,'time_steps':4,'retries':0}
ORIGINAL={name:('artifacts/mechanics-febio-verification-v1/decks/' if name in HEX_CASES else
                'artifacts/mechanics-patient-constraints-runtime-v1/decks/')+name+'.feb' for name in CASES}
SOURCES=('scripts/mechanics_accelerate_controls.py','scripts/mechanics_hbe_backend.py',
         'scripts/mechanics_hbe_access.py','scripts/mechanics_hbe_evaluation.py',
         'scripts/mechanics_patient_constraints_run.py','scripts/mechanics_patient_constraints.py',
         'scripts/mechanics_febio_verification.py','scripts/febio_runtime.py')
CLOSURE=(*SOURCES,'artifacts/febio-runtime-investigation-v1/prospective-runtime.json',
         'artifacts/mechanics-febio-verification-v1/decks/manifest.json',
         'artifacts/mechanics-patient-constraints-runtime-v1/decks/manifest.json',
         DECLARATION,*ORIGINAL.values(),*(BUNDLE+'/decks/'+name+'.feb' for name in CASES))


def load(path,name):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    if Path(module.__file__).resolve()!=Path(path).resolve():raise ValueError('Wrong imported source')
    return module


# Reuse reviewed byte hashing, atomic receipts and failure-preserving rechecks.
common=load(ROOT/'scripts/mechanics_patient_constraints_run.py','accelerate_control_common')
sha,write,unchanged=common.sha,common.write,common.unchanged
backend=load(ROOT/'scripts/mechanics_hbe_backend.py','accelerate_control_backend')


ORIGINAL_MANIFESTS={
 'artifacts/mechanics-febio-verification-v1/decks/manifest.json':'2492e1d20c25a3d0adfc34faf51d88206fe43c4d6897e63db4cf7ea26c96c3d7',
 'artifacts/mechanics-patient-constraints-runtime-v1/decks/manifest.json':'73a20d14551742159f05145bfd39444fde4d8bc81415d6139d85e8d890d0c330'}
CHECKER_PINS={'scripts/mechanics_febio_verification.py':'b3f373743d6d107de929bf3283c167c6a3e0a68dfe74396ea0f1f8ad7cce2be5',
              'scripts/mechanics_patient_constraints.py':'04119660cf0a230428fe862d7897ea17953c51cf8756f87fa3152b426e303c66'}


def original_bindings():
    for name,digest in {**ORIGINAL_MANIFESTS,**CHECKER_PINS}.items():
        if sha(ROOT/name)!=digest:raise ValueError('Original numerical control changed: '+name)
    for path in ORIGINAL_MANIFESTS:
        declared=json.loads((ROOT/path).read_text())
        for name,digest in declared['decks'].items():
            if sha((ROOT/path).parent/name)!=digest:raise ValueError('Original deck changed')


def prepare(directory):
    original_bindings()
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=False)
    (directory/'decks').mkdir()
    manifest={'schema':VERSION+'-declaration','status':'prepared_not_executed',
              'solver_backend':'accelerate','profile_id':backend.PROFILE_ID,
              'case_order':list(CASES),'caps':CAPS,'cases':{},
              'sources':{name:sha(ROOT/name) for name in SOURCES},
              'change_scope':'Only exact existing Skyline linear_solver subtree replaced; all other original deck bytes preserved.',
              'runtime_requirement':'Separate release binds existing repaired runtime identity SHA; missing identity is a refusal.',
              'backend_use_evidence':'Each case requires actual selecting-linear-solver accelerate console line, verified exact solver XML, executed deck hash and same runtime identity.',
              'authority':'Prepared analytical software controls only; no execution authorization or material/patient validation.'}
    for name in CASES:
        original=ROOT/ORIGINAL[name];adapted=directory/'decks'/(name+'.feb')
        adapted.write_text(backend.transform_deck(original.read_bytes()),encoding='utf-8',newline='')
        backend.verify_deck(original.read_bytes(),adapted.read_bytes())
        manifest['cases'][name]={'original_path':ORIGINAL[name],'original_sha256':sha(original),
            'adapted_path':BUNDLE+'/decks/'+name+'.feb','adapted_sha256':sha(adapted),
            'checker':'hex8' if name in HEX_CASES else 'tet10_mpc'}
    write(directory/'declaration.json',manifest)
    return manifest


def declaration():
    original_bindings()
    value=json.loads((ROOT/DECLARATION).read_text())
    if (value.get('schema')!=VERSION+'-declaration' or value.get('status')!='prepared_not_executed'
            or value.get('caps')!=CAPS or value.get('case_order')!=list(CASES)
            or value.get('solver_backend')!='accelerate' or value.get('profile_id')!=backend.PROFILE_ID
            or set(value.get('cases',{}))!=set(CASES) or set(value.get('sources',{}))!=set(SOURCES)):
        raise ValueError('Fixed declaration/cases/caps changed')
    for name,expected in value['sources'].items():
        if sha(ROOT/name)!=expected:raise ValueError('Declared source changed: '+name)
    for name in CASES:
        row=value['cases'][name]
        if (row.get('original_path')!=ORIGINAL[name] or row.get('adapted_path')!=BUNDLE+'/decks/'+name+'.feb'
                or row.get('checker')!=('hex8' if name in HEX_CASES else 'tet10_mpc')):
            raise ValueError('Declared case binding changed')
        for prefix in ('original','adapted'):
            if sha(ROOT/row[prefix+'_path'])!=row[prefix+'_sha256']:raise ValueError('Declared deck changed')
        backend.verify_deck((ROOT/row['original_path']).read_bytes(),(ROOT/row['adapted_path']).read_bytes())
    return value


def runtime_binding(repository,binding):
    # Backend's byte verifier imports these two pure helper modules. Pin the
    # archived import origin; this code never calls their measured-data APIs.
    sys.path.insert(0,str(ROOT))
    from scripts import mechanics_hbe_access, mechanics_hbe_evaluation
    for module in (mechanics_hbe_access,mechanics_hbe_evaluation):
        if Path(module.__file__).resolve()!=ROOT/'scripts'/(module.__name__.split('.')[-1]+'.py'):
            raise ValueError('Runtime verifier helper imported outside released archive')
    return backend.verify_runtime_binding(repository,binding)


def baseline(release_path):
    release_path=Path(release_path).resolve();release=json.loads(release_path.read_text())
    if (release.get('schema')!=VERSION+'-release' or release.get('authorized') is not True
            or not isinstance(release.get('root_release'),str) or not release['root_release'].strip()):
        raise ValueError('Explicit separate execution release required')
    attempt=Path(release.get('attempt_directory',''));commit=release.get('source_commit','')
    if not attempt.is_absolute() or not re.fullmatch(r'[0-9a-f]{40}',commit):
        raise ValueError('One absolute attempt and exact Git commit required')
    if Path(release['source_directory']).resolve()!=ROOT:raise ValueError('Released archive source origin differs')
    repository=Path(release['repository_directory']).resolve()
    if repository==ROOT:raise ValueError('Mutable working-tree execution prohibited')
    if not ROOT.is_relative_to(repository):raise ValueError('Source archive must be inside the repository for evidence publication')
    if not attempt.resolve().is_relative_to(repository):raise ValueError('Attempt must be within bound repository for evidence paths')
    inputs={str(release_path):sha(release_path)}
    for name in CLOSURE:
        raw=subprocess.run(['git','-C',str(repository),'show',f'{commit}:{name}'],
                           capture_output=True,check=True,timeout=5).stdout
        expected=hashlib.sha256(raw).hexdigest()
        if sha(ROOT/name)!=expected:raise ValueError('Source differs from exact Git commit: '+name)
        inputs[str(ROOT/name)]=expected
    declared=declaration()
    bound=runtime_binding(repository,release['runtime_identity']) # Missing identity fails here before a worker.
    inputs.update(bound['inputs'])
    # Bind the actual runtime's build/linkage receipts as well as its executable.
    identity_path=repository/release['runtime_identity']['path']
    for name,digest in bound['runtime'].get('receipts_sha256',{}).items():
        path=(identity_path.parent/name).resolve()
        if not path.is_relative_to(repository):raise ValueError('Runtime receipt escapes repository')
        inputs[str(path)]=digest
    if not all(v['unchanged'] for v in unchanged(inputs).values()):raise ValueError('Input changed before execution')
    return {'schema':VERSION+'-baseline','source_commit':commit,'source_directory':str(ROOT),
            'release_path':str(release_path),'release':release,'attempt_directory':str(attempt.resolve()),
            'input_hashes':inputs,'runtime_identity_sha256':release['runtime_identity']['sha256'],
            'solver_backend':'accelerate','executable':bound['runtime']['executable'],
            'caps':CAPS,'case_order':list(CASES),'declaration':declared,
            'setup_scope':'Source/runtime/hash checks precede one supervised60s worker sequence; no model solves during setup.'}


def initial_result():
    return {'status':'not_started','cases':[{'case':name,'status':'not_executed'} for name in CASES],
            'solver_invocations':0,'no_retry':True,'solver_backend':'accelerate'}


def backend_evidence(console,original,adapted,identity_sha):
    backend.verify_deck(original,adapted)
    selections=[line.strip() for line in console.splitlines() if 'selecting linear solver' in line.lower()]
    if not selections or any(not re.search(r'\bselecting linear solver accelerate\b',line,re.I) for line in selections):
        raise ValueError('Actual Accelerate selection evidence absent or conflicting')
    if not re.fullmatch('[0-9a-f]{64}',identity_sha):raise ValueError('Runtime identity digest missing')
    return {'solver_backend':'accelerate','actual_selection_lines':selections,
            'solver_xml':backend.ACCELERATE_XML,'solver_only_change_verified':True,
            'runtime_identity_sha256':identity_sha,
            'executed_deck_sha256':hashlib.sha256(adapted).hexdigest()}


def worker(output):
    output=Path(output);base=None
    if json.loads((output/'results.json').read_text())!=initial_result():raise ValueError('Worker cannot repeat prior attempt')
    result=initial_result();result['status']='running';start=time.monotonic();failure=None;texts={}
    try:
        base=json.loads((output/'execution-baseline.json').read_text())
        if base!=baseline(base['release_path']):raise ValueError('Worker baseline differs from verified release')
        if output.resolve()!=Path(base['attempt_directory']):raise ValueError('Worker output differs from one released attempt')
        if not all(v['unchanged'] for v in unchanged(base['input_hashes']).values()):raise ValueError('Worker input mismatch')
        hx=load(ROOT/'scripts/mechanics_febio_verification.py','accelerate_hex_checker')
        tet=load(ROOT/'scripts/mechanics_patient_constraints.py','accelerate_tet_checker')
        result.update(runtime_identity_sha256=base['runtime_identity_sha256'],
                      checker_import_paths=[hx.__file__,tet.__file__],numpy_version=hx.np.__version__)
        write(output/'results.json',result)
        for item in result['cases']:
            if failure:item['reason']='blocked_by_first_failure:'+failure;continue
            name=item['case'];directory=output/name
            try:
                remaining=CAPS['aggregate_seconds']-(time.monotonic()-start)
                if remaining<=0:raise TimeoutError('Aggregate worker allowance exhausted')
                directory.mkdir(exist_ok=False)
                source=ROOT/BUNDLE/'decks'/(name+'.feb');deck=directory/source.name;deck.write_bytes(source.read_bytes())
                item.update(status='solver_started',deck_sha256=sha(deck))
                command=[base['executable'],'-noconfig','-no_title','-i',deck.name,'-o',name+'.log']
                item['command']=command;result['solver_invocations']+=1;write(output/'results.json',result)
                before=time.monotonic()
                with (directory/'console.txt').open('x') as log:
                    completed=subprocess.run(command,cwd=directory,stdout=log,stderr=subprocess.STDOUT,timeout=remaining)
                item.update(solver_seconds=time.monotonic()-before,solver_exit_code=completed.returncode)
                if completed.returncode!=0:raise RuntimeError('Nonzero solver exit')
                if sha(deck)!=item['deck_sha256']:raise ValueError('Executed deck changed')
                item['backend_evidence']=backend_evidence(hx.read_bounded_text(directory/'console.txt'),
                    (ROOT/ORIGINAL[name]).read_bytes(),deck.read_bytes(),base['runtime_identity_sha256'])
                raw=[hx.read_bounded_text(directory/(name+'.'+suffix)) for suffix in ('nodes.log','elements.log','log')]
                checked=(hx if name in HEX_CASES else tet).check_outputs(name,*raw)
                write(directory/'checked.json',checked);item['checker_passed']=checked['passed']
                if not checked['passed']:raise RuntimeError('Unchanged primitive checker rejected case')
                texts[name]=raw;item['status']='passed'
                if name=='shear_double_stiffness':
                    scaling=hx.check_stiffness_scaling(texts['shear'][0],texts['shear'][1],raw[0],raw[1])
                    result['stiffness_scaling']=scaling
                    if not scaling['passed']:raise RuntimeError('Unchanged stiffness scaling checker rejected pair')
            except BaseException as error:
                item.update(status='failed',error=f'{type(error).__name__}: {error}');failure=name
            finally:
                try:item['output_hashes']={str(p.relative_to(directory)):sha(p) for p in sorted(directory.rglob('*')) if p.is_file()}
                except BaseException as error:item['output_inventory_error']=f'{type(error).__name__}: {error}';item['status']='failed';failure=name
                write(output/'results.json',result)
        result['status']='completed' if failure is None else 'failed_or_incomplete'
        if failure:result['first_failure']=failure
    except BaseException as error:result.update(status='failed_or_incomplete',worker_error=f'{type(error).__name__}: {error}')
    finally:
        result['worker_seconds']=time.monotonic()-start
        result['inputs_after']=unchanged(base['input_hashes']) if base is not None else {}
        if not result['inputs_after'] or not all(v['unchanged'] for v in result['inputs_after'].values()):result['status']='failed_or_incomplete'
        for item in result['cases']:
            if item['status']=='not_executed':item.setdefault('reason','stopped_without_execution')
        write(output/'results.json',result)
    return 0 if result['status']=='completed' else 1


def accept(supervision,result,after):
    if not (isinstance(result,dict) and isinstance(supervision,dict) and supervision.get('status')=='completed'
            and supervision.get('exit_code')==0 and result.get('status')=='completed' and result.get('solver_invocations')==8
            and result.get('no_retry') is True and result.get('solver_backend')=='accelerate'
            and result.get('stiffness_scaling',{}).get('passed') is True and isinstance(result.get('cases'),list)
            and all(isinstance(row,dict) for row in result['cases'])
            and [row.get('case') for row in result['cases']]==list(CASES)):
        return False
    identity=result.get('runtime_identity_sha256','')
    if not re.fullmatch('[0-9a-f]{64}',identity):return False
    for row in result['cases']:
        evidence=row.get('backend_evidence',{})
        if (row.get('status')!='passed' or row.get('checker_passed') is not True or row.get('solver_exit_code')!=0
                or evidence.get('runtime_identity_sha256')!=identity or evidence.get('solver_backend')!='accelerate'
                or evidence.get('solver_only_change_verified') is not True or evidence.get('solver_xml')!=backend.ACCELERATE_XML
                or evidence.get('executed_deck_sha256')!=row.get('deck_sha256') or not evidence.get('actual_selection_lines')
                or not all(isinstance(line,str) and re.search(r'\bselecting linear solver accelerate\b',line,re.I) for line in evidence['actual_selection_lines'])):
            return False
    prior=result.get('inputs_after')
    return (isinstance(prior,dict) and bool(prior) and isinstance(after,dict) and set(prior)==set(after)
            and all(isinstance(r,dict) and r.get('unchanged') is True for r in [*prior.values(),*after.values()]))


def record(path,repository):
    path=Path(path).resolve();repository=Path(repository).resolve()
    if not path.is_relative_to(repository):raise ValueError('Evidence escapes bound repository')
    return {'path':str(path.relative_to(repository)),'sha256':sha(path)}


def summaries(output,base,result,accepted):
    # No success summary is emitted from worker exit alone: parent cap/identity/
    # byte checks must also pass. Failed rows and every unexecuted case remain.
    repository=Path(base['release']['repository_directory'])
    checker_inputs_after=unchanged({str(ROOT/name):digest for name,digest in CHECKER_PINS.items()})
    summary_common={'source_commit':base['source_commit'],'runtime_identity_sha256':base['runtime_identity_sha256'],
                    'solver_backend':'accelerate','execution_accepted':bool(accepted),'numerical_threads':1,
                    'evidence':{name:record(output/filename,repository) for name,filename in
                                [('execution','execution.json'),('results','results.json'),('baseline','execution-baseline.json')]}}
    for group,names,field,status in [('hex8',HEX_CASES,'rows','passed_all_five_fixed_patch_controls'),
                                   ('tet10_mpc',TET_CASES,'case_rows','three_actual_fixed_software_controls_passed')]:
        rows=[]
        for row in result.get('cases',[]):
            if row.get('case') in names:
                item={**row,'passed':bool(accepted and row.get('status')=='passed')}
                if item['passed']:
                    name=row['case'];directory=output/name
                    paths={'original_deck':ROOT/ORIGINAL[name],'executed_deck':directory/(name+'.feb'),
                           'console':directory/'console.txt','nodes':directory/(name+'.nodes.log'),
                           'elements':directory/(name+'.elements.log'),'solver_log':directory/(name+'.log'),
                           'checked':directory/'checked.json','checker_source':ROOT/'scripts'/
                           ('mechanics_febio_verification.py' if name in HEX_CASES else 'mechanics_patient_constraints.py')}
                    item['evidence']={key:record(path,repository) for key,path in paths.items()}
                rows.append(item)
        summary={**summary_common,'status':status if accepted else 'failed_or_incomplete',field:rows,
                 'solver_invocations':sum('command' in row for row in rows),
                 'case_order':list(names),'original_numerical_checker_unchanged':all(v['unchanged'] for v in checker_inputs_after.values()),
                 'checker_inputs_after':checker_inputs_after}
        if group=='hex8':summary['stiffness_scaling']=result.get('stiffness_scaling',{'passed':False,'status':'not_executed'})
        write(output/(group+'-summary.json'),summary)


def launch(release_path,output):
    output=Path(output).resolve();output.mkdir(parents=True,exist_ok=False)
    started=time.monotonic();receipt={'status':'failed_or_incomplete','worker_started':False,'no_retry':True};base=None;result=initial_result()
    write(output/'results.json',result)
    try:
        base=baseline(release_path)
        if output!=Path(base['attempt_directory']):raise ValueError('Output differs from single released attempt')
        write(output/'execution-baseline.json',base)
        runtime=load(ROOT/'scripts/febio_runtime.py','accelerate_control_supervisor')
        environment=runtime.private_environment(runtime.declaration());receipt['thread_environment']=runtime.declaration()['caps']['thread_environment']
        command=[sys.executable,str(Path(__file__).resolve()),'worker','--output',str(output)]
        receipt['command']=command;receipt['worker_started']=True
        supervision=runtime.supervise(command,output/'supervision',cwd=ROOT,environment=environment,seconds=60,rss_bytes=3*1024**3)
        receipt['supervision']=supervision;result=json.loads((output/'results.json').read_text())
        after=unchanged(base['input_hashes']);receipt['inputs_after']=after
        accepted=accept(supervision,result,after) and result.get('runtime_identity_sha256')==base['runtime_identity_sha256']
        receipt['status']='completed' if accepted else 'failed_or_incomplete'
    except BaseException as error:receipt['error']=f'{type(error).__name__}: {error}'
    finally:
        if base is not None:
            receipt['final_inputs_after']=unchanged(base['input_hashes'])
            if not all(v['unchanged'] for v in receipt['final_inputs_after'].values()):receipt['status']='failed_or_incomplete'
            receipt['launcher_seconds']=time.monotonic()-started
            write(output/'execution.json',receipt) # Summaries bind this saved parent decision.
            try:summaries(output,base,result,receipt['status']=='completed')
            except BaseException as error:receipt.update(status='failed_or_incomplete',summary_error=f'{type(error).__name__}: {error}')
        receipt.setdefault('launcher_seconds',time.monotonic()-started)
        write(output/'execution.json',receipt)
    return 0 if receipt['status']=='completed' else 1


def main():
    parser=argparse.ArgumentParser();sub=parser.add_subparsers(dest='mode',required=True)
    p=sub.add_parser('prepare');p.add_argument('--output',required=True)
    p=sub.add_parser('run');p.add_argument('--release',required=True);p.add_argument('--output',required=True)
    p=sub.add_parser('worker');p.add_argument('--output',required=True)
    args=parser.parse_args()
    if args.mode=='prepare':prepare(args.output);return 0
    return worker(args.output) if args.mode=='worker' else launch(args.release,args.output)


if __name__=='__main__':raise SystemExit(main())

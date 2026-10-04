#!/usr/bin/env python3
"""One released, source-bound three-case attempt using the existing supervisor.

No acquisition/build, patient input, repeat, parameter override or retry path.
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

ROOT = Path(__file__).resolve().parents[1]
CASES = ('tet10_affine', 'mpc_translation', 'mpc_nonrigid')
VERSION = 'mechanics-patient-constraints-runtime-v1'
DECKS = 'artifacts/mechanics-patient-constraints-runtime-v1/decks'
MANIFEST_SHA = '73a20d14551742159f05145bfd39444fde4d8bc81415d6139d85e8d890d0c330'
RUNTIME_IDENTITY_SHA = 'f560f386726d13b399bcfb0c781a4b78104deecf41aa977e0efe205f18fabfc4'
CLOSURE = ('scripts/mechanics_patient_constraints_run.py', 'scripts/mechanics_patient_constraints.py',
           'scripts/mechanics_febio_verification.py', 'scripts/febio_runtime.py',
           'artifacts/febio-runtime-investigation-v1/prospective-runtime.json',
           DECKS+'/manifest.json', *(DECKS+'/'+name+'.feb' for name in CASES))


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024**2),b''):h.update(block)
    return h.hexdigest()


def write(path,data):
    path=Path(path);temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n');temporary.replace(path)


def load(path,name):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    if Path(module.__file__).resolve()!=Path(path).resolve():raise ValueError('Wrong loaded source origin')
    return module


def unchanged(inputs):
    """Missing/unreadable inputs remain recorded failures, never skip final receipt."""
    result={}
    for path,expected in inputs.items():
        try:result[path]={'unchanged':sha(path)==expected}
        except BaseException as error:result[path]={'unchanged':False,'error':f'{type(error).__name__}: {error}'}
    return result


def baseline(release_path):
    release_path=Path(release_path).resolve();release=json.loads(release_path.read_text())
    if (release.get('schema')!=VERSION+'-release' or release.get('authorized') is not True
            or not isinstance(release.get('root_release'),str) or not release['root_release'].strip()):
        raise ValueError('Explicit separately released execution record required')
    attempt=Path(release.get('attempt_directory',''))
    if not attempt.is_absolute():raise ValueError('One absolute attempt directory must be bound in the release')
    commit=release.get('source_commit','')
    if not re.fullmatch(r'[0-9a-f]{40}',commit):raise ValueError('Full immutable Git commit required')
    if Path(release['source_directory']).resolve()!=ROOT:raise ValueError('Run launcher from released source archive')
    repository=Path(release['repository_directory']).resolve()
    if repository==ROOT:raise ValueError('Mutable working-tree execution is prohibited')
    inputs={str(release_path):sha(release_path)}
    for relative in CLOSURE:
        path=ROOT/relative
        committed=subprocess.run(['git','-C',str(repository),'show',f'{commit}:{relative}'],capture_output=True,check=True,timeout=5).stdout
        expected=hashlib.sha256(committed).hexdigest()
        if sha(path)!=expected:raise ValueError('Source differs from exact commit: '+relative)
        inputs[str(path)]=expected
    manifest_path=ROOT/DECKS/'manifest.json'
    if sha(manifest_path)!=MANIFEST_SHA:raise ValueError('Prospective manifest changed')
    manifest=json.loads(manifest_path.read_text())
    if tuple(manifest['cases'])!=CASES or manifest['caps']!={'aggregate_seconds':60,'process_group_rss_bytes':3*1024**3,'numerical_threads':1,'maximum_cases':3,'time_steps':4,'retries':0}:
        raise ValueError('Declared cases/caps changed')
    for name,expected in manifest['decks'].items():
        if sha(ROOT/DECKS/name)!=expected:raise ValueError('Deck differs from declaration')
    if sha(ROOT/'scripts/mechanics_patient_constraints.py')!=manifest['check_source_sha256'] or sha(ROOT/'scripts/mechanics_febio_verification.py')!=manifest['shared_parser_source_sha256']:
        raise ValueError('Checker closure differs from declaration')
    identity_path=Path(release['runtime_identity']).resolve()
    if sha(identity_path)!=RUNTIME_IDENTITY_SHA:raise ValueError('Accepted runtime identity changed')
    identity=json.loads(identity_path.read_text());inputs[str(identity_path)]=RUNTIME_IDENTITY_SHA
    executable=Path(identity['executable']).resolve();prefix=executable.parents[2]
    for name,expected in identity['libraries'].items():inputs[str(prefix/name)]=expected
    inputs[identity['private_openmp']['path']]=identity['private_openmp']['sha256']
    for name,expected in identity['receipts_sha256'].items():inputs[str(identity_path.parent/name)]=expected
    checks=unchanged(inputs)
    if not all(v['unchanged'] for v in checks.values()):raise ValueError('Bound input changed before execution')
    return {'schema':VERSION+'-baseline','source_commit':commit,'source_directory':str(ROOT),
            'release':release,'release_path':str(release_path),'input_hashes':inputs,'attempt_directory':str(attempt.resolve()),
            'executable':str(executable),'caps':manifest['caps'],'case_order':list(CASES),
            'checker_source':str(ROOT/'scripts/mechanics_patient_constraints.py'),
            'setup_scope':'Git/source/runtime validation occurs before the supervised60s worker sequence; no solver runs during setup.'}


def initial_result():
    return {'status':'not_started','cases':[{'case':name,'status':'not_executed'} for name in CASES],
            'solver_invocations':0,'no_retry':True}


def worker(output):
    output=Path(output);base=None
    # Refuse before any write scope: preserve the complete earlier receipt.
    if json.loads((output/'results.json').read_text()) != initial_result():raise ValueError('Worker cannot repeat an attempted sequence')
    result=initial_result();result['status']='running';start=time.monotonic();failure=None
    try:
        base=json.loads((output/'execution-baseline.json').read_text())
        if base != baseline(base['release_path']):raise ValueError('Worker baseline does not match verified release')
        if output.resolve()!=Path(base['attempt_directory']):raise ValueError('Worker output differs from single released attempt')
        if Path(base['source_directory']).resolve()!=ROOT:raise ValueError('Worker source origin mismatch')
        if not all(v['unchanged'] for v in unchanged(base['input_hashes']).values()):raise ValueError('Worker before-input mismatch')
        checker=load(ROOT/'scripts/mechanics_patient_constraints.py','released_mpc_checker')
        result.update(checker_import_path=checker.__file__,numpy_version=checker.np.__version__)
        write(output/'results.json',result)
        for item in result['cases']:
            if failure:item['reason']='blocked_by_first_failure:'+failure;continue
            name=item['case'];directory=output/name
            try:
                remaining=60-(time.monotonic()-start)
                if remaining<=0:raise TimeoutError('Aggregate worker allowance exhausted')
                directory.mkdir(exist_ok=False)
                source_deck=ROOT/DECKS/(name+'.feb');deck=directory/source_deck.name;deck.write_bytes(source_deck.read_bytes())
                item.update(status='solver_started',deck_sha256=sha(deck))
                command=[base['executable'],'-noconfig','-no_title','-i',deck.name,'-o',name+'.log']
                item['command']=command;result['solver_invocations']+=1;write(output/'results.json',result)
                before=time.monotonic()
                with (directory/'console.txt').open('x') as log:
                    completed=subprocess.run(command,cwd=directory,stdout=log,stderr=subprocess.STDOUT,timeout=remaining)
                item.update(solver_seconds=time.monotonic()-before,solver_exit_code=completed.returncode)
                if completed.returncode!=0:raise RuntimeError('Nonzero solver exit')
                if sha(deck)!=item['deck_sha256']:raise ValueError('Executed deck changed')
                texts=[checker.patch.read_bounded_text(directory/(name+'.'+suffix)) for suffix in ('nodes.log','elements.log','log')]
                checked=checker.check_outputs(name,*texts);write(directory/'checked.json',checked)
                console=checker.patch.read_bounded_text(directory/'console.txt')
                item['skyline_console_lines']=[line.strip() for line in console.splitlines() if 'skyline' in line.lower()]
                item['checker_passed']=checked['passed']
                if not checked['passed']:raise RuntimeError('Frozen output checker rejected case')
                if not any('selecting linear solver skyline' in line.lower() for line in item['skyline_console_lines']):raise ValueError('Actual Skyline selection evidence missing')
                item['status']='passed'
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
    return (isinstance(supervision,dict) and supervision.get('status')=='completed' and supervision.get('exit_code')==0
            and isinstance(result,dict) and result.get('status')=='completed' and result.get('solver_invocations')==3
            and result.get('no_retry') is True and isinstance(result.get('cases'),list)
            and all(isinstance(row,dict) for row in result['cases'])
            and [row.get('case') for row in result['cases']]==list(CASES)
            and all(row.get('status')=='passed' and row.get('checker_passed') is True for row in result['cases'])
            and isinstance(result.get('inputs_after'),dict) and bool(result['inputs_after'])
            and isinstance(after,dict) and bool(after) and set(result['inputs_after'])==set(after)
            and all(isinstance(row,dict) and row.get('unchanged') is True for row in result['inputs_after'].values())
            and all(isinstance(row,dict) and row.get('unchanged') is True for row in after.values()))


def launch(release_path,output):
    output=Path(output).resolve();output.mkdir(parents=True,exist_ok=False)
    started=time.monotonic();receipt={'status':'failed_or_incomplete','worker_started':False,'no_retry':True};base=None
    write(output/'results.json',initial_result())
    try:
        base=baseline(release_path)
        if output!=Path(base['attempt_directory']):raise ValueError('Output differs from single released attempt')
        write(output/'execution-baseline.json',base)
        runtime=load(ROOT/'scripts/febio_runtime.py','released_process_supervisor')
        environment=runtime.private_environment(runtime.declaration())
        receipt['thread_environment']=runtime.declaration()['caps']['thread_environment']
        command=[sys.executable,str(Path(__file__).resolve()),'worker','--output',str(output)]
        receipt['command']=command;receipt['worker_started']=True
        supervision=runtime.supervise(command,output/'supervision',cwd=ROOT,environment=environment,seconds=60,rss_bytes=3*1024**3)
        receipt['supervision']=supervision
        result=json.loads((output/'results.json').read_text())
        after=unchanged(base['input_hashes']);receipt['inputs_after']=after
        receipt['result_status']=result.get('status') if isinstance(result,dict) else None
        receipt['status']='completed' if accept(supervision,result,after) else 'failed_or_incomplete'
    except BaseException as error:receipt['error']=f'{type(error).__name__}: {error}'
    finally:
        if base is not None:
            receipt['final_inputs_after']=unchanged(base['input_hashes'])
            if not all(v['unchanged'] for v in receipt['final_inputs_after'].values()):receipt['status']='failed_or_incomplete'
        receipt['launcher_seconds']=time.monotonic()-started
        write(output/'execution.json',receipt)
    return 0 if receipt['status']=='completed' else 1


def main():
    parser=argparse.ArgumentParser();sub=parser.add_subparsers(dest='mode',required=True)
    p=sub.add_parser('run');p.add_argument('--release',required=True);p.add_argument('--output',required=True)
    p=sub.add_parser('worker');p.add_argument('--output',required=True)
    args=parser.parse_args()
    return worker(args.output) if args.mode=='worker' else launch(args.release,args.output)


if __name__=='__main__':raise SystemExit(main())

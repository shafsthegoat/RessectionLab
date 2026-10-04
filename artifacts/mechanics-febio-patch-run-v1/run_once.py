"""One released five-deck sequence; outer committed supervisor owns hard caps."""
from pathlib import Path
import hashlib
import importlib.util
import json
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent


def digest(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path, data):
    temp = path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n'); temp.replace(path)


def main():
    baseline = json.loads((ROOT/'execution-baseline.json').read_text())
    source = Path(baseline['source_directory'])
    result = json.loads((ROOT/'results.json').read_text())
    result.update(status='running', solver_invocations=0)
    result['started_monotonic'] = time.monotonic()
    started = time.monotonic(); failure = None
    try:
        assert digest(__file__)==baseline['wrapper_sha256']
        for path,sha in baseline['input_hashes'].items():
            if digest(path)!=sha: raise ValueError('Before-run input changed: '+path)
        loader=importlib.util.spec_from_file_location('released_patch_checker',source/'scripts/mechanics_febio_verification.py')
        checker=importlib.util.module_from_spec(loader);loader.loader.exec_module(checker)
        result['checker_import_path']=checker.__file__
        result['checker_numpy_version']=checker.np.__version__
        write(ROOT/'results.json',result)
        texts={}
        for item in result['cases']:
            name=item['case']; directory=ROOT/name
            if failure: item['reason']='blocked_by_first_failure:'+failure; continue
            remaining=baseline['applied_patch_cap_seconds']-(time.monotonic()-started)
            if remaining<=0: raise TimeoutError('Aggregate patch allowance exhausted')
            directory.mkdir(exist_ok=False)
            deck=source/'artifacts/mechanics-febio-verification-v1/decks'/f'{name}.feb'
            (directory/deck.name).write_bytes(deck.read_bytes())
            item.update(status='solver_started',deck_sha256=digest(deck))
            command=[baseline['executable'],'-noconfig','-no_title','-i',deck.name,'-o',f'{name}.log']
            item['command']=command;result['solver_invocations']+=1;write(ROOT/'results.json',result)
            try:
                before=time.monotonic()
                with (directory/'console.txt').open('x') as output:
                    process=subprocess.run(command,cwd=directory,stdout=output,stderr=subprocess.STDOUT,timeout=remaining)
                item.update(solver_exit_code=process.returncode,solver_seconds=time.monotonic()-before)
                if process.returncode!=0: raise RuntimeError('Solver returned nonzero status')
                if digest(directory/deck.name)!=item['deck_sha256']: raise ValueError('Executed deck changed')
                actual=[checker.read_bounded_text(directory/f'{name}{suffix}') for suffix in ('.nodes.log','.elements.log','.log')]
                checked=checker.check_outputs(name,*actual);write(directory/'checked.json',checked)
                item['checker_passed']=checked['passed']
                console=checker.read_bounded_text(directory/'console.txt')
                item['skyline_console_lines']=[line.strip() for line in console.splitlines() if 'skyline' in line.lower()]
                if not checked['passed']: raise RuntimeError('Committed primitive output checker rejected case')
                item['status']='passed';texts[name]=actual
            except BaseException as error:
                item.update(status='failed',error={'type':type(error).__name__,'message':str(error)})
                failure=name
            finally:
                item['output_hashes']={str(path.relative_to(directory)):digest(path) for path in sorted(directory.rglob('*')) if path.is_file()}
                write(ROOT/'results.json',result)
        if failure:
            result['status']='failed_or_incomplete';result['first_failure']=failure
        else:
            base=texts['shear'];double=texts['shear_double_stiffness']
            scaling=checker.check_stiffness_scaling(base[0],base[1],double[0],double[1])
            result['scaling_check']=scaling
            result['status']='completed' if scaling['passed'] else 'failed_or_incomplete'
    except BaseException as error:
        result.update(status='failed_or_incomplete',error={'type':type(error).__name__,'message':str(error)})
    finally:
        result['worker_seconds']=time.monotonic()-started
        result['inputs_unchanged_after']={path:Path(path).is_file() and digest(path)==sha for path,sha in baseline['input_hashes'].items()}
        if not all(result['inputs_unchanged_after'].values()): result['status']='failed_or_incomplete'
        for item in result['cases']:
            if item['status']=='not_executed': item.setdefault('reason','stopped_without_execution')
        write(ROOT/'results.json',result)
    return 0 if result['status']=='completed' else 1


if __name__=='__main__':raise SystemExit(main())

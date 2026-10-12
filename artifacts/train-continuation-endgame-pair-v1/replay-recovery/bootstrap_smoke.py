"""Root-run isolated real imports under the exact worker boundary; no payloads."""
import argparse
import json
import os
from pathlib import Path
import signal
import sys
import time
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
from common import ROOT,sha,runtime_roots
from worker import install_boundary,import_execution_modules


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    started=time.monotonic();output=args.output.resolve();output.mkdir(exist_ok=False)
    def timeout(*unused):raise TimeoutError('Bootstrap control25s cap')
    signal.signal(signal.SIGALRM,timeout);signal.setitimer(signal.ITIMER_REAL,25.)
    paths=list((ROOT/'src/resectionlab').glob('*.py'))+list(HERE.glob('*.py'))
    index={'files':{str(p.relative_to(ROOT)):sha(p) for p in paths},'input_metadata':{},'public_arrays':{},'runtime_roots':runtime_roots()}
    payload_attempts=[];runtime_archive_probes=[]
    # One interpreter-declared stdlib archive probe, never arbitrary runtime ZIPs.
    stdlib_archive=(Path(sys.base_prefix)/'lib'/('python%d%d.zip'%sys.version_info[:2])).resolve()
    if stdlib_archive not in {Path(p).resolve() for p in sys.path if p}:
        raise ValueError('Expected isolated interpreter stdlib archive path absent')
    def no_payload(event,values):
        if event=='open' and isinstance(values[0],(str,bytes)):
            path=Path(os.path.abspath(os.fsdecode(values[0])))
            mode,flags=values[1:3]
            writing=(isinstance(mode,str) and any(c in mode for c in 'wax+')) or bool(flags & (os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC|os.O_APPEND))
            if path==stdlib_archive and not writing:
                runtime_archive_probes.append(str(path));return
            if path.suffix in ('.npy','.psckpt','.qckpt','.mat','.tar','.zip'):
                payload_attempts.append(str(path));raise PermissionError('No actual payload in bootstrap smoke')
    sys.addaudithook(no_payload)
    install_boundary(index,output);sys.path.insert(0,str(ROOT/'src'))
    result={'status':'started','payload_open_attempts':payload_attempts,
            'interpreter_stdlib_archive':str(stdlib_archive),'runtime_archive_read_probes':runtime_archive_probes}
    try:
        result['runtime']=import_execution_modules()
        import torch
        torch.set_num_threads(1);torch.set_num_interop_threads(1)
        for path,mode in ((Path(index['runtime_roots'][0])/'__runner_forbidden_write__','wb'),
                          (ROOT/'data'/'__runner_forbidden_read__.json','rb')):
            try:
                with path.open(mode):pass
            except PermissionError:pass
            else:raise AssertionError('Boundary exemption leaked beyond read-only dependencies')
        if payload_attempts:raise AssertionError('Payload access attempted')
        result.update(status='passed',native_calls=0,policy_forwards=0,optimizer_steps=0,
                      runtime_write_denied=True,undeclared_data_read_denied=True)
    except BaseException as error:
        result.update(status='failed',error={'type':type(error).__name__,'message':str(error)})
        raise
    finally:
        result['seconds']=time.monotonic()-started
        (output/'receipt.json').write_text(json.dumps(result,sort_keys=True)+'\n')
        signal.setitimer(signal.ITIMER_REAL,0)


if __name__=='__main__':main()

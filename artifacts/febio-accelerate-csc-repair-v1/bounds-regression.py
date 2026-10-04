"""Extract exact adapter code into an ASan/UBSan fixture; never run a FEBio model."""
from pathlib import Path
import argparse
import hashlib
import importlib.util
import json
import os
import subprocess

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PATCH_SHA = 'c3c6b068ed8e7b2480a64308099fa57d8b02fdbe04076f1497b0b2fdd3b7c3e0'
OLD_SHA = 'f5432e79ab73dd11d4a0550691480edd7fd5f00c392ae2f78eddf9c4b5da51c9'
NEW_SHA = '60e5a3f2350826af1b95376ab93c8a476f2312158763d63d82eb80ec54af79c2'
SUPERVISOR_SHA = '679594d7f3759f5485b9fb868e7e5543ebb242bccd24d112f9ca862bb6d2a01e'
RAW = ROOT / 'build/febio-accelerate-csc-bounds-v1'
COMPILER = '/Library/Developer/CommandLineTools/usr/bin/clang++'

def sha(path):
    with Path(path).open('rb') as handle:return hashlib.file_digest(handle,'sha256').hexdigest()

def save(path,value):
    with path.open('x') as handle:json.dump(value,handle,indent=2);handle.write('\n')

FIXTURE = r'''
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <vector>
struct Matrix {
    int n, nz; std::unique_ptr<int[]> ptr;
    Matrix(int columns, int nonzeros, std::vector<int> offsets, bool allocate)
    :n(columns),nz(nonzeros),ptr(allocate ? new int[offsets.size()] : nullptr) {
        if (allocate) for(size_t i=0;i<offsets.size();++i)ptr[i]=offsets[i];
    }
    int Rows(){return n;} int Columns(){return n;} int NonZeroes(){return nz;}
    int* Pointers(){return ptr.get();}
};
struct LinearSolver {int base_calls=0;bool PreProcess(){++base_calls;return true;}};
struct State {Matrix* m_pA; int m_n=-1,m_nnz=-1,m_nrhs=-1;std::vector<long> pointers;long* colS=nullptr;};
struct Adapter : LinearSolver {
    State* imp;
    explicit Adapter(State* state):imp(state){}
    __attribute__((noinline)) bool PreProcess(){
EXACT_ADAPTER_FRAGMENT
        return true;
    }
};
int check(int id) {
    const std::vector<std::vector<int>> expected={{0,3,6,9},{0,1,2,3},{0,1,1,2,2},{0,1},{0},{0}};
    const std::vector<int> columns={3,3,4,1,0,0};
    const auto& offsets=expected.at(id);int n=columns.at(id),nz=offsets.back();
    Matrix matrix(n,nz,offsets,id!=5);State state;state.m_pA=&matrix;Adapter adapter(&state);
    assert(adapter.PreProcess());
    assert(state.m_n==n && state.m_nnz==nz && state.m_nrhs==1);
    if(n==0){assert(adapter.base_calls==1);assert(state.pointers.empty());assert(state.colS==nullptr);}
    else {
        // Consume the terminal offset BEFORE the size assertion: a too-short
        // destination must produce an actual bounds violation under sanitizers.
        volatile long terminal=state.colS[n];assert(terminal==nz);
        assert(state.pointers.size()==static_cast<size_t>(n+1));
        for(int i=0;i<=n;++i)assert(state.colS[i]==offsets[i]);
        assert(adapter.base_calls==0);
    }
    std::printf("PASS profile %d columns=%d nonzeros=%d\n",id,n,nz);return 0;
}
int main(int argc,char** argv) {
    assert(argc==2);int id=std::atoi(argv[1]);
    if(id>=0)return check(id);
    for(int i=0;i<6;++i)check(i);return 0;
}
'''

def prepare():
    original=ROOT/'data/optional-runtimes/febio-4.13/source/NumCore/AccelerateSparseSolver.cpp'
    assert sha(original)==OLD_SHA and sha(HERE/'accelerate-csc-pointer-count.patch')==PATCH_SHA
    RAW.mkdir(parents=True,exist_ok=False)
    candidate=RAW/'patched/NumCore/AccelerateSparseSolver.cpp';candidate.parent.mkdir(parents=True)
    candidate.write_bytes(original.read_bytes())
    result=subprocess.run(['/usr/bin/patch','--batch','--fuzz=0','-p1','-i',str(HERE/'accelerate-csc-pointer-count.patch')],cwd=RAW/'patched',capture_output=True,text=True,check=True,timeout=5)
    assert sha(candidate)==NEW_SHA
    files={}
    for mode,path in [('original',original),('patched',candidate)]:
        source=path.read_text();start=source.index('    imp->m_n = imp->m_pA->Rows();',source.index('bool AccelerateSparseSolver::PreProcess()'))
        end=source.index('    if (__builtin_available',start)
        fragment=source[start:end]
        assert fragment.endswith('    \n') and 'imp->colS = &imp->pointers[0];' in fragment
        target=RAW/(mode+'.cpp');target.write_text(FIXTURE.replace('EXACT_ADAPTER_FRAGMENT',fragment))
        files[mode]={'source_sha256':sha(path),'fragment_sha256':hashlib.sha256(fragment.encode()).hexdigest(),'fixture':str(target),'fixture_sha256':sha(target)}
    save(HERE/'fixture-preparation.json',{'status':'prepared_not_compiled','patch_apply_stdout':result.stdout,'patch_apply_stderr':result.stderr,'compiler':COMPILER,'compiler_sha256':sha(COMPILER),'files':files,'original_source_unchanged':sha(original)==OLD_SHA,'profiles':['dense3/9','diagonal3/3','empty-columns4/2','single1/1','zero-created','zero-null'],'scope':'Exact source-extracted conversion fragment with mock matrix storage; not full adapter or factorization validation.'})

def run_worker():
    identity=json.loads((HERE/'fixture-preparation.json').read_text())
    assert sha(COMPILER)==identity['compiler_sha256']
    output=HERE/'bounds-controls';output.mkdir(exist_ok=False)
    rows=[]
    def execute(name, command, seconds):
        import time
        start=time.monotonic()
        with (output/(name+'.log')).open('x') as log:
            result=subprocess.run(command,cwd=RAW,stdout=log,stderr=subprocess.STDOUT,timeout=seconds)
        record={'operation':name,'command':command,'exit_code':result.returncode,'elapsed_seconds':time.monotonic()-start,'log_sha256':sha(output/(name+'.log'))}
        rows.append(record)
        return record,(output/(name+'.log')).read_text()
    try:
        for mode,item in identity['files'].items():
            fixture=Path(item['fixture']);assert sha(fixture)==item['fixture_sha256']
            binary=RAW/(mode+'-bounds');assert not binary.exists()
            command=[COMPILER,'-std=c++17','-O1','-g','-fsanitize=address,undefined','-fno-omit-frame-pointer',str(fixture),'-o',str(binary)]
            result,_=execute(mode+'-compile',command,30)
            assert result['exit_code']==0
            item['binary_sha256']=sha(binary)
        for mode,profile in [('original',0),('original',1),('original',5),('patched',-1)]:
            name=mode+'-profile'+str(profile)
            result,log=execute(name,[str(RAW/(mode+'-bounds')),str(profile)],5)
            expected_failure=mode=='original'
            passed=((result['exit_code']!=0 and ('AddressSanitizer: heap-buffer-overflow' in log or 'runtime error:' in log))
                    if expected_failure else result['exit_code']==0 and log.count('PASS profile ')==6)
            result.update(expected_failure=expected_failure,expectation_met=passed)
            if not passed:raise RuntimeError('Bounds expectation failed: '+name)
        status='all_expected_bounds_results_observed'
    except BaseException as error:
        save(output/'summary.json',{'status':'unexpected_failure_stop_no_retry','error':repr(error),'rows':rows});raise
    save(output/'summary.json',{'status':status,'rows':rows,'compiled_fixtures':identity['files'],'original_source_unchanged':sha(ROOT/'data/optional-runtimes/febio-4.13/source/NumCore/AccelerateSparseSolver.cpp')==OLD_SHA,'no_FEBio_runtime_built_or_executed':True,'no_patient_or_measurement_access':True})

def run():
    import sys
    helper_path=ROOT/'scripts/febio_runtime.py';assert sha(helper_path)==SUPERVISOR_SHA
    spec=importlib.util.spec_from_file_location('bounds_supervisor',helper_path);helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
    env=helper.private_environment({'caps':{'thread_environment':{'OMP_NUM_THREADS':'1','VECLIB_MAXIMUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1'}}})
    env.update(ASAN_OPTIONS='detect_leaks=0:abort_on_error=0:symbolize=0',UBSAN_OPTIONS='halt_on_error=1:print_stacktrace=0',PYTHONDONTWRITEBYTECODE='1')
    command=[sys.executable,'-I','-S','-B',str(Path(__file__).resolve()),'control-worker']
    result=helper.supervise(command,HERE/'bounds-supervised',cwd=RAW,environment=env,seconds=60,rss_bytes=1024**3)
    if result['status']!='completed':raise RuntimeError('Supervised bounds diagnostic failed')

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['prepare','run-controls','control-worker']);args=parser.parse_args()
    {'prepare':prepare,'run-controls':run,'control-worker':run_worker}[args.stage]()

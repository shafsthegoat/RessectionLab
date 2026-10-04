"""Check only the repaired offset conversion; no original-code execution."""
from pathlib import Path
import hashlib
import importlib.util
import json
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
RAW = ROOT / 'build/febio-accelerate-csc-positive-v1'
OUT = HERE / 'positive-control-v1'
COMPILER = '/Library/Developer/CommandLineTools/usr/bin/clang++'
SDK = '/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk'
PATCHED = ROOT / 'build/febio-accelerate-csc-bounds-v1/patched/NumCore/AccelerateSparseSolver.cpp'
PATCHED_SHA = '60e5a3f2350826af1b95376ab93c8a476f2312158763d63d82eb80ec54af79c2'
SUPERVISOR_SHA = '679594d7f3759f5485b9fb868e7e5543ebb242bccd24d112f9ca862bb6d2a01e'


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def save(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


FIXTURE = r'''
#include <cassert>
#include <cstdio>
#include <memory>
#include <vector>
struct Matrix {
    int n, nz;
    std::unique_ptr<int[]> ptr;
    Matrix(int columns, const std::vector<int>& offsets, bool allocate)
        : n(columns), nz(offsets.back()),
          ptr(allocate ? new int[offsets.size()] : nullptr) {
        assert(offsets.size() == static_cast<size_t>(n + 1));
        if (allocate) for (size_t i=0; i<offsets.size(); ++i) ptr[i]=offsets[i];
    }
    int Rows() { return n; }
    int Columns() { return n; }
    int NonZeroes() { return nz; }
    int* Pointers() { return ptr.get(); }
};
struct LinearSolver {
    int base_calls=0;
    bool PreProcess() { ++base_calls; return true; }
};
struct State {
    Matrix* m_pA;
    int m_n=-1, m_nnz=-1, m_nrhs=-1;
    std::vector<long> pointers;
    long* colS=nullptr;
};
struct Adapter : LinearSolver {
    State* imp;
    explicit Adapter(State* state) : imp(state) {}
    bool PreProcess() {
EXACT_PATCHED_FRAGMENT
        return true;
    }
};
int main() {
    const std::vector<std::vector<int>> expected={
        {0,3,6,9}, {0,1,2,3}, {0,1,1,2,2}, {0,1}, {0}, {0}};
    const int columns[]={3,3,4,1,0,0};
    for (int id=0; id<6; ++id) {
        const auto& offsets=expected.at(id);
        const int n=columns[id];
        Matrix matrix(n, offsets, id!=5);
        State state; state.m_pA=&matrix;
        Adapter adapter(&state);
        assert(adapter.PreProcess());
        assert(state.m_n==n && state.m_nnz==offsets.back() && state.m_nrhs==1);
        if (n==0) {
            assert(adapter.base_calls==1);
            assert(state.pointers.empty() && state.colS==nullptr);
        } else {
            assert(state.pointers.size()==static_cast<size_t>(n+1));
            assert(state.colS==state.pointers.data());
            for (int i=0; i<=n; ++i) assert(state.colS[i]==offsets.at(i));
            assert(state.colS[n]==matrix.NonZeroes());
            assert(adapter.base_calls==0);
        }
        std::printf("PASS profile %d columns=%d nonzeros=%d\n", id,n,matrix.NonZeroes());
    }
}
'''


def prepare():
    assert sha(PATCHED) == PATCHED_SHA
    RAW.mkdir(parents=True, exist_ok=False)
    source = PATCHED.read_text()
    start = source.index('    imp->m_n = imp->m_pA->Rows();',
                         source.index('bool AccelerateSparseSolver::PreProcess()'))
    end = source.index('    if (__builtin_available', start)
    fragment = source[start:end]
    (RAW/'positive.cpp').write_text(FIXTURE.replace('EXACT_PATCHED_FRAGMENT', fragment))
    pins = {str(path): sha(path) for path in (
        PATCHED, Path(COMPILER), Path(SDK)/'SDKSettings.json',
        ROOT/'scripts/febio_runtime.py', Path(__file__), RAW/'positive.cpp')}
    save(HERE/'positive-release.json', {
        'status': 'released_repaired_code_only', 'pins': pins,
        'fragment_sha256': hashlib.sha256(fragment.encode()).hexdigest(),
        'limits': {'seconds': 60, 'rss_bytes': 1024**3, 'cpu_threads': 1},
        'scope': 'Six positive source-extracted offset-copy controls; no original faulty code, full adapter, factorization, model, or patient execution.',
        'original_setup_failure_preserved': 'bounds-controls/summary.json',
        'worker_interruption': 'Previous preparation agent stopped by automatic cybersecurity screening before the SDK-corrected diagnostic; no corrected attempt was executed.',
        'old_defect_dynamic_reproduction': False})


def worker():
    release = json.loads((HERE/'positive-release.json').read_text())
    for path, expected in release['pins'].items():
        assert sha(path) == expected
    OUT.mkdir(exist_ok=False)
    rows = []
    commands = [
        [COMPILER, '-isysroot', SDK, '-std=c++17', '-O1', '-g',
         '-fsanitize=address,undefined', '-fno-omit-frame-pointer',
         str(RAW/'positive.cpp'), '-o', str(RAW/'positive-check')],
        [str(RAW/'positive-check')]]
    try:
        for name, command, limit in zip(('compile', 'six-positive-cases'), commands, (40, 5)):
            start = time.monotonic()
            with (OUT/(name+'.log')).open('x') as stream:
                result = subprocess.run(command, cwd=RAW, stdout=stream,
                                        stderr=subprocess.STDOUT, timeout=limit)
            rows.append({'name': name, 'command': command, 'exit_code': result.returncode,
                         'seconds': time.monotonic()-start,
                         'log_sha256': sha(OUT/(name+'.log'))})
            if result.returncode != 0:
                raise RuntimeError(name+' failed; no automatic retry')
        assert (OUT/'six-positive-cases.log').read_text().count('PASS profile ') == 6
        for path, expected in release['pins'].items():
            assert sha(path) == expected
        save(OUT/'result.json', {'status': 'six_positive_controls_passed', 'rows': rows,
             'all_pins_unchanged': True, 'binary_sha256': sha(RAW/'positive-check'),
             'original_code_executed': False, 'runtime_built_or_executed': False})
    except BaseException as error:
        save(OUT/'failure.json', {'status': 'failed_no_retry', 'rows': rows, 'error': repr(error)})
        raise


def run():
    helper_path = ROOT/'scripts/febio_runtime.py'
    assert sha(helper_path) == SUPERVISOR_SHA
    spec = importlib.util.spec_from_file_location('checked_supervisor', helper_path)
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    env = helper.private_environment({'caps': {'thread_environment': {
        'OMP_NUM_THREADS': '1', 'VECLIB_MAXIMUM_THREADS': '1',
        'OPENBLAS_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1'}}})
    env.update(ASAN_OPTIONS='detect_leaks=0:symbolize=0',
               UBSAN_OPTIONS='halt_on_error=1:print_stacktrace=0')
    result = helper.supervise([sys.executable, '-I', '-S', '-B', str(Path(__file__)), 'worker'],
        HERE/'positive-supervision-v1', cwd=RAW, environment=env,
        seconds=60, rss_bytes=1024**3)
    if result['status'] != 'completed':
        raise RuntimeError('Positive control failed; original evidence retained')


if __name__ == '__main__':
    {'prepare': prepare, 'worker': worker, 'run': run}[sys.argv[1]]()

"""One canonical saved-native comparison worker; external supervision required.

Root binds this file, exact manifest and fresh runtime before explicitly passing
--execute. This worker is not a supervisor, native solver or saved-stream replay.
"""
import argparse
import json
import os
from pathlib import Path
import sys

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',required=True)
    parser.add_argument('--manifest',required=True)
    parser.add_argument('--manifest-sha256',required=True)
    parser.add_argument('--output-directory',required=True)
    parser.add_argument('--execute',action='store_true')
    args=parser.parse_args()
    if not args.execute:
        raise ValueError('Explicit root-supervised execution required; no comparator imported')
    root=Path(args.root).resolve();out=Path(args.output_directory)
    if not out.is_absolute():out=root/out
    if ('..' in out.parts or not out.is_relative_to(root/'build')
        or not out.is_dir() or out.is_symlink()
        or any(p.is_symlink() for p in out.parents if p.is_relative_to(root))):
        raise ValueError('Existing parent-owned ignored output directory required')
    expected_cache=out/'unused-worker-pycache'
    if (out/'comparison.json').exists() or (out/'comparison.json').is_symlink():
        raise ValueError('Existing comparison output refuses another attempt')
    if (not sys.flags.isolated or not sys.dont_write_bytecode
        or sys.pycache_prefix!=str(expected_cache) or expected_cache.exists()
        or expected_cache.is_symlink()):
        raise ValueError('Isolated -B interpreter with exact absent fresh cache prefix required')
    sys.path.insert(0,str(root))
    from scripts import mechanics_hbe_v5_native_comparison as native
    if Path(native.__file__).resolve()!=root/'scripts/mechanics_hbe_v5_native_comparison.py':
        raise ValueError('Canonical comparator origin required')
    result=native.compare_native_rows(root,{'path':args.manifest,'sha256':args.manifest_sha256})
    if (result.get('schema')!=native.SCHEMA or result.get('native_output_admitted') is not True
        or result.get('native_execution_released') is not False
        or result.get('native_chain_verification_passes')!=2
        or result.get('physical_validation_pass') is not None
        or result.get('calibration_released') is not False
        or result.get('measured_response_accessed') is not False
        or result.get('patient_data_accessed') is not False
        or expected_cache.exists() or expected_cache.is_symlink()):
        raise ValueError('Completed comparison scope or cache policy differs')
    # A valid negative numerical screen is retained, not treated as a worker error.
    raw=(json.dumps(result,indent=2,sort_keys=True,allow_nan=False)+'\n').encode()
    if len(raw)>16*1024**2:raise ValueError('Comparison output exceeds declared aggregate budget')
    fd=os.open(out/'comparison.json',os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
    with os.fdopen(fd,'wb') as stream:stream.write(raw);stream.flush();os.fsync(stream.fileno())
    print(json.dumps({'status':'completed_numerical_screen','native_calls':0,
        'saved_stream_replays':0,'numerical_qualification_passed':result['numerical_qualification_passed'],
        'physical_validation_pass':None,'output_bytes':len(raw)}))
    return 0

if __name__=='__main__':raise SystemExit(main())

"""Use the existing public crop CLI contract with one exact staged scope delta."""
import argparse
import hashlib
from pathlib import Path
import types

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
SOURCE_SHA='565843dac7efeef29640bb1690a7f1f167d38310d8b7e771babad5302571d796'

def main():
    p=argparse.ArgumentParser()
    p.add_argument('phase',choices=('crop-mr-domains',))
    p.add_argument('--case',type=Path,required=True);p.add_argument('--case-sha256',required=True)
    p.add_argument('--headers',type=Path,required=True);p.add_argument('--headers-sha256',required=True)
    p.add_argument('--repository-root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--public-only',action='store_true')
    args=p.parse_args()
    assert args.repository_root.resolve()==ROOT and args.public_only
    assert args.case.resolve()==ROOT/'build/remind-select-public-bindings-v1/ReMIND-013-case.json'
    assert args.case_sha256=='9d4448328e979c856e3c8e08599b9cc2b66fc7d324f1652d7970468099267c65'
    assert args.headers.resolve()==ROOT/'build/remind-select-public-preparation-v1/actual-public-qc-v1/ReMIND-013-headers/result.json'
    assert args.headers_sha256=='fd3a5c5cfded6e4a60ef263c03f13df0adb07ae2f77b84f61aee9295e31b8948'
    assert args.output.resolve()==HERE/'actual-domain-qc-v1/ReMIND-013-crop-mr-domains'
    source=HERE/'stage/resectionlab/remind_planning_qc.py';raw=source.read_bytes()
    assert hashlib.sha256(raw).hexdigest()==SOURCE_SHA
    module=types.ModuleType('exact_expanded_remind_qc');module.__file__=str(source)
    exec(compile(raw,str(source),'exec',dont_inherit=True),module.__dict__)
    return module.run_crop(args)

if __name__=='__main__':raise SystemExit(main())

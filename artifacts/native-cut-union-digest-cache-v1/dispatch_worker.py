"""One generated phase; the root-owned parent supplies lease and total budget."""
import argparse
import importlib.util
import os
from pathlib import Path
import runpy
import select
import sys
import threading

HERE=Path(__file__).resolve().parent
ROOT=next(p for p in HERE.parents if (p/'src/resectionlab').is_dir())


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--phase',choices=('tests','baseline','candidate'),required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    fd=int(os.environ['RESECTIONLAB_PARENT_LEASE_FD'])
    def lease():
        while True:
            if select.select([fd],[],[],.2)[0] and not os.read(fd,1):os._exit(125)
    threading.Thread(target=lease,daemon=True).start()
    if args.phase=='tests':
        sys.path.insert(0,str(ROOT/'src'))
        import resectionlab
        path=HERE/'stage/resectionlab/native_cut_union.py'
        spec=importlib.util.spec_from_file_location('resectionlab.native_cut_union',path)
        helper=importlib.util.module_from_spec(spec)
        sys.modules[spec.name]=helper;spec.loader.exec_module(helper)
        resectionlab.native_cut_union=helper
        import pytest
        return pytest.main(['-q','-p','no:cacheprovider',
            str(HERE/'test_digest_cache.py'),str(ROOT/'tests/test_native_cut_union.py'),
            str(ROOT/'tests/test_independent_cut_union.py'),str(ROOT/'tests/test_native_immutable_state.py')])
    sys.argv=[str(HERE/'profile_fixed_row.py'),'--variant',args.phase,'--output',str(args.output)]
    runpy.run_path(str(HERE/'profile_fixed_row.py'),run_name='__main__')
    return 0

if __name__=='__main__':raise SystemExit(main())

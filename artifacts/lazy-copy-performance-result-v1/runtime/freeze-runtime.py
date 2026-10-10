"""Metadata-only root freeze; creates a pending non-executable release template."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import subprocess
import sys
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
sys.path.insert(0,str(HERE))
from pilot_contract import PAIR_OUTPUT as OUTPUT,TRAIN,INPUT_INDEX,INPUT_SHA,inputs,sha,release_template,validate_release,BASELINE,BASELINE_SHA,CANDIDATE,CANDIDATE_SHA

def module_path(name):
    if name=='resectionlab':return ROOT/'src/resectionlab/__init__.py'
    if not name.startswith('resectionlab.'):return None
    path=ROOT/'src'/name.replace('.','/')
    if path.with_suffix('.py').is_file():return path.with_suffix('.py')
    if (path/'__init__.py').is_file():return path/'__init__.py'
    return None

def source_closure():
    paths={HERE/name for name in ('pilot_contract.py','search_worker.py','run_owned.py','run_pair.py','freeze-runtime.py','test_owned_setup.py')}
    paths.update(ROOT/'build/goal-conditioned-policy-v1'/name for name in ('run_contact_owned.py','darwin_fast_sampler.py'))
    paths.add(ROOT/'src/resectionlab/__init__.py')
    paths.add(ROOT/'build/paired-obstruction-opening-search-v2/pilot_contract.py')
    paths.add(ROOT/'build/immutable-preview-lazy-copy-v2/stage/resectionlab/native_resection.py')
    done=set()
    while paths-done:
        path=sorted(paths-done)[0];done.add(path)
        if path.is_symlink():raise ValueError('Source symlink refused')
        inside=path.is_relative_to(ROOT/'src/resectionlab')
        package='.'.join(path.parent.relative_to(ROOT/'src').parts) if inside else ('resectionlab' if str(path.relative_to(ROOT))==CANDIDATE else None)
        inside=inside or package=='resectionlab'
        for node in ast.walk(ast.parse(path.read_text())):
            modules=[]
            if isinstance(node,ast.Import):modules=[r.name for r in node.names]
            elif isinstance(node,ast.ImportFrom):
                if node.level and inside:
                    base='.'.join(package.split('.')[:len(package.split('.'))-node.level+1]);name=base+('.'+node.module if node.module else '')
                elif node.level==0:name=node.module or ''
                else:continue
                modules=[name]+[name+'.'+r.name for r in node.names if r.name!='*']
            for name in modules:
                found=module_path(name)
                if found is not None:paths.add(found)
    return sorted(paths)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--head',required=True);args=parser.parse_args()
    head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True,timeout=5).strip()
    if args.head!=head:raise ValueError('Root HEAD differs')
    output=ROOT/OUTPUT
    if any(p.exists() or p.is_symlink() for p in (output,output.with_name(output.name+'.supervision'),HERE/'source-index.json',HERE/'release-template.json')):
        raise FileExistsError('Existing freeze or scientific attempt; no overwrite/retry')
    if sha(ROOT/BASELINE)!=BASELINE_SHA or sha(ROOT/CANDIDATE)!=CANDIDATE_SHA:raise ValueError('Engine pins changed')
    controls=ROOT/'build/immutable-preview-lazy-copy-v2/generated-controls.json'
    if __import__('json').loads(controls.read_text())['status']!='PASS':raise ValueError('Rebased controls not passing')
    post_controls=ROOT/'build/immutable-preview-lazy-copy-v2/post-exposure-controls.json'
    if __import__('json').loads(post_controls.read_text())['status']!='PASS':raise ValueError('Existing post-exposure overlay controls not passing')
    setup=HERE/'setup-controls.json'
    setup_record=__import__('json').loads(setup.read_text())
    if setup_record['status']!='PASS' or any(sha(HERE/name)!=digest for name,digest in setup_record['source_pins'].items()):raise ValueError('Current complete parent/pair setup control required')
    records,refs=inputs();metadata={INPUT_INDEX:INPUT_SHA,**{r['path']:r['sha256'] for r in refs.values()}}
    metadata[str(setup.relative_to(ROOT))]=sha(setup)
    metadata[str((HERE/'setup-controls.log').relative_to(ROOT))]=sha(HERE/'setup-controls.log')
    metadata[str(controls.relative_to(ROOT))]=sha(controls)
    metadata[str(post_controls.relative_to(ROOT))]=sha(post_controls)
    metadata['build/immutable-preview-lazy-copy-v2/post-exposure-controls.log']=sha(ROOT/'build/immutable-preview-lazy-copy-v2/post-exposure-controls.log')
    metadata['build/immutable-preview-lazy-copy-v2/generated-controls.log']=sha(ROOT/'build/immutable-preview-lazy-copy-v2/generated-controls.log')
    for row in records['public_index']['cases']:
        path=Path(row['path'])
        if not path.is_absolute() or not path.is_relative_to(ROOT) or path.is_symlink() or path.stat().st_size>1024**2 or sha(path)!=row['sha256']:
            raise ValueError('Pinned public TRAIN manifest changed')
        metadata[str(path.relative_to(ROOT))]=row['sha256']
    sources={str(p.relative_to(ROOT)):sha(p) for p in source_closure()}
    for rel in sources:
        if rel.startswith('src/') and subprocess.check_output(['git','show',head+':'+rel],cwd=ROOT,timeout=5)!=(ROOT/rel).read_bytes():
            raise ValueError('Uncommitted canonical source: '+rel)
    index={'version':'fixed025-lazy-copy-pair-runtime-v2','head':head,'source_files':sources,'metadata_files':metadata,
        'closure_scope':'conservative static Python import closure including lazy imports; helpers pinned explicitly; no scientific imports'}
    raw=(json.dumps(index,indent=2,sort_keys=True,allow_nan=False)+'\n').encode();digest=hashlib.sha256(raw).hexdigest()
    release=release_template(head,digest)
    validate_release({**release,'status':'released_one_attempt'})
    try:validate_release(release)
    except ValueError:pass
    else:raise AssertionError('Pending release admitted')
    if any(sha(ROOT/p)!=v for p,v in {**sources,**metadata}.items()):raise ValueError('Bindings changed while freezing')
    if subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True,timeout=5).strip()!=head:raise ValueError('HEAD changed')
    with (HERE/'source-index.json').open('xb') as stream:stream.write(raw)
    with (HERE/'release-template.json').open('x') as stream:json.dump(release,stream,indent=2,sort_keys=True,allow_nan=False);stream.write('\n')
    if any(k in sys.modules for k in ('torch','numpy','resectionlab')):raise AssertionError('Scientific runtime imported')
    print(json.dumps({'status':'pending_root_release','head':head,'source_count':len(sources),'metadata_count':len(metadata),
        'source_index_sha256':digest,'template_sha256':sha(HERE/'release-template.json')},indent=2))
if __name__=='__main__':main()

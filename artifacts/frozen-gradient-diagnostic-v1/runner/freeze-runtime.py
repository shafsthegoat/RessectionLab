"""Metadata/source-only freeze; no scientific imports or acquired payload reads."""
import argparse,ast,hashlib,json,subprocess,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
sys.dont_write_bytecode=True
sys.pycache_prefix=str(HERE/'never-written-final-freeze-pycache')
if Path(sys.pycache_prefix).exists() or Path(sys.pycache_prefix).is_symlink():raise FileExistsError('Fresh no-write source cache namespace required')
sys.path.insert(0,str(HERE))
from batch_contract import HELPER,ORIGINAL_CONTRACT,OUTPUT,PIN_SHA,metadata,release_template,sha,small,need
def relative(path):return path.relative_to(ROOT).as_posix()

def project_module(name):
    if name=='resectionlab':return ROOT/'src/resectionlab/__init__.py'
    if not name.startswith('resectionlab.'):return None
    base=ROOT/'src'/name.replace('.','/')
    if base.with_suffix('.py').is_file():return base.with_suffix('.py')
    if (base/'__init__.py').is_file():return base/'__init__.py'
    return None

def source_closure(seed):
    paths={HERE/name for name in ('batch_contract.py','initial_inventory_worker.py',
        'run_owned.py','freeze-runtime.py')}
    paths.update(ROOT/'build/goal-conditioned-policy-v1'/name
        for name in ('run_contact_owned.py','darwin_fast_sampler.py'))
    paths.add(ROOT/'src/resectionlab/__init__.py')
    paths.update(ROOT/ref['path'] for ref in (HELPER,ORIGINAL_CONTRACT))
    for name in seed['source_files']:
        if name.startswith('src/resectionlab/'):
            path=Path(name)
            need(not path.is_absolute() and '..' not in path.parts and path.suffix=='.py','canonical_source_name')
            paths.add(ROOT/path)
    visited=set()
    while paths-visited:
        path=sorted(paths-visited)[0];visited.add(path)
        if path.is_symlink() or not path.is_file():raise ValueError('Regular source required: '+str(path))
        tree=ast.parse(path.read_text());in_package=path.is_relative_to(ROOT/'src/resectionlab')
        if in_package:package='.'.join(path.parent.relative_to(ROOT/'src').parts)
        for node in ast.walk(tree):
            modules=[]
            if isinstance(node,ast.Import):modules.extend(alias.name for alias in node.names)
            elif isinstance(node,ast.ImportFrom):
                if node.level and in_package:
                    base='.'.join(package.split('.')[:len(package.split('.'))-node.level+1])
                    module=base+('.'+node.module if node.module else '')
                elif node.level==0:module=node.module or ''
                else:continue
                modules.append(module)
                modules.extend(module+'.'+alias.name for alias in node.names if alias.name!='*')
            for module in modules:
                dependency=project_module(module)
                if dependency is not None:paths.add(dependency)
    return sorted(paths)

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--head',required=True);args=parser.parse_args()
    def current():return subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True,timeout=5).strip()
    head=current();need(head==args.head,'exact_committed_HEAD')
    for path in (OUTPUT,OUTPUT.with_name(OUTPUT.name+'.supervision'),HERE/'source-index.json',HERE/'release-template.json'):
        need(not path.exists() and not path.is_symlink(),'exclusive_new_attempt')
    owner,records,pins=metadata();seed=records['training_index'];bindings={}
    def bind(path,digest):
        p=ROOT/path;need(p.suffix=='.json','metadata_only')
        value=small(p,digest);bindings[relative(p)]=digest;return value
    bind(HERE/'input-pins.json',PIN_SHA)
    for ref in pins.values():bind(ref['path'],ref['sha256'])
    for path,digest in seed['metadata_files'].items():bind(path,digest)
    public=bind(owner.PUBLIC_INDEX,owner.PUBLIC_SHA);bind(owner.COHORT,owner.COHORT_SHA)
    need([row['patient_id'] for row in public['cases']]==list(owner.TRAIN),'ordered_four_TRAIN')
    for row in public['cases']:
        manifest=bind(row['path'],row['sha256'])
        need(manifest['role']=='TRAIN' and manifest['public_only'] and not manifest['private_evaluation_files_included'],'public_only_manifest')
    paths=source_closure(seed);sources={relative(path):sha(path) for path in paths}
    for ref in (HELPER,ORIGINAL_CONTRACT):need(sources[ref['path']]==ref['sha256'],'exact_reused_source')
    need(all(sources.get(name)==pin for name,pin in owner.RANKING_SOURCE_PINS.items()),'unchanged_reviewed_ranking_math')
    for path in paths:
        if path.is_relative_to(ROOT/'src'):
            need(subprocess.check_output(['git','show',head+':'+relative(path)],cwd=ROOT,timeout=5)==path.read_bytes(),'uncommitted_canonical:'+relative(path))
    need(current()==head and all(sha(ROOT/name)==pin for name,pin in {**sources,**bindings}.items()),'freeze_stability')
    index={'version':'frozen-gradient-diagnostic-runtime-v1','head':head,'files':{**sources,**bindings},
        'source_files':sources,'metadata_files':bindings,'closure_scope':'current static canonical imports plus original metadata contract and exact reviewed helper; no source overlays'}
    encode=lambda value:(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n').encode()
    raw=encode(index);digest=hashlib.sha256(raw).hexdigest()
    release=release_template(head,{'path':relative(HERE/'source-index.json'),'sha256':digest})
    need(not any(name in sys.modules for name in ('torch','numpy','resectionlab')),'no_scientific_imports')
    for name,payload in (('source-index.json',raw),('release-template.json',encode(release))):
        with (HERE/name).open('xb') as stream:stream.write(payload)
    print(json.dumps({'status':'pending_root_release','head':head,'sources':len(sources),'metadata':len(bindings),
        'source_index_sha256':digest,'template_sha256':sha(HERE/'release-template.json'),'payload_reads':0,'workers_started':0},indent=2))
if __name__=='__main__':main()

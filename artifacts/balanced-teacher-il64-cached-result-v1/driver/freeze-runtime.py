"""Root-only source/metadata freeze and NON-executable template preparation.

Uses only stdlib and bounded saved TRAIN JSON. Imports no project/ML modules.
The actual worker must separately match the canonical protocol constructor.
Root reviews the resulting template and separately releases its one attempt.
"""
import argparse
import ast
import json
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
from pilot_contract import (OUTPUT, TRAIN, CLOSED, PUBLIC_INDEX, PUBLIC_SHA, COHORT,
    COHORT_SHA, PARENT_SECONDS, SUPERVISION_BYTES, expected_configuration, sha, semantic,
    validate_release, BASELINE_FILES, require_baseline, EXECUTION, BALANCED_FILES,
    FIXED64_INDEX, fixed64_metadata)



def relative(path): return str(path.relative_to(ROOT))


def project_module(name):
    if name == 'resectionlab': return ROOT/'src/resectionlab/__init__.py'
    if not name.startswith('resectionlab.'): return None
    base = ROOT/'src'/name.replace('.', '/')
    if base.with_suffix('.py').is_file(): return base.with_suffix('.py')
    if (base/'__init__.py').is_file(): return base/'__init__.py'
    return None


def source_closure():
    """Conservative static import closure, including lazy imports and package init."""
    paths = {HERE/name for name in ('pilot_contract.py', 'cohort_worker.py', 'run_owned.py', 'freeze-runtime.py')}
    paths.update(ROOT/'build/goal-conditioned-policy-v1'/name
        for name in ('run_contact_owned.py', 'darwin_fast_sampler.py'))
    paths.add(ROOT/'src/resectionlab/__init__.py')
    # Hash-guarded dynamic import in pilot_contract is not visible to AST imports.
    paths.add(ROOT/'build/balanced-teacher-il64-v1/pilot_contract.py')
    visited = set()
    while paths-visited:
        path = sorted(paths-visited)[0]; visited.add(path)
        if path.is_symlink() or not path.is_file(): raise ValueError('Regular source required: '+str(path))
        tree = ast.parse(path.read_text())
        in_package = path.is_relative_to(ROOT/'src/resectionlab')
        if in_package:
            package = '.'.join(path.parent.relative_to(ROOT/'src').parts)
        for node in ast.walk(tree):
            modules = []
            if isinstance(node, ast.Import): modules.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                if node.level and in_package:
                    base = '.'.join(package.split('.')[:len(package.split('.'))-node.level+1])
                    module = base + ('.'+node.module if node.module else '')
                elif node.level == 0: module = node.module or ''
                else: continue
                modules.append(module)
                # from . import preflight-style module imports, as well as
                # package-exported submodules, must not disappear from closure.
                modules.extend(module+'.'+alias.name for alias in node.names if alias.name != '*')
            for module in modules:
                dependency = project_module(module)
                if dependency is not None: paths.add(dependency)
    return sorted(paths)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--head',required=True)
    args=parser.parse_args()
    head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True,timeout=5).strip()
    if head!=args.head:raise ValueError('Root supplied HEAD differs')
    output=ROOT/OUTPUT
    if any(p.exists() or p.is_symlink() for p in (output,output.with_name(output.name+'.supervision'))):
        raise FileExistsError('One attempt or reservation exists')
    for name in ('source-index.json','release-template.json'):
        if (HERE/name).exists():raise FileExistsError('Freeze metadata already exists')
    if sha(ROOT/PUBLIC_INDEX)!=PUBLIC_SHA or sha(ROOT/COHORT)!=COHORT_SHA:
        raise ValueError('Original public index/cohort changed')
    public=json.loads((ROOT/PUBLIC_INDEX).read_text());rows=public['cases']
    if len(rows)!=4 or {r['patient_id'] for r in rows}!=set(TRAIN) or any(r['role']!='TRAIN' for r in rows):
        raise ValueError('Exact four TRAIN manifests required')
    metadata={PUBLIC_INDEX:PUBLIC_SHA,COHORT:COHORT_SHA}
    for row in rows:
        path=Path(row['path'])
        if (not path.is_absolute() or not path.is_relative_to(ROOT) or path.is_symlink()
                or not path.is_file() or path.stat().st_size>1024**2 or sha(path)!=row['sha256']):
            raise ValueError('Pinned public manifest changed')
        metadata[relative(path)]=row['sha256']
    paths=source_closure();bindings={relative(p):sha(p) for p in paths}
    for path in paths:
        if path.is_relative_to(ROOT/'src'):
            committed=subprocess.check_output(['git','show',head+':'+relative(path)],cwd=ROOT,timeout=5)
            if committed!=path.read_bytes():raise ValueError('Canonical source not committed: '+relative(path))
    protocol,limits=expected_configuration();baseline=require_baseline(protocol)
    fixed,refs=fixed64_metadata()
    metadata.update({ref['path']:ref['sha256'] for ref in (*BASELINE_FILES.values(),*BALANCED_FILES.values(),*refs.values(),FIXED64_INDEX)})
    old_sources=fixed['source_index']['source_files']
    changes={p:{'before':old_sources.get(p),'after':bindings.get(p)} for p in sorted(set(old_sources)|set(bindings)) if old_sources.get(p)!=bindings.get(p)}
    if any(sha(ROOT/p)!=v for p,v in bindings.items()) or any(sha(ROOT/p)!=v for p,v in metadata.items()):
        raise ValueError('Source/metadata changed during protocol construction')
    if subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True,timeout=5).strip()!=head:
        raise ValueError('HEAD changed during freeze')
    index={'version':'balanced-teacher-IL64-cached-runtime-v1','head':head,'source_files':bindings,
        'metadata_files':metadata,'source_delta_from_recollected64':changes,
        'closure_scope':'conservative repository Python import closure including lazy imports; external runtime recorded separately',
        'scope':'fixed four TRAIN, no held-out payload; source changes require root review before release'}
    release={'version':'balanced-teacher-IL64-cached-release-v1','status':'pending_root_release',
        'expected_head':head,'output':OUTPUT,'TRAIN':list(TRAIN),'closed_roles':CLOSED,
        'SELECT_EVAL_execution':False,'attempts':1,'automatic_retry':False,
        'parent_seconds':PARENT_SECONDS,'supervision_bytes':SUPERVISION_BYTES,
        'cohort_limits':limits,'limits':{k:v for k,v in limits.items() if k not in ('output_bytes','checkpoint_bytes')},
        'execution_limits':EXECUTION,'learning_protocol':protocol,'learning_protocol_hash':semantic(protocol),
        'public_manifest_index':{'path':PUBLIC_INDEX,'sha256':PUBLIC_SHA},'baseline':BASELINE_FILES,'balanced_baseline':BALANCED_FILES,'fixed64_baseline':FIXED64_INDEX,
        'source_index':{'path':relative(HERE/'source-index.json'),'sha256':None},
        'objective':'unchanged 0.5 mean motion NLL + 0.5 mean STOP NLL',
        'protocol_change':'teacher observation storage and 64MiB payload allowance only',
        'optimizer_lifetime':'baseline final accumulator reference lifetime preserved; separate cleanup deferred',
        'selection':'same fixed64 endpoint, no threshold/intermediate checkpoint/success-metric selection',
        'completion':'all64 update math/tensor hashes,5 exact teacher logits,4 exact full native histories,8 released source visits,320 loss forwards,64 shared Adam updates,1 reload,actual cache receipt',
        'claim':'runtime parity benchmark on unchanged four TRAIN inputs; no new model or SELECT/EVAL execution'}
    index_bytes=(json.dumps(index,indent=2,sort_keys=True,allow_nan=False)+'\n').encode()
    import hashlib
    release['source_index']['sha256']=hashlib.sha256(index_bytes).hexdigest()
    validate_release({**release,'status':'released_one_attempt'})
    try:validate_release(release)
    except ValueError:pass
    else:raise AssertionError('Non-executable template admitted')
    def write(path,value):
        with path.open('x') as stream:json.dump(value,stream,indent=2,sort_keys=True,allow_nan=False);stream.write('\n')
    write(HERE/'source-index.json',index)
    if sha(HERE/'source-index.json')!=release['source_index']['sha256']:raise ValueError('Written index differs')
    write(HERE/'release-template.json',release)
    if any(name in sys.modules for name in ('torch','numpy','resectionlab')):raise AssertionError('Metadata-only freeze imported scientific runtime')
    print(json.dumps({'status':'pending_root_release','head':head,'source_count':len(bindings),
        'metadata_count':len(metadata),'source_delta_count':len(changes),'learning_protocol_hash':release['learning_protocol_hash'],
        'pins':{name:sha(HERE/name) for name in ('pilot_contract.py','cohort_worker.py','run_owned.py','freeze-runtime.py','source-index.json','release-template.json')}},indent=2))

if __name__=='__main__':main()

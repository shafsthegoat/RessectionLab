"""Metadata-only profile freeze, reusing the reviewed SELECT AST closure.

No numerical imports, array stats, model loads or worker execution. Source names
from the completed SELECT freeze seed the closure; bytes always come from the
root-supplied current commit, never from an old source snapshot.
"""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import subprocess
import sys

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.dont_write_bytecode=True
sys.pycache_prefix=str(HERE/'never-written-final-freeze-pycache')
if Path(sys.pycache_prefix).exists() or Path(sys.pycache_prefix).is_symlink():
    raise FileExistsError('Fresh no-write source cache namespace required')
sys.path.insert(0,str(HERE))
from batch_contract import INPUTS,OUTPUT,sha,small,need,release_template

SEED={'path':'build/post-exposure-select013-comparison-v1/source-index.json',
      'sha256':'1882d4b16eac077a0ec7fbe25bf126db27e7e029b962937c91ba771c7a7245c4'}

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
        'profile_window.py','run_owned.py','freeze-runtime.py')}
    paths.update(ROOT/'build/goal-conditioned-policy-v1'/name
        for name in ('run_contact_owned.py','darwin_fast_sampler.py'))
    paths.add(ROOT/'src/resectionlab/__init__.py')
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
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--head',required=True)
    args=parser.parse_args()
    def head_now():
        return subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True,timeout=5).strip()
    head=head_now();need(head==args.head,'exact_committed_HEAD_required')
    for path in (OUTPUT,OUTPUT.with_name(OUTPUT.name+'.supervision'),HERE/'source-index.json',HERE/'release-template.json'):
        if path.exists() or path.is_symlink():raise FileExistsError('Existing freeze/attempt: '+str(path))
    metadata={}
    def bind(ref):
        path=ROOT/ref['path'];need(path.is_relative_to(ROOT) and '..' not in path.parts,'workspace_metadata')
        value=small(path,ref['sha256']);name=relative(path)
        need(name not in metadata or metadata[name]==ref['sha256'],'conflicting_descriptor')
        metadata[name]=ref['sha256'];return value
    seed=bind(SEED)
    records={name:bind(ref) for name,ref in INPUTS.items()}
    original=records['original_release'];public=bind(original['public_index']);cohort=bind(original['cohort'])
    selected=[row for row in public['cases'] if row['patient_id']=='ReMIND-045']
    need(len(selected)==1,'exact_one_TRAIN045_manifest')
    manifest=bind(selected[0]);members=[row for row in cohort['members'] if row['subject']=='ReMIND-045']
    need(len(members)==1 and members[0]['role']==manifest['role']==selected[0]['role']=='TRAIN'
         and members[0]['patient_group']==manifest['patient_group']=='ReMIND:045'
         and manifest['patient_id']=='ReMIND-045' and manifest['public_only'] is True
         and manifest['private_evaluation_files_included'] is False
         and manifest['source_bindings']['cohort_sha256']==original['cohort']['sha256']
         and manifest['source_domain_condition']==original['occupancy_condition']
         and set(manifest['input_files'])=={'image','supplied_support','supplied_whole_tumor','whole_tumor_domain','supplied_support_domain'},'frozen_public_TRAIN_inputs')
    # Inspect descriptors only; do not stat, hash, resolve or open their arrays.
    for ref in manifest['input_files'].values():
        path=Path(ref['path'])
        need(path.is_absolute() and path.is_relative_to(ROOT) and '..' not in path.parts,'public_array_descriptor')
    need(records['case']['status']=='completed_fixed_post_exposure_route'
         and records['audit']['accepted'] is True
         and records['plan']['accounting']['complete'] is True
         and len(records['plan']['actions'])==13 and records['plan']['actions'][-1]=='STOP'
         and records['plan']['actions']==records['case']['actions']
         and records['audit']['outcomes']==records['case']['full_route_outcomes']
         and [row['action_id'] for row in records['metrics']['history']]==records['plan']['actions'],
         'completed_teacher_route')
    paths=source_closure(seed);sources={relative(path):sha(path) for path in paths}
    for path in paths:
        if path.is_relative_to(ROOT/'src'):
            committed=subprocess.check_output(['git','show',head+':'+relative(path)],cwd=ROOT,timeout=5)
            need(committed==path.read_bytes(),'uncommitted_canonical_source:'+relative(path))
    need(all(sha(ROOT/p)==digest for p,digest in {**sources,**metadata}.items()),'bindings_changed_during_freeze')
    need(head_now()==head,'HEAD_changed_during_freeze')
    index={'version':'native-function-profile-runtime-v1','head':head,'files':{**sources,**metadata},
        'source_files':sources,'metadata_files':metadata,
        'closure_scope':'reviewed SELECT canonical path seed plus current static lazy-import closure; no source overlays',
        'scope':'TRAIN045 profile only; no array, checkpoint, model or scientific worker read/run during freeze'}
    encode=lambda value:(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n').encode()
    index_raw=encode(index);index_sha=hashlib.sha256(index_raw).hexdigest()
    release=release_template(head,{'path':relative(HERE/'source-index.json'),'sha256':index_sha})
    release_raw=encode(release)
    need(release['execution_released'] is False,'freeze_cannot_release')
    need(not any(name in sys.modules for name in ('numpy','torch','resectionlab')),'no_scientific_imports')
    with (HERE/'source-index.json').open('xb') as stream:stream.write(index_raw)
    with (HERE/'release-template.json').open('xb') as stream:stream.write(release_raw)
    print(json.dumps({'status':'pending_root_release','head':head,
        'source_count':len(sources),'canonical_source_count':sum(p.startswith('src/') for p in sources),
        'metadata_count':len(metadata),'array_reads':0,'workers_started':0,
        'source_index_sha256':index_sha,'release_template_sha256':hashlib.sha256(release_raw).hexdigest()},indent=2))

if __name__=='__main__':main()

"""Metadata-only freeze after root promotes the one source-validator extraction."""
import argparse,ast,hashlib,json,subprocess,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
sys.dont_write_bytecode=True
sys.path.insert(0,str(HERE))
from batch_contract import OWNER,OUTPUT,SUBJECTS,sha,small,need,metadata,template
SEED={'path':'build/native-capsule-cache-train045-pair-v1/source-index.json'}
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
        'run_owned.py','freeze-runtime.py','hybrid_proposal.py','patient_sparse.py')}
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
    p=argparse.ArgumentParser();p.add_argument('--head',required=True);a=p.parse_args()
    head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip();need(head==a.head,'root_HEAD')
    for name in ('attempt-01','attempt-01.supervision','source-index.json','release-template.json'):
        need(not (HERE/name).exists(),'fresh_freeze_and_attempt')
    owner,prior,baseline=metadata();pins={}
    def bind(ref):
        value=small(ref['path'],ref['sha256']);pins[ref['path']]=ref['sha256'];return value
    refs=dict(owner.INPUTS)
    refs.update({'IL:'+k:owner.BASELINE_REFS[k] for k in ('result','parent','recovery_result','recovery_parent','outcomes')})
    for ref in refs.values():bind(ref)
    original=prior['release'];public=bind(original['public_index']);bind(original['cohort'])
    for subject in SUBJECTS:
        rows=[r for r in public['cases'] if r['patient_id']==subject];need(len(rows)==1,'exact_public_person')
        manifest=bind(rows[0]);need(manifest['role']=='TRAIN' and manifest['public_only'] and not manifest['private_evaluation_files_included'],'public_TRAIN_only')
        need(set(manifest['input_files'])=={'image','supplied_support','supplied_whole_tumor','whole_tumor_domain','supplied_support_domain'},'exact_five_public_arrays')
    seed=small(SEED['path']);paths=source_closure(seed);paths.append(ROOT/OWNER['path'])
    sources={relative(path):sha(path) for path in sorted(set(paths))}
    need(sources[OWNER['path']]==OWNER['sha256'],'exact_teacher_owner')
    for path in paths:
        if path.is_relative_to(ROOT/'src'):
            committed=subprocess.check_output(['git','show',head+':'+relative(path)],cwd=ROOT)
            need(committed==path.read_bytes(),'canonical_source_uncommitted:'+relative(path))
    candidate=HERE/'stage/resectionlab/patient_planning_admission.py'
    need(candidate.read_bytes()==(ROOT/'src/resectionlab/patient_planning_admission.py').read_bytes(),'root_source_validator_promotion_required')
    need(all(sha(ROOT/p)==h for p,h in {**sources,**pins}.items()),'freeze_changed')
    index={'version':'fixed-four-ridge-proposal-runtime-v1','head':head,'files':{**sources,**pins},'source_files':sources,'metadata_files':pins,
        'payloads_opened':False,'scope':'one fixed TRAIN ridge fit plus three proposal comparisons; root release required'}
    encode=lambda v:(json.dumps(v,indent=2,sort_keys=True,allow_nan=False)+'\n').encode()
    raw=encode(index);ref={'path':relative(HERE/'source-index.json'),'sha256':hashlib.sha256(raw).hexdigest()}
    release=template(head,ref);need(not release['execution_released'],'unreleased_template')
    need(not any(n in sys.modules for n in ('numpy','torch','resectionlab')),'metadata_only_freeze')
    (HERE/'source-index.json').write_bytes(raw);(HERE/'release-template.json').write_bytes(encode(release))
    print(json.dumps({'source_index':ref,'release_template_sha256':sha(HERE/'release-template.json'),'execution_released':False}))
if __name__=='__main__':main()

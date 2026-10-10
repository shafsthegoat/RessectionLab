"""Root-only metadata freeze; constructs NON-executable separate endpoint templates."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import subprocess
import sys
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
sys.path.insert(0,str(HERE))
from pilot_contract import (TRAIN,CLOSED,PUBLIC_INDEX,PUBLIC_SHA,COHORT,COHORT_SHA,
    METHODS,INPUTS,PARENT_SECONDS,SUPERVISION_BYTES,expected_configuration,sha,semantic,
    validate_release,inputs,output_for,small,ORIGINAL_IL,ORIGINAL_IL_CHECKPOINT,
    RECOVERY_RECEIPT,RELEASE_VERSION,recovered_il_evidence)

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
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--head',required=True);parser.add_argument('--method',choices=('RL',),default='RL')
    parser.add_argument('--il045-recovery-receipt-sha256',required=True)
    args=parser.parse_args();method=args.method
    head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True,timeout=5).strip()
    if head!=args.head:raise ValueError('Root supplied HEAD differs')
    output=ROOT/output_for(method);folder=output.parent
    if folder.exists() or folder.is_symlink():raise FileExistsError('Freeze/attempt folder exists; no retry')
    if sha(ROOT/PUBLIC_INDEX)!=PUBLIC_SHA or sha(ROOT/COHORT)!=COHORT_SHA:raise ValueError('Fixed public index/cohort changed')
    public=json.loads((ROOT/PUBLIC_INDEX).read_text());rows=public['cases']
    if len(rows)!=4 or {r['patient_id'] for r in rows}!=set(TRAIN) or any(r['role']!='TRAIN' for r in rows):
        raise ValueError('Exact fixed four TRAIN manifests required')
    metadata={PUBLIC_INDEX:PUBLIC_SHA,COHORT:COHORT_SHA}
    for row in rows:
        path=Path(row['path'])
        if (not path.is_absolute() or not path.is_relative_to(ROOT) or path.is_symlink()
                or not path.is_file() or path.stat().st_size>1024**2 or sha(path)!=row['sha256']):
            raise ValueError('Pinned public manifest changed')
        metadata[relative(path)]=row['sha256']
    inputs();metadata.update({ref['path']:ref['sha256'] for ref in INPUTS.values()})
    matched={'path':RECOVERY_RECEIPT,'sha256':args.il045_recovery_receipt_sha256}
    evidence=recovered_il_evidence(matched)
    metadata.update({ref['path']:ref['sha256'] for ref in ORIGINAL_IL.values()})
    metadata[ORIGINAL_IL_CHECKPOINT['path']]=ORIGINAL_IL_CHECKPOINT['sha256']
    metadata.update({ref['path']:ref['sha256'] for ref in evidence['recovery_refs'].values()})
    paths=source_closure();bindings={relative(p):sha(p) for p in paths}
    old_sources=evidence['original']['source_index']['source_files']
    corrected={'src/resectionlab/evaluation.py','src/resectionlab/patient_planning_preflight.py'}
    recovery_sources=small(evidence['recovery_refs']['source_index'])['files']
    for path,digest in old_sources.items():
        if path.startswith('src/'):
            expected=recovery_sources[path] if path in corrected else digest
            if bindings.get(path)!=expected:raise ValueError('Unexpected scientific source change since original IL: '+path)
    for path in paths:
        if path.is_relative_to(ROOT/'src'):
            committed=subprocess.check_output(['git','show',head+':'+relative(path)],cwd=ROOT,timeout=5)
            if committed!=path.read_bytes():raise ValueError('Canonical source not committed: '+relative(path))
    protocol,limits=expected_configuration(method)
    if any(sha(ROOT/p)!=v for p,v in bindings.items()) or any(sha(ROOT/p)!=v for p,v in metadata.items()):
        raise ValueError('Source/metadata changed during preparation')
    if subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True,timeout=5).strip()!=head:raise ValueError('HEAD changed')
    index={'version':RELEASE_VERSION,'method':method,'head':head,'source_files':bindings,
        'metadata_files':metadata,'closure_scope':'conservative repository Python imports, including lazy imports; runtime separately recorded'}
    index_bytes=(json.dumps(index,indent=2,sort_keys=True,allow_nan=False)+'\n').encode()
    release={'version':RELEASE_VERSION,'status':'pending_root_release','method':method,
        'expected_head':head,'output':output_for(method),'TRAIN':list(TRAIN),'closed_roles':CLOSED,
        'SELECT_EVAL_execution':False,'new_four_training':True,'attempts':1,'automatic_retry':False,
        'parent_seconds':METHODS[method]['parent_seconds'],'supervision_bytes':SUPERVISION_BYTES,'cohort_limits':limits,
        'limits':{k:v for k,v in limits.items() if k not in ('output_bytes','checkpoint_bytes')},
        'execution_limits':METHODS[method],'learning_protocol':protocol,'learning_protocol_hash':semantic(protocol),
        'inputs':INPUTS,'source_index':{'path':relative(folder/'source-index.json'),'sha256':hashlib.sha256(index_bytes).hexdigest()},
        'public_manifest_index':{'path':PUBLIC_INDEX,'sha256':PUBLIC_SHA},
        'completion':'fixed own updates and reloaded endpoint,29 teacher logits,4 complete learned replays; no performance threshold',
        'claim':'same world/model/init, distinct endpoint/objective and measured costs; no equal-cost or held-out claim'}
    if matched is not None:release['matched_IL_recovery']=matched
    validate_release({**release,'status':'released_one_attempt'})
    try:validate_release(release)
    except ValueError:pass
    else:raise AssertionError('Non-executable template admitted')
    folder.mkdir(exist_ok=False)
    for name,value in (('source-index.json',index),('release-template.json',release)):
        with (folder/name).open('x') as stream:json.dump(value,stream,indent=2,sort_keys=True,allow_nan=False);stream.write('\n')
    if sha(folder/'source-index.json')!=release['source_index']['sha256']:raise ValueError('Written index differs')
    if any(n in sys.modules for n in ('torch','numpy','resectionlab')):raise AssertionError('Metadata freezer imported runtime')
    print(json.dumps({'status':'pending_root_release','method':method,'head':head,'source_count':len(bindings),
        'metadata_count':len(metadata),'learning_protocol_hash':release['learning_protocol_hash'],
        'pins':{str(p.relative_to(HERE)):sha(p) for p in [*(HERE/n for n in ('pilot_contract.py','cohort_worker.py','run_owned.py','freeze-runtime.py')),folder/'source-index.json',folder/'release-template.json']}},indent=2))

if __name__=='__main__':main()

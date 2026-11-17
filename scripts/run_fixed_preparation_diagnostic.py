#!/usr/bin/env python3
"""Freeze original sources and run the fixed-tip diagnostic without fitting."""
from __future__ import annotations
import argparse,datetime,hashlib,importlib.util,io,json,os,platform,shutil,subprocess,sys,tarfile,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
ORIGINAL_COMMIT='6e04f98c0ac512dc932dc2aaf993df9e9ebc561f'
ORIGINAL=ROOT/'testcases/overnight-sequential-v1/benchmark-task-v2'
DEFAULT=ROOT/'testcases/canonical-endpoint-control-v1/diagnostic'
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text())
def save(path,value):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix(path.suffix+'.tmp')
 tmp.write_text(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n');tmp.replace(path)
def freeze_bundle(out):
 """Literal source archive, never a Git checkout/branch/worktree."""
 bundle=out/'source-bundle-6e04';receipt=bundle/'bundle.json'
 if receipt.exists():
  manifest=read(receipt)
  for path,digest in manifest['files'].items():
   if sha(bundle/path)!=digest:raise ValueError('Frozen source bundle changed: '+path)
  return bundle,manifest
 bundle.mkdir(parents=True,exist_ok=True)
 archive=subprocess.check_output(['git','archive',ORIGINAL_COMMIT,'src/resectionlab'],cwd=ROOT)
 with tarfile.open(fileobj=io.BytesIO(archive)) as stream:stream.extractall(bundle,filter='data')
 package=bundle/'src/resectionlab';current=ROOT/'src/resectionlab';build=read(current/'_geometry_cpp.build.json')
 for name in [build['binaryName'],'_geometry_cpp.build.json',*build['buildIdentity']['sources']]:
  source=current/name
  if (package/name).exists() and sha(package/name)!=sha(source):raise ValueError('Original native source differs: '+name)
  shutil.copy2(source,package/name)
 if sha(package/build['binaryName'])!=build['binarySha256']:raise ValueError('Native binary digest mismatch')
 files={str(p.relative_to(bundle)):sha(p) for p in sorted(package.rglob('*')) if p.is_file()}
 old={str(p.relative_to(ORIGINAL)):sha(p) for p in sorted(ORIGINAL.rglob('*')) if p.is_file() and '__pycache__' not in p.parts}
 manifest=dict(schema='resectionlab.original-source-bundle.v1',commit=ORIGINAL_COMMIT,
  createdUtc=datetime.datetime.now(datetime.timezone.utc).isoformat(),files=files,
  nativeBinarySha256=build['binarySha256'],nativeBuildIdentity=build['buildIdentity'],
  originalEvidenceRoot=str(ORIGINAL),originalEvidenceFiles=old,
  meaning='Literal git archive plus unchanged installed native bytes; no Git checkout/worktree')
 save(receipt,manifest);return bundle,manifest
def load_original(bundle):
 if 'resectionlab' in sys.modules:raise ValueError('Select original package before importing resectionlab')
 sys.path.insert(0,str(bundle/'src'))
 from resectionlab import native_backend
 if native_backend._identity['selected']!='cpp':raise ValueError('Original compiled environment required')
 if native_backend._identity['binarySha256']!=read(bundle/'bundle.json')['nativeBinarySha256']:raise ValueError('Native binary changed')
 spec=importlib.util.spec_from_file_location('fixed_preparation_controller',ROOT/'src/resectionlab/surgical_fixed_preparation_control.py')
 controller=importlib.util.module_from_spec(spec);spec.loader.exec_module(controller)
 return controller,native_backend._identity
def check_original_immutable(manifest):
 for path,digest in manifest['originalEvidenceFiles'].items():
  if sha(ORIGINAL/path)!=digest:raise ValueError('Original experiment changed: '+path)
def run(out,parents):
 if (out/'evaluation.json').exists() or (out/'results').exists():
  raise ValueError('Preserve existing diagnostic attempts; choose a new output version')
 bundle,manifest=freeze_bundle(out);controller,backend=load_original(bundle)
 import torch
 torch.set_num_threads(1);torch.set_num_interop_threads(1)
 from resectionlab.surgical_sequential_task import generated_task
 from resectionlab.surgical_sequential_benchmark import ComputeTier,verify_replay
 protocol=read(ORIGINAL/'protocol.json')
 for path,digest in protocol['sourceHashes'].items():
  actual=sha(bundle/path) if path.startswith('src/') else hashlib.sha256(subprocess.check_output(['git','show',ORIGINAL_COMMIT+':'+path],cwd=ROOT)).hexdigest()
  if actual!=digest:raise ValueError('Historical implementation differs: '+path)
 for c in protocol['cohort']:
  if generated_task(c['parent']).audit!={k:v for k,v in c.items() if k!='split'}:raise ValueError('Original cohort differs')
 started=time.monotonic();record=dict(schema='resectionlab.fixed-preparation-diagnostic.v1',status='running',
  scope='Previously inspected layouts; diagnostic, not untouched evaluation',
  sourceBundle=dict(path=str(bundle),sha256=sha(bundle/'bundle.json')),sourceCommit=ORIGINAL_COMMIT,
  originalProtocol=dict(path=str(ORIGINAL/'protocol.json'),sha256=sha(ORIGINAL/'protocol.json')),
  originalEvaluation=dict(path=str(ORIGINAL/'evaluation.json'),sha256=sha(ORIGINAL/'evaluation.json')),
  controllerSha256=sha(ROOT/'src/resectionlab/surgical_fixed_preparation_control.py'),runnerSha256=sha(Path(__file__)),
  nativeBackend=backend,hardware=dict(platform=platform.platform(),python=sys.version,torch=torch.__version__,threads=1,device='cpu'),
  startedUtc=datetime.datetime.now(datetime.timezone.utc).isoformat(),parents=parents,tiers=protocol['onlineTiers'],
  rows=[],failures=[],optimizerUpdates=0,historicalTiming='Prior comparators retain original wall times; no matched retiming claim')
 save(out/'protocol.json',record)
 for tier in protocol['onlineTiers']:
  for parent in parents:
   row=dict(parent=parent,tier=tier['name'],method=controller.METHOD,seed=0)
   try:
    row=controller.run_fixed_preparation(generated_task(parent),ComputeTier(**tier))
    row['replay']=verify_replay(generated_task(parent),row)
    row['diagnosticProvenance']=dict(sourceCommit=ORIGINAL_COMMIT,sourceBundleManifestSha256=sha(bundle/'bundle.json'),controllerSha256=record['controllerSha256'])
   except Exception as error:
    row['failure']=dict(errorType=type(error).__name__,error=str(error))
    if hasattr(error,'receipt'):row['replay']=error.receipt
    record['failures'].append(dict(parent=parent,tier=tier['name'],**row['failure']))
   save(out/'results'/f'{tier["name"]}-{parent}-{controller.METHOD}-0.json',row)
   record['rows'].append(row);record['wallSeconds']=time.monotonic()-started
   save(out/'evaluation-progress.json',dict(completedRows=len(record['rows']),failures=record['failures'],wallSeconds=record['wallSeconds']))
 check_original_immutable(manifest)
 if record['controllerSha256']!=sha(ROOT/'src/resectionlab/surgical_fixed_preparation_control.py'):raise ValueError('Controller changed during diagnostic')
 record['status']='complete' if not record['failures'] else 'retained-failures';record['originalEvidenceBytesUnchanged']=True
 save(out/'evaluation.json',record)
 print(json.dumps(dict(path=str(out/'evaluation.json'),rows=len(record['rows']),failures=record['failures'],wallSeconds=record['wallSeconds'])))
def main():
 parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,default=DEFAULT)
 parser.add_argument('--freeze-only',action='store_true');parser.add_argument('--parents',type=int,nargs='+',default=list(range(200,216)))
 args=parser.parse_args();out=args.output.resolve()
 if args.freeze_only:print(freeze_bundle(out)[0])
 else:run(out,args.parents)
if __name__=='__main__':main()

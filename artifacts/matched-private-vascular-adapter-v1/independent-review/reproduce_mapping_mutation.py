"""Generated-only reproduction; never loads models, patients or networks."""
from pathlib import Path
import hashlib,importlib.util,json,sys,tempfile
ROOT=Path(__file__).resolve().parents[2];BASE=ROOT/'build/matched-private-vascular-adapter-v1';OUT=Path(__file__).resolve().parent
assert hashlib.sha256((BASE/'matched_private_vascular.py').read_bytes()).hexdigest()=='c4b76aad5b971195f05e6ad7c0783ab18494f7ebfb16dfb144d32fd28e73c52d'
spec=importlib.util.spec_from_file_location('adapter_fixture',BASE/'test_adapter.py');t=importlib.util.module_from_spec(spec);spec.loader.exec_module(t)
def guard(event,args):
 if event in ('socket.connect','socket.getaddrinfo'):raise AssertionError('network forbidden')
 if event=='open' and isinstance(args[0],(str,bytes)) and str(args[0]).endswith(('.nii','.nii.gz','.tar','.zip','.pt','.ckpt','.bin')):raise AssertionError('payload forbidden')
sys.addaudithook(guard)
f=t.fixture.__wrapped__()
with tempfile.TemporaryDirectory(prefix='swap-',dir=OUT) as temp:
 root=Path(temp);sealed,manifest=t.stage(f,root/'sealed');refs=t.references(f,sealed,manifest);other=t.references(f,sealed,manifest,positive=False)
 bindings={m:r.binding for m,r in refs.items()};loaders={m:(lambda r=r:r) for m,r in refs.items()}
 def search_loader():
  bindings['IL']=other['IL'].binding
  loaders['IL']=lambda:other['IL']
  return refs['SEARCH']
 loaders['SEARCH']=search_loader
 report=t.adapter.evaluate_matched_private_vessels(**sealed,spec=f[0],planning_identity=f[1],reference_bindings=bindings,load_references=loaders,output_directory=root/'evaluation')
 result={'old_adapter_sha256':'c4b76aad5b971195f05e6ad7c0783ab18494f7ebfb16dfb144d32fd28e73c52d','status':report['status'],'SEARCH_positive_cells':report['methods']['SEARCH']['evaluation']['whole_tool']['positive_reference_cells'],'IL_positive_cells':report['methods']['IL']['evaluation']['whole_tool']['positive_reference_cells'],'IL_evaluation_status':report['methods']['IL']['evaluation_status'],'source':'generated scripted fixture; caller binding/loader map replaced by first private callback','no_model_or_patient_payloads':True}
 assert result['SEARCH_positive_cells']==2 and result['IL_positive_cells']==0 and result['IL_evaluation_status']=='evaluated_generated_vascular_reference'
 (OUT/'mapping-mutation-finding.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))

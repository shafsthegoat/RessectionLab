"""One-time tiny generated whole-report parity; never an original payload read."""
from pathlib import Path
import hashlib
import importlib.util
import json
import os
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
baseline = ROOT/'src/resectionlab/private_vascular_evaluation.py'
assert hashlib.sha256(baseline.read_bytes()).hexdigest() == 'a4ca5a710fa6b8f02251e7d30d5e3fc6bf3602a8f6b2ca9ce7af6e809281856d'
sys.path[:0] = [str(ROOT/'src'),str(ROOT/'tests')]
import resectionlab
resectionlab.__path__.insert(0,str(HERE/'stage/src/resectionlab'))
from resectionlab import private_vascular_evaluation as tiled
tiled.ROOT = ROOT
spec = importlib.util.spec_from_file_location('legacy_private_for_parity',baseline)
legacy = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = legacy
spec.loader.exec_module(legacy)
import test_private_vascular_evaluation as generated
from resectionlab.research_estimate_planning import research_planning_from_estimates
from resectionlab.core import semantic_digest

prohibited=[]
def guard(event,args):
    if event in ('subprocess.Popen','os.system','os.exec','socket.connect'):
        prohibited.append(event);raise AssertionError('No process/network')
    if event=='open' and isinstance(args[0],(str,bytes)):
        name=os.fsdecode(args[0]).lower()
        if name.endswith(('.nii','.nii.gz','.mat','.tar','.pt','.ckpt','.bin','.npy','.npz','.dcm','.h5')):
            prohibited.append(name);raise AssertionError('No original payloads')
sys.addaudithook(guard)
started=time.monotonic()
records=[]
cases=[('identity',{},{}),('rotated',{}, {'angle':.37}),
       ('partial',{}, {'partial':True}),('empty_partial',{}, {'positive':False,'partial':True}),
       ('outside',{}, {'positive':False,'small_fov':True}),
       ('noncongruent',{}, {'noncongruent':True}),
       ('roi_shifted',{'roi_start':(2,2,0),'roi_stop':(8,8,7)},{}),('stop',{}, {})]
with tempfile.TemporaryDirectory(prefix='legacy-parity-',dir=HERE) as directory:
    directory=Path(directory)
    for name,fixture_kwargs,reference_kwargs in cases:
        case=directory/name;case.mkdir()
        spec=generated.helpers.fixture_spec(**fixture_kwargs)
        if name=='stop':
            plan=generated.helpers.plan(spec)
        else:
            plan=research_planning_from_estimates(spec,method='SEARCH',
                configuration_hash=semantic_digest({'generated_parity_case':name}),planner=generated.fixed_moves)
        public=generated.identity(spec);seal=case/'seal.json'
        sha=tiled.write_strategy_seal(seal,plan=plan,spec=spec,planning_identity=public)
        setup=(spec,plan,public,seal,sha)
        ref=generated.reference(setup,**reference_kwargs)
        old_binding=legacy.VascularReferenceBinding(**ref.binding.record())
        old_ref=legacy.VascularReference(old_binding,ref.mask,ref.coverage,ref.affine_ras_mm)
        outputs=[]
        for module,reference,label in ((legacy,old_ref,'legacy'),(tiled,ref,'tiled')):
            result=module.evaluate_private_vessels(seal_path=seal,seal_sha256=sha,spec=spec,
                planning_identity=public,reference_binding=reference.binding,load_reference=lambda r=reference:r,
                output_directory=case/label,wall_seconds=30.)
            assert result['status']=='evaluated_generated_vascular_reference', (name,label,result)
            result.pop('elapsed_seconds')
            for key in ('contact_work','contact_budget','tile_pruning_padding_mm'):
                result.pop(key,None)
            outputs.append(result)
        assert outputs[0]==outputs[1], (name,outputs)
        records.append({'case':name,'entire_previous_report_equal_except_elapsed_and_additive_work':True,
                        'report_semantic_sha256':semantic_digest(outputs[0]),
                        'whole_tool':outputs[0]['whole_tool'],'removed_overlap':outputs[0]['removed_overlap']})
assert not prohibited
paths=[baseline,HERE/'stage/src/resectionlab/private_vascular_evaluation.py',
       HERE/'stage/src/resectionlab/vascular_contact_streaming.py',Path(__file__)]
record={'status':'all_generated_report_parity_passed','cases':records,'patient_model_native_network_access':False,
        'elapsed_seconds':time.monotonic()-started,
        'sources':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}}
(HERE/'legacy-report-parity.json').write_text(json.dumps(record,indent=2,sort_keys=True,allow_nan=False)+'\n')
print(json.dumps({'status':record['status'],'cases':len(records),'elapsed_seconds':record['elapsed_seconds']}))

"""One saved-case encoding comparison. No patient engine construction or replay."""
from pathlib import Path
import hashlib, json, subprocess, sys, tempfile, time, types
from zipfile import ZipFile
ROOT=next(p for p in Path(__file__).resolve().parents if (p/'src/resectionlab').is_dir())
sys.path.insert(0,str(ROOT/'src'))
from resectionlab import imaging, case_array_payload
from resectionlab.workspace_bundle import load_workspace,save_workspace

def deny_physics(frame,event,arg):
    if event=='call' and frame.f_code.co_filename.endswith('/surgical_engine.py') and frame.f_code.co_name in ('__init__','step','restore','preview'):
        raise AssertionError('Encoding comparison must not construct or replay a patient engine')
    return deny_physics
sys.setprofile(deny_physics)
root=Path(__file__).resolve().parents[1]
source=root/'primary-v2/checkpoints/primary-over-sixty-percent.ressectionlab'
original=ROOT/'testcases/cases/meningioma-sub-06-standard-v2.ressectionlab'
source_sha=hashlib.sha256(source.read_bytes()).hexdigest()
original_sha=hashlib.sha256(original.read_bytes()).hexdigest()
assert original_sha=='ff5a49b8706a8ccda08d8c3e82dad3e079f85dea98cb6bd40783d2262eec9838'
case,artifacts,workspace=load_workspace(source)
before=case.to_manifest()
legacy=types.ModuleType('resectionlab._legacy_save_encoding')
legacy.__package__='resectionlab';sys.modules[legacy.__name__]=legacy
legacy_source=subprocess.check_output(['git','show','a956f6c6:src/resectionlab/imaging.py'],cwd=ROOT,text=True)
exec(compile(legacy_source,'<pinned-a956f6c6-imaging>', 'exec'),legacy.__dict__)
current=imaging.save_case
settings=dict(sources=workspace['sources'],imaging=workspace['imagingState'],episode=workspace['episode'],
              selection=workspace['selection'],originals=workspace['originals'],artifacts=artifacts)
results={}
with tempfile.TemporaryDirectory(prefix='save-encoding-check-',dir=source.parents[2]/'review') as tmp:
    paths={name:Path(tmp)/(name+'.ressectionlab') for name in ('legacy','newCold','newWarm')}
    for name in paths:
        imaging.save_case=legacy.save_case if name=='legacy' else current
        started=time.perf_counter()
        save_workspace(case,paths[name],**settings)
        results[name]={'wallSeconds':time.perf_counter()-started,'bytes':paths[name].stat().st_size}
    imaging.save_case=current
    with ZipFile(paths['legacy']) as baseline:
        primary_arrays=baseline.read('arrays.npz')
        baseline_doc=json.loads(baseline.read('workspace.json'))
        aux_arrays={row['member']:None for row in baseline_doc['images']}
        from io import BytesIO
        for member in aux_arrays:
            with ZipFile(BytesIO(baseline.read(member))) as nested:
                aux_arrays[member]=nested.read('arrays.npz')
        for name in ('newCold','newWarm'):
            with ZipFile(paths[name]) as encoded:
                assert encoded.read('arrays.npz')==primary_arrays
                for member,expected in aux_arrays.items():
                    with ZipFile(BytesIO(encoded.read(member))) as nested:
                        assert nested.read('arrays.npz')==expected
            reopened,decoded,saved=load_workspace(paths[name])
            assert reopened.to_manifest()==before and decoded==artifacts
            assert saved['imagingState']==workspace['imagingState']
            assert saved['originals']==workspace['originals']
            assert saved['selection']==workspace['selection'] and saved['episode']==workspace['episode']
            assert [s.manifest(reopened) for s in saved['sources']]==[s.manifest(case) for s in workspace['sources']]
assert case.to_manifest()==before
assert hashlib.sha256(source.read_bytes()).hexdigest()==source_sha
assert hashlib.sha256(original.read_bytes()).hexdigest()==original_sha
report={'scope':'Saved-source encoding comparison only; no patient engine construction/replay and no whole-action speed claim.',
 'sourceCheckpoint':str(source),'sourceCheckpointSha256':source_sha,'originalSha256':original_sha,
 'legacyCommit':'a956f6c6','actions':len(artifacts['surgical_toolbox']['branches']['full-primary']['actions']),
 'measurements':results,'primaryAndAuxiliaryNpzBytesEqual':True,'allDecodedArtifactsAndWorkspaceStateEqual':True,
 'caseManifestAndOriginalFilesEqual':True,'inputsUnchanged':True,'cachedPayloads':len(case_array_payload._entries),
 'cachedPayloadBytes':sum(len(entry[1]) for entry in case_array_payload._entries.values())}
Path(__file__).with_name('save-array-reuse-source-equivalence.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report))

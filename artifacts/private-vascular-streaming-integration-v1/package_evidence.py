"""Copy an explicit small saved-evidence allowlist; no scientific execution."""
from pathlib import Path
import hashlib
import json
import os
import stat
import sys

ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent
PREP=ROOT/'build/private-vascular-streaming-integration-preparation-v1'
REVIEW=ROOT/'build/private-vascular-streaming-integration-independent-v1'
DEST=ROOT/'build/private-vascular-streaming-integration-evidence-v1'

def guard(event,args):
    if event in ('subprocess.Popen','os.system','os.exec','socket.connect'):
        raise AssertionError('Saved-only evidence packaging')
    if event=='open' and isinstance(args[0],(str,bytes)):
        assert not os.fsdecode(args[0]).lower().endswith(('.nii','.nii.gz','.mat','.tar','.pt','.ckpt','.bin','.npz','.npy','.pkl','.safetensors','.dcm','.h5'))
sys.addaudithook(guard)

def read(path):
    mode=path.lstat().st_mode
    assert stat.S_ISREG(mode) and not path.is_symlink(), path
    assert path.stat().st_size<100_000, path
    with path.open('rb') as stream:data=stream.read(100_001)
    assert len(data)<100_000, path
    return data

def sha(data):return hashlib.sha256(data).hexdigest()
def write(path,data):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('xb') as stream:stream.write(data)

def jsonbytes(value):return (json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n').encode()

assert sha(read(REVIEW/'REPORT.md'))=='bcf3b2f2a2ee0efeae0f752a3ca130916539d4f44738cd184e0d62b596b82a0a'
assert sha(read(PREP/'source.patch'))=='688bd80641fe0584dfc0dd4726aabe6856d51c9f68c1c39674fa07a3a542c101'
pins=json.loads(read(PREP/'candidate-pins.json'))
tracked={}
for relative in ('src/resectionlab/vascular_contact_streaming.py','src/resectionlab/private_vascular_evaluation.py',
                 'tests/test_vascular_contact_streaming.py','tests/test_private_vascular_streaming.py'):
    data=read(ROOT/relative)
    assert sha(data)==pins['files']['stage/'+relative]['sha256'],relative
    tracked[relative]={'sha256':sha(data),'bytes':len(data),'copied':False}
root_log=read(PREP/'root-integration-test.txt')
assert root_log.rstrip().endswith(b'74 passed in 14.37s')
files=[]
for name in ('RESULT.txt','candidate-pins.json','source-delta.json','source.patch','compare_legacy.py',
             'legacy-report-parity.json','run_controls.py','root-integration-test.txt'):
    files.append((PREP/name,Path('author')/name))
for name in ('REPORT.md','test_independent-initial.py','test_independent.py','run_review.py','review_worker.py',
             'run_corrected_controls.py','run-receipt.json','pytest-output.txt','source-before.json','source-after.json',
             'corrected-run-receipt.json','corrected-pytest-output.txt','corrected-source-before.json','corrected-source-after.json'):
    files.append((REVIEW/name,Path('independent')/name))
files.append((Path(__file__),Path('package_evidence.py')))
assert not DEST.exists(), 'Never overwrite a frozen package'
DEST.mkdir()
entries=[]
for source,target in files:
    data=read(source);destination=DEST/target
    write(destination,data)
    assert read(destination)==data==read(source)
    entries.append({'path':str(target),'source':str(source.relative_to(ROOT)),
                    'sha256':sha(data),'bytes':len(data),'byte_compared_to_source':True})
summary={
    'schema':'private-vascular-streaming-integration-evidence-v1',
    'status':'reviewed_candidate_integrated_root_tests_passed',
    'root_terminal':{'source':'root message: session58296 terminal exit0', 'session_id':58296,
                     'exit_code':0,'canonical_tests_passed':74,'pytest_seconds':14.37,
                     'output':'author/root-integration-test.txt','output_sha256':sha(root_log)},
    'review':{'decision':'GO','report':'independent/REPORT.md','initial_passed':87,'initial_fixture_failures':6,
              'corrected_cases_passed':6,'corrected_cases_deselected':13,'corrected_pytest_seconds':.59,
              'one_combined_93_pass_run_claimed':False,'candidate_changed_for_fixture_correction':False},
    'author_whole_report_parity':{'case_count':8,'saved_evidence':'author/legacy-report-parity.json',
        'compared_except':['elapsed_seconds','contact_work','contact_budget','tile_pruning_padding_mm'],
        'independently_inspected_not_rerun':True},
    'boundaries':{'existing_generated_only_admission_unchanged':True,'private_reference_voxel_cap':32**3,
        'planning_voxel_cap':16**3,'microstep_cap':256,'shared_matched_signature_unchanged':True,
        'congruent_removed_overlap_unchanged':True,'patient_payload_access':False,'full_profile_rerun':False},
    'tracked_source_references_not_copied':tracked,
    'scope':'Saved evidence only; package creation performs no tests or scientific, model, patient or network execution.'}
for path,value in [('summary.json',summary)]:
    data=jsonbytes(value);write(DEST/path,data)
    entries.append({'path':path,'source':'generated_from_saved_evidence_and_root_terminal_message',
                    'sha256':sha(data),'bytes':len(data)})
index={'schema':'explicit-small-integration-evidence-index-v1','files':entries,'file_count':len(entries),
       'total_bytes':sum(row['bytes'] for row in entries),'tracked_source_references_not_copied':tracked,
       'excluded':['staged/canonical source duplicates','generated arrays','temporary case files','bulk runtime/source archives'],
       'root_owns_tracked_integration_and_commit':True}
index_bytes=jsonbytes(index);write(DEST/'source-index.json',index_bytes)
for source,target in files:assert sha(read(source))==next(row['sha256'] for row in entries if row['path']==str(target))
for relative,row in tracked.items():assert sha(read(ROOT/relative))==row['sha256']
assert sum(path.stat().st_size for path in DEST.rglob('*') if path.is_file())==index['total_bytes']+len(index_bytes)
print(json.dumps({'destination':str(DEST.relative_to(ROOT)),'index_sha256':sha(index_bytes),
                  'indexed_files':len(entries),'indexed_bytes':index['total_bytes'],'index_bytes':len(index_bytes)}))

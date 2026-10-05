"""Saved receipts/source/blob digests only; no images, mesh decoding or native run."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import stat
import subprocess

ROOT = Path(__file__).resolve().parents[3]
OWNER = ROOT / 'artifacts/mechanics/resect-case4-patient-mesh-v1'
OUT = Path(__file__).resolve().parent
EXPECTED_INDEX = 'd9d50451a40070edf4d8317977ff2fa781f68aaad6db78024922cadad471dfa3'


def sha(path):
    with Path(path).open('rb') as stream: return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path): return json.loads(Path(path).read_text())


def main():
    assert sha(OWNER/'artifact-index.json') == EXPECTED_INDEX
    artifact_index = read(OWNER/'artifact-index.json')
    for name, record in artifact_index['files'].items():
        path = OWNER/name
        assert sha(path) == record['sha256'] and path.stat().st_size == record['bytes'], name
    assert len(artifact_index['files']) >= 11
    release=read(OWNER/'release.json'); baseline=read(OWNER/'execution-baseline.json')
    after=read(OWNER/'post-execution-verification.json'); summary=read(OWNER/'summary.json')
    worker=read(OWNER/'saved-records/worker.json'); acceptance=read(OWNER/'saved-records/acceptance.json')
    supervisor=read(OWNER/'saved-records/supervision/supervision.json')
    source=Path(release['source_directory']); commit=release['source_commit']
    assert commit == 'a49e70ac2a4df747e75fc7bec57c3a887898a6b5'
    assert source != ROOT and len(release['source_sha256']) == 4
    assert len([p for p in source.rglob('*') if p.is_file()]) == 4
    for name, expected in release['source_sha256'].items():
        raw=subprocess.check_output(['git','-C',str(ROOT),'show',f'{commit}:{name}'])
        assert hashlib.sha256(raw).hexdigest() == expected == sha(source/name) == sha(ROOT/name)
        assert stat.S_IMODE((source/name).stat().st_mode) == 0o444
    assert all(stat.S_IMODE(p.stat().st_mode) == 0o555 for p in [source,*[p for p in source.rglob('*') if p.is_dir()]])
    protocol=read(source/'manifests/experiments/resect-case4-patient-mesh-v1.json')
    bindings=baseline['file_sha256']; assert len(bindings)==19 and baseline['all_inputs_match'] and not baseline['attempt_existed']
    assert set(bindings)==set(after['original_archive_context_bindings'])
    excluded=protocol['mask']['path']; rehashed=0
    for name, expected in bindings.items():
        row=after['original_archive_context_bindings'][name]
        assert row['unchanged'] and expected==row['expected_sha256']==row['actual_sha256']
        if name != excluded:
            assert sha(name)==expected; rehashed+=1
    assert rehashed==18
    assert bindings[excluded]=='7902cfbcb5fad15144181c06883bac7f4800e842f7ff05a540f56bac4eb38759'
    assert all(worker['inputs_after'].values()) and len(worker['inputs_after'])==12
    assert worker['input_hashes']=={name:bindings[name] for name in worker['input_hashes']}
    assert len(acceptance['inputs_after'])==12 and all(acceptance['inputs_after'].values())
    assert after['worker_all12_input_checks_unchanged'] and after['launcher_all12_input_checks_unchanged']
    assert release['authorized'] and not release['clinical_validation'] and not release['anatomical_registration_accepted']
    assert supervisor==acceptance['supervision']
    assert supervisor['wall_cap_seconds']==180 and supervisor['rss_cap_bytes']==3*2**30
    assert supervisor['elapsed_seconds']<180 and supervisor['sampled_peak_process_group_rss_bytes']<3*2**30
    assert supervisor['exit_code']==1 and supervisor['kill_reason'] is None and supervisor['no_retry']
    assert supervisor['cleanup_error'] is None and supervisor['error'] is None
    assert supervisor['cwd']==str(source) and supervisor['command'][2]==str(source/'scripts/mechanics_patient_mesh.py')
    assert acceptance['thread_environment']=={'OMP_NUM_THREADS':'1','OMP_DYNAMIC':'FALSE','VECLIB_MAXIMUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1','VTK_SMP_MAX_THREADS':'1','VTK_SMP_IMPLEMENTATION_TYPE':'Sequential'}
    assert [r['status'] for r in worker['levels']]==['failed','not_executed','not_executed']
    row=worker['levels'][0]; cap=protocol['levels'][0]
    assert row['returned_nodes']==2065>cap['maximum_nodes']==2000
    assert row['returned_elements']==1052<cap['maximum_elements']==2000
    assert len(set(row['origin']['gmsh_node_ids']))==2065 and len(set(row['origin']['gmsh_element_ids']))==1052
    assert not {'quality','source_to_mesh','mesh_to_source','relative_volume_error'} & row.keys()
    assert worker['meshing_attempts']==worker['native_generation_calls']==1 and worker['solver_calls']==0
    assert worker['error']=={'type':'ValueError','message':'Returned mesh exceeds declared count cap; counts retained'}
    assert worker['status']==acceptance['status']=='failed_or_incomplete'
    raw_index=read(OWNER/'raw-output-index.json'); raw_directory=Path(raw_index['raw_directory'])
    assert len(raw_index['files'])==5 and raw_index['returned_overcap_volume_mesh_persisted'] is False
    assert set(p.relative_to(raw_directory).as_posix() for p in raw_directory.rglob('*') if p.is_file())==set(raw_index['files'])
    for name, record in raw_index['files'].items():
        path=raw_directory/name
        assert sha(path)==record['sha256'] and path.stat().st_size==record['bytes']
    assert worker['output_sha256']=={'native-surface.npz':raw_index['files']['native-surface.npz']['sha256']}
    native=worker['native_surface']
    assert (native['vertices'],native['triangles'],native['components'],native['euler_characteristic'],native['genus'])==(91951,183902,1,0,1)
    assert summary['native_extracted_surface']==native and summary['status']=='failed_fixed_coarse_node_count_cap'
    assert summary['preserved_raw_volume_mesh'] is False and summary['retries']==0
    prior=read(ROOT/'artifacts/mechanics-patient-mesh-independent-review-v1/receipt.json')['native_control']
    assert prior['overall_fidelity_status']=='failed_fixed_volume_gate' and prior['relative_volume_error']==.08499022677808099 and prior['retries']==0
    for n in [2000,2065]:
        dofs=3*n; assert 8*dofs*(dofs+1)//2>0
    receipt={'status':'saved_negative_result_verified','completed_at':datetime.now(timezone.utc).isoformat(),
      'auditor_sha256':sha(__file__),'owner_artifact_index_sha256':EXPECTED_INDEX,
      'owner_summary_sha256':sha(OWNER/'summary.json'),'owner_raw_index_sha256':sha(OWNER/'raw-output-index.json'),
      'source_commit':commit,'four_archived_and_working_sources_match_git':True,'archive_read_only':True,
      'owner_artifacts_verified':len(artifact_index['files']),'raw_files_blob_hash_verified_without_decoding':5,
      'before_after_bindings_consistent':19,'worker_and_launcher_input_binding_count':12,'nonimage_bindings_independently_rehashed':18,
      'mask_image_reopened_or_rehashed_by_this_audit':False,
      'mask_integrity_evidence':'Saved before/after owner digests and worker/launcher checks, cross-bound to frozen protocol.',
      'native_generation_calls':1,'solver_calls':0,'reruns':0,'image_or_B_or_V_or_motion_reads':0,
      'coarse':{'nodes':2065,'node_cap':2000,'excess_nodes':65,'relative_excess':.0325,'volume_elements':1052,'element_cap':2000},
      'later_levels':'medium/fine not executed','returned_volume_mesh_saved':False,'mesh_quality_and_fidelity':'not executed; no inference of success or failure',
      'native_source_surface_metadata':{'vertices':91951,'triangles':183902,'components':1,'euler':0,'genus':1},
      'resource_observation':{'seconds':supervisor['elapsed_seconds'],'sampled_group_peak_rss_bytes':supervisor['sampled_peak_process_group_rss_bytes'],'wall_cap_seconds':180,'rss_cap_bytes':3*2**30,'resource_cap_fired':False,'one_numerical_thread':True,'isolated_benchmark':False},
      'prior_cube_volume_fidelity_failure_preserved':True,'anatomy_or_registration_approved':False,
      'next_dependency':'Prospective count/memory admission and bounded rejected-geometry retention policy, before any separately authorized new feasibility attempt; current discarded volume geometry cannot be recovered from IDs.',
      'storage_arithmetic_only':{'2065_node_unreduced_dofs':6195,'one_dense_symmetric_value_array_bytes':153536880,'three_value_arrays_bytes':460610640,'2000_node_three_value_arrays_bytes':432072000,'additional_three_value_array_bytes':28538640,'not_full_solver_memory_estimate':True},
      'not_authorized':['raise caps','retry meshing','alter mask/topology or fidelity thresholds','solver','B/V or motion access']}
    target=OUT/'verification.json'; assert not target.exists();target.write_text(json.dumps(receipt,indent=2,sort_keys=True)+'\n')
    print(json.dumps({'status':receipt['status'],'verification_sha256':sha(target)}))


if __name__=='__main__':main()

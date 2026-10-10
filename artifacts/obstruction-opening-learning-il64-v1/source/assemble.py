"""Package only terminal IL64 JSON/source evidence; never load arrays or weights."""
from pathlib import Path
import hashlib,json,shutil
ROOT=Path(__file__).resolve().parents[4]
BASE=ROOT/'build/obstruction-opening-learning-v1';RUN=BASE/'IL64';OUT=RUN/'attempt-01'
AUDIT=ROOT/'build/obstruction-opening-learning-independent-v1'
PACKAGE=Path(__file__).resolve().parent/'artifacts/obstruction-opening-learning-il64-v1'
TRAIN=('ReMIND-008','ReMIND-010','ReMIND-020','ReMIND-025')
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def read(path):
    assert path.is_file() and not path.is_symlink() and path.stat().st_size<=2*1024**2
    return json.loads(path.read_text())
def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x') as stream:json.dump(value,stream,indent=2,sort_keys=True,allow_nan=False);stream.write('\n')
assert not PACKAGE.exists();PACKAGE.mkdir(parents=True)
result=read(OUT/'result.json');parent=read(RUN/'attempt-01.supervision/receipt.json')
audit=read(AUDIT/'IL64/audit-result.json');release=read(RUN/'root-release.json')
assert sha(AUDIT/'IL64/audit-result.json')=='b7ad176a75040bd071f697eea890b8819f343a354b5c9f1e973b8631a6d3ac28'
assert result['status']=='complete_matched_TRAIN_endpoint' and parent['status']=='complete'
assert sha(OUT/'result.json')==parent['result_sha256']==audit['result_sha256']
assert sha(RUN/'root-release.json')==parent['release_sha256']==audit['release_sha256']
assert not parent['cleanup_errors'] and not parent['final_owned_pids'] and parent['worker_termination_confirmed']
entries=[]
def copy(source,target):
    assert source.is_file() and not source.is_symlink() and source.suffix not in ('.psckpt','.npz','.npy')
    assert source.stat().st_size<=2*1024**2
    raw=source.read_bytes();dest=PACKAGE/target;dest.parent.mkdir(parents=True,exist_ok=True)
    with dest.open('xb') as stream:stream.write(raw)
    assert dest.read_bytes()==raw
    entries.append({'path':target,'source_path':str(source.relative_to(ROOT)),'bytes':len(raw),'sha256':sha(dest),'kind':'exact_copy'})
for name in ('pilot_contract.py','cohort_worker.py','run_owned.py','freeze-runtime.py','test_generated_worker_flow.py'):
    copy(BASE/name,'source/'+name)
for name in ('receipt.json','console.txt'):copy(BASE/'generated-attempt-01'/name,'generated-control/'+name)
copy(BASE/'PREPARATION.json','source/PREPARATION.json')
for name in ('root-release.json','release-template.json','source-index.json'):copy(RUN/name,'provenance/'+name)
for name in ('receipt.json','worker-final.json','endpoint-control.json','declaration.json'):
    copy(RUN/'attempt-01.supervision'/name,'supervision/'+name)
for name in ('result.json','configuration.json','costs.json','training-dynamics.json','checkpoint-reload.json','teacher-cache.json','teacher-pins.json','endpoint-teacher-metrics.json'):
    copy(OUT/name,'result/'+name)
for subject in TRAIN:
    copy(OUT/(subject+'-context.json'),'contexts/'+subject+'.json')
    for group in ('teachers','TRAIN-greedy/IL'):
        for name in ('plan.json','native-replay.json'):
            copy(OUT/group/subject/name,group+'/'+subject+'/'+name)
    for step in range(4 if subject=='ReMIND-025' else 1):
        name=f'state-{step:02d}.json';copy(OUT/'teacher-readout'/subject/name,'teacher-readout/'+subject+'/'+name)
copy(AUDIT/'audit_saved.py','audit/audit_saved.py')
for name in ('audit-result.json','input-hashes.json'):copy(AUDIT/'IL64'/name,'audit/'+name)
copy(Path(__file__),'source/assemble.py')
checkpoint={'scope':'metadata only; checkpoint payload remains in ignored local output and is deliberately omitted',
    'source_result_sha256':sha(OUT/'result.json'),'source_path':str((OUT/result['checkpoints']['IL']['path']).relative_to(ROOT)),
    **result['checkpoints']['IL'],'initial_parameter_hash':result['initial_parameter_hash'],
    'learning_protocol_hash':release['learning_protocol_hash'],'completed_updates':64,'reload_record':'result/checkpoint-reload.json'}
write(PACKAGE/'checkpoint-metadata.json',checkpoint)
dynamics=read(OUT/'training-dynamics.json')['updates']
summary={'version':'obstruction-opening-learning-il64-v1','scope':'fixed-four TRAIN imitation endpoint under explicit S OR T occupancy assumption',
    'status':result['status'],'TRAIN':list(TRAIN),'SELECT_EVAL_executed':False,'RL_updates_in_this_run':0,
    'source_head':release['expected_head'],'release_sha256':sha(RUN/'root-release.json'),'result_sha256':sha(OUT/'result.json'),
    'audit_sha256':sha(AUDIT/'IL64/audit-result.json'),'checkpoint':checkpoint,
    'counts':{**audit['counts'],'greedy_forwards':sum(r['steps'] for r in result['TRAIN_greedy'].values()),'teacher_states':7},
    'greedy_routes':audit['greedy_routes'],'endpoint_teacher_metrics':audit['endpoint_metrics_recomputed'],
    'teacher_top1':{'correct':6,'total':7,'route_outcome_equivalence_checked_separately':True},
    'runtime':{'parent_seconds':parent['elapsed_seconds'],'worker_seconds':result['complete_wall_seconds'],
        'sampled_peak_rss_bytes':parent['sampled_peak_rss_bytes'],'owned_output_bytes':parent['output_bytes'],
        'cleanup_errors':parent['cleanup_errors'],'final_owned_pids':parent['final_owned_pids'],'caps':parent['caps']},
    'dynamics':{'updates':len(dynamics),'first_pre_update_loss':dynamics[0]['pre_update_loss'],
        'final_pre_update_loss':dynamics[-1]['pre_update_loss'],'first_gradient_norm':dynamics[0]['gradient_norm_before_clip'],
        'final_gradient_norm':dynamics[-1]['gradient_norm_before_clip'],
        'updates_exceeding_clip_threshold':sum(r['gradient_exceeds_clip_threshold'] for r in dynamics),
        'trajectory_path':'result/training-dynamics.json','trajectory_sha256':sha(OUT/'training-dynamics.json'),
        'endpoint_CE_is_after_update64':True},
    'independent_audit':{'checks':audit['checks'],'input_count':audit['input_count'],
        'source_files':audit['source_files'],'metadata_files':audit['metadata_files'],'limitations':audit['limits']},
    'omissions':['checkpoint payload','patient arrays/images','per-update gradient-contribution JSON bulk','redundant state inventories and repeated source QC'],
    'omitted_evidence_bindings':'audit/input-hashes.json plus all64 before/after parameters and update receipt SHAs in result/training-dynamics.json',
    'interpretation':'Training-set fit on the amended action/occupancy model; no held-out, anatomical/material, clinical, or equal-compute claim'}
write(PACKAGE/'summary.json',summary)
readme='''# Cached balanced IL64 on the union-occupancy obstruction-opening world

The reloaded fixed64 IL policy matches the search outcome on all four fixed TRAIN cases. ReMIND-025 swaps the first two equal-reward motions, then uses the same third physical motion and stops. It removes the same three source cells with the same unordered motion geometry and total path length (12.6953134245 mm): return +2.7583290722, supplied-target volume 2.8610243857 mm³, and 0 mm³ outside the supplied target. ReMIND-008, 010 and 020 stop immediately. This is a small partial-target result, not complete target removal.

Teacher-action top-1 agreement is 6/7. The single different first action on025 does not worsen the complete route outcome. Candidate IDs depend on the current state; the independent comparison uses recorded physical motion geometry, removed cells, path cost and reward. The supplied-target removal fraction for025 is about0.003669%.

This endpoint uses the explicit S OR (T > 0) occupancy/material assumption, h24/cap120, footprint and obstruction-opening proposals, global public target context, seed20261010, and the existing model/optimizer. Its seven replayed search-teacher states contain four STOP and three motion labels. Balanced CE assigns total weight0.5 to each group. All64 shared IL updates ran; no RL or new search ran in this endpoint. The unchanged architecture and common initialization are source-bound; this package makes no comparison across changed worlds.

Accounting is complete:448 loss forwards +7 teacher readouts +7 greedy forwards =462;256 cache reuses;8 sequential source visits;3,224 native previews;one authenticated checkpoint reload. Parent elapsed161.917943 seconds (worker158.160842), sampled peak RSS1,980,448,768 bytes, and1,616,573 output bytes; cleanup had no errors or survivors. Final post-update balanced CE is0.1686177275 (unweighted CE0.1456211400). The last training loss0.1759536648 is measured before update64; it is not the endpoint CE. The full64-row loss/gradient/parameter-hash trajectory is retained.

The independent saved-only audit passed5,932 checks across473 input bindings, including85 source files and20 metadata files. It checked complete teacher and learned plans/replays, cache/reload bindings, scalar probabilities and trajectory accounting. It did not deserialize weights, re-run a model or repeat native replay; recorded tensor identities are not independent tensor recomputation.

`MANIFEST.json` maps every copied file to its original local path and exact hash. Full endpoint plans/native replay JSON, teacher plans/replays, seven readouts, source/test copies, release/source closure, costs/dynamics, checkpoint metadata, and the compact independent audit are included. Checkpoint payloads, patient arrays, repeated inventories and per-update bulk are omitted; their saved audit bindings and update hashes remain available. `checkpoint-metadata.json` locates the ignored checkpoint without including it.

Copied scripts/tests are archival and preserve their original relative-path assumptions. Do not execute them from this package; reproduction requires the manifest's original source layout and a separate owned release. This package does not admit SELECT/EVAL or the newer partial-domain cohort. These are TRAIN numerical results under an unvalidated occupancy assumption, with no physical, clinical, generalization or equal-compute claim.
'''
(PACKAGE/'README.md').write_text(readme)
for name in ('checkpoint-metadata.json','summary.json','README.md'):
    p=PACKAGE/name;entries.append({'path':name,'source_path':None,'bytes':p.stat().st_size,'sha256':sha(p),'kind':'derived_from_packaged_evidence'})
manifest={'version':'compact-result-manifest-v1','destination':'artifacts/obstruction-opening-learning-il64-v1',
    'files':sorted(entries,key=lambda r:r['path']),'file_count':len(entries),'total_bytes':sum(r['bytes'] for r in entries),
    'payloads_excluded':True,'copy_verification':'exact raw bytes, length and SHA256 of every original source/JSON copy; no model/array reads'}
write(PACKAGE/'MANIFEST.json',manifest)
for row in entries:
    p=PACKAGE/row['path'];assert sha(p)==row['sha256'] and p.stat().st_size==row['bytes']
    if row['source_path']:assert p.read_bytes()==(ROOT/row['source_path']).read_bytes()
check={'status':'PASS_exact_saved_package_parity','manifest_sha256':sha(PACKAGE/'MANIFEST.json'),'file_count':len(entries),
    'total_bytes_excluding_manifest':manifest['total_bytes'],'total_bytes_including_manifest':manifest['total_bytes']+(PACKAGE/'MANIFEST.json').stat().st_size,
    'checkpoint_or_patient_payloads_read':False,'destination':str(PACKAGE.relative_to(ROOT))}
write(Path(__file__).resolve().parent/'package-check.json',check);print(json.dumps(check,indent=2))

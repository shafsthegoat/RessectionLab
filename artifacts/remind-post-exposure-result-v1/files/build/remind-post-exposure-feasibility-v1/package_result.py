"""Saved JSON/source packaging only; no arrays, geometry or model execution."""
from pathlib import Path
import json,hashlib,shutil
ROOT=Path(__file__).resolve().parents[2];BASE=Path(__file__).resolve().parent
DEST=ROOT/'build/remind-post-exposure-result-integration-v1'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,v):
 p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('x') as f:json.dump(v,f,indent=2,allow_nan=False);f.write('\n')
resultpath=BASE/'attempt-01/result.json';assert sha(resultpath)=='7a57fd74f58b15d6975be4459fe4f46e6396dfa9e2f3c0fec0ef68cf5e550358'
r=json.loads(resultpath.read_bytes());receipt=json.loads((BASE/'attempt-01.supervision/receipt.json').read_bytes())
assert r['status']=='post_exposure_batch_complete' and receipt['status']=='complete' and receipt['remaining_owned_pids']==[]
index=json.loads((BASE/'source-index.json').read_bytes());paths=set()
for name,digest in index['files'].items():
 p=ROOT/name;assert sha(p)==digest;assert p.suffix in {'.py','.json','.txt','.log'};paths.add(p)
paths.update((BASE/'attempt-01').rglob('*.json'))
paths.update(BASE/n for n in ('source-index.json','prepared-release.json','root-release.json','check-only.log','package_result.py'))
paths.update(BASE/'attempt-01.supervision'/n for n in ('receipt.json','declaration.json','worker.log'))
rows=[]
for x in r['cases']:
 p=BASE/'attempt-01'/x['patient_id'];m=json.loads((p/'episode-metrics.json').read_bytes());e=json.loads((p/'independent-episode.json').read_bytes())
 assert e['accepted'] and e['complete_episode'] and len(m['history'])==len(x['actions'])==x['steps_taken']
 assert e['outcomes']==x['full_route_outcomes']
 rows.append({'patient_id':x['patient_id'],'role':'TRAIN','initial_emitted':x['emitted_count'],'initial_legal':x['accepted_count'],
 'initial_nominal_score_signs':x['initial_nominal_score_signs'],'K_cells':x['post_exposure_start']['seed_cells'],
 'F0_cells':x['post_exposure_start']['initial_free_cells'],'initial_removed_cells':x['initial_removed_cells'],
 'initial_contact_cells':x['initial_contact_cells'],'source_support_knownness_and_assumption':x['source_and_simulated_domains'],
 'full_target_native_voxels':x['target_denominator'],'action_count_including_STOP':len(x['actions']),
 'actions':x['actions'],'outcomes':x['full_route_outcomes'],'independent_accepted':e['accepted'],
 'committed_history_hash':e['committed_history_hash'],'native_preview_entries':x['native_preview_entries'],
 'greedy_seconds':x['greedy_accounting']['planning_seconds'],'independent_seconds':x['independent_episode_seconds'],
 'geometry_unknowns':x['full_route_geometry_unknowns'],'source_hash':x['source_hash'],'context_hash':x['context_hash']})
summary={'status':r['status'],'original_result':{'path':str(resultpath.relative_to(ROOT)),'sha256':sha(resultpath)},
 'condition':r['post_exposure_condition'],'occupancy_condition':r['occupancy_condition'],'cases':rows,
 'planned_cases':4,'completed_cases':4,'positive_routes':2,'STOP_only_routes':2,'teacher_state_count_if_reconstructed':29,
 'motion_decisions':25,'STOP_decisions':4,'native_previews':r['native_budget']['native_preview_entries'],
 'greedy_calls':r['greedy_calls'],'planning_clone_calls':r['planning_clone_calls'],'planning_transition_calls':r['planning_transition_calls'],
 'authoritative_transition_calls':r['transition_calls'],'parent_seconds':receipt['elapsed_seconds'],'sampled_peak_RSS_bytes':receipt['sampled_peak_rss_bytes'],
 'worker_termination_confirmed':receipt['worker_termination_confirmed'],'remaining_owned_pids':receipt['remaining_owned_pids'],
 'model_calls':0,'optimizer_calls':0,'source_arrays_written':0,'training_admitted':False,
 'interpretation':'Complete simulated axial routes with independent numerical replay; no anatomical/material/clinical validation. E and outside-image extent and transfer between withdrawn poses remain unassessed. Source-known Ds, full T and inward U restrictions preserved.',
 'prospective_learning':'All four fixed TRAIN cases retained; reconstruction and trace admission required before optimizer use. No training or SELECT/EVAL extension authorized by completion.'}
DEST.mkdir(exist_ok=False);save(DEST/'summary.json',summary)
records=[]
for p in sorted(paths):
 assert p.is_file() and not p.is_symlink();rel=p.relative_to(ROOT);out=DEST/'files'/rel;out.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,out)
 records.append({'source_path':str(rel),'package_path':str(out.relative_to(DEST)),'sha256':sha(out),'bytes':out.stat().st_size})
report='''Four fixed TRAIN post-exposure routes completed, 2026-10-10.

Patient   Initial legal/emitted   Greedy route      Removed T cells / mm3   Outside-T removed
002       21/117                 STOP              0 / 0                   0
015       39/78                  13 moves + STOP   58 / 55.313110          0
018       17/75                  STOP              0 / 0                   0
045       41/91                  12 moves + STOP   51 / 48.637450          0

All four completed independent episode replay. All legal initial scores were negative for002/018 and positive for015/045; STOP outcomes preserve the fixed denominator. K=93 for every case; F0=1,550,811/1,646,557/1,542,744/1,459,043 known-zero cells. No initialized removal or contact. The measured totals are4,823 native previews,29 planning transitions,29 authoritative transitions, zero model/optimizer calls. Parent62.416s, sampled peak702,431,232B, clean reap.

The two positive routes remove0.74887% and0.20157% of the complete supplied target. Their retained-contact upper bounds are150.680542 and153.541755mm3. Zero outside-target removal is not zero tissue contact, an anatomical normal-tissue claim or clinical safety. Normal-tissue exposure remains unassessed.

This is a separately declared post-exposure simulation: rigid removable S OR T, source-known Ds distinct from D=Ds OR T, and no inward/straddling unknown clearance. The proximal external workspace E is an explicit unvalidated working-space assumption, not acquired air or evidence of surgery. Each insertion and reverse is checked/charged; transfer between withdrawn poses is unassessed/unpriced except the existing tool-switch penalty. Outside-image geometry remains unassessed. Source masks and full T were unchanged. NN sampling, automatic cerebrum ancestry, unavailable exact timing, unknown pretrained/cross-source overlap and no expert anatomical validation persist.

Exact case source bindings, initial inventories and scores, complete histories/observations, greedy costs, authoritative episode metrics and independent replays are preserved under files/. Source and runtime releases/controls are included. No image arrays, DICOM, weights or original bulk payloads are packaged. sha256-manifest.json maps each exact copy to its original workspace path. This report is saved-output interpretation only, not a rerun or training admission.
'''
(DEST/'REPORT.txt').write_text(report)
save(DEST/'sha256-manifest.json',{'files':records,'files_count':len(records),'total_bytes':sum(x['bytes'] for x in records),
 'derived_summary_sha256':sha(DEST/'summary.json'),'derived_report_sha256':sha(DEST/'REPORT.txt'),
 'source_reference_policy':'Restore copies to their source_path under the same repository before reproduction; source index retains original ignored dependencies. Arrays remain outside this package and require original frozen public manifests.'})
print(json.dumps({'package':str(DEST.relative_to(ROOT)),'manifest_sha256':sha(DEST/'sha256-manifest.json'),'files':len(records),'bytes':sum(x['bytes'] for x in records),'summary_sha256':sha(DEST/'summary.json'),'report_sha256':sha(DEST/'REPORT.txt')}))

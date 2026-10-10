"""Copy only compact saved evidence; no replay arrays, weights or runtime calls."""
from pathlib import Path
import hashlib,json,shutil
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent
PREP=ROOT/'build/goal-conditioned-policy-v1/train-refit-integration-v1';AUDIT=ROOT/'build/train-refit-desktop-live-independent-v1';LIVE=ROOT/'build/train-refit-desktop-live-v1'
audit=json.loads((AUDIT/'audit.json').read_text());run=Path(audit['attempt_path']);copies={}
def cp(src,dst):
 target=OUT/dst;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(src,target);copies[dst]=str(src)
for name in ('REPORT.txt','audit.json','audit_saved.py','evidence-hashes.json'):cp(AUDIT/name,name)
for name in ('promotion-index.json','promotion.patch','source-closure.json','HANDOFF.txt','diagnostic-retention-rebase.patch','diagnostic-retention-rebase-receipt.json'):
 cp(PREP/name,'source/'+name)
cp(ROOT/'artifacts/public-contact-train-refit-v1/desktop-release.json','publication/desktop-release.json')
for name in ('root-index.json','root-python-tests.log','root-desktop-tests.log','root-index-tests.log','root-working-build.log','root-index-build.log'):cp(PREP/name,'validation/'+name)
for folder,label in [('train-refit-integration-independent-v1','source-go'),('train-refit-integration-rebase-independent-v1','rebase-go')]:
 for name in ('REPORT.txt','index.json'):cp(ROOT/'build'/folder/name,'reviews/'+label+'-'+name)
for name in ('launch.json','terminal.json'):cp(LIVE/name,'live/'+name)
for name in ('attempt.json','supervision.json','completion.json'):cp(run/name,'live/'+name)
ui={'evidence_kind':'root_reported_live_desktop_observation','auditor_reopened_UI':False,'checks':{'frame39_probe_insertion':True,'frame76_final_retained_goal_contact':True,'save_refused':True,'SELECT_refit_Execute_disabled':True},'old_IL_RL_rerun_in_this_session':False,'scope':'Root explicitly reported visual/workflow checks; no screenshot or final transport envelope added by auditor.'}
(OUT/'live/root-reported-ui-checks.json').write_text(json.dumps(ui,sort_keys=True,indent=2)+'\n')
ledger={'version':'TRAIN-refit-desktop-integration-status-v1','status':'COMPLETED_GENERATED_TRAIN_INTEGRATION','source_promotion_index':'905568afe3b7ff573de6c2f522149c8938874a8e1eed4ca94ce79af187afbb6f','exact_validated_tree':'27d8bf22171d87749bb3ef0093fe43bf396cef42','source_review':'GO_SOURCE_ONLY_then_root_canonical_controls','validation':audit['validation'],'live_result_sha256':audit['result_sha256'],'live_request':audit['request'],'complete_live_episode_equals_previous_TRAIN_10':True,'live_outcome':audit['final_outcome'],'whole_prior_TRAIN_outcome':{'tasks':24,'refit_contacts':6,'saved_SEARCH_contacts':16,'refit_STOP_only':18,'refit_mean_return':0.05633333333333332,'saved_SEARCH_mean_return':0.2561666666666667},'training_budget':{'updates':32,'states_per_update':40,'loss_forwards':1280,'fixed_readout_forwards':80,'original_pilot_loss_forwards':128},'inference_updates':0,'new_training_or_evaluation':False,'original_IL_RL_publication_unchanged':True,'old_IL_RL_live_rerun':False,'UI_checks_evidence':'root_reported_live_desktop_observation','unfinished_scientific_claims':['Unseen-layout/patient generalization is not established.','This refit is weaker than saved SEARCH on the fixed TRAIN corpus.','No relation-policy or STOP-head successor experiment is authorized by this integration receipt.'],'excluded_bulk':'Local 308132-byte live result and original 24 native replay envelopes stay in place; exact hashes are indexed. No weights, image arrays or compressed history archive copied.'}
(OUT/'STATUS.json').write_text(json.dumps(ledger,sort_keys=True,indent=2)+'\n')
(OUT/'copied-evidence.json').write_text(json.dumps(copies,sort_keys=True,indent=2)+'\n')
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
files=[{'path':str(p.relative_to(OUT)),'bytes':p.stat().st_size,'sha256':sha(p)} for p in sorted(OUT.rglob('*')) if p.is_file() and p.name!='index.json']
# Include the copied historical index files; only omit this top-level index.
files=[{'path':str(p.relative_to(OUT)),'bytes':p.stat().st_size,'sha256':sha(p)} for p in sorted(OUT.rglob('*')) if p.is_file() and p!=OUT/'index.json']
index={'status':'PASS_COMPACT_TRAIN_REFIT_DESKTOP_INTEGRATION_PACKAGE','files':files,'file_count':len(files),'total_indexed_bytes':sum(x['bytes'] for x in files),'external_live_result':{'path':str(run/'result.json'),'bytes':(run/'result.json').stat().st_size,'sha256':audit['result_sha256']},'no_bulk_replay_or_weights':True}
(OUT/'index.json').write_text(json.dumps(index,sort_keys=True,indent=2)+'\n')
print(json.dumps({'files':len(files),'bytes':index['total_indexed_bytes'],'index_sha256':sha(OUT/'index.json'),'report_sha256':sha(OUT/'REPORT.txt'),'status_sha256':sha(OUT/'STATUS.json')},indent=2))

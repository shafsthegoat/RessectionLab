"""Small metadata/stat-only snapshot; never opens an IXI archive."""
from pathlib import Path
from datetime import datetime,timezone
from collections import Counter
import hashlib,json,shutil,subprocess
R=Path.cwd();P=R/'build/ixi-paired-intake-preparation-v1';B=R/'data/acquisition/ixi-t1-mra-vessel-v1';sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
identity=json.loads((P/'live-monitor-identity.json').read_text());assert sha(B/'declaration.json')==identity['declaration_sha256'];assert sha(P/'run-intake.py')==identity['runner_sha256'];scope=json.loads((B/'source/scope.json').read_text());assert sha(B/'source/scope.json')==identity['scope_sha256']
pid=identity['pid'];now=datetime.now(timezone.utc)
try:
 command=subprocess.check_output(['ps','-p',str(pid),'-o','command='],text=True).strip();start=subprocess.check_output(['ps','-p',str(pid),'-o','lstart='],text=True).strip();rss=int(subprocess.check_output(['ps','-p',str(pid),'-o','rss='],text=True).strip());alive=True
except (subprocess.CalledProcessError,ValueError):command='';start='';rss=None;alive=False
results=[json.loads(p.read_text()) for p in B.glob('attempts/*/*/*-result.json')];by={}
for row in sorted(results,key=lambda r:r['finished_utc']):by[row['path']]=row
files=[]
for e in scope['files']:
 path=B/'verified'/e['filename'];partial=path.with_name(path.name+'.partial');row=by.get(e['filename'],{});size=path.stat().st_size if path.exists() else 0
 files.append({'filename':e['filename'],'expected_bytes':e['bytes'],'verified':row.get('status') in ['byte_verified','existing_byte_verified'] and size==e['bytes'],'bytes_at_final_path':size,'partial_bytes':partial.stat().st_size if partial.exists() else 0,'latest_result':row.get('status')})
terminal=B/'runs'/identity['run_id']/'completion.json';receipt={'schema':'ixi-live-status-v1','recorded_utc':now.isoformat(),'owner_agent':'/root/acquisition_continuity_audit','session_id':identity['session_id'],'pid':pid,'pgid':identity['pgid'],'process_alive':alive,'exact_process_identity_matches':alive and start==identity['ps_lstart_local'] and hashlib.sha256(command.encode()).hexdigest()==identity['ps_command_stripped_sha256'],'sampled_rss_kib':rss,'verified_files':sum(x['verified'] for x in files),'verified_bytes':sum(x['expected_bytes'] for x in files if x['verified']),'planned_files':3,'planned_bytes':scope['source_bytes'],'files':files,'historical_result_status_counts':dict(Counter(r['status'] for r in results)),'terminal_status':json.loads(terminal.read_text())['status'] if terminal.exists() else None,'terminal_receipt_sha256':sha(terminal) if terminal.exists() else None,'free_bytes':shutil.disk_usage(R).free,'all_payloads_unreviewed':True,'payload_bytes_opened_by_status_tool':0}
raw=json.dumps(receipt,indent=2,sort_keys=True)+'\n';out=P/'status-snapshots'/now.strftime('%Y%m%dT%H%M%S%fZ.json');out.parent.mkdir(exist_ok=True);out.write_text(raw);(P/'latest-status.json').write_text(raw);print(raw,end='')

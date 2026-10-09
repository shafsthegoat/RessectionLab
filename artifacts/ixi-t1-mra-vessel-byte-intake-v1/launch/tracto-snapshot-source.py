"""Read transfer receipts/process identity only; never read source payloads."""
from pathlib import Path
from datetime import datetime,timezone
from collections import Counter
import hashlib,json,os,shutil,subprocess
R=Path.cwd();P=R/'build/tractoinferno-train-preparation-v1';B=R/'data/acquisition/tractoinferno-train-v1';run='20261009T094645035821Z-90fc2160';identity=json.loads((P/'live-monitor-identity.json').read_text());now=datetime.now(timezone.utc);read=lambda field:subprocess.run(['ps','-p',str(identity['pid']),'-o',field+'='],capture_output=True,text=True).stdout.strip()
command=read('command');lstart=read('lstart');rss=read('rss');alive=bool(command);matching=alive and lstart==identity['ps_lstart'] and hashlib.sha256(command.encode()).hexdigest()==identity['ps_command_sha256']
records=[json.loads(p.read_text()) for p in (B/'attempts').rglob('*-result.json')];latest={}
for row in sorted(records,key=lambda x:x.get('finished_utc','')):latest[row['path']]=row
verified=[v for v in latest.values() if v['status'] in ['byte_verified','existing_byte_verified']]
partials=[{'path':str(p.relative_to(B/'verified')),'bytes_present':p.stat().st_size} for p in (B/'verified').rglob('*.partial')]
completion=B/'runs'/run/'completion.json';terminal=json.loads(completion.read_text()) if completion.exists() else None
result={'schema':'tractoinferno-live-status-v1','recorded_utc':now.isoformat(),'run_id':run,'owner_agent':'/root/acquisition_continuity_audit','session_id':35269,'pid':identity['pid'],'pgid':identity['pgid'],'process_alive':alive,'exact_process_identity_matches':matching,'sampled_rss_kib':int(rss) if rss else None,'verified_files':len(verified),'verified_bytes':sum(v['bytes'] for v in verified),'planned_files':7622,'planned_bytes':273788367039,'planned_TRAIN_source_people':198,'latest_status_counts':dict(Counter(v['status'] for v in latest.values())),'historical_result_status_counts':dict(Counter(v['status'] for v in records)),'partial_files':partials,'terminal_status':terminal['status'] if terminal else None,'terminal_receipt_sha256':hashlib.sha256(completion.read_bytes()).hexdigest() if terminal else None,'free_bytes':shutil.disk_usage(B).free,'all_payloads_unreviewed':True,'payload_bytes_opened_by_status_tool':0}
folder=P/'status-snapshots';folder.mkdir(exist_ok=True);payload=(json.dumps(result,indent=2)+'\n').encode();(folder/(now.strftime('%Y%m%dT%H%M%S%fZ')+'.json')).write_bytes(payload);temp=P/'.latest-status.partial';temp.write_bytes(payload);os.replace(temp,P/'latest-status.json');print(json.dumps(result,indent=2))

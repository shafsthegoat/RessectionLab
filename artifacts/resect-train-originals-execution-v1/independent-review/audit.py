"""Read-only independent acquired-byte audit. Writes only sibling review JSON."""
import hashlib
import json
import os
import stat
import subprocess
import time
from collections import Counter
from datetime import datetime, timezone
from email.parser import BytesParser
from pathlib import Path, PurePosixPath

ROOT=Path(__file__).resolve().parents[2]
OUT=Path(__file__).resolve().parent
CACHE=ROOT/'data/acquisition/resect-originals-v1'
EXEC=ROOT/'build/resect-train-originals-execution-v1'
START=time.monotonic(); DEADLINE=START+120
INDEX={}
PINS={
 'scripts/acquire_resect_train_originals.py':'1c04d548348654700ff4f57ea1aae5312d6269b829edeb15dd7c74c11a0fef50',
 'scripts/acquire_resect_cavity.py':'6ece649ba346c9e7a0e93af462a91a9bf863728bea3fba83f5eccd681e1d0d51',
 'scripts/real_intake_io.py':'496821360b1daee79459952f970d299856baacad75fcc57ee589e55115036fd8',
 'manifests/resect-train-originals-v1.json':'1764a0ce75b449d542ef51578c54cab6202b66d72c71b811999cb51a55164135',
}
def encode(value): return (json.dumps(value,indent=2,allow_nan=False)+'\n').encode()
def sha(raw):return hashlib.sha256(raw).hexdigest()
def require(ok,reason):
 if not ok: raise ValueError(reason)
def bounded(path,limit=2*1024**2,expected=None,keep=False):
 require(time.monotonic()<DEADLINE,'audit deadline')
 path=Path(path);require(path.is_relative_to(ROOT),'outside repository')
 require(not any(p.is_symlink() for p in [path,*path.parents] if p.is_relative_to(ROOT)),'symlink')
 h=hashlib.sha256();m=hashlib.md5();total=0;chunks=[]
 with os.fdopen(os.open(path,os.O_RDONLY|os.O_NOFOLLOW),'rb') as f:
  before=os.fstat(f.fileno())
  require(stat.S_ISREG(before.st_mode) and before.st_size<=limit,'type or size bound')
  if expected is not None:require(before.st_size==expected,'size')
  while True:
   require(time.monotonic()<DEADLINE,'audit deadline')
   b=f.read(min(1024**2,limit+1-total))
   if not b:break
   total+=len(b);require(total<=limit,'stream bound');h.update(b);m.update(b)
   if keep:chunks.append(b)
  sig=lambda s:(s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns)
  require(sig(before)==sig(os.fstat(f.fileno()))==sig(path.stat()) and total==before.st_size,'file changed')
 measured={'bytes':total,'sha256':h.hexdigest(),'md5':m.hexdigest()}
 return (measured,b''.join(chunks)) if keep else measured

def metadata(path,expected_sha=None):
 value,raw=bounded(path,keep=True)
 if expected_sha:require(value['sha256']==expected_sha,'metadata SHA '+str(path))
 INDEX[str(path.relative_to(ROOT))]=value['sha256']
 return json.loads(raw)

def claims(value):
 require(all(value[k] is False for k in ('training_admitted','spatial_planning_admitted')),'admission claim')
 require(all(value[k]=='not_run' for k in ('header_qc','geometry_qc','anatomy_qc')),'QC claim')
 require(all(value[k]==0 for k in ('decoded_array_bytes','optimizer_updates','recorded_rl_transitions')),'training claim')

for p,h in PINS.items():require(bounded(ROOT/p)['sha256']==h,'frozen code/manifest')
m=metadata(ROOT/'manifests/resect-train-originals-v1.json',PINS['manifests/resect-train-originals-v1.json'])
proof={}
for p in m['proofs']:
 got,raw=bounded(ROOT/p['path'],expected=p['bytes'],keep=True)
 require(got['sha256']==p['sha256'] and p['path'].startswith('artifacts/'),'portable proof')
 proof[p['original_path']]=raw;INDEX[p['path']]=got['sha256']
cohort=json.loads(proof['manifests/resect-component-cohort-v1.json'])
roles={x['patient_group']:x['role'] for x in cohort['members']}
train={p for p,r in roles.items() if r=='TRAIN'}
require(train=={'RESECT:Case'+str(x) for x in (2,3,5,7,11,12,15,16,17,18,21,23,24,25)},'exact TRAIN cohort')
require(m['protected_roles']=={p.split(':')[1]:r for p,r in roles.items() if r!='TRAIN'},'protected roles')
metadata(ROOT/m['role_source']['path'],m['role_source']['sha256'])
cavity=json.loads(proof['manifests/resect-train-cavity-acquisition-v1.json'])
old={s['source_url'].split('/5686d8fa-2003-4837-8e66-8e887fabe21e/')[1]:s for s in cavity['sources'] if s['kind']=='original_ultrasound'}
excluded={s['source_path']:s for s in m['excluded_existing_originals']}
require(len(old)==25 and set(old)==set(excluded),'exact old25 exclusion')
for p,x in excluded.items():require(x['bytes']==old[p]['bytes'] and x['expected_md5']==old[p]['expected_md5'],'old checksum binding')
sources={s['id']:s for s in m['sources']}
require(len(sources)==len(m['sources'])==104 and sum(s['bytes'] for s in sources.values())==672336857,'scope')
require({s['patient_group'] for s in sources.values()}==train,'all and only TRAIN people')
require(not (set(old)&{s['source_path'] for s in sources.values()}),'duplicated existing original')
require(len({s['destination'] for s in sources.values()})==len({s['source_url'] for s in sources.values()})==104,'duplicate source')
rows=proof['data/mechanics/resect-metadata/table-of-contents.csv'].decode().splitlines()
for s in sources.values():
 row=[c.strip() for c in rows[s['inventory_line']-1].split('|')]
 require(s['role']=='TRAIN' and s['source_path']==row[2] and s['bytes']==int(row[4]) and s['expected_md5']==row[6],'published catalog identity')
 require(s['source_url']=='https://s3.nird.sigma2.no/archive-ro/5686d8fa-2003-4837-8e66-8e887fabe21e/'+row[2],'exact source URL')
require(m['release']['license']=='CC-BY-4.0' and m['release']['version']==1,'release rights')

# Preserve historical old-queue result state without opening its scientific files.
historic_paths=sorted((ROOT/'data/acquisition/continuous-source-v1').rglob('outcome.json'))
historic_paths += [ROOT/'build/resect-exact-mirror-completion-review-v1/verification.json',ROOT/'build/resect-exact-mirror-acquisition-v1/actual-import-result.json']
old_before={str(p.relative_to(ROOT)):bounded(p)['sha256'] for p in historic_paths}
prior=metadata(ROOT/'build/resect-exact-mirror-completion-review-v1/verification.json','8ea4a68ab3aa9c00875bfc36378252bd6f40511395555bda3c4d9fb6c19f2612')
require(prior['coverage']['people_with_qualified_pairs']==13 and prior['coverage']['annotation_files']==25,'prior label coverage')
launch=metadata(EXEC/'actual-launch.json');terminal=metadata(EXEC/'actual-terminal.json')
require(terminal['exit_code']==0 and terminal['source_commit']==launch['source_commit']=='85ac688d04839397051093a75ac6df41415f1725','terminal source binding')
require(terminal['pid']==launch['pid'] and terminal['command']==launch['command'],'terminal run binding')
for p,h in PINS.items():
 raw=subprocess.check_output(['git','show',launch['source_commit']+':'+p],cwd=ROOT,timeout=10)
 require(len(raw)<=2*1024**2 and sha(raw)==h,'launch commit source mismatch')
runs=sorted((CACHE/'runs').iterdir());require(len(runs)==1,'unexpected runs')
run=runs[0];decl=metadata(run/'declaration.json');done=metadata(run/'completion.json')
runtime={p:h for p,h in PINS.items() if p.startswith('scripts/')}
require(decl['runtime']==runtime and decl['bounds']==m['bounds'] and decl['manifest_sha256']==PINS['manifests/resect-train-originals-v1.json'],'declaration binding');claims(decl)
require(done['all_complete'] is True and done['counts']=={'byte_verified':104} and done['local_failures']==[] and done['files']==104,'run completion');claims(done)
for p,h in PINS.items():
 snapshot=run/'source'/p
 require(bounded(snapshot)['sha256']==h,'source snapshot binding');INDEX[str(snapshot.relative_to(ROOT))]=h
_,lograw=bounded(EXEC/'actual-run.log',keep=True)
log=[json.loads(line) for line in lograw.splitlines()]
require(log[-1]==done and len(log)==105 and {x['source_id'] for x in log[:-1]}==set(sources) and all(x['status']=='byte_verified' for x in log[:-1]),'actual log complete sources')
require({p.name for p in (CACHE/'objects').iterdir()}==set(sources),'extra/missing object')

files=[];inodes=set();request_clients=set();attempt_count=0
for sid,s in sources.items():
 obj=CACHE/'objects'/sid
 comp=metadata(obj/'completion.json');binding=sha(encode(s))
 require(comp['source_binding']==binding and comp['bytes']==s['bytes'] and comp['md5']==s['expected_md5'],'canonical receipt source');claims(comp)
 attempts=sorted((obj/'attempts').iterdir());require(len(attempts)==1 and attempts[0].name=='0001','unexpected attempts')
 trial=attempts[0];attempt_count+=1
 intent=metadata(trial/'intent.json');result=metadata(trial/'result.json');supervision=metadata(trial/'supervision.json')
 require(intent['source_id']==sid and intent['source_binding']==binding and intent['manifest_sha256']==decl['manifest_sha256'] and intent['runtime']==runtime and intent['supervisor_pid']==launch['pid'] and intent['attempt']==1,'attempt source/runtime')
 require(intent['prefix']=={'bytes':0,'sha256':sha(b'')},'unexpected resume')
 require(result['intent_sha256']==sha(encode(intent)) and result['source_binding']==binding and result['status']=='byte_verified' and result['curl_exit_code']==0 and result['http_status']==200 and result['completed']==comp,'result completion');claims(result)
 require(supervision['state']=='completed' and supervision['returncode']==0,'supervision failure')
 request=metadata(trial/'native-request.json')
 require(request['source_binding']==binding and request['url_sha256']==sha(s['source_url'].encode()) and request['offset']==0 and request['length']==s['bytes'],'request source')
 client=request['client'];require(client['available'] is True and client['executable']=='/usr/bin/curl' and client['trust']=='existing_macos_system_store','native TLS provenance')
 request_clients.add((client['sha256'],client['version']))
 merge=metadata(trial/'merge.json');merged=metadata(trial/'merged.json')
 require(merge==merged and merge['intent_sha256']==sha(encode(intent)) and merge['source_binding']==binding and merge['before_bytes']==0 and merge['after_bytes']==s['bytes'] and merge['after_sha256']==comp['sha256'] and merge['after_md5']==s['expected_md5'] and merge['suffix_bytes']==s['bytes'] and merge['suffix_sha256']==comp['sha256'],'durable merge chain')
 _,headers=bounded(trial/'headers.local',limit=65536,keep=True)
 lines=headers.split(b'\r\n',1);require(lines[0]==b'HTTP/1.1 200 OK','actual HTTP status')
 parsed=BytesParser().parsebytes(lines[1]);require(parsed.get_all('Content-Length')==[str(s['bytes'])] and not parsed.get_all('Content-Range') and parsed.get_all('Content-Encoding') in (None,['identity']),'actual response framing')
 require(not (obj/'body.partial').exists() and not (trial/'body.partial').exists(),'retained unexpected partial')
 target=ROOT/s['destination'];actual=bounded(target,limit=s['bytes'],expected=s['bytes'])
 require(actual['md5']==s['expected_md5'] and actual['sha256']==comp['sha256'],'actual final full fixity')
 ino=(target.stat().st_dev,target.stat().st_ino);require(ino not in inodes,'duplicate body inode');inodes.add(ino)
 files.append({'source_id':sid,'patient_group':s['patient_group'],'kind':s['kind'],'path':s['destination'],**actual,'completion_receipt_sha256':INDEX[str((obj/'completion.json').relative_to(ROOT))],'source_binding':binding,'status':'published_size_md5_and_local_sha256_match'})
require(old_before=={str(p.relative_to(ROOT)):bounded(p)['sha256'] for p in historic_paths},'historical queue modified during audit')
for p,h in PINS.items():require(bounded(ROOT/p)['sha256']==h,'runtime source changed during audit')
require(len(request_clients)==1,'native client changed')
kind=Counter(s['kind'] for s in sources.values())
require(dict(kind)=={'source_correspondence_landmarks':59,'original_mri':28,'original_ultrasound':17},'kind coverage')
report={
 'schema':'resect-train-originals-output-review-v1','status':'all104_canonical_bodies_and_source_receipt_chains_verified',
 'finished_utc':datetime.now(timezone.utc).isoformat(),'elapsed_seconds':time.monotonic()-START,'source_commit':launch['source_commit'],'source_pins':PINS,
 'auditor_sha256':sha(Path(__file__).read_bytes()),'run':str(run.relative_to(ROOT)),'run_terminal':terminal,
 'coverage':{'new_unique_files':104,'people_in_new_file_set':14,'newly_declared_people':0,'bytes':672336857,'landmark_files':59,'mri_files':28,'ultrasound_files':17,'protected_people_excluded':9,'previous_originals_excluded':25,'combined_unique_train_original_files':129,'combined_unique_train_images':70,'combined_unique_train_landmark_files':59,'combined_train_people':14,'cavity_mask_files_unchanged':25,'cavity_label_people_unchanged':13,'new_label_files':0,'new_verified_trajectories':0},
 'checks':{'published_sizes_match':104,'published_md5_match':104,'local_sha256_matches_completion':104,'source_intent_runtime_merge_completion_chains':104,'full_body_unique_inodes':104,'attempts':attempt_count,'http200_curl0_completed':104,'automatic_retries':0,'portable_proofs':len(proof),'snapshots':len(PINS),'prior_history_receipts_unchanged_during_review':len(old_before)},
 'old_history_fingerprint_sha256':sha(encode(old_before)),
 'rights':m['release'],'protected_roles':m['protected_roles'],'missing_cavity_annotations_unchanged':cavity['missing_annotations'],
 'limits':{'wall_seconds':120,'stream_chunk_bytes':1048576,'body_bytes_cap':'exact published size plus one excess sentinel','metadata_file_bytes':2*1024**2},
 'network_requests':0,'scientific_decoding':False,'header_qc':'not_run','geometry_qc':'not_run','anatomy_qc':'not_run','training_admitted':False,'spatial_planning_admitted':False,'decoded_array_bytes':0,'optimizer_updates':0,'recorded_rl_transitions':0,
 'limitations':['Original release supplies published MD5 and size; SHA256 is local measured fixity, never claimed as published.','This audit establishes acquired bytes and source receipt binding only. Landmark coordinates and anatomy were not parsed.','Before/during/after images add no cavity labels or verified recorded surgical trajectories.','Old890 queue scientific files were not reopened; its retained outcome state and prior independent RESECT completion receipt were preserved.'],
 'files':files,
}
for name,value in [('metadata-sha256-index.json',INDEX),('prior-history-sha256-index.json',old_before),('verification.json',report)]:
 with (OUT/name).open('xb') as f:f.write(encode(value));f.flush();os.fsync(f.fileno())
print(json.dumps({'status':report['status'],'coverage':report['coverage'],'checks':report['checks'],'elapsed_seconds':report['elapsed_seconds'],'report_sha256':sha((OUT/'verification.json').read_bytes())}))

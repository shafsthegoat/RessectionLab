"""Compact copies only; hash saved JSON outputs, never read checkpoint bytes."""
import hashlib,json,os,signal,stat,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];A=Path(__file__).parent;P=ROOT/'build/balanced-teacher-il64-v1';R=P/'attempt-01';S=P/'attempt-01.supervision'
DEST=A/'stage/artifacts/balanced-teacher-il64-result-v1';DEST.mkdir(parents=True,exist_ok=False)
signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('60s packaging cap')));signal.alarm(60)
def raw(p):
 p=Path(p)
 if p.suffix not in ['.json','.py','.txt'] or not p.is_relative_to(ROOT):raise ValueError('metadata/source only')
 fd=os.open(p,os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK)
 try:
  st=os.fstat(fd)
  if not stat.S_ISREG(st.st_mode) or st.st_size>8*1024**2:raise ValueError('bounded regular file')
  with os.fdopen(fd,'rb',closefd=False) as f:b=f.read(8*1024**2+1)
 finally:os.close(fd)
 if len(b)>8*1024**2:raise ValueError('bounded bytes')
 return b
def sha(b):return hashlib.sha256(b).hexdigest()
pairs=[]
for name in ['REPORT.txt','audit_saved.py','audit-result.json','input-hashes.json','audit-run-01.txt','audit-run-02.txt','audit_saved_pre_float32_bound.py','diagnose_score_arithmetic.py','score-arithmetic-diagnosis.json','score-arithmetic-diagnosis.txt','package_saved.py']:pairs.append((A/name,'audit/'+name))
for name in ['pilot_contract.py','cohort_worker.py','run_owned.py','freeze-runtime.py']:pairs.append((P/name,'driver/'+name))
for name in ['root-release.json','source-index.json']:pairs.append((P/name,'bindings/'+name))
for name in ['result.json','configuration.json','costs.json','teacher-pins.json','checkpoint-reload.json','endpoint-teacher-metrics.json','training-dynamics.json','first-eight-update-control.json']:pairs.append((R/name,'run/'+name))
for subject in ['ReMIND-008','ReMIND-010','ReMIND-020','ReMIND-025']:
 for step in range(2 if subject=='ReMIND-025' else 1):pairs.append((R/'teacher-readout'/subject/f'state-{step:02d}.json',f'run/readouts/{subject}-state-{step:02d}.json'))
for name in ['receipt.json','worker-final.json','endpoint-control.json','declaration.json']:pairs.append((S/name,'supervision/'+name))
entries=[]
for source,relative in pairs:
 b=raw(source);dest=DEST/relative;dest.parent.mkdir(parents=True,exist_ok=True)
 with dest.open('xb') as f:f.write(b)
 if dest.read_bytes()!=b:raise ValueError('copy mismatch')
 entries.append({'path':relative,'source':str(source.relative_to(ROOT)),'bytes':len(b),'sha256':sha(b)})
receipt=json.loads(raw(S/'receipt.json'));copied={str(p.relative_to(ROOT)) for p,_ in pairs};inventory=[]
for dirpath,dirs,files in os.walk(R):
 for name in sorted(files):
  p=Path(dirpath)/name;st=p.lstat()
  if not stat.S_ISREG(st.st_mode):raise ValueError('nonregular run output')
  row={'path':str(p.relative_to(ROOT)),'bytes':st.st_size,'included_in_compact_package':str(p.relative_to(ROOT)) in copied}
  if p.suffix=='.json':row.update(sha256=sha(raw(p)),hash_authority='independent_saved_json_byte_hash')
  elif p.name=='IL-final.psckpt':
   saved=receipt['checkpoints']['IL']
   if st.st_size!=saved['bytes']:raise ValueError('checkpoint size differs')
   row.update(sha256=saved['sha256'],hash_authority='owned_parent_receipt; checkpoint not opened by packaging')
  else:raise ValueError('unapproved output type '+name)
  inventory.append(row)
inv={'scope':'All run output paths/bytes; saved JSON hashes independently verified; checkpoint digest taken from authenticated parent, not reopened. No weights or bulk traces copied. Source index binds canonical dependencies without copying them.','run_directory':str(R.relative_to(ROOT)),'files':inventory,'total_bytes':sum(x['bytes'] for x in inventory)}
if inv['total_bytes']!=receipt['output_bytes']:raise ValueError('total output size differs')
payload=(json.dumps(inv,sort_keys=True,indent=2)+'\n').encode();(DEST/'full-output-bindings.json').write_bytes(payload);entries.append({'path':'full-output-bindings.json','source':'generated from saved JSON bytes and parent checkpoint record','bytes':len(payload),'sha256':sha(payload)})
for source,relative in pairs:
 if sha(raw(source))!=next(e['sha256'] for e in entries if e['path']==relative):raise ValueError('source changed during copy')
index={'version':'balanced-teacher-il64-result-package-v1','scope':'Compact exact driver/runtime/metadata/readout/64-update-dynamics/audit package; both audit runs and rounding diagnosis preserved; no weights, patient arrays or full traces.','files':entries,'file_count':len(entries),'bytes':sum(e['bytes'] for e in entries)}
(DEST/'source-index.json').write_text(json.dumps(index,sort_keys=True,indent=2)+'\n')
print(json.dumps({'destination':str(DEST.relative_to(ROOT)),'index_sha256':sha((DEST/'source-index.json').read_bytes()),'files':len(entries),'bytes':index['bytes'],'bound_output_files':len(inventory),'bound_output_bytes':inv['total_bytes']}))

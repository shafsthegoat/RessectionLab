"""Compare saved JSON records only. Safe to rerun as the canonical trace grows."""
from pathlib import Path
import hashlib,json,math,struct,collections,os
from datetime import datetime,timezone
if not __debug__: raise RuntimeError('Exact comparison requires assertions enabled')
HERE=Path(__file__).resolve().parent
LIVE_SHA='e601eba917d6cc84ddf1e6d97659cb3db8d2b1456e2ffe374de3a1ec15ebdc80'
TOTAL=2148

def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()

def complete_rows(path):
 data=path.read_bytes();last=data.rfind(b'\n')
 if last<0:return [],len(data)
 return [json.loads(x) for x in data[:last+1].splitlines()],len(data)-last-1

def exact(a,b):
 if type(a) is not type(b):return False
 if type(a) is float:return struct.pack('!d',a)==struct.pack('!d',b)
 if type(a) is list:return len(a)==len(b) and all(exact(x,y) for x,y in zip(a,b))
 if type(a) is dict:return a.keys()==b.keys() and all(exact(a[k],b[k]) for k in a)
 return a==b

live_path=HERE/'live-primary-history.jsonl';trace_path=HERE/'primary-canonical.actions.jsonl'
assert sha(live_path)==LIVE_SHA
live,trailing=complete_rows(live_path);assert not trailing and len(live)==TOTAL
export=json.loads((HERE/'live-primary-history.receipt.json').read_text());freeze=json.loads((HERE/'artifact-freeze.json').read_text())
assert export['historySha256']==LIVE_SHA and export['actions']==TOTAL and export['physicalActionsDuringExport']==0
assert export['stateId']==freeze['branches']['full-primary']['stateId']
assert sha(Path(freeze['native']))==freeze['nativeSha256']
canonical,trailing=complete_rows(trace_path)
assert len(canonical)<=TOTAL
fields=('action','stateBefore','stateAfter','status','reason','reasonCode','completedFraction','elapsedSeconds','events')
counts=collections.Counter();event_counts=collections.Counter();omitted_contact=0;previous=None
for index,row in enumerate(live,1):
 assert row['position']==index and set(row)=={'position','outcome'}
 outcome=row['outcome'];assert set(outcome)==set(fields)|{'contactPosition','deltas'}
 assert set(outcome['deltas'])=={'removedCells','geometryRevision'}
 assert outcome['deltas']['geometryRevision']==outcome['stateAfter']
 assert previous is None or outcome['stateBefore']==previous
 previous=outcome['stateAfter'];omitted_contact+=outcome['contactPosition'] is not None
assert previous==export['stateId']
mismatches=[]
for index,row in enumerate(canonical,1):
 assert row['position']==index
 o=live[index-1]['outcome']
 for key in fields:
  if key not in row or not exact(o[key],row[key]):mismatches.append({'position':index,'field':key})
  else:counts[key]+=1
 if not exact(o['deltas']['removedCells'],row['removedCells']):mismatches.append({'position':index,'field':'deltas.removedCells→removedCells'})
 else:counts['deltas.removedCells→removedCells']+=1
 if row['stateAfter']!=o['deltas']['geometryRevision']:mismatches.append({'position':index,'field':'deltas.geometryRevision→stateAfter'})
 else:counts['deltas.geometryRevision→stateAfter']+=1
 event_counts.update(e['kind'] for e in o['events'])
canonical_receipt=HERE/'primary-canonical.json';receipt=json.loads(canonical_receipt.read_text()) if canonical_receipt.exists() else None
complete=len(canonical)==TOTAL and trailing==0 and receipt is not None and receipt.get('auditStatus')=='passed'
if complete:
 assert receipt['actions']==TOTAL and receipt['stateId']==export['stateId'] and receipt['branch']=='full-primary'
 assert Path(receipt['canonicalReplay']['trace']).resolve()==trace_path.resolve()
 assert receipt['nativeSha256']==freeze['nativeSha256'] and receipt['implementationFingerprint']==freeze['implementationFingerprint']
 assert receipt['canonicalReplay']['forced'] is True and receipt['canonicalReplay']['cacheUsed'] is False
report={'schema':'resectionlab.original-live-vs-final-canonical-record-comparison.v1','status':'failed' if mismatches else ('passed' if complete else 'provisional'),'checkedUtc':datetime.now(timezone.utc).isoformat(),'expectedActions':TOTAL,'originalLiveRows':len(live),'completeCanonicalRowsCompared':len(canonical),'trailingIncompleteCanonicalBytesIgnored':trailing,'canonicalSuccessfulReceiptPresent':receipt is not None and receipt.get('auditStatus')=='passed','all2148AndCanonicalSuccessRequiredForPass':True,'nativeSha256':freeze['nativeSha256'],'implementationFingerprint':freeze['implementationFingerprint'],'originalLiveHistory':str(live_path),'originalLiveHistorySha256':LIVE_SHA,'canonicalTrace':str(trace_path),'canonicalTraceSha256':sha(trace_path) if complete else None,'exactEqualCounts':dict(counts),'comparedEventKindCounts':dict(event_counts),'mismatches':mismatches[:30],'mismatchCount':len(mismatches),'comparisonSemantics':'Exact recursive JSON types and values; float IEEE754 bits including signed zero; list/event ordering preserved; object key order ignored.','omittedOriginalLiveFields':{'contactPosition':{'totalRows':TOTAL,'nonNullRows':omitted_contact,'reason':'Canonical trace does not serialize this top-level descriptive outcome field; no equality claim. Contact/material/aspiration/drill payloads present in events are compared exactly.'},'deltas.geometryRevision':{'reason':'Canonical trace omits nested field; all original values independently equal stateAfter, which is compared exactly.'}},'canonicalOnlyFields':['fluid','vascularState','drillingAndDebris'],'materialAndFieldOutcomeScope':'Complete events arrays (including removal, drill_work, aspiration and contact payloads), removed-cell deltas and full resulting state IDs are compared; canonical-only derived cumulative ledgers are not independently present in live outcomes. State-ID equality binds complete canonical state, not a separate direct field-by-field live-state export.','physicsCalls':0,'guiCalls':0,'limits':['Saved record equality is independent of replay execution but relies on original live export provenance. No new physical replay performed.','A growing trace or its last complete row never implies final completion; passed requires all2148 rows and matching successful forced-canonical receipt.']}
progress=HERE/'live-vs-canonical-comparison-progress.json';temp=progress.with_suffix('.tmp');temp.write_text(json.dumps(report,indent=2)+'\n');temp.replace(progress)
if complete and not mismatches:
 final=HERE/'live-vs-canonical-exact-comparison.json'
 if final.exists():assert json.loads(final.read_text())['canonicalTraceSha256']==report['canonicalTraceSha256']
 else:final.open('x').write(json.dumps(report,indent=2)+'\n')
print(json.dumps({k:report[k] for k in ('status','completeCanonicalRowsCompared','expectedActions','canonicalSuccessfulReceiptPresent','mismatchCount','omittedOriginalLiveFields')}))
if mismatches:raise SystemExit('Original live vs canonical outcome mismatch; inspect progress receipt')

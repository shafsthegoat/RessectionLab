"""Independent receipt and saved-trace audit only; stdlib, no Engine imports."""
from pathlib import Path
import collections,hashlib,json,math,struct
from datetime import datetime,timezone
B=Path(__file__).resolve().parent

def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for x in iter(lambda:f.read(1048576),b''):h.update(x)
 return h.hexdigest()

def exact(a,b):
 if type(a) is not type(b):return False
 if type(a) is float:return struct.pack('!d',a)==struct.pack('!d',b)
 if type(a) is list:return len(a)==len(b) and all(exact(x,y) for x,y in zip(a,b))
 if type(a) is dict:return a.keys()==b.keys() and all(exact(a[k],b[k]) for k in a)
 return a==b

def rows(p):return [json.loads(x) for x in p.read_text().splitlines()]
freeze=json.loads((B/'artifact-freeze.json').read_text());source_review=json.loads((B/'independent-source-record-review.json').read_text())
assert sha(Path(freeze['native']))==freeze['nativeSha256']==source_review['nativeSha256']
assert sha(Path(freeze['original']))==freeze['originalSha256']==source_review['originalSha256']
canonical={role:json.loads((B/(role+'-canonical.json')).read_text()) for role in ('primary','vascular')}
traces={};checks={}
for role,c in canonical.items():
 name='full-'+role;n=freeze['branches'][name]['actions'];original=source_review['branches'][name]
 assert c['auditStatus']=='passed' and c['branch']==name and c['actions']==n
 assert c['nativeSha256']==freeze['nativeSha256'] and c['implementationFingerprint']==freeze['implementationFingerprint']
 assert c['originalSha256']==freeze['originalSha256'] and c['sceneId']==freeze['sceneId']
 assert c['sourceArrays']==source_review['sourceArrayHashes'] and c['labelArrayHash']==source_review['labelArrayHash']
 assert c['stateId']==freeze['branches'][name]['stateId']
 assert c['canonicalReplay']['forced'] is True and c['canonicalReplay']['cacheUsed'] is False
 assert c['canonicalReplay']['verifiedCachePublication']['publications']==1
 assert c['canonicalReplay']['perActionStateAssertions']==n and c['canonicalReplay']['rejectedCommittedActions']==0
 assert c['accounting']==original['accounting'] and c['rawAnnotationMemberships']==original['rawAnnotationMemberships']
 p=c['protection'];assert p['mode']=='strict-research' and p['clearanceMm']==2 and p['criticalMaxDisplacementMm']==0
 assert p['contractId']==source_review['branchRecords'][name]['protection']['contractId'] and len(p['structures'])==11 and p['missingRequired']==[]
 assert c['endpointHolds']=={} and all(not s['active'] and not s['inserted'] and not s['grip'] for s in c['endpointTools'].values())
 assert c['vascular']['activeWounds']==0
 path=Path(c['canonicalReplay']['trace']);trace=rows(path);traces[role]=trace;assert len(trace)==n
 previous=None;seen=set();generated=0.;maximum={'fluidMm3':0.,'bloodReservoirMm3':0.,'debrisMm3':0.,'thermalJ':0.};statuses=collections.Counter()
 for position,row in enumerate(trace,1):
  assert row['position']==position and row['status'] in ('accepted','partial')
  assert previous is None or row['stateBefore']==previous
  previous=row['stateAfter'];statuses[row['status']]+=1
  assert not seen.intersection(row['removedCells']);seen.update(row['removedCells'])
  f=row['fluid'];res=f['addedMm3']-math.fsum([f['presentMm3'],f['trappedOutletMm3'],f['removedMm3'],f['escapedMm3']]);assert abs(res)<1e-8
  maximum['fluidMm3']=max(maximum['fluidMm3'],abs(res));v=f['vascular'];res=math.fsum([v['initialMm3'],v['refilledMm3'],-v['remainingMm3'],-v['generatedMm3']]);assert abs(res)<1e-8
  assert math.isclose(v['emittedMm3'],v['generatedMm3'],abs_tol=1e-8);maximum['bloodReservoirMm3']=max(maximum['bloodReservoirMm3'],abs(res))
  for event in row['events']:
   if event['kind']=='drill_work':generated+=event['removedVolumeMm3']
   elif event['kind']=='bone_kerf':generated+=event['cells']
  d=row['drillingAndDebris']['debris'];res=generated-d['presentMm3']-d['removedMm3']-d['escapedMm3'];assert abs(res)<1e-7
  assert math.isclose(generated,d['generatedByActualDrillAndKerfEventsMm3'],abs_tol=1e-7);maximum['debrisMm3']=max(maximum['debrisMm3'],abs(res))
  d=row['drillingAndDebris'].get('drilling')
  if d:
   energy=d['inputEnergyJ']-d['cuttingWorkJ']-d['unusedEnergyJ'];work=d['cuttingWorkJ']-d['removedVolumeMm3']*.5-d['pendingWorkJ'];partition=d['heatGeneratedJ']-d['cuttingWorkJ']*.35;thermal=d['heatGeneratedJ']-d['storedHeatJ']-d['coolingLossJ']
   assert all(abs(x)<1e-6 for x in (energy,work,partition,thermal));maximum['thermalJ']=max(maximum['thermalJ'],abs(thermal))
 assert previous==c['stateId'] and dict(statuses)==c['canonicalReplay']['statuses']
 assert len(seen)==sum(x['removedCells'] for x in c['accounting']['materials'].values())
 checks[role]={'actions':n,'statuses':dict(statuses),'traceSha256':sha(path),'uniqueRemovedSourceCells':len(seen),'generatedDebrisFromActualSavedEventsMm3':generated,'maximumIndependentlyRecomputedResiduals':maximum,'sourceAccountingAndEndpointsMatchPriorIndependentSourceAudit':True}
assert exact(traces['primary'][:42],traces['vascular'][:42])
assert traces['primary'][41]['stateAfter']==freeze['commonPrefix']['stateId']
v=traces['vascular'];samples=[r['vascularState'] for r in v]
assert not samples[46]['faces'] and len(samples[47]['faces'])==5 and samples[47]['activeWounds']==1
for position in (50,51):assert samples[position-1]['generatedMm3']>samples[position-2]['generatedMm3'] and all(not x['active'] for x in samples[position-1]['slots'].values())
for position in (55,56):assert sum(x['coagulationDose'] for x in samples[position-1]['faces'].values())>sum(x['coagulationDose'] for x in samples[position-2]['faces'].values())
assert all(x['coagulationDose']>=2 and x['clot']==1 and x['patent'] is False for x in samples[55]['faces'].values())
for position in (58,72,73):assert samples[position-1]['activeWounds']==0 and samples[position-1]['generatedMm3']==samples[position-2]['generatedMm3']
for position in (62,63):assert samples[position-1]['addedMm3']>samples[position-2]['addedMm3']
for position in (68,69):assert samples[position-1]['removedMm3']>samples[position-2]['removedMm3'] and samples[position-1]['generatedMm3']==samples[position-2]['generatedMm3'] and all(x['patent'] is False for x in samples[position-1]['faces'].values())
assert canonical['vascular']['causalVascularChain']['completeChain'] is True
live=B/'live-primary-history.jsonl';assert sha(live)=='e601eba917d6cc84ddf1e6d97659cb3db8d2b1456e2ffe374de3a1ec15ebdc80'
live_rows=rows(live);assert len(live_rows)==2148
fields=('action','stateBefore','stateAfter','status','reason','reasonCode','completedFraction','elapsedSeconds','events')
for original,row in zip(live_rows,traces['primary']):
 assert original['position']==row['position']
 for k in fields:assert exact(original['outcome'][k],row[k])
 assert exact(original['outcome']['deltas']['removedCells'],row['removedCells'])
auth=json.loads((B/'live-vs-authenticated-canonical-history.json').read_text());assert auth['status']=='passed' and auth['actionsCompared']==2148 and auth['allOutcomeFieldsCompared'] is True and auth['omittedOutcomeFields']==[] and auth['mismatchCount']==0 and auth['nonNullContactPositionsCompared']==202
assert auth['nativeSha256']==freeze['nativeSha256'] and auth['implementationFingerprint']==freeze['implementationFingerprint']
assert auth['canonicalReceiptSha256']==sha(B/'primary-canonical.json') and auth['scriptSha256']==sha(B/'compare_live_sealed_history.py')
assert all(n==2148 for n in auth['exactEqualOutcomeFieldCounts'].values())
progress=json.loads((B/'verification-progress.json').read_text())
result={'status':'passed: completed canonical/source/history gates reviewed','utc':datetime.now(timezone.utc).isoformat(),'nativeSha256':freeze['nativeSha256'],'implementationFingerprint':freeze['implementationFingerprint'],'evidenceSha256':{p.name:sha(p) for p in [B/'artifact-freeze.json',B/'primary-canonical.json',B/'vascular-canonical.json',B/'independent-source-record-review.json',B/'live-vs-authenticated-canonical-history.json']},'canonicalTraceRecomputation':checks,'commonPrefix42ExactRowsIncludingStatesEventsAndDerivedLedgers':True,'vascularCausalChainIndependentRecordChecks':{'injury':48,'passivePersistentBleeding':[50,51],'actualDoseIncrease':[55,56],'clot1DoseAtLeast2NonpatentBy':56,'stableIdle':[58,72,73],'irrigation':[62,63],'suctionWithoutVesselRepair':[68,69]},'primaryTarget':{'ownedOriginalMm3':43573,'removedMm3':35444,'removedPercent':100*35444/43573,'retainedMm3':8129,'residualSixConnectedRegions':161,'ownedNonTargetSoftRemovalMm3':0,'rawBrainRemovedMembership':36402,'rawTumorRemovedMembership':35810},'historicalEquality':{'independentTraceVsOriginalLiveComparison':'all2148 mapped fields/events/removed deltas exact types and floatbits','authenticatedCompleteOutcomeReceipt':'all2148 complete outcomes, including202 nonnull contactPositions, zero mismatch; receipt and comparator hashes match','objectKeyOrder':'ignored because canonical JSON sorts keys; list/event ordering remains exact'},'criticalRemainingGates':['Final primary/vascular descriptive telemetry and all-frame authenticated cache coverage not complete at this review','Actual final Electron playback/branch/persistence/usability checks pending'],'videoRequirement':'Canceled by user; not an acceptance gate and no video pass claimed','observedPipelineStatus':progress,'limitations':['Budget-limited endpoint, not maximal feasible removal or proof residual unresectable.','Source ownership/raw annotation overlap is not reviewed clinical truth; zero owned soft removal is not zero clinical healthy-brain injury. Strict11masks do not supply missing functional/dura/attachment evidence.','Primary15280.0297706mm3 debris remains after1271.9702294mm3 aspiration; bounded cleanup is not a fully cleared operative-field claim.','This review used saved records and authenticated-history comparison receipt; no Engine call, app use or new physical replay.'],'engineCalls':0,'computerUseCalls':0,'productionEdits':0}
out=B/'independent-completed-canonical-review-20261021.json';out.open('x').write(json.dumps(result,indent=2)+'\n');print(json.dumps({'receipt':str(out),'status':result['status'],'remainingGates':result['criticalRemainingGates']}))

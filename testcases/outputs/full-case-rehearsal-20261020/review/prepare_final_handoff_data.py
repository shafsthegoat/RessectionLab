"""Summarize immutable saved records and existing receipts; no Engine imports."""
from pathlib import Path
import sys,json,hashlib
from collections import Counter
from datetime import datetime,timezone
ROOT=next(p for p in Path(__file__).resolve().parents if (p/'src/resectionlab').is_dir())
sys.path.insert(0,str(ROOT/'src'))
from resectionlab.imaging import read_case_artifacts
B=ROOT/'testcases/outputs/full-case-rehearsal-20261020'; F=B/'final'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text())
def ref(p):return {'path':str(p),'sha256':sha(p)}
freeze=read(F/'artifact-freeze.json'); native=Path(freeze['native']); native_sha=sha(native)
assert native_sha==freeze['nativeSha256']
source=Path(freeze['original']);assert sha(source)==freeze['originalSha256']
record=read_case_artifacts(native)['surgical_toolbox']
invpath=F/'tool-family-inventory-source-records.json';inventory=read(invpath)
source_review=read(F/'independent-source-record-review.json')
assert source_review['nativeSha256']==native_sha
history_receipt=read(F/'live-primary-history.receipt.json');historypath=Path(history_receipt['history'])
assert sha(historypath)==history_receipt['historySha256']
history=[json.loads(line) for line in historypath.read_text().splitlines()]
primary=record['branches']['full-primary']
assert len(history)==len(primary['actions'])==2148
for index,row in enumerate(history):
 o=row['outcome'];assert row['position']==index+1 and o['action']==primary['actions'][index]
 if index:assert o['stateBefore']==history[index-1]['outcome']['stateAfter']
assert history[-1]['outcome']['stateAfter']==primary['stateId']==history_receipt['stateId']
statuses=Counter(row['outcome']['status'] for row in history)
partial_reasons=Counter(row['outcome'].get('reason') for row in history if row['outcome']['status']=='partial')
branches={}
for role,branch in [('primary','full-primary'),('vascular','full-vascular')]:
 snap=record['branches'][branch];state=snap['checkpoint']['state'];entry=inventory['episodes'][role];effects=entry['endpointEffects']
 receiptpath=B/(role+'-v2')/'execution-receipt.json';receipt=read(receiptpath);meta=record['fullCase'][branch]
 assert receipt['stateId']==snap['stateId'] and receipt['actions']==len(snap['actions']) and receipt['sceneId']==snap['sceneId']
 owned=effects['initialOwnedTargetMm3'];removed=effects['removedOwnedTargetMm3']; raw_counts=inventory['sourceCounts']['rawCompartments']
 branches[role]={
  'branch':branch,'savedActionCount':len(snap['actions']),'endpointStateId':snap['stateId'],
  'simulatedElapsedSeconds':source_review['branches'][branch]['elapsedSimulationSeconds'],
  'sourceRecordedStatus':meta['status'],'sourceRecordedStopReason':meta['stopReason'],
  'ownedTarget':{'initialMm3':owned,'removedMm3':removed,'retainedMm3':owned-removed,'removedPercent':100*removed/owned},
  'rawSourceAnnotationDenominatorsCells':raw_counts,
  'removedSourceCellRawMemberships':effects['removedSourceCellRawCompartmentMemberships'],
  'ownedNonTargetRemoval':{'softTissueMm3':effects['removedOwnedMaterialMm3'].get('soft_tissue',0.),'vesselMm3':effects['removedOwnedMaterialMm3'].get('vessel',0.),'unknownMm3':receipt['accounting']['unknownRemovedMm3'],'boneMm3':effects['removedOwnedMaterialMm3'].get('bone',0.),'qualifier':'Owned material precedence under the supplied segmentation. Zero owned non-target soft-tissue removal does NOT establish zero clinical healthy-brain injury, zero raw brain membership, or safe clinical outcome.'},
  'boneAndDebris':{k:effects[k] for k in ['debrisPresentMm3','debrisRemovedMm3','debrisEscapedMm3','boneDebrisConservationResidualMm3']},
  'fluid':{k:effects[k] for k in ['fluidAddedMm3','fluidRemovedMm3','fluidEscapedMm3','fluidPresentIncludingOutletsMm3','fluidConservationResidualMm3']},
  'wounds':effects['wounds'],'bothHandsOutInactiveReleased':source_review['branches'][branch]['bothHandsOutInactiveReleased'],
  'wallTimeEvidence':{**ref(receiptpath),'recordedWallSeconds':receipt.get('wallSeconds'),'executionMode':receipt.get('executionMode'),'basis':('Original live recovery-session elapsed wall includes console idle time; not compute time or a replay benchmark.' if role=='primary' else 'Existing vascular execution-receipt timer begins after source load/prefix restoration, before initial persist and31 new actions; includes persistence. It is not total preparation or cold replay time.')},
  'evidenceBasis':'Saved endpoint/source-record review and exactly matching endpoint execution receipt; final frozen-implementation canonical/observed verification remains separate.',
  'existingMatchedObservedReplay':entry.get('verifiedTelemetry')}
 branches[role]['boneAndDebris']['generatedBoneDebrisMm3']=effects['drilling']['removedVolumeMm3']
 branches[role]['fluid']['modeledVascularBloodGeneratedMm3']=effects['wounds']['generatedMm3']
 branches[role]['fluid']['irrigationAddedMm3']=effects['fluidAddedMm3']-effects['wounds']['generatedMm3']
 if role=='primary':
  cleanup=meta.get('fieldCleanup',{}); results=cleanup.get('results',[])
  branches[role]['finalBoundedCleanup']={'status':cleanup.get('status'),'stations':len(cleanup.get('stations',[])),'actionPositions1Based':[r['position']for r in results], 'debrisAspiratedDuringRecordedCleanupMm3':sum(r.get('debrisRemovedMm3',0.)for r in results),'qualification':cleanup.get('qualification'),'residualDebrisRetainedMm3':effects['debrisPresentMm3']}
  branches[role]['savedLiveOutcomeSanityCheck']={**ref(historypath),'receipt':ref(F/'live-primary-history.receipt.json'),'allActionsExactlyMatchSavedRequests':True,'stateIdChainAndEndpointMatch':True,'statuses':dict(statuses),'partialReasons':dict(partial_reasons),'qualification':'Preserved original live outcomes only; no independent canonical comparison or replay performed by this summary. Canonical comparison belongs to acceptance reviewer.'}
  branches[role]['blockersAndUntriedRoutes']={'endpointCause':'User/work-budget stop with retained opportunities; not exhaustive geometric failure or maximal resection.', 'actualCommittedPartialReasons':dict(partial_reasons),'savedDirectorEngineReasonCounters':meta.get('engineRejectionReasons',{}),'uncommittedProposalReasonCounters':entry['savedUncommittedProposalReasons'],'counterQualification':'Director counters and proposal records include attempted/rejected proposals and are not extra saved physical actions.', 'superiorCursor':meta.get('accessIndex'),'superiorOpportunityListLength':len(meta.get('accesses',[])),'pausedStage':meta.get('budgetLimitedEndpoint',{}).get('pausedStage'),'qualifier':'Remaining listed/untried routes and incomplete later passes were not established unreachable. No new reachability search performed.'}
 else:
  branches[role]['fluid']['irrigationAddedMm3']=6.0
  branches[role]['fluid']['irrigationTotalResidualMm3']=effects['fluidAddedMm3']-6.0-effects['wounds']['generatedMm3']
  branches[role]['branchPurpose']='Peripheral supplied vessel complication and deliberate bipolar sealing/field-fluid handling; retains common-prefix target residual, not a tumor-clearance alternative.'
progress=read(F/'verification-progress.json')
output={'schema':'resectionlab.measured-final-handoff-data.v1','createdUtc':datetime.now(timezone.utc).isoformat(),
 'scope':'Read-only saved source/action/outcome/endpoint records and existing evidence; no Engine construction, restore, preview, step, replay, GUI, video or production edits.',
 'frozenArtifact':ref(native),'freezeReceipt':ref(F/'artifact-freeze.json'),'source':ref(source),'sceneId':freeze['sceneId'],
 'originalSourcePreserved':{'fileHashMatchesOriginal':True,'sourceArrayHashes':source_review['sourceArrayHashes'],'embeddedSourceAndReferencesPreserved':source_review['sourceReferencesAndEmbeddedOriginalsPreserved'],'sourceRecordReview':ref(F/'independent-source-record-review.json')},
 'denominatorAndHealthyQualification':inventory['sourceCounts']['qualification']+' Owned-target denominator43573mm3, full raw tumor annotation45115cells; source voxels are1mm3. Raw brain annotation is a compartment, not a separate brain_mask in this file.',
 'commonPrefix':freeze['commonPrefix'],'branches':branches,'toolFamilyInventory':ref(invpath),
 'toolEvidenceSummary':{'actualPatientActionFamilies':[r['tool']for r in inventory['tools']if r['patientActionEvidence']], 'matchedExistingObservedEffectFamilies':[r['tool']for r in inventory['tools']if r['patientProof']['observedEffectsBySelectedEpisode']], 'syntheticFixtureOnlyFamilies':[r['tool']for r in inventory['tools']if not r['patientActionEvidence']], 'qualification':'Existing targeted synthetic tool receipts are retained with explicit fixture provenance in the inventory; they do not establish patient anatomy or clinical force/outcome calibration.'},
 'missingAndUncalibratedEvidence':inventory['anatomyEvidence'],
 'assumptions':['Unreviewed supplied masks and owned-material precedence; overlapping annotations are not independent clinical truth.','Generic declared tool/material, flow, coagulation, thermal and wound laws are uncalibrated research assumptions.','Complete vascular topology, perfusion, scalp/dura attachments, bone anchors and functional localization are not established.','Residual connected regions/mass and conservative geometric blockers do not prove clinical resectability or unresectability.'],
 'pendingAcceptance':{'finalCanonicalAndTelemetry':{'statusAtRead':progress['status'],'progressReceipt':ref(F/'verification-progress.json'),'qualification':'Active pipeline owned by root; no final success inferred from source records or prior branch receipts.'},'finalObservedFrameCoverage':'pending; do not infer from older vascular receipts','finalGuiPersistencePlaybackAndVideo':'pending/separate; this summary makes no GUI, video or verification pass claim.'}}
assert sha(native)==native_sha and sha(source)==freeze['originalSha256']
path=F/'measured-final-handoff-data.json'
with path.open('x')as f:json.dump(output,f,indent=2,allow_nan=False);f.write('\n')
print(json.dumps({'output':str(path),'primaryActions':2148,'primaryOwnedTargetPercent':branches['primary']['ownedTarget']['removedPercent'],'primarySimSeconds':branches['primary']['simulatedElapsedSeconds'],'vascularSimSeconds':branches['vascular']['simulatedElapsedSeconds'],'primaryLiveStatuses':dict(statuses),'vascularBloodMm3':branches['vascular']['fluid']['modeledVascularBloodGeneratedMm3'],'vascularIrrigationMm3':6.,'totalVascularFluidAddedMm3':branches['vascular']['fluid']['fluidAddedMm3']}))

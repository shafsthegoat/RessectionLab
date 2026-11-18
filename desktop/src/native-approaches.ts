import type {Point3} from './viewer/coordinates';
import type {ViewerRoute} from './viewer/contracts';
import type {EnvironmentOptions} from './native-anatomy-data';

export interface ApproachSensitivityConfig {translationMm:number;boundaryMm:number}
export const defaultApproachSensitivity:ApproachSensitivityConfig={translationMm:1,boundaryMm:.5};

export interface ApproachTargetOption {
  layerId:string;label:string;target:Point3|null;entry?:Point3;
  candidateBinding?:string;kind?:string;origin?:string;sourceSeries?:string;layerRevision?:number;labelValue?:number;
  reviewStatus?:string;coverageMeaning?:string;model?:string|null;provenance?:string|null;focusMeaning?:string;reason?:string;
}

export interface SensitivityMetric {
  units:string;adverseDirection:string;nominal:number|null;worst:number|null;
  worstScenarioId:string|null;tailMean:number|null;tailFraction:number;
  assessedCount:number;unknownCount:number;
}

export interface SensitivityScenarioResult {
  id:string;status:string;values:Record<string,number|null>;unknowns?:string[];
  coverageIncomplete?:boolean;
  conflictLayerIds?:string[];constraintFailures?:string[];
  sourceReadings?:Record<string,ApproachSourceReading>;
}
export type ApproachSourceReading=[coverage:string,clearanceLowerBoundMm:number|null,contact:string|null,count:number|null];

export interface ApproachSensitivity {
  scenarioSetId:string;scenarioCount:number;metrics:Record<string,SensitivityMetric>;
  scenarios:SensitivityScenarioResult[];scenarioSet?:ApproachScenarioSet;assumptions?:string[];meaning?:string;
}

export interface ApproachScenarioSet {
  id:string;config:ApproachSensitivityConfig;assumptions:string[];
  scenarios:Array<{id:string;sourceTranslationsMm?:Record<string,Point3>;boundaryMm?:number}>;
}

export interface ApproachProposalCandidate {
  id:string;entry:Point3;target:Point3;radiusMm:number;direction:Point3|null;disposition:string;
  reasons?:string[];objectives?:Record<string,number|null>;pareto?:boolean;detectedConflictLayerIds?:string[];
  sourceReadings?:Record<string,ApproachSourceReading>;
  scenarioGuard?:ApproachScenarioGuard;
  accessAssessment?:CorticalAccessAssessment;
}
export interface ApproachScenarioGuard {disposition:string;vetoScenarioIds:string[];incompleteScenarioIds:string[];meaning?:string}

export interface ApproachProposalReport {
  evaluated:number;accepted:number;eligible?:number;enumerated?:number;rejected:number;
  reasons:unknown;selectionRule:string;whyFewer?:string|null;targetLayerId?:string|null;
  candidates?:ApproachProposalCandidate[];scenarioSet?:ApproachScenarioSet;
  selectedCandidateIds?:string[];sourceSnapshotId?:string;assessmentStatus?:string;invalidationReason?:string;
  finalistOutcomes?:Record<string,{metrics:Record<string,SensitivityMetric>;
    scenarios?:Array<Pick<SensitivityScenarioResult,'id'|'constraintFailures'>>}>;
  task?:ApproachTask;
}

export interface ApproachTask {
  targetLayerId?:string|null; entry?:Point3; target?:Point3; radiusMm?:number;
  entryPatch?:NativeApproach['entryPatch']; constraints?:NativeApproach['constraints'];
  sensitivityConfig?:ApproachSensitivityConfig;
}

export interface ApproachInputNomination {
  candidateId:string;status:'refused'|'incomplete-coverage'|'eligible-static';reasons:string[];
  objectives?:Record<string,number|null>;detectedConflictLayerIds?:string[];sensitivity?:ApproachSensitivity;
  scenarioGuard?:ApproachScenarioGuard;
  accessAssessment?:CorticalAccessAssessment;
}

export interface ApproachInputArm {
  inputWhitelist:string[];layerIds:string[];sourceSeriesIds:string[];tractIds:string[];
  selectedCandidateIds:string[];candidates:ApproachProposalCandidate[];nominated:ApproachInputNomination[];
  refusalCount:number;abstentionCount:number;
}

export interface ApproachInputComparisonReport {
  arms:{rich:ApproachInputArm;restricted:ApproachInputArm};task?:ApproachTask;
  candidateBudget:number;scenarioSet:ApproachScenarioSet;candidateFreeze?:{id:string};guardProtection?:string;
  sourceSnapshotId?:string;assessmentStatus?:string;
}

export interface ApproachStructure {
  layerId:string;label:string;kind:string;available:boolean;
  clearanceLowerBoundMm?:number|null;intersectsOrUnresolved?:boolean;confirmedIntersection?:boolean;
  excluded?:boolean;reason?:string;reviewStatus?:string;sourceSeries?:string;coverageMeaning?:string;
  sourceIdentity?:{dataSha256?:string;[key:string]:unknown};
  coverage?:ApproachCoverage;acquisitionCoverage?:ApproachCoverage;
  closestPointRasMm?:Point3;nearestPoint?:Point3;[key:string]:unknown;
}
export interface ApproachCoverage {wholeEnvelopeWithinField?:boolean;outsideFaces?:string[]}
export interface ApproachTargetMeasurement {
  label?:string;sourceSeries?:string;available:boolean;coverage?:ApproachCoverage;acquisitionCoverage?:ApproachCoverage;
  totalVolumeMm3?:number;coveredVoxelCenterFraction?:number;coveredVoxelCenterVolumeMm3?:number;targetPointInsideLabel?:boolean;
}
export interface CorticalAccessAssessment {
  status:'supported'|'incomplete'|'rejected';scope:string;hemisphere?:'left'|'right'|null;
  missingEvidence?:string[];violations?:string[];sourceSnapshotId?:string;surfaceEntry?:unknown;
  sourceLayers?:Record<string,{layerId:string;sourceSeries:string;revision:number;evidence:string}>;
  postEntryEnvelopeSupported?:boolean|null;distalCapSupported?:boolean|null;
}
export interface NativeApproach {
  id:string;name:string;sourceSnapshotId:string;parameterHash:string;
  entry:Point3;target:Point3;radiusMm:number;direction:Point3;
  entryPatch?:{center:Point3;normal:Point3;radiusMm:number};
  targetLayerId?:string; sensitivityConfig?:ApproachSensitivityConfig;
  constraints:{maxLengthMm?:number;minClearanceMm?:number;excludedLayerIds?:string[]};
  assessmentStatus:'current'|'stale'|'unassessed';
  measurements?:{lengthMm:number;orientation:{azimuthDeg:number;elevationDeg:number};structures:ApproachStructure[];
    coverage:Record<string,unknown>;targets:ApproachTargetMeasurement[];normalTraversal:Record<string,unknown>&{coverage?:ApproachCoverage;acquisitionCoverage?:ApproachCoverage};
    sensitivity?:ApproachSensitivity;
    accessAssessment?:CorticalAccessAssessment;
    status:'known-conflict'|'incomplete'|'no-detected-conflict';unknowns:string[];assumptions:string[];constraintFailures?:string[];[key:string]:unknown};
  [key:string]:unknown;
}
export interface ApproachResponse {
  rehearsalOptions?:EnvironmentOptions;
  snapshotId:string;approaches:NativeApproach[];selectedId?:string|null;targetOptions?:ApproachTargetOption[];
  activeTargetLayerId?:string|null;targetSelectionIssue?:string|null;
  proposalReport?:ApproachProposalReport|null;inputComparison?:ApproachInputComparisonReport|null;
}
export const approachStatus=(a:NativeApproach)=>a.assessmentStatus!=='current'?a.assessmentStatus:a.measurements?.status??'unassessed';

/** Separate observations from evidence gaps and user-declared constraint failures. */
export function approachAssessment(a:NativeApproach){
  const m=a.assessmentStatus==='current'?a.measurements:undefined;
  const structures=m?.structures??[],assessed=structures.filter(s=>s.available),intersections=assessed.filter(s=>s.confirmedIntersection);
  const observed=intersections.length?'intersection':assessed.length?'no-intersection':'unavailable';
  const observedLabel=observed==='intersection'?'Detected anatomical intersection':observed==='no-intersection'?'No detected intersection in assessed anatomy':'Geometric measurement unavailable';
  const unavailable=structures.filter(s=>!s.available),unresolved=assessed.filter(s=>s.intersectsOrUnresolved&&!s.confirmedIntersection);
  const limitations=[...(m?.unknowns??[])];
  if(m&&!assessed.length)limitations.push('No supplied anatomical structure could be measured');
  for(const s of unavailable)if(!limitations.some(text=>text.includes(s.layerId)||text.includes(s.label)))limitations.push(`${s.label}: measurement unavailable`);
  for(const s of unresolved)if(!limitations.some(text=>text.includes(s.layerId)||text.includes(s.label)))limitations.push(`${s.label}: conservative contact unresolved`);
  if(m?.coverage?.withinReferenceEnvelope===false&&!limitations.some(text=>text.includes('outside reference')))limitations.push('Corridor envelope extends outside the reference image field');
  const incomplete=!!m&&(m.status==='incomplete'||limitations.length>0||unavailable.length>0||unresolved.length>0);
  const completeness=a.assessmentStatus==='stale'?'stale':!m?'unavailable':incomplete?'incomplete':'complete';
  const completenessLabel=completeness==='stale'?'Reassessment required':completeness==='unavailable'?'Assessment unavailable':incomplete?'Incomplete evidence':'Recorded evidence assessed';
  const finite=(value:unknown):value is number=>typeof value==='number'&&Number.isFinite(value);
  const bounds=assessed.map(s=>s.clearanceLowerBoundMm).filter(finite);
  const targets=m?.targets??[];
  const target=targets.length===1?targets[0]:undefined;
  const routeTargetReached=target?.available&&finite(target.coveredVoxelCenterVolumeMm3)
    &&target.coverage?.wholeEnvelopeWithinField&&target.acquisitionCoverage?.wholeEnvelopeWithinField
    ?target.coveredVoxelCenterVolumeMm3:null;
  return {observed,observedLabel,completeness,completenessLabel,assessedCount:assessed.length,totalCount:structures.length,
    intersections,unavailable,unresolved,limitations:limitations.map(text=>structures.reduce((result,s)=>result.replaceAll(`${s.layerId}:`,`${s.label}:`),text)),
    constraintFailures:m?.constraintFailures??[],minimumBoundMm:bounds.length?Math.min(...bounds):null,
    targetReachedMm3:routeTargetReached,
    traversalMm3:m?.normalTraversal?.available&&finite(m.normalTraversal.conservativeVolumeUpperBoundMm3)?m.normalTraversal.conservativeVolumeUpperBoundMm3:null};
}

export function approachRoutes(approaches:NativeApproach[],selectedId:string|null):ViewerRoute[]{const displayed=approaches.slice(0,3),selected=approaches.find(a=>a.id===selectedId);if(selected&&!displayed.includes(selected))displayed[2]=selected;return displayed.map((a,index)=>({route_id:a.id,comparisonSlot:(['A','B','C'] as const)[index],scope:'geometric-approach',label:a.name,selected:a.id===selectedId,entry_mm:a.entry,target_mm:a.target,
  tool:{working_length_mm:Math.hypot(...a.target.map((v,i)=>v-a.entry[i])),tip_length_mm:0,shaft_radius_mm:a.radiusMm,tip_radius_mm:a.radiusMm},
  window:a.entryPatch?{center_mm:a.entryPatch.center,normal_inward:a.entryPatch.normal,radius_mm:a.entryPatch.radiusMm}:undefined}));}

/** Round down so a displayed lower bound never overstates source clearance. */
export const lowerBoundText=(value:unknown)=>typeof value==='number'&&Number.isFinite(value)?(Math.floor(value*10)/10).toFixed(1):'—';

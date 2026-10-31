import type {Point3} from './viewer/coordinates';
import type {AnatomicalEncounter,TwoToolComparison,TwoToolSummary} from './two-tool-data';

export interface ExecutionOffsetOutcome {
  scenarioId:string;translationRasMm:Point3;completeExecution:boolean;withdrawalComplete:boolean;
  actionProgram:Array<{pose?:{position:Point3}}>;
  failure:{actionIndex:number;operation:string;phase:string|null;reason:string}|null;
  summary:TwoToolSummary;anatomicalEncounter:AnatomicalEncounter;meetsTask:boolean;violations:string[];
}
export interface ExecutionSensitivityStudy {
  schema:string;studySeal:string;comparisonReportSeal:string;
  assessment:{assessmentSeal:string;controls:{offsetMm:number};scenarioCount:number;scope:string;
    policies:{nominal:string;fixedMargin:string;allDeclaredScenarios:string};
    candidates:Array<{id:string;outcomes:ExecutionOffsetOutcome[]}>;
    preferenceChanges:Array<{scenarioId:string;selectedId:string}>;
    calibratedUncertainty:{supported:false;reason:string};
    fixedMarginDefinition:{nonTargetClearanceMm:number;contextMarginMm:number;protectedExtraMm:number}};
}
export interface ExecutionSensitivityAttachment {revision:number;comparisonRevision:number;study:ExecutionSensitivityStudy}
export function currentSensitivity(report:TwoToolComparison,offset:number):ExecutionSensitivityStudy|null {
  const study=report.executionSensitivity?.study;
  return study&&Number.isFinite(offset)&&study.comparisonReportSeal===report.reportSeal&&study.assessment.controls.offsetMm===offset?study:null;
}
export function failedCommandPoint(outcome:ExecutionOffsetOutcome):Point3|null {
  const point=outcome.failure?outcome.actionProgram[outcome.failure.actionIndex]?.pose?.position:null;
  return point?.length===3&&point.every(Number.isFinite)?[...point]:null;
}
export function executionIssueText(outcome:ExecutionOffsetOutcome):string {
  const labels:Record<string,string>={INCOMPLETE_EXECUTION:'Sequence did not complete',INCOMPLETE_WITHDRAWAL:'Withdrawal did not complete',ANATOMICAL_CONTEXT_LIMIT:'Context limit exceeded',PROGRESS_THRESHOLD:'Progress below the declared threshold'};
  return outcome.meetsTask?'Task met':outcome.violations.map(v=>labels[v]??v.replaceAll('_',' ').toLowerCase()).join('; ');
}
export function executionFailureText(reason:string):string {
  for(const [code,text] of [['SHAFT_COLLISION','The shaft intersects retained tissue.'],['RETAINED_TISSUE_COLLISION','The commanded tool intersects retained tissue.'],['TOOL_COLLISION','The instruments intersect.'],['WITHDRAWAL_PATH_BLOCKED','The requested withdrawal is blocked.'],['UNSUPPORTED_MATERIAL','The tool meets an unsupported material.'],['PROTECTED_STRUCTURE','The declared protected-structure constraint fails.']]){
    if(reason.includes(code))return text;
  }
  return reason.replace('STRATEGY_SENSITIVITY: ','').replace('execution refused: ','');
}

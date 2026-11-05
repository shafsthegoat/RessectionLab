import {transformPoint} from './viewer/coordinates.ts';
import type {Point3,Shape3,Affine} from './viewer/coordinates';
import type {TwoToolComparison,TwoToolSummary} from './two-tool-data';

export interface LabelEncounter {
  nearMm3:number;newRemovedContextMm3:number;newContextWorkMm3:number;unknownEncounteredMm3:number;
  baselineRemovedContextMm3:number;nearPositionsMm:Point3[];withinLimit:boolean;
  newContextWorkCells:number[];unknownEncounteredCells:number[];
}
export interface AnatomicalSensitivityStudy {
  schema:string;studySeal:string;comparisonReportSeal:string;
  assessment:{controls:{contextVoxelShift:number};scenarioCount:number;scope:string;
    scenarioConfiguration:{maximumTranslationMm:number;scenarios:Array<{id:string;voxelShift:Point3;translationRasMm:Point3;outOfFieldContextCells:number}>};
    candidates:Array<{id:string;completeExecution:boolean;withdrawalComplete:boolean;summary:TwoToolSummary;
      outcomes:Array<{scenarioId:string;encounter:LabelEncounter;violations:string[];meetsTask:boolean}>}>;
    policies:{nominal:string;fixedContextMargin:string;allDeclaredScenarios:string};
    fixedMarginDefinition:{contextMarginMm:number;isGuarantee:false;scope:string};
    preferenceChanges:Array<{scenarioId:string;selectedId:string}>;
    calibratedUncertainty:{supported:false;reason:string};};
}
export interface AnatomicalSensitivityAttachment {revision:number;comparisonRevision:number;study:AnatomicalSensitivityStudy}
export function currentAnatomicalSensitivity(report:TwoToolComparison,step:number):AnatomicalSensitivityStudy|null {
  const study=report.anatomicalSensitivity?.study;
  return study&&Number.isInteger(step)&&step>=0&&step<=2&&study.schema==='resectionlab.two-tool-context-label-sensitivity.v1'&&
    study.comparisonReportSeal===report.reportSeal&&study.assessment.controls.contextVoxelShift===step?study:null;
}
export function anatomicalIssueText(violations:string[]):string {
  const labels:Record<string,string>={ANATOMICAL_CONTEXT_LIMIT:'Context proximity limit exceeded',NEW_CONTEXT_MANIPULATION:'New CUSA work in hypothetical context',UNKNOWN_CONTEXT_COVERAGE:'Context coverage unknown',PROGRESS_THRESHOLD:'Progress below the declared threshold'};
  return violations.length?violations.map(v=>labels[v]??v.replaceAll('_',' ').toLowerCase()).join('; '):'Declared task met';
}
export function contextCellPoint(cell:number|undefined,scene:{shape:Shape3;affine:Affine}):Point3|null {
  const {shape,affine}=scene;
  if(cell===undefined||!Number.isSafeInteger(cell)||cell<0||shape.some(n=>!Number.isSafeInteger(n)||n<=0)||cell>=shape[0]*shape[1]*shape[2]||
    affine.length!==4||affine.some(row=>row.length!==4||row.some(n=>!Number.isFinite(n))))return null;
  const voxel:Point3=[Math.floor(cell/(shape[1]*shape[2])),Math.floor(cell/shape[2])%shape[1],cell%shape[2]];
  const point=transformPoint(affine,voxel);
  return point.every(Number.isFinite)?point:null;
}

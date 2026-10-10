import type { CasePayload, Mat4, ViewerCase } from './types';
export type DisplayModality = 'T1'|'T1CE'|'T2'|'FLAIR'|'CT'|'CTA'|'TOF-MRA'|'MRA'|'SWI'|'other-3D-scalar';
export type AnnotationKind = 'none'|'source-provided'|'estimated';
export interface ImportDisplaySeriesRequest { caseHash:string; modality:DisplayModality; annotationKind:AnnotationKind }
/** Extends a case workspace, not the patient's planning model or actor inputs. */
export interface DisplaySeriesPayload {
  schema:'workspace-display-series-v1'; seriesId:string; referenceCaseHash:string;
  referenceFrameHash:string; sourceFrameHash:string;
  association:{kind:'explicit_user_selected';samePersonVerified:false;sameTimeVerified:false};
  registration:{status:'unreviewed';reason:string;sourceToReferenceRasMm:Mat4|null;registrationHash:string|null;overlayPermitted:false};
  scope:'display-only-native-grid'; planningEligible:false; evaluationEligible:false;
  modality:DisplayModality;modalityOrigin:'user_declared';annotationKind:AnnotationKind;
  annotationCoverage:string;acquisitionDatetime:null;expandedBytes:number;volume:CasePayload;
}
export interface DisplaySeriesView { descriptor:DisplaySeriesPayload; volume:ViewerCase }
export interface DisplayImagingApi {
  importDisplaySeries?(request:ImportDisplaySeriesRequest):Promise<DisplaySeriesPayload|null>;
}

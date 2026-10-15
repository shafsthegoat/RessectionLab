import type {Binding, NativeScanState} from './native-anatomy-data';

export function studyTimepoint(b:Binding):string {
  return b.document.study_timepoint??b.document.series[0]?.timepoint??'preoperative';
}
export function studyLabel(value:string):string {
  return value==='postoperative'?'Post-op':value==='preoperative'?'Pre-op':value;
}
export function scanLabel(series:{id:string;modality:string;label?:string}):string {
  return series.label??(/^[a-f\d-]{30,}$/i.test(series.id)?series.modality:series.id.toUpperCase().replaceAll('-',' '));
}
/** Scan contrast/camera stays per scan; selected anatomy belongs to the study. */
export function nextScanState(current:NativeScanState, saved:NativeScanState):NativeScanState {
  return {...saved,visible:{...current.visible},settings:{...saved.settings,
    layerColor:current.settings.layerColor,layerOpacity:current.settings.layerOpacity}};
}

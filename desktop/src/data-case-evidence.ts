/** Keep acquired backgrounds distinct from invented primary image signal. */
export function caseImageEvidence(metadata:Record<string,unknown>|undefined){
  const synthetic=!!metadata?.is_synthetic;
  const data=metadata?.data_case as Record<string,unknown>|undefined;
  const observations=data?.observations as Record<string,unknown>|undefined;
  const onAcquiredHost=synthetic&&['synthetic_scenario_on_acquired_host','synthetic_scenario_on_acquired_healthy_host'].includes(String(data?.classification));
  const geometryOnly=onAcquiredHost&&((observations?.primary_image??observations?.primary_T1)==='baseline_acquired_background_geometry_only'||
    observations?.T1==='acquired_unmodified_baseline'&&observations?.tumor==='synthetic_geometry_only');
  return {synthetic,onAcquiredHost,geometryOnly,generatedImage:synthetic&&!geometryOnly};
}

/** Display the exposure assumption for the case's declared research task. */
export function conditionalRehearsalNotice(metadata:Record<string,unknown>|undefined){
  const rehearsal=metadata?.integrated_rehearsal as Record<string,unknown>|undefined;
  if(rehearsal?.stage==='conditional_post_scalp_exposure')return 'Conditional research stage: scalp exposure is assumed; dural opening is not simulated. The acquired CTA remains a baseline reference. Intact-head access and clinical safety are unsupported.';
  if(rehearsal?.stage==='conditional_exposed_brain_geometry')return 'Conditional exposed-brain geometry task: scalp exposure, a local skull opening and dural opening are assumed and not simulated. T1/MRA remain baseline references; skull CT is unavailable. Intact-head access and clinical navigation are unsupported.';
  return undefined;
}

/** Some older native layer records omit provenance, while their explicit
 * component contract retains the supplied-class union. Do not infer it from a
 * synthetic case flag or a layer name alone. Explicit layer evidence wins. */
export function caseLayerEvidence(metadata:Record<string,unknown>|undefined,name:string){
  const layers=metadata?.anatomy_layers as Record<string,Record<string,unknown>>|undefined;
  const record=layers?.[name];
  if(record?.provenance)return record;
  const key=({supplied_arterial_annotation_union:'arteries',supplied_venous_sinus_annotation_union:'veins_sinuses'} as Record<string,string>)[name];
  const data=metadata?.data_case as Record<string,unknown>|undefined;
  const components=data?.components as Record<string,Record<string,unknown>>|undefined;
  const component=key?components?.[key]:undefined;
  return component?.provenance==='supplied_annotation'?{...record,provenance:'source-provided',
    coverage:`Derived union of supplied source classes; incomplete coverage. ${component.coverage??''}`} : record;
}

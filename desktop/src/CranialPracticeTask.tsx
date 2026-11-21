import type {SurgicalResponse} from './surgical-types';

// The sealed parent binds this generated exercise to its source, scene,
// instrument catalog, original target, model, predecessor actions and boundary.
const parentSeal='sha256:b54fff661810b8798c961181d57ac699d4759a3e4465949426adcfeec70afbe3';
const stageContract='sha256:00e861fa8802c8dacd44ab13d0f55402ab94b335c55f26a3b5402da2aeaa820d';
const finite=(n:unknown):n is number=>typeof n==='number'&&Number.isFinite(n);
const equal=(n:unknown,expected:number)=>finite(n)&&Math.abs(n-expected)<1e-8;
const volume=(n:unknown)=>finite(n)?n.toLocaleString(undefined,{maximumFractionDigits:3})+' mm³':'Unknown';

export function CranialPracticeTask({result}:{result:SurgicalResponse}) {
  const parent=result.practiceEpisode?.parent;
  if(!parent||parent.checkpointSeal!==parentSeal||parent.position!==99||parent.caseHash!==result.caseHash||
    parent.sceneId!==result.view.scene.sceneId||result.practiceEpisode?.episodeId!==result.branchId||
    result.playback||result.position<parent.position)return null;
  const {view}=result,stage=view.cranialStages,membrane=stage?.membrane,m=view.metrics;
  const current=stage?.stateId===view.stateId&&stage.sceneId===parent.sceneId&&stage.contractHash===stageContract&&
    stage.assessmentCurrent&&stage.bonePrefixVerified&&stage.boundaryCutComplete;
  const extracted=membrane?.removedMembraneMm3;
  const extraRemoved=finite(m.nonTargetRemovedMm3)&&finite(extracted)?m.nonTargetRemovedMm3-285-extracted:null;
  const target=view.originalTargetResidualMm3, debris=m.debrisMm3, collected=m.debrisRemovedMm3, fluid=m.fluidPresentMm3;
  let objective:string;
  if(!current||!membrane||!equal(result.practiceMetrics?.originalTargetVolumeMm3,27)||
    ![extracted,extraRemoved,target,m.targetRemovedMm3,debris,collected,fluid].every(finite))
    objective='Current task evidence unavailable';
  else if(!membrane.retainedRimIntact||!equal(extraRemoved,0))
    objective='Review retained rim and non-target removal';
  else if(!stage.patchExtracted)
    objective=stage.patchHeld?'Transport the held patch outside and extract it':'Grasp and extract the cut membrane patch';
  else if(!stage.workEntrySupported||!equal(extracted,49)||!membrane.actualNativeExtractionEvents)
    objective='Review native membrane clearance';
  else if(!equal(target,0)||!equal(m.targetRemovedMm3,27)) objective='Aspirate the remaining original target';
  else if(!equal(debris,0)||!equal(collected,236)) objective='Collect the remaining kerf debris';
  else if(!equal(fluid,0)) objective='Clear the remaining modeled fluid';
  else if(!stage.wholeExitComplete) objective='Release and withdraw both instruments';
  else objective='Modeled endpoint reached · save and verify replay';
  return <details className="practice-task" aria-label="Practice task" data-task-state={view.stateId}>
    <summary><strong>Practice task</strong> · {objective}</summary>
    <p>Generated square cranial control, boundary 99. The bone opening and membrane perimeter cut are complete.</p>
    <ul>
      <li>Extract the 49 mm³ patch and preserve the rim. Extracted {volume(current?extracted:null)}. Rim {current&&membrane?(membrane.retainedRimIntact?'intact':'disrupted'):'unknown'}.</li>
      <li>Remove the 27 mm³ original target. Remaining {volume(target)}. Additional non-target removal beyond the patch {volume(current?extraRemoved:null)}.</li>
      <li>Collect 236 mm³ of bone debris. Collected {volume(collected)}, remaining {volume(debris)}. Fluid remaining {volume(fluid)}. Both residuals must reach zero.</li>
      <li>Release and withdraw both hands, save the child and verify its reopened endpoint. Native exit {current?(stage.wholeExitComplete?'complete':'pending'):'unknown'}.</li>
    </ul>
    <p className="surgical-note">Patient protection coverage and neurological outcomes are unknown.</p>
  </details>;
}

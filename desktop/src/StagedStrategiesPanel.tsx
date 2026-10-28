import {useEffect,useState} from 'react';
import type {SurgicalController} from './use-surgical';
import {stagedGenerationEligibility,stagedStrategyDisplay} from './staged-strategy-data';
import {cusaContactModel} from './cusa-research';

const mm=(value:number)=>value.toFixed(1);

export function StagedStrategiesPanel({c,artifact,pause,play}:{c:SurgicalController;
  artifact:unknown;pause:()=>void;play:(branchId:string)=>void}){
  const [acknowledged,setAcknowledged]=useState(false);
  const [narrowAccess,setNarrowAccess]=useState(false);
  const [exploreFaces,setExploreFaces]=useState(false);
  const [cumulative,setCumulative]=useState(false);
  useEffect(()=>{setAcknowledged(false);setNarrowAccess(false);setExploreFaces(false);setCumulative(false);},[c.result?.caseHash,c.result?.view.scene.sceneId]);
  const record=artifact&&typeof artifact==='object'?artifact as Record<string,unknown>:null;
  const saved=!!record&&(Object.hasOwn(record,'stagedPlanning')||Object.hasOwn(record,'stagedSummary'));
  const live=c.result?.stagedPlanning!=null;
  const paired=c.result?.planning?.task==='anatomy-access'&&
    c.result?.sectorPlanning?.task==='anatomy-strategy';
  if(!saved&&!live&&!paired)return null;
  const data=stagedStrategyDisplay(artifact,c.result);
  const eligibility=stagedGenerationEligibility(artifact,c.result);
  if(!data){
    if(saved||live)return <p role="alert" className="surgical-warning">Staged strategy evidence could not be bound to this scene and its saved native branches.</p>;
    return <details className="research-planning staged-strategies"><summary>Staged source strategies</summary>
      <p className="surgical-warning">This structural source lacks skull opening, complete vessels and functional mapping. Native CUSA contact, withdrawal and tissue effects remain modeled research behavior.</p>
      <p>Run bounded, deterministic staged work from the common intact source. Source geometry proposes tiles; native previews, commits and checked exits decide which work is retained. The larger 240-action objective and the original 48-action sector gate are reported separately. No actor or optimizer runs.</p>
      {!cusaContactModel(c.result?.view)&&<label>Resection sequence<select aria-label="Staged resection sequence" disabled={c.pending}
        value={cumulative?'reentry':'local'} onChange={event=>setCumulative(event.target.value==='reentry')}>
        <option value="local">Local staged work</option><option value="reentry">Planned external re-entry</option>
      </select></label>}
      {cumulative&&<p>Plans complete entries through each selected surface, with tool size changes after checked withdrawal. Initial access and all later removal count toward the displayed effects. Source contact predicts a work order; native checks determine which stages complete.</p>}
      {!cusaContactModel(c.result?.view)&&<label>Initial CUSA access configuration<select aria-label="Staged initial access configuration" disabled={c.pending||cumulative}
        value={cumulative?'automatic':narrowAccess?'narrow':'saved'} onChange={event=>setNarrowAccess(event.target.value==='narrow')}>
        {cumulative&&<option value="automatic">Selected with the re-entry sequence</option>}
        <option value="saved">Saved selected tip radius</option><option value="narrow">Fixed narrow tip · 0.6 mm</option>
      </select><small>A narrow initial tip can reduce approach removal while limiting resection progress. It is configured outside tissue; native checks and all task gates remain unchanged.</small></label>}
      <label><input aria-label="Explore additional entry surfaces" type="checkbox" checked={exploreFaces&&!cumulative}
        disabled={c.pending||cumulative} onChange={event=>setExploreFaces(event.target.checked)}/>
        Explore additional entry surfaces</label>
      {exploreFaces&&!cumulative&&<p>Includes entry surfaces omitted by the saved corridor inventory. The finite search proposes candidates; native execution and checked withdrawal determine feasibility. Saved measured routes remain available.</p>}
      <label><input aria-label="Acknowledge staged source assumptions" type="checkbox" checked={acknowledged}
        disabled={c.pending} onChange={event=>setAcknowledged(event.target.checked)}/>
        I acknowledge the missing anatomy and research-only source assumptions for this staged comparison.</label>
      <button disabled={c.pending||!acknowledged||!eligibility.ready} onClick={()=>{
        pause();void c.run('plan',{opponent:'anatomy-staged-strategy',acknowledgeResearchUnknowns:true,
          ...(cumulative?{cumulativeReentry:true}:{...(exploreFaces?{exploreExteriorFaces:true}:{}),
          ...(narrowAccess?{initialAccessRadiusMm:.6}:{})})});
      }}>Generate staged source strategies</button>
      {!eligibility.ready&&<p role="status">{eligibility.reason}</p>}
    </details>;
  }
  const disabled=c.pending||c.result?.verification?.status==='unverified-import'||c.result?.readOnly;
  const inspect=(branchId:string,position:number)=>{
    pause();void c.run('selectBranch',{branchId,position});
  };
  const passes=data.alternatives.filter(row=>row.outcome.taskGateMet).length;
  const legacyPasses=data.alternatives.filter(row=>row.outcome.legacyTask.taskGateMet).length;
  return <details className="research-planning staged-strategies"><summary>{data.comparisonScope==='same-route-configurations'?'Access configuration comparison':'Staged source strategies'} · {passes} / {data.alternatives.length} larger task gates passed</summary>
    <p className="surgical-warning">Deterministic research comparison on partial anatomy. Skull, complete vessels and functional mapping are unavailable; these modeled effects are not clinical risk estimates.</p>
    {data.cusaModel==='legacy-instantaneous-with-v2-mechanics'&&<p className="surgical-note">This scene uses legacy instantaneous CUSA removal with V2 tissue mechanics. It does not establish contact-capacity CUSA v3 or device fidelity.</p>}
    <p>{passes} / {data.alternatives.length} larger staged task gates passed. The original 48-action two-sector gate passed in {legacyPasses} / {data.alternatives.length} branches; that gate is a separate measure.</p>
    {data.controllerMode==='cumulative-reentry'&&<p>Planned external re-entry: complete source-selected strokes with checked withdrawal and external configuration between stages. Playback shows their recorded native actions; it does not run new inference.</p>}
    {data.comparisonScope==='same-route-configurations'&&<p>These are initial tool configurations on the same access geometry, not distinct surgical approaches. Each row shows its whole approach and resection tradeoffs.</p>}
    <p>Earlier sector strategies can change working radius while inserted. Staged work keeps tip radius and depth fixed until a checked external exchange, so their tool-configuration assumptions differ.</p>
    <p>{data.physicalCalls} / {data.maxPhysicalCalls} physical calls for staged planning · {data.wallSeconds.toFixed(1)} wall seconds · zero optimizer updates. Failed work probes and checked exits are included in the planning cost.</p>
    {data.exportRecovery&&<p className="surgical-warning">Export required recovery. {data.exportRecovery.additionalReplayCalls===null?'The additional native replay cost is unavailable.':`${data.exportRecovery.additionalReplayCalls} additional native replay calls; ${data.physicalCalls+data.exportRecovery.additionalReplayCalls} observed calls including planning.`} The full execution-and-export wall-time requirement {data.exportRecovery.fullRunWithinWallCap===null?'is unverified':data.exportRecovery.fullRunWithinWallCap?'passed':'did not pass'}. Recovery is recorded separately from planning.</p>}
    {data.notes.length>0&&<details><summary>Recorded scope and execution limits</summary><ul>{data.notes.map((note,index)=><li key={index}>{note}</li>)}</ul></details>}
    {data.sourceBranchId&&<button disabled={disabled}
      data-staged-branch={data.sourceBranchId} data-staged-position={0}
      onClick={()=>inspect(data.sourceBranchId!,0)}>Inspect common intact source</button>}
    <section aria-label="Staged whole approach summaries">{data.alternatives.map(row=><article key={row.branchId}>
      <h4>{row.sourceMethod} · {row.hand} hand · {row.status}</h4>
      <p>Whole target {mm(row.outcome.wholeTargetRemovedMm3)} mm³ · whole non-target {mm(row.outcome.wholeNonTargetRemovedMm3)} mm³.</p>
      <p>Residual target {mm(row.outcome.residualTargetMm3)} mm³ · {(100*row.outcome.sourceTargetFractionRemoved).toFixed(2)}% of source target removed.</p>
      {row.executionMetrics&&<p>Recorded whole approach: {mm(row.executionMetrics.simulatedSeconds)} simulated seconds · {mm(row.executionMetrics.instrumentTravelMm)} mm instrument travel · {row.executionMetrics.workingConfigurations} contributing tool configurations.</p>}
      <p>Post-access target: {mm(row.outcome.legacyTask.additionalTargetMm3)} mm³ total; {mm(row.outcome.legacyTask.additionalTargetMm3-row.unassignedPostAccessTargetMm3)} mm³ in admitted work stages and {mm(row.unassignedPostAccessTargetMm3)} mm³ in other native effects such as exit, re-entry or recovery.</p>
      <p>Checked cleanup {row.outcome.cleanupComplete?'complete':'incomplete'} · independent native replay exact · {row.resumeVerified?'checkpoint replay verified':'checkpoint replay not verified'}.</p>
      <p>{row.stages.length} admitted work stages · {row.workAttempts} work attempts · {row.exitRefusals} refused checked exits · {row.nativeActions} accepted native actions.</p>
      <div className="surgical-actions"><button disabled={disabled}
        data-staged-branch={row.branchId} data-staged-position={0}
        onClick={()=>inspect(row.branchId,0)}>Inspect source start</button>
        <button disabled={disabled} data-staged-branch={row.branchId}
          data-staged-position={row.nativeActions}
          onClick={()=>inspect(row.branchId,row.nativeActions)}>Inspect endpoint</button>
        <button disabled={disabled||row.nativeActions===0} data-staged-play={row.branchId}
          onClick={()=>play(row.branchId)}>Play recorded native actions</button></div>
      <details><summary>Stages and configuration changes</summary>
        {row.stages.length?<ol>{row.stages.map(stage=><li key={`${stage.stage}:${stage.tileId}`}
          data-staged-execution={stage.executionMode??'requested_motion'}>
          Stage {stage.stage+1} · {stage.configuration} · tile {stage.tileId} · +{mm(stage.targetIncrementMm3)} mm³ target.
          {stage.executionMode==='checked_collision_prefix'&&<p>A shortened command passed fresh native checks after a shaft refusal. The requested waypoint {stage.requestedPoseReached?'was reached':'remains incomplete'}; subsequent work uses the actual accepted pose.</p>}
          <button disabled={disabled} data-staged-branch={row.branchId}
            data-staged-position={stage.nativeActions}
            onClick={()=>inspect(row.branchId,stage.nativeActions)}>
            Inspect recorded action {stage.nativeActions}</button>
          <p>Checked exit from this stage: {stage.checkedExitActions} actions · state {stage.checkedExitStateId}. The saved branch continues through its retained native history.</p>
        </li>)}</ol>:<p>No work stage was admitted.</p>}
      </details>
      <details><summary>Task gates and recorded limits</summary>
        <p>Larger task: {row.outcome.taskGateMet?'passed':'unpassed'}{row.outcome.failedGates.length?` · failed ${row.outcome.failedGates.join(', ')}`:''}. Stop reason: {row.stopReason.replaceAll('_',' ')}.</p>
        <p>Zones: {Object.entries(row.outcome.zoneTargetRemovedMm3).map(([zone,value])=>`${zone}: ${mm(value)}`).join(' · ')} mm³.</p>
        <p>Configurations: {Object.entries(row.outcome.configurationTargetRemovedMm3).map(([configuration,value])=>`${configuration}: ${mm(value)}`).join(' · ')||'none'} mm³.</p>
        <p>Original 48-action gate: {row.outcome.legacyTask.taskGateMet?'passed':'unpassed'} · post-access target +{mm(row.outcome.legacyTask.additionalTargetMm3)}, non-target +{mm(row.outcome.legacyTask.additionalNonTargetMm3)} mm³ · near {mm(row.outcome.legacyTask.perSectorRemovedMm3.near)}, far {mm(row.outcome.legacyTask.perSectorRemovedMm3.far)} mm³.</p>
        {row.outcome.legacyTask.orderPassed!==undefined&&<p>Original far-then-near order: {row.outcome.legacyTask.orderPassed?'passed':'unpassed'} · observed {row.outcome.legacyTask.observedFirstProgressOrder?.join(' → ')||'no sector progress'}{row.outcome.legacyTask.firstProgressSimultaneous?' · simultaneous first stroke':''}.</p>}
      </details>
    </article>)}</section>
    {!data.alternatives.length&&<p>No source-derived route reached an admitted staged alternative.</p>}
    <details><summary>Declared larger objective and source identity</summary>
      <p>Whole target ≥{mm(data.task.minimumWholeTargetMm3)} mm³. At least 3 source zones each ≥{mm(data.task.minimumTargetPerZoneMm3)} mm³ and 2 tool configurations each ≥{mm(data.task.minimumTargetPerConfigurationMm3)} mm³.</p>
      <p>Whole non-target ≤200 mm³ and at most 0.2 times whole target removal; continuation non-target ≤20 mm³; accepted actions ≤240. Checked final withdrawal and exact replay are required.</p>
      <p>Case {data.caseHash} · scene {data.sceneId} · intact state {data.sourceStateId} · comparison {data.comparisonSeal}.</p>
    </details>
  </details>;
}

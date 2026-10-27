import {useEffect,useState} from 'react';
import type {SurgicalController} from './use-surgical';
import {jointGenerationEligibility,jointStrategyDisplay} from './joint-strategy-data';
import type {JointBranchRow} from './joint-strategy-data';

export function JointStrategiesPanel({c,artifact,pause}:{c:SurgicalController;artifact:unknown;pause:()=>void}){
  const [classicOpen,setClassicOpen]=useState(false);
  const [acknowledged,setAcknowledged]=useState(false);
  useEffect(()=>{setAcknowledged(false);setClassicOpen(false);},
    [c.result?.caseHash,c.result?.view.scene.sceneId]);
  const hasSavedJoint=!!artifact&&typeof artifact==='object'&&'jointPlanning' in artifact;
  const hasLiveJoint=c.result?.jointPlanning!=null;
  const classicContext=c.result?.planning?.task==='anatomy-access'&&
    c.result?.sectorPlanning?.task==='anatomy-strategy';
  if(!hasSavedJoint&&!hasLiveJoint&&!classicContext)return null;
  const data=jointStrategyDisplay(artifact,c.result);
  const eligibility=jointGenerationEligibility(artifact,c.result);
  if(!data){
    if(hasSavedJoint||hasLiveJoint)return <p role="alert" className="surgical-warning">Joint strategy evidence could not be bound to this scene and native branch inventory.</p>;
    return <details className="research-planning joint-strategies"><summary>Deterministic joint strategies</summary>
      <p className="surgical-warning">The source has no modeled skull opening. Brain corridors use mapped source tissue as a partial estimate; complete vessels, functional anatomy, and clinical feasibility remain unknown.</p>
      <p>One source-selected route per native axis is executed from the common intact state, followed by one fixed far-first left-CUSA sector strategy. Native work is limited to 48 actions per route, 1500 physical calls, and 120 seconds. No optimizer runs.</p>
      <label><input aria-label="Acknowledge joint source assumptions" type="checkbox" checked={acknowledged} disabled={c.pending} onChange={event=>setAcknowledged(event.target.checked)}/>I acknowledge the partial skull and mapped-brain source assumptions for this bounded comparison.</label>
      <button disabled={c.pending||!acknowledged||!eligibility.ready} onClick={()=>{
        pause();void c.run('plan',{opponent:'anatomy-joint-strategy',
          acknowledgeResearchUnknowns:true,allowModeledBrainCorridor:true});
      }}>Generate joint source strategies</button>
      {!eligibility.ready&&<p role="status">{eligibility.reason}</p>}
    </details>;
  }
  const disabled=c.pending||c.result?.verification?.status==='unverified-import';
  const inspect=(row:JointBranchRow,position:number)=>{
    pause();void c.run('selectBranch',{branchId:row.branchId,position});
  };
  const branchActions=(row:JointBranchRow)=>
    <span className="surgical-actions" key={row.branchId}>
      <button disabled={disabled} data-joint-branch={row.branchId} data-joint-position={row.start}
        onClick={()=>inspect(row,row.start)}>{row.label} · recorded start</button>
      <button disabled={disabled} data-joint-branch={row.branchId} data-joint-position={row.end}
        onClick={()=>inspect(row,row.end)}>Inspect endpoint</button>
    </span>;
  const amount=(value:number|null)=>value===null?'Unavailable':value.toFixed(1);
  const entry=(row:JointBranchRow)=>row.entryRas?
    row.entryRas.map(value=>value.toFixed(1)).join(', '):'Unavailable';
  const withdrawal=(row:JointBranchRow)=>{
    if(!row.withdrawal)return 'Not recorded';
    const receipt=row.withdrawal;
    return `${receipt.fastRefusal?`Fast ${receipt.fastRefusal.replaceAll('_',' ')}; `:''}`+
      `${receipt.slowStatus??'slow status unavailable'}`+
      `${receipt.slowSeconds===null?'':` (${receipt.slowSeconds.toFixed(1)} s)`}`;
  };
  const tradeoffRow=(row:JointBranchRow,label:string)=>
    <tr key={`${label}:${row.branchId}`} aria-selected={row.branchId===c.result?.branchId}>
      <th>{label}<small>{row.method}</small></th>
      <td>{entry(row)}</td><td>{row.hand??'Unavailable'} / {row.order??'Unavailable'}</td>
      <td>{row.status==='passed'?'Passed':'Unpassed'} / {row.cleanupComplete?'Complete':'Incomplete'}</td>
      <td>{amount(row.wholeTargetMm3)}</td><td>{amount(row.wholeNonTargetMm3)}</td>
      <td>{amount(row.residualTargetMm3)}{row.sourceFractionRemoved===null?'':` · ${(100*row.sourceFractionRemoved).toFixed(2)}% of source removed`}</td>
      <td>{amount(row.additionalTargetMm3)} / {amount(row.additionalNonTargetMm3)}</td>
      <td>{amount(row.simulatedSeconds)}</td>
      <td>{row.optionPhysicalCalls===null?'Unavailable':row.optionPhysicalCalls} / {row.end}</td>
      <td>{withdrawal(row)}</td><td>{branchActions(row)}</td>
    </tr>;
  const passedClassic=data.classicSectors.filter(row=>row.status==='passed');
  const measuredClassic=data.classicSectors.filter(row=>row.wholeTargetMm3!==null);
  const passedMeasured=measuredClassic.filter(row=>row.status==='passed');
  const classicPool=passedMeasured.length?passedMeasured:measuredClassic;
  const classicReference=[...classicPool].sort((left,right)=>
    (right.wholeTargetMm3??-1)-(left.wholeTargetMm3??-1)||
    (left.wholeNonTargetMm3??Infinity)-(right.wholeNonTargetMm3??Infinity)||
    left.branchId.localeCompare(right.branchId))[0];
  const tradeoffHead=<thead><tr><th>Approach</th><th>Entry RAS · mm</th><th>Hand / sector order</th>
    <th>Task gate / cleanup</th><th>Whole target · mm³</th><th>Whole non-target · mm³</th>
    <th>Residual target · mm³ / source fraction</th><th>Sector Δ target / non-target · mm³</th>
    <th>Simulated s</th><th>Option calls / actions</th><th>Withdrawal probe</th><th>Native inspection</th>
  </tr></thead>;
  const summaryCard=(row:JointBranchRow,label:string)=><article key={row.branchId}>
    <h4>{label} · {row.method}</h4>
    <p>Task {row.status==='passed'?'passed':'unpassed'} · withdrawal {row.cleanupComplete?'complete':'incomplete'}</p>
    <p>Whole target {amount(row.wholeTargetMm3)} mm³<br/>Whole non-target {amount(row.wholeNonTargetMm3)} mm³</p>
    <p>Residual target {amount(row.residualTargetMm3)} mm³{row.sourceFractionRemoved===null?'':` · ${(100*row.sourceFractionRemoved).toFixed(2)}% removed`}</p>
    <details><summary>Geometry, sequence and resources</summary>
      <p>Entry RAS: {entry(row)} mm. {row.hand??'Unavailable'} hand · {row.order??'order unavailable'}.</p>
      <p>Sector increment: target {amount(row.additionalTargetMm3)}, non-target {amount(row.additionalNonTargetMm3)} mm³.</p>
      <p>{amount(row.simulatedSeconds)} simulated seconds · {row.end} actions · {row.optionPhysicalCalls??'unavailable'} option calls.</p>
      <p>Access withdrawal probe: {withdrawal(row)}.</p>
    </details>{branchActions(row)}
  </article>;
  return <details className="research-planning joint-strategies"><summary>Deterministic joint strategies · {data.sectorPassed} / {data.selectedRoutes} sector task gates passed</summary>
    <p className="surgical-warning">Source anatomy supports a bounded research simulation. Skull opening, complete vessels, functional anatomy, and clinical feasibility remain unknown.</p>
    <p role="status">{data.selectedRoutes} distinct source geometry routes selected · {data.accessExecuted} native access executions · {data.sectorPassed} native sector task gates passed. Three-approach gate: {data.threeApproachTaskGateMet?'met':'unmet'}.</p>
    <p className="surgical-note">Recorded sector task gates: joint fixed left/far-first {data.sectorPassed} / {data.sectors.length}; original classic hand/order comparison {passedClassic.length} / {data.classicSectors.length}. These option sets differ.</p>
    <p className="surgical-note">Fixed joint execution: one far-first strategy with left CUSA per selected route. No optimizer updates. {data.planningPhysicalCalls} physical calls belong to joint planning. {hasSavedJoint?'Saved export replay calls are counted separately in the export receipt.':'This live session has not performed a saved export replay.'}</p>
    <p className="surgical-note">To generate another joint comparison, open an original classic source workspace at its intact state.</p>
    <button disabled={disabled} onClick={()=>{pause();void c.run('selectBranch',{branchId:data.sourceBranchId,position:0});}}>Inspect common intact source</button>
    <section aria-label="Whole approach summaries">{data.sectors.map(row=>summaryCard(row,'Joint'))}
      {classicReference&&summaryCard(classicReference,passedMeasured.length?'Classic passed reference':'Classic observed reference · unpassed')}
    </section>
    <details><summary>Detailed whole approach comparison table</summary><div className="rescue-comparison"><table aria-label="Whole approach joint and classic reference outcomes"><caption>Whole native endpoint from the common intact source. The classic reference is the measured passing branch with the most full target removal, or the highest measured unpassed branch when no passing reference is available.</caption>
      {tradeoffHead}<tbody>{data.sectors.map(row=>tradeoffRow(row,'Joint'))}
        {classicReference&&tradeoffRow(classicReference,passedMeasured.length?'Classic passed reference':'Classic observed reference · unpassed')}
      </tbody></table></div></details>
    {!classicReference&&<p className="surgical-note">No finite whole-endpoint classic reference is available in this receipt.</p>}
    <p className="surgical-note">Whole target, non-target, residual, and source fraction come from the recorded endpoint. Sector Δ is additional removal after access. Option calls count that option’s recorded native planning work; the aggregate joint planning cost above covers all attempted options. Withdrawal is a modeled access probe, not a clinical clearance claim. Unavailable means the receipt did not provide a finite measured value.</p>
    {data.access.length>0&&<section aria-label="Joint access executions"><h4>Recorded access branches</h4>
      {data.access.map(branchActions)}</section>}
    {data.sectors.length>0&&<section aria-label="Joint sector executions"><h4>Recorded sector branches</h4>
      {data.sectors.map(row=><div key={row.branchId}>
        {branchActions(row)} <span>{row.status==='passed'?'Task gate passed':'Task gate unpassed'} · cleanup {row.cleanupComplete?'complete':'incomplete'}</span>
      </div>)}</section>}
    {data.failures.length>0&&<section aria-label="Retained stopped native branches"><h4>Stopped native executions · unpassed</h4>
      {data.failures.map(row=><div key={row.branchId}>{branchActions(row)}
        <span>Unpassed · cleanup {row.cleanupComplete?'complete':'incomplete'} · {row.reason}</span></div>)}
      <p className="surgical-note">These stopped prefixes are retained for inspection. They are not admitted strategy alternatives.</p>
    </section>}
    <details onToggle={event=>setClassicOpen(event.currentTarget.open)}><summary>Original classic comparison · {data.classicAccess.length} access and {data.classicSectors.length} sector branches</summary>
      {classicOpen&&<>
        <p>The original source case and branch actions remain in this workspace. Open their recorded access or sector start and endpoint in the native viewer.</p>
        <section aria-label="Original access comparison">{data.classicAccess.map(branchActions)}</section>
        <section aria-label="Original sector comparison"><div className="rescue-comparison"><table aria-label="All original classic whole approach outcomes"><caption>All original hand and order variants · whole native endpoints</caption>{tradeoffHead}<tbody>{data.classicSectors.map(row=>tradeoffRow(row,'Classic'))}</tbody></table></div></section>
      </>}
    </details>
    <details><summary>Source and limits</summary><p>Case: {data.caseHash}</p><p>Scene: {data.sceneId}</p><p>Intact state: {data.startStateId}</p><p>Joint receipt: {data.jointSeal}</p><p>Geometry selection is a source-only heuristic. Every displayed execution is tied to a retained native branch; these results are not clinical validation.</p></details>
  </details>;
}

import {useState} from 'react';
import type {SurgicalController} from './use-surgical';
import {jointStrategyDisplay} from './joint-strategy-data';
import type {JointBranchRow} from './joint-strategy-data';

export function JointStrategiesPanel({c,artifact,pause}:{c:SurgicalController;artifact:unknown;pause:()=>void}){
  const [classicOpen,setClassicOpen]=useState(false);
  if(!artifact||typeof artifact!=='object'||!('jointPlanning' in artifact))return null;
  const data=jointStrategyDisplay(artifact,c.result);
  if(!data)return <p role="alert" className="surgical-warning">Saved joint strategy evidence could not be bound to this scene and native branch inventory.</p>;
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
  return <details className="research-planning joint-strategies"><summary>Deterministic joint strategies · {data.sectorPassed} / {data.selectedRoutes} sector task gates passed</summary>
    <p className="surgical-warning">Source anatomy supports a bounded research simulation. Skull opening, complete vessels, functional anatomy, and clinical feasibility remain unknown.</p>
    <p role="status">{data.selectedRoutes} distinct source geometry routes selected · {data.accessExecuted} native access executions · {data.sectorPassed} native sector task gates passed. Three-approach gate: {data.threeApproachTaskGateMet?'met':'unmet'}.</p>
    <p className="surgical-note">Fixed joint execution: one far-first strategy with left CUSA per selected route. No optimizer updates. {data.planningPhysicalCalls} physical calls belong to joint planning; saved export replay calls are counted separately in the export receipt.</p>
    <button disabled={disabled} onClick={()=>{pause();void c.run('selectBranch',{branchId:data.sourceBranchId,position:0});}}>Inspect common intact source</button>
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
        <section aria-label="Original sector comparison">{data.classicSectors.map(row=><div key={row.branchId}>{branchActions(row)}
          <span>{row.status==='passed'?'Task gate passed':'Task gate unpassed'} · cleanup {row.cleanupComplete?'complete':'incomplete'}</span></div>)}</section>
      </>}
    </details>
    <details><summary>Source and limits</summary><p>Case: {data.caseHash}</p><p>Scene: {data.sceneId}</p><p>Intact state: {data.startStateId}</p><p>Joint receipt: {data.jointSeal}</p><p>Geometry selection is a source-only heuristic. Every displayed execution is tied to a retained native branch; these results are not clinical validation.</p></details>
  </details>;
}

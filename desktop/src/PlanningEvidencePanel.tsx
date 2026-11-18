import type {SurgicalController} from './use-surgical';

interface Quantity {
  cumulative?:number|null;incremental?:number|null;unit:string;status:string;reason?:string|null;
  incrementalMeaning?:string;source?:{labelName?:string;material?:string};
}
interface StructureReading {
  source:{provenance:string;reviewStatus:string}|null;
  removedMm3:Quantity;
  cutSourceMm3:Quantity;
  recordedContactCellSeconds:Quantity;
}
interface Decision {
  status:'continue'|'reassess'|'stop_with_residual'|'stop_after_treatment';
  reason:string;
  selectedId:string|null;
  frontierIds:string[];
  preferenceHarmMetrics:string[];
  controllingHarmMetrics:string[];
  supportingBoundaries:{id:string;label:string;residualMm3:number}[];
}
interface DecisionReceipt {
  selectionIndex:number;
  sourcePosition:number;
  sourceStateId:string;
  decision:Decision;
  selectedSkillId:string;
  alternatives:{id:string;treatmentMm3:number;harms:Record<string,number>;admissible:boolean|null}[];
  residual:{residualMm3:number;unexplainedResidualMm3:number};
}
interface EvidenceRow {
  branchId:string;
  position:number;
  stateId:string;
  refusals:number;
  endpoint:{progress:{originalTargetTreatmentMm3:number;originalTargetRemovedMm3:number};classification:string;validWithdrawal:boolean;truncated:boolean};
  verification:{verified:boolean;adapterStopOutcome?:{status:string;reasonCode:string}|null};
  computation:{planningPhysicalCalls:number;executionPhysicalCalls:number};
  consequences:{
    residualTargetMm3:number;
    dimensions:Record<string,Quantity>;
    regionalMetricIds:string[];
    mechanismEvidence:{
      functional:{structures:Record<string,StructureReading>;evidence:Record<string,{status:string;reason?:string}>};
      vascular:{status:string;reason?:string};
      mechanical:{status:string;reason?:string};
    };
    treatmentIntent?:{lesionId:string|null;lesionRevision:string|null;originalBurdenMm3:number|null;baselineFunction:{summary:string|null}};
  };
  intentDecisions:DecisionReceipt[];
  intentExecution?:{executionStatus:string};
}
export interface PlanningEvidence {
  schema:'resectionlab.recorded-planning-evidence.v1';
  sourceSceneId:string;
  baselineStateId:string;
  alternatives:EvidenceRow[];
  clinicalValidation:false;
  scope:string;
  seal:string;
}

const labels:Record<string,string>={
  'conditional':'Legacy conditional controller','harm-aware':'Named preservation controller','search':'Bounded search',
  'corticospinal':'Corticospinal tract (CST)','motor-cortex':'Motor cortex',
  'language-cortex':'Language cortex','brainstem':'Brainstem','cerebellum':'Cerebellum',
  neuralRemovedMm3:'aggregate neural membership removal',
};
const intents:Record<Decision['status'],string>={continue:'Continue',reassess:'Reassess',stop_with_residual:'Finish with residual',stop_after_treatment:'Finish after treatment'};
const amount=(n?:number|null)=>typeof n==='number'?new Intl.NumberFormat(undefined,{maximumFractionDigits:3}).format(n):'Unknown';
const units:Record<string,string>={mm3:'mm³','cell*s':'cell·s','mm3*s':'mm³·s','Pa2*mm3*s':'Pa²·mm³·s','N*s':'N·s','bond*s':'bond·s'};
const effectLabels:Record<string,string>={removedMm3:'removal',cutSourceMm3:'cut-edge source incidence',supplyDeficitSeconds:'modeled supply deficit',drainageDeficitSeconds:'modeled drainage deficit'};
const metric=(name:string)=>{
  const [,subject,effect]=name.split(':');
  return subject&&effect?`${labels[subject]??subject} ${effectLabels[effect]??effect}`:labels[name]??name.replaceAll('_',' ');
};
const value=(q:Quantity)=>q.cumulative==null?'Unknown':`${amount(q.cumulative)} ${units[q.unit]??q.unit} cumulative · ${amount(q.incremental)} since baseline`;

function DecisionSummary({receipt,current,onInspect}:{receipt:DecisionReceipt;current:boolean;onInspect:()=>void}){
  const d=receipt.decision,chosen=receipt.alternatives.find(row=>row.id===d.selectedId);
  const projectedHarms=chosen?d.controllingHarmMetrics.filter(key=>Number.isFinite(chosen.harms[key]))
    .map(key=>`${metric(key)} ${amount(chosen.harms[key])} mm³`):[];
  return <article data-intent={d.status} className="planning-evidence-decision">
    <h4>{intents[d.status]} · action {receipt.sourcePosition}</h4>
    {d.status==='continue'&&<p>Selected a continuation with {amount(chosen?.treatmentMm3)} mm³ projected additional treatment.</p>}
    {projectedHarms.length>0&&<p>Projected controlling harm · {projectedHarms.join(', ')}.</p>}
    {d.status==='reassess'&&<p>{d.reason==='unresolved_treatment_harm_tradeoff'?'Remaining treatment trades off against the declared preservation priorities.':'The available evidence does not resolve the remaining treatment choice.'}</p>}
    {d.status==='stop_with_residual'&&<p>Retain {amount(receipt.residual.residualMm3)} mm³ where additional work meets the declared preservation boundary.</p>}
    {d.status==='stop_after_treatment'&&<p>The original target burden has been exhausted in the simulation.</p>}
    {!!d.controllingHarmMetrics.length&&<p>Decision affected by {d.controllingHarmMetrics.map(metric).join(', ')}.</p>}
    {!!d.supportingBoundaries.length&&<ul>{d.supportingBoundaries.map(row=><li key={row.id}>{labels[row.label]??row.label} · {amount(row.residualMm3)} mm³ residual</li>)}</ul>}
    {current?<p>At this recorded checkpoint.</p>:<button onClick={onInspect}>Inspect decision at action {receipt.sourcePosition}</button>}
    <details><summary>Candidate projections and decision evidence</summary>
      <p>Declared priorities {d.preferenceHarmMetrics.map(metric).join(', ')}.</p>
      <p>Full benefit–harm frontier {d.frontierIds.join(', ')||'empty'}.</p>
      <p>Candidate volumes are estimated source contacts. Unexplained residual {amount(receipt.residual.unexplainedResidualMm3)} mm³.</p>
      <pre>{JSON.stringify(receipt,null,2)}</pre>
    </details>
  </article>;
}

function Mechanisms({row}:{row:EvidenceRow}){
  const evidence=row.consequences.mechanismEvidence;
  const regional=(prefix:string)=>row.consequences.regionalMetricIds.filter(key=>key.startsWith(prefix));
  const details=(keys:string[])=>keys.length>0&&<ul>{keys.map(key=>{
    const q=row.consequences.dimensions[key];
    return <li key={key}>{q.source?.labelName&&`${q.source.labelName} · `}{metric(key)} · {value(q)} {q.incrementalMeaning??q.reason}</li>;
  })}</ul>;
  return <details data-planning-mechanisms><summary>Mechanism evidence · {labels[row.branchId]}</summary>
    <h4>Direct functional-source effects</h4>
    <p>Shared cells can contribute to multiple anatomical memberships. Each quantity covers the named structure separately.</p>
    <ul>{Object.entries(evidence.functional.structures).map(([name,structure])=><li key={name}>
      <strong>{labels[name]??name}</strong> · removal {value(structure.removedMm3)} · cut-edge incidence {value(structure.cutSourceMm3)} · retained contact {value(structure.recordedContactCellSeconds)}
      {structure.source&&<small> · {structure.source.reviewStatus} · {structure.source.provenance}</small>}
    </li>)}</ul>
    <h4>Vascular exposure</h4>
    <p>{evidence.vascular.reason??evidence.vascular.status.replaceAll('_',' ')}</p>
    {row.consequences.dimensions.bloodGeneratedMm3?.cumulative!=null&&<p>Modeled blood generated · {value(row.consequences.dimensions.bloodGeneratedMm3)}</p>}
    {details(regional('vascular:'))}
    <h4>Mechanical exposure</h4>
    <p>{evidence.mechanical.reason??evidence.mechanical.status.replaceAll('_',' ')}</p>
    {details(regional('mechanical:'))}
    <h4>Mapping, monitoring and outcome evidence</h4>
    <ul>{Object.entries(evidence.functional.evidence).filter(([,source])=>source.reason).map(([name,source])=><li key={name}>{source.reason}</li>)}</ul>
  </details>;
}

export function PlanningEvidencePanel({c,pause}:{c:SurgicalController;pause:()=>void}){
  const packet=c.result?.planningEvidence,view=c.view;
  if(!view||packet?.schema!=='resectionlab.recorded-planning-evidence.v1'||packet.sourceSceneId!==view.scene.sceneId)return null;
  const harm=packet.alternatives.find(row=>row.branchId==='harm-aware');
  if(!harm)return null;
  const chosen=packet.alternatives.find(row=>row.branchId===c.result?.branchId)??harm;
  const decisions=harm.intentDecisions;
  const inspect=(branchId:string,position:number)=>{if(!c.pending){pause();void c.run('selectBranch',{branchId,position});}};
  const progress=chosen.endpoint.progress,intent=chosen.consequences.treatmentIntent;
  const original=intent?.originalBurdenMm3??(progress.originalTargetRemovedMm3+chosen.consequences.residualTargetMm3);
  const showDecision=(r:DecisionReceipt)=><DecisionSummary key={r.selectionIndex} receipt={r}
    current={r.sourceStateId===view.stateId} onInspect={()=>inspect(harm.branchId,r.sourcePosition)}/>;
  return <section className="planning-evidence" aria-label="Evidence-aware planning comparison" data-planning-evidence={packet.seal}>
    <h3>Evidence-aware planning comparison</h3>
    <p>{view.scene.regions.filter(r=>r.target).map(r=>r.name).join(', ')} · original labelled burden {amount(original)} mm³</p>
    <p>{packet.scope} Baseline function and clinical outcomes are unknown.</p>
    <div className="planning-evidence-alternatives">
      {packet.alternatives.map(row=><article key={row.branchId} data-planning-alternative={row.branchId}>
        <h4>{labels[row.branchId]??row.branchId}</h4>
        <p>{amount(row.endpoint.progress.originalTargetTreatmentMm3)} mm³ simulated treatment · {amount(row.consequences.residualTargetMm3)} mm³ original residual</p>
        <p>CST source removal {amount(row.consequences.dimensions['structure:corticospinal:removedMm3']?.cumulative)} mm³ · motor cortex source removal {amount(row.consequences.dimensions['structure:motor-cortex:removedMm3']?.cumulative)} mm³</p>
        <p>{row.endpoint.truncated?'Truncated':row.verification.verified&&row.endpoint.validWithdrawal&&row.verification.adapterStopOutcome?.reasonCode==='PROCEDURE_STOP'?'Checked STOP · whole tools withdrawn':row.endpoint.classification.replaceAll('-',' ')}{row.intentExecution&&` · ${row.intentExecution.executionStatus.replaceAll('_',' ')}`}</p>
        <p>{row.computation.executionPhysicalCalls} execution · {row.computation.planningPhysicalCalls} preview requests · {row.refusals} execution refusals</p>
        <button disabled={!!c.pending} onClick={()=>inspect(row.branchId,row.position)}>Inspect {labels[row.branchId]??row.branchId}</button>
      </article>)}
    </div>
    <h3>Recorded decisions</h3>
    {decisions.filter((_,i)=>i===0||i===decisions.length-1).map(showDecision)}
    {decisions.length>2&&<details><summary>Intermediate decisions</summary>{decisions.slice(1,-1).map(showDecision)}</details>}
    <Mechanisms row={chosen}/>
    <details><summary>Source and checkpoint binding</summary><p>Scene <code>{packet.sourceSceneId}</code></p><p>Baseline <code>{packet.baselineStateId}</code></p>
      <pre>{JSON.stringify(intent??null,null,2)}</pre></details>
  </section>;
}

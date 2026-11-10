import type {ReactNode} from 'react';
import type {SurgicalWorkflowStage,WorkflowAlternative} from './surgical-workflow-data';

export interface WorkflowPlanningPresentation {stage?:SurgicalWorkflowStage;onCompared?:()=>void;onInstalled?:()=>void}
export function WorkflowPlanningShell({stage,title,className,children}:{stage?:SurgicalWorkflowStage;title:string;className?:string;children:ReactNode}) {
  return stage?<section className={className} hidden={stage!=='plan'&&stage!=='compare'} aria-label={title}><h3>{title}</h3>{children}</section>:
    <details className={className} open={title==='Plan an approach · post-exposure'}><summary>{title}</summary>{children}</details>;
}
const amount=(v:number|null,unit:string)=>v===null?'Unassessed':`${v.toFixed(2)} ${unit}`;
export function WorkflowAlternatives({rows,selectedId,onSelect,disabled=false}:{rows:WorkflowAlternative[];selectedId?:string|null;onSelect:(id:string)=>void;disabled?:boolean}) {
  return <div className="workflow-alternatives" aria-label="Strategy alternative cards">{rows.map(row=><article key={row.id} className="workflow-alternative" data-workflow-alternative={row.id} aria-label={row.label}>
    <strong>{row.label}</strong><p>{row.status}{row.recommended?' · modeled preference':''}</p>
    <dl><div><dt title={row.coverageHelp??"Fraction of the original source target removed"}>{row.coverageLabel??"Target coverage"}</dt><dd>{row.fractionRemoved===null?'Unassessed':`${(row.fractionRemoved*100).toFixed(1)}%`}</dd></div>
      <div><dt>{row.quantityScope==='whole-sequence'?'Whole target treatment':'New target'}</dt><dd>{amount(row.removedMm3,'mm³')}</dd></div><div><dt>Residual</dt><dd>{amount(row.residualMm3,'mm³')}</dd></div>
      <div><dt>{row.quantityScope==='whole-sequence'?'Whole non-target':'New non-target'}</dt><dd>{amount(row.nonTargetMm3,'mm³')}</dd></div><div><dt>Modeled time</dt><dd>{amount(row.seconds,'s')}</dd></div></dl>
    <p className="surgical-note">{row.residualLocation}</p><p>{row.reason.replaceAll('_',' ')}</p>
    <details><summary>Burdens and unknowns</summary><p>{row.unknowns}</p><p>Preparation and access costs are included when supplied. Prior recorded burdens persist separately. Percentages require the original source-target denominator; unavailable fractions stay unassessed. These modeled preferences are research comparisons.</p></details>
    <button disabled={disabled} aria-pressed={selectedId===row.id} onClick={()=>onSelect(row.id)}>Inspect {row.label}</button>
  </article>)}</div>;
}

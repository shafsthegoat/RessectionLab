import * as Dialog from '@radix-ui/react-dialog';
import {useState} from 'react';
import type {ResectionApi} from './types';
import './recorded-comparison.css';

type Point = [number,number,number];
interface SectionPlane {
  center:Point;
  u:Point;
  v:Point;
  spacingMm:number;
  size:number;
  arms:{png:string;removedPixels:number}[];
}
interface ReferenceSections {
  seriesId:string;
  modality:string;
  sourceSha256:string;
  planes:SectionPlane[];
}

interface RecordedArm {
  name:string;
  actions:number;
  actionsHash:string;
  endpointStateId:string;
  generator:{kind:string;name:string}|null;
  lengthMm:number;
  radiusMm:number;
  targetRemovedMm3:number;
  sourceTargetMm3:number|null;
  residualMm3:number|null;
  otherRemovedMm3:number;
  materialRemovalMm3:Record<string,number>;
  simulatedSeconds:number;
  withdrawn:boolean;
}

export interface RecordingComparison {
  patientId:string;
  sourceSnapshotId:string;
  targetLayerId:string;
  arms:[RecordedArm,RecordedArm];
  sections?:ReferenceSections|{unavailable:string};
}

type Request = {collectionId:string;snapshotId:string;targetLayerId:string};
type Compare = NonNullable<ResectionApi['compareBoundSurgicalReplays']>;

function value(amount:number|null|undefined,unit:string) {
  return amount==null?'Unmeasured':amount.toLocaleString(undefined,{maximumFractionDigits:2})+' '+unit;
}

function directionLabel(direction:Point) {
  const signs=[['L','R'],['P','A'],['I','S']];
  return direction.map((component,axis)=>({weight:Math.abs(component),text:signs[axis][Number(component>0)]}))
    .filter(item=>item.weight>.30).sort((a,b)=>b.weight-a.weight).map(item=>item.text).join('');
}

function MatchedSections({result,sections}:{result:RecordingComparison;sections:ReferenceSections}) {
  const [selected,setSelected]=useState(0);
  const plane=sections.planes[selected];
  const barPercent=100*10/(plane.spacingMm*plane.size);
  const up=plane.v.map(n=>-n) as Point;
  return <section className="endpoint-sections" aria-label="Matched endpoint sections">
    <div className="section-toolbar">
      <h3>Matched {sections.modality} sections</h3>
      <div role="group" aria-label="Shared section plane">{['Along A','Along B'].map((label,i)=>
        <button type="button" key={label} aria-pressed={selected===i} onClick={()=>setSelected(i)}>{label}</button>)}</div>
    </div>
    <div className="endpoint-pair">{result.arms.map((arm,i)=>{
      const letter=i===0?'A':'B';
      const view=plane.arms[i];
      const fraction=arm.sourceTargetMm3&&arm.residualMm3!=null
        ? (100*(arm.sourceTargetMm3-arm.residualMm3)/arm.sourceTargetMm3).toLocaleString(undefined,{maximumFractionDigits:2})+'% of original target removed'
        : null;
      return <figure key={arm.actionsHash} data-endpoint-state={arm.endpointStateId}>
        <figcaption><span className="endpoint-letter">{letter}</span><strong>{arm.name}</strong></figcaption>
        <div className="endpoint-image"><img src={'data:image/png;base64,'+view.png}
          width={plane.size} height={plane.size} alt={arm.name+' endpoint on the shared '+sections.modality+' section'}/>
          <span className="section-up">{directionLabel(up)}</span><span className="section-right">{directionLabel(plane.u)}</span>
          <span className="section-scale" style={{width:barPercent+'%'}}>10 mm</span>
        </div>
        {fraction&&<div className="endpoint-fraction">{fraction}</div>}
      </figure>;
    })}</div>
    <div className="endpoint-legend"><span><i className="endpoint-target"/>Original target contour</span>
      <span><i className="endpoint-removal"/>Removed source cells</span></div>
    <p className="endpoint-explanation">Both views show the same reference scan through the two entries and the selected approach.</p>
  </section>;
}

export function RecordedStrategyComparison({compare,request,disabled}:{compare:Compare;request:Request;disabled:boolean}) {
  const [pending,setPending]=useState(false);
  const [open,setOpen]=useState(false);
  const [result,setResult]=useState<RecordingComparison|null>(null);
  const [error,setError]=useState('');
  const sections=result?.sections;
  const reference=sections&&'planes' in sections?sections:null;
  async function compareFiles() {
    setPending(true);setError('');
    try {
      const next=await compare(request);
      if(next){setResult(next);setOpen(true);}
    } catch(cause) {
      setResult(null);setError(cause instanceof Error?cause.message:String(cause));
    } finally {setPending(false);}
  }
  const rows:[string,(arm:RecordedArm)=>string][]=[
    ['Path length',arm=>value(arm.lengthMm,'mm')],
    ['Corridor radius',arm=>value(arm.radiusMm,'mm')],
    ['Target removed¹',arm=>value(arm.targetRemovedMm3,'mm³')],
    ['Whole target remaining',arm=>value(arm.residualMm3,'mm³')],
    ['Other tissue removed',arm=>value(arm.otherRemovedMm3,'mm³')],
    ['Mapped soft tissue removed',arm=>value(arm.materialRemovalMm3.soft_tissue,'mm³')],
    ['Mapped bone removed',arm=>value(arm.materialRemovalMm3.bone,'mm³')],
    ['Mapped vessel removed',arm=>value(arm.materialRemovalMm3.vessel,'mm³')],
    ['Simulated time',arm=>value(arm.simulatedSeconds,'s')],
    ['Recorded actions',arm=>String(arm.actions)],
    ['Withdrawal complete',arm=>arm.withdrawn?'Yes':'No'],
  ];
  return <section className="operative-inspector" aria-label="Recorded procedure comparison">
    <button disabled={disabled||pending} onClick={()=>void compareFiles()}>{pending?'Checking recordings…':'Compare two recordings'}</button>
    {error&&<p role="alert">{error}</p>}
    {result&&<Dialog.Root open={open} onOpenChange={setOpen}>
      <Dialog.Trigger asChild><button>Review comparison</button></Dialog.Trigger>
      <Dialog.Portal><Dialog.Overlay className="dialog-overlay"/>
        <Dialog.Content className="study-dialog recorded-comparison">
      <div className="study-dialog-heading"><Dialog.Title>Recorded endpoints</Dialog.Title>
        <Dialog.Close asChild><button className="outline-button">Close</button></Dialog.Close></div>
      <Dialog.Description>{result.patientId} · original target {value(result.arms[0].sourceTargetMm3,'mm³')}</Dialog.Description>
      {reference?<MatchedSections key={result.arms.map(arm=>arm.actionsHash).join('/')} result={result} sections={reference}/>
        :<p>{sections&&'unavailable' in sections?sections.unavailable:'Matched reference sections are unavailable for these recordings.'}</p>}
      <table><thead><tr><th scope="col">Measure</th>{result.arms.map((arm,i)=><th key={arm.actionsHash} scope="col">
        {String.fromCharCode(65+i)} · {arm.name}</th>)}</tr></thead>
        <tbody>{rows.map(([label,read])=><tr key={label}><th scope="row">{label}</th>{result.arms.map(arm=>
          <td key={arm.actionsHash}>{read(arm)}</td>)}</tr>)}</tbody>
      </table>
      <p>¹ Treatment credit uses cells assigned to the target. Remaining volume includes all original target memberships.</p>
      <p>Removal measures describe mapped tissue. Patient functional outcomes remain unknown.</p>
      <details><summary>Recording identities</summary>
        <p>{reference&&<>Reference series · {reference.seriesId}<br/>{reference.sourceSha256}<br/></>}
          Target layer · {result.targetLayerId}</p>{result.arms.map(arm=><p key={arm.actionsHash}>
        {arm.name} · {arm.generator?.name??'Generator unavailable'}<br/>{arm.endpointStateId}</p>)}</details>
        </Dialog.Content></Dialog.Portal>
    </Dialog.Root>}
  </section>;
}

import * as Dialog from '@radix-ui/react-dialog';
import {useState} from 'react';
import type {ResectionApi} from './types';

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
}

type Request = {collectionId:string;snapshotId:string;targetLayerId:string};
type Compare = NonNullable<ResectionApi['compareBoundSurgicalReplays']>;

function value(amount:number|null|undefined,unit:string) {
  return amount==null?'Unmeasured':amount.toLocaleString(undefined,{maximumFractionDigits:2})+' '+unit;
}

export function RecordedStrategyComparison({compare,request,disabled}:{compare:Compare;request:Request;disabled:boolean}) {
  const [pending,setPending]=useState(false);
  const [open,setOpen]=useState(false);
  const [result,setResult]=useState<RecordingComparison|null>(null);
  const [error,setError]=useState('');
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
      <table><thead><tr><th scope="col">Measure</th>{result.arms.map((arm,i)=><th key={arm.actionsHash} scope="col">
        {String.fromCharCode(65+i)} · {arm.name}</th>)}</tr></thead>
        <tbody>{rows.map(([label,read])=><tr key={label}><th scope="row">{label}</th>{result.arms.map(arm=>
          <td key={arm.actionsHash}>{read(arm)}</td>)}</tr>)}</tbody>
      </table>
      <p>¹ Treatment credit uses cells assigned to the target. Remaining volume includes all original target memberships.</p>
      <p>Removal measures describe mapped tissue. Patient functional outcomes remain unknown.</p>
      <details><summary>Recording identities</summary>{result.arms.map(arm=><p key={arm.actionsHash}>
        {arm.name} · {arm.generator?.name??'Generator unavailable'}<br/>{arm.endpointStateId}</p>)}</details>
        </Dialog.Content></Dialog.Portal>
    </Dialog.Root>}
  </section>;
}

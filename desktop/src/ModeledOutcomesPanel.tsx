import {useState} from 'react';

const object=(value:unknown):value is Record<string,unknown>=>!!value&&typeof value==='object'&&!Array.isArray(value);
const label=(value:string)=>value.replace(/([a-z])([A-Z])/g,'$1 $2').replaceAll('_',' ');
function Value({value,depth=0}:{value:unknown;depth?:number}){
  if(value===null||value===undefined)return <>Unavailable</>;
  if(typeof value==='number')return <>{Number.isFinite(value)?Number(value.toPrecision(6)).toLocaleString():'Unavailable'}</>;
  if(typeof value==='string'||typeof value==='boolean')return <>{String(value)}</>;
  if(Array.isArray(value))return <>{value.length<=8?value.map((row,index)=><div key={index}><Value value={row} depth={depth+1}/></div>):'See source receipt'}</>;
  if(!object(value)||depth>5)return <>See source receipt</>;
  if('status' in value&&'value' in value)return <>{value.value===null?<>Unavailable · {label(String(value.reason??value.status))}</>:<><Value value={value.value} depth={depth+1}/>{value.unit?` ${String(value.unit)}`:''}</>}</>;
  return <dl className="surgical-declared-fields">{Object.entries(value).filter(([key])=>!['status','reason','schema'].includes(key)).map(([key,item])=><div key={key}><dt>{label(key)}</dt><dd><Value value={item} depth={depth+1}/></dd></div>)}</dl>;
}
export function ModeledOutcomesPanel({outcome}:{outcome:unknown}){
  const [open,setOpen]=useState(false);
  if(!object(outcome)||outcome.schema!=='resectionlab.surgical-modeled-outcomes.v1')return null;
  return <details onToggle={e=>setOpen(e.currentTarget.open)}><summary>Recorded model consequences</summary>{open&&<>
    <p className="surgical-note">Source-bound simulator outputs. Contact time, mechanics, heat and local fluid are uncalibrated; neurological and perfusion probabilities remain unknown.</p>
    {['target','nonTargetByLabel','contact','mechanics','thermal','fluid','localVessels','protection','globalPerfusion','neurologicalRisk'].map(group=><details key={group}><summary>{label(group)}</summary><Value value={outcome[group]}/></details>)}
  </>}</details>;
}

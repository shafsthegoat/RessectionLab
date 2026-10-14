import {useState} from 'react';
import type {VolumeRenderingSettings} from './volumeTransfer';
export function TransferEditor({value,onChange}:{value:VolumeRenderingSettings;onChange:(value:VolumeRenderingSettings)=>void}){
  const [low,setLow]=useState(value.window?.[0]??0),[high,setHigh]=useState(value.window?.[1]??1000);
  const [points,setPoints]=useState<Array<[number,string,number]>>([[0,'#000000',0],[.25,'#4d6680',0],[.6,'#a6bfd9',.1],[1,'#fff2e6',.8]]);
  const [error,setError]=useState('');
  return <details><summary>Window & transfer colors</summary><div className="viewer-field-row"><label>Low<input aria-label="Window low" type="number" value={low} onChange={e=>setLow(+e.target.value)}/></label><label>High<input aria-label="Window high" type="number" value={high} onChange={e=>setHigh(+e.target.value)}/></label><button disabled={!(high>low)} onClick={()=>onChange({...value,window:[low,high]})}>Apply</button></div>
    {points.map((p,i)=><div className="viewer-field-row" key={i}><input aria-label={`Transfer position ${i+1}`} type="number" min="0" max="1" step=".01" value={p[0]} onChange={e=>setPoints(v=>v.map((x,j)=>i===j?[+e.target.value,x[1],x[2]]:x))}/><input aria-label={`Transfer color ${i+1}`} type="color" value={p[1]} onChange={e=>setPoints(v=>v.map((x,j)=>i===j?[x[0],e.target.value,x[2]]:x))}/><input aria-label={`Transfer opacity ${i+1}`} type="range" min="0" max="1" step=".01" value={p[2]} onChange={e=>setPoints(v=>v.map((x,j)=>i===j?[x[0],x[1],+e.target.value]:x))}/></div>)}
    <button onClick={()=>{const sorted=[...points].sort((a,b)=>a[0]-b[0]);if(sorted.some((p,i)=>!Number.isFinite(p[0])||p[0]<0||p[0]>1||(i>0&&p[0]===sorted[i-1][0]))){setError('Use distinct positions between zero and one.');return;}setError('');onChange({...value,points:sorted.map(p=>[p[0],parseInt(p[1].slice(1,3),16)/255,parseInt(p[1].slice(3,5),16)/255,parseInt(p[1].slice(5,7),16)/255,p[2]])})}}>Apply colors</button><button onClick={()=>onChange({...value,window:undefined,points:undefined})}>Reset</button>{error&&<p role="alert">{error}</p>}
  </details>;
}

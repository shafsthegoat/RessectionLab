import {useState} from 'react';
import {ctaCoverageInputs} from './native-study-state';
import type {NativeWorkspace} from './use-native-workspace';

type Selection='focused'|'wider'|'omit';

export function NativeCtaCoverage({workspace:w,disabled}:{workspace:NativeWorkspace;disabled:boolean}){
  const current=w.ctaCoverage;
  const [selection,setSelection]=useState<Record<string,Selection>>(()=>Object.fromEntries([
    ...(current?.focusIds??[]).map(id=>[id,'focused'] as const),
    ...(current?.widerIds??[]).map(id=>[id,'wider'] as const),
  ]));
  if(!w.binding||w.binding.readOnlyStudy)return null;
  const layers=ctaCoverageInputs(w.binding.document,w.series);
  if(layers.length<2)return null;
  const focus=layers.filter(layer=>selection[layer.id]==='focused').map(layer=>layer.id);
  const wider=layers.filter(layer=>selection[layer.id]==='wider').map(layer=>layer.id);
  return <details className="unified-details cta-coverage-panel" data-cta-coverage-panel>
    <summary>Combine supplied arteries</summary>
    <p>Choose focused branches and wider coverage from this CTA. Their supplied positives form one display layer.</p>
    <div className="cta-coverage-inputs">
      {layers.map(layer=><label key={layer.id}>{layer.label}<select aria-label={'Arterial coverage role for '+layer.label}
        data-arterial-layer={layer.id} value={selection[layer.id]??'omit'} disabled={disabled}
        onChange={e=>setSelection({...selection,[layer.id]:e.target.value as Selection})}>
        <option value="omit">Omit</option><option value="focused">Focused</option><option value="wider">Wider</option>
      </select></label>)}
    </div>
    <button className="wide-button" disabled={disabled||!focus.length||!wider.length||focus.length+wider.length>128}
      onClick={()=>void w.combineCtaCoverage(focus,wider)}>{current?'Update combined arteries':'Build combined arteries'}</button>
  </details>;
}

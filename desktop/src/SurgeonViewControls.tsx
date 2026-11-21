import {Vector3} from 'three';
import type {SurgicalController} from './use-surgical';
import type {ViewerWorkspaceSettings,MicroscopeSettings} from './viewer/workspaceSettings';
import {defaultSurgeonView,exposureMicroscope,microscopeCamera} from './viewer/microscope';

/** Surface inspection is display state, never a claim of operative exposure. */
export function inspectionNotice(settings:ViewerWorkspaceSettings):string|null {
  const surgeon=settings.operativeView==='surgeon';
  const transparent=surgeon?settings.microscope?.inspectionTransparency:Object.values(settings.layerOpacity??{}).some(value=>typeof value==='number'&&value<1);
  const cutaway=surgeon?settings.microscope?.inspectionCutaway:settings.clipping?.enabled||settings.clipBox?.enabled||settings.obliqueClip?.enabled;
  return transparent||cutaway?`Inspection only · ${[transparent?'transparent anatomy':'',cutaway?'cutaway':''].filter(Boolean).join(' and ')}`:null;
}

export function SurgeonViewControls({c,settings,onSettings,onFitAnatomy}:{onFitAnatomy?:()=>void;c:SurgicalController;settings:ViewerWorkspaceSettings;onSettings:(s:ViewerWorkspaceSettings)=>void}) {
  const enabled=settings.operativeView==='surgeon',m=settings.microscope;
  const inspection=!!m&&(m.inspectionCutaway||m.inspectionTransparency);
  function enter(section=false){
    if(!c.view)return;
    const next=enabled?settings:defaultSurgeonView(c.view,{...settings,operativeView:undefined});
    if(!next.microscope)return;
    const camera=next.camera??microscopeCamera(next.microscope);
    const towardField=new Vector3(...camera.target).sub(new Vector3(...camera.position)).normalize();
    // Keep the far side in any patient orientation, including oblique scans.
    onSettings({...next,microscope:{...next.microscope,inspectionTransparency:false,inspectionCutaway:section},
      obliqueClip:section?{enabled:true,normal:towardField.toArray(),offset:-towardField.dot(new Vector3(...camera.target))}:undefined,
      clipping:undefined,clipBox:undefined});
  }
  function external(){
    if(!enabled)return;
    onSettings({...settings,operativeView:'external',camera:settings.externalCamera??null,
      clipping:undefined,clipBox:undefined,obliqueClip:undefined});
    if(!settings.externalCamera)onFitAnatomy?.();
  }
  function scalpOverview(){
    if(!c.view?.operativeExposureV1)return;
    onSettings({...settings,operativeView:'external',camera:null,showPlane:false,clipping:undefined,clipBox:undefined,obliqueClip:undefined,
      layerOpacity:Object.fromEntries(c.view.scene.regions.map(region=>[region.name,1])),
      layerColor:Object.fromEntries(c.view.scene.regions.map(region=>[region.name,region.color]))});
    onFitAnatomy?.();
  }
  function update(patch:Partial<MicroscopeSettings>){if(!m)return;const next={...m,...patch};onSettings({...settings,microscope:next,camera:microscopeCamera(next)});}
  return <section className="surgeon-view" aria-label="Surgeon microscope view" data-operative-view={enabled?'surgeon':'external'}>
    <div role="group" aria-label="Field presentation" className="surgeon-view-modes">
      <button disabled={!c.view} aria-pressed={enabled&&!inspection} onClick={()=>enter()}>Surgeon / Microscope</button>
      {c.view?.operativeExposureV1&&<button onClick={scalpOverview}>External scalp overview</button>}
      <button disabled={!c.view} aria-pressed={!enabled} onClick={external}>External anatomy view</button>
      <button disabled={!c.view} aria-pressed={enabled&&!!m?.inspectionCutaway} onClick={()=>enter(true)}>Inspection cutaway</button>
    </div>
    {enabled&&m&&<>
      {(m.inspectionTransparency||m.inspectionCutaway)&&<strong role="status">INSPECTION ONLY · view through tissue enabled</strong>}
      <details><summary>Microscope settings and inspection</summary>
      {c.result?.researchSetup&&<p className="surgical-warning" aria-label="Operative exposure assumption">{c.result.researchSetup.description}</p>}
      <p className="surgical-note">The microscope stays at its focus while the tools move. Intact tissue occludes deeper structures.</p>
      <label>Surface material<select aria-label="Operative surface material" value={m.tissueStyle??'operative'} onChange={e=>update({tissueStyle:e.target.value as 'operative'|'labels'})}><option value="operative">Illustrative tissue material</option><option value="labels">Source segmentation colours</option></select></label>
      <p className="surgical-note">{m.tissueStyle==='labels'
        ?'Source colours identify labels. Cyan marks cumulative contact and orange marks heat.'
        :c.view?.operativeExposureV1?'Wound-only output appears red until another fluid enters the simulation. Mixed fluid has neutral translucent shading. Granular deposits show retained debris.':'Retained mixed fluid appears as translucent patches and local sheen. Granular material marks debris. Read the field ledger for volumes in mm³.'} Linked slices retain source label colours.</p>
      {([
        ['angleDeg','Viewing angle',-70,70,1,'°'],['azimuthDeg','Angle direction',-180,180,5,'°'],['workingDistanceMm','Working distance',60,350,5,'mm'],['fieldMm','Field / magnification',12,120,1,'mm'],['focusOffsetMm','Focus plane offset',-30,30,.5,'mm'],['light','Operative light',0,6,.1,'']
      ] as const).map(([key,label,min,max,step,unit])=><label key={key}>{label} <input aria-label={label} type="range" min={min} max={max} step={step} value={m[key]} onChange={e=>update({[key]:Number(e.target.value)})}/><output>{m[key]} {unit}</output></label>)}
      <p className="surgical-note">Focus RAS {m.target.map(n=>n.toFixed(1)).join(', ')} mm. Drag to orbit or pan around the opening.</p>
      {c.view?.operativeExposureV1&&<button aria-label="Frame operative opening" onClick={()=>{const frame=exposureMicroscope(c.view!);if(frame)update(frame);}}>Frame operative opening</button>}
      <button disabled={c.pending||!c.view} onClick={()=>{const slot=c.view?.slots[c.slot];if(slot)update({target:[...slot.position],focusOffsetMm:0});}}>Center field on accepted tip once</button>
      {c.view?.operativeExposureV1&&<div role="group" aria-label="Frame an accepted instrument">
        {(['left','right'] as const).map(hand=><button key={hand} disabled={!c.view?.slots[hand]} aria-label={`Center field on ${hand} accepted tip`}
          onClick={()=>{const pose=c.view?.slots[hand];if(pose)update({target:[...pose.position],focusOffsetMm:0});}}>{hand==='left'?'Frame left tip':'Frame right tip'}</button>)}
      </div>}
      <label><input type="checkbox" checked={!!m.inspectionTransparency} onChange={e=>update({inspectionTransparency:e.target.checked})}/>Inspection-only transparency / hidden anatomy</label>
      </details>
    </>}
  </section>;
}

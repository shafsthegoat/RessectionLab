import * as THREE from 'three';
import type {SurgicalController} from './use-surgical';
import type {ViewerWorkspaceSettings,MicroscopeSettings} from './viewer/workspaceSettings';
import {microscopeCamera} from './viewer/microscope';
import type {Point3} from './viewer/coordinates';

/** Surface inspection is display state, never a claim of operative exposure. */
export function inspectionNotice(settings:ViewerWorkspaceSettings):string|null {
  const surgeon=settings.operativeView==='surgeon';
  const transparent=surgeon?settings.microscope?.inspectionTransparency:Object.values(settings.layerOpacity??{}).some(value=>typeof value==='number'&&value<1);
  const cutaway=surgeon?settings.microscope?.inspectionCutaway:settings.clipping?.enabled||settings.clipBox?.enabled||settings.obliqueClip?.enabled;
  return transparent||cutaway?`Inspection only · ${[transparent?'transparent anatomy':'',cutaway?'cutaway':''].filter(Boolean).join(' and ')}`:null;
}

export function SurgeonViewControls({c,settings,onSettings}:{c:SurgicalController;settings:ViewerWorkspaceSettings;onSettings:(s:ViewerWorkspaceSettings)=>void}) {
  const enabled=settings.operativeView==='surgeon',m=settings.microscope;
  function enter(){
    if(enabled)return;
    const slots=Object.values(c.view?.slots??{}),slot=slots.find(s=>s.inserted)??slots[0];if(!slot)return;
    const blade=c.view?.planarBladeManeuver?.opticalFrame;
    const microscope:MicroscopeSettings=m??{target:blade?[...blade.targetMm]:[...slot.position],outward:blade?[...blade.outward]:new THREE.Vector3(0,0,-1).applyQuaternion(new THREE.Quaternion(...slot.quaternion)).toArray() as Point3,angleDeg:blade?30:0,azimuthDeg:0,workingDistanceMm:160,fieldMm:blade?12:42,focusOffsetMm:0,light:2.5};
    onSettings({...settings,operativeView:'surgeon',microscope,externalCamera:settings.camera,camera:microscopeCamera(microscope),layout:'3d-focus',showPlane:false,contextStyle:'uniform',clipping:undefined,clipBox:undefined,obliqueClip:undefined,volumeRendering:{...settings.volumeRendering,enabled:false}});
  }
  function update(patch:Partial<MicroscopeSettings>){if(!m)return;const next={...m,...patch};onSettings({...settings,microscope:next,camera:microscopeCamera(next)});}
  return <section className="surgeon-view" aria-label="Surgeon microscope view" data-operative-view={enabled?'surgeon':'external'}>
    <div><button disabled={!c.view} aria-pressed={enabled} onClick={enter}>Surgeon / Microscope</button> <button disabled={!enabled} onClick={()=>onSettings({...settings,operativeView:'external',camera:settings.externalCamera??settings.camera})}>External anatomy view</button></div>
    {enabled&&m&&<>
      {(m.inspectionTransparency||m.inspectionCutaway)&&<strong role="status">INSPECTION ONLY · view through tissue enabled</strong>}
      <details><summary>Microscope settings and inspection</summary>
      {c.result?.researchSetup&&<p className="surgical-warning" aria-label="Operative exposure assumption">{c.result.researchSetup.description}</p>}
      <p className="surgical-note">Fixed optical viewpoint · opaque tissue occludes the field. Place the view above an actual opening; intact tissue remains in the way. No exposure is created by the camera.</p>
      <label>Surface material<select aria-label="Operative surface material" value={m.tissueStyle??'operative'} onChange={e=>update({tissueStyle:e.target.value as 'operative'|'labels'})}><option value="operative">Illustrative tissue material</option><option value="labels">Source segmentation colours</option></select></label>
      <p className="surgical-note">{m.tissueStyle==='labels'?'Source label colours.':'Illustrative tissue colours and roughness; no acquired tissue colour, vessels or surface anatomy are added.'} Linked MRI retains source label colours. Lighting and shadows use the retained surfaces.</p>
      {([
        ['angleDeg','Viewing angle',-70,70,1,'°'],['azimuthDeg','Angle direction',-180,180,5,'°'],['workingDistanceMm','Working distance',60,350,5,'mm'],['fieldMm','Field / magnification',12,120,1,'mm'],['focusOffsetMm','Focus plane offset',-30,30,.5,'mm'],['light','Operative light',0,6,.1,'']
      ] as const).map(([key,label,min,max,step,unit])=><label key={key}>{label} <input aria-label={label} type="range" min={min} max={max} step={step} value={m[key]} onChange={e=>update({[key]:Number(e.target.value)})}/><output>{m[key]} {unit}</output></label>)}
      <p className="surgical-note">Focus center RAS mm: {m.target.map(n=>n.toFixed(1)).join(', ')}. Orbit/pan adjusts the optical view; tools never move it. Focus sets the optical plane; depth-of-field blur is not modeled.</p>
      <button disabled={c.pending||!c.view} onClick={()=>{const slot=c.view?.slots[c.slot];if(slot)update({target:[...slot.position],focusOffsetMm:0});}}>Center field on accepted tip once</button>
      <label><input type="checkbox" checked={!!m.inspectionTransparency} onChange={e=>update({inspectionTransparency:e.target.checked})}/>Inspection-only transparency / hidden anatomy</label>
      <label><input type="checkbox" checked={!!m.inspectionCutaway} onChange={e=>{const next={...m,inspectionCutaway:e.target.checked};onSettings({...settings,microscope:next,clipping:{axis:'z',value:m.target[2],enabled:e.target.checked}});}}/>Inspection-only cutaway</label>
      </details>
    </>}
  </section>;
}

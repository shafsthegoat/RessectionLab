import {useEffect,useState} from 'react';
import type {SurgicalController} from './use-surgical';
import type {ViewerVolume} from './viewer/contracts';
import type {Point3} from './viewer/coordinates';
import {validCameraState} from './viewer/workspaceSettings';
import type {ViewerWorkspaceSettings} from './viewer/workspaceSettings';
import {acceptedCloseTool,closeToolFrame,closeToolPresentation} from './full-case-presentation';
import {replayAuthorityKey} from './surgical-authority';

interface Props {c:SurgicalController;source:ViewerVolume|null;settings:ViewerWorkspaceSettings;cursor:Point3|null;onSettings:(settings:ViewerWorkspaceSettings)=>void;onCursor:(point:Point3|null)=>void;onFitAnatomy:()=>void}
type Snapshot={source:ViewerVolume;branch:string;scene:string;authority:string;settings:ViewerWorkspaceSettings;cursor:Point3|null};
type Pin={name:string;instrument:string;fieldMm:number;stateId:string};

/** Optional display command. No pose editing, replay dispatch or automatic follow. */
export function SurgicalCloseViewControls({c,source,settings,cursor,onSettings,onCursor,onFitAnatomy}:Props){
  const [snapshot,setSnapshot]=useState<Snapshot|null>(null),[pin,setPin]=useState<Pin|null>(null),[translucent,setTranslucent]=useState(true);
  const view=c.view,branch=c.result?.branchId??'',scene=view?.scene.sceneId??'',authority=c.result?replayAuthorityKey(c.result):'';
  const owns=!!snapshot&&snapshot.source===source&&snapshot.branch===branch&&snapshot.scene===scene&&snapshot.authority===authority;
  useEffect(()=>{if(snapshot&&!owns){setSnapshot(null);setPin(null);}},[snapshot,owns]);
  const effective=c.result?.playback?.effectiveActivation,tool=acceptedCloseTool(view,c.slot,effective),disabled=c.pending||!!c.preview||!tool;
  function frame(){
    if(disabled||!view||!source)return;
    const field=closeToolFrame(view,c.slot,effective,settings,translucent);if(!field)return;
    if(!owns)setSnapshot({source,branch,scene,authority,settings:structuredClone(settings),cursor:cursor?[...cursor]:null});
    setPin({name:field.name,instrument:field.instrument,fieldMm:field.fieldMm,stateId:field.stateId});onCursor(field.center);onSettings(field.settings);
  }
  function restore(){
    if(!owns||!snapshot)return;
    onSettings(snapshot.settings);onCursor(snapshot.cursor);if(!validCameraState(snapshot.settings.camera))onFitAnatomy();setSnapshot(null);setPin(null);
  }
  return <section aria-label="Close instrument view" data-pinned-state-id={owns?pin?.stateId:undefined} data-pinned-slot={owns?pin?.name:undefined} data-field-mm={owns?pin?.fieldMm:undefined}>
    <button disabled={disabled} onClick={frame}>{owns?'Refocus accepted tool':'Close tool view'}</button>
    {!tool&&<p className="surgical-note">An accepted inserted instrument is needed for a close view.</p>}
    {owns&&pin&&<>
      <button onClick={restore}>Restore previous view</button>
      <p className="surgical-note">Pinned to {pin.name} hand · {pin.instrument} · {pin.fieldMm} mm field. Orbit freely; refocus to use the latest accepted tip.</p>
      <label><input type="checkbox" checked={translucent} onChange={event=>{
        const enabled=event.target.checked;setTranslucent(enabled);if(!view||!snapshot)return;
        const base={...settings,layerOpacity:snapshot.settings.layerOpacity};onSettings(enabled?closeToolPresentation(view,base):base);
      }}/>Context transparency · display only</label>
      <p className="surgical-note">{translucent?'Display alpha: context soft tissue 4.5%, non-target bone 55%, membrane 14%, unknown 8%; declared targets 90%, vessels 100%.':'Original layer opacities restored.'} Visibility and clipping controls still apply. Linked slices retain the source image.</p>
    </>}
  </section>;
}

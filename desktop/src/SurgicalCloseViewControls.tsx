import {useEffect,useRef,useState} from 'react';
import type {SurgicalController} from './use-surgical';
import type {ViewerVolume} from './viewer/contracts';
import type {Point3} from './viewer/coordinates';
import {validCameraState} from './viewer/workspaceSettings';
import type {ViewerWorkspaceSettings} from './viewer/workspaceSettings';
import {acceptedCloseTool,closeToolFrame,followCloseToolFrame,closeToolPresentation,closeToolWindowPresentation,sourceContainsPoint} from './full-case-presentation';
import type {CloseToolAnchor} from './full-case-presentation';
import {replayAuthorityKey} from './surgical-authority';

interface Props {c:SurgicalController;source:ViewerVolume|null;settings:ViewerWorkspaceSettings;cursor:Point3|null;onSettings:(settings:ViewerWorkspaceSettings)=>void;onCursor:(point:Point3|null)=>void;onFitAnatomy:()=>void}
type Snapshot={source:ViewerVolume;branch:string;scene:string;authority:string;settings:ViewerWorkspaceSettings;cursor:Point3|null};
type Pin=CloseToolAnchor&{instrument:string;stateId:string;slicesAtTip:boolean};

/** Optional display commands consume accepted views only, never desired frames. */
export function SurgicalCloseViewControls({c,source,settings,cursor,onSettings,onCursor,onFitAnatomy}:Props){
  const [snapshot,setSnapshot]=useState<Snapshot|null>(null),[pin,setPin]=useState<Pin|null>(null),[translucent,setTranslucent]=useState(true);
  const [cutaway,setCutaway]=useState(true);
  const [follow,setFollow]=useState(false),lastFollow=useRef<string|null>(null);
  const view=c.view,branch=c.result?.branchId??'',scene=view?.scene.sceneId??'',authority=c.result?replayAuthorityKey(c.result):'';
  const owns=!!snapshot&&snapshot.source===source&&snapshot.branch===branch&&snapshot.scene===scene&&snapshot.authority===authority;
  useEffect(()=>{if(snapshot&&!owns){setSnapshot(null);setPin(null);setFollow(false);lastFollow.current=null;}},[snapshot,owns]);
  const effective=c.result?.playback?.effectiveActivation,tool=acceptedCloseTool(view,c.slot,effective),disabled=c.pending||!!c.preview||!tool||!source;
  const acceptedKey=JSON.stringify([branch,c.result?.playback?.position??c.result?.position,c.result?.playback?.frameIndex,view?.stateId]);
  useEffect(()=>{
    if(!follow||!owns||!snapshot||!pin||!view||!source||c.pending||c.preview||lastFollow.current===acceptedKey)return;
    // Commit the identity before publishing settings: orbit callbacks and
    // settings rerenders cannot recursively follow the same accepted frame.
    lastFollow.current=acceptedKey;
    const field=followCloseToolFrame(view,c.slot,effective,settings,pin,snapshot.settings,cutaway);if(!field)return;
    const slicesAtTip=sourceContainsPoint(source,field.center);
    setPin({...field,slicesAtTip});if(slicesAtTip)onCursor(field.center);onSettings(field.settings);
  },[follow,owns,acceptedKey,view,c.pending,c.preview]);
  function frame(){
    if(disabled||!view||!source)return;
    const field=closeToolFrame(view,c.slot,effective,settings,translucent,cutaway);if(!field)return;
    if(!owns)setSnapshot({source,branch,scene,authority,settings:structuredClone(settings),cursor:cursor?[...cursor]:null});
    const slicesAtTip=sourceContainsPoint(source,field.center);
    lastFollow.current=acceptedKey;setPin({...field,slicesAtTip});
    if(slicesAtTip)onCursor(field.center);
    onSettings(field.settings);
  }
  function restore(){
    if(!owns||!snapshot)return;
    setFollow(false);lastFollow.current=null;onSettings(snapshot.settings);onCursor(snapshot.cursor);if(!validCameraState(snapshot.settings.camera))onFitAnatomy();setSnapshot(null);setPin(null);
  }
  return <section aria-label="Close instrument view" data-pinned-state-id={owns?pin?.stateId:undefined} data-pinned-slot={owns?pin?.name:undefined} data-field-mm={owns?pin?.fieldMm:undefined} data-follow-accepted-tool={follow&&owns}>
    <button disabled={disabled} onClick={frame}>{owns?'Refocus accepted tool':'Close tool view'}</button>
    <label><input type="checkbox" checked={follow&&owns} disabled={!follow&&disabled} onChange={event=>{
      const enabled=event.target.checked;if(enabled&&!owns)frame();else if(enabled)lastFollow.current=null;setFollow(enabled);
    }}/>Follow accepted tool</label>
    <label><input type="checkbox" checked={cutaway} onChange={event=>{
      const enabled=event.target.checked;setCutaway(enabled);
      if(owns&&snapshot&&pin)onSettings(closeToolWindowPresentation(settings,pin.windowClipping,snapshot.settings,enabled));
    }}/>Local anatomy cutaway · display only</label>
    {!tool&&<p className="surgical-note">An accepted inserted instrument is needed for a close view.</p>}
    {owns&&pin&&<>
      <button onClick={restore}>Restore previous view</button>
      <p className="surgical-note">{follow?'Following':'Pinned to'} {pin.name} hand · {pin.instrument} · {pin.fieldMm} mm field. {follow?'Camera and local window move only with newly accepted states; orbit and zoom are preserved for the same instrument.':'Orbit freely; refocus to use the latest accepted tip.'}</p>
      <p className="surgical-note">{cutaway?(follow?'Local anatomy window follows the accepted tip.':'Local anatomy window is pinned to this tip.'):'Original clipping restored.'} Instruments remain complete; MRI source images stay unchanged. Restore previous view disables follow and returns the saved context and clipping.</p>
      {!pin.slicesAtTip&&<p className="surgical-note">No source image slice is available at this tip. Linked slices keep their previous position.</p>}
      <label><input type="checkbox" checked={translucent} onChange={event=>{
        const enabled=event.target.checked;setTranslucent(enabled);if(!view||!snapshot)return;
        const base={...settings,layerOpacity:snapshot.settings.layerOpacity};onSettings(enabled?closeToolPresentation(view,base):base);
      }}/>Context transparency · display only</label>
      <p className="surgical-note">{translucent?'Display alpha: context soft tissue 4.5%, non-target bone 55%, membrane 14%, unknown 8%; declared targets 90%, vessels 100%.':'Original layer opacities restored.'} Visibility and clipping controls still apply. Linked slices retain the source image.</p>
    </>}
  </section>;
}

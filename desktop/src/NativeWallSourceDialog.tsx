import * as Dialog from '@radix-ui/react-dialog';
import {FolderOpen, X} from 'lucide-react';
import {useEffect, useRef, useState} from 'react';
import {hydrateAnatomy} from './native-anatomy-data';
import type {NativeAnatomyApi, NativeWallSourceInspection, NativeWallSourceScan} from './native-anatomy-data';
import {operationMessage} from './operation-feedback';
import type {ResectionApi} from './types';
import {ViewerWorkspace} from './viewer/ViewerWorkspace';
import type {ViewerVolume} from './viewer/contracts';
import type {Point3} from './viewer/coordinates';
import type {ViewerWorkspaceSettings} from './viewer/workspaceSettings';

function fieldSettings({centerRasMm:target, layers}:NativeWallSourceScan):ViewerWorkspaceSettings {
  const showsVessels=layers.some(layer=>['vessel','vessels','artery','vein','sinus'].includes(layer.kind));
  return {
    layout:showsVessels?'3d-focus':'mri-review', showPlane:false, sliceOverlay:'contour',
    camera:{target, position:[target[0]+24,target[1]-98,target[2]+59], up:[0,0,1]},
    vesselShading:'balanced',
    volumeRendering:showsVessels?{enabled:false,density:.22}:undefined,
    layerOpacity:Object.fromEntries(layers.filter(layer=>layer.kind==='brain').map(layer=>[layer.id,showsVessels?.055:.14])),
  };
}

export function NativeWallSourceDialog({api,caseHash,close}:{api:ResectionApi&NativeAnatomyApi;caseHash:string;close:()=>void}) {
  const lease=useRef<NativeWallSourceInspection|null>(null);
  const generation=useRef(0);
  const activeRequest=useRef<string|null>(null);
  const [source,setSource]=useState<NativeWallSourceInspection|null>(null);
  const [scan,setScan]=useState<NativeWallSourceScan|null>(null);
  const [volume,setVolume]=useState<ViewerVolume|null>(null);
  const [cursor,setCursor]=useState<Point3|null>(null);
  const [settings,setSettings]=useState<ViewerWorkspaceSettings>({});
  const [visible,setVisible]=useState<Record<string,boolean>>({});
  const [loading,setLoading]=useState(true);
  const [error,setError]=useState('');

  async function loadScan(bound:NativeWallSourceInspection, next:NativeWallSourceScan) {
    const current=++generation.current;
    const requestId=`wall-source-view-${crypto.randomUUID()}`;
    activeRequest.current=requestId;
    setLoading(true);setError('');setVolume(null);setScan(next);
    setCursor(next.centerRasMm);setSettings(fieldSettings(next));
    setVisible(Object.fromEntries(next.layers.map(layer=>[layer.id,true])));
    try{
      const payload=await api.anatomyView({collectionId:bound.collectionId,snapshotId:bound.snapshotId,seriesId:next.seriesId,requestId});
      if(current!==generation.current)return;
      if(payload.snapshotId!==bound.snapshotId||payload.unavailableLayers.length||payload.compartments.length!==next.layers.length||
          payload.compartments.some(l=>l.sourceSeries!==next.seriesId||!next.layers.some(expected=>expected.id===l.name))||
          (next.independent?payload.registrationStatus!=='independent source frame':!payload.patientFrameReady))
        throw Error('The view differs from the verified source field');
      const loaded=await hydrateAnatomy(payload,api);
      if(current===generation.current)setVolume(loaded);
    }catch(failure){if(current===generation.current)setError(operationMessage(failure));}
    finally{
      if(activeRequest.current===requestId)activeRequest.current=null;
      if(current===generation.current)setLoading(false);
    }
  }

  async function openSource(choose=false) {
    const current=++generation.current;
    const requestId=`wall-source-${crypto.randomUUID()}`;
    activeRequest.current=requestId;
    setLoading(true);setError('');
    try{
      const bound=await api.inspectNativeWallSource!({caseHash,choose,requestId});
      if(!bound)return;
      if(current!==generation.current||bound.caseHash!==caseHash){
        await api.closeNativeWallSource?.({collectionId:bound.collectionId});
        return;
      }
      lease.current=bound;setSource(bound);
      await loadScan(bound,bound.scans[0]);
    }catch(failure){if(current===generation.current)setError(operationMessage(failure));}
    finally{
      if(activeRequest.current===requestId)activeRequest.current=null;
      if(current===generation.current)setLoading(false);
    }
  }

  useEffect(()=>{
    const current=generation.current;
    queueMicrotask(()=>{if(current===generation.current)void openSource();});
    return ()=>{
      generation.current++;
      const request=activeRequest.current;activeRequest.current=null;
      if(request)void api.cancel(request).catch(()=>{});
      const active=lease.current;lease.current=null;
      if(active)void api.closeNativeWallSource?.({collectionId:active.collectionId}).catch(()=>{});
    };
  },[api,caseHash]);

  return <Dialog.Root open onOpenChange={open=>{if(!open)close();}}>
    <Dialog.Portal>
      <Dialog.Overlay className="dialog-overlay"/>
      <Dialog.Content className="study-dialog native-wall-source-dialog">
        <div className="study-dialog-heading">
          <Dialog.Title>Rehearsal source study</Dialog.Title>
          <Dialog.Close className="outline-button"><X size={14}/> Return to rehearsal</Dialog.Close>
        </div>
        <Dialog.Description>Inspect the acquired scans at the saved field. Your rehearsal stays open.</Dialog.Description>
        {source&&<>
          <div className="native-wall-source-bar">
            <div role="group" aria-label="Source acquisition">{source.scans.map(item=>
              <button key={item.seriesId} className="outline-button" aria-pressed={scan===item} disabled={loading}
                onClick={()=>void loadScan(source,item)}>{item.modality} · {item.seriesId}</button>)}</div>
            <span>Rehearsal crop {source.cropShape.join(' × ')} voxels</span>
            {scan&&<button className="outline-button" disabled={!volume} onClick={()=>{setCursor(scan.centerRasMm);setSettings(fieldSettings(scan));}}>Reset to saved field</button>}
          </div>
          <p className="native-wall-source-scope">{scan?.independent?'Native acquisition coordinates · study-wide alignment unreviewed':'Registered patient coordinates'}
            {source.locallyReviewed?' · Cross-acquisition review applies to the saved crop':''}
            {scan?.layers.length?` · ${scan.layers.map(layer=>layer.label).join(' · ')}`:''}</p>
        </>}
        {error&&<p role="alert">{error}</p>}
        {loading&&<p role="status">{source?'Loading the acquired scan…':'Verifying the original study and rehearsal field…'}</p>}
        {!source&&!loading&&<button className="outline-button" onClick={()=>void openSource(true)}><FolderOpen size={15}/> Locate source study…</button>}
        {source&&scan&&!volume&&!loading&&<button className="outline-button" onClick={()=>void loadScan(source,scan)}>Retry scan</button>}
        {volume&&scan&&<div className="native-wall-source-view">
          <ViewerWorkspace key={scan.seriesId} caseData={volume} visibleLayers={visible} cursor={cursor} onCursorChange={setCursor}
            workspaceSettings={settings} onWorkspaceSettingsChange={setSettings} sourceLabel={scan.modality}
            overlayOpacity={.34} cameraMode="anatomy" routes={[]}/>
        </div>}
      </Dialog.Content>
    </Dialog.Portal>
  </Dialog.Root>;
}

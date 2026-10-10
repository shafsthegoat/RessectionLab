import {useCallback,useEffect,useRef,useState} from 'react';
import {hydrateDisplaySeries} from './workspace-imaging-data';
import {defaultImageView} from './workspace-session';
import type {CasePayload,ResectionApi,Vec3} from './types';
import type {DisplayImagingApi,DisplaySeriesView,ImportDisplaySeriesRequest} from './workspace-imaging-types';
import type {ImageViewState,ImagingSessionState} from './workspace-session-types';
export function useWorkspaceImaging(primary:CasePayload|null,api:(ResectionApi&DisplayImagingApi)|null) {
  const [series,setSeries]=useState<DisplaySeriesView[]>([]);
  const [selectedId,setSelectedId]=useState<string|null>(null);
  const [states,setStates]=useState<Record<string,ImageViewState>>({});
  const [pending,setPending]=useState(false);
  const current=useRef(primary?.caseHash);current.current=primary?.caseHash;
  const owner=useRef<string|undefined>(primary?.caseHash);
  const generation=useRef(0);
  const mounted=useRef(true);useEffect(()=>{mounted.current=true;return()=>{mounted.current=false;generation.current++}},[]);
  const invalidate=useCallback(()=>{generation.current++;setPending(false)},[]);
  const restore=useCallback((caseHash:string,views:DisplaySeriesView[],saved:ImagingSessionState|null)=>{
    generation.current++;owner.current=caseHash;current.current=caseHash;setPending(false);
    setSeries(views);setSelectedId(saved?.selectedSeriesId??null);
    setStates(saved?structuredClone(saved.states):Object.fromEntries(views.map(view=>[view.descriptor.seriesId,defaultImageView(view.volume)])));
  },[]);
  useEffect(()=>{
    if(owner.current===primary?.caseHash)return;
    generation.current++;owner.current=primary?.caseHash;setPending(false);setSeries([]);setSelectedId(null);setStates({});
  },[primary?.caseHash]);
  const selected=series.find(s=>s.descriptor.seriesId===selectedId&&s.descriptor.referenceCaseHash===primary?.caseHash)??null;
  const resetDisplayToPrimary=useCallback(()=>setSelectedId(null),[]);
  const select=useCallback((id:string|null)=>{
    if(id===null){resetDisplayToPrimary();return}
    const view=series.find(s=>s.descriptor.seriesId===id&&s.descriptor.referenceCaseHash===current.current);
    if(!view)throw new Error('Choose an image attached to the current workspace.');
    setSelectedId(id);setStates(all=>all[id]?all:{...all,[id]:defaultImageView(view.volume)});
  },[series,resetDisplayToPrimary]);
  const setCursor=useCallback((cursor:Vec3)=>{
    if(!selected)return;const id=selected.descriptor.seriesId;
    setStates(all=>({...all,[id]:{...(all[id]??defaultImageView(selected.volume)),cursor:[...cursor]}}));
  },[selected]);
  const setVisibleLayers=useCallback((update:Record<string,boolean>|((previous:Record<string,boolean>)=>Record<string,boolean>))=>{
    if(!selected)return;const id=selected.descriptor.seriesId;
    setStates(all=>{const view=all[id]??defaultImageView(selected.volume);return {...all,[id]:{...view,visibleLayers:typeof update==='function'?update(view.visibleLayers):{...update}}}});
  },[selected]);
  const snapshot=useCallback((primaryView:ImageViewState):ImagingSessionState=>({selectedSeriesId:selected?.descriptor.seriesId??null,
    states:{primary:structuredClone(primaryView),...Object.fromEntries(series.map(view=>[view.descriptor.seriesId,structuredClone(states[view.descriptor.seriesId]??defaultImageView(view.volume))]))}}),[selected,series,states]);
  const importSeries=useCallback(async(request:ImportDisplaySeriesRequest)=>{
    if(!primary||!api?.importDisplaySeries||request.caseHash!==primary.caseHash||pending)return;
    const source=primary,requestGeneration=++generation.current;setPending(true);
    const live=()=>mounted.current&&generation.current===requestGeneration&&current.current===source.caseHash;
    try{
      const response=await api.importDisplaySeries(request);
      if(!response||!live())return;
      const view=await hydrateDisplaySeries(source,response,api);
      if(!live())return;
      setSeries(all=>[...all.filter(s=>s.descriptor.seriesId!==view.descriptor.seriesId),view]);
      setSelectedId(view.descriptor.seriesId);
      setStates(all=>({...all,[view.descriptor.seriesId]:all[view.descriptor.seriesId]??defaultImageView(view.volume)}));
    }catch(error){if(live())throw error}
    finally{if(live())setPending(false)}
  },[api,primary,pending]);
  const view=selected?(states[selected.descriptor.seriesId]??defaultImageView(selected.volume)):null;
  return {series,selected,selectedId:selected?.descriptor.seriesId??null,pending,cursor:view?.cursor??null,setCursor,
    visibleLayers:view?.visibleLayers??{},setVisibleLayers,select,importSeries,resetDisplayToPrimary,snapshot,restore,invalidate};
}

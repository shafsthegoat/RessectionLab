import {useCallback,useEffect,useRef,useState} from 'react';
import {initialCursor} from './case-data.ts';
import {hydrateDisplaySeries} from './workspace-imaging-data';
import type {CasePayload,ResectionApi,Vec3} from './types';
import type {DisplayImagingApi,DisplaySeriesView,ImportDisplaySeriesRequest} from './workspace-imaging-types';
export function useWorkspaceImaging(primary:CasePayload|null,api:(ResectionApi&DisplayImagingApi)|null) {
  const [series,setSeries]=useState<DisplaySeriesView[]>([]);
  const [selectedId,setSelectedId]=useState<string|null>(null);
  const [pending,setPending]=useState(false);
  const [cursor,setCursor]=useState<Vec3|null>(null);
  const [visibleLayers,setVisibleLayers]=useState<Record<string,boolean>>({});
  const current=useRef(primary?.caseHash);current.current=primary?.caseHash;
  const mounted=useRef(true);useEffect(()=>{mounted.current=true;return()=>{mounted.current=false}},[]);
  useEffect(()=>{setSeries([]);setSelectedId(null);setCursor(null);setVisibleLayers({})},[primary?.caseHash]);
  const selected=series.find(s=>s.descriptor.seriesId===selectedId&&s.descriptor.referenceCaseHash===primary?.caseHash)??null;
  const resetDisplayToPrimary=useCallback(()=>setSelectedId(null),[]);
  const select=useCallback((id:string|null)=>{
    if(id===null){resetDisplayToPrimary();return}
    const view=series.find(s=>s.descriptor.seriesId===id&&s.descriptor.referenceCaseHash===current.current);
    if(!view)throw new Error('Choose an image attached to the current workspace.');
    setSelectedId(id);setCursor(initialCursor(view.volume));
    setVisibleLayers(Object.fromEntries(view.volume.compartments.map(layer=>[layer.name,true])));
  },[series,resetDisplayToPrimary]);
  const importSeries=useCallback(async(request:ImportDisplaySeriesRequest)=>{
    if(!primary||!api?.importDisplaySeries||request.caseHash!==primary.caseHash||pending)return;
    const source=primary;setPending(true);
    try{
      const response=await api.importDisplaySeries(request);
      if(!response||!mounted.current||current.current!==source.caseHash)return;
      const view=await hydrateDisplaySeries(source,response,api);
      if(!mounted.current||current.current!==source.caseHash)return;
      setSeries(all=>[...all.filter(s=>s.descriptor.seriesId!==view.descriptor.seriesId),view]);
      setSelectedId(view.descriptor.seriesId);setCursor(initialCursor(view.volume));
      setVisibleLayers(Object.fromEntries(view.volume.compartments.map(layer=>[layer.name,true])));
    }finally{if(mounted.current)setPending(false)}
  },[api,primary,pending]);
  return {series,selected,selectedId:selected?.descriptor.seriesId??null,pending,cursor,setCursor,visibleLayers,setVisibleLayers,select,importSeries,resetDisplayToPrimary};
}

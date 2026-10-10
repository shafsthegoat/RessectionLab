import {useCallback,useEffect,useRef,useState} from 'react';
import type {ResectionApi} from './types';
import type {DisplaySeriesView} from './workspace-imaging-types';
import {hydrateDiagnostic} from './scan-diagnostic-data';
import type {DiagnosticView} from './scan-diagnostic-data';
export function useDiagnosticLayer(selected:DisplaySeriesView|null,api:ResectionApi|null){
  const [view,setView]=useState<DiagnosticView|null>(null),[pending,setPending]=useState(false),[error,setError]=useState<string|null>(null);
  const current=useRef(selected);current.current=selected;const generation=useRef(0),mounted=useRef(true);
  useEffect(()=>{mounted.current=true;return()=>{mounted.current=false;generation.current++}},[]);
  const clear=useCallback(()=>{generation.current++;setPending(false);setView(null);setError(null)},[]);
  useEffect(clear,[selected,clear]);
  const load=useCallback(async()=>{
    if(!selected||!api?.importDiagnosticLayer||pending)return;
    const source=selected,id=++generation.current;setPending(true);setError(null);setView(null);
    const live=()=>mounted.current&&current.current===source&&generation.current===id;
    try{const result=await api.importDiagnosticLayer({caseHash:source.descriptor.referenceCaseHash,seriesId:source.descriptor.seriesId});
      if(!result||!live())return;const hydrated=await hydrateDiagnostic(result,source,api);if(live())setView(hydrated);
    }catch(cause){if(live())setError(cause instanceof Error?cause.message:String(cause));}
    finally{if(live())setPending(false);}
  },[selected,api,pending]);
  return {view:view?.source===selected?view:null,pending,error,load,clear};
}

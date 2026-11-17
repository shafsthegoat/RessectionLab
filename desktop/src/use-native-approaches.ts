import {useEffect,useRef,useState} from 'react';
import type {Binding,NativeAnatomyApi,Args} from './native-anatomy-data';
import type {ApproachResponse,NativeApproach} from './native-approaches';
import {approachRoutes} from './native-approaches';
export function useNativeApproaches(binding:Binding|null,api:NativeAnatomyApi|null,ready:boolean){
  const [approaches,setApproaches]=useState<NativeApproach[]>([]),[selectedId,setSelectedId]=useState<string|null>(null);
  const [pending,setPending]=useState(false),[error,setError]=useState(''),[report,setReport]=useState<ApproachResponse['proposalReport']>();
  const generation=useRef(0),active=useRef(false);
  const loadedKey=useRef('');
  useEffect(()=>{const seq=++generation.current,key=binding?.collectionId+'/'+binding?.document.snapshot_id;
    if(key!==loadedKey.current){setApproaches([]);setError('');setReport(undefined);setSelectedId(binding?.viewState?.selectedApproachId??null);loadedKey.current=key;}
    if(!binding||!api||!ready)return;
    void api.anatomyPlan({collectionId:binding.collectionId,snapshotId:binding.document.snapshot_id,action:'list'}).then(raw=>{if(seq!==generation.current)return;const result=raw as ApproachResponse;setApproaches(result.approaches);setSelectedId(id=>result.approaches.some(a=>a.id===id)?id:result.approaches[0]?.id??null)}).catch(e=>{if(seq===generation.current)setError(String(e))});
  },[binding?.collectionId,binding?.document.snapshot_id,api,ready]);
  async function request(input:Args){if(!api||!binding||active.current)return;const seq=++generation.current;active.current=true;setPending(true);setError('');try{
    const result=await api.anatomyPlan({collectionId:binding.collectionId,snapshotId:binding.document.snapshot_id,...input}) as ApproachResponse;
    if(seq!==generation.current)return;setApproaches(result.approaches);setSelectedId(id=>result.selectedId??(result.approaches.some(a=>a.id===id)?id:result.approaches[0]?.id??null));setReport(result.proposalReport);return result;
  }catch(e){if(seq===generation.current)setError(e instanceof Error?e.message:String(e))}finally{active.current=false;setPending(false)}}
  return {approaches,selectedId,select:setSelectedId,pending,error,report,request,selected:approaches.find(a=>a.id===selectedId)??null,routes:approachRoutes(approaches,selectedId)};
}

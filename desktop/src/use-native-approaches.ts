import {useEffect,useRef,useState} from 'react';
import type {Binding,NativeAnatomyApi,Args} from './native-anatomy-data';
import type {ApproachResponse} from './native-approaches';
import {approachRoutes} from './native-approaches';

const targetKey=(id?:string|null)=>id??'';
type Selection={key:string;id:string|null;byTarget:Record<string,string|null>};

export function useNativeApproaches(binding:Binding|null,api:NativeAnatomyApi|null,ready:boolean){
  const key=binding?binding.collectionId+'/'+binding.document.snapshot_id:'';
  const view=binding?.viewState;
  const savedSelection=view?.selectedApproachId;
  const savedTarget=view?.selectedPlanningTargetId;
  const [result,setResult]=useState<{key:string;data:ApproachResponse}>();
  const [selection,setSelection]=useState<Selection>();
  const [pending,setPending]=useState(false),[error,setError]=useState('');
  const load=useRef<{key:string;api:NativeAnatomyApi}|null>(null),active=useRef(false),selectionVersion=useRef(0);
  const comparisonRefresh=useRef<{key:string;api:NativeAnatomyApi}|null>(null);
  const liveKey=useRef(key);
  liveKey.current=key;

  function install(data:ApproachResponse,action:unknown,version:number){
    setResult(previous=>({key,data:{...(previous?.key===key?previous.data:undefined),...data,
      consequenceEvidence:data.consequenceEvidence??null}}));
    setSelection(current=>{
      if(current?.key===key&&selectionVersion.current!==version)return current;
      const scope=targetKey(data.activeTargetLayerId);
      const byTarget=current?.key===key?current.byTarget:{...view?.selectedApproachIds};
      const previous=Object.hasOwn(byTarget,scope)?byTarget[scope]:
        (action==='initial'&&(savedTarget===undefined||targetKey(savedTarget)===scope)?savedSelection:undefined);
      const preferred=action==='suggest'||data.selectedId?data.selectedId:previous;
      const first=data.approaches.find(a=>targetKey(a.targetLayerId)===scope)??data.approaches[0];
      const id=preferred===null?null:data.approaches.find(a=>a.id===preferred)?.id??first?.id??null;
      return {key,id,byTarget:{...byTarget,[scope]:id}};
    });
  }

  useEffect(()=>{
    if(load.current?.key!==key||load.current.api!==api)setError('');
    if(!binding||!api||!ready||(load.current?.key===key&&load.current.api===api))return;
    const started={key,api},version=selectionVersion.current;
    load.current=started;
    void api.anatomyPlan({collectionId:binding.collectionId,snapshotId:binding.document.snapshot_id,action:'list',
      ...(savedTarget===undefined?{}:{targetLayerId:savedTarget}),
      ...(savedSelection?{reviewApproachId:savedSelection}:{})}).then(raw=>{
      if(liveKey.current===key&&load.current===started)install(raw as ApproachResponse,'initial',version);
    }).catch(e=>{
      if(liveKey.current===key&&load.current===started)setError(e instanceof Error?e.message:String(e));
    });
  },[key,api,ready]);

  const data=result?.key===key?result.data:undefined;
  const activeTargetLayerId=data?.activeTargetLayerId;
  function select(id:string|null){
    selectionVersion.current++;
    setSelection(current=>({key,id,byTarget:{...(current?.key===key?current.byTarget:view?.selectedApproachIds),[targetKey(activeTargetLayerId)]:id}}));
  }

  async function request(input:Args){
    if(!api||!binding||!ready||active.current)return;
    const version=selectionVersion.current;
    active.current=true;
    setPending(true);
    setError('');
    try{
      const response=await api.anatomyPlan({collectionId:binding.collectionId,snapshotId:binding.document.snapshot_id,...input}) as ApproachResponse;
      if(liveKey.current!==key)return;
      install(response,input.action,version);
      return response;
    }catch(e){
      if(liveKey.current===key)setError(e instanceof Error?e.message:String(e));
    }finally{
      active.current=false;
      setPending(false);
    }
  }

  const approaches=data?.approaches??[],selectedId=selection?.key===key?selection.id:null;
  const evidence=data?.consequenceEvidence;
  const consequenceEvidence=evidence&&evidence.sourceSnapshotId===binding?.document.snapshot_id
    &&evidence.targetLayerId===activeTargetLayerId?evidence:null;
  const routes=approachRoutes(approaches,selectedId,activeTargetLayerId);
  const needsComparison=!!consequenceEvidence&&(routes.length!==consequenceEvidence.routes.length||
    routes.some(view=>!consequenceEvidence.routes.some(row=>row.approachId===view.route_id&&row.comparisonSlot===view.comparisonSlot)));
  // Selection updates the geometry immediately. Fetch evidence when the displayed group changes.
  const comparisonKey=JSON.stringify([key,activeTargetLayerId,selectedId,selectionVersion.current,routes.map(route=>route.route_id)]);
  useEffect(()=>{
    if(!needsComparison){comparisonRefresh.current=null;return;}
    if(!api||!ready||pending||active.current||
      (comparisonRefresh.current?.key===comparisonKey&&comparisonRefresh.current.api===api))return;
    comparisonRefresh.current={key:comparisonKey,api};
    void request({action:'list',targetLayerId:activeTargetLayerId,reviewApproachId:selectedId??undefined});
  },[needsComparison,comparisonKey,api,ready,pending]);
  return {
    approaches,selectedId,select,pending,error,loading:!!binding&&!data&&!error,activeTargetLayerId,targetSelectionIssue:data?.targetSelectionIssue,
    selectedApproachIds:selection?.key===key?selection.byTarget:view?.selectedApproachIds,
    targetOptions:data?.targetOptions??[],rehearsalOptions:data?.rehearsalOptions,report:data?.proposalReport,distinctEntryReport:data?.distinctEntryReport,inputComparison:data?.inputComparison,
    consequenceEvidence,request,
    selected:approaches.find(a=>a.id===selectedId)??null,routes,
  };
}

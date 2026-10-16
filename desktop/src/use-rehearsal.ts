import {useEffect,useMemo,useRef,useState} from 'react';
import type {ViewerVolume} from './viewer/contracts';
import type {RehearsalAction,RehearsalApi,RehearsalResponse,RehearsalViewPreferences} from './rehearsal-types';
import {checkedRehearsalResponse,rehearsalViewerState} from './rehearsal-data';

export function useRehearsal(volume:ViewerVolume|null,api:Partial<RehearsalApi>|null) {
  const [result,setResult]=useState<RehearsalResponse|null>(null),[selectedBranch,setSelectedBranch]=useState('live'),
    [selectedTool,setSelectedTool]=useState(''),[frameIndex,setFrame]=useState(0),[playing,setPlaying]=useState(false),
    [visible,setVisible]=useState(false),[pending,setPending]=useState(false),[error,setError]=useState<string|null>(null);
  const current=useRef({volume,api,result,pending});current.current={volume,api,result,pending};
  const generation=useRef(0),busy=useRef(false);
  const caseHash=volume?.caseHash;
  useEffect(()=>{generation.current++;busy.current=false;setResult(null);setSelectedBranch('live');setSelectedTool('');setFrame(0);setPlaying(false);setVisible(false);setPending(false);setError(null);return()=>{generation.current++;};},[caseHash]);
  const checked=result?.caseHash===caseHash?result:null;
  const activeEpisode=(selectedBranch==='live'?checked?.episode:checked?.comparison?.alternatives.find(a=>a.id===selectedBranch)?.episode)??null;
  const selectedFrame=Math.min(frameIndex,Math.max(0,(activeEpisode?.replayFrames.length??1)-1));
  const rendered=useMemo(()=>{
    if(!visible||!volume||!activeEpisode)return {state:null,error:null};
    try{return {state:rehearsalViewerState(volume,activeEpisode,selectedFrame),error:null};}
    catch(cause){return {state:null,error:cause instanceof Error?cause.message:String(cause)};}
  },[visible,volume,activeEpisode,selectedFrame]);
  useEffect(()=>{
    if(!playing||!activeEpisode)return;
    if(selectedFrame>=activeEpisode.replayFrames.length-1){setPlaying(false);return;}
    const now=activeEpisode.replayFrames[selectedFrame],next=activeEpisode.replayFrames[selectedFrame+1];
    const timer=setTimeout(()=>setFrame(selectedFrame+1),Math.max(100,Math.min(1400,(next.elapsedSeconds-now.elapsedSeconds)*650)));
    return()=>clearTimeout(timer);
  },[playing,activeEpisode,selectedFrame]);
  async function run(kind:'create'|'inspect'|'step'|'compare',action?:RehearsalAction) {
    const v=current.current.volume,a=current.current.api,old=current.current.result;
    if(!v||!a||busy.current||((kind==='step'||kind==='compare')&&(!old||old.caseHash!==v.caseHash)))return;
    if(!a.createRehearsal||!a.inspectRehearsal||!a.stepRehearsal||!a.compareRehearsal){setError('Research rehearsal is unavailable in this application session.');return;}
    const token=++generation.current;busy.current=true;setPending(true);setPlaying(false);setError(null);
    const valid=()=>token===generation.current&&current.current.volume?.caseHash===v.caseHash;
    try {
      const input={caseHash:v.caseHash,expectedStateId:old?.stateId??''};
      const raw=kind==='create'?await a.createRehearsal({caseHash:v.caseHash}):kind==='inspect'?await a.inspectRehearsal({caseHash:v.caseHash}):
        kind==='compare'?await a.compareRehearsal(input):await a.stepRehearsal({...input,action:action!});
      if(!valid())return;
      if(!raw){setError('No saved rehearsal exists for this source. Prepare a rehearsal to begin.');return;}
      const next=checkedRehearsalResponse(v,raw);
      if(kind==='step'&&raw.accepted===false){setError(raw.reason??'The bounded action was rejected; the saved state was preserved.');setResult(next);return;}
      setResult(next);setSelectedBranch('live');setSelectedTool(previous=>next.episode.tools.some(t=>t.toolId===previous)?previous:next.episode.tools[0].toolId);
      const saved=kind==='inspect'?raw.savedView:null;
      const savedEpisode=saved?.branchId==='live'?next.episode:next.comparison?.alternatives.find(a=>a.id===saved?.branchId)?.episode;
      if(saved&&savedEpisode&&saved.episodeInitialStateId===next.episode.initialStateId&&Number.isInteger(saved.frameIndex)&&saved.frameIndex>=0&&saved.frameIndex<savedEpisode.replayFrames.length&&typeof saved.visible==='boolean'&&savedEpisode.tools.some(t=>t.toolId===saved.selectedTool)){
        setSelectedBranch(saved.branchId);setSelectedTool(saved.selectedTool);setFrame(saved.frameIndex);setVisible(saved.visible);
      }else{setFrame(kind==='create'?0:next.episode.replayFrames.length-1);setVisible(true);}
    }catch(cause){if(valid())setError(cause instanceof Error?cause.message:String(cause));}
    finally{if(valid()){busy.current=false;setPending(false);}}
  }
  function selectBranch(id:string) {
    if(pending||!checked||(id!=='live'&&!checked.comparison?.alternatives.some(a=>a.id===id)))return;
    const branch=checked.comparison?.alternatives.find(a=>a.id===id);
    const branchStart=branch?.episode.replayFrames.findIndex(f=>f.stateAfter===checked.comparison?.snapshotId)??0;
    setPlaying(false);setSelectedBranch(id);setFrame(id==='live'?checked.episode.replayFrames.length-1:Math.max(0,branchStart));setVisible(true);
  }
  const viewPreferences:RehearsalViewPreferences|null=checked?{branchId:selectedBranch,frameIndex:selectedFrame,selectedTool,visible,episodeInitialStateId:checked.episode.initialStateId}:null;
  return {viewPreferences,episode:checked?.episode??null,comparison:checked?.comparison??null,activeEpisode,
    suggestedActions:checked?.suggestedActions??[],viewerState:rendered.state,selectedBranch,selectedTool,frameIndex:selectedFrame,
    playing,visible,pending,error:error??rendered.error,create:()=>run('create'),inspect:()=>run('inspect'),step:(action:RehearsalAction)=>run('step',action),compare:()=>run('compare'),
    selectBranch,selectTool:(id:string)=>{if(activeEpisode?.tools.some(t=>t.toolId===id))setSelectedTool(id);},
    setFrameIndex:(index:number)=>{if(activeEpisode&&Number.isInteger(index)&&index>=0&&index<activeEpisode.replayFrames.length){setPlaying(false);setFrame(index);setVisible(true);}},
    play:()=>{if(activeEpisode&&activeEpisode.replayFrames.length>1){if(selectedFrame===activeEpisode.replayFrames.length-1)setFrame(0);setVisible(true);setPlaying(true);}},
    pause:()=>setPlaying(false),show:()=>setVisible(true),hide:()=>{setVisible(false);setPlaying(false);}};
}
export type RehearsalController=ReturnType<typeof useRehearsal>;

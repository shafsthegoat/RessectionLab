import {useEffect,useRef,useState} from 'react';
import type {SurgicalController} from './use-surgical';

const paint=()=>new Promise<void>(resolve=>requestAnimationFrame(()=>resolve()));

function presentedCaptureArea(stateId:string) {
  const pane=Array.from(document.querySelectorAll<HTMLElement>('[data-accepted-tools]'))
    .find(element=>element.dataset.surgicalStateId===stateId&&element.closest<HTMLElement>('.rl-viewer')?.dataset.surfacesPending==='0');
  if(!pane)return null;
  const layout=pane.closest<HTMLElement>('[data-engine-replay-layout]');
  if(!layout)return pane;
  const inspection=layout.querySelector<HTMLElement>('canvas[data-engine-inspection-state]');
  if(inspection?.dataset.engineInspectionState!==stateId)return null;
  const result=pane.closest<HTMLElement>('.rl-viewer[data-result-inspection]');
  if(result&&result.querySelector<HTMLElement>('.engine-result-orbit')?.dataset.resultState!==stateId)return null;
  const preparing=Array.from(layout.querySelectorAll<HTMLElement>('.rl-viewer'))
    .some(view=>view.clientWidth>0&&view.dataset.rendererReady!=='true');
  return preparing?null:layout;
}

/** Walk the manual seekFrame path. Canvas and controller must both acknowledge
 * the exact observation before the compositor image can enter the movie. */
export function SurgicalMovieExport({controller,disabled}:{controller:SurgicalController;disabled:boolean}) {
  const latest=useRef(controller);
  latest.current=controller;
  const active=useRef<AbortController|null>(null);
  const [running,setRunning]=useState(false),[message,setMessage]=useState('');
  const record=globalThis.window?.resectionApi?.recordEngineMovie;
  const source=controller.result?.caseHash,branch=controller.result?.branchId;
  useEffect(()=>()=>{active.current?.abort();},[source,branch]);
  if(!record) return null;

  async function exportMovie() {
    const initial=latest.current.result;
    if(!initial||!record||active.current) return;
    const steps=initial.branches.find(b=>b.id===initial.branchId)?.steps??0;
    const task=new AbortController();
    active.current=task;setRunning(true);setMessage('Choosing movie destination…');
    let id:string|null=null,moved=false;
    function check() {
      task.signal.throwIfAborted();
      if(latest.current.result?.caseHash!==source || latest.current.result?.branchId!==branch)
        throw Error('The recorded branch changed during export.');
    }
    const cancel=()=>{if(id)void record({command:'cancel',id}).catch(()=>{});};
    task.signal.addEventListener('abort',cancel,{once:true});
    try {
      const opened=await record({command:'begin',caseHash:initial.caseHash,branchId:initial.branchId});
      if(typeof opened!=='string'){setMessage('');return;}
      id=opened;check();
      // Seed the recorder with a full timeline before compact frame responses.
      moved=true;
      if(!await latest.current.run('seek',{branchId:branch,position:0})) throw Error('Unable to open the initial recorded state.');
      for(let position=0;position<steps;position++) {
        let frameCount=1;
        for(let frameIndex=0;frameIndex<frameCount;frameIndex++) {
          check();
          const selected={branchId:branch,position,frameIndex};
          if(!await latest.current.run('seekFrame',selected)) throw Error(latest.current.error??'Engine observation unavailable.');
          const deadline=performance.now()+120000;
          for(;;) {
            await paint();check();
            const c=latest.current,r=c.result,p=r?.playback;
            const area=r&&presentedCaptureArea(r.view.stateId);
            if(performance.now()>deadline) throw Error(c.error??'Timed out waiting for the displayed operative field.');
            if(!c.pending && r && p?.position===position && p.frameIndex===frameIndex && area) {
              frameCount=p.frameCount;
              await paint();check();
              if(presentedCaptureArea(r.view.stateId)!==area)continue;
              const bounds=area.getBoundingClientRect(),x=Math.ceil(bounds.left),y=Math.ceil(bounds.top);
              const rect={x,y,width:Math.floor(bounds.right)-x,height:Math.floor(bounds.bottom)-y};
              await record({command:'capture',id,stateId:r.view.stateId,rect});
              break;
            }
          }
        }
        setMessage('Captured action '+(position+1)+' of '+steps+'.');
      }
      check();setMessage('Encoding the observed replay…');
      const result=await record({command:'finish',id});
      id=null;
      setMessage(typeof result==='object'&&result?'Exported '+result.captured+' engine observations · '+result.simulatedSeconds.toFixed(2)+' simulated s.':'Exported observed replay.');
    } catch(error) {
      setMessage(task.signal.aborted?'Export cancelled.':String(error));
    } finally {
      task.signal.removeEventListener('abort',cancel);
      if(id) await record({command:'cancel',id}).catch(()=>{});
      if(moved && latest.current.result?.caseHash===source && latest.current.result?.branchId===branch) {
        const p=initial.playback;
        await latest.current.run(p?'seekFrame':'seek',p?{branchId:branch,position:p.position,frameIndex:p.frameIndex}:{branchId:branch,position:initial.position});
      }
      active.current=null;setRunning(false);
    }
  }
  return <div className="surgical-actions">
    <button disabled={!running&&disabled} onClick={()=>running?active.current?.abort():void exportMovie()}>{running?'Cancel movie export':'Export observed replay'}</button>
    {message&&<small role="status" aria-label="Replay export status">{message}</small>}
  </div>;
}

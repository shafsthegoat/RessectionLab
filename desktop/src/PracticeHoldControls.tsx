import {useEffect, useRef, useState} from 'react';
import {practiceHoldLimit, practiceHoldPose, practiceHoldStepMs, practiceHoldStop} from './practice-input';
import type {PracticeHoldKind} from './practice-input';
import type {SurgicalAction, SurgicalResponse} from './surgical-types';
import type {SurgicalController} from './use-surgical';

type Hold = {
  kind: PracticeHoldKind;
  input: number | string;
  slot: string;
  instrument: string;
  origin: SurgicalResponse;
  receipt: SurgicalResponse | null;
  abort: AbortController;
  steps: number;
  nextAt: number;
  deadline?: ReturnType<typeof setTimeout>;
};

const labels = {advance:'Hold to advance',retract:'Hold to retract',apply:'Hold to apply'} as const;
const operating = {advance:'Advancing',retract:'Retracting',apply:'Applying at tip'} as const;

export function PracticeHoldControls({c, blocked, params, onRequest}: {
  c:SurgicalController;
  blocked:boolean;
  params:Record<string,number>;
  onRequest:(action:SurgicalAction, origin:SurgicalResponse)=>void;
}) {
  const active = useRef<Hold | null>(null);
  const latest = useRef({c,blocked,params,onRequest});
  latest.current = {c,blocked,params,onRequest};
  const [version,refresh] = useState(0);
  const [status,setStatus] = useState('');
  function stop(hold: Hold, reason: string, discard = false) {
    if (active.current !== hold) return;
    active.current = null;
    clearTimeout(hold.deadline);
    hold.abort.abort();
    if (discard && latest.current.c.result === hold.origin) latest.current.c.cancel();
    setStatus(reason);
    refresh(n=>n+1);
  }
  function stopInput(input?: number | string) {
    const hold=active.current;
    if (hold && (input === undefined || input === hold.input)) stop(hold,'Hold released.',true);
  }
  async function step(hold: Hold) {
    if (active.current !== hold || hold.abort.signal.aborted) return;
    const {c,blocked,params,onRequest} = latest.current;
    const slot = c.view?.slots[hold.slot];
    if (blocked || !slot || c.result !== hold.origin || c.slot !== hold.slot || slot.instrument !== hold.instrument) {
      stop(hold,'Practice state changed.');return;
    }
    const pose = practiceHoldPose(hold.kind,slot);
    const action = c.action(hold.kind === 'apply' ? 'apply' : 'move',params,practiceHoldStepMs/1000);
    action.path=[pose];
    hold.steps++;
    hold.nextAt=performance.now()+practiceHoldStepMs;
    c.setPose(pose);
    onRequest(action,hold.origin);
    try {
      const receipt=await c.runPracticeAction(action,hold.origin,hold.abort.signal);
      if (active.current !== hold) return;
      if (!receipt) {stop(hold,'Hold stopped.',true);return;}
      hold.receipt=receipt;
      refresh(n=>n+1);
    } catch {
      stop(hold,'Instrument request failed.',true);
    }
  }
  function start(kind: PracticeHoldKind, input: number | string) {
    const {c,blocked} = latest.current;
    const result=c.result, slot=c.view?.slots[c.slot];
    if (blocked || !result?.practiceEpisode || !slot || active.current) return;
    const hold:Hold={kind,input,slot:c.slot,instrument:slot.instrument,origin:result,receipt:null,
      abort:new AbortController(),steps:0,nextAt:0};
    active.current=hold;
    hold.deadline=setTimeout(()=>stop(hold,'Time limit reached. Hold again to continue.',true),8000);
    setStatus(operating[kind]);
    refresh(n=>n+1);
    void step(hold);
  }

  // A backend receipt can arrive before its surfaces. Re-arm only after
  // React has received that exact GPU-admitted snapshot and all work is idle.
  useEffect(()=>{
    const hold=active.current;
    if (!hold) return;
    if (c.slot !== hold.slot || c.result?.branchId !== hold.origin.branchId || c.result?.playback || c.result?.readOnly) {
      stop(hold,'Practice state changed.');return;
    }
    if (c.pending || !hold.receipt) return;
    if (c.result !== hold.receipt || blocked) {stop(hold,'Displayed checkpoint changed.');return;}
    const reason=practiceHoldStop(hold.origin,hold.receipt);
    if (reason || hold.steps >= practiceHoldLimit) {stop(hold,reason ?? '20 steps complete. Hold again to continue.');return;}
    const accepted=hold.receipt;
    const timer=setTimeout(()=>{
      if (active.current !== hold) return;
      hold.origin=accepted;
      hold.receipt=null;
      void step(hold);
    },Math.max(0,hold.nextAt-performance.now()));
    return ()=>clearTimeout(timer);
  },[version,c.result,c.pending,c.slot,blocked]);

  useEffect(()=>{
    const blur=()=>stopInput();
    const visibility=()=>{if(document.hidden)stopInput();};
    const escape=(event:globalThis.KeyboardEvent)=>{if(event.key==='Escape'&&active.current){event.preventDefault();stopInput();}};
    window.addEventListener('blur',blur);
    window.addEventListener('keydown',escape,true);
    document.addEventListener('visibilitychange',visibility);
    return ()=>{
      window.removeEventListener('blur',blur);
      window.removeEventListener('keydown',escape,true);
      document.removeEventListener('visibilitychange',visibility);
      const hold=active.current;
      active.current=null;
      if(hold){clearTimeout(hold.deadline);hold.abort.abort();}
    };
  },[]);
  const held=active.current?.kind;
  return <section aria-label="Continuous instrument input" data-practice-hold={held ?? 'idle'}>
    <div className="practice-hold-buttons">{(Object.keys(labels) as PracticeHoldKind[]).map(kind=>
      <button type="button" key={kind} aria-pressed={held === kind} disabled={held ? held !== kind : blocked}
        onPointerDown={e=>{
          if(e.button!==0 || !e.isPrimary || blocked || active.current)return;
          e.currentTarget.focus({preventScroll:true});
          e.currentTarget.setPointerCapture(e.pointerId);
          start(kind,e.pointerId);
        }}
        onPointerUp={e=>stopInput(e.pointerId)} onPointerCancel={e=>stopInput(e.pointerId)}
        onLostPointerCapture={e=>stopInput(e.pointerId)} onBlur={()=>stopInput()}
        onKeyDown={e=>{
          if(e.key!==' ' && e.key!=='Enter')return;
          e.preventDefault();if(!e.repeat)start(kind,e.key);
        }}
        onKeyUp={e=>{if(e.key===' '||e.key==='Enter'){e.preventDefault();stopInput(e.key);}}}
      >{labels[kind]}</button>)}</div>
    <p className="surgical-note">Hold with mouse, Space or Enter. Advance or retract 0.5 mm per step, or apply at the tip for 0.1 s. Release to stop. A step already committing finishes. Each hold allows up to 20 steps.</p>
    {status&&<p role="status" className="practice-hold-status">{status}</p>}
  </section>;
}

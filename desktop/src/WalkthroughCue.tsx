import {useState} from 'react';
import type {SurgicalAction, SurgicalResponse} from './surgical-types';
import type {SurgicalController} from './use-surgical';
import type {Point3} from './viewer/coordinates';

export type PracticeCue = {
  status:'ready'; position:number; count:number; label:string; phase:string;
  action:SurgicalAction;
} | {status:'complete'|'modified'; position:number; count:number};

export function WalkthroughCue({c, cue, blocked, origin}:{
  c:SurgicalController; cue:PracticeCue; blocked:boolean; origin:SurgicalResponse;
}) {
  const [seconds, setSeconds] = useState(cue.status==='ready' ? cue.action.duration??1 : 1);
  const [editedTip, setEditedTip] = useState<Point3|null>(null);
  if (cue.status==='complete') return <p role="status">Recorded rehearsal complete · {cue.count} actions</p>;
  if (cue.status!=='ready') return <p role="status">Your episode follows a different path. Continue with the manual controls.</p>;
  const action = {...cue.action,duration:seconds};
  const lastPose = cue.action.path?.at(-1) ?? cue.action.pose;
  if (editedTip && lastPose) {
    const pose = {...lastPose, position:editedTip};
    if (action.path?.length) action.path = [...action.path.slice(0,-1),pose];
    else action.pose = pose;
  }
  const toolId = action.instrument ?? origin.view.slots[action.slot]?.instrument;
  const tool = origin.view.catalog.find(item=>item.id===toolId)?.name ?? toolId;
  const valid = Number.isFinite(seconds) && seconds>0 && seconds<=30 && (!editedTip||editedTip.every(Number.isFinite));
  return <section className="walkthrough-cue" aria-label="Guided rehearsal">
    <small>{cue.phase} · Action {cue.position+1} of {cue.count}</small>
    <h3>{cue.label}</h3><p>{action.slot} hand · {tool}</p>
    <div className="surgical-actions">
      <button disabled={blocked||!valid} onClick={()=>void c.run('preview',{action})}>Preview maneuver</button>
      <button className="primary-button" disabled={blocked||!valid} onClick={()=>void c.runPracticeAction(action,origin)}>Perform maneuver</button>
      {c.preview&&<button onClick={c.clearPreview}>Dismiss preview</button>}
    </div>
    <details><summary>Maneuver settings</summary>
      <label>Duration · s<input aria-label="Guided maneuver duration" type="number" min=".01" max="30" step=".01"
        disabled={blocked} value={Number.isNaN(seconds)?'':seconds} onChange={e=>setSeconds(e.currentTarget.valueAsNumber)}/></label>
      {lastPose && <fieldset disabled={blocked}><legend>Final tip · RAS mm</legend>
        <div className="surgical-actions">{['R','A','S'].map((axis,i)=>{
          const tip = editedTip ?? lastPose.position;
          return <label key={axis}>{axis}<input type="number" aria-label={'Guided tip '+axis} step=".1"
            value={Number.isNaN(tip[i])?'':tip[i]} onChange={e=>{
              const next:Point3 = [...tip];
              next[i] = e.currentTarget.valueAsNumber;
              setEditedTip(next);
            }}/></label>;
        })}</div>{editedTip&&<button onClick={()=>setEditedTip(null)}>Restore recorded tip</button>}
      </fieldset>}
      <p className="surgical-note">{action.operation} · {action.path?.length??(action.pose?1:0)} pose samples</p>
    </details>
  </section>;
}

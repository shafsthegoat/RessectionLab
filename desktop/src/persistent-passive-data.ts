import type {PersistentPassiveEpisode,PassiveOperativeState,PhysicalPose,PassiveActionId} from './persistent-passive-types';
import {sha256Bytes} from './source-integrity.ts';

const HASH=/^sha256:[a-f0-9]{64}$/;
const ACTIONS:readonly PassiveActionId[]=['PASSIVE_MOVE','ACQUIRE_GENERATED_STATUS','PASSIVE_WITHDRAW','PASSIVE_WITHDRAW_STATIONARY','STOP'];
const PHASES=['ready','held','inspected','withdrawn','both_withdrawn','stopped'] as const;
function need(ok:unknown,reason:string):asserts ok {if(!ok)throw Error(`Persistent passive replay withheld: ${reason}`);}
function stable(v:unknown):unknown {
  return Array.isArray(v)?v.map(stable):v&&typeof v==='object'?
    Object.fromEntries(Object.entries(v).sort(([a],[b])=>a.localeCompare(b)).map(([k,x])=>[k,stable(x)])):v;
}
const equal=(a:unknown,b:unknown)=>JSON.stringify(stable(a))===JSON.stringify(stable(b));
const pose=(v:unknown):v is PhysicalPose=>{
  const p=v as PhysicalPose;
  return !!p&&typeof p==='object'&&Array.isArray(p.tip_mm)&&Array.isArray(p.axis_unit)&&
    p.tip_mm.length===3&&p.axis_unit.length===3&&[...p.tip_mm,...p.axis_unit].every(Number.isFinite)&&
    Math.abs(Math.hypot(...p.axis_unit)-1)<1e-10;
};
const zero=(v:unknown)=>Array.isArray(v)&&v.length===0;

/** Validate a separate generated paired-tool history, never legacy auto-withdraw frames.
 * Backend fresh replay remains the authority for full-tool and pair geometry. */
export async function checkedPairedPassiveEpisode(raw:PersistentPassiveEpisode,canonical:string):Promise<PersistentPassiveEpisode>{
  const e=structuredClone(raw);
  need(typeof canonical==='string'&&canonical.length>0&&canonical.length<=2*1024*1024&&
    `sha256:${await sha256Bytes(new TextEncoder().encode(canonical))}`===e?.episodeId,'episode canonical digest changed');
  let parsed:unknown;
  try {parsed=JSON.parse(canonical);} catch {throw Error('Persistent passive replay withheld: invalid canonical episode JSON');}
  const {episodeId:_,...body}=e;
  need(equal(parsed,body)&&e.schema==='resectionlab.generated-persistent-passive-episode.v1'&&
    [e.episodeId,e.sourceHash,e.decisionModelHash,e.physicalHistoryHash,e.finalStateHash].every(h=>HASH.test(h)),
    'schema, hashes or canonical episode body changed');
  need(e.scope==='generated passive geometry and declared time only; no surgical or sensor validation'&&
    e.frameMeaning==='operative state after each committed passive action; not legacy native stroke frames',
    'physical or clinical scope changed');
  need(typeof e.physicalHistoryCanonicalJson==='string'&&e.physicalHistoryCanonicalJson.length<=1024*1024&&
    `sha256:${await sha256Bytes(new TextEncoder().encode(e.physicalHistoryCanonicalJson))}`===e.physicalHistoryHash,
    'history bytes changed');
  let physicalHistory:unknown;
  try {physicalHistory=JSON.parse(e.physicalHistoryCanonicalJson);} catch {throw Error('Persistent passive replay withheld: invalid canonical history JSON');}
  need(equal(physicalHistory,e.history)&&equal(e.actions,ACTIONS)&&e.history.length===5&&e.frames.length===6,
    'paired action/frame history changed');
  const p=e.protocol;
  need(p?.version==='generated-persistent-passive-v1'&&
    typeof p.moving_tool_id==='string'&&p.moving_tool_id.length>0&&
    typeof p.stationary_tool_id==='string'&&p.stationary_tool_id.length>0&&p.stationary_tool_id!==p.moving_tool_id&&
    pose(p.start_pose)&&pose(p.hold_pose)&&pose(p.stationary_pose)&&pose(p.stationary_exit_pose)&&
    Array.isArray(p.generated_seconds)&&p.generated_seconds.length===5&&
    p.generated_seconds.every((n,i)=>Number.isFinite(n)&&n>=0&&n<=30&&(i===4||n>0))&&
    p.time_meaning==='declared generated schedule; not measured surgical duration'&&
    p.movement_scope==='one passive rigid tool, other tool stationary; no active cutting',
    'paired protocol, tool identity or generated time changed');
  need(equal(p.start_pose.axis_unit,p.hold_pose.axis_unit)&&
    equal(p.stationary_pose.axis_unit,p.stationary_exit_pose.axis_unit),
    'fixed-axis protocol changed');
  const info=e.history[1]?.information_event;
  need(info?.version==='generated-registration-status-v1'&&
    (info.status==='verified'||info.status==='unresolved')&&HASH.test(info.input_fingerprint),
    'generated inspect event unavailable');
  let elapsed=0;
  const native=e.history[0]?.source_state_hash;
  need(HASH.test(native),'native ancestry absent');
  for(let i=0;i<6;i++){
    const frame=e.frames[i],s=frame?.operative as PassiveOperativeState;
    need(frame?.frameIndex===i&&s?.phase===PHASES[i]&&Number.isFinite(s.elapsed_generated_seconds)&&
      Math.abs(s.elapsed_generated_seconds-elapsed)<=1e-9&&
      s.inspection_status===(i<2?'not_acquired':info.status),
      'frame order, generated clock or information reveal changed');
    const movingPose=i===1||i===2?p.hold_pose:p.start_pose;
    const otherPose=i<4?p.stationary_pose:p.stationary_exit_pose;
    const movingHeld=i===1||i===2,otherHeld=i<4;
    need(s.moving_tool?.tool_id===p.moving_tool_id&&equal(s.moving_tool.pose,movingPose)&&
      s.moving_tool.retained_at_depth===movingHeld&&s.moving_tool.activation===(movingHeld?'passive_only':'off')&&
      s.moving_tool.pose_status===(movingHeld?'held_in_declared_void':'access_pose_recorded')&&
      s.moving_tool.physical_absence_claim===false&&
      s.other_tool?.tool_id===p.stationary_tool_id&&equal(s.other_tool.pose,otherPose)&&
      s.other_tool.retained_at_depth===otherHeld&&s.other_tool.activation==='off'&&
      s.other_tool.pose_status===(otherHeld?'held_in_declared_void':'access_pose_recorded')&&
      s.other_tool.physical_absence_claim===false,
      'paired poses, activation or physical presence changed');
    if(i===5)continue;
    const row=e.history[i];
    const beforePose=i===0?p.start_pose:i===1?p.hold_pose:i===2?p.hold_pose:i===3?p.stationary_pose:p.start_pose;
    const afterPose=i===0?p.hold_pose:i===1?p.hold_pose:i===2?p.start_pose:i===3?p.stationary_exit_pose:p.start_pose;
    const actionTool=i===3?p.stationary_tool_id:i===4?null:p.moving_tool_id;
    need(row?.action_id===ACTIONS[i]&&row.duration_generated_seconds===p.generated_seconds[i]&&
      row.tool_id===actionTool&&equal(row.pose_before,beforePose)&&equal(row.pose_after,afterPose)&&
      row.time_source==='fixed_generated_schedule_not_measured_surgical_time'&&
      row.source_state_hash===native&&row.result_state_hash===native&&
      HASH.test(row.operative_before)&&HASH.test(row.operative_after)&&
      (i===0||row.operative_before===e.history[i-1].operative_after)&&
      zero(row.removed_indices_native)&&zero(row.contact_indices_native)&&row.reward===0&&
      row.clinical_deficit_probability===null&&row.physiological_response===null,
      'action order, native/operative ancestry or no-effect contract changed');
    const expectedMode=i===1?'generated_information':i===4?'stop':'passive_motion';
    need(row.interaction_mode===expectedMode,'interaction type changed');
    if(expectedMode==='passive_motion')
      need(row.geometry?.feasible===true&&row.independent_tissue_geometry?.feasible===true&&
        row.independent_tissue_geometry.stationary_tool_pair_independently_checked===false&&
        row.information_event===null,'motion certificate or evidence scope changed');
    else need(row.geometry===null&&row.independent_tissue_geometry===null&&
      (i===1?equal(row.information_event,info):row.information_event===null),
      'information or STOP acquired a physical effect');
    if(i!==1)need(row.information_event===null,'generated status leaked before inspect');
    need(row.stop_reason===(i===4?(info.status==='unresolved'?'generated_registration_unresolved':'declared_stop_without_injury_claim'):null),
      'STOP reason changed');
    elapsed+=p.generated_seconds[i];
    need(Number.isFinite(elapsed)&&Math.abs(row.elapsed_generated_seconds-elapsed)<=1e-9,
      'action and frame clocks disagree');
  }
  return e;
}

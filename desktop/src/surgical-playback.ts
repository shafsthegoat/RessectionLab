/** Elapsed time recorded by the authoritative action result, never inferred
 * from frame index or requested duration (partial actions may stop early). */
export function surgicalPlaybackDelay(timeline:readonly {elapsedSeconds?:number}[],position:number):{milliseconds:number;recorded:boolean} {
  const elapsed=timeline[position]?.elapsedSeconds;
  if(typeof elapsed==='number'&&Number.isFinite(elapsed)&&elapsed>=0)return {milliseconds:elapsed*1000,recorded:true};
  // Historical results without timing remain explicit state-by-state review.
  return {milliseconds:600,recorded:false};
}

/** Wheel controls emit start/end in the same task. A gesture spans quiet time,
 * not that synchronous pair. Camera persistence remains the caller's job. */
export class InteractionSession {
  active = false;
  private timer: ReturnType<typeof setTimeout> | undefined;
  private readonly change: (active:boolean)=>void;
  private readonly quietMs:number;
  constructor(change:(active:boolean)=>void,quietMs=150) {this.change=change;this.quietMs=quietMs;}
  start() {
    clearTimeout(this.timer);
    this.timer = undefined;
    if (!this.active) { this.active = true; this.change(true); }
  }
  end() {
    clearTimeout(this.timer);
    this.timer = setTimeout(() => this.finish(), this.quietMs);
  }
  finish() {
    clearTimeout(this.timer); this.timer = undefined;
    if (this.active) { this.active = false; this.change(false); }
  }
  dispose() { clearTimeout(this.timer); this.timer = undefined; }
}

/** GPU completion polling is an upper bound, not CPU submission time. */
export class InteractionBudget {
  private tier = 1;
  private fast = 0;
  private slow = 0;
  readonly edges = [240, 360, 540, 720];
  get longEdge() { return this.edges[this.tier]; }
  sample(milliseconds: number) {
    if (!Number.isFinite(milliseconds)) return;
    this.slow = milliseconds > 18 ? this.slow + 1 : 0;
    this.fast = milliseconds < 14 ? this.fast + 1 : 0;
    if (this.slow >= 4 && this.tier > 0) { this.tier--; this.slow = this.fast = 0; }
    if (this.fast >= 32 && this.tier < this.edges.length - 1) { this.tier++; this.slow = this.fast = 0; }
  }
}

/** Path-traced motion is drawn about once per display refresh: the fence of
 * a frame still running at the next refresh is then polled every millisecond,
 * so a slightly late frame costs little. One seen more than half a refresh
 * late is a miss; two misses in eight frames lower the resolution at once. A
 * larger image is tried after a long run of hits when the slowest recent GPU
 * time scaled by area leaves wide room (clock scaling stretches GPU time two-
 * to threefold), or, where GPU timer queries are unavailable, after a run of
 * hits twice as long; a tier that failed asks for a longer run again. */
export class RefreshBudget {
  readonly edges = [200, 240, 280, 320, 360, 400, 440, 480];
  private tier = 4;
  private refresh = 1000 / 60;
  private lastFrame = -Infinity;
  private readonly deltas: number[] = [];
  private readonly recent: boolean[] = [];
  private readonly gpu: number[] = [];
  private hits = 0;
  private readonly failed = new Set<number>();
  get longEdge() { return this.edges[this.tier]; }
  get refreshMs() { return this.refresh; }
  /** Display refresh from animation-frame timestamps: the least step seen lately. */
  frame(time: number) {
    const delta = time - this.lastFrame;
    this.lastFrame = time;
    if (!(delta > 2 && delta < 50)) return;
    this.deltas.push(delta);
    if (this.deltas.length > 32) this.deltas.shift();
    this.refresh = Math.min(...this.deltas);
  }
  gpuTime(milliseconds: number) {
    if (!Number.isFinite(milliseconds)) return;
    this.gpu.push(milliseconds);
    if (this.gpu.length > 64) this.gpu.shift();
  }
  /** Submission to observed fence completion of one motion frame. */
  sample(elapsed: number) {
    if (!Number.isFinite(elapsed)) return;
    const miss = elapsed > 1.5 * this.refresh;
    this.recent.push(miss);
    if (this.recent.length > 8) this.recent.shift();
    this.hits = miss ? 0 : this.hits + 1;
    if (this.tier > 0 && this.recent.filter(Boolean).length >= 2) { this.failed.add(this.tier); this.tier--; this.reset(); return; }
    const next = this.edges[this.tier + 1];
    if (!next) return;
    const run = this.failed.has(this.tier + 1) ? 240 : 64;
    if (this.gpu.length >= 32 ? this.hits >= run && Math.max(...this.gpu) * (next / this.longEdge) ** 2 <= .6 * this.refresh :
      this.hits >= 2 * run) { this.tier++; this.reset(); }
  }
  private reset() { this.recent.length = 0; this.gpu.length = 0; this.hits = 0; }
}

export class SliceScrollAccumulator {
  private pending: [number, number, number] = [0, 0, 0];
  add(axis: number, step: number) { this.pending[axis] += step; }
  flush(cursor: readonly number[], low: readonly number[], high: readonly number[]): [number, number, number] {
    const next = cursor.map((value, axis) => Math.max(low[axis], Math.min(high[axis], value + this.pending[axis]))) as [number,number,number];
    this.pending = [0, 0, 0];
    return next;
  }
}

type Port=MessagePort&{ref?:()=>void;unref?:()=>void};
let taskChannel:MessageChannel|null=null;
const taskQueue:Array<()=>void>=[];
const reportTaskError=(error:unknown)=>{
  const report=(globalThis as {reportError?:(error:unknown)=>void}).reportError;
  if(report)report(error);else console.error(error);
};
/** Run a callback in a new task as soon as possible: unlike nested timers it
 * is not clamped, and unlike animation frames it does not wait for a refresh.
 * A callback that throws is reported without dropping those queued with it. */
export function nextTask(callback:()=>void){
  if(!taskChannel){
    const channel=taskChannel=new MessageChannel();
    channel.port1.onmessage=()=>{
      const run=taskQueue.splice(0);
      try{for(const task of run){try{task();}catch(error){reportTaskError(error);}}}
      finally{if(!taskQueue.length)(channel.port1 as Port).unref?.();}
    };
  }
  taskQueue.push(callback);
  if(taskQueue.length===1){(taskChannel.port1 as Port).ref?.();taskChannel.port2.postMessage(0);}
}

/** Run a callback after about a millisecond, to poll a GPU fence without
 * holding a core. The timer is set from a message task, outside any timer
 * nesting, so it is not clamped to four milliseconds however long the poll runs. */
export function pollSoon(callback:()=>void){
  nextTask(()=>{setTimeout(()=>callback(),1);});
}

/** Decides when a pending fence is polled back to back instead of on the
 * millisecond timer. A timer poll lands a few milliseconds late, which a
 * progressive render pays on every sample; polling without a pause for the
 * whole GPU frame would hold a core. So the core is held only for a short
 * window around the completion time the previous samples predict. */
export class FenceWindow {
  private expected = 0;
  private readonly lead: number;
  private readonly slack: number;
  constructor(lead = 3, slack = 6) { this.lead = lead; this.slack = slack; }
  /** Whether a fence submitted `elapsed` ms ago is polled without a pause. */
  eager(elapsed: number) { return this.expected > 0 && elapsed >= this.expected - this.lead && elapsed <= this.expected + this.slack; }
  /** Record the submission-to-completion time of a finished sample. */
  completed(elapsed: number) { if (Number.isFinite(elapsed) && elapsed > 0) this.expected = this.expected ? this.expected * .7 + elapsed * .3 : elapsed; }
  reset() { this.expected = 0; }
}

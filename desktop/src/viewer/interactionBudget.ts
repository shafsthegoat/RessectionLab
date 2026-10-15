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

export class SliceScrollAccumulator {
  private pending: [number, number, number] = [0, 0, 0];
  add(axis: number, step: number) { this.pending[axis] += step; }
  flush(cursor: readonly number[], low: readonly number[], high: readonly number[]): [number, number, number] {
    const next = cursor.map((value, axis) => Math.max(low[axis], Math.min(high[axis], value + this.pending[axis]))) as [number,number,number];
    this.pending = [0, 0, 0];
    return next;
  }
}

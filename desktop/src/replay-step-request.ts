/** A bounded replay queue: one active request and one replaceable latest value. */
export class LatestReplayRequests<T> {
  private latest: T | undefined;
  private timer: ReturnType<typeof setTimeout> | undefined;
  private running = false;
  private generation = 0;
  private readonly execute: (
    value: T,
    isCurrent: () => boolean,
  ) => Promise<void>;
  private readonly pending: (value: boolean) => void;
  private readonly delayMs: number;
  constructor(
    execute: (value: T, isCurrent: () => boolean) => Promise<void>,
    pending: (value: boolean) => void,
    delayMs = 150,
  ) {
    this.execute = execute;
    this.pending = pending;
    this.delayMs = delayMs;
  }
  request(value: T): void {
    this.generation++;
    this.latest = value;
    this.pending(true);
    if (this.timer) clearTimeout(this.timer);
    this.timer = setTimeout(() => {
      this.timer = undefined;
      void this.drain();
    }, this.delayMs);
  }
  invalidate(): void {
    this.generation++;
    this.latest = undefined;
    if (this.timer) clearTimeout(this.timer);
    this.timer = undefined;
    this.pending(false);
  }
  private async drain(): Promise<void> {
    if (this.running || this.latest === undefined) return;
    const value = this.latest,
      generation = this.generation;
    this.latest = undefined;
    this.running = true;
    try {
      await this.execute(value, () => generation === this.generation);
    } finally {
      this.running = false;
      if (this.latest !== undefined) {
        // A timer still pending means the newest interaction has not settled yet.
        if (!this.timer) void this.drain();
      } else if (!this.timer) this.pending(false);
    }
  }
}

/**
 * Read-only value HUD for controls that change the 3D image.
 *
 * Controls emit ordinary `input` events (range sliders) or the explicit
 * `rl-viewport-hud` event below; the 3D pane shows the control name and value
 * while the user drags and hides it shortly after release. Controls stay where
 * they are; nothing here changes a setting.
 */
export const VIEWPORT_HUD_EVENT = 'rl-viewport-hud';
export const VIEWPORT_HUD_HIDE_MS = 600;

export interface ViewportHudReading {
  label: string;
  value: string;
}

export interface ViewportHudDetail extends Partial<ViewportHudReading> {
  /** `change` shows or updates the HUD; `end` starts the hide delay. */
  phase: 'change' | 'end';
}

/** For a control that is not a range input: report a value change or its end. */
export function emitViewportHud(detail: ViewportHudDetail, target: EventTarget = document): void {
  target.dispatchEvent(new CustomEvent<ViewportHudDetail>(VIEWPORT_HUD_EVENT, {detail}));
}

/** The parts of a range input the HUD reads. */
export interface ViewportHudControl {
  ariaLabel?: string | null;
  /** `data-viewport-hud`: explicit HUD name; opts any range into the HUD. */
  hudLabel?: string | null;
  /** `data-viewport-hud-unit`, appended to a numeric value. */
  hudUnit?: string | null;
  /** `aria-valuetext`, preferred when present. */
  valueText?: string | null;
  value: number;
  min: number;
  max: number;
}

/**
 * Known 3D-image controls by accessible name, so inspector rows owned
 * elsewhere need no markup change. Slice scrubbers and timelines never match.
 */
const HUD_NAME_RULES: ReadonlyArray<readonly [RegExp, (match: RegExpMatchArray) => string]> = [
  [/^Registered blend opacity$/i, () => 'Blend'],
  [/^Scan volume density$/i, () => 'Scan density'],
  [/^Cutaway position$/i, () => 'Cutaway'],
  [/window width$/i, () => 'Window width'],
  [/^Transfer opacity (\d+)$/i, match => `Transfer opacity ${match[1]}`],
  [/^(.+) opacity$/i, match => `${match[1]} opacity`],
];

function formatNumber(value: number): string {
  if (!Number.isFinite(value)) return '—';
  const rounded = Math.abs(value) >= 100 ? value.toFixed(0) : Math.abs(value) >= 10 ? value.toFixed(1) : value.toFixed(2);
  return String(Number(rounded));
}

/** The HUD line for a control, or null when the control does not change the 3D image. */
export function viewportHudReading(control: ViewportHudControl): ViewportHudReading | null {
  let label = control.hudLabel?.trim() || null;
  if (!label) {
    const name = control.ariaLabel?.trim() ?? '';
    for (const [pattern, describe] of HUD_NAME_RULES) {
      const match = name.match(pattern);
      if (match) {
        label = describe(match);
        break;
      }
    }
  }
  if (!label) return null;
  const valueText = control.valueText?.trim();
  let value: string;
  if (valueText) value = valueText;
  else if (control.hudUnit) value = `${formatNumber(control.value)} ${control.hudUnit}`;
  else if (control.min >= 0 && control.max <= 1 && control.max > control.min) value = `${Math.round(control.value * 100)}%`;
  else value = formatNumber(control.value);
  return {label, value};
}

export interface HudTimers<T = unknown> {
  set(callback: () => void, ms: number): T;
  clear(handle: T): void;
}

const browserTimers: HudTimers<ReturnType<typeof setTimeout>> = {
  set: (callback, ms) => setTimeout(callback, ms),
  clear: handle => clearTimeout(handle),
};

/**
 * Show while a HUD control changes or stays pressed; hide `delayMs` after the
 * pointer or key is released. Keyboard steps without a held pointer each
 * restart the delay.
 */
export class ViewportHudController<T = unknown> {
  private pressed = false;
  private shown = false;
  private timer: T | null = null;
  private readonly view: {show(reading: ViewportHudReading): void; hide(): void};
  private readonly delayMs: number;
  private readonly timers: HudTimers<T>;

  constructor(
    view: {show(reading: ViewportHudReading): void; hide(): void},
    delayMs = VIEWPORT_HUD_HIDE_MS,
    timers: HudTimers<T> = browserTimers as unknown as HudTimers<T>,
  ) {
    this.view = view;
    this.delayMs = delayMs;
    this.timers = timers;
  }

  /** A pointer went down on a HUD control. */
  press(): void {
    this.pressed = true;
    this.cancel();
  }

  change(reading: ViewportHudReading): void {
    this.cancel();
    this.shown = true;
    this.view.show(reading);
    if (!this.pressed) this.schedule();
  }

  release(): void {
    this.pressed = false;
    if (this.shown && this.timer === null) this.schedule();
  }

  get visible(): boolean {
    return this.shown;
  }

  dispose(): void {
    this.cancel();
  }

  private schedule(): void {
    this.cancel();
    this.timer = this.timers.set(() => {
      this.timer = null;
      this.shown = false;
      this.view.hide();
    }, this.delayMs);
  }

  private cancel(): void {
    if (this.timer !== null) {
      this.timers.clear(this.timer);
      this.timer = null;
    }
  }
}

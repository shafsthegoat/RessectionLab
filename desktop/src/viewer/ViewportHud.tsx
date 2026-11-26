import {useEffect, useRef, useState} from 'react';
import type {RefObject} from 'react';
import {VIEWPORT_HUD_EVENT, ViewportHudController, viewportHudReading} from './viewportHud';
import type {ViewportHudDetail, ViewportHudReading} from './viewportHud';
import {cameraTargetDistance, formatScaleLength, niceScaleBar, perspectiveMmPerPixel} from './scaleBar';
import './viewport-hud.css';

function rangeReading(target: EventTarget | null, viewer: Element | null): ViewportHudReading | null {
  if (!(target instanceof HTMLInputElement) || target.type !== 'range' || target.disabled) return null;
  // A slider inside another workspace belongs to that workspace's HUD.
  const owner = target.closest('.rl-viewer');
  if (owner && owner !== viewer) return null;
  return viewportHudReading({
    ariaLabel: target.getAttribute('aria-label'),
    hudLabel: target.dataset.viewportHud,
    hudUnit: target.dataset.viewportHudUnit,
    valueText: target.getAttribute('aria-valuetext'),
    value: target.valueAsNumber,
    min: target.min === '' ? 0 : Number(target.min),
    max: target.max === '' ? 100 : Number(target.max),
  });
}

/**
 * Read-only name and value of a 3D-image control while it is dragged,
 * top-centre of the 3D pane. The slider itself stays where it is and remains
 * the accessible source of the value, so the HUD is hidden from assistive tech.
 */
export function ViewportSliderHud({scope}: {scope: RefObject<HTMLElement | null>}) {
  const [hud, setHud] = useState<{reading: ViewportHudReading; visible: boolean} | null>(null);
  useEffect(() => {
    const controller = new ViewportHudController({
      show: reading => setHud(current => current?.visible && current.reading.label === reading.label &&
        current.reading.value === reading.value ? current : {reading, visible: true}),
      hide: () => setHud(current => current && {...current, visible: false}),
    });
    const viewer = () => scope.current?.closest('.rl-viewer') ?? null;
    let pending: HTMLInputElement | null = null;
    let frame = 0;
    // Read after React commits the change, so aria-valuetext matches the value.
    const onInput = (event: Event) => {
      if (!(event.target instanceof HTMLInputElement) || event.target.type !== 'range') return;
      pending = event.target;
      if (!frame) frame = requestAnimationFrame(() => {
        frame = 0;
        const reading = rangeReading(pending, viewer());
        pending = null;
        if (reading) controller.change(reading);
      });
    };
    const onPointerDown = (event: PointerEvent) => {
      if (rangeReading(event.target, viewer())) controller.press();
    };
    const release = () => controller.release();
    const onKeyUp = (event: KeyboardEvent) => {
      if (event.target instanceof HTMLInputElement && event.target.type === 'range') release();
    };
    const onDetail = (event: Event) => {
      const detail = (event as CustomEvent<ViewportHudDetail>).detail;
      if (!detail) return;
      if (detail.phase === 'end') release();
      else if (detail.label && detail.value != null) controller.change({label: detail.label, value: detail.value});
    };
    document.addEventListener('input', onInput, true);
    document.addEventListener('pointerdown', onPointerDown, true);
    document.addEventListener('pointerup', release, true);
    document.addEventListener('pointercancel', release, true);
    document.addEventListener('keyup', onKeyUp, true);
    document.addEventListener(VIEWPORT_HUD_EVENT, onDetail);
    window.addEventListener('blur', release);
    return () => {
      cancelAnimationFrame(frame);
      controller.dispose();
      document.removeEventListener('input', onInput, true);
      document.removeEventListener('pointerdown', onPointerDown, true);
      document.removeEventListener('pointerup', release, true);
      document.removeEventListener('pointercancel', release, true);
      document.removeEventListener('keyup', onKeyUp, true);
      document.removeEventListener(VIEWPORT_HUD_EVENT, onDetail);
      window.removeEventListener('blur', release);
    };
  }, [scope]);
  if (!hud) return null;
  return (
    <div className="rl-viewport-hud" data-visible={hud.visible ? 'true' : 'false'} aria-hidden="true">
      <span className="rl-viewport-hud-label">{hud.reading.label}</span>
      <span className="rl-viewport-hud-value">{hud.reading.value}</span>
    </div>
  );
}

/**
 * Physical scale for the perspective 3D pane, measured at the orbit target
 * from each rendered frame's camera (RAS millimetres).
 */
export function ViewportScaleBar({canvas, pane, fieldOfView}: {
  canvas: RefObject<HTMLCanvasElement | null>;
  pane: RefObject<HTMLElement | null>;
  /** The 3D camera's current vertical field of view in degrees. */
  fieldOfView: () => number | null | undefined;
}) {
  const [bar, setBar] = useState<{mm: number; px: number} | null>(null);
  const latestFov = useRef(fieldOfView);
  latestFov.current = fieldOfView;
  useEffect(() => {
    const element = canvas.current;
    if (!element) return;
    const receive = (event: Event) => {
      const camera = (event as CustomEvent<{cameraState?: {position: number[]; target: number[]}}>).detail?.cameraState;
      const fov = latestFov.current();
      const height = pane.current?.clientHeight ?? 0;
      const next = camera && fov
        ? niceScaleBar(perspectiveMmPerPixel(cameraTargetDistance(camera.position, camera.target), fov, height))
        : null;
      setBar(current => current?.mm === next?.mm && Math.round(current?.px ?? 0) === Math.round(next?.px ?? 0) ? current : next);
    };
    element.addEventListener('anatomy-frame', receive);
    return () => element.removeEventListener('anatomy-frame', receive);
  }, [canvas, pane]);
  if (!bar) return null;
  const label = formatScaleLength(bar.mm);
  return (
    <div className="rl-viewport-scale" role="img" aria-label={`Scale bar, ${label} at the orbit centre`}
      title={`${label} at the orbit centre. Perspective: nearer anatomy appears larger.`}>
      <span>{label}</span>
      <i style={{width: Math.max(8, Math.round(bar.px))}} />
    </div>
  );
}

import { useEffect, useMemo, useRef, useState } from "react";
import type { KeyboardEvent, PointerEvent } from "react";
import { VolumeRenderer } from "./VolumeRenderer";
import {
  PLANE_AXES,
  PLANE_LABELS,
  clampCursor,
  inverseAffine,
  rasAffine,
  slicePoint,
  sliceRect,
  volumeBounds,
} from "./coordinates";
import type { Point3, SlicePlane } from "./coordinates";
import type { ViewerWorkspaceProps } from "./contracts";
import "./viewer.css";

const PLANES: SlicePlane[] = ["axial", "coronal", "sagittal"];
const TITLES: Record<SlicePlane, string> = {
  axial: "Axial",
  coronal: "Coronal",
  sagittal: "Sagittal",
};

export function ViewerWorkspace(props: ViewerWorkspaceProps) {
  const {
    caseData,
    cursor,
    onCursorChange,
    visibleLayers,
    overlayOpacity,
    routes,
    cameraMode,
    replay,
  } = props;
  const container = useRef<HTMLDivElement>(null),
    canvas = useRef<HTMLCanvasElement>(null),
    anatomy = useRef<HTMLDivElement>(null);
  const axial = useRef<HTMLDivElement>(null),
    coronal = useRef<HTMLDivElement>(null),
    sagittal = useRef<HTMLDivElement>(null);
  const engine = useRef<VolumeRenderer | null>(null),
    latestProps = useRef(props);
  latestProps.current = props;
  const [error, setError] = useState<string | null>(null),
    [remaining, setRemaining] = useState(0),
    [ready, setReady] = useState(false);
  const [activePlane, setActivePlane] = useState<SlicePlane>("axial"),
    [contrast, setContrast] = useState(1),
    [showPlane, setShowPlane] = useState(false),
    [replayError, setReplayError] = useState<string | null>(null);
  const [paneSizes, setPaneSizes] = useState<
    Record<SlicePlane, [number, number]>
  >({ axial: [1, 1], coronal: [1, 1], sagittal: [1, 1] });
  const geometry = useMemo(() => {
    if (!caseData) return null;
    try {
      const affine = rasAffine(caseData.affine, caseData.frame);
      return {
        affine,
        inverse: inverseAffine(affine),
        bounds: volumeBounds(caseData.shape, affine),
      };
    } catch {
      return null;
    }
  }, [caseData]);
  const currentCursor =
    cursor ??
    (geometry
      ? (geometry.bounds[0].map(
          (v, i) => (v + geometry.bounds[1][i]) / 2,
        ) as Point3)
      : [0, 0, 0]);

  useEffect(() => {
    if (
      !caseData ||
      !container.current ||
      !canvas.current ||
      !anatomy.current ||
      !axial.current ||
      !coronal.current ||
      !sagittal.current
    )
      return;
    setError(null);
    setReady(false);
    setContrast(1);
    try {
      const renderer = new VolumeRenderer(
        container.current,
        canvas.current,
        {
          anatomy: anatomy.current,
          axial: axial.current,
          coronal: coronal.current,
          sagittal: sagittal.current,
        },
        caseData,
        setError,
        setRemaining,
        setReplayError,
      );
      engine.current = renderer;
      setReady(true);
      return () => {
        renderer.dispose();
        engine.current = null;
      };
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : String(cause));
    }
  }, [caseData]);

  useEffect(() => {
    if (!engine.current) return;
    engine.current.update(
      currentCursor,
      visibleLayers,
      overlayOpacity,
      routes,
      cameraMode,
    );
  }, [
    ready,
    currentCursor[0],
    currentCursor[1],
    currentCursor[2],
    visibleLayers,
    overlayOpacity,
    routes,
    cameraMode,
  ]);

  useEffect(() => {
    engine.current?.setSourcePlane(showPlane ? activePlane : null);
  }, [activePlane, showPlane, ready, caseData]);

  useEffect(() => {
    if (!engine.current) return;
    setReplayError(null);
    try {
      engine.current.setReplay(replay ?? null);
    } catch (cause) {
      setReplayError(cause instanceof Error ? cause.message : String(cause));
    }
    // The app may recreate its props object on cursor movement; the accepted
    // mask and certificate inputs, rather than that wrapper, define a new step.
  }, [
    ready,
    caseData,
    replay?.removedMask,
    replay?.step,
    replay?.stepCount,
    replay?.scope,
    replay?.caseHash,
    replay?.shape,
    replay?.affine,
    replay?.independentlyAccepted,
    replay?.removedTargetVolumeMm3,
    replay?.removedNormalVolumeMm3,
    replay?.residualTargetVolumeMm3,
  ]);

  useEffect(() => {
    const renderer = engine.current;
    if (!renderer) return;
    const [low, high] = renderer.defaultWindow,
      centre = (low + high) / 2,
      width = (high - low) * contrast;
    renderer.setWindow(centre - width / 2, centre + width / 2);
  }, [contrast, ready]);

  useEffect(() => {
    const refs = { axial, coronal, sagittal };
    const observer = new ResizeObserver(() => {
      const sizes = {} as Record<SlicePlane, [number, number]>;
      PLANES.forEach((plane) => {
        const rect = refs[plane].current?.getBoundingClientRect();
        sizes[plane] = [rect?.width ?? 1, rect?.height ?? 1];
      });
      setPaneSizes(sizes);
    });
    const listeners: Array<() => void> = [];
    for (const plane of PLANES) {
      const node = refs[plane].current;
      if (!node) continue;
      observer.observe(node);
      const listener = (event: WheelEvent) => {
        if (event.deltaY === 0) return;
        event.preventDefault();
        const p = latestProps.current,
          volume = p.caseData;
        if (!volume) return;
        const aff = rasAffine(volume.affine, volume.frame),
          inverse = inverseAffine(aff),
          bounds = volumeBounds(volume.shape, aff);
        const next = p.cursor
          ? ([...p.cursor] as Point3)
          : (bounds[0].map((v, i) => (v + bounds[1][i]) / 2) as Point3);
        const axis = PLANE_AXES[plane][2],
          step =
            1 /
            Math.hypot(inverse[0][axis], inverse[1][axis], inverse[2][axis]);
        next[axis] +=
          (event.deltaY > 0 ? -1 : 1) * step * (event.shiftKey ? 5 : 1);
        p.onCursorChange(clampCursor(next, bounds));
      };
      node.addEventListener("wheel", listener, { passive: false });
      listeners.push(() => node.removeEventListener("wheel", listener));
    }
    return () => {
      observer.disconnect();
      listeners.forEach((remove) => remove());
    };
  }, [caseData]);

  function moveCursor(
    event: PointerEvent<HTMLDivElement>,
    plane: SlicePlane,
  ): void {
    if (!geometry || (event.button !== 0 && event.buttons !== 1)) return;
    const rect = event.currentTarget.getBoundingClientRect(),
      image = sliceRect(geometry.bounds, plane, rect.width, rect.height);
    const point = slicePoint(
      geometry.bounds,
      plane,
      currentCursor,
      image,
      event.clientX - rect.left,
      event.clientY - rect.top,
    );
    if (point) {
      event.preventDefault();
      onCursorChange(point);
    }
  }

  function keySlice(
    event: KeyboardEvent<HTMLDivElement>,
    plane: SlicePlane,
  ): void {
    if (!geometry || !["ArrowUp", "ArrowDown"].includes(event.key)) return;
    event.preventDefault();
    const axis = PLANE_AXES[plane][2],
      next = [...currentCursor] as Point3;
    const inverse = geometry.inverse,
      step =
        1 / Math.hypot(inverse[0][axis], inverse[1][axis], inverse[2][axis]);
    next[axis] += (event.key === "ArrowUp" ? 1 : -1) * step;
    onCursorChange(clampCursor(next, geometry.bounds));
  }

  if (!caseData)
    return (
      <div
        className="rl-viewer-empty"
        aria-label="Imaging workspace awaiting a case"
      >
        <div className="rl-viewer-empty-mark" aria-hidden="true">
          <span />
          <span />
          <span />
        </div>
        <h2>Your case, in perspective.</h2>
        <p>
          Open a prepared case to inspect the source MRI, its annotations and
          instrument-aware routes.
        </p>
        <span className="rl-viewer-empty-meta">
          LOCAL IMAGING · LINKED 3D + MRI · PHYSICAL COORDINATES
        </span>
      </div>
    );

  const refs = { axial, coronal, sagittal };
  const modeled = replay && !replayError;
  return (
    <div
      className="rl-viewer"
      ref={container}
      aria-label="Linked patient imaging workspace"
    >
      <canvas ref={canvas} className="rl-viewer-canvas" aria-hidden="true" />
      <div
        className="rl-viewer-anatomy rl-viewer-pane"
        ref={anatomy}
        onDoubleClick={(event) => {
          const point = engine.current?.pick(event.clientX, event.clientY);
          if (point && geometry)
            onCursorChange(clampCursor(point, geometry.bounds));
        }}
        aria-label="3-D source anatomy. Drag to rotate, scroll to zoom, double-click to set linked cursor."
      >
        <div className="rl-viewer-heading">
          <span className="rl-viewer-tag">3D</span>
          <span>Spatial workspace</span>
          <span className="rl-viewer-frame">RAS · mm</span>
        </div>
        <div
          className="rl-viewer-toolbar"
          onPointerDown={(event) => event.stopPropagation()}
          onDoubleClick={(event) => event.stopPropagation()}
        >
          <button
            className={`rl-viewer-reset rl-viewer-plane-toggle ${showPlane ? "is-active" : ""}`}
            aria-pressed={showPlane}
            onClick={() => setShowPlane(!showPlane)}
            title="Show or hide the source MRI plane in 3D; linked MRI views remain visible"
          >
            MRI plane {showPlane ? "on" : "off"}
          </button>
          <div className="rl-viewer-segment" aria-label="Source MRI plane">
            {PLANES.map((plane) => (
              <button
                key={plane}
                className={
                  showPlane && activePlane === plane ? "is-active" : ""
                }
                onClick={() => {
                  setActivePlane(plane);
                  setShowPlane(true);
                }}
                title={`Show source ${plane} MRI plane`}
              >
                {TITLES[plane]}
              </button>
            ))}
          </div>
          <button
            className="rl-viewer-reset"
            onClick={() => engine.current?.fitCamera(cameraMode)}
            title="Fit visible anatomy and selected camera preset"
          >
            ↺ <span>Fit view</span>
          </button>
        </div>
        <div className="rl-viewer-annotations">
          <span className="rl-viewer-dot" />{" "}
          {modeled
            ? `Modeled residual annotations · step ${replay.step}/${replay.stepCount}`
            : "Source annotation surfaces"}
          {modeled && (
            <span className="rl-viewer-replay-key">
              Mint mesh: modeled removal · source MRI unchanged
            </span>
          )}
          {remaining > 0 && (
            <span className="rl-viewer-preparing" role="status">
              Preparing {remaining} {remaining === 1 ? "surface" : "surfaces"}…
            </span>
          )}
        </div>
        <div
          className="rl-viewer-compass"
          aria-label="3-D patient coordinates: R is right, A anterior, S superior"
        >
          <span>R</span>
          <span>A</span>
          <span>S</span>
          <small>Patient axes</small>
        </div>
        <div className="rl-viewer-scene-footer">
          <span>
            Drag to rotate <i>·</i> Scroll to zoom
          </span>
          <span>Double-click to link cursor</span>
        </div>
      </div>
      {PLANES.map((plane) => {
        const [left, right, top, bottom] = PLANE_LABELS[plane],
          axis = PLANE_AXES[plane][2];
        const [width, height] = paneSizes[plane];
        const image = geometry
          ? sliceRect(geometry.bounds, plane, width, height)
          : null;
        const mmWidth = geometry
          ? geometry.bounds[1][PLANE_AXES[plane][0]] -
            geometry.bounds[0][PLANE_AXES[plane][0]]
          : 1;
        const pxPerMm = image ? image.width / mmWidth : 1;
        const scaleMm = pxPerMm * 20 < 78 ? 20 : 10;
        return (
          <div
            key={plane}
            ref={refs[plane]}
            className={`rl-viewer-pane rl-viewer-slice rl-viewer-${plane}`}
            tabIndex={0}
            aria-label={`${TITLES[plane]} source MRI, neurological convention, slice position ${currentCursor[axis].toFixed(1)} millimeters. Click to set cursor; scroll or arrow keys change slice.`}
            onPointerDown={(event) => {
              if (event.button === 0) {
                event.currentTarget.focus({ preventScroll: true });
                event.currentTarget.setPointerCapture(event.pointerId);
                moveCursor(event, plane);
              }
            }}
            onPointerMove={(event) => {
              if (event.buttons === 1) moveCursor(event, plane);
            }}
            onKeyDown={(event) => keySlice(event, plane)}
          >
            <div className="rl-viewer-heading">
              <span className={`rl-viewer-plane-dot ${plane}`} />
              <span>{TITLES[plane]}</span>
              <span className="rl-viewer-slice-position">
                {currentCursor[axis].toFixed(1)} <small>mm</small>
              </span>
            </div>
            <span className="rl-orientation rl-orientation-left">{left}</span>
            <span className="rl-orientation rl-orientation-right">{right}</span>
            <span className="rl-orientation rl-orientation-top">{top}</span>
            <span className="rl-orientation rl-orientation-bottom">
              {bottom}
            </span>
            <span
              className="rl-viewer-scale"
              style={{ width: Math.max(8, scaleMm * pxPerMm) }}
            >
              <i />
              {scaleMm} mm
            </span>
            <span className="rl-viewer-slice-note">SOURCE MRI</span>
          </div>
        );
      })}
      <div
        className="rl-viewer-statusbar"
        onPointerDown={(event) => event.stopPropagation()}
      >
        <span>
          <i className="rl-viewer-linked" /> Linked views <b>RAS</b>{" "}
          {currentCursor.map((v) => v.toFixed(1)).join(" / ")} mm
        </span>
        <label>
          Contrast{" "}
          <input
            type="range"
            min="0.4"
            max="2"
            step="0.05"
            value={contrast}
            onChange={(event) => setContrast(Number(event.target.value))}
            aria-label="Source MRI window width"
          />
        </label>
        <span>Neurological convention</span>
      </div>
      {(error || replayError) && (
        <div className="rl-viewer-error" role="alert">
          <strong>Imaging view needs attention</strong>
          <p>{error || replayError}</p>
        </div>
      )}
    </div>
  );
}

export default ViewerWorkspace;

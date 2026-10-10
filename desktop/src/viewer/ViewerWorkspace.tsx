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
  transformPoint,
  volumeBounds,
} from "./coordinates";
import type { Point3, SlicePlane } from "./coordinates";
import type { ViewerPriorLayer, ViewerWorkspaceProps } from "./contracts";
import { INSPECTION_TOOL_COLOR, InspectionSelectionGate } from "./inspectionTool";
import type { InspectionToolDisplay } from "./inspectionTool";
import { formatPriorValue, samplePriorVoxel } from "./priorLayer";
import { FAILURE_COLOR, routeAppearance } from "./routeAppearance";
import {
  STRUCTURAL_PROPOSAL_COLOR,
  structuralProposalReviewLabel,
} from "./structuralProposal";
import "./viewer.css";

const PLANES: SlicePlane[] = ["axial", "coronal", "sagittal"];
const TITLES: Record<SlicePlane, string> = {
  axial: "Axial",
  coronal: "Coronal",
  sagittal: "Sagittal",
};
const PRIOR_UNAVAILABLE_LABELS: Record<string, string> = {
  "outside-atlas-coverage": "Outside sampled atlas · no value",
  "incomplete-interpolation-support":
    "Interpolation support unavailable · no value",
  "numerical-boundary-uncertainty":
    "Atlas field boundary · display precision unknown",
  "numerical-precision-unavailable":
    "Atlas coordinates exceed display precision · no value",
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
    structuralProposal,
    priorLayer,
    inspectionTool,
  } = props;
  const signalName = props.generatedSignal ? "analytic signal" : "MRI";
  const container = useRef<HTMLDivElement>(null),
    canvas = useRef<HTMLCanvasElement>(null),
    anatomy = useRef<HTMLDivElement>(null);
  const axial = useRef<HTMLDivElement>(null),
    coronal = useRef<HTMLDivElement>(null),
    sagittal = useRef<HTMLDivElement>(null);
  const engine = useRef<VolumeRenderer | null>(null),
    latestProps = useRef(props);
  latestProps.current = props;
  const inspectionSelection = useRef(new InspectionSelectionGate());
  const [displayedInspection, setDisplayedInspection] = useState<InspectionToolDisplay | null>(null);
  const [inspectionError, setInspectionError] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null),
    [remaining, setRemaining] = useState(0),
    [ready, setReady] = useState(false);
  const [activePlane, setActivePlane] = useState<SlicePlane>("axial"),
    [contrast, setContrast] = useState(1),
    [showPlane, setShowPlane] = useState(false),
    [replayError, setReplayError] = useState<string | null>(null),
    [proposalError, setProposalError] = useState<string | null>(null),
    [proposalReviewLabel, setProposalReviewLabel] = useState<string | null>(
      null,
    );
  const [layout, setLayout] = useState<"3d-focus" | "mri-review">("3d-focus"),
    [expanded, setExpanded] = useState<SlicePlane | null>(null);
  const [displayedPrior, setDisplayedPrior] = useState<ViewerPriorLayer | null>(
      null,
    ),
    [priorError, setPriorError] = useState<string | null>(null);
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
  const priorSample = useMemo(() => {
    if (!displayedPrior || displayedPrior.caseHash !== caseData?.caseHash)
      return null;
    return samplePriorVoxel(
      displayedPrior,
      transformPoint(inverseAffine(displayedPrior.affine), currentCursor),
    );
  }, [
    displayedPrior,
    caseData?.caseHash,
    currentCursor[0],
    currentCursor[1],
    currentCursor[2],
  ]);

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
    replay?.recordedTool,
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
    setInspectionError(null);
    setDisplayedInspection(null);
    const suppressed = Boolean(replay || structuralProposal || priorLayer);
    if (inspectionTool && suppressed) {
      if (inspectionSelection.current.clear(inspectionTool)) latestProps.current.onClearInspection?.();
    }
    try {
      setDisplayedInspection(renderer.setInspectionTool(
        inspectionSelection.current.select(inspectionTool ?? null, suppressed),
      ));
    } catch (cause) {
      inspectionSelection.current.clear(inspectionTool ?? null);
      setInspectionError(cause instanceof Error ? cause.message : String(cause));
      latestProps.current.onClearInspection?.();
    }
  }, [ready, caseData, inspectionTool, Boolean(replay), Boolean(structuralProposal), Boolean(priorLayer)]);

  useEffect(() => {
    const renderer = engine.current;
    if (!renderer) return;
    const [low, high] = renderer.defaultWindow,
      centre = (low + high) / 2,
      width = (high - low) * contrast;
    renderer.setWindow(centre - width / 2, centre + width / 2);
  }, [contrast, ready]);

  useEffect(() => {
    if (!engine.current) return;
    setProposalError(null);
    setProposalReviewLabel(null);
    try {
      const proposal = replay ? null : (structuralProposal ?? null);
      engine.current.setStructuralProposal(proposal);
      if (proposal)
        setProposalReviewLabel(
          structuralProposalReviewLabel(proposal.reviewStatus),
        );
    } catch (cause) {
      setProposalError(cause instanceof Error ? cause.message : String(cause));
    }
  }, [
    ready,
    caseData,
    Boolean(replay),
    structuralProposal?.caseHash,
    structuralProposal?.evidenceId,
    structuralProposal?.mask,
    structuralProposal?.shape,
    structuralProposal?.affine,
    structuralProposal?.frame,
    structuralProposal?.scope,
    structuralProposal?.provenance,
    structuralProposal?.reviewStatus,
  ]);

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

  useEffect(() => {
    if (!engine.current) return;
    setPriorError(null);
    setDisplayedPrior(null);
    try {
      setDisplayedPrior(
        engine.current.setPriorLayer(
          replay || structuralProposal ? null : (priorLayer ?? null),
        ),
      );
    } catch (cause) {
      setPriorError(cause instanceof Error ? cause.message : String(cause));
    }
  }, [
    ready,
    caseData,
    Boolean(replay),
    Boolean(structuralProposal),
    priorLayer?.caseHash,
    priorLayer?.proposalId,
    priorLayer?.mapId,
    priorLayer?.title,
    priorLayer?.component,
    priorLayer?.values,
    priorLayer?.coverage,
    priorLayer?.affine,
    priorLayer?.shape,
    priorLayer?.frame,
    priorLayer?.mapKind,
    priorLayer?.scope,
    priorLayer?.provenance,
    priorLayer?.reviewStatus,
    priorLayer?.planningEligible,
    priorLayer?.patientSpecificFunction,
    priorLayer?.valueUnits,
    priorLayer?.spatialUnits,
  ]);

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
  const inspected = !replay && !structuralProposal && !priorLayer &&
    displayedInspection?.caseHash === caseData.caseHash ? displayedInspection : null;
  const clearInspection = () => {
    inspectionSelection.current.clear(inspectionTool ?? null);
    engine.current?.setInspectionTool(null);
    setDisplayedInspection(null);
    setInspectionError(null);
    latestProps.current.onClearInspection?.();
  };
  return (
    <div
      className={`rl-viewer rl-layout-${layout}${expanded ? " has-expanded" : ""}`}
      ref={container}
      data-layout={layout}
      data-expanded={expanded ?? undefined}
      aria-label="Linked patient imaging workspace"
    >
      <canvas ref={canvas} className="rl-viewer-canvas" aria-hidden="true" />
      <div
        className="rl-viewer-layoutbar"
        role="toolbar"
        aria-label="Imaging layout and 3D controls"
      >
        <div
          className="rl-viewer-segment"
          role="group"
          aria-label="View layout"
        >
          <button
            aria-pressed={layout === "3d-focus" && !expanded}
            className={layout === "3d-focus" && !expanded ? "is-active" : ""}
            onClick={() => {
              setLayout("3d-focus");
              setExpanded(null);
            }}
          >
            3D focus
          </button>
          <button
            aria-pressed={layout === "mri-review" && !expanded}
            className={layout === "mri-review" && !expanded ? "is-active" : ""}
            onClick={() => {
              setLayout("mri-review");
              setExpanded(null);
            }}
          >
            {props.generatedSignal ? "Signal review" : "MRI review"}
          </button>
        </div>
        <div className="rl-viewer-toolbar" hidden={Boolean(expanded)}>
          <button
            className={`rl-viewer-reset rl-viewer-plane-toggle ${showPlane ? "is-active" : ""}`}
            aria-pressed={showPlane}
            onClick={() => setShowPlane(!showPlane)}
            title={`Show or hide the source ${signalName} plane in 3D; linked ${signalName} views remain visible`}
          >
            {props.generatedSignal ? "Signal" : "MRI"} plane {showPlane ? "on" : "off"}
          </button>
          <select
            className="rl-viewer-plane-select"
            aria-label={`${signalName} plane shown in 3D`}
            value={activePlane}
            onChange={(event) => {
              setActivePlane(event.target.value as SlicePlane);
              setShowPlane(true);
            }}
          >
            {PLANES.map((plane) => (
              <option key={plane} value={plane}>
                {TITLES[plane]}
              </option>
            ))}
          </select>
          <button
            className="rl-viewer-reset"
            onClick={() => engine.current?.fitCamera(cameraMode)}
            aria-label="Fit 3D view"
            title="Fit anatomy and the selected 3D camera preset"
          >
            ↺ <span>Fit 3D</span>
          </button>
        </div>
        {expanded && (
          <span className="rl-viewer-expanded-note">
            {TITLES[expanded]} {signalName} expanded
          </span>
        )}
        {displayedPrior && priorSample && (
          <div
            className="rl-viewer-prior-readout"
            role="status"
            aria-live="polite"
          >
            <span>
              <strong>Population prior</strong> · alignment review required
            </span>
            <span className="rl-viewer-prior-value">
              {priorSample.covered
                ? `${displayedPrior.mapKind === "functional_concordance" ? "Interpolated atlas sample" : "Nearest released-mask cell"}: ${priorSample.value === 0 ? "covered zero · " : ""}${formatPriorValue(priorSample.value!)} · unitless`
                : (PRIOR_UNAVAILABLE_LABELS[priorSample.reason ?? ""] ??
                  "Atlas sample unavailable · no value")}
            </span>
            <span className="rl-viewer-prior-note">
              Patient function unknown · hatching marks unavailable atlas
              support
            </span>
          </div>
        )}
      </div>
      <div className="rl-viewer-anatomy rl-viewer-pane">
        <div className="rl-viewer-heading">
          <span className="rl-viewer-tag">3D</span>
          <span>Spatial workspace</span>
          <span className="rl-viewer-frame">RAS · mm</span>
        </div>
        <div
          className="rl-viewer-anatomy-image"
          ref={anatomy}
          onDoubleClick={(event) => {
            const point = engine.current?.pick(event.clientX, event.clientY);
            if (point && geometry)
              onCursorChange(clampCursor(point, geometry.bounds));
          }}
          aria-label="3-D source anatomy. Drag to rotate, scroll to zoom, double-click to set linked cursor."
        >
          <div className="rl-viewer-annotations">
            <span className="rl-viewer-dot" />{" "}
            {modeled
              ? `Modeled residual annotations · step ${replay.step}/${replay.stepCount}`
              : "Source annotation surfaces"}
            {modeled && (
              <span className="rl-viewer-replay-key">
                Mint mesh: modeled removal · source {signalName} unchanged
              </span>
            )}
            {inspected && (
              <span className="rl-viewer-route-key" style={{ color: INSPECTION_TOOL_COLOR }}>
                ● Inspection tool · unexecuted preview · no tissue removed
                <button className="rl-viewer-inspection-clear" onClick={clearInspection}>Clear tool</button>
              </span>
            )}
            {inspected?.unknowns.includes("tool_geometry_outside_image_unassessed") && (
              <span className="rl-viewer-route-key">Tool extends beyond imaged anatomy · outside tissue unassessed</span>
            )}
            {!modeled && !inspected &&
              routes.slice(0, 2).map((route, index) => {
                const { slot, color } = routeAppearance(route, index);
                return (
                  <span
                    key={route.route_id}
                    className="rl-viewer-route-key"
                    style={{ color }}
                  >
                    ● Route {slot}
                    {route.category === "rejected" ? " · rejected" : ""}
                  </span>
                );
              })}
            {!modeled && !inspected &&
              routes.some((route) => route.geometry?.failures?.length) && (
                <span
                  className="rl-viewer-route-key"
                  style={{ color: FAILURE_COLOR }}
                >
                  ● Constraint failure
                </span>
              )}
            {remaining > 0 && (
              <span className="rl-viewer-preparing" role="status">
                Preparing {remaining} {remaining === 1 ? "surface" : "surfaces"}
                …
              </span>
            )}
            {proposalReviewLabel && (
              <span
                className="rl-viewer-proposal-key"
                style={{ color: STRUCTURAL_PROPOSAL_COLOR }}
              >
                <span aria-hidden="true">┄</span> Estimated envelope on {signalName} ·{" "}
                {proposalReviewLabel}
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
            className={`rl-viewer-pane rl-viewer-slice rl-viewer-${plane}${expanded === plane ? " is-expanded" : ""}`}
          >
            <div className="rl-viewer-heading">
              <span className={`rl-viewer-plane-dot ${plane}`} />
              <span>{TITLES[plane]}</span>
              <span className="rl-viewer-slice-position">
                {currentCursor[axis].toFixed(1)} <small>mm</small>
              </span>
              <button
                className="rl-viewer-expand"
                aria-label={
                  expanded === plane
                    ? "Restore linked views"
                    : `Expand ${TITLES[plane]} ${signalName}`
                }
                aria-pressed={expanded === plane}
                title={
                  expanded === plane
                    ? "Restore linked views"
                    : `Expand ${TITLES[plane]} ${signalName}`
                }
                onClick={() => setExpanded(expanded === plane ? null : plane)}
              >
                {expanded === plane ? "↙" : "⤢"}
              </button>
            </div>
            <div
              ref={refs[plane]}
              className="rl-viewer-slice-image"
              tabIndex={0}
              aria-label={`${TITLES[plane]} source ${signalName}, neurological convention, slice position ${currentCursor[axis].toFixed(1)} millimeters. ${inspected ? "Inspection tool, unexecuted initial preview. No tissue removed. " : ""}${proposalReviewLabel ? `Estimated envelope contour, view only: ${proposalReviewLabel}. ` : ""}${displayedPrior ? "Population prior, alignment review required. Patient function unknown. Hatching marks unavailable atlas support. " : ""}Click to set cursor; scroll or arrow keys change slice.`}
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
              <span className="rl-orientation rl-orientation-left">{left}</span>
              <span className="rl-orientation rl-orientation-right">
                {right}
              </span>
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
              <span className="rl-viewer-slice-note">
                {props.generatedSignal ? "GENERATED ANALYTIC SIGNAL" : "SOURCE MRI"}
                {inspected
                  ? " · UNEXECUTED TOOL PREVIEW"
                  : proposalReviewLabel
                  ? " · ESTIMATE"
                  : displayedPrior
                    ? " · PRIOR"
                    : ""}
              </span>
            </div>
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
            aria-label={`Source ${signalName} window width`}
          />
        </label>
        <span>Neurological convention</span>
      </div>
      {(error || replayError || proposalError || priorError || inspectionError) && (
        <div className="rl-viewer-error" role="alert">
          <strong>Imaging view needs attention</strong>
          <p>{error || replayError || proposalError || priorError || inspectionError}</p>
        </div>
      )}
    </div>
  );
}

export default ViewerWorkspace;

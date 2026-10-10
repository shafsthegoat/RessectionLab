import { ImagingWorkspacePanel } from "./ImagingWorkspacePanel";
import { useWorkspaceImaging } from "./use-workspace-imaging";
import { workspaceDisplay } from "./workspace-imaging-data";
import { hydrateWorkspace } from "./workspace-session";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import * as Dialog from "@radix-ui/react-dialog";
import * as Tabs from "@radix-ui/react-tabs";
import {
  ArrowDownToLine,
  ArrowUpFromLine,
  Check,
  ChevronDown,
  ChevronRight,
  CircleDot,
  Command,
  Expand,
  FlaskConical,
  Focus,
  FolderOpen,
  Info,
  Layers3,
  LoaderCircle,
  PanelLeftClose,
  Scan,
  ShieldCheck,
  SlidersHorizontal,
  Sparkles,
  Target,
  Waypoints,
  X,
} from "lucide-react";
import {
  annotationCenter,
  initialCursor,
  readableName,
  rasPoint,
} from "./case-data";
import { AnnotationCenterButton } from "./AnnotationCenterButton";
import { WorkspaceBreadcrumb } from "./WorkspaceBreadcrumb";
import { readOnlyPreview } from "./preview-api";
import { EpisodePanel } from "./EpisodePanel";
import { hydrateDevelopmentEpisode } from "./episode-data";
import type { EpisodeView } from "./episode-data";
import type { DevelopmentEpisodeRequest } from "./episode-types";
import { RefinementPanel } from "./RefinementPanel";
import { hydratePriorProposal } from "./prior-data";
import { PriorInventory, PriorProvenance } from "./PriorInventory";
import { hydrateStructuralProposal } from "./structural-proposal-data";
import { isOperationCancelled, operationMessage } from "./operation-feedback";
import { routeComparisonPrompt } from "./route-comparison-prompt";
import { StructuralEvidenceInventory } from "./StructuralEvidenceInventory";
import { StructuralImportDialog } from "./StructuralImportDialog";
import { researchSupportGate } from "./case-support";
import { selectComparisonRoutes } from "./route-selection";
import type { SelectedRoute } from "./route-selection";
import type {
  BridgeEvent,
  CasePayload,
  CertifiedReplay,
  ResectionApi,
  RouteCandidate,
  StructuralProposalView,
  PriorLayerView,
  SearchResult,
  Vec3,
  ViewerCase,
} from "./types";
import { ViewerWorkspace } from "./viewer/ViewerWorkspace";

interface Operation {
  id: string;
  op: string;
  message: string;
  fraction: number;
}
const categoryLabels = {
  pareto: "Retained alternatives",
  dominated: "Dominated alternatives",
  rejected: "Rejected candidates",
};
const mL = (value: number | null | undefined) =>
  value == null ? "Unassessed" : `${(value / 1000).toFixed(3)} mL`;
const millimeters = (value: number | null | undefined) =>
  value == null ? "Unassessed" : `${value.toFixed(1)} mm`;

function routeName(route: RouteCandidate, all: RouteCandidate[]): string {
  const ordinal =
    all.findIndex((candidate) => candidate.route_id === route.route_id) + 1;
  const tool = route.tool_id.replace(/^generic_/, "").replace(/_/g, " ");
  return `Route ${String(ordinal).padStart(2, "0")} · ${tool.charAt(0).toUpperCase() + tool.slice(1)}`;
}

function Logo() {
  return (
    <span className="brand-symbol" aria-hidden="true">
      <i />
      <i />
      <i />
    </span>
  );
}

function EvidenceDrawer({
  caseData,
  routes,
  canImport,
  onImport,
  priorView,
}: {
  priorView: PriorLayerView | null;
  caseData: CasePayload | null;
  routes: RouteCandidate[];
  canImport: boolean;
  onImport: (variant: "main" | "nocsf") => Promise<void>;
}) {
  const [advanced, setAdvanced] = useState(false);
  const collection = (caseData?.metadata.source_collection ?? {}) as Record<
    string,
    unknown
  >;
  return (
    <Dialog.Root>
      <Dialog.Trigger asChild>
        <button className="quiet-button inspector-trigger" disabled={!caseData}>
          <Info size={14} /> Inspect evidence <ChevronRight size={14} />
        </button>
      </Dialog.Trigger>
      <Dialog.Portal>
        <Dialog.Overlay className="dialog-overlay" />
        <Dialog.Content className="evidence-drawer">
          <div className="drawer-heading">
            <span className="eyebrow">CASE RECORD</span>
            <Dialog.Close
              className="icon-button"
              aria-label="Close evidence inspector"
            >
              <X size={18} />
            </Dialog.Close>
          </div>
          <Dialog.Title>Evidence & assumptions</Dialog.Title>
          <Dialog.Description>
            Review the source, physical coordinate frame and unresolved inputs
            behind this workspace.
          </Dialog.Description>
          {caseData && (
            <>
              <section className="record-section">
                <h3>Source identity</h3>
                <dl>
                  <dt>Case</dt>
                  <dd>{caseData.caseId}</dd>
                  <dt>Collection</dt>
                  <dd>
                    {String(
                      collection.name ??
                        collection.accession ??
                        (caseData.metadata.is_synthetic
                          ? "Synthetic geometry fixture"
                          : "Imported source"),
                    )}
                  </dd>
                  <dt>Source equivalence</dt>
                  <dd>
                    {String(
                      caseData.metadata.primary_source_equivalence ??
                        "See source record",
                    )}
                  </dd>
                  <dt>Annotation</dt>
                  <dd>
                    {String(
                      caseData.metadata.annotation_review ?? "Review pending",
                    )}
                  </dd>
                </dl>
              </section>
              <StructuralImportDialog
                disabled={!canImport}
                onImport={onImport}
              />
              <section className="record-section">
                <h3>Physical frame</h3>
                <dl>
                  <dt>Convention</dt>
                  <dd>{caseData.frame} · millimeters</dd>
                  <dt>Source grid</dt>
                  <dd>{caseData.shape.join(" × ")}</dd>
                  <dt>Voxel spacing</dt>
                  <dd>
                    {caseData.spacingMm
                      .map((value) => value.toFixed(2))
                      .join(" × ")}{" "}
                    mm
                  </dd>
                </dl>
                <div className="affine-label">Voxel → source world</div>
                <div className="affine-matrix">
                  {caseData.affine.map((row, index) => (
                    <div key={index}>
                      {row.map((value, column) => (
                        <span key={column}>{value.toFixed(2)}</span>
                      ))}
                    </div>
                  ))}
                </div>
              </section>
              {!!caseData.sourceRefs?.length && (
                <section className="record-section">
                  <h3>Recorded source inputs</h3>
                  <div className="source-inputs">
                    {caseData.sourceRefs.map((source) => (
                      <div key={source.source_id}>
                        <strong>{readableName(source.source_id)}</strong>
                        <span>{source.uri.split("/").pop() ?? source.uri}</span>
                        <small>
                          {source.license ?? "License not recorded"} ·{" "}
                          {source.provenance ?? "Provenance unassessed"}
                        </small>
                        {source.sha256 && (
                          <code title={source.sha256}>
                            SHA256 {source.sha256.slice(0, 16)}…
                          </code>
                        )}
                      </div>
                    ))}
                  </div>
                </section>
              )}
              {caseData.metadata.fractional_annotation != null && (
                <section className="record-section">
                  <h3>Annotation derivation</h3>
                  <p className="structural-method">
                    The binary target is derived from the creator-supplied
                    fractional annotation using an explicit research threshold.
                    The source intensities are not calibrated probabilities.
                  </p>
                  <dl>
                    <dt>Threshold</dt>
                    <dd>
                      {String(
                        (
                          caseData.metadata.fractional_annotation as Record<
                            string,
                            unknown
                          >
                        ).threshold ?? "unrecorded",
                      )}
                    </dd>
                    <dt>Review</dt>
                    <dd>
                      {String(
                        caseData.metadata.annotation_review ?? "unassessed",
                      )}
                    </dd>
                  </dl>
                </section>
              )}
              <StructuralEvidenceInventory
                detailed
                evidence={caseData.structuralEvidence ?? []}
              />
              {priorView && <PriorProvenance proposal={priorView.proposal} />}
              <section className="record-section">
                <h3>Unresolved inputs</h3>
                <ul className="unknown-list">
                  {caseData.unknowns.map((item) => (
                    <li key={item}>
                      <span className="unknown-dot" />
                      {readableName(item)}
                    </li>
                  ))}
                </ul>
              </section>
              <button
                className="quiet-button advanced-toggle"
                onClick={() => setAdvanced(!advanced)}
                aria-expanded={advanced}
              >
                <ChevronDown size={14} className={advanced ? "rotated" : ""} />{" "}
                Advanced audit record
              </button>
              {advanced && (
                <pre className="audit-json">
                  {JSON.stringify(
                    { case: caseData, selectedRoutes: routes },
                    null,
                    2,
                  )}
                </pre>
              )}
            </>
          )}
        </Dialog.Content>
      </Dialog.Portal>
    </Dialog.Root>
  );
}

export function RouteSelectionControls({
  routes,
  available,
  category,
  routeA,
  routeB,
  onCategory,
  onRoute,
}: {
  routes: RouteCandidate[];
  available: RouteCandidate[];
  category: keyof typeof categoryLabels;
  routeA: string;
  routeB: string;
  onCategory: (category: keyof typeof categoryLabels) => void;
  onRoute: (letter: "A" | "B", routeId: string) => void;
}) {
  if (!routes.length) return null;
  return (
    <>
      <div className="category-row">
        <div className="select-wrap category-select">
          <select
            aria-label="Candidate category"
            value={category}
            onChange={(event) =>
              onCategory(event.target.value as typeof category)
            }
          >
            {Object.entries(categoryLabels).map(([key, value]) => (
              <option value={key} key={key}>
                {value}
              </option>
            ))}
          </select>
          <ChevronDown size={12} />
        </div>
        <span className="count-pill">{available.length}</span>
      </div>
      <div className="route-selectors">
        {(["A", "B"] as const).map((letter) => (
          <label className="route-selector" key={letter}>
            <span
              className={`route-letter ${letter === "A" ? "route-a" : "route-b"}`}
            >
              {letter}
            </span>
            <div className="select-wrap">
              <select
                aria-label={
                  letter === "A" ? "Primary route A" : "Comparison route B"
                }
                value={letter === "A" ? routeA : routeB}
                onChange={(event) => onRoute(letter, event.target.value)}
                disabled={!available.length}
              >
                <option value="">
                  {letter === "A" ? "Select a route" : "Add a comparison"}
                </option>
                {available.map((route) => (
                  <option value={route.route_id} key={route.route_id}>
                    {routeName(route, routes)}
                  </option>
                ))}
              </select>
              <ChevronDown size={12} />
            </div>
          </label>
        ))}
      </div>
    </>
  );
}

export function RouteComparison({
  selected,
  all,
  emptyPrompt,
  onFailure,
}: {
  selected: SelectedRoute[];
  all: RouteCandidate[];
  emptyPrompt: ReturnType<typeof routeComparisonPrompt>;
  onFailure: (point: Vec3) => void;
}) {
  if (!selected.length)
    return (
      <div className="empty-comparison">
        <Waypoints size={28} strokeWidth={1.1} />
        <h3>{emptyPrompt.title}</h3>
        <p>{emptyPrompt.description}</p>
      </div>
    );
  const metrics: { label: string; get: (route: RouteCandidate) => string }[] = [
    {
      label: "Accessible target",
      get: (route) => mL(route.accessible_target_volume_mm3),
    },
    {
      label: "Route length",
      get: (route) => millimeters(route.route_length_mm),
    },
    {
      label: "Clearance¹",
      get: (route) => millimeters(route.geometry.clearance_mm),
    },
    {
      label: "Normal exposure²",
      get: (route) => mL(route.normal_tissue_exposure_mm3),
    },
    {
      label: "Assessment",
      get: (route) =>
        route.category === "rejected"
          ? "Rejected"
          : route.assessment === "incomplete"
            ? "Incomplete"
            : route.assessment,
    },
  ];
  return (
    <div className="comparison-content">
      <table className="comparison-table">
        <thead>
          <tr>
            <th>Modeled metric</th>
            {selected.map((route, index) => (
              <th key={route.route_id}>
                <span
                  className={`route-letter ${route.comparisonSlot === "B" ? "route-b" : "route-a"}`}
                >
                  {route.comparisonSlot}
                </span>
                <span>{routeName(route, all).split(" · ")[0]}</span>
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {metrics.map((metric) => (
            <tr key={metric.label}>
              <th>{metric.label}</th>
              {selected.map((route) => (
                <td
                  key={route.route_id}
                  className={
                    metric.label === "Assessment" ? "assessment-cell" : ""
                  }
                >
                  {metric.get(route)}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
      <div className="metric-notes">
        <p>¹ Conservative lower bound</p>
        <p>² Geometric exposure, not tissue removal</p>
        <p>
          Removed target volume <span>Not simulated</span>
        </p>
      </div>
      {selected.map(
        (route, index) =>
          route.geometry.failures.length > 0 && (
            <div className="failure-card" key={route.route_id}>
              <div>
                <span
                  className={`route-letter ${route.comparisonSlot === "B" ? "route-b" : "route-a"}`}
                >
                  {route.comparisonSlot}
                </span>
                <strong>
                  {readableName(route.geometry.failures[0].reason)}
                </strong>
              </div>
              <p>{route.geometry.failures[0].detail}</p>
              <button
                className="text-button"
                onClick={() =>
                  onFailure(route.geometry.failures[0].position_mm)
                }
              >
                Inspect failure location <Focus size={13} />
              </button>
            </div>
          ),
      )}
      <details className="compartment-details">
        <summary>
          Target compartments <ChevronDown size={14} />
        </summary>
        <table>
          <tbody>
            {Object.keys(
              selected[0].accessible_target_volume_mm3_by_compartment,
            ).map((name) => (
              <tr key={name}>
                <th>{readableName(name)}</th>
                {selected.map((route) => (
                  <td key={route.route_id}>
                    {mL(
                      route.accessible_target_volume_mm3_by_compartment[name],
                    )}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </details>
    </div>
  );
}

export default function App() {
  const [api, setApi] = useState<ResectionApi | null>(
    window.resectionApi ?? null,
  );
  const [payload, setPayload] = useState<CasePayload | null>(null);
  const [caseData, setCaseData] = useState<ViewerCase | null>(null);
  const [visibleLayers, setVisibleLayers] = useState<Record<string, boolean>>(
    {},
  );
  const [overlayOpacity, setOverlayOpacity] = useState(0.32);
  const [cursor, setCursor] = useState<Vec3 | null>(null);
  const annotationCenterMm = useMemo(
    () => annotationCenter(payload, caseData),
    [payload, caseData],
  );
  const [cameraMode, setCameraMode] = useState<"anatomy" | "instruments">(
    "anatomy",
  );
  const [routes, setRoutes] = useState<RouteCandidate[]>([]);
  const [episodeView, setEpisodeView] = useState<EpisodeView | null>(null);
  const [episodeStep, setEpisodeStep] = useState(0);
  const [episodeVisible, setEpisodeVisible] = useState(false);
  const [episodePending, setEpisodePending] = useState(false);
  const [episodeSelector, setEpisodeSelector] = useState<DevelopmentEpisodeRequest["selector"]>("scripted");
  const episodeGeneration = useRef(0);
  const [certifiedReplay, setCertifiedReplay] =
    useState<CertifiedReplay | null>(null);
  const [proposalView, setProposalView] =
    useState<StructuralProposalView | null>(null);
  const [proposalLoadingId, setProposalLoadingId] = useState<string | null>(
    null,
  );
  const proposalRequest = useRef<AbortController | null>(null);
  const clearProposal = useCallback(() => {
    proposalRequest.current?.abort();
    proposalRequest.current = null;
    setProposalLoadingId(null);
    setProposalView(null);
  }, []);
  const [priorView, setPriorView] = useState<PriorLayerView | null>(null);
  const [priorLoadingId, setPriorLoadingId] = useState<string | null>(null);
  const priorRequest = useRef<AbortController | null>(null);
  const clearPrior = useCallback(() => {
    priorRequest.current?.abort();
    priorRequest.current = null;
    setPriorLoadingId(null);
    setPriorView(null);
  }, []);
  useEffect(() => () => priorRequest.current?.abort(), []);
  const receiveReplay = useCallback(
    (replay: CertifiedReplay | null) => {
      if (replay) {
        setEpisodeVisible(false);
        clearProposal();
        clearPrior();
      }
      setCertifiedReplay(replay);
    },
    [clearProposal, clearPrior],
  );
  useEffect(() => () => proposalRequest.current?.abort(), []);
  const [category, setCategory] = useState<"pareto" | "dominated" | "rejected">(
    "pareto",
  );
  const [routeA, setRouteA] = useState("");
  const [routeB, setRouteB] = useState("");
  const [planningTab, setPlanningTab] = useState("routes");
  const [instrument, setInstrument] = useState("all");
  const [allowEstimatedSupport, setAllowEstimatedSupport] = useState(false);
  const [operation, setOperation] = useState<Operation | null>(null);
  const [hydrating, setHydrating] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [message, setMessage] = useState(
    "Local workspace · Open imaging to begin",
  );
  const [searchSeconds, setSearchSeconds] = useState<number | null>(null);
  const [combinedModels, setCombinedModels] = useState(false);
  const [casePanel, setCasePanel] = useState(true);
  const [engineOperations, setEngineOperations] = useState<Set<string>>(
    new Set(),
  );
  const [engineStopped, setEngineStopped] = useState(false);
  const stoppedEngine = useRef(false);
  const mounted = useRef(true);
  const caseGeneration = useRef(0);
  const activeCaseHash = useRef<string | null>(null);
  const imaging = useWorkspaceImaging(payload, api);
  const resetDisplayToPrimary = imaging.resetDisplayToPrimary;
  const viewerCase = useMemo(() => caseData ? {...caseData, planningHash: payload?.planningHash} : null,
    [caseData, payload?.planningHash]);
  const display = workspaceDisplay(viewerCase, imaging.selected);
  const displayedCase = display.caseData;
  const displayedSource = imaging.selected?.descriptor.volume ?? payload;
  const displayedCursor = imaging.selected ? imaging.cursor : cursor;
  const displayedLayers = imaging.selected ? imaging.visibleLayers : visibleLayers;
  const setDisplayedLayers = imaging.selected ? imaging.setVisibleLayers : setVisibleLayers;
  const busy = !!operation || hydrating || episodePending || imaging.pending;
  const readonly = !!api?.readOnly;
  const controlsBlocked = busy || engineStopped || !display.planningInteractionPermitted;
  const fileNavigationBlocked = !api || busy || engineStopped || readonly;
  const restoreSourceView = () => {
    resetDisplayToPrimary();
    setEpisodeVisible(false);
    clearPrior();
    clearProposal();
    setCertifiedReplay(null);
    setMessage("Source imaging restored · Source annotations preserved");
  };
  const reportError = (failure: unknown) =>
    setError(
      stoppedEngine.current
        ? "The local computation engine stopped. Your visible case is preserved. Reopen the app to reconnect."
        : operationMessage(failure),
    );

  const installSearch = useCallback((result: SearchResult) => {
    if (
      result.case_hash !== activeCaseHash.current ||
      result.candidates.some((route) => route.case_hash !== result.case_hash)
    ) {
      setError("A route result belongs to another case and was withheld.");
      return;
    }
    setRoutes(result.candidates);
    setSearchSeconds(result.elapsed_seconds);
    setCombinedModels(result.combined_models === true);
    setAllowEstimatedSupport(
      result.assumptions.some((value) =>
        value.includes("Hypothetical window support: estimated"),
      ),
    );
    const retained = result.candidates.filter(
      (route) => route.category === "pareto",
    );
    setCategory("pareto");
    setRouteA(retained[0]?.route_id ?? "");
    setRouteB(retained[1]?.route_id ?? "");
    setMessage(
      `${result.candidates.length} candidates evaluated · Search completed in ${result.elapsed_seconds.toFixed(2)} s`,
    );
  }, []);

  const installCase = useCallback(
    async (sourceInput: CasePayload, runtime: ResectionApi, requestedGeneration?: number) => {
      const generation = requestedGeneration ?? ++caseGeneration.current;
      if (generation !== caseGeneration.current) return false;
      imaging.invalidate();
      setEpisodeVisible(false);
      setHydrating(true);
      setError(null);
      try {
        const reopened = await hydrateWorkspace(sourceInput, runtime,
          () => mounted.current && generation === caseGeneration.current);
        const { source, volume: loaded } = reopened;
        if (!mounted.current || generation !== caseGeneration.current)
          return false;
        activeCaseHash.current = source.caseHash;
        clearPrior();
        clearProposal();
        resetDisplayToPrimary();
        setPayload(source);
        setCaseData(loaded);
        setCertifiedReplay(null);
        const saved = source.artifacts?.workspace as
          | Record<string, unknown>
          | undefined;
        const workspace = saved?.case_hash === source.caseHash ? saved : {};
        const layers = (workspace.visibleLayers ?? {}) as Record<
          string,
          unknown
        >;
        setVisibleLayers(
          Object.fromEntries(
            loaded.compartments.map((layer) => [
              layer.name,
              typeof layers[layer.name] === "boolean"
                ? Boolean(layers[layer.name])
                : true,
            ]),
          ),
        );
        if (
          typeof workspace.overlayOpacity === "number" &&
          workspace.overlayOpacity >= 0 &&
          workspace.overlayOpacity <= 0.75
        )
          setOverlayOpacity(workspace.overlayOpacity);
        else setOverlayOpacity(0.32);
        const savedCursor = workspace.cursor;
        setCursor(
          Array.isArray(savedCursor) &&
            savedCursor.length === 3 &&
            savedCursor.every(
              (value) => typeof value === "number" && Number.isFinite(value),
            )
            ? (savedCursor as Vec3)
            : source.targetCentroidMm
              ? rasPoint(source.targetCentroidMm, source.frame)
              : initialCursor(loaded),
        );
        const restored =
          source.artifacts?.savedRouteValidation ===
            "matches_current_session_evaluation" &&
          Array.isArray(workspace.routes)
            ? (workspace.routes as RouteCandidate[])
            : [];
        const checked = restored.filter(
          (route) =>
            route.case_hash === source.caseHash &&
            route.clinical_deficit_probability == null &&
            route.simulated_removed_target_volume_mm3 == null,
        );
        setRoutes(checked);
        const nextCategory = ["pareto", "dominated", "rejected"].includes(
          String(workspace.category),
        )
          ? (workspace.category as typeof category)
          : "pareto";
        setCategory(nextCategory);
        const choices = checked.filter(
          (route) => route.category === nextCategory,
        );
        setRouteA(
          choices.some((route) => route.route_id === workspace.routeA)
            ? String(workspace.routeA)
            : (choices[0]?.route_id ?? ""),
        );
        setRouteB(
          choices.some((route) => route.route_id === workspace.routeB)
            ? String(workspace.routeB)
            : (choices[1]?.route_id ?? ""),
        );
        setSearchSeconds(null);
        setCombinedModels(
          new Set(
            checked.map((route) => route.planning_model_hash).filter(Boolean),
          ).size > 1,
        );
        setAllowEstimatedSupport(
          checked.some((route) =>
            route.assumptions.some((value) =>
              value.includes("Hypothetical window support: estimated"),
            ),
          ),
        );
        imaging.restore(source.caseHash, reopened.series, reopened.imagingState);
        if (reopened.imagingState) {
          setCursor(reopened.imagingState.states.primary.cursor);
          setVisibleLayers(reopened.imagingState.states.primary.visibleLayers);
        }
        setEpisodeView(reopened.episodeView);
        setEpisodeStep(reopened.episodeStep);
        setEpisodeVisible(reopened.episodeVisible);
        if (reopened.episodeView) setEpisodeSelector(reopened.episodeView.episode.selector);
        if (reopened.episodeVisible) { setCameraMode("instruments"); setPlanningTab("episode"); }
        setMessage(
          reopened.session ? `Workspace reopened · native image views restored${reopened.episodeView ? " · saved replay revalidated" : ""}` :
          `${source.metadata.is_synthetic ? "Synthetic fixture" : source.caseId} ready · Source imaging preserved`,
        );
        return true;
      } catch (failure) {
        if (generation === caseGeneration.current)
          setError(
            failure instanceof Error ? failure.message : String(failure),
          );
        return false;
      } finally {
        if (generation === caseGeneration.current) setHydrating(false);
      }
    },
    [],
  );

  const showEpisode = () => {
    resetDisplayToPrimary();
    if (!episodeView || episodeView.source.caseHash !== activeCaseHash.current) return;
    clearPrior(); clearProposal(); setCertifiedReplay(null); setEpisodeVisible(false);
    setEpisodeVisible(true); setCameraMode("instruments");
    setMessage("Generated episode · recorded tool poses and modeled cavity · no patient admission");
  };
  const requestEpisodeStep = (step: number) => {
    if (!episodeView || !Number.isInteger(step) || step < 0 || step >= episodeView.frames.length) return;
    setEpisodeStep(step); showEpisode();
  };
  const executeEpisode = async () => {
    if (!api?.executeDevelopmentEpisode || controlsBlocked || readonly) return;
    const generation = ++episodeGeneration.current;
    setEpisodePending(true); setError(null);
    try {
      const result = await api.executeDevelopmentEpisode({fixture:"generated-sequential-v1",selector:episodeSelector});
      const checked = await hydrateDevelopmentEpisode(result,api);
      if (!mounted.current || generation !== episodeGeneration.current) return;
      if (!await installCase(checked.source,api)) return;
      setEpisodeView(checked); setEpisodeStep(0); setEpisodeVisible(true);
      setCameraMode("instruments"); setPlanningTab("episode");
      setMessage("Generated software episode executed · all recorded frame masks checked · no patient admission");
    } catch (failure) { if (mounted.current && generation === episodeGeneration.current) reportError(failure); }
    finally { if (mounted.current && generation === episodeGeneration.current) setEpisodePending(false); }
  };

  useEffect(() => {
    mounted.current = true;
    let disposed = false;
    if (window.resectionApi) {
      const runtime = window.resectionApi;
      runtime
        .startupCase()
        .then((source) => {
          if (source && !disposed) return installCase(source, runtime);
        })
        .catch((failure) => setError(String(failure)));
      return () => {
        disposed = true;
        mounted.current = false;
      };
    }
    readOnlyPreview()
      .then(async (preview) => {
        if (!preview || disposed) return;
        setApi(preview.api);
        const installed = await installCase(preview.caseData, preview.api);
        if (installed && !disposed && preview.search)
          installSearch(preview.search);
      })
      .catch(() =>
        setMessage(
          "Desktop connection unavailable · Open this workspace in the Mac app",
        ),
      );
    return () => {
      disposed = true;
      mounted.current = false;
    };
  }, [installCase, installSearch]);

  useEffect(() => {
    if (!api) return;
    return api.onEvent((event: BridgeEvent) => {
      if (event.event === "engineStopped") {
        clearPrior();
        clearProposal();
        stoppedEngine.current = true;
        setEngineStopped(true);
        setEngineOperations(new Set());
        setOperation(null);
        setError(
          "The local computation engine stopped. Your visible case is preserved. Reopen the app to reconnect.",
        );
        return;
      }
      if (["cancel", "ping", "inspectEvidence"].includes(event.op ?? ""))
        return;
      if (event.event === "started")
        setOperation({
          id: event.id,
          op: event.op ?? "",
          fraction: 0,
          message: "Preparing local operation…",
        });
      if (event.event === "progress")
        setOperation((current) =>
          current?.id === event.id
            ? { ...current, ...event.progress }
            : current,
        );
      if (["result", "cancelled", "error"].includes(event.event))
        setOperation((current) => (current?.id === event.id ? null : current));
      if (event.event === "cancelled")
        setMessage("Operation cancelled · Current case preserved");
      if (event.event === "error")
        setError(
          event.error?.message ?? "The operation could not be completed.",
        );
    });
  }, [api]);

  useEffect(() => {
    let disposed = false;
    setEngineOperations(new Set());
    if (api && !api.readOnly)
      api
        .ping()
        .then((value) => {
          if (!disposed) {
            const operations = (value as { operations?: unknown }).operations;
            setEngineOperations(
              new Set(
                Array.isArray(operations)
                  ? operations.filter(
                      (value): value is string => typeof value === "string",
                    )
                  : [],
              ),
            );
          }
        })
        .catch(() => {});
    return () => {
      disposed = true;
    };
  }, [api]);

  const act = async (work: () => Promise<unknown>) => {
    setError(null);
    try {
      await work();
    } catch (failure) {
      if (isOperationCancelled(failure))
        setMessage("Operation cancelled · Current case preserved");
      else reportError(failure);
    }
  };
  const load = (kind: "openCase" | "importNifti" | "createSyntheticCase") =>
    act(async () => {
      if (!api || fileNavigationBlocked) return;
      const generation = ++caseGeneration.current;
      imaging.invalidate();
      const source = await api[kind]();
      if (source && mounted.current && generation === caseGeneration.current)
        await installCase(source, api, generation);
    });
  const viewPrior = async (proposalId: string) => {
    if (!payload || !caseData || !api || episodeVisible || certifiedReplay || controlsBlocked)
      return;
    clearPrior();
    clearProposal();
    const controller = new AbortController();
    priorRequest.current = controller;
    setPriorLoadingId(proposalId);
    setMessage("Loading selected population prior · Source imaging preserved");
    setError(null);
    try {
      const view = await hydratePriorProposal(
        payload,
        caseData,
        proposalId,
        api,
        controller.signal,
      );
      if (
        controller.signal.aborted ||
        priorRequest.current !== controller ||
        activeCaseHash.current !== view.caseHash
      )
        return;
      setPriorView(view);
      setMessage(
        "Population prior shown · alignment review required · not used in route scoring",
      );
    } catch (failure) {
      if (!controller.signal.aborted && priorRequest.current === controller) {
        setMessage("Prior withheld · Source imaging preserved");
        reportError(failure);
      }
    } finally {
      if (priorRequest.current === controller) {
        priorRequest.current = null;
        setPriorLoadingId(null);
      }
    }
  };
  const viewProposal = async (evidenceId: string) => {
    if (!payload || !caseData || !api || episodeVisible || certifiedReplay || controlsBlocked)
      return;
    clearPrior();
    clearProposal();
    const controller = new AbortController();
    proposalRequest.current = controller;
    setProposalLoadingId(evidenceId);
    setMessage(
      "Loading selected brain-envelope estimate · Source imaging preserved",
    );
    setError(null);
    try {
      const view = await hydrateStructuralProposal(
        payload,
        caseData,
        evidenceId,
        api,
        controller.signal,
      );
      if (
        controller.signal.aborted ||
        proposalRequest.current !== controller ||
        activeCaseHash.current !== view.caseHash
      )
        return;
      setProposalView(view);
      setMessage(
        "Estimated envelope shown on source MRI · display only · no working anatomy changed",
      );
    } catch (failure) {
      if (
        !controller.signal.aborted &&
        proposalRequest.current === controller
      ) {
        setMessage("Estimate withheld · Source imaging preserved");
        reportError(failure);
      }
    } finally {
      if (proposalRequest.current === controller) {
        proposalRequest.current = null;
        setProposalLoadingId(null);
      }
    }
  };
  const importStructural = (variant: "main" | "nocsf") =>
    act(async () => {
      if (!api?.importStructuralEvidence || !payload) return;
      const source = await api.importStructuralEvidence({
        caseHash: payload.caseHash,
        variant,
      });
      if (source && (await installCase(source, api)))
        setMessage(
          "Structural proposal added locally · review required · working anatomy unchanged",
        );
    });
  const generateNativeAlternatives = () =>
    act(async () => {
      if (!api?.generateNativeRoutes || !payload) return;
      const oldIds = new Set(routes.map((route) => route.route_id));
      const previous = routes.find((route) => route.route_id === routeA);
      const result = await api.generateNativeRoutes({
        caseHash: payload.caseHash,
      });
      if (result.case_hash !== activeCaseHash.current) return;
      installSearch(result);
      const added = result.candidates.filter(
        (route) => !oldIds.has(route.route_id),
      );
      const chosen =
        added.find((route) => route.category === "pareto") ?? added[0];
      if (chosen) {
        setCategory(chosen.category);
        setRouteA(chosen.route_id);
        setRouteB(
          previous?.category === chosen.category
            ? previous.route_id
            : (added.find(
                (route) =>
                  route.route_id !== chosen.route_id &&
                  route.category === chosen.category,
              )?.route_id ?? ""),
        );
      }
      setMessage(
        `${added.length} additional research routes added · original routes preserved · review the new geometry`,
      );
    });
  const save = () =>
    act(async () => {
      if (!api || !payload) return;
      const result = await api.saveCase({
        caseHash: payload.caseHash,
        workspace: {
          routeA,
          routeB,
          category,
          visibleLayers,
          overlayOpacity,
          cursor,
          imaging: imaging.snapshot({cursor, visibleLayers}),
          episodeReplay: episodeView?.source.caseHash === payload.caseHash ? {
            episodeId: episodeView.episode.episodeId, frameIndex: episodeStep,
            visible: episodeVisible && !imaging.selected,
          } : null,
        },
      });
      if (result?.saved) setMessage("Workspace saved locally · images, views and recorded replay preserved");
    });

  useEffect(() => {
    if (!api) return;
    return api.onEvent((event) => {
      if (event.event !== "menuAction" || busy || engineStopped || readonly) return;
      if (event.action === "openCase") void load("openCase");
      if (event.action === "saveCase" && payload) void save();
    });
  }, [
    api,
    busy,
    engineStopped,
    readonly,
    payload,
    imaging.snapshot,
    imaging.selected,
    episodeView,
    episodeStep,
    episodeVisible,
    routeA,
    routeB,
    category,
    visibleLayers,
    overlayOpacity,
    cursor,
  ]);

  const available = useMemo(
    () => routes.filter((route) => route.category === category),
    [routes, category],
  );
  const selected = useMemo(
    () => selectComparisonRoutes(routes, routeA, routeB),
    [routes, routeA, routeB],
  );
  const viewerRoutes = useMemo(
    () =>
      selected.map((route) =>
        payload?.frame.startsWith("LPS")
          ? {
              ...route,
              entry_mm: rasPoint(route.entry_mm, payload.frame),
              target_mm: rasPoint(route.target_mm, payload.frame),
              window: {
                ...route.window,
                center_mm: rasPoint(route.window.center_mm, payload.frame),
                normal_inward: rasPoint(
                  route.window.normal_inward,
                  payload.frame,
                ),
              },
              geometry: {
                ...route.geometry,
                failures: route.geometry.failures.map((failure) => ({
                  ...failure,
                  position_mm: rasPoint(failure.position_mm, payload.frame),
                })),
              },
            }
          : route,
      ),
    [selected, payload],
  );
  const viewerReplay = useMemo(
    () =>
      episodeVisible && episodeView && episodeView.source.caseHash === caseData?.caseHash
        ? episodeView.frames[episodeStep]
        : certifiedReplay && caseData
        ? {
            removedMask: certifiedReplay.mask,
            step: certifiedReplay.result.step,
            stepCount: certifiedReplay.result.stepCount,
            scope: "native-source-grid" as const,
            caseHash: certifiedReplay.result.caseHash,
            shape: caseData.shape,
            affine: certifiedReplay.result.affine,
            independentlyAccepted: true,
            removedTargetVolumeMm3:
              certifiedReplay.result.simulatedRemovedTargetVolumeMm3,
            removedNormalVolumeMm3:
              certifiedReplay.result.simulatedRemovedNormalVolumeMm3,
            residualTargetVolumeMm3:
              certifiedReplay.result.modeledResidualTargetVolumeMm3,
          }
        : null,
    [certifiedReplay, caseData, episodeVisible, episodeView, episodeStep],
  );
  const synthetic = !!payload?.metadata.is_synthetic;
  const supportGate = researchSupportGate(payload);
  const needsSupport = supportGate.requiresEstimatedSupport;
  const canGenerate =
    !!payload &&
    !!caseData?.compartments.length &&
    !controlsBlocked &&
    !readonly &&
    !supportGate.blocked &&
    (!needsSupport || allowEstimatedSupport);
  const fractionalAnnotation = payload?.metadata.fractional_annotation as
    | Record<string, unknown>
    | undefined;
  const annotationDescription = synthetic ? "Generated target" :
    fractionalAnnotation?.derived_provenance === "estimated"
      ? "Threshold-derived annotation"
      : "Supplied annotation";
  const title = synthetic
    ? "Synthetic access fixture"
    : (payload?.caseId ?? "No case open");

  function changeCategory(value: typeof category) {
    setCategory(value);
    const next = routes.filter((route) => route.category === value);
    setRouteA(next[0]?.route_id ?? "");
    setRouteB(next[1]?.route_id ?? "");
  }

  return (
    <div className={`app-shell ${casePanel ? "" : "case-panel-collapsed"}`}>
      <header className="app-header">
        <div className="window-traffic-space" />
        <div className="brand">
          <Logo />
          <span>
            Ressection<span className="brand-light">Lab</span>
          </span>
        </div>
        <div className="header-divider" />
        <span className="header-purpose">Patient-specific planning</span>
        <div className="header-spacer" />
        <span className="research-tag">
          <FlaskConical size={11} /> Research workspace
        </span>
        <button
          className="header-button"
          onClick={() => load("openCase")}
          disabled={fileNavigationBlocked}
        >
          <FolderOpen size={15} />
          <span>Open case</span>
          <kbd>⌘O</kbd>
        </button>
        <button
          className="header-button"
          onClick={() => load("importNifti")}
          disabled={fileNavigationBlocked}
        >
          <ArrowUpFromLine size={15} />
          <span>Import MRI</span>
        </button>
        <button
          className="header-button save-button"
          onClick={save}
          disabled={!payload || busy || engineStopped || readonly}
        >
          <ArrowDownToLine size={15} />
          <span>Save</span>
        </button>
      </header>

      <nav className="navigation-rail" aria-label="Workspace navigation">
        <button
          className={`rail-button ${casePanel ? "selected" : ""}`}
          onClick={() => setCasePanel(!casePanel)}
          aria-label="Toggle case and evidence panel"
          aria-expanded={casePanel}
          aria-controls="case-evidence-panel"
        >
          <Layers3 size={19} />
          <span>Case</span>
        </button>
        <button
          className="rail-button"
          onClick={() =>
            document.querySelector<HTMLElement>("[data-route-panel]")?.focus()
          }
          aria-label="Focus route comparison"
        >
          <Waypoints size={19} />
          <span>Plan</span>
        </button>
        <div className="rail-spacer" />
        <span className="rail-local" title="Processing stays on this Mac">
          <ShieldCheck size={17} />
        </span>
      </nav>

      <aside className="case-panel" id="case-evidence-panel">
        <div className="panel-heading">
          <span className="eyebrow">CASE & EVIDENCE</span>
          <button
            className="icon-button"
            aria-label="Collapse case panel"
            onClick={() => setCasePanel(false)}
          >
            <PanelLeftClose size={15} />
          </button>
        </div>
        <h1>{title}</h1>
        <p className="case-subtitle">
          {displayedSource
            ? displayedSource.metadata.is_synthetic
              ? "Geometry demonstration · no patient data"
              : displayedSource.compartments.length > 0
                ? "Annotation-assisted structural workspace"
                : "Source-image workspace · no target annotations"
            : "A focused workspace for inspecting anatomy and candidate routes."}
        </p>
        {payload?.metadata.primary_source_equivalence === "unverified" && (
          <span className="source-badge">
            <CircleDot size={11} /> Public mirror · source match unverified
          </span>
        )}
        {!payload && (
          <button
            className="outline-button demo-button"
            onClick={() => setPlanningTab("episode")}
            disabled={!api || readonly || controlsBlocked}
          >
            <FlaskConical size={15} /> Open generated episode
          </button>
        )}
        {payload && <ImagingWorkspacePanel caseHash={payload.caseHash} series={imaging.series} selectedId={imaging.selectedId}
          busy={busy || engineStopped} importAvailable={!!api?.importDisplaySeries && engineOperations.has("importDisplaySeries") && !readonly}
          onSelect={imaging.select} onImport={request => void act(() => imaging.importSeries(request))} />}
        <section className="case-section">
          <h2>
            Displayed image <span>{displayedSource ? "01" : "—"}</span>
          </h2>
          <div className="sequence-card">
            <span className="sequence-icon">
              <Scan size={19} />
            </span>
            <div>
              <strong>
                {displayedSource
                  ? String(
                      displayedSource.metadata.is_synthetic ? "Generated analytic signal" : imaging.selected?.descriptor.modality ?? displayedSource.metadata.selected_modality ?? "Structural MRI",
                    )
                  : "No MRI loaded"}
              </strong>
              <p>
                {displayedSource
                  ? `${displayedSource.shape.join(" × ")} voxels`
                  : "NIfTI source volume"}
              </p>
            </div>
            {displayedSource && <Check size={13} className="subtle-check" />}
          </div>
          {displayedSource && (
            <p className="grid-note">
              {displayedSource.spacingMm
                .map((value) => `${value.toFixed(1)}`)
                .join(" × ")}{" "}
              mm · {displayedSource.frame}
            </p>
          )}
          {displayedSource?.metadata.source_anatomy_review === "not_planning_accepted" && (
            <p className="muted-note" role="note">
              Source-reported coordinates. Anatomical orientation and planning suitability remain unreviewed.
            </p>
          )}
        </section>
        <section className="case-section">
          <h2>
            {imaging.selected ? (imaging.selected.descriptor.annotationKind === "estimated" ? "Estimated labels · display only" : "Source labels · display only") : "Target annotations"}{" "}
            <span>
              {displayedCase?.compartments.length.toString().padStart(2, "0") ?? "—"}
            </span>
          </h2>
          <AnnotationCenterButton
            center={imaging.selected ? null : annotationCenterMm}
            onCenter={setCursor}
          />
          {displayedCase?.compartments.map((layer) => (
            <label className="layer-row" key={layer.name}>
              <input
                type="checkbox"
                checked={displayedLayers[layer.name] ?? false}
                onChange={(event) =>
                  setDisplayedLayers((current) => ({
                    ...current,
                    [layer.name]: event.target.checked,
                  }))
                }
              />
              <span
                className="layer-swatch"
                style={{ background: layer.color }}
              />
              <div>
                <strong>{readableName(layer.name)}</strong>
                <span>
                  {(layer.volumeMm3 / 1000).toFixed(2)} mL ·{" "}
                  {imaging.selected ? imaging.selected.descriptor.annotationKind : annotationDescription.toLowerCase()}
                </span>
              </div>
            </label>
          ))}
          {!displayedCase?.compartments.length && (
            <p className="muted-note">
              Import a supplied mask to inspect its target compartments.
            </p>
          )}
          <div className="opacity-control">
            <label htmlFor="opacity">
              Overlay opacity <span>{Math.round(overlayOpacity * 100)}%</span>
            </label>
            <input
              id="opacity"
              type="range"
              min="0"
              max="0.75"
              step="0.01"
              value={overlayOpacity}
              onChange={(event) =>
                setOverlayOpacity(Number(event.target.value))
              }
              disabled={!caseData}
            />
          </div>
        </section>
        <StructuralEvidenceInventory
          evidence={payload?.structuralEvidence ?? []}
          inspection={{
            selectedId: proposalView?.evidenceId,
            loadingId: proposalLoadingId ?? undefined,
            disabled: controlsBlocked || episodeVisible || !!certifiedReplay,
            onSelect: (id) => void viewProposal(id),
            onClear: restoreSourceView,
            outsideCount: proposalView?.annotationOutsideVoxelCount,
            onOutside: proposalView?.outsideAnnotationPointMm
              ? () => setCursor(proposalView.outsideAnnotationPointMm)
              : undefined,
          }}
        />
        {!!payload?.structuralEvidence?.length && certifiedReplay && (
          <p className="proposal-mode-hint">
            Return to source view to inspect structural estimates.
          </p>
        )}
        <PriorInventory
          items={payload?.priorProposals ?? []}
          selected={priorView}
          loadingId={priorLoadingId}
          disabled={controlsBlocked || episodeVisible || !!certifiedReplay}
          onSelect={(id) => void viewPrior(id)}
          onClear={restoreSourceView}
          cursor={cursor}
        />
        {!payload?.priorProposals?.length && (
          <section className="case-section functional-section">
            <h2>Functional evidence</h2>
            <div className="evidence-line">
              <span>Motor network</span>
              <span className="unassessed-pill">Unassessed</span>
            </div>
            <div className="evidence-line">
              <span>Language network</span>
              <span className="unassessed-pill">Unassessed</span>
            </div>
            <p className="muted-note">
              Missing evidence remains unknown. Population priors require
              registration and review.
            </p>
          </section>
        )}
        <div className="case-panel-bottom">
          <EvidenceDrawer
            caseData={payload}
            routes={selected}
            canImport={
              !!api?.importStructuralEvidence &&
              engineOperations.has("importStructuralEvidence") &&
              !controlsBlocked &&
              !readonly
            }
            onImport={importStructural}
            priorView={priorView}
          />
          <div className="local-note">
            <span className="status-dot" />
            {readonly
              ? "Read-only public-case preview"
              : "Local processing · source preserved"}
          </div>
        </div>
      </aside>

      <main className="anatomy-panel">
        <div className="workspace-heading">
          <div>
            <WorkspaceBreadcrumb caseId={caseData?.caseId ?? null} />
            <h2>Anatomy, in context</h2>
          </div>
          <div
            className="view-presets"
            role="group"
            aria-label="Camera framing"
          >
            <button
              className={cameraMode === "anatomy" ? "active" : ""}
              onClick={() => setCameraMode("anatomy")}
              aria-pressed={cameraMode === "anatomy"}
            >
              <Focus size={14} /> Focus anatomy
            </button>
            <button
              className={cameraMode === "instruments" ? "active" : ""}
              onClick={() => setCameraMode("instruments")}
              aria-pressed={cameraMode === "instruments"}
            >
              <Expand size={14} /> Fit instruments
            </button>
          </div>
        </div>
        {priorView && (
          <div className="prior-view-notice">
            <span className="prior-population-badge">Population prior</span>
            <div>
              <strong>{priorView.title}</strong>
              <span>
                Alignment review required · inspection only · not used in route
                scoring
              </span>
            </div>
            <button className="text-button" onClick={restoreSourceView}>
              Source view <X size={12} />
            </button>
          </div>
        )}
        {proposalView && (
          <div className="proposal-view-notice" role="status">
            <span className="proposal-contour-key" />
            <div>
              <strong>{proposalView.label}</strong>
              <span>
                {proposalView.reviewStatus === "review_required"
                  ? "Review required"
                  : proposalView.reviewStatus === "rejected"
                    ? "Rejected estimate"
                    : "Reviewed envelope"}{" "}
                · view only · cortical access not certified
              </span>
            </div>
            <button className="text-button" onClick={restoreSourceView}>
              Source view <X size={12} />
            </button>
          </div>
        )}
        {episodeVisible && episodeView && (
          <div className="modeled-replay-notice" role="status"><span>GENERATED · modeled cavity · frame {episodeStep}/{episodeView.frames.length-1} · {episodeView.episode.replayFrames[episodeStep].phase} · no patient admission</span><button className="text-button" onClick={restoreSourceView}>Source view <X size={12}/></button></div>
        )}
        {certifiedReplay && (
          <div className="modeled-replay-notice">
            <span>
              <span className="layer-swatch" />
              Modeled removal · native source grid · selection step{" "}
              {certifiedReplay.result.step}/{certifiedReplay.result.stepCount}
            </span>
            <button className="text-button" onClick={restoreSourceView}>
              Source view <X size={12} />
            </button>
          </div>
        )}
        <div className="viewer-shell">
          {caseData && (
            <ViewerWorkspace
              generatedSignal={display.primaryOverlaysPermitted && synthetic}
              caseData={displayedCase}
              visibleLayers={displayedLayers}
              overlayOpacity={overlayOpacity}
              cursor={displayedCursor}
              onCursorChange={imaging.selected ? imaging.setCursor : setCursor}
              routes={display.primaryOverlaysPermitted ? viewerRoutes : []}
              cameraMode={cameraMode}
              replay={display.primaryOverlaysPermitted ? viewerReplay : null}
              structuralProposal={display.primaryOverlaysPermitted ? proposalView : null}
              priorLayer={display.primaryOverlaysPermitted ? priorView : null}
            />
          )}
          {!caseData && (
            <div className="welcome-overlay">
              <div className="welcome-symbol">
                <Scan size={34} strokeWidth={1} />
              </div>
              <span className="eyebrow">YOUR CASE IS THE STARTING POINT</span>
              <h2>
                From source imaging
                <br />
                to inspectable alternatives.
              </h2>
              <p>
                Load a case to explore linked MRI views, source annotations and
                complete instrument geometry.
              </p>
              <button
                className="primary-button"
                disabled={fileNavigationBlocked}
                onClick={() => load("openCase")}
              >
                <FolderOpen size={16} /> Open a case
              </button>
              <button
                className="text-button"
                disabled={!api || readonly || controlsBlocked}
                onClick={() => setPlanningTab("episode")}
              >
                Open the generated episode workflow <ChevronRight size={13} />
              </button>
            </div>
          )}
          {hydrating && (
            <div className="viewer-loading">
              <LoaderCircle size={22} className="spin" />
              <span>Preparing source images…</span>
            </div>
          )}
        </div>
        <div className="workspace-caption">
          <span>
            <span className="mini-crosshair" /> Linked physical coordinates ·
            RAS+ mm
          </span>
          <span>
            {cameraMode === "anatomy"
              ? "Anatomy focus may crop instruments"
              : "Complete rendered instrument envelopes"}
          </span>
        </div>
      </main>

      <aside className="planning-panel" data-route-panel tabIndex={-1}>
        <Tabs.Root value={planningTab} onValueChange={(tab) => {
                setPlanningTab(tab);
        }} className="planning-tabs">
          <Tabs.List
            className="planning-tab-list"
            aria-label="Planning workflows"
          >
            <Tabs.Trigger value="routes">
              <Waypoints size={14} /> Routes
            </Tabs.Trigger>
            <Tabs.Trigger value="episode"><FlaskConical size={14} /> Episode</Tabs.Trigger>
            <Tabs.Trigger value="refinement">
              <Sparkles size={14} /> Refine
            </Tabs.Trigger>
          </Tabs.List>
          <Tabs.Content value="episode" className="planning-tab-content">
            <EpisodePanel vascularApi={!readonly && !engineStopped && engineOperations.has("evaluateDevelopmentEpisodeVascular") ? api : null} view={episodeView?.source.caseHash === payload?.caseHash ? episodeView : null} step={episodeStep}
              selector={episodeSelector} onSelector={setEpisodeSelector} busy={controlsBlocked}
              onExecute={executeEpisode} onStep={requestEpisodeStep} onShow={showEpisode} onSource={restoreSourceView} visible={episodeVisible}
              unavailableReason={engineStopped ? "The local engine stopped. Reopen the app to execute." : readonly ? "This browser preview is read only. Open the desktop app to execute." : !api?.executeDevelopmentEpisode || !engineOperations.has("executeDevelopmentEpisode") ? "This engine does not provide generated development episodes." : undefined}/>
          </Tabs.Content>
          <Tabs.Content value="routes" className="planning-tab-content">
            <div className="planning-title">
              <span className="eyebrow">INSTRUMENT-AWARE SEARCH</span>
              <h2>Compare the alternatives</h2>
              {!routes.length && (
                <p>
                  Inspect modeled access under shared geometric assumptions.
                </p>
              )}
            </div>
            <details className="search-setup" open={!routes.length}>
              <summary>
                Search settings <ChevronDown size={13} />
              </summary>
              <label className="field-label" htmlFor="instrument">
                Instrument configuration
              </label>
              <div className="select-wrap">
                <select
                  id="instrument"
                  value={instrument}
                  onChange={(event) => setInstrument(event.target.value)}
                  disabled={controlsBlocked}
                >
                  <option value="all">Compare both generic tools</option>
                  <option value="generic_suction">
                    Suction · shaft Ø2.8 mm
                  </option>
                  <option value="generic_aspirator">
                    Aspirator · shaft Ø5.4 mm
                  </option>
                </select>
                <ChevronDown size={14} />
              </div>
              <p className="instrument-note">
                Generic research geometry · whole shaft and active tip
              </p>
              {needsSupport && !supportGate.blocked && (
                <label className="support-choice">
                  <input
                    type="checkbox"
                    checked={allowEstimatedSupport}
                    onChange={(event) =>
                      setAllowEstimatedSupport(event.target.checked)
                    }
                    disabled={controlsBlocked || readonly}
                  />
                  <span>
                    Use estimated image support for{" "}
                    <strong>hypothetical</strong> access windows
                  </span>
                </label>
              )}
              {supportGate.reason && (
                <div className="capability-note">{supportGate.reason}</div>
              )}
              <button
                className="primary-button generate-button"
                disabled={!canGenerate}
                onClick={() =>
                  act(async () => {
                    if (api)
                      installSearch(
                        await api.generateRoutes({
                          caseHash: payload?.caseHash,
                          allowEstimatedSupport,
                          toolIds:
                            instrument === "all" ? undefined : [instrument],
                        }),
                      );
                  })
                }
              >
                {operation?.op === "generateRoutes" ? (
                  <LoaderCircle size={16} className="spin" />
                ) : (
                  <Waypoints size={16} />
                )}{" "}
                Generate candidate routes
              </button>
            </details>
            {!!payload?.artifacts?.withheldRoutesReason && !routes.length && (
              <div className="capability-note">
                Saved routes need a fresh search in this engine session. Your
                source case and view settings are preserved.
              </div>
            )}
            <div className="routes-divider" />
            <RouteSelectionControls
              routes={routes}
              available={available}
              category={category}
              routeA={routeA}
              routeB={routeB}
              onCategory={changeCategory}
              onRoute={(letter, routeId) =>
                letter === "A" ? setRouteA(routeId) : setRouteB(routeId)
              }
            />
            {combinedModels && (
              <div className="model-scope-note">
                Separate tool/access models are shown together. Retained sets
                are evaluated within each declared model.
              </div>
            )}
            <RouteComparison
              selected={selected}
              all={routes}
              emptyPrompt={routeComparisonPrompt({
                hasCase: !!payload,
                hasTargetAnnotations: !!caseData?.compartments.length,
                candidateCount: routes.length,
                availableCount: available.length,
                support: supportGate,
                estimatedSupportChosen: allowEstimatedSupport,
                readOnly: readonly,
                engineStopped,
                generating:
                  operation?.op === "generateRoutes" ||
                  operation?.op === "generateNativeRoutes",
              })}
              onFailure={(point) => {
                setCursor(rasPoint(point, payload?.frame ?? "RAS+"));
                setCameraMode("anatomy");
              }}
            />
            {searchSeconds != null && (
              <div className="search-provenance">
                <Check size={12} />
                {combinedModels
                  ? `${routes.length} candidates across separate search models`
                  : `${routes.length} evaluations · ${searchSeconds.toFixed(2)} s · Search`}
              </div>
            )}
          </Tabs.Content>
          <Tabs.Content
            forceMount
            value="refinement"
            className="planning-tab-content refinement-content"
          >
            <RefinementPanel
              api={api}
              caseData={payload}
              route={routes.find((route) => route.route_id === routeA)}
              busy={controlsBlocked}
              onReplay={receiveReplay}
              replayVisible={certifiedReplay !== null}
              onError={reportError}
              routeLabel={
                routes.find((route) => route.route_id === routeA)
                  ? routeName(
                      routes.find((route) => route.route_id === routeA)!,
                      routes,
                    )
                  : "Choose route A"
              }
              canInspect={engineOperations.has("inspectRefinement")}
              canGenerateNative={
                engineOperations.has("generateNativeRoutes") &&
                !supportGate.blocked &&
                routes.length > 0 &&
                !readonly
              }
              onGenerateNative={generateNativeAlternatives}
            />
          </Tabs.Content>
        </Tabs.Root>
        <div className="research-note">
          <FlaskConical size={15} />
          <p>
            <strong>Research use only</strong>
            <span>
              Clinical outcomes are not predicted. Functional and vascular
              anatomy remain unassessed.
            </span>
          </p>
        </div>
      </aside>

      {error && (
        <div className="error-notice" role="alert">
          <Info size={16} />
          <div>
            <strong>Unable to complete the operation</strong>
            <p>{error}</p>
          </div>
          <button
            className="icon-button"
            aria-label="Dismiss error"
            onClick={() => setError(null)}
          >
            <X size={16} />
          </button>
        </div>
      )}
      <footer className="status-bar">
        <span className={`status-dot ${busy ? "working" : ""}`} />
        <span
          className="status-message"
          role="status"
          aria-live="polite"
          aria-atomic="true"
        >
          {engineStopped
            ? "Local engine stopped · Reopen the app to reconnect"
            : hydrating
              ? "Transferring source imaging to the local viewer…"
              : (operation?.message ?? message)}
        </span>
        {operation && (
          <>
            <div className="operation-track">
              <i
                style={{
                  width: `${Math.max(0, Math.min(100, operation.fraction * 100))}%`,
                }}
              />
            </div>
            <button
              className="cancel-operation"
              onClick={() =>
                void act(async () => {
                  await api?.cancel(operation.id);
                })
              }
            >
              Cancel <X size={11} />
            </button>
          </>
        )}
        <span className="status-frame">
          {readonly && <strong>READ-ONLY PREVIEW</strong>}
          {displayedCursor
            ? `RAS  ${displayedCursor.map((value) => value.toFixed(1)).join("  /  ")} mm`
            : "RAS+ · millimeters"}
        </span>
      </footer>
    </div>
  );
}

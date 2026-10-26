import {useSurgical} from "./use-surgical";
import {SurgicalPanel} from "./SurgicalPanel";
import {EnvironmentPreparationPanel} from './EnvironmentPreparationPanel';
import {rehearsalPresentation,initialRehearsalPresentation} from './rehearsal-presentation';
import {operativeFieldFrame} from './full-case-presentation';
import {SurgicalCloseViewControls} from './SurgicalCloseViewControls';
import {hasSavedFullCaseBranch} from './full-case-workspace';
import {useRehearsal} from "./use-rehearsal";
import {RehearsalPanel} from "./RehearsalPanel";
import type {RehearsalApi} from "./rehearsal-types";
import {scanLabel} from './native-study-state';
import {useNativeWorkspace} from './use-native-workspace';
import {NativeSidebar,NativeCorrectionBar,NativeResearch} from './NativeWorkspacePanels';
import {isNativeBinding} from './native-anatomy-data';
import type {NativeAnatomyApi} from './native-anatomy-data';
import {useDiagnosticLayer} from "./use-diagnostic-layer";
import { ImagingWorkspacePanel } from "./ImagingWorkspacePanel";
import { useWorkspaceImaging } from "./use-workspace-imaging";
import { workspaceDisplay, displaySeriesLabel } from "./workspace-imaging-data";
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
import {useEpisodeComparison} from "./use-episode-comparison";
import {requirePersistableEpisode} from "./episode-comparison-data";
import {ContactFamilyPanel} from "./ContactFamilyPanel";
import {hydrateContactFamilyEpisode} from "./contact-family-data";
import type {ContactFamilyView} from "./contact-family-data";
import type {ContactFamilyAvailability,ContactFamilyRequest} from "./contact-family-types";
import {PublicContactPanel} from "./PublicContactPanel";
import {hydratePublicContactEpisode,requirePersistableContact,contactGoalMarker} from "./public-contact-data";
import type {PublicContactView} from "./public-contact-data";
import type {ContactGoalId,ContactSelector} from "./public-contact-types";
import { EpisodePanel } from "./EpisodePanel";
import { hydrateDevelopmentEpisode } from "./episode-data";
import {hydratePersistentPassiveEpisode} from './persistent-passive-view';
import type {PersistentPassiveView} from './persistent-passive-view';
import {PassiveEpisodePanel} from './PassiveEpisodePanel';
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
  CaseLibraryFolder,
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
import { AnatomyPanel } from "./AnatomyPanel";
import { CaseLibraryPanel, WorkbenchDataPanel, anatomyDescription } from "./WorkbenchPanels";
import type { LibraryCase, AnatomyRecord } from "./WorkbenchPanels";
import type { ViewerWorkspaceSettings } from "./viewer/workspaceSettings";
import "./workbench.css";

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
  const native = useNativeWorkspace(api);
  const nativeApi = api as (ResectionApi&NativeAnatomyApi)|null;
  const [newPatient,setNewPatient]=useState('');
  const [newStudyTimepoint,setNewStudyTimepoint]=useState('preoperative');
  const [payload, setPayload] = useState<CasePayload | null>(null);
  const [caseData, setCaseData] = useState<ViewerCase | null>(null);
  const [visibleLayers, setVisibleLayers] = useState<Record<string, boolean>>(
    {},
  );
  const [overlayOpacity, setOverlayOpacity] = useState(0.32);
  const [primaryViewer, setPrimaryViewer] = useState<ViewerWorkspaceSettings>({});
  const [workbenchTab,setWorkbenchTab] = useState<"cases"|"data"|"anatomy"|"rehearsal"|"research">("anatomy");
  const [rehearsalModel,setRehearsalModel]=useState<'standard'|'legacy'>('standard');
  const [libraryCases,setLibraryCases] = useState<LibraryCase[]>([]);
  const [libraryFolder,setLibraryFolder] = useState<CaseLibraryFolder|null>(null);
  const [libraryError,setLibraryError] = useState<string|null>(null);
  const [choosingLibrary,setChoosingLibrary] = useState(false);
  useEffect(()=>{
    let live=true;
    api?.caseLibraryInfo?.().then(info=>{if(live)setLibraryFolder(info);}).catch(error=>{if(live)setLibraryError(error instanceof Error?error.message:String(error));});
    api?.listCases?.().then(rows=>{if(live){setLibraryCases(rows);setLibraryError(null);}}).catch(error=>{if(live){setLibraryCases([]);setLibraryError(error instanceof Error?error.message:String(error));}});
    return ()=>{live=false;};
  },[api,payload,native.binding?.collectionId,native.binding?.document.snapshot_id]);
  const chooseCaseLibrary=async()=>{
    if(!api?.chooseCaseLibrary||choosingLibrary)return;
    setChoosingLibrary(true);
    try {
      const selected=await api.chooseCaseLibrary();
      if(selected){setLibraryFolder(selected);const rows=await api.listCases?.();if(rows)setLibraryCases(rows);setLibraryError(null);}
    } catch(error){setLibraryError(error instanceof Error?error.message:String(error));}
    finally {setChoosingLibrary(false);}
  };
  const [cursor, setCursor] = useState<Vec3 | null>(null);
  const annotationCenterMm = useMemo(
    () => annotationCenter(payload, caseData),
    [payload, caseData],
  );
  const [cameraMode, setCameraMode] = useState<"anatomy" | "instruments">(
    "anatomy",
  );
  const [cameraFitRequest,setCameraFitRequest]=useState(0);
  const [routes, setRoutes] = useState<RouteCandidate[]>([]);
  const [episodeView, setEpisodeView] = useState<EpisodeView | null>(null);
  const [passiveView,setPassiveView] = useState<PersistentPassiveView|null>(null);
  const [passiveStep,setPassiveStep] = useState(0);
  const [episodeWorkflow,setEpisodeWorkflow] = useState<"aspiration"|"public-contact"|"contact-family"|"passive">("aspiration");
  const [publicContactView,setPublicContactView] = useState<PublicContactView|null>(null);
  const [contactStep,setContactStep] = useState(0);
  const [familyView,setFamilyView] = useState<ContactFamilyView|null>(null);
  const [familyStep,setFamilyStep] = useState(0);
  const [contactSelector,setContactSelector] = useState<ContactSelector>("SEARCH");
  const [contactGoal,setContactGoal] = useState<ContactGoalId>("near");
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
  useEffect(()=>{if(native.binding){setEpisodeVisible(false);setCertifiedReplay(null);clearProposal();clearPrior();setRoutes([])}},[native.binding?.collectionId]);
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
  const diagnostic = useDiagnosticLayer(imaging.selected, api);
  const displayedCase = native.binding ? native.volume : display.caseData;
  const displayedSource = imaging.selected?.descriptor.volume ?? payload;
  const displayedCursor = native.binding ? native.state.cursor : imaging.selected ? imaging.cursor : cursor;
  const displayedLayers = native.binding ? native.state.visible : imaging.selected ? imaging.visibleLayers : visibleLayers;
  const setDisplayedLayers = native.binding ? native.setVisible : imaging.selected ? imaging.setVisibleLayers : setVisibleLayers;
  const viewerSettings = native.binding ? native.state.settings : imaging.selected ? imaging.viewer ?? {} : primaryViewer;
  const setViewerSettings = native.binding ? native.setSettings : imaging.selected ? imaging.setViewer : setPrimaryViewer;
  const fusionSeries = !imaging.selected && primaryViewer.fusion?.seriesId ? imaging.series.find(s=>s.descriptor.seriesId===primaryViewer.fusion?.seriesId && s.descriptor.registration.overlayPermitted) : null;
  const registeredFusion = useMemo(()=>fusionSeries && payload && fusionSeries.descriptor.registration.sourceToReferenceRasMm ? {
    seriesId:fusionSeries.descriptor.seriesId,referenceCaseHash:payload.caseHash,
    registration:{status:"source-provided" as const,overlayPermitted:true as const,sourceToReferenceRasMm:fusionSeries.descriptor.registration.sourceToReferenceRasMm},
    volume:fusionSeries.volume,label:fusionSeries.descriptor.modality,
  }:null,[fusionSeries,payload]);
  const surgicalApi = useMemo(()=>api?.surgicalRehearsal ? {
    surgicalRehearsal:(input:import('./surgical-types').SurgicalRequest)=>api.surgicalRehearsal!({...input,...(native.binding?{source:{collectionId:native.binding.collectionId,snapshotId:native.binding.document.snapshot_id,seriesId:native.series}}:{})}),
  }:null,[api,native.binding?.collectionId,native.binding?.document.snapshot_id,native.series]);
  const surgical = useSurgical(displayedCase, surgicalApi, workbenchTab === "rehearsal" && rehearsalModel==='standard' && !native.binding,
    !!(payload?.artifacts?.surgical_toolbox as {archivePreviewAvailable?:boolean}|undefined)?.archivePreviewAvailable);
  const presentedScene=useRef<string|null>(null);
  useEffect(()=>{
    if(surgical.view&&presentedScene.current!==surgical.view.scene.sceneId){
      presentedScene.current=surgical.view.scene.sceneId;
      setPrimaryViewer(settings=>initialRehearsalPresentation(surgical.view!.scene.regions,settings));
    }
  },[surgical.view?.scene.sceneId]);
  const rehearsal = useRehearsal(!native.binding && !imaging.selected ? viewerCase : null,
    api?.createRehearsal && !api.readOnly ? api as RehearsalApi : null);
  useEffect(()=>{
    if (viewerCase && payload?.artifacts?.patient_rehearsal && !native.binding) {
      const saved = (payload.artifacts.workspace as {rehearsalView?:{visible?:boolean}}|undefined)?.rehearsalView;
      if (saved?.visible) setWorkbenchTab('rehearsal');
      void rehearsal.inspect();
    }
  },[viewerCase?.caseHash,payload?.workspaceSession?.sessionHash,payload?.artifacts?.patient_rehearsal,native.binding?.collectionId]);
  const rehearsalUnavailable = native.binding ? "Open a glioma research bundle with source-bound brain and tumor layers."
    : !payload ? "Open a patient case to begin."
    : imaging.selected ? "Select the reference scan before preparing a rehearsal."
    : !payload.brainMask || !Object.values((payload.metadata.anatomy_layers??{}) as Record<string,{kind?:string}>).some(layer=>layer.kind==='tumor')
      ? "This episode requires brain support and a named tumor layer. Missing anatomy remains unassessed." : undefined;
  const busy = surgical.pending || rehearsal.pending || native.busy || !!operation || hydrating || episodePending || imaging.pending || diagnostic.pending;
  const readonly = !!api?.readOnly;
  const controlsBlocked = !!native.binding || busy || engineStopped || !display.planningInteractionPermitted;
  const comparisonAvailable = !readonly && !engineStopped && engineOperations.has("inspectDevelopmentEpisodeComparison");
  const comparison = useEpisodeComparison(episodeView, api, comparisonAvailable, payload?.caseHash ?? null);
  const displayedEpisodeView = comparison.activeView;
  const displayedEpisodeStep = comparison.arm === "search" ? comparison.companionStep : episodeStep;
  const activeContactView = episodeWorkflow === "contact-family" ? familyView : episodeWorkflow === "public-contact" ? publicContactView : null;
  const activeContactStep = episodeWorkflow === "contact-family" ? familyStep : contactStep;
  const publicGoalMarker = useMemo(() => episodeVisible && activeContactView && activeContactView.source.caseHash === payload?.caseHash ? contactGoalMarker(activeContactView,activeContactStep) : null,
    [episodeVisible,activeContactView,activeContactStep,payload?.caseHash]);
  const actorReplayVisible = useRef(false);
  const fileNavigationBlocked = !!native.draft || !api || busy || comparison.pending || engineStopped || readonly;
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
      setHydrating(true);
      setError(null);
      try {
        const reopened = await hydrateWorkspace(sourceInput, runtime,
          () => mounted.current && generation === caseGeneration.current);
        const { source, volume: loaded } = reopened;
        if (!mounted.current || generation !== caseGeneration.current)
          return false;
        // Commit only after the complete replacement has hydrated successfully.
        native.clear();
        imaging.invalidate();
        diagnostic.clear();
        setEpisodeVisible(false);
        activeCaseHash.current = source.caseHash;
        setRehearsalModel('standard');
        await runtime.workspaceInstalled?.({caseHash:source.caseHash,caseId:source.caseId});
        clearPrior();
        clearProposal();
        resetDisplayToPrimary();
        setPayload(source);
        setCaseData(loaded);
        const opensFullCase=!!source.artifacts?.surgical_toolbox&&hasSavedFullCaseBranch(source.artifacts);
        setWorkbenchTab(opensFullCase?'rehearsal':'anatomy');
        if(opensFullCase)setCameraMode('instruments');
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
        setPrimaryViewer(reopened.imagingState?.states.primary.viewer ?? {});
        if (reopened.imagingState) {
          setCursor(reopened.imagingState.states.primary.cursor);
          setVisibleLayers(reopened.imagingState.states.primary.visibleLayers);
        }
        setPublicContactView(null); setContactStep(0); setFamilyView(null); setFamilyStep(0);
        setPassiveView(reopened.passiveView);setPassiveStep(reopened.passiveView?reopened.episodeStep:0);
        setEpisodeWorkflow(reopened.passiveView?"passive":"aspiration");
        setEpisodeView(reopened.episodeView);
        setEpisodeStep(reopened.passiveView?0:reopened.episodeStep);
        setEpisodeVisible(reopened.episodeVisible);
        if (reopened.episodeView && reopened.episodeView.episode.selector !== "RL256_ASPIRATION_MATCHED_SEARCH") setEpisodeSelector(reopened.episodeView.episode.selector);
        if (reopened.episodeVisible) { setCameraMode("instruments"); setPlanningTab("episode"); }
        setMessage(
          reopened.session ? `Workspace reopened · native image views restored${reopened.episodeView||reopened.passiveView ? " · saved replay revalidated" : ""}` :
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
    if (!displayedEpisodeView || displayedEpisodeView.source.caseHash !== activeCaseHash.current) return;
    setEpisodeWorkflow("aspiration");
    clearPrior(); clearProposal(); setCertifiedReplay(null); setEpisodeVisible(false);
    setEpisodeVisible(true); setCameraMode("instruments");
    setMessage("Generated episode · recorded tool poses and modeled cavity · no patient admission");
  };
  const requestEpisodeStep = (step: number) => {
    if (!displayedEpisodeView || !Number.isInteger(step) || step < 0 || step >= displayedEpisodeView.frames.length) return;
    if (comparison.arm === "search") comparison.setStep(step); else setEpisodeStep(step);
    showEpisode();
  };
  const showPassive = () => {
    resetDisplayToPrimary();
    if(!passiveView||passiveView.source.caseHash!==activeCaseHash.current)return;
    setEpisodeWorkflow('passive');clearPrior();clearProposal();setCertifiedReplay(null);
    setEpisodeVisible(true);setCameraMode('instruments');setPlanningTab('episode');
    setMessage('Generated paired passive episode · discrete retained and access poses · no patient admission');
  };
  const requestPassiveStep = (step:number) => {
    if(!passiveView||!Number.isInteger(step)||step<0||step>=passiveView.frames.length)return;
    setPassiveStep(step);showPassive();
  };
  const selectComparisonArm = (arm: "actor" | "search") => {
    if (controlsBlocked || comparison.pending || !comparison.result || arm === comparison.arm) return;
    if (arm === "search") { actorReplayVisible.current = episodeVisible; comparison.select(arm); showEpisode(); }
    else { comparison.select(arm); setEpisodeVisible(actorReplayVisible.current); }
  };
  const showContact = () => {
    if (!publicContactView || publicContactView.source.caseHash !== activeCaseHash.current) return;
    resetDisplayToPrimary(); clearPrior(); clearProposal(); setCertifiedReplay(null);
    setEpisodeVisible(true); setCameraMode("instruments");
  };
  const requestContactStep = (step:number) => {
    if (!publicContactView || !Number.isInteger(step) || step < 0 || step >= publicContactView.frames.length) return;
    setContactStep(step); showContact();
  };
  const showFamily = () => {
    if (!familyView || familyView.source.caseHash !== activeCaseHash.current) return;
    setEpisodeWorkflow("contact-family"); resetDisplayToPrimary(); clearPrior(); clearProposal(); setCertifiedReplay(null);
    setEpisodeVisible(true); setCameraMode("instruments");
  };
  const requestFamilyStep = (step:number) => {
    if (!familyView || !Number.isInteger(step) || step < 0 || step >= familyView.frames.length) return;
    setFamilyStep(step); showFamily();
  };
  const executeFamily = async (request:ContactFamilyRequest,catalog:ContactFamilyAvailability) => {
    if (!api?.executePublicContactFamilyEpisode || controlsBlocked || readonly) return;
    const generation=++episodeGeneration.current,sourceGeneration=caseGeneration.current;
    setEpisodePending(true); setError(null); setEpisodeVisible(false);
    try {
      const result=await api.executePublicContactFamilyEpisode(request);
      const checked=await hydrateContactFamilyEpisode(result,api,request,catalog);
      if (!mounted.current || generation!==episodeGeneration.current || sourceGeneration!==caseGeneration.current) return;
      if (!await installCase(checked.source,api)) return;
      setFamilyView(checked); setFamilyStep(0); setEpisodeWorkflow("contact-family"); setEpisodeVisible(true);
      setCursor([...checked.episode.publicGoal.rasMm]); setCameraMode("instruments"); setPlanningTab("episode");
      setMessage("Generated family contact · source, public objective and native replay checked · temporary, not saved");
    } catch(failure) {if(mounted.current && generation===episodeGeneration.current)reportError(failure);}
    finally {if(mounted.current && generation===episodeGeneration.current)setEpisodePending(false);}
  };
  const executeContact = async () => {
    if (!api?.executePublicSurfaceContactEpisode || controlsBlocked || readonly) return;
    const generation=++episodeGeneration.current,sourceGeneration=caseGeneration.current;
    setEpisodePending(true); setError(null); setEpisodeVisible(false);
    try {
      const result=await api.executePublicSurfaceContactEpisode({fixture:"generated-public-surface-contact-v1",goalId:contactGoal,selector:contactSelector});
      const checked=await hydratePublicContactEpisode(result,api);
      if (!mounted.current || generation!==episodeGeneration.current || sourceGeneration!==caseGeneration.current) return;
      if (!await installCase(checked.source,api)) return;
      setPublicContactView(checked); setContactStep(0); setEpisodeWorkflow("public-contact"); setEpisodeVisible(true);
      setCursor([...checked.episode.publicGoal.rasMm]); setCameraMode("instruments"); setPlanningTab("episode");
      setMessage("Generated public-contact task · exact replay and public objective checked · temporary, not saved");
    } catch(failure) { if(mounted.current && generation===episodeGeneration.current)reportError(failure); }
    finally { if(mounted.current && generation===episodeGeneration.current)setEpisodePending(false); }
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
      setEpisodeWorkflow("aspiration"); setEpisodeView(checked); setEpisodeStep(0); setEpisodeVisible(true);
      setCameraMode("instruments"); setPlanningTab("episode");
      setMessage("Generated software episode executed · all recorded frame masks checked · no patient admission");
    } catch (failure) { if (mounted.current && generation === episodeGeneration.current) reportError(failure); }
    finally { if (mounted.current && generation === episodeGeneration.current) setEpisodePending(false); }
  };
  const executePassive = async () => {
    if(!api?.executePersistentPassiveEpisode||controlsBlocked||readonly)return;
    const generation=++episodeGeneration.current;
    setEpisodePending(true);setError(null);
    try{
      const result=await api.executePersistentPassiveEpisode({fixture:'generated-persistent-passive-v1',paired:true});
      const checked=await hydratePersistentPassiveEpisode(result,api);
      if(!mounted.current||generation!==episodeGeneration.current)return;
      if(!await installCase(checked.source,api))return;
      setPassiveView(checked);setPassiveStep(0);setEpisodeWorkflow('passive');setEpisodeVisible(true);
      setCameraMode('instruments');setPlanningTab('episode');
      setMessage('Generated paired passive episode · six backend states checked · no patient admission');
    }catch(failure){if(mounted.current&&generation===episodeGeneration.current)reportError(failure);}
    finally{if(mounted.current&&generation===episodeGeneration.current)setEpisodePending(false);}
  };

  useEffect(() => {
    mounted.current = true;
    let disposed = false;
    if (window.resectionApi) {
      const runtime = window.resectionApi;
      const startup=(runtime as ResectionApi&NativeAnatomyApi).startupWorkspace;
      (startup?startup():runtime.startupCase())
        .then(async (source) => {
          if(source&&!disposed){if(isNativeBinding(source)){setPayload(null);setCaseData(null);return native.install(source)}await installCase(source,runtime);}
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
  }, [installCase, installSearch,native.install]);

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
      const source = kind==="openCase"&&nativeApi?.openWorkspace ? await nativeApi.openWorkspace() : await api[kind]();
      if (source && mounted.current && generation === caseGeneration.current) {
        if(isNativeBinding(source)){setPayload(null);setCaseData(null);setEpisodeVisible(false);setCertifiedReplay(null);clearProposal();clearPrior();await native.install(source);await api.workspaceInstalled?.({caseHash:source.collectionId,caseId:source.document.patient_id});setWorkbenchTab('anatomy');api.listCases?.().then(setLibraryCases);return;}
        const installed = await installCase(source, api, generation);
        if (installed && kind === "importNifti")
          setMessage("Source image imported · Original file preserved · Save to keep this workspace");
      }
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
      requirePersistableContact(publicContactView);
      requirePersistableContact(familyView);
      requirePersistableEpisode(episodeWorkflow==='aspiration'?displayedEpisodeView:null);
      rehearsal.pause();
      surgical.pauseFullCase();
      surgical.pausePlayback();
      if(surgical.view&&!await surgical.run('inspect'))throw new Error('The recorded surgical state could not be synchronized before saving.');
      const result = await api.saveCase({
        caseHash: payload.caseHash,
        workspace: {
          routeA,
          routeB,
          category,
          visibleLayers,
          overlayOpacity,
          cursor,
          imaging: imaging.snapshot({cursor, visibleLayers, viewer:primaryViewer}),
          ...(rehearsal.viewPreferences ? {rehearsalView:rehearsal.viewPreferences} : {}),
          episodeReplay: episodeWorkflow==='passive'&&passiveView?.source.caseHash===payload.caseHash ? {
            episodeId:passiveView.episode.episodeId,frameIndex:passiveStep,
            visible:episodeVisible&&episodeWorkflow==='passive'&&!imaging.selected,
          } : episodeWorkflow==='aspiration'&&episodeView?.source.caseHash === payload.caseHash ? {
            episodeId: episodeView.episode.episodeId, frameIndex: episodeStep,
            visible: episodeVisible && !imaging.selected,
          } : null,
        },
      });
      if (result?.saved) {
        setMessage("Workspace saved locally · images, layers and views preserved");
        api.listCases?.().then(setLibraryCases).catch(()=>{});
      }
    });

  useEffect(() => {
    if (!api) return;
    return api.onEvent((event) => {
      if (event.event !== "menuAction" || busy || comparison.pending || engineStopped || readonly) return;
      if (event.action === "openCase") void load("openCase");
      if (event.action === "saveCase" && (payload||native.binding)) void (native.binding?native.save():save());
    });
  }, [
    api,
    native.binding,native.series,native.state,native.draft,
    busy,
    engineStopped,
    readonly,
    payload,
    imaging.snapshot,
    imaging.selected,
    episodeView,
    passiveView,
    displayedEpisodeView,
    publicContactView,
    familyView,
    comparison.pending,
    episodeStep,
    passiveStep,
    episodeWorkflow,
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
      episodeVisible && activeContactView && activeContactView.source.caseHash === caseData?.caseHash
        ? activeContactView.frames[activeContactStep]
        : episodeVisible && episodeWorkflow === "aspiration" && displayedEpisodeView && displayedEpisodeView.source.caseHash === caseData?.caseHash
        ? displayedEpisodeView.frames[displayedEpisodeStep]
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
    [certifiedReplay, caseData, episodeVisible, displayedEpisodeView, displayedEpisodeStep,episodeWorkflow,activeContactView,activeContactStep],
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
  const annotationDescription = synthetic ? "Generated engineering geometry" :
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
    <div className={`app-shell poc-workbench ${native.binding?"native-workspace":""} ${workbenchTab === "research" && !native.binding ? "research-open" : "anatomy-only"} ${casePanel ? "" : "case-panel-collapsed"}`}>
      <header className="app-header">
        <div className="window-traffic-space" />
        <div className="brand">
          <Logo />
          <span>
            Ressection<span className="brand-light">Lab</span>
          </span>
        </div>
        <div className="header-divider" />
        <span className="header-purpose">Studio</span>

        <div className="header-spacer" />
        <span className="research-tag">
          <span className="status-dot" /> Local research
        </span>
        <button
          className="header-button"
          onClick={() => load("openCase")}
          disabled={fileNavigationBlocked}
        >
          <FolderOpen size={15} />
          <span>Open</span>
          <kbd>⌘O</kbd>
        </button>
        <button
          className="header-button"
          onClick={() => native.binding ? setWorkbenchTab("data") : load("importNifti")}
          disabled={fileNavigationBlocked}
        >
          <ArrowUpFromLine size={15} />
          <span>Add data</span>
        </button>
        <button
          className="header-button save-button"
          onClick={native.binding?()=>native.save():save}
          disabled={(!payload&&!native.binding) || busy || !!native.draft || engineStopped || readonly}
        >
          <ArrowDownToLine size={15} />
          <span>Save</span>
        </button>
      </header>

      <nav className="navigation-rail" aria-label="Workspace navigation">
        {([{id:"cases",label:"Cases",icon:FolderOpen},{id:"data",label:"Data",icon:Scan},{id:"anatomy",label:"Anatomy",icon:Layers3},{id:"rehearsal",label:"Rehearse",icon:Target}] as const).map(item=><button key={item.id}
          className={`rail-button ${workbenchTab===item.id?"selected":""}`} aria-label={item.id==='anatomy'?'3D workspace':item.label}
          aria-current={workbenchTab===item.id?'page':undefined} title={item.id==='anatomy'?'3D workspace':item.label}
          onClick={()=>{if(item.id==='rehearsal'){resetDisplayToPrimary();setEpisodeVisible(false);setCertifiedReplay(null);setCameraMode('instruments')}else rehearsal.hide();setWorkbenchTab(item.id);setCasePanel(true)}}><item.icon size={19}/><span>{item.label}</span></button>)}
        <div className="rail-spacer" />
        <button className={`rail-button ${workbenchTab==='research'?'selected':''}`} aria-label="Advanced research workflows"
          title="Earlier experimental workflows" onClick={()=>{setWorkbenchTab('research');setCasePanel(true)}}><FlaskConical size={18}/><span>Research</span></button>
        <span className="rail-local" title="Processing stays on this Mac"><ShieldCheck size={17}/></span>
      </nav>

      <aside className="case-panel" id="case-evidence-panel">
        {workbenchTab==='rehearsal'?<>
          <div className="panel-heading"><span className="eyebrow">SURGICAL RESEARCH</span><button className="icon-button" aria-label="Collapse case panel" onClick={()=>setCasePanel(false)}><PanelLeftClose size={15}/></button></div>
          {native.binding ? <EnvironmentPreparationPanel key={native.binding.collectionId} binding={native.binding} busy={busy} onAnatomy={()=>setWorkbenchTab('data')} onPrepare={async options=>{
            if(!api?.prepareSurgicalEnvironment)throw Error('Restart the app to load environment preparation.');
            const prepared=await api.prepareSurgicalEnvironment(options);
            if(prepared&&await installCase(prepared,api))setWorkbenchTab('rehearsal');
          }}/> : <>
            <label className="rehearsal-model">Rehearsal model<select aria-label="Rehearsal model" value={rehearsalModel} onChange={e=>{rehearsal.pause();surgical.cancel();setRehearsalModel(e.target.value as typeof rehearsalModel)}}><option value="standard">Standard instrument environment</option><option value="legacy">Legacy bounded episode</option></select></label>
            {rehearsalModel==='standard'?<>
              <SurgicalPanel c={surgical} learningReport={payload?.artifacts?.patient_rescue_learning} executionReport={payload?.artifacts?.patient_rescue_ppo_bc??payload?.artifacts?.patient_rescue_bc} cusaLearning={payload?.artifacts?.surgical_toolbox} microscopeControls={<SurgicalCloseViewControls c={surgical} source={displayedCase} settings={viewerSettings} cursor={displayedCursor} onSettings={setViewerSettings} onCursor={point=>{if(imaging.selected&&point)imaging.setCursor(point);else setCursor(point);}} onFitAnatomy={()=>{setCameraMode('anatomy');setCameraFitRequest(request=>request+1);}}/>} fieldDisplayControls={<label><input type="checkbox" checked={viewerSettings.smoothSurfaces!==false} onChange={e=>setViewerSettings({...viewerSettings,smoothSurfaces:e.target.checked})}/>Smooth 3D surfaces</label>} onFrameField={()=>{if(surgical.view){const field=operativeFieldFrame(surgical.view,imaging.selected?null:annotationCenterMm,viewerSettings,displayedCase);if(field){if(imaging.selected)imaging.setCursor(field.center);else setCursor(field.center);setViewerSettings(field.settings);setMessage(field.anchor==='target'?'Framed supplied annotation.':'Framed accepted instrument in source scan.');}else setMessage('No accepted instrument or supplied annotation lies within this source scan.');}}} anatomyControls={!!surgical.view&&<details className="rehearsal-anatomy"><summary>Anatomy visibility and appearance</summary><button onClick={()=>setViewerSettings(rehearsalPresentation(surgical.view!.scene.regions,viewerSettings))}>Reference appearance</button>
                <AnatomyPanel layers={surgical.view.scene.regions.map(r=>({name:r.name,kind:r.material,color:r.color,provenance:`${r.material} · ${r.target?'declared target':'context anatomy'}`}))} visibleLayers={displayedLayers} onVisibilityChange={setDisplayedLayers} settings={viewerSettings} onSettingsChange={setViewerSettings}/>
              </details>} onImport={async options=>{if(api?.importSurgicalScene){const imported=await api.importSurgicalScene(options);if(imported&&await installCase(imported,api))setWorkbenchTab('rehearsal')}}} available={!!displayedCase} onFixture={()=>void act(async()=>{if(api?.createSurgicalFixture){const fixture=await api.createSurgicalFixture();if(fixture&&await installCase(fixture,api))setWorkbenchTab('rehearsal')}})}/>

            </>:<RehearsalPanel controller={rehearsal} unavailableReason={rehearsalUnavailable}/>}
          </>}
        </>:native.binding&&workbenchTab!=='cases'?<>
          <div className="panel-heading"><span className="eyebrow">PATIENT WORKSPACE</span><button className="icon-button" aria-label="Collapse case panel" onClick={()=>setCasePanel(false)}><PanelLeftClose size={15}/></button></div>
          <div className="workspace-tabs"><button className={workbenchTab==='data'?'selected':''} onClick={()=>setWorkbenchTab('data')}>Scans</button><button className={workbenchTab==='anatomy'?'selected':''} onClick={()=>setWorkbenchTab('anatomy')}>Anatomy</button></div>
          {workbenchTab==='research'?<NativeResearch workspace={native}/>:<NativeSidebar workspace={native} tab={workbenchTab==='data'?'data':'anatomy'}/>}
        </>:<>
        {workbenchTab!=="research" && <>
          <div className="panel-heading"><span className="eyebrow">{workbenchTab==='cases'?'PATIENT LIBRARY':workbenchTab==='data'?'DATA & REGISTRATION':'ANATOMY LAYERS'}</span>
            <button className="icon-button" aria-label="Collapse case panel" onClick={()=>setCasePanel(false)}><PanelLeftClose size={15}/></button></div>
          {workbenchTab==='cases' ? <><div className="unified-sidebar"><details className="unified-details"><summary>New patient study</summary><p className="subtle">One patient, one timepoint. Add multiple scans to build the workspace.</p><input aria-label="New patient ID" placeholder="Patient ID" value={newPatient} onChange={e=>setNewPatient(e.target.value)}/><label>Study<select aria-label="New study timepoint" value={newStudyTimepoint} onChange={e=>setNewStudyTimepoint(e.target.value)}><option value="preoperative">Pre-op</option><option value="postoperative">Post-op</option></select></label><button disabled={fileNavigationBlocked||!newPatient.trim()} onClick={async()=>{const b=await native.create(newPatient.trim(),newStudyTimepoint);if(b){setPayload(null);setCaseData(null);setWorkbenchTab('data')}}}>Create study…</button></details></div><CaseLibraryPanel cases={libraryCases} busy={fileNavigationBlocked||choosingLibrary} onBrowse={()=>load('openCase')}
            libraryFolder={libraryFolder} libraryError={libraryError} onChooseLibrary={api?.chooseCaseLibrary?()=>void chooseCaseLibrary():undefined}
            onOpen={id=>void act(async()=>{if(api?.openListedCase){const next=nativeApi?.openListedWorkspace?await nativeApi.openListedWorkspace(id):await api.openListedCase(id);if(isNativeBinding(next)){setPayload(null);setCaseData(null);await native.install(next);await api.workspaceInstalled?.({caseHash:next.collectionId,caseId:next.document.patient_id});setWorkbenchTab('anatomy')}else if(next){await installCase(next,api)}else return;}})}/></> : <>
            <div className="workspace-tabs"><button className={workbenchTab==='data'?'selected':''} onClick={()=>setWorkbenchTab('data')}>Scans</button><button className={workbenchTab==='anatomy'?'selected':''} onClick={()=>setWorkbenchTab('anatomy')}>Anatomy</button></div><h1 title={title}>{title}</h1><p className="case-subtitle">{displayedSource ? String(displayedSource.metadata.selected_modality??'Source image') : 'Open a case to begin'}</p>
            {workbenchTab==='data' && payload && displayedSource && <WorkbenchDataPanel source={payload} displayed={displayedSource} series={imaging.series} selectedId={imaging.selectedId}
              busy={busy||engineStopped} api={api} onSelect={imaging.select} onImport={request=>void act(()=>imaging.importSeries(request))}
              settings={primaryViewer} onSettingsChange={setPrimaryViewer}/>}
            {workbenchTab==='anatomy' && <>
              {payload && <label className="poc-scan-selector">Source scan<select aria-label="Source scan" value={imaging.selectedId??''} disabled={busy}
                onChange={e=>imaging.select(e.target.value||null)}><option value="">{String(payload.metadata.selected_modality??'Reference image')} · reference</option>
                {imaging.series.map(s=><option key={s.descriptor.seriesId} value={s.descriptor.seriesId}>{displaySeriesLabel(s.descriptor)}</option>)}</select></label>}
              <AnatomyPanel layers={(displayedCase?.compartments??[]).map(layer=>({...layer,label:(displayedSource?.metadata.anatomy_layers as Record<string,{label?:string}>|undefined)?.[layer.name]?.label,provenance:anatomyDescription((displayedSource?.metadata.anatomy_layers as Record<string,AnatomyRecord>|undefined)?.[layer.name],imaging.selected?.descriptor.annotationKind==='estimated'?'Model estimate':annotationDescription)}))}
                visibleLayers={displayedLayers} onVisibilityChange={setDisplayedLayers} settings={viewerSettings} onSettingsChange={setViewerSettings}
                missingStructures={Array.isArray(displayedSource?.metadata.missing_anatomy)?displayedSource.metadata.missing_anatomy as string[]:['Skull','Vessels','Brain structures'].filter(name=>!displayedCase?.compartments.some(layer=>{
                  const kind=(displayedSource?.metadata.anatomy_layers as Record<string,{kind?:string}>|undefined)?.[layer.name]?.kind?.toLowerCase();
                  const kinds=name==='Skull'?['skull','bone']:name==='Vessels'?['vessel','vessels']:['brain','soft_tissue'];
                  return (kind!==undefined&&kinds.includes(kind))||layer.name.toLowerCase().includes(name==='Brain structures'?'brain':name.toLowerCase());
                }))}
                provenanceLabel="Cached masks · no inference on view changes"/>
              <AnnotationCenterButton center={imaging.selected?null:annotationCenterMm} onCenter={setCursor}/>
              <button className="text-button" onClick={()=>setWorkbenchTab('data')}>Source files & registration <ChevronRight size={13}/></button>
              <p className="muted-note poc-research-note">Research estimates. Missing structures are unassessed.</p>
            </>}
          </>}
        </>}
        {workbenchTab==="research" && <>

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
        {imaging.selected && <section className="control-section" aria-label="Saved model diagnostic">
          <h2>Saved diagnostic</h2>
          <p className="muted-note">Select the prepared atlas T1c, then load its saved diagnostic. This layer is unreviewed, temporary and excluded from planning.</p>
          <button className="outline-button" disabled={busy || engineStopped || readonly || !api?.importDiagnosticLayer || !engineOperations.has("importDiagnosticLayer")}
            onClick={()=>void diagnostic.load()}>{diagnostic.pending ? "Loading diagnostic…" : "Load diagnostic"}</button>
          {diagnostic.view && <><p className="muted-note">Coral contour: model-positive candidate. Neutral hatching: unknown output coverage. Covered negatives leave the MRI unchanged.</p>
            <button className="text-button" onClick={diagnostic.clear}>Remove diagnostic layer</button></>}
          {diagnostic.error && <p role="alert" className="muted-note">{diagnostic.error}</p>}
        </section>}
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
            {imaging.selected ? (imaging.selected.descriptor.annotationKind === "estimated" ? "Estimated labels · display only" : "Source labels · display only") : (publicContactView || familyView) ? "Fixture background annotations" : "Target annotations"}{" "}
            <span>
              {displayedCase?.compartments.length.toString().padStart(2, "0") ?? "—"}
            </span>
          </h2>
          {(publicContactView || familyView) && !imaging.selected && <p className="muted-note">These fixture annotations are unused by the contact reward. The public goal has its own marker.</p>}
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
                  setDisplayedLayers({
                    ...displayedLayers,
                    [layer.name]: event.target.checked,
                  })
                }
              />
              <span
                className="layer-swatch"
                style={{ background: layer.color }}
              />
              <div>
                <strong>{readableName(layer.name)}</strong>
                <span>
                  {((layer.volumeMm3??0) / 1000).toFixed(2)} mL ·{" "}
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
              {diagnostic.view ? "Diagnostic opacity" : "Overlay opacity"} <span>{Math.round(overlayOpacity * 100)}%</span>
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
        </>}
        </>}
      </aside>

      <main className="anatomy-panel">
        <div className="workspace-heading">
          <div>
            <WorkspaceBreadcrumb caseId={native.binding?.document.patient_id??caseData?.caseId ?? null} />
            <h2>{workbenchTab==='rehearsal'?'Surgical research rehearsal':native.binding?scanLabel(native.binding.document.series.find(s=>s.id===native.series)??{id:'',modality:'Study'}):workbenchTab==='data'?'Source imaging':'Anatomy workspace'}</h2>
          </div>
          <div
            className={`view-presets ${workbenchTab==='research'?'':'poc-hidden'}`}
            role="group"
            aria-label="Camera framing"
          >
            <button
              className={cameraMode === "anatomy" ? "active" : ""}
              onClick={() => {setCameraMode("anatomy");setCameraFitRequest(value=>value+1);}}
              aria-pressed={cameraMode === "anatomy"}
            >
              <Focus size={14} /> Focus anatomy
            </button>
            <button
              className={cameraMode === "instruments" ? "active" : ""}
              onClick={() => {setCameraMode("instruments");setCameraFitRequest(value=>value+1);}}
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
        {episodeVisible && episodeWorkflow === "aspiration" && displayedEpisodeView && (
          <div className="modeled-replay-notice" role="status"><span>GENERATED · modeled cavity · {comparison.arm === "search" ? "matched search" : "selected episode"} · frame {displayedEpisodeStep}/{displayedEpisodeView.frames.length-1} · {displayedEpisodeView.episode.replayFrames[displayedEpisodeStep].phase} · no patient admission</span><button className="text-button" onClick={restoreSourceView}>Source view <X size={12}/></button></div>
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
        {episodeVisible && activeContactView && <div className="modeled-replay-notice" role="status"><span>GENERATED · public retained-contact · frame {activeContactStep}/{activeContactView.frames.length-1} · no patient admission</span><button className="text-button" onClick={restoreSourceView}>Source view <X size={12}/></button></div>}
        {episodeVisible&&episodeWorkflow==='passive'&&passiveView&&caseData&&passiveView.source.caseHash===caseData.caseHash&&
          <div className="modeled-replay-notice" role="status"><span>GENERATED · two persistent tool poses · state {passiveStep}/5 · declared {passiveView.frames[passiveStep].elapsedGeneratedSeconds.toFixed(1)} s · no patient admission</span>
            <button className="text-button" onClick={restoreSourceView}>Source view <X size={12}/></button></div>}
        {native.binding&&native.registration==='alignment preview'&&<div className="workspace-alignment-banner" role="status">Estimated scan alignment <span>{Object.values(native.volume?.alignmentQuality??{}).flatMap(q=>q.warnings)[0]??'Shared structures · verify in Scans before planning'}</span></div>}
        <NativeCorrectionBar workspace={native}/>
        <div className="viewer-shell">
          {displayedCase && (
            <ViewerWorkspace
              surgicalInteraction={workbenchTab==="rehearsal"?surgical.interaction:undefined}
              rehearsalState={surgical.interaction.view?null:rehearsal.viewerState}
              workspaceSettings={viewerSettings}
              onWorkspaceSettingsChange={setViewerSettings}
              sourceLabel={native.binding?native.binding.document.series.find(s=>s.id===native.series)?.modality:String(displayedSource?.metadata.selected_modality??'Image')}
              onSliceStroke={native.binding?native.onStroke:undefined}
              registeredFusion={native.binding?null:registeredFusion}
              diagnosticLayer={native.binding?null:diagnostic.view?.layer ?? null}
              diagnosticSourceSha256={diagnostic.view?.sourceSha256 ?? null}
              generatedSignal={display.primaryOverlaysPermitted && synthetic}
              segmentationOnly={!native.binding && displayedSource?.metadata.generated_signal===true}
              publicGoal={!native.binding && display.primaryOverlaysPermitted ? publicGoalMarker : null}
              caseData={surgical.previewVolume??surgical.volume??displayedCase}
              visibleLayers={displayedLayers}
              overlayOpacity={overlayOpacity}
              cursor={displayedCursor}
              onCursorChange={native.binding ? native.setCursor : imaging.selected ? imaging.setCursor : setCursor}
              routes={!native.binding && display.primaryOverlaysPermitted ? viewerRoutes : []}
              cameraMode={cameraMode}
              cameraFitRequest={cameraFitRequest}
              replay={!native.binding && display.primaryOverlaysPermitted ? viewerReplay : null}
              passiveReplay={display.primaryOverlaysPermitted&&episodeVisible&&episodeWorkflow==='passive'&&passiveView&&caseData&&passiveView.source.caseHash===caseData.caseHash?passiveView.frames[passiveStep]:null}
              structuralProposal={!native.binding && display.primaryOverlaysPermitted ? proposalView : null}
              priorLayer={!native.binding && display.primaryOverlaysPermitted ? priorView : null}
            />
          )}
          {!caseData && !native.binding && (
            <div className="welcome-overlay">
              <div className="welcome-symbol">
                <Scan size={34} strokeWidth={1} />
              </div>
              <span className="eyebrow">RESSECTIONLAB STUDIO</span>
              <h2>
                From source imaging
                <br />
                to inspectable anatomy.
              </h2>
              <p>
                Explore your imaging in 3D, inspect linked slices, and work with your anatomical layers.
              </p>
              <button
                className="primary-button"
                disabled={fileNavigationBlocked}
                onClick={() => load("openCase")}
              >
                <FolderOpen size={16} /> Open a case
              </button>
              <div className="welcome-workflow" aria-label="Workspace workflow">
                <span><Scan size={15}/> Source imaging</span>
                <ChevronRight size={12}/>
                <span><Layers3 size={15}/> Anatomy</span>
                <ChevronRight size={12}/>
                <span><Target size={15}/> Rehearsal</span>
              </div>
            </div>
          )}
          {native.binding&&!displayedCase&&!native.busy&&<div className="welcome-overlay"><h2>{native.binding.document.series.length?'Scan unavailable':'Add your first scan'}</h2><p>{native.message}</p>{native.binding.document.series.length?<button className="primary-button" onClick={()=>native.run(native.refresh)}>Retry</button>:<button className="primary-button" onClick={()=>setWorkbenchTab('data')}>Add data</button>}</div>}
          {(hydrating||native.busy) && (
            <div className="viewer-loading">
              <LoaderCircle size={22} className="spin" />
              <span>{native.binding?native.message:"Preparing source images…"}</span>{native.busy&&<button className="header-button" onClick={native.cancel}>Cancel</button>}
            </div>
          )}
        </div>
        <div className="workspace-caption">
          <span>
            <span className="mini-crosshair" /> Linked physical coordinates ·
            RAS+ mm
          </span>
          <span>
            {workbenchTab!=="research" ? "Local imaging · cached anatomy" : cameraMode === "anatomy" ? "Anatomy focus may crop instruments" : "Complete rendered instrument envelopes"}
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
            <label className="field-label" htmlFor="episode-workflow">Generated task</label><select id="episode-workflow" value={episodeWorkflow} disabled={controlsBlocked||comparison.pending} onChange={event=>{setEpisodeWorkflow(event.target.value as "aspiration"|"public-contact"|"contact-family"|"passive");setEpisodeVisible(false);}}><option value="aspiration">Aspiration & trained-transfer comparison</option><option value="public-contact">Public retained-contact goal</option><option value="contact-family">Generated family · learned / search</option><option value="passive">Persistent passive pair · generated</option></select>
            {episodeWorkflow==='passive'?<PassiveEpisodePanel view={passiveView?.source.caseHash===payload?.caseHash?passiveView:null} step={passiveStep} busy={controlsBlocked||comparison.pending} visible={episodeVisible} onExecute={executePassive} onStep={requestPassiveStep} onShow={showPassive} onSource={restoreSourceView} unavailableReason={engineStopped?'The local engine stopped.':readonly?'This browser preview is read only.':!api?.executePersistentPassiveEpisode||!engineOperations.has('executePersistentPassiveEpisode')?'This engine does not provide generated passive episodes.':undefined}/>
            :episodeWorkflow === "contact-family" ? <ContactFamilyPanel api={api} view={familyView?.source.caseHash===payload?.caseHash?familyView:null} step={familyStep} busy={controlsBlocked||comparison.pending} visible={episodeVisible} onExecute={executeFamily} onStep={requestFamilyStep} onShow={showFamily} onSource={restoreSourceView} onFocusGoal={()=>{if(familyView){setCursor([...familyView.episode.publicGoal.rasMm]);showFamily();}}} unavailableReason={engineStopped?"The local engine stopped.":readonly?"This browser preview is read only.":!api?.executePublicContactFamilyEpisode||!engineOperations.has("executePublicContactFamilyEpisode")?"This engine does not provide the contact family.":undefined}/> : episodeWorkflow === "public-contact" ? <PublicContactPanel view={publicContactView?.source.caseHash===payload?.caseHash?publicContactView:null} step={contactStep} selector={contactSelector} goalId={contactGoal} busy={controlsBlocked||comparison.pending} visible={episodeVisible} onSelector={setContactSelector} onGoal={setContactGoal} onExecute={executeContact} onStep={requestContactStep} onShow={showContact} onSource={restoreSourceView} onFocusGoal={()=>{if(publicContactView){setCursor([...publicContactView.episode.publicGoal.rasMm]);showContact();}}} unavailableReason={engineStopped?"The local engine stopped.":readonly?"This browser preview is read only.":!api?.executePublicSurfaceContactEpisode||!engineOperations.has("executePublicSurfaceContactEpisode")?"This engine does not provide the public-contact task.":undefined}/> : <>
            <EpisodePanel vascularApi={!readonly && !engineStopped && engineOperations.has("evaluateDevelopmentEpisodeVascular") ? api : null} view={displayedEpisodeView?.source.caseHash === payload?.caseHash ? displayedEpisodeView : null} step={displayedEpisodeStep}
              comparison={{actor:episodeView, result:comparison.result, arm:comparison.arm, available:comparisonAvailable, pending:comparison.pending, error:comparison.error, inspect:comparison.inspect, select:selectComparisonArm}}
              selector={episodeSelector} onSelector={setEpisodeSelector} busy={controlsBlocked || comparison.pending}
              onExecute={executeEpisode} onStep={requestEpisodeStep} onShow={showEpisode} onSource={restoreSourceView} visible={episodeVisible}
              unavailableReason={engineStopped ? "The local engine stopped. Reopen the app to execute." : readonly ? "This browser preview is read only. Open the desktop app to execute." : !api?.executeDevelopmentEpisode || !engineOperations.has("executeDevelopmentEpisode") ? "This engine does not provide generated development episodes." : undefined}/>
            </>}
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
              : (operation?.message ?? (native.binding?native.message:message))}
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

import {surgicalEffectMarkers,clipSurgicalMarkersToSlab,surgicalMarkerRadiusPx} from './surgicalGeometry';
import {modeledSourceIdentity,modeledAnatomy} from './modeledAnatomy';
import type {SurgicalInteraction} from '../surgical-types';
import {SliceScrollAccumulator} from './interactionBudget';
import {TransferEditor} from './TransferEditor';
import {sampleDiagnosticLayer} from "./diagnosticLayer";
import type {CheckedDiagnosticLayer} from "./diagnosticLayer";
import {checkedPublicGoal,PUBLIC_GOAL_COLOR} from './publicGoal';
import {mprView,mprPoint,mprPixel,sliceFraming,sourceSignalBounds} from './mprFraming';
import { useEffect, useMemo, useRef, useState } from "react";
import {
  Columns2,
  SlidersHorizontal,
  ScanLine,
  Focus,
  Layers3,
  Box,
  Maximize2,
  Minimize2,
} from "lucide-react";
import type { KeyboardEvent, PointerEvent } from "react";
import { VolumeRenderer } from "./VolumeRenderer";
import { clippingForBounds } from './workspaceSettings';
import type { ViewerWorkspaceSettings } from './workspaceSettings';
import { volumeSettings, isCtModality } from './volumeTransfer';
import type { VolumePreset } from './volumeTransfer';
import {
  PLANE_AXES,
  PLANE_LABELS,
  clampCursor,
  inverseAffine,
  rasAffine,
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
    routes: requestedRoutes,
    cameraMode,
    replay: requestedReplay,
    passiveReplay: requestedPassiveReplay,
    rehearsalState: requestedRehearsal,
    structuralProposal: requestedProposal,
    priorLayer: requestedPrior,
    inspectionTool,
    diagnosticLayer,
    diagnosticSourceSha256,
  } = props;
  const rendererIdentity=caseData?modeledSourceIdentity(caseData):caseData;
  const geometrySource=caseData?(modeledAnatomy(caseData)?.source??caseData):null;
  // This independent image layer cannot share planning/replay overlays.
  const diagnosticRequested = Boolean(diagnosticLayer);
  const rehearsalState = diagnosticRequested ? null : requestedRehearsal;
  const rehearsalRequested = Boolean(rehearsalState);
  const passiveReplay = diagnosticRequested || rehearsalRequested ? null : requestedPassiveReplay;
  const passiveRequested = Boolean(passiveReplay);
  const routes = diagnosticRequested || passiveRequested || rehearsalRequested ? [] : requestedRoutes;
  const replay = diagnosticRequested || passiveRequested || rehearsalRequested ? null : requestedReplay;
  const structuralProposal = diagnosticRequested || passiveRequested || rehearsalRequested ? null : requestedProposal;
  const priorLayer = diagnosticRequested || passiveRequested || rehearsalRequested ? null : requestedPrior;
  const [diagnosticVisible, setDiagnosticVisible] = useState(true);
  const [displayedDiagnostic, setDisplayedDiagnostic] = useState<CheckedDiagnosticLayer|null>(null);
  const [diagnosticError, setDiagnosticError] = useState<string|null>(null);
  const goalSelection = useMemo(() => {
    try { return {goal:caseData ? checkedPublicGoal(caseData,diagnosticRequested || passiveRequested || rehearsalRequested ? null : (props.publicGoal ?? null)):null,error:null}; }
    catch (cause) { return {goal:null,error:cause instanceof Error?cause.message:String(cause)}; }
  },[caseData,props.publicGoal,diagnosticRequested,passiveRequested,rehearsalRequested]);
  const publicGoal = goalSelection.goal;
  const signalName = props.segmentationOnly ? 'segmentation only' : props.generatedSignal ? "analytic signal" : (props.sourceLabel ?? "MRI");
  const [localSettings,setLocalSettings] = useState<ViewerWorkspaceSettings>({});
  const settings=props.workspaceSettings ?? localSettings;
  const scanVolume=volumeSettings(settings.volumeRendering,signalName);
  const latestSettings=useRef(settings);latestSettings.current=settings;
  const changeSettings=(next:ViewerWorkspaceSettings)=>{
    latestSettings.current=next;
    if(latestProps.current.onWorkspaceSettingsChange)latestProps.current.onWorkspaceSettingsChange(next);
    else setLocalSettings(next);
  };
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
  const [fusionError,setFusionError]=useState<string|null>(null);
  const [graphicsLost,setGraphicsLost]=useState(false),[graphicsGeneration,setGraphicsGeneration]=useState(0);
  const [committedSurgical,setCommittedSurgical]=useState<{identity:typeof rendererIdentity;interaction:SurgicalInteraction|null}|null>(null);
  const [modelDisplayPending,setModelDisplayPending]=useState(false);
  const [modelDisplayError,setModelDisplayError]=useState<string|null>(null);
  const lastCameraFitRequest=useRef(props.cameraFitRequest??0);
  const activePlane=settings.activePlane??'axial',contrast=settings.contrast??1,showPlane=settings.showPlane??false;
  const displayedSurgical=committedSurgical?.identity===rendererIdentity?committedSurgical.interaction:null;
  const surgicalView=displayedSurgical?.preview??displayedSurgical?.view;
  const surgicalMarkers=useMemo(()=>surgicalView?surgicalEffectMarkers(surgicalView,displayedSurgical?.outcome):[],[surgicalView,displayedSurgical?.outcome]);
  const setActivePlane=(activePlane:SlicePlane)=>changeSettings({...latestSettings.current,activePlane});
  const setContrast=(contrast:number)=>changeSettings({...latestSettings.current,contrast});
  const setShowPlane=(showPlane:boolean)=>changeSettings({...latestSettings.current,showPlane});
  const framing=sliceFraming(settings.sliceFraming);
  const setFraming=(fit:'image'|'anatomy',zoom=1)=>changeSettings({...latestSettings.current,sliceFraming:{fit,zoom}});
  const zoomSlices=(amount:number)=>{const current=sliceFraming(latestSettings.current.sliceFraming);setFraming(current.fit,Math.min(4,Math.max(.5,Math.round((current.zoom+amount)*100)/100)));};
  const layout=settings.layout??'3d-focus';
  const setLayout=(layout:'3d-focus'|'mri-review')=>changeSettings({...latestSettings.current,layout});
  const [replayError, setReplayError] = useState<string | null>(null),
    [passiveError, setPassiveError] = useState<string | null>(null),
    [rehearsalError, setRehearsalError] = useState<string | null>(null),
    [proposalError, setProposalError] = useState<string | null>(null),
    [proposalReviewLabel, setProposalReviewLabel] = useState<string | null>(
      null,
    );
  const [expanded, setExpanded] = useState<SlicePlane | null>(null);
  const [navigationMode,setNavigationMode]=useState<'orbit'|'pan'>('orbit');
  const [controlPanel,setControlPanel]=useState<'layout'|'appearance'|'clip'|null>(null);
  const [obliqueAngle,setObliqueAngle]=useState(0);
  const smoothSurfaces=settings.smoothSurfaces!==false;
  const setSmoothSurfaces=(smoothSurfaces:boolean)=>changeSettings({...latestSettings.current,smoothSurfaces});
  const [displayedPrior, setDisplayedPrior] = useState<ViewerPriorLayer | null>(
      null,
    ),
    [priorError, setPriorError] = useState<string | null>(null);
  const [paneSizes, setPaneSizes] = useState<
    Record<SlicePlane, [number, number]>
  >({ axial: [1, 1], coronal: [1, 1], sagittal: [1, 1] });
  const geometry = useMemo(() => {
    if (!geometrySource) return null;
    try {
      const affine = rasAffine(geometrySource.affine, geometrySource.frame);
      return {
        affine,
        inverse: inverseAffine(affine),
        bounds: volumeBounds(geometrySource.shape, affine),
        support: sourceSignalBounds(geometrySource.mri,geometrySource.shape,affine),
      };
    } catch {
      return null;
    }
  }, [geometrySource]);
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

  const [surfaceReports,setSurfaceReports]=useState<Record<string,{diagnostics?:import('./surfacePolicy').SurfaceDiagnostics;empty?:boolean}>>({});
  useEffect(()=>{
    setSurfaceReports({});const element=canvas.current;if(!element)return;
    const receive=(event:Event)=>{const d=(event as CustomEvent).detail;setSurfaceReports(old=>({...old,[d.name]:d}));};
    element.addEventListener('anatomy-surface',receive);return ()=>element.removeEventListener('anatomy-surface',receive);
  },[rendererIdentity,graphicsGeneration]);

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
    setError(null);setModelDisplayError(null);
    setReady(false);
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
        camera=>{
          if(latestProps.current.caseData&&modeledSourceIdentity(latestProps.current.caseData)===rendererIdentity)
            changeSettings({...latestSettings.current,camera});
        },
        status=>{
          if(engine.current!==renderer)return;
          if(status==='lost')setGraphicsLost(true);
          else setGraphicsGeneration(generation=>generation+1);
        },
      );
      engine.current = renderer;
      setGraphicsLost(false);
      setReady(true);
      return () => {
        renderer.dispose();
        engine.current = null;
      };
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : String(cause));
    }
  }, [rendererIdentity,graphicsGeneration]);

  useEffect(()=>{
    if(caseData?.displayIdentity&&!modeledAnatomy(caseData)&&engine.current){
      try{engine.current.setAnatomyLayers(caseData);}catch(cause){setError(cause instanceof Error?cause.message:String(cause));}
    }
  },[caseData,ready,graphicsGeneration]);

  useEffect(()=>{engine.current?.setWorkspaceSettings(settings);},[ready,graphicsGeneration,caseData,settings]);
  useEffect(()=>{engine.current?.restoreCamera(settings.camera);},[ready,graphicsGeneration,caseData,settings.camera]);
  useEffect(()=>{
    const renderer=engine.current,requested=props.surgicalInteraction??null;if(!renderer||!caseData)return;
    if(!requested?.view){setCommittedSurgical({identity:rendererIdentity,interaction:null});setModelDisplayPending(false);setModelDisplayError(null);}
    else setModelDisplayPending(true);
    const admitted=renderer.installSurgicalState(caseData,requested?{...requested,onDisplayReady:view=>{
      if(engine.current!==renderer)return;setCommittedSurgical({identity:rendererIdentity,interaction:requested});setModelDisplayPending(false);setModelDisplayError(null);requested.onDisplayReady?.(view);
    },onDisplayError:(view,message)=>{if(engine.current!==renderer)return;setModelDisplayPending(false);setModelDisplayError(message);requested.onDisplayError?.(view,message);}}:null);
    if(!admitted)setModelDisplayPending(false);
  },[ready,graphicsGeneration,caseData,props.surgicalInteraction]);
  useEffect(()=>{engine.current?.setNavigationMode(navigationMode);},[ready,graphicsGeneration,caseData,navigationMode]);
  useEffect(()=>{engine.current?.setSurfaceSmoothing(smoothSurfaces);},[ready,graphicsGeneration,caseData,smoothSurfaces]);
  useEffect(()=>{engine.current?.setVolumeRendering(settings.volumeRendering,signalName);},[ready,graphicsGeneration,caseData,settings.volumeRendering,signalName]);
  useEffect(()=>{
    if(!engine.current)return;
    setFusionError(null);
    try{engine.current.setRegisteredFusion(props.registeredFusion??null);}
    catch(cause){setFusionError(cause instanceof Error?cause.message:String(cause));}
  },[ready,graphicsGeneration,caseData,props.registeredFusion]);

  useEffect(() => {if(engine.current)engine.current.setPublicGoal(publicGoal);},[ready,graphicsGeneration,caseData,publicGoal]);

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
    graphicsGeneration,
    caseData,
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
  }, [activePlane, showPlane, ready, graphicsGeneration, caseData]);

  useEffect(()=>{
    const request=props.cameraFitRequest??0;
    if(!engine.current||request===lastCameraFitRequest.current)return;
    lastCameraFitRequest.current=request;
    engine.current.frameCamera(cameraMode);
  },[ready,graphicsGeneration,props.cameraFitRequest,cameraMode]);

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
    graphicsGeneration,
    caseData,
    replay?.recordedTool,
    replay?.recordedEpisodeBounds,
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
    if (!engine.current) return;
    setPassiveError(null);
    try {
      engine.current.setPassiveReplay(passiveReplay ?? null);
    } catch (cause) {
      setPassiveError(cause instanceof Error ? cause.message : String(cause));
    }
  }, [ready, graphicsGeneration, caseData, passiveReplay?.episodeId, passiveReplay?.frameIndex,
    passiveReplay?.recordedTools, passiveReplay?.recordedEpisodeBounds,
    passiveReplay?.primaryPairClearanceChecked, passiveReplay?.independentTissueMotionChecked,
    passiveReplay?.shape, passiveReplay?.affine, passiveReplay?.phase,
    passiveReplay?.elapsedGeneratedSeconds, passiveReplay?.inspectionStatus,
    passiveReplay?.event]);

  useEffect(() => {
    if (!engine.current) return;
    setRehearsalError(null);
    try { engine.current.setRehearsal(rehearsalState ?? null); }
    catch (cause) { setRehearsalError(cause instanceof Error ? cause.message : String(cause)); }
  }, [ready, graphicsGeneration, caseData, rehearsalState]);

  useEffect(() => {
    const renderer = engine.current;
    if (!renderer) return;
    setInspectionError(null);
    setDisplayedInspection(null);
    const suppressed = diagnosticRequested || passiveRequested || rehearsalRequested || Boolean(replay || structuralProposal || priorLayer);
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
  }, [ready, graphicsGeneration, caseData, inspectionTool, Boolean(replay), passiveRequested, Boolean(structuralProposal), Boolean(priorLayer), diagnosticRequested, rehearsalRequested]);

  useEffect(() => {
    const renderer = engine.current;
    if (!renderer) return;
    const [low, high] = renderer.defaultWindow,
      centre = (low + high) / 2,
      width = (high - low) * contrast;
    renderer.setWindow(centre - width / 2, centre + width / 2);
  }, [contrast, ready, graphicsGeneration, caseData]);

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
    graphicsGeneration,
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
    const scroll=new SliceScrollAccumulator();let scrollFrame=0;
    const aff=caseData?rasAffine(caseData.affine,caseData.frame):null;
    const inverse=aff?inverseAffine(aff):null;
    const bounds=aff&&caseData?volumeBounds(caseData.shape,aff):null;
    for (const plane of PLANES) {
      const node = refs[plane].current;
      if (!node) continue;
      observer.observe(node);
      const listener = (event: WheelEvent) => {
        if (event.deltaY === 0) return;
        event.preventDefault();
        if(!inverse||!bounds)return;
        engine.current?.beginInteraction();engine.current?.endInteraction();
        const axis=PLANE_AXES[plane][2];
        const step=1/Math.hypot(inverse[0][axis],inverse[1][axis],inverse[2][axis]);
        scroll.add(axis,(event.deltaY>0?-1:1)*step*(event.shiftKey?5:1));
        if(!scrollFrame)scrollFrame=requestAnimationFrame(()=>{
          scrollFrame=0;const p=latestProps.current;
          const cursor=p.cursor??bounds[0].map((v,i)=>(v+bounds[1][i])/2);
          p.onCursorChange(scroll.flush(cursor,bounds[0],bounds[1]));
        });
      };
      node.addEventListener("wheel", listener, { passive: false });
      listeners.push(() => node.removeEventListener("wheel", listener));
    }
    return () => {
      observer.disconnect();
      cancelAnimationFrame(scrollFrame);
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
          replay || passiveRequested || rehearsalRequested || structuralProposal ? null : (priorLayer ?? null),
        ),
      );
    } catch (cause) {
      setPriorError(cause instanceof Error ? cause.message : String(cause));
    }
  }, [
    ready,
    graphicsGeneration,
    caseData,
    Boolean(replay),
    passiveRequested,
    rehearsalRequested,
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

  useEffect(() => {setDiagnosticVisible(true);}, [diagnosticLayer, diagnosticSourceSha256, caseData]);

  useEffect(() => {
    if (!engine.current) return;
    setDiagnosticError(null);
    setDisplayedDiagnostic(null);
    try {
      setDisplayedDiagnostic(engine.current.setDiagnosticLayer(
        diagnosticVisible ? (diagnosticLayer ?? null) : null,
        diagnosticSourceSha256 ?? null,
      ));
    } catch (cause) {
      setDiagnosticError(cause instanceof Error ? cause.message : String(cause));
    }
  }, [ready, graphicsGeneration, caseData, diagnosticLayer, diagnosticSourceSha256, diagnosticVisible]);
  const diagnosticSample = displayedDiagnostic && diagnosticVisible &&
    displayedDiagnostic.layer.descriptor.source_sha256.t1c === diagnosticSourceSha256
      ? sampleDiagnosticLayer(displayedDiagnostic, currentCursor) : null;

  function moveCursor(
    event: PointerEvent<HTMLDivElement>,
    plane: SlicePlane,
    phase: 'start' | 'move' = 'move',
  ): void {
    if (!geometry || (event.button !== 0 && event.buttons !== 1)) return;
    const rect = event.currentTarget.getBoundingClientRect(),
      view = mprView(geometry.bounds,geometry.support,plane,rect.width,rect.height,framing);
    const point = mprPoint(
      view,
      plane,
      currentCursor,
      event.clientX - rect.left,
      event.clientY - rect.top,
      geometry.inverse,
      caseData!.shape,
    );
    if (point) {
      event.preventDefault();
      onCursorChange(point);
      props.onSliceStroke?.(point, phase);
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
  const displayBounds=geometry?.bounds??[[0,0,0],[1,1,1]] as [Point3,Point3];
  const clipping=clippingForBounds(settings.clipping,displayBounds);
  const clipAxis={x:0,y:1,z:2}[clipping.axis];
  const modeled = replay && !replayError;
  const passiveMode = passiveReplay && !passiveError;
  const rehearsalMode = rehearsalState && !rehearsalError && !replayError;
  const volumeDisplayed = scanVolume.enabled && !modeled && !passiveMode && !rehearsalMode;
  const inspected = !diagnosticRequested && !rehearsalRequested && !replay && !passiveRequested && !structuralProposal && !priorLayer &&
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
      onKeyDownCapture={event=>{if(event.key==='Escape')setControlPanel(null)}}
      onPointerDownCapture={event=>{if((event.target as HTMLInputElement).type==='range')engine.current?.beginInteraction();if(controlPanel&&!(event.target as HTMLElement).closest('.rl-control-popover,.rl-primary-tools'))setControlPanel(null)}}
      onPointerUpCapture={event=>{if((event.target as HTMLInputElement).type==='range')engine.current?.endInteraction();}}
      onPointerCancelCapture={()=>engine.current?.endInteraction()}
      data-layout={layout}
      data-surfaces-pending={remaining}
      data-expanded={expanded ?? undefined}
      aria-label="Linked patient imaging workspace"
    >
      <canvas ref={canvas} className="rl-viewer-canvas" aria-hidden="true" />
      <div
        className="rl-viewer-layoutbar"
        role="toolbar"
        aria-label="Imaging layout and 3D controls"
      >
        <div className="rl-primary-tools">
          {(['layout', 'appearance', 'clip'] as const).map(id => {
            const Icon = {layout: Columns2, appearance: SlidersHorizontal, clip: ScanLine}[id];
            return (
              <button key={id} aria-expanded={controlPanel === id}
                onClick={() => setControlPanel(controlPanel === id ? null : id)}>
                <Icon size={14}/>{id[0].toUpperCase() + id.slice(1)}
              </button>
            );
          })}
          <button onClick={()=>engine.current?.cameraPreset('oblique')} aria-label="Fit 3D view"><Focus size={14}/>Fit 3D</button>
          <span className="viewer-tool-divider"/>
          <button aria-label="Show structures only" aria-pressed={!scanVolume.enabled} onClick={()=>changeSettings({...settings,volumeRendering:{...settings.volumeRendering,enabled:false}})}><Layers3 size={14}/>Structures</button>
          <button aria-label="Show scan volume with structures" aria-pressed={scanVolume.enabled} disabled={props.segmentationOnly||props.generatedSignal||Boolean(replay)||passiveRequested||rehearsalRequested} onClick={()=>changeSettings({...settings,volumeRendering:{...settings.volumeRendering,enabled:true}})}><Box size={14}/>Volume</button>
        </div>
        <div className="rl-slice-framing" role="group" aria-label="Source slice framing">
          <select aria-label="Source slice fit" title="Display framing only. Full image restores the complete source image." value={framing.fit} onChange={event=>setFraming(event.target.value as 'image'|'anatomy')}>
            <option value="anatomy">Fit scan content</option><option value="image">Fit full image</option>
          </select>
          <button aria-label="Zoom source slices out" disabled={framing.zoom<=.5} onClick={()=>zoomSlices(-.25)}>−</button>
          <button className="rl-slice-zoom" aria-label="Reset source slice zoom" title="Reset slice zoom to 100%" onClick={()=>setFraming(framing.fit)}>{Math.round(framing.zoom*100)}%</button>
          <button aria-label="Zoom source slices in" disabled={framing.zoom>=4} onClick={()=>zoomSlices(.25)}>+</button>
        </div>
        <span className="rl-viewer-expanded-note">{expanded?TITLES[expanded]:layout==='3d-focus'?'3D + slices':'Image review'}</span>
        {controlPanel&&<div className="rl-control-popover" role="region" aria-label={`${controlPanel} controls`}>
          <div className="popover-heading"><strong>{controlPanel[0].toUpperCase()+controlPanel.slice(1)}</strong><button aria-label="Close view controls" onClick={()=>setControlPanel(null)}>×</button></div>
          {controlPanel==='layout'&&<>
            <div className="viewer-field-row"><button aria-pressed={layout==='3d-focus'} onClick={()=>{setLayout('3d-focus');setExpanded(null);setControlPanel(null)}}>3D + slices</button><button aria-pressed={layout==='mri-review'} onClick={()=>{setLayout('mri-review');setExpanded(null);setControlPanel(null)}}>Four views</button></div>
            <label>Camera<select aria-label="Camera preset" defaultValue="oblique" onChange={e=>engine.current?.cameraPreset(e.target.value as 'oblique'|'anterior'|'left'|'superior')}><option value="oblique">Oblique</option><option value="anterior">Anterior</option><option value="left">Left</option><option value="superior">Superior</option></select></label>
            <label>Left drag<select aria-label="3D drag mode" value={navigationMode} onChange={e=>setNavigationMode(e.target.value as 'orbit'|'pan')}><option value="orbit">Rotate</option><option value="pan">Pan</option></select></label>
          </>}
          {controlPanel==='appearance'&&<>
            <label><input type="checkbox" aria-label="Scan volume" checked={scanVolume.enabled} disabled={props.segmentationOnly||props.generatedSignal||Boolean(replay)||passiveRequested||rehearsalRequested} onChange={e=>changeSettings({...settings,volumeRendering:{...settings.volumeRendering,enabled:e.target.checked}})}/>Scan volume</label>
            <label>Preset<select aria-label="Volume rendering preset" value={settings.volumeRendering?.preset??'auto'} onChange={e=>changeSettings({...settings,volumeRendering:{...settings.volumeRendering,preset:e.target.value as VolumePreset}})}><option value="auto">Automatic · {signalName}</option><option value="ct-bone" disabled={!isCtModality(signalName)}>CT bone</option><option value="ct-tissue" disabled={!isCtModality(signalName)}>CT tissue</option><option value="ct-angiography" disabled={!isCtModality(signalName)}>CT angiography · contrast scan</option><option value="mri">{/MRA|TOF|angiogra/i.test(signalName)?'Bright-blood MR signal':'MRI tissue'}</option></select></label>
            <label>Scan density<input aria-label="Scan volume density" type="range" min="0" max="2" step=".01" value={scanVolume.density} onChange={e=>changeSettings({...settings,volumeRendering:{...settings.volumeRendering,density:+e.target.value}})}/></label>
            <label><input type="checkbox" aria-label="Smooth 3D surfaces" checked={smoothSurfaces} onChange={e=>setSmoothSurfaces(e.target.checked)}/>Smooth surfaces</label>
            <label>Context appearance<select aria-label="Context appearance" value={settings.contextStyle??'uniform'} onChange={e=>changeSettings({...settings,contextStyle:e.target.value as 'shell'|'uniform'})}><option value="shell">Edge shell · structures visible</option><option value="uniform">Uniform transparency</option></select></label>
            <label>Slice annotations<select aria-label="Slice annotation style" value={settings.sliceOverlay??'fill'} onChange={e=>changeSettings({...settings,sliceOverlay:e.target.value as 'contour'|'fill'})}><option value="contour">Contours · source detail</option><option value="fill">Filled labels</option></select></label>
            <label><input type="checkbox" checked={settings.volumeRendering?.shadows!==false} onChange={e=>changeSettings({...settings,volumeRendering:{...settings.volumeRendering,shadows:e.target.checked}})}/>Volume shadows</label>
            <label><input type="checkbox" checked={settings.volumeRendering?.isolateLabels===true} onChange={e=>changeSettings({...settings,volumeRendering:{...settings.volumeRendering,isolateLabels:e.target.checked}})}/>Isolate selected tissue</label>
            {diagnosticLayer&&<label><input type="checkbox" checked={diagnosticVisible} onChange={e=>setDiagnosticVisible(e.target.checked)}/>Diagnostic overlay</label>}
            <details aria-label="Anatomy display quality"><summary>Anatomy quality</summary><p>Source slices and masks remain the basis for review. Anatomical accuracy is unmeasured.</p>{caseData.compartments.map(layer=>{const d=surfaceReports[layer.name]?.diagnostics??layer.surfaceDiagnostics;return <div key={layer.name}><strong>{layer.name}</strong>{layer.qa?.issues.map(issue=><p key={issue}>{issue}</p>)}{d&&<p>{d.fallbackReason?d.fallbackReason:d.restoredComponents?`${d.restoredComponents} components use their original surface after geometry checks.`:'Surface geometry checks passed.'}{d.backedOffVertices?` Smoothing reduced locally at ${d.backedOffVertices.toLocaleString()} vertices.`:''}</p>}{surfaceReports[layer.name]?.empty&&<p>Empty segmentation: anatomy is unavailable.</p>}</div>})}</details>
            <TransferEditor value={settings.volumeRendering??{}} onChange={volumeRendering=>changeSettings({...settings,volumeRendering})}/>
          </>}
          {controlPanel==='clip'&&<>
            <label><input type="checkbox" checked={clipping.enabled} onChange={e=>changeSettings({...settings,clipping:{...clipping,enabled:e.target.checked}})}/>Cutaway</label>
            <label>Plane<select aria-label="Cutaway axis" value={clipping.axis} onChange={e=>changeSettings({...settings,clipping:{...clipping,axis:e.target.value as 'x'|'y'|'z'}})}><option value="z">Axial</option><option value="y">Coronal</option><option value="x">Sagittal</option></select></label>
            <input aria-label="Cutaway position" type="range" min={displayBounds[0][clipAxis]} max={displayBounds[1][clipAxis]} step=".1" value={clipping.value} disabled={!clipping.enabled} onChange={e=>changeSettings({...settings,clipping:{...clipping,value:+e.target.value}})}/>
            <button disabled={!clipping.enabled} onClick={()=>changeSettings({...settings,clipping:{...clipping,flipped:!clipping.flipped}})}>Flip side</button>
            <label><input type="checkbox" checked={showPlane} onChange={e=>setShowPlane(e.target.checked)}/>Source plane in 3D</label><select aria-label="Source plane orientation" value={activePlane} onChange={e=>setActivePlane(e.target.value as SlicePlane)}>{PLANES.map(p=><option key={p} value={p}>{TITLES[p]}</option>)}</select>
            <details><summary>Oblique & box cuts</summary><label>Angle · degrees<input aria-label="Oblique angle" type="number" min="-180" max="180" value={obliqueAngle} onChange={e=>setObliqueAngle(+e.target.value)}/></label><button onClick={()=>{const normal:Point3=[Math.sin(obliqueAngle*Math.PI/180),0,Math.cos(obliqueAngle*Math.PI/180)];changeSettings({...settings,obliqueClip:{enabled:true,normal,offset:-normal.reduce((sum,n,i)=>sum+n*currentCursor[i],0)}})}}>Cut at cursor</button><button onClick={()=>changeSettings({...settings,clipBox:{enabled:true,min:currentCursor.map(n=>n-50) as Point3,max:currentCursor.map(n=>n+50) as Point3}})}>100 mm box at cursor</button><button onClick={()=>changeSettings({...settings,clipBox:undefined,obliqueClip:undefined})}>Clear extra cuts</button></details>
          </>}
        </div>}
        {props.registeredFusion&&!fusionError&&<div className="rl-fusion-readout" role="status">Fused source images · {signalName} + {props.registeredFusion.label}<span>{props.registeredFusion.registration.status==='source-provided'?'Source-provided registration':'Verified registration'} · {Math.round((settings.fusion?.opacity??.5)*100)}% blend</span></div>}
        {expanded && (
          <span className="rl-viewer-expanded-note">
            {TITLES[expanded]} {signalName} expanded
          </span>
        )}
        {diagnosticLayer && <div className="rl-viewer-prior-readout" role="status" aria-live="polite">
          <span><strong>Model diagnostic · unreviewed</strong> · prepared T1c grid</span>
          <span className="rl-viewer-prior-value">{!diagnosticVisible ? "Overlay hidden" :
            diagnosticSample === "candidate_positive" ? "Cursor: model-positive candidate" :
            diagnosticSample === "candidate_negative" ? "Cursor: model-negative within output coverage" :
            "Cursor: unknown — no output coverage"}</span>
          <span className="rl-viewer-prior-note">Inferior support is omitted; training overlap is unknown. Hatching marks unknown coverage. Display only; not planning or clinical evidence.</span>
        </div>}
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
          <span>{settings.operativeView==='surgeon'?'Operative field · illustrative material':volumeDisplayed?'Volume + structures':'Segmentation surfaces'}</span>
          <span className="rl-viewer-frame">RAS · mm</span>
        </div>
        <div
          className="rl-viewer-anatomy-image"
          ref={anatomy}
          onPointerDown={event=>{if(props.surgicalInteraction?.mode==='draw'){const p=engine.current?.pick(event.clientX,event.clientY);if(p){event.currentTarget.setPointerCapture(event.pointerId);props.surgicalInteraction.onStroke(p,'start');}}}}
          onPointerMove={event=>{if(event.buttons===1&&props.surgicalInteraction?.mode==='draw'){const p=engine.current?.pick(event.clientX,event.clientY);if(p)props.surgicalInteraction.onStroke(p,'move');}}}
          onPointerUp={event=>{if(props.surgicalInteraction?.mode==='draw'){const p=engine.current?.pick(event.clientX,event.clientY);if(p)props.surgicalInteraction.onStroke(p,'end');}}}
          onDoubleClick={(event) => {
            const point = engine.current?.pick(event.clientX, event.clientY);
            if (point && geometry)
              onCursorChange(clampCursor(point, geometry.bounds));
          }}
          aria-label={props.surgicalInteraction?.mode==='draw'?'3-D source anatomy. Drag on anatomy to draw a proposed instrument stroke.':`3-D source anatomy. Drag to ${navigationMode==='pan'?'pan':'rotate'}, scroll to zoom, double-click to set linked cursor.`}
        >
          {settings.operativeView==='surgeon'&&surgicalView&&<div className="operative-readout" data-operative-readout-state={surgicalView.stateId}>
            <strong>Native operative state · {Number(surgicalView.metrics.elapsedSeconds).toFixed(1)} s</strong>
            <span>Target removed {Number(surgicalView.metrics.targetRemovedMm3).toFixed(1)} mm³ · remaining {Math.max(0,(surgicalView.scene.environment?.target.initialVolumeMm3??0)-Number(surgicalView.metrics.targetRemovedMm3)).toFixed(1)} mm³</span>
            <span>CUSA pending work {Number(surgicalView.metrics.cusaPendingCapacityMm3??0).toFixed(2)} mm³ · suction {Number(surgicalView.metrics.fluidRemovedMm3??0).toFixed(2)} mm³</span>
            <span>Field fluid {Number(surgicalView.metrics.fluidPresentMm3??0).toFixed(2)} mm³ · source grid {surgicalView.scene.resolutionMm.map(n=>n.toFixed(1)).join(' × ')} mm</span>
          </div>}
          <div className="rl-viewer-annotations">
            <span className="rl-viewer-dot" />{" "}
            {rehearsalMode
              ? `Research rehearsal · frame ${rehearsalState.frameIndex}/${rehearsalState.stepCount}`
              : passiveMode
              ? `Generated passive replay · frame ${passiveReplay.frameIndex}/${passiveReplay.stepCount}`
              : modeled
              ? `Modeled residual annotations · step ${replay.step}/${replay.stepCount}`
              : `${signalName} · ${volumeDisplayed?'source volume + labels':'source-derived labels'}`}
            {rehearsalMode && <>
              <span className="rl-viewer-replay-key">Mint: modeled removal · violet: assumed initial exposure · amber: field-management contact</span>
              <span className="rl-viewer-replay-key">Two persistent tools · {rehearsalState.elapsedSeconds.toFixed(1)} simulated s · source {signalName} unchanged · uncalibrated physics</span>
            </>}
            {passiveMode && (
              <span className="rl-viewer-replay-key">
                Blue moving tool · amber other tool · {passiveReplay.phase.replaceAll('_', ' ')} · declared {passiveReplay.elapsedGeneratedSeconds.toFixed(1)} s · generated status {passiveReplay.inspectionStatus.replaceAll('_', ' ')} · source {signalName} unchanged · no tissue removal
              </span>
            )}
            {passiveMode && (
              <span className="rl-viewer-replay-key">
                Primary pair clearance checked · independent tissue motion checked separately · generated geometry only
              </span>
            )}
            {modeled && (
              <span className="rl-viewer-replay-key">
                Mint mesh: modeled removal · source {signalName} unchanged
              </span>
            )}
            {publicGoal && <span className="rl-viewer-route-key" style={{color:PUBLIC_GOAL_COLOR}}>◇ Public goal · {publicGoal.state.replaceAll('-',' ')} · declared location, not anatomy</span>}
            {inspected && (
              <span className="rl-viewer-route-key" style={{ color: INSPECTION_TOOL_COLOR }}>
                ● Inspection tool · unexecuted preview · no tissue removed
                <button className="rl-viewer-inspection-clear" onClick={clearInspection}>Clear tool</button>
              </span>
            )}
            {inspected?.unknowns.includes("tool_geometry_outside_image_unassessed") && (
              <span className="rl-viewer-route-key">Tool extends beyond imaged anatomy · outside tissue unassessed</span>
            )}
            {!modeled && !passiveRequested && !rehearsalRequested && !inspected &&
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
            {!modeled && !passiveRequested && !rehearsalRequested && !inspected &&
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
            {modelDisplayPending&&<span className="rl-viewer-preparing" role="status">Preparing requested tissue state · previous accepted display retained</span>}
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
              {props.surgicalInteraction?.mode==='draw'?'Drag on anatomy to draw a proposed stroke':props.surgicalInteraction?.mode==='translate'?'Drag the arrows to translate the proposed tool':props.surgicalInteraction?.mode==='rotate'?'Drag the rings to rotate the proposed tool':`Drag to ${navigationMode==='pan'?'pan':'rotate'} · Scroll to zoom · Right-drag to pan`}
            </span>
            <span>Double-click to link cursor</span>
          </div>
        </div>
      </div>
      {PLANES.map((plane) => {
        const [left, right, top, bottom] = PLANE_LABELS[plane],
          axis = PLANE_AXES[plane][2];
        const [width, height] = paneSizes[plane];
        const view = geometry
          ? mprView(geometry.bounds,geometry.support,plane,width,height,framing)
          : null;
        const image=view?.rect??null;
        const mmWidth = view
          ? view.bounds[1][PLANE_AXES[plane][0]] -
            view.bounds[0][PLANE_AXES[plane][0]]
          : 1;
        const pxPerMm = image ? image.width / mmWidth : 1;
        const scaleMm = pxPerMm * 20 < 78 ? 20 : 10;
        const goalPixel = publicGoal && view && Math.abs(publicGoal.rasMm[axis]-currentCursor[axis])<=1e-6 ? mprPixel(view,plane,publicGoal.rasMm) : null;
        const slab=(surgicalView?Math.max(...surgicalView.scene.resolutionMm):1)*.6;
        const marks=view?clipSurgicalMarkersToSlab(surgicalMarkers,axis as 0|1|2,currentCursor[axis],slab).map(m=>({color:m.color,radius:surgicalMarkerRadiusPx(m,pxPerMm),points:m.points.map(p=>mprPixel(view,plane,p))})):[];
        return (
          <div
            key={plane}
            className={`rl-viewer-pane rl-viewer-slice rl-viewer-${plane}${expanded === plane ? " is-expanded" : ""}`}
          >
            <div className="rl-viewer-heading">
              <span className={`rl-viewer-plane-dot ${plane}`} />
              <span>{TITLES[plane]}</span>
              <span className="rl-slice-fit-badge" title="Display framing only; choose Fit full image to see the complete source extent.">{framing.fit==='image'?'Full image':'Content fit'}{framing.zoom!==1?` · ${Math.round(framing.zoom*100)}%`:''}</span>
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
                {expanded === plane ? <Minimize2 size={13}/> : <Maximize2 size={13}/>}
              </button>
            </div>
            <div
              ref={refs[plane]}
              className="rl-viewer-slice-image"
              tabIndex={0}
              aria-label={`${TITLES[plane]} source ${signalName}, ${framing.fit==='image'?'full image':'scan content fit'}, zoom ${Math.round(framing.zoom*100)} percent, neurological convention, slice position ${currentCursor[axis].toFixed(1)} millimeters. ${inspected ? "Inspection tool, unexecuted initial preview. No tissue removed. " : ""}${proposalReviewLabel ? `Estimated envelope contour, view only: ${proposalReviewLabel}. ` : ""}${displayedPrior ? "Population prior, alignment review required. Patient function unknown. Hatching marks unavailable atlas support. " : ""}Click to set cursor; scroll or arrow keys change slice.`}
              onPointerDown={(event) => {
                if (event.button === 0) {
                  engine.current?.beginInteraction();
                  event.currentTarget.focus({ preventScroll: true });
                  event.currentTarget.setPointerCapture(event.pointerId);
                  moveCursor(event, plane, 'start');
                }
              }}
              onPointerMove={(event) => {
                if (event.buttons === 1) moveCursor(event, plane);
              }}
              onPointerUp={() => {engine.current?.endInteraction();props.onSliceStroke?.(null, 'end');}}
              onPointerCancel={() => {engine.current?.endInteraction();props.onSliceStroke?.(null, 'end');}}
              onKeyDown={(event) => keySlice(event, plane)}
            >
              {surgicalView&&<svg aria-label="Modeled surgical effects" width="100%" height="100%" style={{position:'absolute',inset:0,pointerEvents:'none',zIndex:3}}>{marks.map((m,i)=>m.points.length===1?<circle key={i} cx={m.points[0][0]} cy={m.points[0][1]} r={m.radius} fill={m.color}/>:<polyline key={i} points={m.points.map(p=>p.join(',')).join(' ')} stroke={m.color} strokeWidth={2} fill="none"/>)}</svg>}
              {goalPixel && <span aria-label="Declared public goal" style={{position:'absolute',left:goalPixel[0],top:goalPixel[1],transform:'translate(-50%,-50%)',color:PUBLIC_GOAL_COLOR,fontSize:20,lineHeight:1,pointerEvents:'none',zIndex:4}}>◇</span>}
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
                {props.segmentationOnly ? 'SEGMENTATION ONLY · NO ACQUIRED IMAGE' : props.generatedSignal ? "GENERATED ANALYTIC SIGNAL" : `SOURCE ${signalName.toUpperCase()}`}
                {surgicalView ? (displayedSurgical?.preview?" · MODELED PREVIEW":" · MODELED EFFECT OVERLAY") : rehearsalMode ? " · RESEARCH EFFECT OVERLAY" : inspected
                  ? " · UNEXECUTED TOOL PREVIEW"
                  : proposalReviewLabel
                  ? " · ESTIMATE"
                  : displayedPrior
                    ? " · PRIOR"
                    : displayedDiagnostic
                      ? " · UNREVIEWED DIAGNOSTIC"
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
      {(error || modelDisplayError || replayError || passiveError || rehearsalError || proposalError || priorError || diagnosticError || inspectionError || goalSelection.error || fusionError) && (
        <div className="rl-viewer-error" role="alert">
          <strong>Imaging view needs attention</strong>
          <p>{error || modelDisplayError || replayError || passiveError || rehearsalError || proposalError || priorError || diagnosticError || inspectionError || goalSelection.error || fusionError}</p>
          {graphicsLost&&<button onClick={()=>engine.current?.retryGraphics()}>Retry graphics</button>}
        </div>
      )}
    </div>
  );
}

export default ViewerWorkspace;

import {surgicalFieldAmounts} from '../full-case-presentation';
import {PracticeHud} from './PracticeHud';
import {membraneTransport} from './membraneTransport';
import {resultMembership} from './resultInspectionData';
import {EngineOperatedSection} from './EngineOperatedSection';
import {OperativeSourceLocator} from './OperativeSourceLocator';
import {OperativeDepthSection} from './OperativeDepthSection';
import {OperativeLegend} from './OperativeLegend';
import {planningSliceSegments,planningSphereSliceRadius,planningSliceMemberships} from '../interactive-planning-data';
import {surgicalEffectMarkers,clipSurgicalMarkersToSlab,surgicalMarkerRadiusPx} from './surgicalGeometry';
import {modeledSourceIdentity,modeledAnatomy} from './modeledAnatomy';
import {tractColor,tractPreview,TractSliceIndex} from './tractDisplay';
import {approachCamera,approachClipBox} from './approachView';
import type {SurgicalInteraction} from '../surgical-types';
import {SliceScrollAccumulator} from './interactionBudget';
import {patientSliceCursor,patientSliceNavigation} from './sliceNavigation';
import {followSourceSection,restoreSourceSection,sourceDepthSlabAtCursor,sourceSectionAtCursor,sourceSectionSnapshot} from './sourceSection';
import type {SourceSectionSnapshot} from './sourceSection';
import {TransferEditor} from './TransferEditor';
import {sampleDiagnosticLayer} from "./diagnosticLayer";
import type {CheckedDiagnosticLayer} from "./diagnosticLayer";
import {checkedPublicGoal,PUBLIC_GOAL_COLOR} from './publicGoal';
import {mprView,mprPoint,mprPixel,sliceFraming,sourceSignalBounds} from './mprFraming';
import { useEffect, useId, useImperativeHandle, useMemo, useRef, useState } from "react";
import {
  Columns2,
  SlidersHorizontal,
  ScanLine,
  Focus,
  Download,
  Layers2,
  Layers3,
  Box,
  Maximize2,
  Minimize2,
  Ellipsis,
  ChevronUp,
  ChevronDown,
} from "lucide-react";
import type { KeyboardEvent, PointerEvent } from "react";
import { VolumeRenderer } from "./VolumeRenderer";
import {mixedVesselLesion} from '../native-study-state';
import { clippingForBounds, clippingPositionInBounds, newClippingPosition, usesPatientCutaway, viewerCutaway } from './workspaceSettings';
import type { ViewerWorkspaceSettings } from './workspaceSettings';
import { canBalanceTofContext, volumeSettings, isCtAngiography, isCtModality } from './volumeTransfer';
import type { CinematicBone, VolumePreset, VolumeRenderingMode } from './volumeTransfer';
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
import {operativeAnatomyTitle} from "./operativeProvenance";
import "./viewer.css";
import "./operativePresentation.css";

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
    ctaCoverage,
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
  const generatedSignalName = props.generatedSignalLabel ?? "generated image signal";
  const signalName = props.segmentationOnly ? 'segmentation only' : props.generatedSignal ? generatedSignalName : (props.sourceLabel ?? "MRI");
  const [localSettings,setLocalSettings] = useState<ViewerWorkspaceSettings>({});
  const settings=props.workspaceSettings ?? localSettings;
  const [overlapInspection,setOverlapInspection]=useState(false);
  const overlapLesion=mixedVesselLesion(caseData,settings);
  const overlapAvailable=overlapLesion&&visibleLayers[overlapLesion.name]&&
    (settings.layerOpacity?.[overlapLesion.name]??1)>0&&(settings.layerOpacity?.[overlapLesion.name]??1)<1;
  const inspectLesionOverlap=!!overlapAvailable&&overlapInspection;
  useEffect(()=>setOverlapInspection(false),[caseData?.displayIdentity,overlapAvailable]);
  const ctaCutaway=Boolean(caseData?.displayIdentity&&isCtAngiography(signalName)&&
    settings.presentation!=='scan'&&!settings.operativeView&&settings.clipping?.enabled&&
    settings.clipping.axis==='z'&&settings.clipping.flipped);
  const patientCutawayActive=usesPatientCutaway(settings,!!caseData?.patientContext?.boundsRasMm);
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
  const orientationGuide=useRef<HTMLDivElement>(null);
  const axial = useRef<HTMLDivElement>(null),
    coronal = useRef<HTMLDivElement>(null),
    sagittal = useRef<HTMLDivElement>(null);
  const engine = useRef<VolumeRenderer | null>(null),
    latestProps = useRef(props);
  latestProps.current = props;
  const exportJob=useRef<AbortController|null>(null);
  const [exportBusy,setExportBusy]=useState(false);
  const [exportStatus,setExportStatus]=useState('');
  const exportAnatomy=async()=>{
    if(exportJob.current){
      exportJob.current.abort(new Error('3D export cancelled.'));
      return;
    }
    const renderer=engine.current;
    if(!renderer)return;
    const job=new AbortController();
    exportJob.current=job;
    setExportBusy(true);
    setExportStatus('Rendering 3D PNG…');
    try{
      const blob=await renderer.exportAnatomyPng(job.signal);
      if(engine.current!==renderer)return;
      const url=URL.createObjectURL(blob);
      const link=document.createElement('a');
      link.href=url;
      link.download='resectionlab-3d.png';
      try{link.click();}finally{window.setTimeout(()=>URL.revokeObjectURL(url),1000);}
      setExportStatus('3D PNG ready.');
    }catch(cause){
      if(engine.current===renderer)setExportStatus(cause instanceof Error?cause.message:String(cause));
    }finally{
      if(exportJob.current===job){exportJob.current=null;setExportBusy(false);}
    }
  };
  useImperativeHandle(props.readCamera,()=>()=>engine.current?.getCameraState()??null,[]);
  const inspectionSelection = useRef(new InspectionSelectionGate());
  const [displayedInspection, setDisplayedInspection] = useState<InspectionToolDisplay | null>(null);
  const [inspectionError, setInspectionError] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null),
    [remaining, setRemaining] = useState(0),
    [ready, setReady] = useState(false);
  const [fusionError,setFusionError]=useState<string|null>(null);
  const [graphicsLost,setGraphicsLost]=useState(false),[graphicsGeneration,setGraphicsGeneration]=useState(0);
  const [committedSurgical,setCommittedSurgical]=useState<{identity:typeof rendererIdentity;volume:NonNullable<typeof caseData>;interaction:SurgicalInteraction|null}|null>(null);
  const [modelDisplayPending,setModelDisplayPending]=useState(false);
  const [modelDisplayError,setModelDisplayError]=useState<string|null>(null);
  const lastCameraFitRequest=useRef(props.cameraFitRequest??0);
  const lastStructureFocusRequest=useRef(props.structureFocusRequest??0);
  const previousApproachCamera=useRef<import('./workspaceSettings').ViewerCameraState|null>(null);
  const [approachViewMode,setApproachViewMode]=useState<'along'|'oblique'|null>(null);
  const [cropApproach,setCropApproach]=useState(false);
  const [showSliceTracts,setShowSliceTracts]=useState(true);
  const tractClipId=useId();
  const activePlane=settings.activePlane??'axial',contrast=settings.contrast??1,showPlane=settings.showPlane??false;
  const displayedSurgical=committedSurgical?.identity===rendererIdentity?committedSurgical.interaction:null;
  const surgicalView=displayedSurgical?.preview??displayedSurgical?.view;
  const operativeVessels=committedSurgical&&modeledAnatomy(committedSurgical.volume)?.vesselSurfaces;
  const sourceVolumeBoundary=operativeVessels&&[...operativeVessels].some(([name,surface])=>
    surface.sourceVolumeBoundary&&visibleLayers[name]!==false);
  const acquiredBrainSection=settings.operativeView==='surgeon'&&settings.microscope?.tissueStyle!=='labels'&&committedSurgical?.volume.compartments.some(layer=>
    layer.appearanceSource?.nativeWallCrop&&visibleLayers[layer.name]!==false);
  const showEngineSection=!!(props.engineInspection&&surgicalView&&!displayedSurgical?.preview&&committedSurgical);
  const resultOrbit=useRef<HTMLDivElement>(null);
  const inspectable=showEngineSection&&committedSurgical&&!!modeledAnatomy(committedSurgical.volume)?.surfaceReference;
  const resultParts=inspectable?settings.resultInspection:undefined;
  useEffect(()=>{
    const renderer=engine.current;if(!renderer||!ready)return;
    renderer.setResultInspection(resultParts?resultOrbit.current:null,resultParts,settings.resultCamera,
      pose=>changeSettings({...latestSettings.current,resultCamera:pose}));
  },[ready,graphicsGeneration,resultParts,settings.resultCamera,rendererIdentity]);
  const resultCells=resultParts&&committedSurgical&&surgicalView?resultMembership(committedSurgical.volume,surgicalView):null;
  const inspectResult=(mode:ViewerWorkspaceSettings['resultInspection'])=>changeSettings({...latestSettings.current,resultInspection:mode,
    ...(resultParts&&mode&&resultParts!==mode?{resultCamera:undefined}:{})});
  const [pickedLayer,setPickedLayer]=useState<{sceneId:string;name:string}|null>(null);
  const inspectRegion=(name:string|null)=>setPickedLayer(name&&surgicalView?{sceneId:surgicalView.scene.sceneId,name}:null);
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
  const [fieldExpanded,setFieldExpanded]=useState(false);
  const fieldPresentable=!!surgicalView&&(settings.operativeView==='surgeon'||!!surgicalView.operativeExposureV1)&&!resultParts;
  const fieldPresentation=fieldExpanded&&fieldPresentable;
  useEffect(()=>{
    if(!fieldPresentation)return;
    const escape=(event:globalThis.KeyboardEvent)=>{if(event.key==='Escape')setFieldExpanded(false);};
    document.addEventListener('keydown',escape);
    return ()=>document.removeEventListener('keydown',escape);
  },[fieldPresentation]);
  useEffect(()=>{engine.current?.setFieldPresentation(fieldPresentation);},[fieldPresentation,ready,graphicsGeneration,rendererIdentity]);
  const [expanded, setExpanded] = useState<SlicePlane | null>(null);
  const [navigationMode,setNavigationMode]=useState<'orbit'|'pan'>('orbit');
  const [controlPanel,setControlPanel]=useState<'layout'|'appearance'|'clip'|'info'|null>(null);
  const controlPanelElement=useRef<HTMLDivElement>(null),controlTrigger=useRef<HTMLButtonElement|null>(null);
  const controlsId=useId();
  const toggleControls=(panel:NonNullable<typeof controlPanel>,trigger:HTMLButtonElement,forceOpen=false)=>{controlTrigger.current=trigger;setControlPanel(controlPanel===panel&&!forceOpen?null:panel);};
  const closeControls=()=>{setControlPanel(null);controlTrigger.current?.focus({preventScroll:true});};
  useEffect(()=>{if(controlPanel)controlPanelElement.current?.querySelector<HTMLButtonElement>('button')?.focus({preventScroll:true});},[controlPanel]);
  const [sourceSection,setSourceSection]=useState<{identity:typeof rendererIdentity;patientCutaway:boolean;previous:SourceSectionSnapshot}|null>(null);
  const sourceSectionActive=Boolean(sourceSection&&sourceSection.identity===rendererIdentity&&sourceSection.patientCutaway===patientCutawayActive);
  const sourceSectionAllowed=!props.segmentationOnly&&settings.operativeView!=='surgeon';
  const [depthSlab,setDepthSlab]=useState<{identity:typeof rendererIdentity;patientCutaway:boolean;context:boolean}|null>(null);
  const [depthSlabThickness,setDepthSlabThickness]=useState(44);
  const depthSlabAllowed=!props.sourcePreview&&sourceSectionAllowed&&settings.presentation==='vascular'&&!props.generatedSignal&&
    !displayedSurgical&&!replay&&!passiveRequested&&!rehearsalRequested;
  const depthSlabActive=depthSlabAllowed&&depthSlab?.identity===rendererIdentity&&depthSlab?.patientCutaway===patientCutawayActive;
  useEffect(()=>{if(depthSlab&&!depthSlabActive)setDepthSlab(null);},[depthSlab,depthSlabActive]);
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
        sourceClipBounds:volumeBounds(geometrySource.shape,affine,true),
        patientClipBounds:geometrySource.patientContext?.boundsRasMm,
        support: sourceSignalBounds(geometrySource.mri,geometrySource.shape,affine),
      };
    } catch {
      return null;
    }
  }, [geometrySource]);
  const cutawayBounds=(patientCutawayActive?geometry?.patientClipBounds:undefined)??geometry?.sourceClipBounds;
  const currentCursor =
    cursor ??
    (geometry
      ? (geometry.bounds[0].map(
          (v, i) => (v + geometry.bounds[1][i]) / 2,
        ) as Point3)
      : [0, 0, 0]);
  const depthSlabBox=useMemo(()=>depthSlabActive&&cutawayBounds
    ?sourceDepthSlabAtCursor(currentCursor,cutawayBounds,activePlane,depthSlabThickness):undefined,
  [depthSlabActive,cutawayBounds,activePlane,depthSlabThickness,currentCursor[0],currentCursor[1],currentCursor[2]]);
  const exitSourceSection=()=>{
    if(sourceSectionActive&&sourceSection)changeSettings(restoreSourceSection(latestSettings.current,sourceSection.previous));
    setSourceSection(null);
  };
  const toggleSourceSection=()=>{
    if(sourceSectionActive){exitSourceSection();return;}
    if(!sourceSectionAllowed||!cutawayBounds)return;
    setDepthSlab(null);
    const previous=sourceSectionSnapshot(latestSettings.current);
    const cameraPosition=engine.current?.getCameraState()?.position??settings.camera?.position;
    changeSettings(sourceSectionAtCursor(latestSettings.current,currentCursor,cutawayBounds,cameraPosition));
    setSourceSection({identity:rendererIdentity,patientCutaway:patientCutawayActive,previous});
  };
  const toggleDepthSlab=()=>{
    if(depthSlabActive){setDepthSlab(null);return;}
    if(!depthSlabAllowed||!cutawayBounds)return;
    if(sourceSectionActive)exitSourceSection();
    setDepthSlab({identity:rendererIdentity,patientCutaway:patientCutawayActive,context:false});
  };
  useEffect(()=>{
    if(!sourceSection)return;
    // A source/presentation change or an explicit manual hide ends the shortcut
    // without overwriting the newly selected display with its previous state.
    if(!sourceSectionActive||!sourceSectionAllowed||!showPlane||!settings.clipping?.enabled){setSourceSection(null);return;}
    if(!cutawayBounds)return;
    const next=followSourceSection(settings,currentCursor,cutawayBounds);
    if(next!==settings)changeSettings(next);
  },[sourceSection,sourceSectionActive,sourceSectionAllowed,showPlane,settings.clipping,activePlane,cutawayBounds,currentCursor[0],currentCursor[1],currentCursor[2]]);
  const tractDisplays=useMemo(()=>geometrySource&&geometry?(geometrySource.tractography??[]).map((tract,index)=>({tract,color:tractColor(index),index:new TractSliceIndex(tractPreview(tract,geometry.affine))})):[],[geometrySource?.tractography,geometry?.affine]);
  const selectedApproach=routes.find(route=>route.scope==='geometric-approach'&&route.selected)??routes.find(route=>route.scope==='geometric-approach');
  const setApproachView=(mode:'along'|'oblique')=>{
    if(!selectedApproach||!caseData)return;
    const rect=anatomy.current?.getBoundingClientRect();
    const camera=approachCamera(selectedApproach,caseData.frame,mode,(rect?.width??1)/Math.max(1,rect?.height??1));
    if(!camera)return;
    if(!previousApproachCamera.current)previousApproachCamera.current=engine.current?.getCameraState()??null;
    changeSettings({...latestSettings.current,camera});setApproachViewMode(mode);
    onCursorChange(selectedApproach.target_mm.map((v,i)=>v*(caseData.frame.startsWith('LPS')&&i<2?-1:1)) as Point3);
  };
  const restoreApproachView=()=>{if(previousApproachCamera.current)changeSettings({...latestSettings.current,camera:previousApproachCamera.current});previousApproachCamera.current=null;setApproachViewMode(null);setCropApproach(false);};
  const approachViewSignature=selectedApproach?JSON.stringify([selectedApproach.route_id,selectedApproach.entry_mm,selectedApproach.target_mm,selectedApproach.tool,selectedApproach.window?.radius_mm]):null;
  const corridorCrop=useMemo(()=>cropApproach&&selectedApproach&&caseData?approachClipBox(selectedApproach,caseData.frame):undefined,[cropApproach,approachViewSignature,caseData?.frame]);
  const slabContextAllowed=!!depthSlabBox&&!corridorCrop&&!props.vesselTrace?.isolationRadiusMm;
  const slabContextActive=slabContextAllowed&&depthSlab?.context===true;
  const priorSlabCut=useMemo(()=>viewerCutaway(settings),[settings.clipping,settings.clipBox,settings.obliqueClip]);
  useEffect(()=>{previousApproachCamera.current=null;setApproachViewMode(null);setCropApproach(false);},[rendererIdentity]);
  useEffect(()=>{
    if(!approachViewMode||!selectedApproach||!caseData)return;
    const rect=anatomy.current?.getBoundingClientRect();
    const camera=approachCamera(selectedApproach,caseData.frame,approachViewMode,(rect?.width??1)/Math.max(1,rect?.height??1));
    if(camera)changeSettings({...latestSettings.current,camera});
  },[approachViewMode,approachViewSignature]);
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
      !orientationGuide.current ||
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
          orientationGuide: orientationGuide.current,
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
      setExportStatus('');
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

  useEffect(()=>{
    const presentation=corridorCrop?{...settings,clipBox:corridorCrop}:depthSlabBox
      ?{...settings,clipping:undefined,obliqueClip:undefined,clipBox:depthSlabBox}:settings;
    engine.current?.setWorkspaceSettings(presentation,inspectLesionOverlap);
  },[ready,graphicsGeneration,caseData,settings,corridorCrop,depthSlabBox,inspectLesionOverlap]);
  useEffect(()=>{
    const axis=PLANE_AXES[activePlane][2];
    engine.current?.setVesselSlabContext(slabContextActive&&depthSlabBox
      ?{axis,min:depthSlabBox.min[axis],max:depthSlabBox.max[axis],prior:priorSlabCut}:undefined);
  },[ready,graphicsGeneration,caseData,depthSlabBox,activePlane,slabContextActive,priorSlabCut]);
  useEffect(()=>{engine.current?.restoreCamera(settings.camera);},[ready,graphicsGeneration,caseData,settings.camera]);
  useEffect(()=>{engine.current?.setSelectedStructure(props.selectedStructure??null);},[ready,graphicsGeneration,caseData,props.selectedStructure]);
  useEffect(()=>{engine.current?.setVesselTrace(props.vesselTrace??null);},[ready,graphicsGeneration,caseData,props.vesselTrace]);
  useEffect(()=>{
    const request=props.structureFocusRequest??0;
    if(!engine.current||request===lastStructureFocusRequest.current)return;
    if(props.selectedStructure){const point=engine.current.focusStructure(props.selectedStructure,props.structureFocusPoint);if(point){lastStructureFocusRequest.current=request;onCursorChange(point);}}
  },[ready,graphicsGeneration,caseData,props.selectedStructure,props.structureFocusRequest,props.structureFocusPoint]);
  useEffect(()=>{
    const renderer=engine.current,requested=props.surgicalInteraction??null;if(!renderer||!caseData)return;
    if(!requested?.view){setCommittedSurgical({identity:rendererIdentity,volume:caseData,interaction:null});setModelDisplayPending(false);setModelDisplayError(null);}
    else setModelDisplayPending(true);
    const admitted=renderer.installSurgicalState(caseData,requested?{...requested,insertedInstrumentsOnly:!!props.engineInspection&&settings.operativeView==='surgeon'&&!requested.preview,onDisplayReady:view=>{
      if(engine.current!==renderer)return;setCommittedSurgical({identity:rendererIdentity,volume:caseData,interaction:requested});setModelDisplayPending(false);setModelDisplayError(null);requested.onDisplayReady?.(view);
    },onDisplayError:(view,message)=>{if(engine.current!==renderer)return;setModelDisplayPending(false);setModelDisplayError(message);requested.onDisplayError?.(view,message);}}:null);
    if(!admitted)setModelDisplayPending(false);
  },[ready,graphicsGeneration,caseData,props.surgicalInteraction,props.engineInspection,settings.operativeView]);
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
    const aff=geometrySource?rasAffine(geometrySource.affine,geometrySource.frame):null;
    const inverse=aff?inverseAffine(aff):null;
    const bounds=aff&&geometrySource?volumeBounds(geometrySource.shape,aff):null;
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
          const requested=scroll.flush(cursor,bounds[0],bounds[1]);
          let next=[...cursor] as Point3;
          for(const slicePlane of PLANES){const sliceAxis=PLANE_AXES[slicePlane][2];next=patientSliceCursor(aff!,inverse,geometrySource!.shape,next,slicePlane,requested[sliceAxis]);}
          p.onCursorChange(next);
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
  }, [geometrySource]);

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
      geometrySource!.shape,
    );
    if (point) {
      event.preventDefault();
      onCursorChange(point);
      props.onSliceStroke?.(point, phase);
    }
  }

  function changeSlice(plane: SlicePlane, value: number): void {
    if (!geometry || !geometrySource) return;
    onCursorChange(patientSliceCursor(geometry.affine, geometry.inverse, geometrySource.shape, currentCursor, plane, value));
  }

  function keySlice(event: KeyboardEvent<HTMLElement>, plane: SlicePlane): void {
    if (!geometry || !geometrySource) return;
    const navigation = patientSliceNavigation(geometry.affine, geometry.inverse, geometrySource.shape, currentCursor, plane);
    if (!navigation) return;
    const direction = {ArrowUp:1, ArrowRight:1, ArrowDown:-1, ArrowLeft:-1, PageUp:10, PageDown:-10}[event.key];
    if (direction === undefined && event.key !== 'Home' && event.key !== 'End') return;
    event.preventDefault();
    changeSlice(plane, event.key === 'Home' ? navigation.min : event.key === 'End' ? navigation.max : navigation.value + direction! * navigation.step * (event.shiftKey ? 5 : 1));
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
  const displayBounds=cutawayBounds??[[0,0,0],[1,1,1]] as [Point3,Point3];
  const clipping=clippingForBounds(settings.clipping,displayBounds);
  const clipAxis={x:0,y:1,z:2}[clipping.axis];
  const modeled = replay && !replayError;
  const passiveMode = passiveReplay && !passiveError;
  const rehearsalMode = rehearsalState && !rehearsalError && !replayError;
  const volumeDisplayed = scanVolume.enabled && !modeled && !passiveMode && !rehearsalMode;
  const inspected = !diagnosticRequested && !rehearsalRequested && !replay && !passiveRequested && !structuralProposal && !priorLayer &&
    displayedInspection?.caseHash === caseData.caseHash ? displayedInspection : null;
  const membrane=surgicalView&&!displayedSurgical?.preview?membraneTransport(surgicalView):null;
  const fieldAmounts=useMemo(()=>surgicalView?surgicalFieldAmounts(surgicalView):null,[surgicalView]);
  const operativeMeasurements=surgicalView&&fieldAmounts&&<>
            {!showEngineSection&&membrane&&<span>{membrane}</span>}
            <span>Owned target removed {Number(surgicalView.metrics.targetRemovedMm3).toFixed(1)} mm³ · Whole target remaining {surgicalView.originalTargetResidualMm3==null?'unknown':`${surgicalView.originalTargetResidualMm3.toFixed(1)} mm³`}</span>
            <span>CUSA pending work {Number(surgicalView.metrics.cusaPendingCapacityMm3??0).toFixed(2)} mm³ · suction {Number(surgicalView.metrics.fluidRemovedMm3??0).toFixed(2)} mm³</span>
            <span>Free-field fluid {fieldAmounts.spatialFluidMm3.toFixed(2)} mm³ · outlet reservoir {fieldAmounts.reservoirFluidMm3?.toFixed(2)??'unknown'} mm³ · retained total {fieldAmounts.retainedFluidMm3?.toFixed(2)??'unknown'} mm³</span>
            <span>Source grid {surgicalView.scene.resolutionMm.map(n=>n.toFixed(1)).join(' × ')} mm</span>
  </>;
  const clearInspection = () => {
    inspectionSelection.current.clear(inspectionTool ?? null);
    engine.current?.setInspectionTool(null);
    setDisplayedInspection(null);
    setInspectionError(null);
    latestProps.current.onClearInspection?.();
  };
  return (
    <div
      className={`rl-viewer rl-layout-${layout}${props.engineInspection?' engine-operated-view':''}${expanded ? " has-expanded" : ""}`}
      ref={container}
      onKeyDownCapture={event=>{if(event.key==='Escape'&&controlPanel){event.preventDefault();closeControls();}}}
      onPointerDownCapture={event=>{if((event.target as HTMLInputElement).type==='range')engine.current?.beginInteraction();if(controlPanel&&!(event.target as HTMLElement).closest('.rl-control-popover,.rl-primary-tools,.rl-viewer-details'))setControlPanel(null)}}
      onPointerUpCapture={event=>{if((event.target as HTMLInputElement).type==='range')engine.current?.endInteraction();}}
      onPointerCancelCapture={()=>engine.current?.endInteraction()}
      data-layout={layout}
      data-field-presentation={fieldPresentation||undefined}
      data-engine-section-layout={showEngineSection||undefined}
      data-result-inspection={resultParts}
      data-source-case-hash={caseData?.caseHash}
      data-source-display-identity={caseData?.displayIdentity}
      data-source-modality={props.sourceLabel}
      data-selected-structure={props.selectedStructure??''}
      data-visible-layer-ids={JSON.stringify(Object.keys(visibleLayers).filter(id=>visibleLayers[id]))}
      data-approach-view-mode={approachViewMode??''}
      data-approach-routes={JSON.stringify(routes)}
      data-source-preview={props.sourcePreview||undefined}
      data-renderer-ready={!props.sourcePreview&&ready&&!error&&!modelDisplayPending&&remaining===0?'true':'false'}
      data-surfaces-pending={remaining}
      data-expanded={expanded ?? undefined}
      aria-label={props.approachOverview?"Selected approach on intact anatomy":"Linked patient imaging workspace"}
    >
      {props.approachOverview&&<div className="operative-reference-heading">Selected approach · intact anatomy</div>}
      <canvas ref={canvas} className="rl-viewer-canvas" aria-hidden="true" />
      <div
        className="rl-viewer-layoutbar"
        role="toolbar"
        aria-label="Imaging layout and 3D controls"
      >
        {resultParts&&<div className="result-heading"><strong>3D result</strong>
          <button onClick={()=>inspectResult(undefined)}>Return to operative field</button></div>}
        <div className="rl-primary-tools">
          {fieldPresentable&&<button className="field-expand"
            aria-label={fieldPresentation?'Restore operative workspace':'Expand operative field'}
            title={fieldPresentation?'Restore operative workspace (Esc)':'Expand operative field'}
            aria-pressed={fieldPresentation} onClick={()=>{setFieldExpanded(!fieldPresentation);setControlPanel(null);setExpanded(null);}}>
            {fieldPresentation?<Minimize2 size={16}/>:<Maximize2 size={16}/>}
          </button>}
          <div className="rl-viewer-title">
            <strong>3D View</strong>
            <span title={settings.operativeView==='surgeon' ? 'Operative field · illustrative material' : `${signalName} · ${volumeDisplayed ? 'source volume + structures' : 'segmentation surfaces'}`}>
              {settings.operativeView==='surgeon' ? 'Operative field' : signalName}
            </span>
          </div>
          {(['layout', 'appearance', 'clip'] as const).map(id => {
            const Icon = {layout: Columns2, appearance: SlidersHorizontal, clip: ScanLine}[id];
            const label = id[0].toUpperCase() + id.slice(1);
            return (
              <button key={id} aria-label={label} title={id==='clip'?'Clip · Section at cursor':label} aria-expanded={controlPanel === id}
                aria-controls={controlsId} onClick={event => toggleControls(id,event.currentTarget)}>
                <Icon size={16}/><span className="rl-viewer-tool-label">{label}</span>
              </button>
            );
          })}
          {depthSlabAllowed&&<button aria-label="Vessel slab at cursor" aria-pressed={depthSlabActive}
            title={depthSlabActive?'Restore prior 3D depth':'Inspect a vessel slab at the linked cursor'}
            disabled={!cutawayBounds||!!corridorCrop} onClick={toggleDepthSlab}>
            <Layers2 size={16}/><span className="rl-viewer-tool-label">Vessel slab</span>
          </button>}
          <button onClick={()=>engine.current?.cameraPreset('oblique')} aria-label="Fit 3D view" title="Fit 3D view"><Focus size={16}/><span className="rl-viewer-tool-label">Fit 3D</span></button>
          {!resultParts&&<button onClick={()=>void exportAnatomy()} aria-label={exportBusy?'Cancel 3D export':'Export 3D PNG'}
            title={exportBusy?'Cancel 3D export':'Export high-resolution 3D PNG'}
            disabled={!exportBusy&&(props.sourcePreview||!ready||!!error||remaining>0||modelDisplayPending)}>
            <Download size={16}/><span className="rl-viewer-tool-label">PNG</span>
          </button>}
          <span className="viewer-tool-divider"/>
          <button aria-label="Show structures only" title="Structures" aria-pressed={!scanVolume.enabled} onClick={()=>changeSettings({...settings,volumeRendering:{...settings.volumeRendering,enabled:false}})}><Layers3 size={16}/><span className="rl-viewer-tool-label">Structures</span></button>
          <button aria-label="Show scan volume with structures" title="Volume" aria-pressed={scanVolume.enabled} disabled={props.segmentationOnly||passiveRequested} onClick={()=>changeSettings({...settings,volumeRendering:{...settings.volumeRendering,enabled:true}})}><Box size={16}/><span className="rl-viewer-tool-label">Volume</span></button>
          <button aria-label="View information and navigation" title="View information and navigation" aria-expanded={controlPanel==='info'} aria-controls={controlsId} onClick={event=>toggleControls('info',event.currentTarget)}><Ellipsis size={17}/></button>
        </div>
        <div ref={controlPanelElement} id={controlsId} className="rl-control-popover" role="region" aria-label={`${controlPanel??'view'} controls`} hidden={!controlPanel}>
          <div className="popover-heading"><strong>{controlPanel==='info'?'View information':controlPanel?controlPanel[0].toUpperCase()+controlPanel.slice(1):'View controls'}</strong><button aria-label="Close view controls" onClick={closeControls}>×</button></div>
          <div hidden={controlPanel!=='layout'}>
            <div className="viewer-field-row" hidden={showEngineSection}><button aria-pressed={layout==='3d-focus'} onClick={()=>{setLayout('3d-focus');setExpanded(null);closeControls()}}>3D + slices</button><button aria-pressed={layout==='mri-review'} onClick={()=>{setLayout('mri-review');setExpanded(null);closeControls()}}>Four views</button></div>
            <label>Camera<select aria-label="Camera preset" defaultValue="oblique" onChange={e=>engine.current?.cameraPreset(e.target.value as 'oblique'|'anterior'|'left'|'superior')}><option value="oblique">Oblique</option><option value="anterior">Anterior</option><option value="left">Left</option><option value="superior">Superior</option></select></label>
            <label>Left drag<select aria-label="3D drag mode" value={navigationMode} onChange={e=>setNavigationMode(e.target.value as 'orbit'|'pan')}><option value="orbit">Rotate</option><option value="pan">Pan</option></select></label>
            <div className="rl-slice-framing" role="group" aria-label="Source slice framing" hidden={showEngineSection}>
              <label>Slice fit<select aria-label="Source slice fit" title="Display framing only. Full image restores the complete source image." value={framing.fit} onChange={event=>setFraming(event.target.value as 'image'|'anatomy')}>
                <option value="anatomy">Fit scan content</option><option value="image">Fit full image</option>
              </select></label>
              <div className="viewer-field-row"><button aria-label="Zoom source slices out" disabled={framing.zoom<=.5} onClick={()=>zoomSlices(-.25)}>−</button>
                <button className="rl-slice-zoom" aria-label="Reset source slice zoom" title="Reset slice zoom to 100%" onClick={()=>setFraming(framing.fit)}>{Math.round(framing.zoom*100)}%</button>
                <button aria-label="Zoom source slices in" disabled={framing.zoom>=4} onClick={()=>zoomSlices(.25)}>+</button></div>
            </div>
            {selectedApproach&&<div className="rl-corridor-view-controls">
              <p>Selected corridor</p>
              <div className="viewer-field-row"><button aria-label="Look along selected corridor" aria-pressed={approachViewMode==='along'} onClick={()=>setApproachView('along')}>Along corridor</button>
              <button aria-label="Oblique view of selected corridor" aria-pressed={approachViewMode==='oblique'} onClick={()=>setApproachView('oblique')}>Corridor oblique</button></div>
              <button aria-pressed={!!corridorCrop} onClick={()=>setCropApproach(value=>!value)}>Crop around corridor</button>
              {approachViewMode&&<button onClick={restoreApproachView}>Restore view</button>}
            </div>}
          </div>
          {controlPanel==='appearance'&&<>
            <label><input type="checkbox" aria-label="Scan volume" checked={scanVolume.enabled} disabled={props.segmentationOnly||passiveRequested} onChange={e=>changeSettings({...settings,volumeRendering:{...settings.volumeRendering,enabled:e.target.checked}})}/>Scan volume</label>
            {caseData?.displayIdentity&&settings.presentation==='scan'&&!settings.operativeView&&volumeDisplayed&&isCtModality(signalName)&&
              caseData.compartments.some(layer=>layer.kind==='bone'||layer.kind==='skull')&&
              <label title="Show skull and bone estimates over the acquired CT. Linked slice contours stay available.">
                <input type="checkbox" checked={settings.volumeRendering?.showCtBoneOverlay!==false||settings.volumeRendering?.isolateLabels===true}
                  disabled={settings.volumeRendering?.isolateLabels===true}
                  onChange={event=>changeSettings({...settings,volumeRendering:{...settings.volumeRendering,showCtBoneOverlay:event.target.checked}})}/>
                Show 3D bone overlay
              </label>}
            <label title="Cinematic lighting traces shadows and light bounces through the acquired scan and refines the still image while the view is at rest. Vessel layers shown here color the volume instead of drawing their surfaces.">Volume lighting<select aria-label="Volume lighting" value={settings.volumeRendering?.mode??'standard'} disabled={!scanVolume.enabled} onChange={e=>changeSettings({...settings,volumeRendering:{...settings.volumeRendering,mode:e.target.value as VolumeRenderingMode}})}><option value="standard">Standard</option><option value="cinematic">Cinematic · refines at rest</option></select></label>
            {settings.volumeRendering?.mode==='cinematic'&&isCtModality(signalName)&&
              <label title="Solid bone is lit and opaque and hides vessels inside an uncut skull; see-through keeps them visible behind a thin bone shell. Automatic uses solid bone at a cut.">Bone<select aria-label="Cinematic bone" value={settings.volumeRendering.cinematicBone??'auto'} disabled={!scanVolume.enabled}
                onChange={e=>{
                  const volumeRendering={...settings.volumeRendering};
                  if(e.target.value==='auto')delete volumeRendering.cinematicBone;
                  else volumeRendering.cinematicBone=e.target.value as CinematicBone;
                  changeSettings({...settings,volumeRendering});
                }}><option value="auto">Automatic · solid when cut</option><option value="shell">See-through</option><option value="solid">Solid</option></select></label>}
            <label>Preset<select aria-label="Volume rendering preset" value={settings.volumeRendering?.preset??'auto'} onChange={e=>changeSettings({...settings,volumeRendering:{...settings.volumeRendering,preset:e.target.value as VolumePreset}})}><option value="auto">Automatic · {signalName}</option><option value="ct-bone" disabled={!isCtModality(signalName)}>CT bone</option><option value="ct-tissue" disabled={!isCtModality(signalName)}>CT tissue</option><option value="ct-angiography" disabled={!isCtModality(signalName)}>CT angiography · contrast scan</option><option value="mri">{/MRA|TOF|angiogra/i.test(signalName)?'Bright-blood MR signal':'MRI tissue'}</option></select></label>
            <label>Scan density<input aria-label="Scan volume density" type="range" min="0" max="2" step=".01" value={scanVolume.density}
              onChange={e=>{
                const volumeRendering={...settings.volumeRendering,density:+e.target.value};
                delete volumeRendering.balanceTofContext;
                changeSettings({...settings,volumeRendering});
              }}/></label>
            {caseData?.displayIdentity&&settings.presentation==='vascular'&&settings.contextStyle==='shell'&&!settings.operativeView&&
              !settings.clipping?.enabled&&!settings.clipBox?.enabled&&!settings.obliqueClip?.enabled&&!corridorCrop&&!depthSlabBox&&
              canBalanceTofContext(settings.volumeRendering,signalName)&&
              <label title="Set volume strength from this acquisition's signal and physical depth. Adjusting scan density switches to manual.">
                <input type="checkbox" checked={settings.volumeRendering?.balanceTofContext===true}
                  onChange={event=>changeSettings({...settings,volumeRendering:{...settings.volumeRendering,balanceTofContext:event.target.checked}})}/>
                Balance TOF context
              </label>}
            {caseData?.ctExterior&&(settings.presentation==='segmentation'||settings.presentation==='vascular')&&
              !settings.clipping?.enabled&&!settings.clipBox?.enabled&&!settings.obliqueClip?.enabled&&
              !corridorCrop&&!depthSlabBox&&!settings.volumeRendering?.isolateLabels&&
              settings.volumeRendering?.preset!=='mri'&&!settings.volumeRendering?.points&&!settings.volumeRendering?.window&&
              <label><input type="checkbox" checked={settings.volumeRendering?.showCtExteriorSignal===true}
                onChange={event=>changeSettings({...settings,volumeRendering:{...settings.volumeRendering,showCtExteriorSignal:event.target.checked}})}/>
                Show detached CT signal</label>}
            {caseData.displayIdentity&&!settings.operativeView&&(settings.presentation==='vascular'||settings.presentation==='segmentation')&&
              caseData.compartments.some(layer=>['artery','vein','sinus','vessel','vessels'].includes(layer.kind??''))&&
              <label title="Separate overlapping courses and light acquired CT cuts from an angle.">
                <input type="checkbox" checked={settings.vesselDepthSeparation!==false}
                  onChange={event=>changeSettings({...settings,vesselDepthSeparation:event.target.checked})}/>
                Vessel depth lighting
              </label>}
            {overlapAvailable&&<label title="Inspect projected overlap using the displayed scan alignment.">
              <input type="checkbox" checked={inspectLesionOverlap} onChange={event=>setOverlapInspection(event.target.checked)}/>
              Show vessels through lesion
            </label>}
            <label><input type="checkbox" aria-label="Smooth 3D surfaces" disabled={props.sourcePreview} checked={smoothSurfaces} onChange={e=>setSmoothSurfaces(e.target.checked)}/>Smooth surfaces</label>
            <label><input type="checkbox" aria-label="Body orientation guide" checked={settings.showOrientationGuide!==false} onChange={e=>changeSettings({...settings,showOrientationGuide:e.target.checked})}/>Body orientation guide</label>
            <label>Context appearance<select aria-label="Context appearance" value={settings.contextStyle??'uniform'} onChange={e=>changeSettings({...settings,contextStyle:e.target.value as 'shell'|'uniform'})}><option value="shell">Edge shell · structures visible</option><option value="uniform">Uniform transparency</option></select></label>
            <label>Slice annotations<select aria-label="Slice annotation style" value={settings.sliceOverlay??'fill'} onChange={e=>changeSettings({...settings,sliceOverlay:e.target.value as 'contour'|'fill'})}><option value="contour">Contours · source detail</option><option value="fill">Filled labels</option></select></label>
            {tractDisplays.length>0&&<label><input type="checkbox" checked={showSliceTracts} onChange={e=>setShowSliceTracts(e.target.checked)}/>Supplied tracts in source slices</label>}
            <label><input type="checkbox" checked={settings.volumeRendering?.shadows!==false} onChange={e=>changeSettings({...settings,volumeRendering:{...settings.volumeRendering,shadows:e.target.checked}})}/>Volume shadows</label>
            <label><input type="checkbox" checked={settings.volumeRendering?.isolateLabels===true} onChange={e=>changeSettings({...settings,volumeRendering:{...settings.volumeRendering,isolateLabels:e.target.checked}})}/>Isolate selected tissue</label>
            {diagnosticLayer&&<label><input type="checkbox" checked={diagnosticVisible} onChange={e=>setDiagnosticVisible(e.target.checked)}/>Diagnostic overlay</label>}
            <details aria-label="Anatomy display quality"><summary>Anatomy quality</summary><p>Source slices and masks remain the basis for review. Anatomical accuracy is unmeasured.</p>{caseData.compartments.map(layer=>{const d=surfaceReports[layer.name]?.diagnostics??layer.surfaceDiagnostics;return <div key={layer.name}><strong>{layer.name}</strong>{layer.qa?.issues.map(issue=><p key={issue}>{issue}</p>)}{d?.policy==='engine-shared-implicit-v1'&&<p>Shared material interfaces · maximum display displacement {d.maxDisplacementMm.toFixed(3)} mm{d.displacementLimitMm!=null&&<> of {d.displacementLimitMm.toFixed(3)} mm allowed</>}.</p>}{d&&<p>{d.fallbackReason?d.fallbackReason:d.restoredComponents?`${d.restoredComponents} components use their original surface after geometry checks.`:'Surface geometry checks passed.'}{d.backedOffVertices?` Smoothing reduced locally at ${d.backedOffVertices.toLocaleString()} vertices.`:''}{d.volumeRestoredComponents?` Lesion volume checks restored ${d.volumeRestoredComponents} source components.`:''}</p>}{surfaceReports[layer.name]?.empty&&<p>Empty segmentation: anatomy is unavailable.</p>}</div>})}</details>
            <TransferEditor value={settings.volumeRendering??{}} onChange={volumeRendering=>changeSettings({...settings,volumeRendering})}/>
          </>}
          {controlPanel==='clip'&&<>
            <button className="rl-section-action" aria-label={sourceSectionActive?'Exit section':'Section at cursor'} aria-pressed={sourceSectionActive} disabled={!sourceSectionAllowed} onClick={toggleSourceSection}><ScanLine size={16}/>{sourceSectionActive?'Exit section':'Section at cursor'}</button>
            <p className="rl-section-help">{props.segmentationOnly?'Segmentation only · no acquired image to section.':!sourceSectionAllowed?'Exit the operative view to inspect a source image section.':sourceSectionActive?`${props.generatedSignal?generatedSignalName.charAt(0).toUpperCase()+generatedSignalName.slice(1):`Source ${signalName}`} · ${TITLES[activePlane]} section follows the linked cursor. Exit restores the previous cut.`:'Show the source image across the cut, at the linked cursor. Choose its plane below.'}</p>
            {depthSlabActive?<>
              <label>Slab plane<select aria-label="Slab plane" value={activePlane} onChange={event=>setActivePlane(event.target.value as SlicePlane)}>{PLANES.map(plane=><option key={plane} value={plane}>{TITLES[plane]}</option>)}</select></label>
              <label>Slab thickness<select aria-label="Slab thickness" value={depthSlabThickness} onChange={event=>setDepthSlabThickness(Number(event.target.value))}>{[20,28,36,44,60,80].map(mm=><option key={mm} value={mm}>{mm} mm</option>)}</select></label>
            </>:<>
            <label><input type="checkbox" checked={clipping.enabled} onChange={e=>changeSettings({...settings,clipping:{...clipping,enabled:e.target.checked}})}/>Cutaway</label>
            <label>Plane<select aria-label="Cutaway axis" value={clipping.axis} onChange={e=>{
              const axis=e.target.value as 'x'|'y'|'z',i={x:0,y:1,z:2}[axis];
              changeSettings({...settings,clipping:{...clipping,axis,value:newClippingPosition(axis,displayBounds,currentCursor[i])}});
            }}><option value="z">Axial</option><option value="y">Coronal</option><option value="x">Sagittal</option></select></label>
            {settings.clipping&&!clippingPositionInBounds(settings.clipping.axis,settings.clipping.value,displayBounds)&&<p role="status">Cutaway is outside this field. Choose a new plane or turn Cutaway on to reset its position.</p>}
            <input aria-label="Cutaway position" type="range" min={displayBounds[0][clipAxis]} max={displayBounds[1][clipAxis]} step=".1" value={clipping.value} disabled={!clipping.enabled||sourceSectionActive} title={sourceSectionActive?'Section follows the linked cursor':'Cutaway position'} onChange={e=>changeSettings({...settings,clipping:{...clipping,value:+e.target.value}})}/>
            <button disabled={!clipping.enabled} onClick={()=>changeSettings({...settings,clipping:{...clipping,flipped:!clipping.flipped}})}>Flip side</button>
            <label><input type="checkbox" checked={showPlane} disabled={props.segmentationOnly} onChange={e=>setShowPlane(e.target.checked)}/>Source plane in 3D</label><select aria-label="Source plane orientation" value={activePlane} disabled={sourceSectionActive||props.segmentationOnly} onChange={e=>setActivePlane(e.target.value as SlicePlane)}>{PLANES.map(p=><option key={p} value={p}>{TITLES[p]}</option>)}</select>
            <details><summary>Oblique & box cuts</summary><label>Angle · degrees<input aria-label="Oblique angle" type="number" min="-180" max="180" value={obliqueAngle} onChange={e=>setObliqueAngle(+e.target.value)}/></label><button onClick={()=>{const normal:Point3=[Math.sin(obliqueAngle*Math.PI/180),0,Math.cos(obliqueAngle*Math.PI/180)];changeSettings({...settings,obliqueClip:{enabled:true,normal,offset:-normal.reduce((sum,n,i)=>sum+n*currentCursor[i],0)}})}}>Cut at cursor</button><button onClick={()=>changeSettings({...settings,clipBox:{enabled:true,min:currentCursor.map(n=>n-50) as Point3,max:currentCursor.map(n=>n+50) as Point3}})}>100 mm box at cursor</button><button onClick={()=>changeSettings({...settings,clipBox:undefined,obliqueClip:undefined})}>Clear extra cuts</button></details>
            </>}
          </>}
          <section className="rl-viewer-information" hidden={controlPanel!=='info'} aria-label="Linked view information">
            <p className="rl-info-source"><strong>{signalName}</strong> · {settings.operativeView==='surgeon' ? (settings.microscope?.tissueStyle==='labels'?'operative field · source labels':'operative field · illustrative material') : volumeDisplayed?'source volume + structures':'segmentation surfaces'}</p>
            <div className="rl-viewer-statusbar" onPointerDown={event=>event.stopPropagation()}>
              <span><i className="rl-viewer-linked"/>Linked views <b>RAS</b> {currentCursor.map(v=>v.toFixed(1)).join(' / ')} mm</span>
              <label>Contrast<input type="range" min="0.4" max="2" step="0.05" value={contrast} onChange={event=>setContrast(Number(event.target.value))} aria-label={`Source ${signalName} window width`}/></label>
              <span>Neurological convention</span>
            </div>
            {geometry&&<p>Source voxel · {transformPoint(geometry.inverse,currentCursor).map(v=>v.toFixed(1)).join(' / ')}</p>}
            <p>Patient-frame orthogonal views · {framing.fit==='image'?'full source image':'scan content fit'} · {Math.round(framing.zoom*100)}% zoom. Counters describe patient-frame planes; ≈ marks a resampled oblique plane. The scrubber follows source coverage through the linked cursor.</p>
            <details open><summary>Navigation</summary><p>Click a slice or double-click anatomy to link the cursor. Scroll a slice or use its mint scrubber to move through the source image. Arrow keys move one sampling step; Shift moves five; Page Up / Down move ten.</p><p>{props.surgicalInteraction?.mode==='draw'?'Drag on anatomy to draw a proposed stroke.':props.surgicalInteraction?.mode==='translate'?'Drag the arrows to translate the proposed tool.':props.surgicalInteraction?.mode==='rotate'?'Drag the rings to rotate the proposed tool.':`Drag anatomy to ${navigationMode==='pan'?'pan':'rotate'}. Scroll to zoom; right-drag to pan.`}</p></details>
          </section>
        </div>
        {props.registeredFusion&&!fusionError&&<div className="rl-fusion-readout" role="status">Fused source images · {signalName} + {props.registeredFusion.label}<span>{props.registeredFusion.registration.status==='source-provided'?'Source-provided registration':'Reviewed alignment'} · {Math.round((settings.fusion?.opacity??.5)*100)}% blend</span></div>}
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
        <div
          className="rl-viewer-anatomy-image"
          data-practice-interaction={displayedSurgical?.practiceHud?displayedSurgical.mode:undefined}
          ref={anatomy}
          onPointerDown={event=>{if(props.surgicalInteraction?.mode==='draw'){const p=engine.current?.pick(event.clientX,event.clientY);if(p){event.currentTarget.setPointerCapture(event.pointerId);props.surgicalInteraction.onStroke(p,'start');}}}}
          onPointerMove={event=>{if(event.buttons===1&&props.surgicalInteraction?.mode==='draw'){const p=engine.current?.pick(event.clientX,event.clientY);if(p)props.surgicalInteraction.onStroke(p,'move');}}}
          onPointerUp={event=>{if(props.surgicalInteraction?.mode==='draw'){const p=engine.current?.pick(event.clientX,event.clientY);if(p)props.surgicalInteraction.onStroke(p,'end');}}}
          onDoubleClick={(event) => {
            let sourceLayer:string|null=null;
            const point = engine.current?.pick(event.clientX, event.clientY, name=>{
              if (surgicalView) inspectRegion(name);
              else sourceLayer=name;
            });
            if (point && geometry){
              onCursorChange(patientCutawayActive?point:clampCursor(point, geometry.bounds));
              if(sourceLayer)props.onPickStructure?.(sourceLayer,point);
            }
          }}
          aria-label={props.surgicalInteraction?.mode==='draw'?'3-D source anatomy. Drag on anatomy to draw a proposed instrument stroke.':`3-D source anatomy. Drag to ${navigationMode==='pan'?'pan':'rotate'}, scroll to zoom, double-click to set linked cursor.`}
        >
          {fieldPresentation&&surgicalView&&<div className="field-presentation-readout" role="status">
            <b>{surgicalView.operativeExposureV1?operativeAnatomyTitle(geometrySource?.compartments??[]):'Operative field'}</b>
            <span>{displayedSurgical?.preview?'Preview':modelDisplayPending?'Preparing accepted state':settings.microscope?.inspectionTransparency||settings.microscope?.inspectionCutaway?'Inspection only · anatomy exposed by view':settings.operativeView==='surgeon'?'Opaque microscope':Object.values(settings.layerOpacity??{}).some(o=>o<1)?'Inspection only · transparent anatomy':'External scalp overview'}</span>
            <span>{Object.values(surgicalView.slots).filter(s=>s.inserted).map(s=>s.instrument.replaceAll('_',' ')).join(' · ')}</span>
            {acquiredBrainSection?<span>Gray end faces mark the source volume boundary</span>:
              sourceVolumeBoundary&&<span>Gray vessel ends mark the source volume boundary</span>}
          </div>}
          {surgicalView&&!fieldPresentation&&<OperativeLegend scene={surgicalView.scene} settings={settings} cursor={currentCursor}
            visible={visibleLayers} picked={pickedLayer?.sceneId===surgicalView.scene.sceneId?pickedLayer.name:null}
            onPick={inspectRegion} onSmooth={setSmoothSurfaces} sourceVolumeBoundary={sourceVolumeBoundary||acquiredBrainSection||undefined}/>}
          {!fieldPresentation&&settings.operativeView==='surgeon'&&settings.microscope&&geometrySource&&geometry&&surgicalView&&
            (!displayedSurgical?.preview||displayedSurgical.recordedFrame)&&
            <OperativeSourceLocator source={geometrySource} bounds={geometry.bounds} inverse={geometry.inverse}
              microscope={settings.microscope} labelsOnly={!!props.segmentationOnly} stateId={surgicalView.stateId}/>}
          {!fieldPresentation&&settings.operativeView==='surgeon'&&settings.microscope?.inspectionCutaway&&geometrySource&&geometry&&
            surgicalView?.operativeExposureV1&&!displayedSurgical?.preview&&
            <OperativeDepthSection source={geometrySource} inverse={geometry.inverse} microscope={settings.microscope} stateId={surgicalView.stateId}/>}
          {!fieldPresentation&&displayedSurgical?.practiceHud&&surgicalView&&<PracticeHud interaction={displayedSurgical} view={surgicalView} membrane={membrane}/>}
          {!fieldPresentation&&!displayedSurgical?.practiceHud&&settings.operativeView==='surgeon'&&surgicalView&&(showEngineSection?
            <details className="engine-frame-metrics" data-operative-readout-state={surgicalView.stateId}>
              <summary>{membrane??'Engine'} · {Number(surgicalView.metrics.elapsedSeconds).toFixed(1)} s</summary>
              {operativeMeasurements}
            </details>:
            <div className="operative-readout" data-operative-readout-state={surgicalView.stateId}>
              <strong>Native operative state · {Number(surgicalView.metrics.elapsedSeconds).toFixed(1)} s</strong>
              {operativeMeasurements}
            </div>)}
          <div className="rl-viewer-annotations">
            {exportStatus&&<span className="rl-viewer-preparing" role="status" aria-label="3D PNG export status">{exportStatus}</span>}
            {settings.operativeView==='surgeon'&&<span>{settings.microscope?.tissueStyle==='labels'?'Operative field · source labels':'Operative field · illustrative material'}</span>}
            {(rehearsalMode||passiveMode||modeled)&&<span className="rl-viewer-dot" />}
            {rehearsalMode
              ? `Research rehearsal · frame ${rehearsalState.frameIndex}/${rehearsalState.stepCount}`
              : passiveMode
              ? `Generated passive replay · frame ${passiveReplay.frameIndex}/${passiveReplay.stepCount}`
              : modeled
              ? `Modeled residual annotations · step ${replay.step}/${replay.stepCount}`
              : null}
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
              routes.slice(0, 3).map((route, index) => {
                const { slot, color } = routeAppearance(route, index);
                return (
                  <span
                    key={route.route_id}
                    className="rl-viewer-route-key"
                    style={{ color }}
                  >
                    ● {route.scope==='geometric-approach'?(route.label??`Approach ${slot}`):`Route ${slot}`}
                    {route.selected ? ' · selected' : ''}
                    {route.category === "rejected" ? " · rejected" : ""}
                  </span>
                );
              })}
            {selectedApproach&&<span className="rl-viewer-route-key">Corridor + virtual axis · geometric planning only · no tissue removed</span>}
            {tractDisplays.length>0&&<span className="rl-viewer-route-key">Supplied tract bundles · streamline display preview</span>}
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
          {sourceSectionActive&&<div className="rl-source-section-readout" role="status">
            <span>{props.generatedSignal?`Section of ${generatedSignalName}`:`Source ${signalName} section`} · {TITLES[activePlane]} · {'XYZ'[PLANE_AXES[activePlane][2]]} {currentCursor[PLANE_AXES[activePlane][2]].toFixed(1)} mm</span>
            <button aria-label="Exit source section" onClick={exitSourceSection}>Exit section</button>
          </div>}
          {depthSlabBox&&!corridorCrop&&<div className="rl-source-section-readout" role="status">
            <span>{TITLES[activePlane]} {depthSlabThickness} mm slab · linked slices stay full</span>
            {slabContextAllowed&&<button aria-label="Faint vessel context outside slab" aria-pressed={slabContextActive}
              title="Show more distant vessels faintly behind the focused slab"
              onClick={()=>setDepthSlab(current=>current&&{...current,context:!current.context})}>Faint context</button>}
            <button aria-label="Exit vessel slab in 3D" onClick={()=>setDepthSlab(null)}>Exit slab</button>
          </div>}
          {ctaCoverage&&!surgicalView&&<div className="rl-cta-coverage" role="group" aria-label="CTA arterial layers">
            <button aria-pressed={ctaCoverage.mode==='combined'} onClick={()=>ctaCoverage.onChange(true)}>Combined arteries</button>
            <button aria-pressed={ctaCoverage.mode==='focused'} onClick={()=>ctaCoverage.onChange(false)}>Focused branches</button>
          </div>}
          {ctaCutaway&&!surgicalView&&!sourceSectionActive&&!depthSlabBox&&!corridorCrop&&!settings.clipBox?.enabled&&
            !settings.obliqueClip?.enabled&&!props.vesselTrace?.isolationRadiusMm&&<div className="rl-source-section-readout" role="status">
            <span>Axial CTA cut · Z {settings.clipping!.value.toFixed(1)} mm · linked slices stay full</span>
            <button aria-label="Show full CTA head" onClick={()=>{
              engine.current?.fitCamera('anatomy');
              changeSettings({...settings,clipping:undefined,
                camera:engine.current?.getCameraState()??null});
            }}>Full head</button>
          </div>}
          {!sourceSectionActive&&(!depthSlabBox||corridorCrop)&&!surgicalView&&(corridorCrop||settings.clipBox?.enabled||props.vesselTrace?.isolationRadiusMm)&&
            <div className="rl-source-section-readout" role="status">
              <span>{props.vesselTrace?.isolationRadiusMm?'3D branch crop · linked slices retain full anatomy':corridorCrop?'3D corridor crop':'3D box crop'}</span>
              {corridorCrop?<button onClick={()=>setCropApproach(false)}>Exit corridor crop</button>:
                !props.vesselTrace?.isolationRadiusMm&&settings.clipBox?.enabled&&<button onClick={()=>changeSettings({...settings,clipBox:undefined})}>Clear box</button>}
            </div>}
          <div className="rl-body-guide" hidden={fieldPresentation||settings.showOrientationGuide===false}>
            <div ref={orientationGuide} className="rl-body-guide-image" role="img" aria-label="Body orientation guide. Mint head follows the brain. Anatomical reference.">
              <span data-guide-side="right" aria-hidden="true">R</span>
              <span data-guide-side="left" aria-hidden="true">L</span>
              <span data-guide-gravity hidden>
                <svg width="24" height="24" viewBox="0 0 24 24" aria-hidden="true"><line data-guide-arrow x1="12" y1="12" x2="12" y2="21"/><path data-guide-arrowhead d="M 8 17 L 12 21 L 16 17"/></svg>
                <span data-guide-axial hidden aria-hidden="true"/><small>Down</small>
              </span>
            </div>
            <small data-guide-caption>Anatomical reference</small>
            <div className="rl-head-axes" hidden={!surgicalView?.operativeExposureV1}>
              <svg data-head-axes viewBox="0 0 64 64" role="img"
                aria-label="Head axes: R patient right, A anterior, S superior; L, P and I opposite">
                {(['R','A','S'] as const).map((end,axis)=><g key={end} data-head-axis={axis} className={`axis-${end}`}>
                  <line x1="32" y1="32" x2="32" y2="32"/><text x="32" y="32">{end}</text><text x="32" y="32">{['L','P','I'][axis]}</text>
                </g>)}
              </svg>
              <small>Head axes</small>
            </div>
          </div>
        </div>
        {resultParts&&<>
          <div className="engine-result-orbit" ref={resultOrbit} aria-label="Rotatable three-dimensional cavity and residual target"
            onDoubleClick={()=>engine.current?.fitResultInspection()}>
            <span className="result-empty">No cells in the selected result at this state.</span>
          </div>
          <div className="result-controls" role="group" aria-label="Result surfaces">
            {(['both','cavity','residual'] as const).map(mode=><button key={mode} aria-pressed={resultParts===mode}
              aria-label={mode==='both'?'Show cavity and residual':`Show ${mode} only`} onClick={()=>inspectResult(mode)}>
              {mode==='both'?'Both':mode==='cavity'?'Cavity':'Residual'}</button>)}
            <button aria-label="Fit result" onClick={()=>{changeSettings({...latestSettings.current,resultCamera:undefined});engine.current?.fitResultInspection();}}>Fit</button>
          </div>
          {resultCells&&<div className="result-key">
            <span>Removed space · cavity cast{resultParts==='both'&&resultCells.cavity.length>0&&resultCells.residual.length>0?' · muted where behind target':''}</span><span>Owned residual target · {(resultCells.residual.length*resultCells.cellMm3).toFixed(1)} mm³</span>
            {resultCells.overlapOriginal>0&&<small>{resultCells.overlapRemaining.toLocaleString()} of {resultCells.overlapOriginal.toLocaleString()} overlapping target cells remain assigned to other tissues.</small>}
          </div>}
        </>}
      </div>
      {showEngineSection&&committedSurgical&&surgicalView&&<EngineOperatedSection key={surgicalView.scene.sceneId}
        onInspect={inspectable&&!resultParts?()=>inspectResult('both'):undefined}
        volume={committedSurgical.volume} view={surgicalView} labelsOnly={!!props.segmentationOnly} sourceLabel={signalName}/>}
      {PLANES.map((plane) => {
        const [left, right, top, bottom] = PLANE_LABELS[plane],
          axis = PLANE_AXES[plane][2];
        const navigation = geometry&&geometrySource ? patientSliceNavigation(geometry.affine,geometry.inverse,geometrySource.shape,currentCursor,plane) : null;
        const planeDescription = `${navigation?.resampled?'Resampled patient-frame plane':'Patient-frame slice'} ${navigation?.index??'—'} of ${navigation?.count??'—'} · ${'XYZ'[axis]} ${currentCursor[axis].toFixed(1)} mm`;
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
        const planning=displayedSurgical?.planningOverlay;
        const planningSegments=planning&&view?planningSliceSegments(planning,axis as 0|1|2,currentCursor[axis],slab).map(segment=>({...segment,pixels:segment.points.map(p=>mprPixel(view,plane,p))})):[];
        const planningRegions=planning&&!planning.recorded&&view?([['Target region',planning.constraints.target,'#ff91ae'],['Entry region',planning.constraints.entry,'#70e0ff']] as const).flatMap(([name,region,color])=>{
          const radius=planningSphereSliceRadius(region.centerMm,region.radiusMm,axis as 0|1|2,currentCursor[axis]);
          return radius===null?[]:[{name,color,center:mprPixel(view,plane,region.centerMm),radius:radius*pxPerMm}];
        }):[];
        const planningMemberships=planning&&view?planningSliceMemberships(planning,axis as 0|1|2,currentCursor[axis],slab).map(mark=>({...mark,pixel:mprPixel(view,plane,mark.positionMm)})):[];
        const planningLimits=planning&&view?planning.limitingStructures.filter(s=>Math.abs(s.positionMm[axis]-currentCursor[axis])<=slab).map(s=>({...s,pixel:mprPixel(view,plane,s.positionMm)})):[];
        const tractPaths=showSliceTracts&&view?tractDisplays.filter(({tract})=>visibleLayers[tract.id]!==false).map(({tract,color,index})=>{
          const selected=props.selectedStructure===tract.id;
          const opacity=settings.layerOpacity?.[tract.id]??.66;
          const segments=index.segments(axis as 0|1|2,currentCursor[axis],.65,1600);
          const dimmed=Boolean(props.selectedStructure&&tractDisplays.some(t=>t.tract.id===props.selectedStructure));
          return {id:tract.id,label:tract.label,color:settings.layerColor?.[tract.id]??color,selected,opacity:Math.min(1,opacity)*(dimmed&&!selected ? .4 : 1),path:segments.map(points=>points.map((point,i)=>`${i?'L':'M'}${mprPixel(view,plane,point).map(n=>n.toFixed(1)).join(',')}`).join('')).join('')};
        }):[];
        return (
          <div
            key={plane}
            className={`rl-viewer-pane rl-viewer-slice rl-viewer-${plane}${expanded === plane ? " is-expanded" : ""}`}
          >
            <div className="rl-viewer-heading">
              <span className="rl-viewer-plane-title">{TITLES[plane]}</span>
              <output className="rl-viewer-slice-position" aria-live="off" aria-label={`${TITLES[plane]} ${planeDescription}`} title={planeDescription}>
                {navigation?.resampled&&<span className="rl-slice-resampled" aria-hidden="true">≈</span>}
                {navigation?.index??'—'}<span>/{navigation?.count??'—'}</span>
              </output>
              <span className="rl-slice-fit-badge" title="Display framing only; choose Fit full image in Layout to see the complete source extent.">{framing.fit==='image'?'Full image':'Content fit'}{framing.zoom!==1?` · ${Math.round(framing.zoom*100)}%`:''}</span>
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
              <button className="rl-viewer-expand rl-viewer-details" aria-label={`${TITLES[plane]} view information`} title="View information" aria-expanded={controlPanel==='info'&&activePlane===plane} aria-controls={controlsId} onClick={event=>{setActivePlane(plane);toggleControls('info',event.currentTarget,activePlane!==plane)}}><Ellipsis size={16}/></button>
            </div>
            <div
              ref={refs[plane]}
              className="rl-viewer-slice-image"
              tabIndex={0}
              aria-label={`${TITLES[plane]} source ${signalName}, ${planeDescription}, ${framing.fit==='image'?'full image':'scan content fit'}, zoom ${Math.round(framing.zoom*100)} percent, neurological convention, slice position ${currentCursor[axis].toFixed(1)} millimeters. ${inspected ? "Inspection tool, unexecuted initial preview. No tissue removed. " : ""}${proposalReviewLabel ? `Estimated envelope contour, view only: ${proposalReviewLabel}. ` : ""}${displayedPrior ? "Population prior, alignment review required. Patient function unknown. Hatching marks unavailable atlas support. " : ""}Click to set cursor; scroll or arrow keys change slice.`}
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
              {tractPaths.length>0&&image&&<svg aria-label="Supplied tract intersections with source slice" width="100%" height="100%" style={{position:'absolute',inset:0,pointerEvents:'none',zIndex:2}}>
                <defs><clipPath id={`${tractClipId}-${plane}`}><rect x={image.left} y={image.top} width={image.width} height={image.height}/></clipPath></defs>
                <g clipPath={`url(#${tractClipId}-${plane})`}>{tractPaths.map(tract=><path key={tract.id} d={tract.path} stroke={tract.color} strokeWidth={tract.selected?2:1} opacity={tract.opacity} fill="none"><title>{tract.label} · fibers within ±0.65 mm of this source plane · display preview</title></path>)}</g>
              </svg>}
              {surgicalView&&<svg aria-label="Modeled surgical effects" width="100%" height="100%" style={{position:'absolute',inset:0,pointerEvents:'none',zIndex:3}}>{marks.map((m,i)=>m.points.length===1?<circle key={i} cx={m.points[0][0]} cy={m.points[0][1]} r={m.radius} fill={m.color}/>:<polyline key={i} points={m.points.map(p=>p.join(',')).join(' ')} stroke={m.color} strokeWidth={2} fill="none"/>)}</svg>}
              {planning&&<svg aria-label="Approach guides intersecting current MRI slab" width="100%" height="100%" style={{position:'absolute',inset:0,pointerEvents:'none',zIndex:3}}>
                {planningRegions.map(region=><circle key={region.name} cx={region.center[0]} cy={region.center[1]} r={Math.max(2,region.radius)} stroke={region.color} strokeDasharray="4 3" strokeWidth={1.5} fill="none"><title>{region.name}</title></circle>)}
                {planningSegments.map(segment=>Math.hypot(segment.pixels[0][0]-segment.pixels[1][0],segment.pixels[0][1]-segment.pixels[1][1])<1?<circle key={segment.id} cx={segment.pixels[0][0]} cy={segment.pixels[0][1]} r={segment.selected?4:2.5} fill={segment.color}/>:<polyline key={segment.id} points={segment.pixels.map(p=>p.join(',')).join(' ')} stroke={segment.color} strokeWidth={segment.selected?4:2} opacity={segment.selected?1:.65} fill="none"/>)}
                {planningMemberships.map(mark=><circle key={mark.id} cx={mark.pixel[0]} cy={mark.pixel[1]} r={mark.radiusPx} stroke={mark.color} strokeWidth={2} fill="none"><title>{mark.label}</title></circle>)}
                {planningLimits.map((limit,i)=><g key={i}><title>{limit.name}: {limit.reason}</title><path d={`M${limit.pixel[0]-5},${limit.pixel[1]-5}l10,10m-10,0l10,-10`} stroke="#ffb65c" strokeWidth="2"/></g>)}
              </svg>}
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
                {props.segmentationOnly ? 'SEGMENTATION ONLY · NO ACQUIRED IMAGE' : props.generatedSignal ? generatedSignalName.toUpperCase() : `SOURCE ${signalName.toUpperCase()}`}
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
            <div className="rl-slice-scrubber" role="group" aria-label={`${TITLES[plane]} slice navigation`}>
              <button aria-label={`Next ${TITLES[plane]} slice`} title="Next slice" disabled={!navigation||navigation.value>=navigation.max-1e-6} onClick={()=>navigation&&changeSlice(plane,navigation.value+navigation.step)}><ChevronUp size={15}/></button>
              <input type="range" className="rl-slice-range" aria-label={`${TITLES[plane]} slice position`} aria-orientation="vertical" aria-valuetext={planeDescription} title={planeDescription} min={navigation?.min??0} max={navigation?.max??1} step="any" value={navigation?.value??0} disabled={!navigation||navigation.max-navigation.min<1e-6} onChange={event=>changeSlice(plane,Number(event.target.value))} onKeyDown={event=>keySlice(event,plane)}/>
              <button aria-label={`Previous ${TITLES[plane]} slice`} title="Previous slice" disabled={!navigation||navigation.value<=navigation.min+1e-6} onClick={()=>navigation&&changeSlice(plane,navigation.value-navigation.step)}><ChevronDown size={15}/></button>
            </div>
          </div>
        );
      })}
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

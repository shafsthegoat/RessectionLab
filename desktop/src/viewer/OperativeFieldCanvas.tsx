import {installRetractionMorph} from './retractionMorph';
import { useEffect, useMemo, useRef, useState } from 'react';
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { RoomEnvironment } from 'three/addons/environments/RoomEnvironment.js';
import type { OperativeField, OperativePhase } from '../operative-field';
import { buildOperativeModel, clamp01, embeddedLayers, layerNames, retractionOffset, phaseAmount, smooth, type FieldLayer, type OperativeModel } from './operativeFieldModel.ts';
import type {AspirationPattern} from './operativeAspiration';
import { operativeVoxelSurface } from './operativeVoxelSurface.ts';
import { operativeSurface } from './operativeSurface.ts';
import {duraReflectionAngle} from './operativeDuralFlap.ts';
import { boneFlapTransfer, prepareOperativeOpening } from './operativeOpening.ts';
import { operativeInstruments } from './operativeInstruments.ts';
import { operativeSignalTexture,operativeTissueMaterial } from './operativeFieldAppearance.ts';

import type {CortexWindow} from './operativeCortexWindow';
import {entryApproachEnd,operativeCameraPose,operativeSourceFit} from './operativeCameraPath.ts';
import {microscopeFov} from './microscope.ts';
import {operativeSectionContext,type OperativeViewChoice} from './operativeSectionContext.ts';
import type {ViewerVolume} from './contracts';
import {pickOperativeStructure,type OperativePick} from './operativeFieldPick.ts';
import {operativeEntryCover,operativeFieldContext,limitToOperativePatch} from './operativeFieldContext.ts';
import type {Point3} from './coordinates';
import {fieldGrid,fieldLocalToRas,formatLabelDistance,labelDistance,labeledCells} from './toolNavigation.ts';

/** The visible modeled instrument tip in field-local rest coordinates (the
 * frame of the sampled labels, before display-only retraction morphing). */
export type OperativeToolTip = {tool:'Drill'|'Scalpel'|'CUSA'; local:Point3; ras:Point3};

type Props = {onToolTip?:(tip:OperativeToolTip|null)=>void;visibleLayers?:Readonly<Record<string,boolean>>;sweepPattern:AspirationPattern;sourceVolume:ViewerVolume|null;collectionId:string;field:OperativeField; phase:OperativePhase; progress:number; withRetraction:boolean; view:OperativeViewChoice; raw:boolean; onContext:(layers:string[])=>void;
  onVesselDisplay:(display:OperativeModel['vesselDisplay'])=>void};

function fieldScene(host:HTMLDivElement, field:OperativeField, sourceVolume:ViewerVolume|null,collectionId:string,sweepPattern:AspirationPattern,visibleLayers:Props['visibleLayers'],pick:(value:OperativePick|null)=>void,lost:()=>void) {
  const renderer=new THREE.WebGLRenderer({antialias:true,alpha:true,powerPreference:'high-performance'});
  renderer.setClearColor(0,0); renderer.setPixelRatio(Math.min(window.devicePixelRatio||1,2));
  renderer.outputColorSpace=THREE.SRGBColorSpace;
  renderer.toneMapping=THREE.ACESFilmicToneMapping;
  renderer.shadowMap.enabled=true;
  renderer.shadowMap.type=THREE.PCFShadowMap;
  renderer.domElement.style.cssText='width:100%; height:100%; display:block; touch-action:none; outline-offset:-3px;';
  renderer.domElement.tabIndex=0;
  renderer.domElement.setAttribute('aria-label','3D operative field. Drag to orbit, scroll to zoom. Arrow keys rotate. Home resets the view. Click tissue to inspect its source. Enter inspects the center.');
  host.appendChild(renderer.domElement);
  host.dataset.aspirationPattern=sweepPattern;
  const model=buildOperativeModel(field,sweepPattern), scene=new THREE.Scene();
  const inspectionCamera=new THREE.OrthographicCamera(-1,1,1,-1,.1,1600);
  const opticalCamera=new THREE.PerspectiveCamera(34,1,.1,1600);
  let workingDistanceMm=160;
  let camera:THREE.OrthographicCamera|THREE.PerspectiveCamera=opticalCamera;
  inspectionCamera.up.set(0,0,1);
  const patchRadius=Math.min(field.radiusMm,model.opening*1.55);
  const exterior={value:false};
  const context=operativeFieldContext(sourceVolume,field,collectionId,patchRadius);scene.add(context);
  camera.up.set(0,0,1); const direction=new THREE.Vector3(.36,-.91,.90).normalize();
  camera.position.copy(direction).multiplyScalar(400);
  const controls=new OrbitControls<THREE.Camera>(camera,renderer.domElement);
  controls.enablePan=false;controls.enableDamping=false;controls.minZoom=.45;controls.maxZoom=7;
  controls.minPolarAngle=.07;controls.maxPolarAngle=Math.PI*.92;controls.zoomSpeed=.85;
  const room=new RoomEnvironment(), pmrem=new THREE.PMREMGenerator(renderer);
  const environment=pmrem.fromScene(room,.06);
  room.dispose();pmrem.dispose();
  scene.environment=environment.texture;
  scene.environmentIntensity=.20;
  scene.add(new THREE.HemisphereLight('#fff8f1','#645b60',.36));
  const keyLight=new THREE.DirectionalLight('#fff5eb',2.05);
  const range=field.radiusMm*1.8;
  keyLight.castShadow=true;
  keyLight.shadow.mapSize.set(1024,1024);
  const shadowCamera=keyLight.shadow.camera;
  Object.assign(shadowCamera,{left:-range,right:range,top:range,bottom:-range,near:1,far:field.entryDepthMm+range*6});
  shadowCamera.updateProjectionMatrix();
  keyLight.shadow.normalBias=.09;
  keyLight.shadow.bias=-.00022;
  scene.add(keyLight,keyLight.target);
  const rim=new THREE.DirectionalLight('#d6e8ef',.43);
  rim.position.set(.78,1,1.15);scene.add(rim);
  const hingeX=model.duralOpening*.74, boneZ=field.stations.bone??0;
  const hingeZ=model.surfaceHeight('dura',hingeX,0)??field.stations.dura??0;
  const texture=field.intensity?operativeSignalTexture(field.intensity):null;
  const sectionY={value:field.origin[1]-field.spacingMm*.5};
  const sectionPlaneY=field.origin[1]+(model.firstSectionRow-.5)*field.spacingMm;
  let sectionContext:ReturnType<typeof operativeSectionContext>|undefined;

  const meshes=Object.fromEntries(layerNames.map(name=>{
    const offset=new THREE.Vector3();
    if(name==='duraFlap')offset.set(hingeX,0,hingeZ);
    if(name==='boneFlap')offset.z=boneZ;
    const material=operativeTissueMaterial(name,field,texture,offset,sectionY,model.vesselDisplay.conflictingVoxels);
    const mesh=new THREE.Mesh(new THREE.BufferGeometry(),material);
    const role=name==='boneFlap'?'bone':name==='duraFlap'?'dura':name==='vessel'?'vessels':name;
    mesh.userData.sourceLayer={role,offset,detail:['brain','bone','dura','tumor','vessels','lesionContext'].includes(role)?null:name};
    mesh.name=name;mesh.castShadow=name!=='csf'&&name!=='ventricles';mesh.receiveShadow=true;
    scene.add(mesh);return [name,mesh];
  })) as Record<FieldLayer,THREE.Mesh<THREE.BufferGeometry,THREE.MeshStandardMaterial>>;
  const transferBone=boneFlapTransfer(model);
  const openings={
    bone:prepareOperativeOpening('bone',model,meshes.bone,meshes.boneFlap,
      operativeTissueMaterial('bone',field,texture,new THREE.Vector3(),sectionY),new THREE.Vector3(0,0,boneZ)),
    dura:prepareOperativeOpening('dura',model,meshes.dura,meshes.duraFlap,
      operativeTissueMaterial('dura',field,texture,new THREE.Vector3(),sectionY),new THREE.Vector3(hingeX,0,hingeZ)),
  };
  const cortex=context.children.find(mesh=>mesh.userData.sourceLayer.cortex) as THREE.Mesh|undefined;
  const cortexWindow:CortexWindow|undefined=cortex?.userData.sourceLayer.cortex;
  for(const source of context.children){
    const role=source.userData.sourceLayer.role as 'brain'|'bone'|'grayMatter';
    // Source labels are user data and can be named "brain" or "bone" too.
    const local=role==='bone'?[meshes.bone,openings.bone.kerf]:[meshes[role]];
    for(const display of local){
      const surfaceWindow=role==='grayMatter'?cortexWindow:undefined;
      if(surfaceWindow){
        display.userData.sourceLayer.cortexLocal=surfaceWindow;
        display.customDepthMaterial=new THREE.MeshDepthMaterial({depthPacking:THREE.RGBADepthPacking});
      }
      limitToOperativePatch(display.material,patchRadius,exterior,surfaceWindow);
      if(display.customDepthMaterial)limitToOperativePatch(display.customDepthMaterial,patchRadius,exterior,surfaceWindow);
      display.userData.sourceLayer.patchRadius=patchRadius;
      display.userData.sourceLayer.exterior=exterior;
    }
  }
  const cover=context.children.some(mesh=>mesh.userData.sourceLayer.role==='bone')
    ?operativeEntryCover(sourceVolume,field,collectionId,visibleLayers):null;
  if(cover)context.add(cover);
  host.dataset.surroundingLayers=JSON.stringify(context.children.map(mesh=>mesh.name));
  host.dataset.surroundingSources=JSON.stringify(context.children.map(mesh=>mesh.userData.sourceLayer),(key,value)=>key==='activeFloor'?undefined:value);
  const equipment=operativeInstruments(model.tipRadius,model.opening,field.entryDepthMm);
  Object.values(equipment).forEach(group=>{
    group.traverse(object=>{if(object instanceof THREE.Mesh)object.castShadow=true});
    scene.add(group);
  });
  const cache=new Map<FieldLayer,string>();
  let viewSize=field.radiusMm*2.8, aspect=1, frame=0, disposed=false;
  const scale=host.querySelector<HTMLElement>('.operative-scale');
  const entryMarker=host.querySelector<HTMLElement>('.operative-entry-marker');
  const plannedEntry=new THREE.Vector3(0,0,field.entryDepthMm),screenEntry=new THREE.Vector3();
  let markerOpacity=0;
  const lightDirection=new THREE.Vector3();
  const draw=()=>{
    frame=0;
    if(disposed)return;
    if(camera===opticalCamera){
      // Rest the lamp beside the optics as the user orbits the opening.
      lightDirection.copy(camera.position).sub(controls.target).normalize();
      keyLight.target.position.copy(controls.target);
      keyLight.position.copy(controls.target).addScaledVector(lightDirection,range*2.6);
      keyLight.position.x-=range*.7;keyLight.position.y-=range*.34;
    }
    // In perspective the bar measures the optical focus plane.
    const heightMm=camera instanceof THREE.PerspectiveCamera
      ?2*camera.position.distanceTo(controls.target)*Math.tan(THREE.MathUtils.degToRad(camera.getEffectiveFOV())/2)
      :(camera.top-camera.bottom)/camera.zoom;
    const pxPerMm=host.clientHeight/heightMm;
    if(scale&&pxPerMm>0){
      const mm=[1,2,5,10,20,50,100].filter(mm=>mm*pxPerMm<85).pop()??1;
      scale.style.width=(mm*pxPerMm)+'px';
      scale.textContent=mm+' mm';
    }
    renderer.render(scene,camera);
    if(entryMarker){
      screenEntry.copy(plannedEntry).project(camera);
      entryMarker.hidden=markerOpacity<.01||Math.abs(screenEntry.x)>1||Math.abs(screenEntry.y)>1;
      entryMarker.style.opacity=String(markerOpacity);
      entryMarker.style.left=(50+50*screenEntry.x)+'%';
      entryMarker.style.top=(50-50*screenEntry.y)+'%';
    }
    renderer.domElement.dataset.camera=JSON.stringify({position:camera.position.toArray(),target:controls.target.toArray(),
      up:camera.up.toArray(),zoom:camera.zoom,...(camera instanceof THREE.PerspectiveCamera?{projection:'perspective',fov:camera.fov}:{projection:'orthographic',top:camera.top,right:camera.right})});
  };
  const queue=()=>{if(!frame && !disposed) frame=requestAnimationFrame(draw);};
  controls.addEventListener('change',queue);
  const updateProjection=()=>{
    const half=viewSize*.5;
    if(camera instanceof THREE.PerspectiveCamera){
      camera.aspect=aspect;camera.far=Math.max(1600,workingDistanceMm*4);
      camera.fov=microscopeFov({fieldMm:viewSize/Math.min(1,aspect),workingDistanceMm});
    }else{
      camera.top=half/Math.min(1,aspect);camera.bottom=-camera.top;
      camera.right=half*Math.max(1,aspect);camera.left=-camera.right;
    }
    camera.updateProjectionMatrix();
  };
  const resize=()=>{
    const w=Math.max(1,host.clientWidth), h=Math.max(1,host.clientHeight);
    aspect=w/h;updateProjection();
    renderer.setSize(w,h,false);queue();
  };
  const observer=new ResizeObserver(resize);observer.observe(host);
  const sourceFit=operativeSourceFit(context,plannedEntry);
  let nominalBearing:THREE.Vector3|undefined;
  function setCamera(center:THREE.Vector3,size:number,bearing:THREE.Vector3,reset=false){
    const direction=camera.position.clone().sub(controls.target).normalize();
    // Preserve the user's orbit while the nominal view changes. With no orbit
    // the pose is identical whether the user plays forward or seeks directly.
    if(reset||!nominalBearing||direction.distanceToSquared(nominalBearing)<1e-14)direction.copy(bearing);
    else if(!nominalBearing.equals(bearing)){
      const from=new THREE.Quaternion().setFromUnitVectors(camera.up,nominalBearing);
      const to=new THREE.Quaternion().setFromUnitVectors(camera.up,bearing);
      direction.applyQuaternion(to.multiply(from.invert()));
    }
    nominalBearing=bearing.clone();
    // The introductory wide field needs a longer stand-off to fit the near
    // side of the head. Keep the user's dolly ratio as the scope approaches.
    const dolly=reset?1:camera.position.distanceTo(controls.target)/workingDistanceMm;
    workingDistanceMm=Math.max(160,3*size);
    const optical=camera===opticalCamera;
    controls.minDistance=optical?workingDistanceMm*.48:0;
    controls.maxDistance=optical?workingDistanceMm*2.2:Infinity;
    const distance=optical?workingDistanceMm*dolly:400;
    controls.target.copy(center);camera.position.copy(center).addScaledVector(direction,distance);
    viewSize=size;
    keyLight.target.position.set(0,0,center.z);
    if(camera===inspectionCamera)keyLight.position.set(-range*1.2,-range*1.8,center.z+range*2.6);
    if(reset)camera.zoom=1;
    controls.update();updateProjection();queue();
  }
  const onKey=(event:KeyboardEvent)=>{
    if(event.key==='Enter'){pick(pickOperativeStructure(scene,camera,new THREE.Vector2(),field));}
    else if(event.key==='Home'&&nominalBearing)setCamera(controls.target,viewSize,nominalBearing,true);
    else if(['ArrowUp','ArrowDown','ArrowLeft','ArrowRight'].includes(event.key)) {
      const offset=camera.position.clone().sub(controls.target);
      const axis=event.key==='ArrowLeft'||event.key==='ArrowRight'?camera.up:new THREE.Vector3().crossVectors(camera.up,offset).normalize();
      offset.applyAxisAngle(axis,event.key==='ArrowDown'||event.key==='ArrowLeft'?-.13:.13);
      camera.position.copy(controls.target).add(offset);controls.update();
    } else if(['+','=','-'].includes(event.key)) {
      camera.zoom=THREE.MathUtils.clamp(camera.zoom*(event.key==='-'?.85:1/.85),controls.minZoom,controls.maxZoom);camera.updateProjectionMatrix();queue();
    } else return;
    event.preventDefault();
  };
  let down:THREE.Vector2|null=null;
  const pointerDown=(e:PointerEvent)=>{down=e.button===0?new THREE.Vector2(e.clientX,e.clientY):null};
  const pointerUp=(e:PointerEvent)=>{
    if(!down||Math.hypot(e.clientX-down.x,e.clientY-down.y)>4){down=null;return;}
    down=null;
    const r=renderer.domElement.getBoundingClientRect();
    const point=new THREE.Vector2((e.clientX-r.left)/r.width*2-1,1-(e.clientY-r.top)/r.height*2);
    pick(pickOperativeStructure(scene,camera,point,field));
  };
  renderer.domElement.addEventListener('pointerdown',pointerDown);
  renderer.domElement.addEventListener('pointerup',pointerUp);
  const contextLost=(event:Event)=>{event.preventDefault();lost();};
  renderer.domElement.addEventListener('webglcontextlost',contextLost);
  renderer.domElement.addEventListener('keydown',onKey);
  const brainZ=field.stations.brain??0, opening=model.opening;
  // Keep the two original presentations. Previously changing the section row
  // synchronously rebuilt the whole cortex even when revisiting the same view.
  const originalGeometry=new Map<string,THREE.BufferGeometry>();
  const retained=new Set<THREE.BufferGeometry>();
  function fieldGeometry(name:FieldLayer,state:string,row:number,raw:boolean){
    const sourceKey=name+'/'+row+'/'+raw;
    const original=state==='source'||state==='0';
    const cached=original&&originalGeometry.get(sourceKey);
    if(cached)return cached;
    const layer=model.layers[name];
    const surfaceKind=['artery','vein','sinus'].includes(name)?'vessel':name;
    const next=raw?operativeVoxelSurface(field,layer.mask,layer.bounds,row):operativeSurface(field,layer.mask,layer.bounds,row,surfaceKind);
    if(field.stations.brain!==null)installRetractionMorph(next,opening,brainZ);
    if(original){originalGeometry.set(sourceKey,next);retained.add(next);}
    return next;
  }
  function geometry(name:FieldLayer,state:string,cutaway:boolean,raw:boolean){
    const row=cutaway?model.firstSectionRow:0,key=state+'/'+row+'/'+raw;
    if(cache.get(name)===key)return;
    const mesh=meshes[name],next=fieldGeometry(name,state,row,raw);
    if(!retained.has(mesh.geometry))mesh.geometry.dispose();
    mesh.geometry=next;mesh.updateMorphTargets();cache.set(name,key);
  }
  function prepareSectionContext(){
    if(sectionContext!==undefined)return;
    sectionContext=operativeSectionContext(sourceVolume,collectionId,field,sectionPlaneY);
    if(sectionContext)scene.add(sectionContext);
  }
  let previousStep=-1,removedCells=0,duralProgramReady=false;
  let previousCutaway:boolean|undefined,previousView:OperativeViewChoice|undefined;
  function firstOperativePhase(retract:boolean){
    return (['bone','dura','brain','tumor'] as const).find(id=>field.stations[id]!==null&&(id!=='brain'||retract));
  }
  function update(props:Props){
    const {phase,progress,withRetraction,view}=props;
    const cutaway=view==='section';
    const pose=operativeCameraPose(field,model.opening,sourceFit,phase,progress,withRetraction,view);
    camera=view==='operative'?opticalCamera:inspectionCamera;
    controls.object=camera;
    markerOpacity=pose.marker;
    if(cover){
      const first=firstOperativePhase(withRetraction);
      const coverOpacity=phase==='overview'||view==='exterior'?1:phase===first?1-THREE.MathUtils.smootherstep(progress,.29,.43):0;
      cover.material.opacity=coverOpacity;cover.visible=!cutaway&&coverOpacity>.001;
      host.dataset.entryCaption=cover.visible?'Outer head · MRI estimate':'';
    }
    const removed=presentStep(props,cutaway);
    const flap=meshes.duraFlap;
    if(!duralProgramReady&&flap.geometry.morphAttributes.position?.length){
      // The two-sided fold has its own morph shader. Compile it while the
      // field opens, before the first reflected flap enters the color pass.
      renderer.compile(flap,camera,scene);
      duralProgramReady=true;
    }
    setCamera(pose.center,pose.size,pose.bearing,previousView!==view);
    previousView=view;
    return removed;
  }
  function presentStep({phase:next,progress,withRetraction,raw}:Props,cutaway:boolean){
    let activeTip:Omit<OperativeToolTip,'ras'>|null=null;
    if(previousCutaway!==cutaway){
      // Look down the saved approach at intact and opened layers; orbit remains available.
      if(cutaway)prepareSectionContext();
      if(sectionContext)sectionContext.visible=cutaway;
      host.dataset.sectionSource=cutaway&&sectionContext?JSON.stringify(sectionContext.userData.sectionSource):'';
      host.dataset.sectionCaption=cutaway&&sectionContext?sectionContext.name:'';
      exterior.value=!cutaway;context.visible=!cutaway;
      meshes.csf.material.opacity=cutaway?.42:.075;
      meshes.ventricles.material.opacity=cutaway?.42:.21;
      previousCutaway=cutaway;
    }
    sectionY.value=field.origin[1]+(cutaway?model.firstSectionRow-.5:-2)*field.spacingMm;
    const entering=sourceFit&&next===firstOperativePhase(withRetraction);
    const actionProgress=entering?clamp01((progress-entryApproachEnd)/(1-entryApproachEnd)):progress;
    const amounts={bone:phaseAmount(next,actionProgress,'bone'),dura:phaseAmount(next,actionProgress,'dura')};
    for(const kind of ['bone','dura'] as const){
      const {angle,fraction,lift,cutting}=openings[kind].update(cutaway?model.firstSectionRow:0,amounts[kind],raw);
      if(kind==='bone'){
        const [dx,dy,dz]=transferBone(lift);
        meshes.boneFlap.position.set(dx,dy,boneZ+dz);
      }else{
        meshes.duraFlap.position.set(hingeX,0,hingeZ);meshes.duraFlap.rotation.y=lift*duraReflectionAngle;
      }
      const tool=kind==='bone'?equipment.drill:equipment.scalpel;
      const radius=kind==='bone'?opening:model.duralOpening, x=radius*Math.cos(angle),y=radius*Math.sin(angle);
      const active=next===kind && cutting && (!entering||progress>=entryApproachEnd);
      const z=active?model.surfaceHeight(kind,x,y):null;tool.visible=z!==null;
      if(z!==null){tool.position.set(x,y,z);tool.rotation.set(.24,.21,angle,'ZXY');
        activeTip={tool:kind==='bone'?'Drill':'Scalpel',local:[x,y,z]};
        const bit=kind==='bone'?tool.getObjectByName('instrument:drill-bit'):null;if(bit)bit.rotation.z=fraction*Math.PI*48;
      }
    }

    const retract=withRetraction&&field.stations.brain!==null?phaseAmount(next,actionProgress,'brain'):0;
    const held=smooth((retract-.16)/.84);
    if(cortexWindow&&cortex?.morphTargetInfluences){
      cortexWindow.activeFloor.value=!raw&&!cutaway?cortexWindow.floor:cortexWindow.top+field.spacingMm;
      cortex.morphTargetInfluences[0]=!raw&&!cutaway?held:0;
      host.dataset.corticalSurface=!raw&&!cutaway?'registered-planning-surface':raw?'raw-field':'conditioned-section';
    }
    const approaching=entering&&progress<entryApproachEnd;
    equipment.blade.visible=!approaching&&withRetraction && field.stations.brain!==null && (next==='brain'||next==='tumor');
    equipment.blade.position.set(opening*(.16+.40*held),0,brainZ+opening*(.55*(1-smooth(retract/.16))-.31*held));
    equipment.blade.rotation.y=.32+.14*held;
    const tumor=phaseAmount(next,actionProgress,'tumor');
    const sweep=clamp01((tumor-.16)/.78);
    const pathProgress=sweepPattern==='spiral'?Math.round(sweep*120)/120:sweep;
    const step=Math.round(pathProgress*model.stepCount);
    if(step!==previousStep){removedCells=model.applyAspiration(step);previousStep=step;}
    for(const name of embeddedLayers){
      geometry(name,name==='tumor'?String(step):'source',cutaway,raw);
      const influence=meshes[name].morphTargetInfluences;
      if(influence?.length)influence[0]=held;
    }
    const tip=model.tipAt(step);
    equipment.cusa.visible=!approaching&&next==='tumor' && !!tip;
    if(tip){
      const travel=opening*.8*(1-smooth(tumor/.16))+opening*.32*smooth((tumor-.94)/.06);
      const [dx,,dz]=retractionOffset(...tip,opening,brainZ);
      equipment.cusa.position.set(tip[0]+held*dx,tip[1],tip[2]+travel+held*dz);equipment.cusa.rotation.set(.055,-.10,0);
      // The retraction offset only follows the morphed display; the sampled
      // labels stay in rest space, where the tip is the path cell plus travel.
      if(equipment.cusa.visible)activeTip={tool:'CUSA',local:[tip[0],tip[1],tip[2]+travel]};
    }
    return {removed:model.stepCount ? removedCells : null,tip:activeTip};
  }
  function dispose(){
    disposed=true;cancelAnimationFrame(frame);observer.disconnect();controls.dispose();
    renderer.domElement.removeEventListener('pointerdown',pointerDown);renderer.domElement.removeEventListener('pointerup',pointerUp);
    renderer.domElement.removeEventListener('webglcontextlost',contextLost);renderer.domElement.removeEventListener('keydown',onKey);
    const materials=new Set<THREE.Material>();
    openings.bone.dispose();openings.dura.dispose();meshes.grayMatter.customDepthMaterial?.dispose();
    scene.traverse(object=>{if(object instanceof THREE.Mesh){if(!retained.has(object.geometry))object.geometry.dispose();for(const m of Array.isArray(object.material)?object.material:[object.material]) materials.add(m);}});
    retained.forEach(item=>item.dispose());
    materials.forEach(m=>m.dispose());sectionContext?.material.map?.dispose();texture?.dispose();environment.dispose();keyLight.shadow.dispose();
    scene.clear();renderer.dispose();renderer.forceContextLoss();renderer.domElement.remove();
  }
  return {update,dispose,vesselDisplay:model.vesselDisplay,contextLabels:context.children.map(mesh=>[mesh.name,mesh.userData.sourceLayer.appearance?.modality].filter(Boolean).join(' · '))};
}

export function OperativeFieldCanvas(props:Props) {
  const host=useRef<HTMLDivElement>(null), scene=useRef<ReturnType<typeof fieldScene>|null>(null);
  const [error,setError]=useState('');
  const [picked,setPicked]=useState<OperativePick|null>(null);
  const [removed,setRemoved]=useState<number|null>(null);
  const [tip,setTip]=useState<Omit<OperativeToolTip,'ras'>|null>(null);
  const onToolTip=useRef(props.onToolTip);onToolTip.current=props.onToolTip;
  useEffect(()=>{
    if(!host.current)return;
    setError('');setPicked(null);
    try{
      scene.current=fieldScene(host.current,props.field,props.sourceVolume,props.collectionId,props.sweepPattern,props.visibleLayers,setPicked,()=>setError('The 3D graphics context was interrupted. Reopen the operative field to restore it.'));
      props.onVesselDisplay(scene.current.vesselDisplay);props.onContext(scene.current.contextLabels);
    }
    catch(cause){setError(cause instanceof Error?cause.message:'3D graphics could not be initialized.');}
    return ()=>{scene.current?.dispose();scene.current=null;};
  },[props.field,props.sourceVolume,props.collectionId,props.sweepPattern,props.visibleLayers,props.onVesselDisplay,props.onContext]);
  useEffect(()=>{if(scene.current){
    const state=scene.current.update(props);setRemoved(state.removed);
    setTip(previous=>previous&&state.tip&&previous.tool===state.tip.tool&&previous.local.every((v,i)=>Math.abs(v-state.tip!.local[i])<1e-6)?previous:state.tip);
  }},[props.field,props.sourceVolume,props.collectionId,props.sweepPattern,props.visibleLayers,props.phase,props.progress,props.withRetraction,props.view,props.raw]);
  const field=props.field;
  const labels=useMemo(()=>{
    const {grid,fromLocal}=fieldGrid(field),bits=field.materialBits,voxels=field.voxels;
    const target=(i:number)=>(voxels[i]&bits.tumor)!==0,vessel=(i:number)=>(voxels[i]&bits.vessels)!==0;
    return {grid,fromLocal,target,vessel,targetCells:labeledCells(voxels.length,target),vesselCells:labeledCells(voxels.length,vessel)};
  },[field]);
  const shownTip=error||props.phase==='overview'?null:tip;
  const navigation=useMemo(()=>{
    if(!shownTip||!labels.targetCells.length)return null;
    // Distances refer to the labels sampled in this local field: outside the
    // field no label is assumed, and the readout says so.
    const q=labels.fromLocal(shownTip.local),limit={complete:true,maxMm:Math.hypot(...field.shape)*field.spacingMm};
    const target=labelDistance(labels.grid,labels.target,q,{...limit,cells:labels.targetCells});
    const vessel=labels.vesselCells.length?labelDistance(labels.grid,labels.vessel,q,{...limit,cells:labels.vesselCells}):null;
    return target&&{tool:shownTip.tool,target,vessel};
  },[shownTip,labels,field]);
  useEffect(()=>{onToolTip.current?.(shownTip?{...shownTip,ras:fieldLocalToRas(field,shownTip.local)}:null);},[shownTip,field]);
  useEffect(()=>()=>onToolTip.current?.(null),[]);
  const targetLabel=field.layers.find(l=>l.id===field.targetLayerId)?.label??'Target';
  return <div ref={host} className="operative-field-canvas">
    <div className="operative-field-readouts">
    {!error&&navigation&&<aside className="operative-navigation" aria-label="Modeled tip distances" data-tip-tool={navigation.tool}
      data-target-distance-mm={(navigation.target.inside?-1:1)*navigation.target.distanceMm}
      title={`Signed distance from the modeled ${navigation.tool} tip to the boundary of the labels sampled in this local field (${field.spacingMm.toFixed(2)} mm grid); negative inside. Labels outside the field are not included. Geometric preview, not tracked navigation.`}>
      <strong>Modeled tip · {navigation.tool} · field labels</strong>
      <dl>
        <div data-inside={navigation.target.inside||undefined}><dt>{targetLabel}</dt><dd>{formatLabelDistance(navigation.target)}</dd></div>
        {navigation.vessel&&<div><dt>Vessel label</dt><dd>{formatLabelDistance(navigation.vessel)}</dd></div>}
      </dl>
    </aside>}
    {!error&&props.phase==='tumor'&&<span className="operative-cavity" data-removed-cells={removed??0}>{removed===null
      ?'No eligible lesion cells at this tip position'
      :'Modeled aspiration · '+(removed*props.field.spacingMm**3).toFixed(1)+' mm³ of local lesion label'}</span>}
    {picked&&!error&&<aside className="operative-picked" aria-label="Picked tissue provenance">
      <button aria-label="Close picked tissue" onClick={()=>setPicked(null)}>×</button>
      {picked.layers.map(l=><div key={l.id}><strong>{l.label}</strong><small>{l.evidence} · {l.reviewStatus?.replaceAll('_',' ')}</small>
        <small>Source {l.sourceSeries} · {l.coverageMeaning?.replaceAll('_',' ')??'Coverage unreported'}</small></div>)}
      {picked.layers.length>1&&<small>These layers share the sampled display class.</small>}
      <small>Rest position RAS {picked.ras.map(n=>n.toFixed(1)).join(', ')} mm</small>
      {picked.voxel&&<small>Scan {picked.series} · voxel {picked.voxel.join(', ')}</small>}
    </aside>}
    </div>
    {!error&&<span className="operative-entry-marker" hidden>Planned entry</span>}
    {!error&&<span className="operative-scale" aria-label="Physical scale"/>}
    {error&&<div role="alert">{error}</div>}
  </div>;
}

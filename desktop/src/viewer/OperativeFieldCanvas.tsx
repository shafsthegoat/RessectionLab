import { useEffect, useRef, useState } from 'react';
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { RoomEnvironment } from 'three/addons/environments/RoomEnvironment.js';
import type { OperativeField, OperativePhase } from '../operative-field';
import { buildOperativeModel, clamp01, layerNames, phaseAmount, smooth, type FieldLayer } from './operativeFieldModel.ts';
import { operativeSurface } from './operativeSurface.ts';
import { prepareOperativeOpening } from './operativeOpening.ts';
import { operativeInstruments } from './operativeInstruments.ts';
import { operativeSignalTexture,operativeTissueMaterial } from './operativeFieldAppearance.ts';

type Props = {field:OperativeField; phase:OperativePhase; progress:number; withRetraction:boolean; cutaway:boolean;
  onSuperficialVessels:(count:number)=>void};

function fieldScene(host:HTMLDivElement, field:OperativeField, lost:()=>void) {
  const renderer=new THREE.WebGLRenderer({antialias:true,alpha:true,powerPreference:'high-performance'});
  renderer.setClearColor(0,0); renderer.setPixelRatio(Math.min(window.devicePixelRatio||1,2));
  renderer.outputColorSpace=THREE.SRGBColorSpace;
  renderer.toneMapping=THREE.ACESFilmicToneMapping;
  renderer.shadowMap.enabled=true;
  renderer.shadowMap.type=THREE.PCFShadowMap;
  renderer.domElement.style.cssText='width:100%; height:100%; display:block; touch-action:none; outline-offset:-3px;';
  renderer.domElement.tabIndex=0;
  renderer.domElement.setAttribute('aria-label','3D operative field. Drag to orbit, scroll to zoom. Arrow keys rotate. Home resets the view.');
  host.appendChild(renderer.domElement);
  const model=buildOperativeModel(field), scene=new THREE.Scene(), camera=new THREE.OrthographicCamera(-1,1,1,-1,.1,1600);
  camera.up.set(0,0,1); const direction=new THREE.Vector3(.36,-.91,.90).normalize();
  camera.position.copy(direction).multiplyScalar(400);
  const controls=new OrbitControls(camera,renderer.domElement);
  controls.enablePan=false;controls.enableDamping=false;controls.minZoom=.45;controls.maxZoom=7;
  controls.minPolarAngle=.07;controls.maxPolarAngle=Math.PI*.92;controls.zoomSpeed=.85;
  const room=new RoomEnvironment(), pmrem=new THREE.PMREMGenerator(renderer);
  const environment=pmrem.fromScene(room,.06);
  room.dispose();pmrem.dispose();
  scene.environment=environment.texture;
  scene.environmentIntensity=.40;
  scene.add(new THREE.HemisphereLight('#fff7ed','#87747b',.64));
  const keyLight=new THREE.DirectionalLight('#fff2e4',2.45);
  const range=field.radiusMm*1.8;
  keyLight.castShadow=true;
  keyLight.shadow.mapSize.set(1024,1024);
  const shadowCamera=keyLight.shadow.camera;
  Object.assign(shadowCamera,{left:-range,right:range,top:range,bottom:-range,near:1,far:field.entryDepthMm+range*6});
  shadowCamera.updateProjectionMatrix();
  keyLight.shadow.normalBias=.09;
  keyLight.shadow.bias=-.00022;
  scene.add(keyLight,keyLight.target);
  const rim=new THREE.DirectionalLight('#d6e8ef',1.05);
  rim.position.set(.78,1,1.15);scene.add(rim);
  const hingeX=model.duralOpening*.74, boneZ=field.stations.bone??0;
  const hingeZ=model.surfaceHeight('dura',hingeX,0)??field.stations.dura??0;
  const texture=field.intensity?operativeSignalTexture(field.intensity):null;
  const sectionY={value:field.origin[1]-field.spacingMm*.5};
  const meshes=Object.fromEntries(layerNames.map(name=>{
    const offset=new THREE.Vector3();
    if(name==='duraFlap')offset.set(hingeX,0,hingeZ);
    if(name==='boneFlap')offset.z=boneZ;
    const material=operativeTissueMaterial(name,field,texture,offset,sectionY);
    const mesh=new THREE.Mesh(new THREE.BufferGeometry(),material);
    mesh.name=name;mesh.castShadow=true;mesh.receiveShadow=true;
    scene.add(mesh);return [name,mesh];
  })) as Record<FieldLayer,THREE.Mesh<THREE.BufferGeometry,THREE.MeshStandardMaterial>>;
  const openings={
    bone:prepareOperativeOpening('bone',model,meshes.bone,meshes.boneFlap,
      operativeTissueMaterial('bone',field,texture,new THREE.Vector3(),sectionY),new THREE.Vector3(0,0,boneZ)),
    dura:prepareOperativeOpening('dura',model,meshes.dura,meshes.duraFlap,
      operativeTissueMaterial('dura',field,texture,new THREE.Vector3(),sectionY),new THREE.Vector3(hingeX,0,hingeZ)),
  };
  const equipment=operativeInstruments(model.tipRadius,model.opening,field.entryDepthMm);
  Object.values(equipment).forEach(group=>{
    group.traverse(object=>{if(object instanceof THREE.Mesh)object.castShadow=true});
    scene.add(group);
  });
  const cache=new Map<FieldLayer,string>();
  let viewSize=field.radiusMm*2.8, aspect=1, frame=0, disposed=false;
  const draw=()=>{frame=0;if(!disposed) renderer.render(scene,camera);};
  const queue=()=>{if(!frame && !disposed) frame=requestAnimationFrame(draw);};
  controls.addEventListener('change',queue);
  const updateProjection=()=>{
    const half=viewSize*.5;
    camera.top=half/Math.min(1,aspect);camera.bottom=-camera.top;
    camera.right=half*Math.max(1,aspect);camera.left=-camera.right;camera.updateProjectionMatrix();
  };
  const resize=()=>{
    const w=Math.max(1,host.clientWidth), h=Math.max(1,host.clientHeight);
    aspect=w/h;updateProjection();
    renderer.setSize(w,h,false);queue();
  };
  const observer=new ResizeObserver(resize);observer.observe(host);
  const fieldTop=Math.max(0,...layerNames.map(name=>model.layers[name].bounds.high[2]).filter(v=>v>=0).map(z=>field.origin[2]+z*field.spacingMm));
  const fieldBottom=Math.min(fieldTop,...layerNames.map(name=>model.layers[name].bounds.low[2]).filter(Number.isFinite).map(z=>field.origin[2]+z*field.spacingMm));
  const overviewZ=(fieldTop+fieldBottom)*.5;
  const overviewSize=Math.max(field.radiusMm*3.15,(fieldTop-fieldBottom)*1.12+field.radiusMm*.95);
  const closeSize=Math.max(model.opening*4,field.radiusMm*1.82);
  function focus(z:number,size:number,reset=false) {
    if(!reset && z===controls.target.z && size===viewSize)return;
    const offset=reset?direction.clone().multiplyScalar(400):camera.position.clone().sub(controls.target);
    controls.target.set(0,0,z);camera.position.copy(controls.target).add(offset);
    viewSize=size;
    keyLight.target.position.set(0,0,z);
    keyLight.position.set(-range*1.2,-range*1.8,z+range*2.6);
    if (reset) camera.zoom=1;
    controls.update();updateProjection();queue();
  }
  const stageOrder=['bone','dura','brain','tumor'] as const;
  function followStation(phase:OperativePhase,progress:number,retraction:boolean){
    if(phase==='overview'){focus(overviewZ,overviewSize);return;}
    let previousZ=overviewZ,previousSize=overviewSize;
    for(const id of stageOrder){
      const z=field.stations[id];
      if(z===null || (id==='brain'&&!retraction))continue;
      if(id===phase){
        const t=smooth(progress/.24);
        focus(THREE.MathUtils.lerp(previousZ,z,t),THREE.MathUtils.lerp(previousSize,closeSize,t));
        return;
      }
      previousZ=z;previousSize=closeSize;
    }
  }
  const onKey=(event:KeyboardEvent)=>{
    if(event.key==='Home')focus(controls.target.z,viewSize,true);
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
  const contextLost=(event:Event)=>{event.preventDefault();lost();};
  renderer.domElement.addEventListener('webglcontextlost',contextLost);
  renderer.domElement.addEventListener('keydown',onKey);
  const brainZ=field.stations.brain??0, opening=model.opening;
  const brainOffset=(x:number,y:number,z:number)=>{
    const distance=((x-opening*.16)/(opening*.62))**2+(y/(opening*.55))**2+((z-brainZ)/(opening*.85))**2;
    return Math.exp(-distance*1.65);
  };
  function geometry(name:FieldLayer, state:string, cutaway:boolean) {
    const row=cutaway?model.firstSectionRow:0, key=`${state}/${row}`;
    if(cache.get(name)===key) return;
    const layer=model.layers[name], mesh=meshes[name];
    const next=operativeSurface(field,layer.mask,layer.bounds,row,name);
    const position=next.getAttribute('position');
    if(name==='brain' && position && field.stations.brain!==null) {
      const displaced=new Float32Array(position.count*3);
      for(let i=0;i<position.count;i++){
        const weight=brainOffset(position.getX(i),position.getY(i),position.getZ(i));
        displaced[i*3]=opening*.40*weight;displaced[i*3+2]=-opening*.31*weight;
      }
      next.morphAttributes.position=[new THREE.BufferAttribute(displaced,3)];next.morphTargetsRelative=true;
    }
    mesh.geometry.dispose();mesh.geometry=next;mesh.updateMorphTargets();cache.set(name,key);
  }
  let previousStep=-1,removedCells=0;
  function update({phase:next,progress,withRetraction,cutaway}:Props) {
    sectionY.value=field.origin[1]+((cutaway?model.firstSectionRow:0)-.5)*field.spacingMm;
    followStation(next,progress,withRetraction);
    const amounts={bone:phaseAmount(next,progress,'bone'),dura:phaseAmount(next,progress,'dura')};
    for(const kind of ['bone','dura'] as const){
      const {angle,fraction,lift,cutting}=openings[kind].update(cutaway?model.firstSectionRow:0,amounts[kind]);
      if(kind==='bone'){
        meshes.boneFlap.position.set(-opening*1.6*lift,opening*.22*lift,boneZ+opening*(.49*lift+.4*Math.sin(Math.PI*lift)));
        meshes.boneFlap.rotation.y=-.33*lift;
      }else{
        meshes.duraFlap.position.set(hingeX,0,hingeZ);meshes.duraFlap.rotation.y=lift*Math.PI*.84;
      }
      const tool=kind==='bone'?equipment.drill:equipment.scalpel;
      const radius=kind==='bone'?opening:model.duralOpening, x=radius*Math.cos(angle),y=radius*Math.sin(angle);
      const active=next===kind && cutting;
      const z=active?model.surfaceHeight(kind,x,y):null;tool.visible=z!==null;
      if(z!==null){tool.position.set(x,y,z);tool.rotation.set(.24,.21,angle,'ZXY');
        const bit=kind==='bone'?tool.getObjectByName('instrument:drill-bit'):null;if(bit)bit.rotation.z=fraction*Math.PI*48;
      }
    }
    geometry('brain','source',cutaway);geometry('vessel','source',cutaway);
    const retract=withRetraction&&field.stations.brain!==null?phaseAmount(next,progress,'brain'):0;
    const held=smooth((retract-.16)/.84), influence=meshes.brain.morphTargetInfluences;
    if(influence)influence[0]=held;
    equipment.blade.visible=withRetraction && field.stations.brain!==null && (next==='brain'||next==='tumor');
    equipment.blade.position.set(opening*(.16+.40*held),0,brainZ+opening*(.55*(1-smooth(retract/.16))-.31*held));
    equipment.blade.rotation.y=.32+.14*held;
    const tumor=phaseAmount(next,progress,'tumor');
    const sweep=clamp01((tumor-.16)/.78), step=Math.round(Math.round(sweep*120)/120*model.stepCount);
    if(step!==previousStep){removedCells=model.applyAspiration(step);previousStep=step;}
    geometry('tumor',String(step),cutaway);
    const tip=model.tipAt(step);
    equipment.cusa.visible=next==='tumor' && !!tip;
    if(tip){
      const travel=opening*.8*(1-smooth(tumor/.16))+opening*.32*smooth((tumor-.94)/.06);
      equipment.cusa.position.set(tip[0],tip[1],tip[2]+travel);equipment.cusa.rotation.set(.055,-.10,0);
    }
    queue();
    return model.stepCount ? removedCells : null;
  }
  function dispose(){
    disposed=true;cancelAnimationFrame(frame);observer.disconnect();controls.dispose();
    renderer.domElement.removeEventListener('webglcontextlost',contextLost);renderer.domElement.removeEventListener('keydown',onKey);
    const materials=new Set<THREE.Material>();
    openings.bone.dispose();openings.dura.dispose();
    scene.traverse(object=>{if(object instanceof THREE.Mesh){object.geometry.dispose();for(const m of Array.isArray(object.material)?object.material:[object.material]) materials.add(m);}});
    materials.forEach(m=>m.dispose());texture?.dispose();environment.dispose();keyLight.shadow.dispose();
    scene.clear();renderer.dispose();renderer.forceContextLoss();renderer.domElement.remove();
  }
  return {update,dispose,vesselDisplay:model.vesselDisplay};
}

export function OperativeFieldCanvas(props:Props) {
  const host=useRef<HTMLDivElement>(null), scene=useRef<ReturnType<typeof fieldScene>|null>(null);
  const [error,setError]=useState('');
  const [removed,setRemoved]=useState<number|null>(null);
  useEffect(()=>{
    if(!host.current)return;
    setError('');
    try{
      scene.current=fieldScene(host.current,props.field,()=>setError('The 3D graphics context was interrupted. Reopen the operative field to restore it.'));
      props.onSuperficialVessels(scene.current.vesselDisplay.superficialVoxels);
    }
    catch(cause){setError(cause instanceof Error?cause.message:'3D graphics could not be initialized.');}
    return ()=>{scene.current?.dispose();scene.current=null;};
  },[props.field,props.onSuperficialVessels]);
  useEffect(()=>{if(scene.current)setRemoved(scene.current.update(props));},[props.field,props.phase,props.progress,props.withRetraction,props.cutaway]);
  return <div ref={host} className="operative-field-canvas">
    {error&&<div role="alert">{error}</div>}
    {!error&&props.phase==='tumor'&&<span className="operative-cavity" data-removed-cells={removed??0}>{removed===null
      ?'No eligible lesion cells at this tip position'
      :'Modeled aspiration · '+(removed*props.field.spacingMm**3).toFixed(1)+' mm³ of local lesion label'}</span>}
  </div>;
}

import { useEffect, useRef, useState } from 'react';
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import type { OperativeField, OperativePhase } from '../operative-field';
import { buildOperativeModel, clamp01, layerNames, phaseAmount, smooth, type FieldLayer } from './operativeFieldModel.ts';
import { operativeSurface } from './operativeSurface.ts';
import { operativeInstruments } from './operativeInstruments.ts';

type Props = {field:OperativeField; phase:OperativePhase; progress:number; withRetraction:boolean; cutaway:boolean};
const colors: Record<FieldLayer,string> = {bone:'#ded2af',boneFlap:'#ded2af',dura:'#debb9e',duraFlap:'#debb9e',brain:'#cf9391',tumor:'#b778bb',vessel:'#d94d5a'};

function fieldScene(host:HTMLDivElement, field:OperativeField, lost:()=>void) {
  const renderer=new THREE.WebGLRenderer({antialias:true,alpha:true,powerPreference:'high-performance'});
  renderer.setClearColor(0,0); renderer.setPixelRatio(Math.min(window.devicePixelRatio,1.55));
  renderer.domElement.style.cssText='width:100%; height:100%; display:block; touch-action:none; outline-offset:-3px;';
  renderer.domElement.tabIndex=0;
  renderer.domElement.setAttribute('aria-label','3D operative field. Drag to orbit, scroll to zoom. Arrow keys rotate. Home resets the view.');
  host.appendChild(renderer.domElement);
  const model=buildOperativeModel(field), scene=new THREE.Scene(), camera=new THREE.OrthographicCamera(-1,1,1,-1,.1,1600);
  camera.up.set(0,0,1); const direction=new THREE.Vector3(.34,-.94,.74).normalize();
  camera.position.copy(direction).multiplyScalar(400);
  const controls=new OrbitControls(camera,renderer.domElement);
  controls.enablePan=false;controls.enableDamping=false;controls.minZoom=.45;controls.maxZoom=7;
  controls.minPolarAngle=.07;controls.maxPolarAngle=Math.PI*.92;controls.zoomSpeed=.85;
  scene.add(new THREE.HemisphereLight('#fff8ec','#626a7a',1.65));
  function light(color:string, intensity:number, position:number[]) {
    const l=new THREE.DirectionalLight(color,intensity);l.position.set(position[0],position[1],position[2]);scene.add(l);
  }
  light('#fff1d9',2.5,[-.7,-1,1.5]);light('#aac9df',1.35,[1,.35,.58]);
  const meshes=Object.fromEntries(layerNames.map(name=>{
    const material=new THREE.MeshStandardMaterial({color:colors[name],roughness:name==='vessel'?.38:name==='bone'||name==='boneFlap'?.83:.64,metalness:0});
    const mesh=new THREE.Mesh(new THREE.BufferGeometry(),material);mesh.name=name;scene.add(mesh);return [name,mesh];
  })) as Record<FieldLayer,THREE.Mesh<THREE.BufferGeometry,THREE.MeshStandardMaterial>>;
  const equipment=operativeInstruments(model.tipRadius,model.opening,field.entryDepthMm);
  Object.values(equipment).forEach(group=>scene.add(group));
  const cache=new Map<FieldLayer,string>();
  let phase:OperativePhase|null=null, viewSize=field.radiusMm*2.8, frame=0, disposed=false;
  const draw=()=>{frame=0;if(!disposed) renderer.render(scene,camera);};
  const queue=()=>{if(!frame && !disposed) frame=requestAnimationFrame(draw);};
  controls.addEventListener('change',queue);
  const resize=()=>{
    const w=Math.max(1,host.clientWidth), h=Math.max(1,host.clientHeight), aspect=w/h;
    const half=viewSize*.5;
    camera.top=half/Math.min(1,aspect);camera.bottom=-camera.top;
    camera.right=half*Math.max(1,aspect);camera.left=-camera.right;camera.updateProjectionMatrix();
    renderer.setSize(w,h,false);queue();
  };
  const observer=new ResizeObserver(resize);observer.observe(host);
  const fieldTop=Math.max(0,...layerNames.map(name=>model.layers[name].bounds.high[2]).filter(v=>v>=0).map(z=>field.origin[2]+z*field.spacingMm));
  const fieldBottom=Math.min(fieldTop,...layerNames.map(name=>model.layers[name].bounds.low[2]).filter(Number.isFinite).map(z=>field.origin[2]+z*field.spacingMm));
  function focus(next:OperativePhase, reset=false) {
    const z=next==='overview'?(fieldTop+fieldBottom)*.5:field.stations[next]??0;
    const offset=reset?direction.clone().multiplyScalar(400):camera.position.clone().sub(controls.target);
    controls.target.set(0,0,z);camera.position.copy(controls.target).add(offset);
    viewSize=next==='overview'?Math.max(field.radiusMm*3.15,(fieldTop-fieldBottom)*1.12+field.radiusMm*.95):Math.max(model.opening*3.6,field.radiusMm*1.66);
    if (reset) camera.zoom=1;
    controls.update();resize();
  }
  const onKey=(event:KeyboardEvent)=>{
    if(event.key==='Home')focus(phase??'overview',true);
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
  const hingeX=model.duralOpening*.74, boneZ=field.stations.bone??0;
  const hingeZ=model.surfaceHeight('dura',hingeX,0)??field.stations.dura??0;
  const brainZ=field.stations.brain??0, opening=model.opening;
  const brainOffset=(x:number,y:number,z:number)=>{
    const distance=((x-opening*.16)/(opening*.62))**2+(y/(opening*.55))**2+((z-brainZ)/(opening*.85))**2;
    return Math.exp(-distance*1.65);
  };
  function geometry(name:FieldLayer, state:string, cutaway:boolean) {
    const row=cutaway&&name!=='vessel'?model.firstSectionRow:0, key=`${state}/${row}`;
    if(cache.get(name)===key) return;
    const layer=model.layers[name], mesh=meshes[name];
    const next=operativeSurface(field,layer.mask,layer.bounds,row);
    if(name==='boneFlap')next.translate(0,0,-boneZ);
    if(name==='duraFlap')next.translate(-hingeX,0,-hingeZ);
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
  const openKeys={bone:'',dura:''}; let previousStep=-1,removedCells=0;
  function update({phase:next,progress,withRetraction,cutaway}:Props) {
    if(next!==phase){phase=next;focus(next);}
    const amounts={bone:phaseAmount(next,progress,'bone'),dura:phaseAmount(next,progress,'dura')};
    for(const kind of ['bone','dura'] as const){
      const flap=kind==='bone'?'boneFlap':'duraFlap';
      const {cut,lift}=model.openingState(kind,field.stations[kind]===null?0:amounts[kind]);
      const key=`${cut}/${lift>0}`;
      if(openKeys[kind]!==key){model.applyOpening(kind,cut,lift>0);openKeys[kind]=key;}
      geometry(kind,key,cutaway);meshes[flap].visible=lift>0;
      if(lift>0)geometry(flap,'source',cutaway);
      if(kind==='bone'){
        meshes.boneFlap.position.set(-opening*1.6*lift,opening*.22*lift,boneZ+opening*(.49*lift+.4*Math.sin(Math.PI*lift)));
        meshes.boneFlap.rotation.y=-.33*lift;
      }else{
        meshes.duraFlap.position.set(hingeX,0,hingeZ);meshes.duraFlap.rotation.y=lift*Math.PI*.84;
      }
      const tool=kind==='bone'?equipment.drill:equipment.scalpel;
      const cutoff=kind==='bone'?.68:.50, fraction=clamp01(progress/cutoff);
      const angle=kind==='bone'?fraction*2*Math.PI:.70+fraction*(2*Math.PI-1.40);
      const radius=kind==='bone'?opening:model.duralOpening, x=radius*Math.cos(angle),y=radius*Math.sin(angle);
      const active=next===kind && progress<=cutoff;
      const z=active?model.surfaceHeight(kind,x,y):null;tool.visible=z!==null;
      if(z!==null){tool.position.set(x,y,z);tool.rotation.set(.24,.21,angle,'ZXY');
        const bit=kind==='bone'?tool.getObjectByName('instrument:drill-bit'):null;if(bit)bit.rotation.z=fraction*Math.PI*48;
      }
    }
    geometry('brain','source',cutaway);geometry('vessel','source',false);
    // Keep the entire vascular course visible through the inspection section
    // while allowing a foreground vessel to reveal the tip and cavity behind it.
    const vesselMaterial=meshes.vessel.material;
    if(vesselMaterial.transparent!==cutaway){
      vesselMaterial.transparent=cutaway;
      vesselMaterial.opacity=cutaway ? 0.38 : 1;
      vesselMaterial.depthWrite=!cutaway;
      vesselMaterial.needsUpdate=true;
    }
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
    scene.traverse(object=>{if(object instanceof THREE.Mesh){object.geometry.dispose();for(const m of Array.isArray(object.material)?object.material:[object.material]) materials.add(m);}});
    materials.forEach(m=>m.dispose());scene.clear();renderer.dispose();renderer.forceContextLoss();renderer.domElement.remove();
  }
  return {update,dispose};
}

export function OperativeFieldCanvas(props:Props) {
  const host=useRef<HTMLDivElement>(null), scene=useRef<ReturnType<typeof fieldScene>|null>(null);
  const [error,setError]=useState('');
  const [removed,setRemoved]=useState<number|null>(null);
  useEffect(()=>{
    if(!host.current)return;
    setError('');
    try{scene.current=fieldScene(host.current,props.field,()=>setError('The 3D graphics context was interrupted. Reopen the operative field to restore it.'));}
    catch(cause){setError(cause instanceof Error?cause.message:'3D graphics could not be initialized.');}
    return ()=>{scene.current?.dispose();scene.current=null;};
  },[props.field]);
  useEffect(()=>{if(scene.current)setRemoved(scene.current.update(props));},[props.field,props.phase,props.progress,props.withRetraction,props.cutaway]);
  return <div ref={host} className="operative-field-canvas">
    {error&&<div role="alert">{error}</div>}
    {!error&&props.phase==='tumor'&&<span className="operative-cavity" data-removed-cells={removed??0}>{removed===null
      ?'No eligible lesion cells at this tip position'
      :'Modeled aspiration · '+(removed*props.field.spacingMm**3).toFixed(1)+' mm³ of local lesion label'}</span>}
  </div>;
}

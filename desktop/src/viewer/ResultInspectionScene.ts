import * as THREE from 'three';
import {OrbitControls} from 'three/addons/controls/OrbitControls.js';
import type {SurgicalView} from '../surgical-types';
import type {ViewerVolume} from './contracts';
import type {ViewerCameraState,ViewerWorkspaceSettings} from './workspaceSettings';
import {resultMaskCrop,resultMaskHash,resultMembership} from './resultInspectionData.ts';
import {prepareSurfaceSoup} from './prepareSurface.ts';
import type {PreparedSurface} from './prepareSurface.ts';
import {maskSurface} from './surface.ts';
import {SURFACE_POLICY} from './surfacePolicy.ts';
import {placeInSourceFrame} from './sceneGeometry.ts';

type Parts=NonNullable<ViewerWorkspaceSettings['resultInspection']>;
// Keep the topology audit for larger cuts off the orbit and replay thread.
const INLINE_CAVITY_TRIANGLES=4096;
export class ResultInspectionScene {
  private scene=new THREE.Scene();
  private camera=new THREE.PerspectiveCamera(34,1,.05,2500);
  private controls:OrbitControls;
  private residual=new THREE.Group();
  private cavity=new THREE.Mesh(new THREE.BufferGeometry(),new THREE.MeshStandardMaterial({color:0xe5a456,roughness:.54,side:THREE.DoubleSide}));
  // Only the cavity silhouette behind retained tissue gets a muted ink. Draw this
  // depth-tested pass before the normally lit cast so exposed surfaces stay solid.
  private buriedCast=new THREE.Mesh(this.cavity.geometry,new THREE.MeshBasicMaterial({
    color:0xa58158,depthFunc:THREE.GreaterDepth,depthWrite:false,
  }));
  private tissue=new THREE.MeshStandardMaterial({color:0x499ca6,roughness:.78,side:THREE.DoubleSide});
  private lastCavity?:{sceneId:string;indices:Uint32Array};
  private conditioning?:Worker;
  private settling?:ReturnType<typeof setTimeout>;
  private volume:ViewerVolume|null=null;
  private parts:Parts='both';
  private followFit=true;
  private fitNeeded=true;
  private aspect=0;
  private cavityCount=0;
  private residualCount=0;
  private cavityBounds=new THREE.Box3();
  private residualBounds=new THREE.Box3();
  private consumedPose:ViewerCameraState|null|undefined;
  readonly element:HTMLElement;
  private renderAgain:()=>void;
  private saveCamera:(camera:ViewerCameraState)=>void;
  constructor(element:HTMLElement,renderAgain:()=>void,saveCamera:(camera:ViewerCameraState)=>void){
    this.element=element;
    this.renderAgain=renderAgain;
    this.saveCamera=saveCamera;
    this.scene.background=new THREE.Color(0x10191c);
    this.buriedCast.renderOrder=1;
    this.cavity.renderOrder=2;
    this.scene.add(this.buriedCast,this.cavity,this.residual,new THREE.HemisphereLight(0xfff7e8,0x7090a0,1.5));
    const key=new THREE.DirectionalLight(0xfff0db,2.35);key.position.set(-.4,-.65,1);this.scene.add(key);
    const rim=new THREE.DirectionalLight(0xc9efff,1.1);rim.position.set(.7,1,.2);this.scene.add(rim);
    this.camera.up.set(0,0,1);
    this.controls=new OrbitControls(this.camera,element);
    this.controls.minDistance=.4;this.controls.maxDistance=1600;
    // Keep every interactive orbit inside the persisted camera's non-polar range.
    this.controls.minPolarAngle=.017;this.controls.maxPolarAngle=Math.PI-.017;
    this.controls.addEventListener('change',this.renderAgain);
    this.controls.addEventListener('end',()=>{this.followFit=false;this.saveCamera(this.pose());});
  }
  private pose():ViewerCameraState {
    return {position:this.camera.position.toArray(),target:this.controls.target.toArray(),up:this.camera.up.toArray()};
  }
  configure(parts:Parts,pose:ViewerCameraState|null|undefined) {
    if(this.parts!==parts&&!pose){this.followFit=true;this.fitNeeded=true;}
    this.parts=parts;
    if(pose&&pose!==this.consumedPose){
      this.followFit=false;this.fitNeeded=false;
      if(!this.camera.position.equals(new THREE.Vector3(...pose.position))||!this.controls.target.equals(new THREE.Vector3(...pose.target))||!this.camera.up.equals(new THREE.Vector3(...pose.up))){
        this.camera.up.set(...pose.up);this.camera.position.set(...pose.position);
        this.controls.target.set(...pose.target);this.controls.update();
      }
    }
    this.consumedPose=pose;
    this.buriedCast.visible=parts==='both'&&this.cavityCount>0&&this.residualCount>0;
    this.cavity.visible=parts!=='residual'&&this.cavityCount>0;
    this.residual.visible=parts!=='cavity';
  }
  requestFit(){this.fitNeeded=true;this.followFit=true;}
  private frame(){
    const {cavityBounds,residualBounds}=this;
    const bounds=new THREE.Box3();
    if(this.cavity.visible)bounds.union(cavityBounds);
    if(this.residual.visible)bounds.union(residualBounds);
    if(bounds.isEmpty())return;
    const center=bounds.getCenter(new THREE.Vector3());
    const direction=new THREE.Vector3(1,-1,.65);
    // See the long access cavity in profile with the target beyond its end.
    if(!cavityBounds.isEmpty()&&!residualBounds.isEmpty()){
      const axis=cavityBounds.getCenter(new THREE.Vector3()).sub(residualBounds.getCenter(new THREE.Vector3()));
      if(axis.lengthSq()>1){
        axis.normalize();const up=Math.abs(axis.z)<.83?new THREE.Vector3(0,0,1):new THREE.Vector3(0,1,0);
        direction.crossVectors(axis,up).addScaledVector(up,.42).addScaledVector(axis,.15);
      }
    }
    direction.normalize();
    const right=new THREE.Vector3().crossVectors(this.camera.up,direction).normalize();
    const up=new THREE.Vector3().crossVectors(direction,right);
    const tanY=Math.tan(THREE.MathUtils.degToRad(this.camera.fov/2)),tanX=tanY*this.camera.aspect;
    // Fit the projected corners. A diagonal bounding sphere wastes most of the
    // vertical field for an elongated cast, particularly in the narrow layout.
    let distance=2;
    for(const x of [bounds.min.x,bounds.max.x])for(const y of [bounds.min.y,bounds.max.y])for(const z of [bounds.min.z,bounds.max.z]){
      const offset=new THREE.Vector3(x,y,z).sub(center);
      const extent=Math.max(Math.abs(offset.dot(right))/tanX,Math.abs(offset.dot(up))/tanY);
      distance=Math.max(distance,offset.dot(direction)+1.18*extent);
    }
    this.controls.target.copy(center);this.camera.position.copy(center).addScaledVector(direction,distance);
    this.controls.update();this.fitNeeded=false;
  }

  render(renderer:THREE.WebGLRenderer,volume:ViewerVolume,view:SurgicalView,surfaces:Map<string,THREE.Mesh>,width:number,height:number) {
    const data=this.element.dataset;
    if(volume!==this.volume){
      const cells=resultMembership(volume,view);
      this.residual.clear();
      // Borrow the registered current target meshes. Their owner retains the
      // geometry and releases it through the normal engine frame lifecycle.
      for(const name of cells.targetNames){
        const source=surfaces.get(name);if(!source)continue;
        const mesh=new THREE.Mesh(source.geometry,this.tissue);placeInSourceFrame(mesh,view.scene.affine);
        this.residual.add(mesh);
      }
      // Many observed samples move the tool without editing a single cell.
      // Retain just the most recent cast and check exact membership before extraction.
      const previous=this.lastCavity;
      if(previous?.sceneId!==view.scene.sceneId||previous.indices.length!==cells.cavity.length||
          !cells.cavity.every((value,index)=>previous.indices[index]===value)){
        this.installCavity(resultMaskCrop(cells.cavity,view));
        this.lastCavity={sceneId:view.scene.sceneId,indices:cells.cavity};
      }
      data.resultCavityCells=String(this.cavityCount=cells.cavity.length);
      data.resultResidualCells=String(this.residualCount=cells.residual.length);
      data.resultCavityHash=resultMaskHash(cells.cavity);data.resultResidualHash=resultMaskHash(cells.residual);
      data.resultResidualTriangles=String(this.residual.children.reduce((n,m)=>n+(((m as THREE.Mesh).geometry.index?.count??(m as THREE.Mesh).geometry.getAttribute('position')?.count??0)/3),0));
      data.resultCavityTriangles=String((this.cavity.geometry.index?.count??0)/3);
      // Stable geometry can retain unused historical vertices. Fit only the
      // triangle-indexed vertices actually submitted for this result.
      this.residualBounds=drawnBounds(this.residual);
      data.resultCavityBounds=rasBounds(this.cavityBounds);data.resultResidualBounds=rasBounds(this.residualBounds);
      this.volume=volume;this.configure(this.parts,this.consumedPose);
      if(this.followFit)this.fitNeeded=true;
    }
    this.camera.aspect=width/height;this.camera.updateProjectionMatrix();
    if(this.followFit&&this.aspect!==this.camera.aspect)this.fitNeeded=true;
    this.aspect=this.camera.aspect;
    if(this.fitNeeded)this.frame();
    renderer.render(this.scene,this.camera);
    data.resultCamera=JSON.stringify(this.pose());data.resultParts=this.parts;
    data.resultState=view.stateId;
    data.resultEmpty=String(!(this.cavity.visible&&this.cavityCount||this.residual.visible&&this.residualCount));
  }
  private installCavity(crop:ReturnType<typeof resultMaskCrop>){
    const data=this.element.dataset;
    this.cancelConditioning();
    this.cavity.geometry.dispose();
    const geometry=new THREE.BufferGeometry();
    this.cavity.geometry=this.buriedCast.geometry=geometry;
    this.cavityBounds.makeEmpty();
    data.resultCavitySourceBounds='null';
    data.resultCavityDisplacementMm='0';
    if(!crop)return;
    const soup=maskSurface(crop.mask,crop.size);
    const triangles=soup.length/9;
    const inline=triangles<=INLINE_CAVITY_TRIANGLES;
    const surface=prepareSurfaceSoup(soup,crop.mask,crop.size,crop.affine,'cavity',inline);
    geometry.setIndex(new THREE.BufferAttribute(surface.indices,1));
    geometry.setAttribute('position',new THREE.BufferAttribute(surface.positions,3));
    placeInSourceFrame(this.cavity,crop.affine);
    placeInSourceFrame(this.buriedCast,crop.affine);
    data.resultCavitySourceBounds=rasBounds(drawnBounds(this.cavity));
    const useSurface=(prepared:PreparedSurface)=>{
      geometry.setAttribute('position',new THREE.BufferAttribute(prepared.smoothPositions??prepared.positions,3));
      geometry.setAttribute('normal',new THREE.BufferAttribute(prepared.smoothNormals??prepared.normals,3));
      geometry.computeBoundingSphere();
      this.cavityBounds=drawnBounds(this.cavity);
      data.resultCavityBounds=rasBounds(this.cavityBounds);
      data.resultCavityDisplacementMm=String(prepared.diagnostics?.maxDisplacementMm??0);
    };
    useSurface(surface);
    if(inline||triangles>SURFACE_POLICY.maxContextTriangles)return;
    // Show current raw geometry throughout a seek. Condition the latest
    // settled edit and retire the worker if that membership changes.
    this.settling=setTimeout(()=>{
      const worker=new Worker(new URL('./surface.worker.ts',import.meta.url),{type:'module'});
      this.conditioning=worker;
      worker.onmessage=(event:MessageEvent<PreparedSurface&{error?:string}>)=>{
        if(this.conditioning!==worker)return;
        if(!event.data.error){
          // Dispose the uploaded attributes before replacing their buffers.
          geometry.dispose();
          useSurface(event.data);
          if(this.followFit)this.fitNeeded=true;
          this.renderAgain();
        }
        this.cancelConditioning();
      };
      worker.onerror=()=>{if(this.conditioning===worker)this.cancelConditioning();};
      worker.postMessage({name:'result-cavity',kind:'cavity',mask:crop.mask,shape:crop.size,
        affine:crop.affine,displaySmoothing:true},[crop.mask.buffer]);
    },140);
  }
  private cancelConditioning(){
    clearTimeout(this.settling);
    this.conditioning?.terminate();this.conditioning=undefined;
  }
  dispose(){
    this.cancelConditioning();
    this.controls.dispose();this.residual.clear();this.cavity.geometry.dispose();
    this.cavity.material.dispose();this.buriedCast.material.dispose();this.tissue.dispose();
  }
}

function drawnBounds(root:THREE.Object3D):THREE.Box3 {
  const bounds=new THREE.Box3(),point=new THREE.Vector3();
  root.updateWorldMatrix(true,true);
  root.traverse(object=>{
    if(!(object instanceof THREE.Mesh))return;
    const position=object.geometry.getAttribute('position');if(!position)return;
    const index=object.geometry.index;
    for(let k=0;k<(index?.count??position.count);k++){
      point.fromBufferAttribute(position,index?index.getX(k):k).applyMatrix4(object.matrixWorld);
      bounds.expandByPoint(point);
    }
  });
  return bounds;
}

function rasBounds(bounds:THREE.Box3){
  return JSON.stringify(bounds.isEmpty()?null:[bounds.min.toArray(),bounds.max.toArray()]);
}

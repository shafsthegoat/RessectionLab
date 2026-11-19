import * as THREE from 'three';
import {OrbitControls} from 'three/addons/controls/OrbitControls.js';
import type {SurgicalView} from '../surgical-types';
import type {ViewerVolume} from './contracts';
import type {ViewerCameraState,ViewerWorkspaceSettings} from './workspaceSettings';
import {resultMaskCrop,resultMaskHash,resultMembership} from './resultInspectionData.ts';
import {prepareSurface} from './prepareSurface.ts';
import {placeInSourceFrame} from './sceneGeometry.ts';

type Parts=NonNullable<ViewerWorkspaceSettings['resultInspection']>;
export class ResultInspectionScene {
  private scene=new THREE.Scene();
  private camera=new THREE.PerspectiveCamera(34,1,.05,2500);
  private controls:OrbitControls;
  private residual=new THREE.Group();
  private cavity=new THREE.Mesh(new THREE.BufferGeometry(),new THREE.MeshStandardMaterial({color:0xe5a456,roughness:.54,side:THREE.DoubleSide}));
  private tissue=new THREE.MeshStandardMaterial({color:0x53c6d1,roughness:.58,side:THREE.DoubleSide});
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
  constructor(element:HTMLElement,render:()=>void,private saveCamera:(camera:ViewerCameraState)=>void){
    this.element=element;
    this.scene.background=new THREE.Color(0x10191c);
    this.scene.add(this.cavity,this.residual,new THREE.HemisphereLight(0xfff7e8,0x7090a0,1.5));
    const key=new THREE.DirectionalLight(0xfff0db,2.35);key.position.set(-.4,-.65,1);this.scene.add(key);
    const rim=new THREE.DirectionalLight(0xc9efff,1.1);rim.position.set(.7,1,.2);this.scene.add(rim);
    this.camera.up.set(0,0,1);
    this.controls=new OrbitControls(this.camera,element);
    this.controls.minDistance=.4;this.controls.maxDistance=1600;
    // Keep every interactive orbit inside the persisted camera's non-polar range.
    this.controls.minPolarAngle=.017;this.controls.maxPolarAngle=Math.PI-.017;
    this.controls.addEventListener('change',render);
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
    // The cavity can be entirely enclosed within a larger residual target.
    // Keep it visible through the retained target shell in the combined view.
    this.tissue.transparent=parts==='both'&&this.cavityCount>0;
    this.tissue.opacity=this.tissue.transparent?.48:1;
    this.tissue.depthWrite=!this.tissue.transparent;
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
    const fov=Math.atan(Math.tan(THREE.MathUtils.degToRad(this.camera.fov/2))*Math.min(1,this.camera.aspect));
    const distance=Math.max(2,bounds.getSize(new THREE.Vector3()).length()*.56/Math.sin(fov));
    this.controls.target.copy(center);this.camera.position.copy(center).addScaledVector(direction.normalize(),distance);
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
      const crop=resultMaskCrop(cells.cavity,view);
      this.cavity.geometry.dispose();this.cavity.geometry=new THREE.BufferGeometry();
      if(crop){
        const surface=prepareSurface(crop.mask,crop.size,crop.affine,'cavity',false);
        this.cavity.geometry.setAttribute('position',new THREE.BufferAttribute(surface.positions,3));
        this.cavity.geometry.setAttribute('normal',new THREE.BufferAttribute(surface.normals,3));
        this.cavity.geometry.setIndex(new THREE.BufferAttribute(surface.indices,1));
        placeInSourceFrame(this.cavity,crop.affine);
      }
      data.resultCavityCells=String(this.cavityCount=cells.cavity.length);
      data.resultResidualCells=String(this.residualCount=cells.residual.length);
      data.resultCavityHash=resultMaskHash(cells.cavity);data.resultResidualHash=resultMaskHash(cells.residual);
      data.resultResidualTriangles=String(this.residual.children.reduce((n,m)=>n+(((m as THREE.Mesh).geometry.index?.count??(m as THREE.Mesh).geometry.getAttribute('position')?.count??0)/3),0));
      data.resultCavityTriangles=String((this.cavity.geometry.index?.count??0)/3);
      // Stable geometry can retain unused historical vertices. Fit only the
      // triangle-indexed vertices actually submitted for this result.
      this.cavityBounds=drawnBounds(this.cavity);
      this.residualBounds=drawnBounds(this.residual);
      const rasBounds=(b:THREE.Box3)=>JSON.stringify(b.isEmpty()?null:[b.min.toArray(),b.max.toArray()]);
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
  dispose(){this.controls.dispose();this.residual.clear();this.cavity.geometry.dispose();this.cavity.material.dispose();this.tissue.dispose();}
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

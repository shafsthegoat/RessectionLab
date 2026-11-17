import * as THREE from 'three';
import type {PositionEffects} from '../surgical-types';
import {paneViewport} from './layout';
import type {PaneBounds} from './layout';
import {orientationMannequin} from './orientationMannequin';

type Rotation3 = [[number,number,number],[number,number,number],[number,number,number]];
const POSTURES:Record<string,string>={supine:'Supine',prone:'Prone',left_lateral:'Left lateral',right_lateral:'Right lateral',park_bench:'Park-bench',semisitting:'Semisitting'};
function rotation(value:unknown):THREE.Matrix4|null {
  if(!Array.isArray(value)||value.length!==3||value.some(row=>!Array.isArray(row)||row.length!==3||row.some(n=>typeof n!=='number'||!Number.isFinite(n))))return null;
  const r=value as Rotation3,m=new THREE.Matrix4().set(r[0][0],r[0][1],r[0][2],0,r[1][0],r[1][1],r[1][2],0,r[2][0],r[2][1],r[2][2],0,0,0,0,1);
  const product=m.clone().transpose().multiply(m),identity=new THREE.Matrix4();
  return Math.abs(m.determinant()-1)<1e-5&&product.elements.every((n,i)=>Math.abs(n-identity.elements[i])<1e-5)?m:null;
}
/** Image/head RAS stays fixed. A room pose must not be applied twice to it. */
export function orientationPose(effects:PositionEffects|null|undefined){
  const body=rotation(effects?.bodyRasToRoom),head=rotation(effects?.headRasToRoom);
  if(!body||!head||!effects||!Object.hasOwn(POSTURES,effects.preset))return null;
  const roomToHead=head.transpose();
  return {bodyToHead:roomToHead.clone().multiply(body),down:new THREE.Vector3(0,-1,0).transformDirection(roomToHead),label:POSTURES[effects.preset]};
}

/** A scissored overlay in the existing drawing buffer, never a second canvas. */
export class OrientationGuide {
  readonly scene=new THREE.Scene();
  readonly camera=new THREE.OrthographicCamera(-1,1,1,-1,.1,10);
  readonly mannequin=orientationMannequin();
  private readonly light=new THREE.DirectionalLight(0xffffff,2.6);
  private readonly centre=new THREE.Vector3();
  private readonly sphere=new THREE.Sphere();
  private readonly projected=new THREE.Vector3();
  private readonly cameraRotation=new THREE.Quaternion();
  private readonly bodyRotation=new THREE.Matrix4();
  private readonly oldViewport=new THREE.Vector4();
  private readonly oldScissor=new THREE.Vector4();
  private down:THREE.Vector3|null=null;
  private poseSignature='';
  private label='Anatomical reference';
  private readonly sides:HTMLElement[];
  private readonly caption:HTMLElement|null;
  private readonly gravity:HTMLElement|null;
  private readonly arrow:SVGLineElement|null;
  private readonly arrowhead:SVGPathElement|null;
  private readonly axial:HTMLElement|null;
  constructor(private readonly element:HTMLElement){
    this.scene.add(this.mannequin.model,new THREE.AmbientLight(0xffffff,1.2),this.light,this.light.target);
    this.sides=Array.from(element.querySelectorAll<HTMLElement>('[data-guide-side]'));
    this.caption=element.parentElement?.querySelector('[data-guide-caption]')??null;
    this.gravity=element.querySelector('[data-guide-gravity]');
    this.arrow=element.querySelector('[data-guide-arrow]');this.arrowhead=element.querySelector('[data-guide-arrowhead]');
    this.axial=element.querySelector('[data-guide-axial]');
    this.fit();
  }
  setPosture(effects:PositionEffects|null|undefined):void {
    const signature=JSON.stringify([effects?.preset,effects?.bodyRasToRoom,effects?.headRasToRoom]);
    if(signature===this.poseSignature)return;
    this.poseSignature=signature;
    const pose=orientationPose(effects);
    this.bodyRotation.copy(pose?.bodyToHead??new THREE.Matrix4());
    this.mannequin.body.quaternion.setFromRotationMatrix(this.bodyRotation);
    this.down=pose?.down??null;this.label=pose?.label??'Anatomical reference';
    this.fit();
    if(this.caption)this.caption.textContent=this.label;
    if(this.gravity)this.gravity.hidden=!this.down;
    this.element.setAttribute('aria-label',`Body orientation guide. Mint head follows the brain. ${this.label}${pose?', recorded body and head orientation. Down indicates room gravity.':'.'}`);
  }
  private fit(){
    this.mannequin.model.updateMatrixWorld(true);
    new THREE.Box3().setFromObject(this.mannequin.model).getBoundingSphere(this.sphere);
    this.centre.copy(this.sphere.center);
  }
  /** Separate projection ignores orbit target, pan, perspective FOV and zoom. */
  syncCamera(main:THREE.Camera,width:number,height:number):void {
    main.getWorldQuaternion(this.cameraRotation);
    this.camera.quaternion.copy(this.cameraRotation);
    this.camera.position.set(0,0,4).applyQuaternion(this.cameraRotation).add(this.centre);
    const halfHeight=this.sphere.radius*1.13*Math.max(1,height/width),halfWidth=halfHeight*width/height;
    this.camera.left=-halfWidth;this.camera.right=halfWidth;this.camera.top=halfHeight;this.camera.bottom=-halfHeight;
    this.camera.updateProjectionMatrix();this.camera.updateMatrixWorld(true);
    this.light.position.set(-3,4,5).applyQuaternion(this.cameraRotation).add(this.centre);
    this.light.target.position.copy(this.centre);
    for(const side of this.sides){
      this.projected.set(side.dataset.guideSide==='right'?.255:-.255,0,-.16).applyMatrix4(this.bodyRotation).project(this.camera);
      side.style.left=`${(this.projected.x+1)*width/2}px`;side.style.top=`${(1-this.projected.y)*height/2}px`;
    }
    if(this.down&&this.gravity){
      this.projected.copy(this.down).applyQuaternion(this.cameraRotation.clone().invert());
      const axial=Math.abs(this.projected.z)>.95,x=this.projected.x*9,y=-this.projected.y*9,angle=Math.atan2(y,x),ex=12+x,ey=12+y;
      if(this.arrow){this.arrow.style.display=axial?'none':'';this.arrow.setAttribute('x2',String(ex));this.arrow.setAttribute('y2',String(ey));}
      if(this.arrowhead){this.arrowhead.style.display=axial?'none':'';this.arrowhead.setAttribute('d',`M ${ex-4*Math.cos(angle-.6)} ${ey-4*Math.sin(angle-.6)} L ${ex} ${ey} L ${ex-4*Math.cos(angle+.6)} ${ey-4*Math.sin(angle+.6)}`);}
      if(this.axial){this.axial.hidden=!axial;this.axial.textContent=this.projected.z>0?'⊙':'⊗';}
      this.gravity.setAttribute('aria-label',`Down${axial?(this.projected.z>0?' toward viewer':' away from viewer'):''}`);
    }
  }
  render(renderer:THREE.WebGLRenderer,main:THREE.Camera,root:PaneBounds,rect:PaneBounds):boolean {
    const viewport=paneViewport(root,rect);if(!viewport)return false;
    this.syncCamera(main,viewport.width,viewport.height);
    renderer.getViewport(this.oldViewport);renderer.getScissor(this.oldScissor);
    const scissorTest=renderer.getScissorTest(),autoClear=renderer.autoClear;
    try{
      renderer.autoClear=false;renderer.setScissorTest(true);
      renderer.setViewport(viewport.left,viewport.bottom,viewport.width,viewport.height);
      renderer.setScissor(viewport.left,viewport.bottom,viewport.width,viewport.height);
      renderer.clearDepth();renderer.render(this.scene,this.camera);
    }finally{
      renderer.setViewport(this.oldViewport);renderer.setScissor(this.oldScissor);renderer.setScissorTest(scissorTest);renderer.autoClear=autoClear;
    }
    return true;
  }
  snapshot(){
    return {cameraQuaternion:this.camera.quaternion.toArray(),headQuaternion:this.mannequin.head.quaternion.toArray(),
      bodyQuaternion:this.mannequin.body.quaternion.toArray(),down:this.down?.toArray()??null,posture:this.label};
  }
  dispose():void {
    for(const mesh of [this.mannequin.body,this.mannequin.head]){mesh.geometry.dispose();mesh.material.dispose();}
  }
}

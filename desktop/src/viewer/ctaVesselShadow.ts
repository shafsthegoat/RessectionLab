import * as THREE from 'three';
import type {Bounds3} from './coordinates.ts';
import {renderTargetBytes} from './renderResourceEstimate.ts';

/** Light-space depth of the displayed, clipped vessels. */
export class CtaVesselShadow {
  private readonly target=new THREE.WebGLRenderTarget(768,768);
  private readonly depth=new THREE.MeshDepthMaterial({side:THREE.DoubleSide,colorWrite:false});
  private readonly scene=new THREE.Scene();
  private readonly camera:THREE.OrthographicCamera;
  private readonly center:THREE.Vector3;
  private readonly radius:number;
  private signature='';
  readonly projection=new THREE.Matrix4();

  constructor(bounds:Bounds3) {
    const low=new THREE.Vector3(...bounds[0]),high=new THREE.Vector3(...bounds[1]);
    this.center=low.clone().lerp(high,.5);
    this.radius=low.distanceTo(high)*.5+2;
    const r=this.radius;
    this.camera=new THREE.OrthographicCamera(-r,r,r,-r,1,2*r+3);
    this.target.depthTexture=new THREE.DepthTexture(this.target.width,this.target.height,THREE.UnsignedIntType);
  }

  get texture(){return this.target.depthTexture!;}
  get bytes(){return renderTargetBytes(this.target);}
  get depthMm(){return this.camera.far-this.camera.near;}
  get penumbraUv(){return .85/(2*this.radius);}

  capture(renderer:THREE.WebGLRenderer,objects:THREE.Object3D,direction:THREE.Vector3,up:THREE.Vector3,plane:THREE.Vector4) {
    const vessels:THREE.Mesh[]=[];
    objects.traverseVisible(object=>{
      if(object instanceof THREE.Mesh&&!Array.isArray(object.material)&&
        object.material.userData.ctaVesselReceiver===true&&object.material.opacity===1){
        object.updateWorldMatrix(true,false);
        vessels.push(object);
      }
    });
    const signature=JSON.stringify([direction.toArray(),up.toArray(),plane.toArray(),
      vessels.map(mesh=>[mesh.geometry.id,mesh.matrixWorld.elements])]);
    if(signature===this.signature)return;
    this.camera.position.copy(this.center).addScaledVector(direction,this.radius+2);
    this.camera.up.copy(up);
    this.camera.lookAt(this.center);
    this.camera.updateMatrixWorld();
    this.projection.multiplyMatrices(this.camera.projectionMatrix,this.camera.matrixWorldInverse);
    this.depth.clippingPlanes=[new THREE.Plane(new THREE.Vector3(plane.x,plane.y,plane.z),plane.w)];
    for(const vessel of vessels){
      const mesh=new THREE.Mesh(vessel.geometry,this.depth);
      mesh.matrixAutoUpdate=false;
      mesh.matrix.copy(vessel.matrixWorld);
      this.scene.add(mesh);
    }
    const previous=renderer.getRenderTarget();
    const viewport=renderer.getViewport(new THREE.Vector4());
    const scissor=renderer.getScissor(new THREE.Vector4());
    const scissorTest=renderer.getScissorTest();
    try{
      renderer.setRenderTarget(this.target);
      renderer.clear(false,true,false);
      renderer.render(this.scene,this.camera);
      this.signature=signature;
    }finally{
      this.scene.clear();
      renderer.setRenderTarget(previous);
      renderer.setViewport(viewport);
      renderer.setScissor(scissor);
      renderer.setScissorTest(scissorTest);
    }
  }

  dispose(){this.depth.dispose();this.target.dispose();}
}

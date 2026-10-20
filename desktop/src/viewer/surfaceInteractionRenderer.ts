import * as THREE from 'three';
import {renderTargetBytes} from './renderResourceEstimate';
/** Pixel workload only: every resident triangle and all clipping/material state
 * remain in the same scene. Settling draws directly at the normal pane size.
 * Interaction uses linear HDR followed by the renderer's normal output mapping. */
export class SurfaceInteractionRenderer {
  private readonly target=new THREE.WebGLRenderTarget(1,1,{type:THREE.HalfFloatType,depthBuffer:true});
  private readonly scene=new THREE.Scene();
  private readonly camera=new THREE.Camera();
  private readonly geometry=new THREE.PlaneGeometry(2,2);
  private readonly material=new THREE.MeshBasicMaterial({map:this.target.texture,depthTest:false,depthWrite:false,toneMapped:true});
  constructor(){this.scene.add(new THREE.Mesh(this.geometry,this.material));}
  get estimatedTargetBytes(){return renderTargetBytes(this.target);}
  render(renderer:THREE.WebGLRenderer,scene:THREE.Scene,camera:THREE.Camera,width:number,height:number,interactive:boolean,longEdge:number,restore:()=>void){
    if(!interactive){renderer.render(scene,camera);return {adaptive:false,width:Math.round(width*renderer.getPixelRatio()),height:Math.round(height*renderer.getPixelRatio())};}
    const ratio=renderer.getPixelRatio(),scale=Math.min(ratio,longEdge/Math.max(width,height));
    const w=Math.max(1,Math.round(width*scale)),h=Math.max(1,Math.round(height*scale));
    if(this.target.width!==w||this.target.height!==h)this.target.setSize(w,h);
    try{
      renderer.setRenderTarget(this.target);renderer.setScissorTest(false);renderer.setViewport(0,0,w/ratio,h/ratio);
      renderer.clear();renderer.render(scene,camera);
    }finally{renderer.setRenderTarget(null);renderer.setScissorTest(true);restore();}
    renderer.render(this.scene,this.camera);return {adaptive:true,width:w,height:h};
  }
  dispose(){this.target.dispose();this.material.dispose();this.geometry.dispose();}
}

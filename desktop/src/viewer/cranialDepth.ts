import * as THREE from 'three';

type Boundary = 'cortex' | 'head';
export type CorticalSignal = {texture:THREE.Data3DTexture;window:[number,number];worldToVoxel:THREE.Matrix4};
type Surface = {mesh:THREE.Mesh;material:THREE.Material};

/** The outer intersection of a folded tissue remains a translucent surface.
 * Interior walls of that same tissue would otherwise accumulate until opaque.
 * Each group gets its own depth so the scalp cannot erase the cortex. */
export class CranialDepth {
  readonly targets=new Map<Boundary,THREE.WebGLRenderTarget>();
  private readonly depth=new THREE.MeshDepthMaterial({side:THREE.DoubleSide,colorWrite:false});
  private readonly cameraRange={value:new THREE.Vector2()};
  private readonly projectionScale={value:1};

  capture(renderer:THREE.WebGLRenderer,scene:THREE.Scene,camera:THREE.PerspectiveCamera,
    surfaces:Surface[],width:number,height:number):void {
    this.cameraRange.value.set(camera.near,camera.far);
    this.projectionScale.value=height*camera.projectionMatrix.elements[5]/2;
    for(const group of ['head','cortex'] as const){
      const members=surfaces.filter(({mesh})=>mesh.userData.cranialBoundary===group);
      if(!members.length)continue;
      let target=this.targets.get(group);
      if(!target){
        target=new THREE.WebGLRenderTarget(1,1);
        target.depthTexture=new THREE.DepthTexture(1,1,THREE.UnsignedIntType);
        this.targets.set(group,target);
      }
      if(target.width!==width||target.height!==height)target.setSize(width,height);
      // All participating meshes are native source surfaces and share the
      // patient clipping planes. A section therefore exposes its next wall.
      const previous=scene.overrideMaterial;
      try{
        surfaces.forEach(({mesh})=>{mesh.visible=mesh.userData.cranialBoundary===group;});
        this.depth.clippingPlanes=members[0].material.clippingPlanes;
        scene.overrideMaterial=this.depth;
        renderer.setRenderTarget(target);renderer.clear();renderer.render(scene,camera);
      }finally{
        scene.overrideMaterial=previous;
        surfaces.forEach(({mesh})=>{mesh.visible=true;});
      }
    }
  }

  install(material:THREE.Material,group:Boundary,signal?:CorticalSignal):void {
    const previous=material.onBeforeCompile;
    const key=material.customProgramCacheKey;
    material.onBeforeCompile=(shader,renderer)=>{
      previous.call(material,shader,renderer);
      shader.uniforms.uBoundaryDepth={value:this.targets.get(group)!.depthTexture};
      shader.uniforms.uCameraRange=this.cameraRange;
      shader.uniforms.uProjectionScale=this.projectionScale;
      if(signal&&group==='cortex'){
        shader.uniforms.uCorticalScan={value:signal.texture};
        shader.uniforms.uCorticalWindow={value:new THREE.Vector2(...signal.window)};
        shader.uniforms.uCorticalWorldToVoxel={value:signal.worldToVoxel};
        shader.vertexShader='uniform mat4 uCorticalWorldToVoxel; varying vec3 vCorticalVoxel;\n'+shader.vertexShader.replace('#include <begin_vertex>',
          '#include <begin_vertex>\nvCorticalVoxel=(uCorticalWorldToVoxel*modelMatrix*vec4(position,1.)).xyz;');
        shader.fragmentShader='uniform highp sampler3D uCorticalScan; uniform vec2 uCorticalWindow; varying vec3 vCorticalVoxel;\n'+
          shader.fragmentShader.replace('#include <color_fragment>',`#include <color_fragment>
          vec3 uvw=(vCorticalVoxel.zyx+0.5)/vec3(textureSize(uCorticalScan,0));
          if(all(greaterThanEqual(uvw,vec3(0.)))&&all(lessThanEqual(uvw,vec3(1.)))){
            float signalValue=texture(uCorticalScan,uvw).r;
            float brightness=clamp((signalValue-uCorticalWindow.x)/(uCorticalWindow.y-uCorticalWindow.x),0.,1.);
            diffuseColor.rgb*=exp2((brightness-.45)*1.15);
          }`);
      }
      shader.fragmentShader=`
        uniform sampler2D uBoundaryDepth;
        uniform vec2 uCameraRange;
        uniform float uProjectionScale;
        float eyeDistance(float d){
          return uCameraRange.x*uCameraRange.y/mix(uCameraRange.y,uCameraRange.x,d);
        }
        float neighborDepth(vec2 uv, float center){
          float sampleDepth=texture2D(uBoundaryDepth,uv).r;
          return sampleDepth>=1.0 ? center : clamp(eyeDistance(sampleDepth)-center,-8.,8.)+center;
        }
      `+shader.fragmentShader.replace('#include <clipping_planes_fragment>',`
        #include <clipping_planes_fragment>
        vec2 boundaryUv=gl_FragCoord.xy/uSurfaceSize;
        float nearestDepth=texture2D(uBoundaryDepth,boundaryUv).r;
        if(gl_FragCoord.z>nearestDepth+0.00000012)discard;
      `).replace('#include <opaque_fragment>',group==='cortex'?`
        // A symmetric depth difference is zero on an inclined plane.
        // Shade recessed folds over a 2 mm projected neighborhood.
        float center=eyeDistance(nearestDepth);
        vec2 offset=clamp(2.*uProjectionScale/center,1.,12.)/uSurfaceSize;
        vec2 dx=vec2(offset.x,0.),dy=vec2(0.,offset.y);
        float nearby=.25*(neighborDepth(boundaryUv-dx,center)+neighborDepth(boundaryUv+dx,center)
                        +neighborDepth(boundaryUv-dy,center)+neighborDepth(boundaryUv+dy,center));
        float recess=smoothstep(.15,2.4,center-nearby);
        float lip=smoothstep(.35,2.8,nearby-center);
        float rim=pow(1.-abs(dot(normal,normalize(vViewPosition))),2.5);
        outgoingLight=outgoingLight*(1.-.58*recess)+diffuseColor.rgb*(.42*rim+.40*lip);
        #include <opaque_fragment>`:`
        vec2 px=1./uSurfaceSize;
        float edge=max(max(texture2D(uBoundaryDepth,boundaryUv+vec2(px.x,0.)).r,texture2D(uBoundaryDepth,boundaryUv-vec2(px.x,0.)).r),
                       max(texture2D(uBoundaryDepth,boundaryUv+vec2(0.,px.y)).r,texture2D(uBoundaryDepth,boundaryUv-vec2(0.,px.y)).r));
        float rim=max(step(.9999999,edge),pow(1.-abs(dot(normal,normalize(vViewPosition))),3.));
        diffuseColor.a=opacity*(.009+.91*rim);
        outgoingLight=diffuseColor.rgb*(.18+1.12*rim)+totalEmissiveRadiance;
        #include <opaque_fragment>`);
    };
    material.customProgramCacheKey=()=>key.call(material)+'/front-'+group+(signal?'/scan':'');
  }

  dispose():void{
    this.depth.dispose();
    for(const target of this.targets.values())target.dispose();
  }
}

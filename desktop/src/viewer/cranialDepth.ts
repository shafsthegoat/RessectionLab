import * as THREE from 'three';

const cranialBoundaries = ['head', 'cortex', 'lesion'] as const;
export type CranialBoundary = typeof cranialBoundaries[number];
export type CorticalSignal = {texture:THREE.Data3DTexture;window:[number,number];worldToVoxel:THREE.Matrix4;contrast?:number;nativeWallCrop?:true};
type Surface = {mesh:THREE.Mesh;material:THREE.Material};

const cropSampling = `
float operativeSoftTissueSample(ivec3 index){
  float signalValue=texelFetch(uCorticalScan,index.zyx,0).r;
  return clamp((signalValue-uCorticalWindow.x)/(uCorticalWindow.y-uCorticalWindow.x),0.,1.);
}
float operativeSoftTissueSignal(vec3 p,vec3 lastSample){
  vec3 voxel=clamp(p,vec3(0.),lastSample);
  ivec3 low=ivec3(floor(voxel)),high=min(low+1,ivec3(lastSample));
  vec3 w=fract(voxel);
  // Window neighbors before blending so dense bone or contrast cannot flood a tissue sample.
  float n00=mix(operativeSoftTissueSample(low),operativeSoftTissueSample(ivec3(low.xy,high.z)),w.z);
  float n10=mix(operativeSoftTissueSample(ivec3(high.x,low.yz)),operativeSoftTissueSample(ivec3(high.x,low.y,high.z)),w.z);
  float n01=mix(operativeSoftTissueSample(ivec3(low.x,high.y,low.z)),operativeSoftTissueSample(ivec3(low.x,high.yz)),w.z);
  float n11=mix(operativeSoftTissueSample(ivec3(high.xy,low.z)),operativeSoftTissueSample(high),w.z);
  return mix(mix(n00,n10,w.x),mix(n01,n11,w.x),w.y);
}
`;

/** Sample the originating acquisition on both solid and translucent cortical
 * surfaces. The volume peels reuse this same material hook. */
export function installCorticalSignal(material:THREE.MeshStandardMaterial,signal:CorticalSignal,enabled:boolean) {
  const active={value:enabled,transport:{value:false}};
  const previous=material.onBeforeCompile,key=material.customProgramCacheKey();
  material.onBeforeCompile=(shader,renderer)=>{
    previous.call(material,shader,renderer);
    shader.uniforms.uCorticalSignalEnabled=active;
    shader.uniforms.uCorticalTransport=active.transport;
    shader.uniforms.uCorticalScan={value:signal.texture};
    shader.uniforms.uCorticalWindow={value:new THREE.Vector2(...signal.window)};
    shader.uniforms.uCorticalContrast={value:signal.contrast??1.15};
    shader.uniforms.uCorticalWorldToVoxel={value:signal.worldToVoxel};
    shader.vertexShader='uniform bool uCorticalTransport; uniform mat4 uCorticalWorldToVoxel; attribute vec3 corticalRestOffset; attribute float corticalUnknown; varying float vCorticalUnknown; varying vec3 vCorticalVoxel;\n'+shader.vertexShader.replace('#include <begin_vertex>',
      '#include <begin_vertex>\nvCorticalVoxel=(uCorticalWorldToVoxel*modelMatrix*vec4(position+(uCorticalTransport?corticalRestOffset:vec3(0.)),1.)).xyz; vCorticalUnknown=uCorticalTransport?corticalUnknown:0.;');
    const header='uniform bool uCorticalSignalEnabled; uniform highp sampler3D uCorticalScan; uniform vec2 uCorticalWindow; uniform float uCorticalContrast; varying float vCorticalUnknown; varying vec3 vCorticalVoxel;\n';
    if(signal.nativeWallCrop){
      shader.fragmentShader=header+cropSampling+shader.fragmentShader.replace('#include <color_fragment>',`#include <color_fragment>
        float acquisitionEnd=0.;
        if(uCorticalSignalEnabled&&vCorticalUnknown<.0001){
          vec3 lastSample=vec3(textureSize(uCorticalScan,0).zyx-1);
          vec3 excess=max(-vCorticalVoxel,vCorticalVoxel-lastSample);
          acquisitionEnd=smoothstep(.035,.35,max(excess.x,max(excess.y,excess.z)));
          float brightness=operativeSoftTissueSignal(vCorticalVoxel,lastSample);
          vec3 section=mix(vec3(.045,.056,.064),vec3(.29,.33,.36),brightness);
          diffuseColor.rgb=mix(diffuseColor.rgb*exp2((brightness-.45)*uCorticalContrast),section,acquisitionEnd);
        }`).replace('#include <roughnessmap_fragment>',`#include <roughnessmap_fragment>
        roughnessFactor=mix(roughnessFactor,.8,acquisitionEnd);`);
    }else{
      shader.fragmentShader=header+shader.fragmentShader.replace('#include <color_fragment>',`#include <color_fragment>
        vec3 uvw=(vCorticalVoxel.zyx+0.5)/vec3(textureSize(uCorticalScan,0));
        if(uCorticalSignalEnabled&&vCorticalUnknown<.0001&&all(greaterThanEqual(uvw,vec3(0.)))&&all(lessThanEqual(uvw,vec3(1.)))){
          float signalValue=texture(uCorticalScan,uvw).r;
          float brightness=clamp((signalValue-uCorticalWindow.x)/(uCorticalWindow.y-uCorticalWindow.x),0.,1.);
          diffuseColor.rgb*=exp2((brightness-.45)*uCorticalContrast);
        }`);
    }
  };
  material.customProgramCacheKey=()=>key+(signal.nativeWallCrop?'/native-wall-crop-signal':'/transported-cortical-signal');
  return active;
}

/** The outer intersection of a folded tissue remains a translucent surface.
 * Interior walls of that same tissue would otherwise accumulate until opaque.
 * The head, cortex and single lesion each keep an independent depth. */
export class CranialDepth {
  readonly targets=new Map<CranialBoundary,THREE.WebGLRenderTarget>();
  private readonly depth=new THREE.MeshDepthMaterial({side:THREE.DoubleSide,colorWrite:false});
  private readonly cameraRange={value:new THREE.Vector2()};
  private readonly projectionScale={value:1};

  capture(renderer:THREE.WebGLRenderer,scene:THREE.Scene,camera:THREE.PerspectiveCamera,
    surfaces:Surface[],width:number,height:number):void {
    this.cameraRange.value.set(camera.near,camera.far);
    this.projectionScale.value=height*camera.projectionMatrix.elements[5]/2;
    for(const group of cranialBoundaries){
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

  install(material:THREE.Material,group:CranialBoundary):void {
    const previous=material.onBeforeCompile;
    const key=material.customProgramCacheKey;
    material.onBeforeCompile=(shader,renderer)=>{
      previous.call(material,shader,renderer);
      shader.uniforms.uBoundaryDepth={value:this.targets.get(group)!.depthTexture};
      shader.uniforms.uCameraRange=this.cameraRange;
      shader.uniforms.uProjectionScale=this.projectionScale;
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
      `);
      if(group==='lesion'){
        shader.fragmentShader=shader.fragmentShader.replace('#include <opaque_fragment>',`
          // Keep the outline of the estimate as its facing surface recedes.
          float center=eyeDistance(nearestDepth);
          vec2 offset=clamp(.55*uProjectionScale/center,1.,2.5)/uSurfaceSize;
          vec2 dx=vec2(offset.x,0.),dy=vec2(0.,offset.y);
          float edge=max(max(texture2D(uBoundaryDepth,boundaryUv-dx).r,texture2D(uBoundaryDepth,boundaryUv+dx).r),
                         max(texture2D(uBoundaryDepth,boundaryUv-dy).r,texture2D(uBoundaryDepth,boundaryUv+dy).r));
          float rim=max(step(1.,edge),pow(1.-abs(dot(normal,normalize(vViewPosition))),2.));
          diffuseColor.a*=mix(.48,1.,rim);
          #include <opaque_fragment>`);
        return;
      }
      shader.fragmentShader=shader.fragmentShader.replace('#include <opaque_fragment>',group==='cortex'?`
        // A symmetric depth difference is zero on an inclined plane.
        // Shade recessed folds over a 2 mm projected neighborhood.
        float center=eyeDistance(nearestDepth);
        vec2 offset=clamp(2.*uProjectionScale/center,1.,12.)/uSurfaceSize;
        vec2 dx=vec2(offset.x,0.),dy=vec2(0.,offset.y);
        float nearby=.25*(neighborDepth(boundaryUv-dx,center)+neighborDepth(boundaryUv+dx,center)
                        +neighborDepth(boundaryUv-dy,center)+neighborDepth(boundaryUv+dy,center));
        float recess=smoothstep(.15,2.4,center-nearby);
        // The physical lights already brighten the crowns. Add contact shade
        // only where the surface recedes behind its local neighborhood.
        outgoingLight=mix(outgoingLight,totalEmissiveRadiance,.40*recess);
        #include <opaque_fragment>`:`
        vec2 px=1./uSurfaceSize;
        float edge=max(max(texture2D(uBoundaryDepth,boundaryUv+vec2(px.x,0.)).r,texture2D(uBoundaryDepth,boundaryUv-vec2(px.x,0.)).r),
                       max(texture2D(uBoundaryDepth,boundaryUv+vec2(0.,px.y)).r,texture2D(uBoundaryDepth,boundaryUv-vec2(0.,px.y)).r));
        float rim=max(step(.9999999,edge),pow(1.-abs(dot(normal,normalize(vViewPosition))),3.));
        diffuseColor.a=opacity*(.009+.91*rim);
        outgoingLight=diffuseColor.rgb*(.18+1.12*rim)+totalEmissiveRadiance;
        #include <opaque_fragment>`);
    };
    material.customProgramCacheKey=()=>key.call(material)+'/front-v2-'+group;
  }

  dispose():void{
    this.depth.dispose();
    for(const target of this.targets.values())target.dispose();
  }
}

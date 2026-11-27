import * as THREE from 'three';

const vertexShader=`
  #include <clipping_planes_pars_vertex>
  varying vec3 vWorld;
  varying vec3 vNormal;
  void main(){
    vec4 mvPosition=modelViewMatrix*vec4(position,1.0);
    vWorld=(modelMatrix*vec4(position,1.0)).xyz;
    vNormal=normalMatrix*normal;
    gl_Position=projectionMatrix*mvPosition;
    #include <clipping_planes_vertex>
  }
`;
const fragmentShader=`
  #include <clipping_planes_pars_fragment>
  uniform int uAxis;
  uniform vec2 uRange;
  uniform vec3 uColor;
  uniform float uOpacity;
  varying vec3 vWorld;
  varying vec3 vNormal;
  void main(){
    #include <clipping_planes_fragment>
    float p=vWorld[uAxis];
    float distanceMm=max(uRange.x-p,p-uRange.y);
    if(distanceMm<=0.0)discard;
    float fade=mix(0.145,0.018,smoothstep(1.0,45.0,distanceMm));
    float lighting=0.55+0.45*abs(dot(normalize(vNormal),vec3(0.24,0.32,0.9165)));
    vec3 tint=mix(vec3(0.17,0.18,0.23),uColor,0.30);
    gl_FragColor=vec4(tint*(fade*lighting*uOpacity),1.0);
    #include <tonemapping_fragment>
    #include <colorspace_fragment>
  }
`;

type SourceMesh=THREE.Mesh<THREE.BufferGeometry,THREE.MeshStandardMaterial>;

export class VesselSlabContext {
  private readonly scene=new THREE.Scene();
  private readonly layers=new Map<SourceMesh,THREE.Mesh<THREE.BufferGeometry,THREE.ShaderMaterial>>();
  private readonly range=new THREE.Vector2();
  private axis=0;
  private planes:THREE.Plane[]=[];

  configure(axis:number, min:number, max:number, planes:THREE.Plane[]) {
    this.axis=axis;
    this.range.set(min,max);
    this.planes=planes;
  }

  update(surfaces:readonly SourceMesh[]) {
    const visible=new Set(surfaces.filter(mesh=>mesh.visible&&mesh.material.opacity>0));
    for(const [source,layer] of this.layers){
      if(visible.has(source))continue;
      layer.removeFromParent();
      layer.material.dispose();
      this.layers.delete(source);
    }
    for(const source of visible){
      let layer=this.layers.get(source);
      if(!layer){
        const material=new THREE.ShaderMaterial({vertexShader,fragmentShader,clipping:true,side:THREE.DoubleSide,
          uniforms:{uAxis:{value:this.axis},uRange:{value:this.range},uColor:{value:new THREE.Color()},uOpacity:{value:1}}});
        layer=new THREE.Mesh(source.geometry,material);
        layer.matrixAutoUpdate=false;
        this.scene.add(layer);
        this.layers.set(source,layer);
      }
      // The foreground may switch between raw and conditioned buffers or
      // move to another source affine while this inspection mode is open.
      source.updateWorldMatrix(true,false);
      layer.geometry=source.geometry;
      layer.matrix.copy(source.matrixWorld);
      layer.matrixWorldNeedsUpdate=true;
      layer.material.uniforms.uAxis.value=this.axis;
      layer.material.uniforms.uColor.value.copy(source.material.color);
      layer.material.uniforms.uOpacity.value=source.material.opacity;
      layer.material.clippingPlanes=this.planes;
    }
    return this.layers.size>0;
  }

  render(renderer:THREE.WebGLRenderer,camera:THREE.Camera) {
    renderer.render(this.scene,camera);
  }

  dispose(){
    // The foreground owns all geometry and attributes.
    this.update([]);
  }
}

/** Composite only the closest dim context surface. Clear its depth before
 * foreground rendering so it never occludes the slab or limits scalar rays. */
export function renderAnatomyScene(renderer:THREE.WebGLRenderer,scene:THREE.Scene,camera:THREE.Camera,context?:VesselSlabContext){
  if(context){
    context.render(renderer,camera);
    renderer.clearDepth();
  }
  renderer.render(scene,camera);
}

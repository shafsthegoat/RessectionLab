import * as THREE from 'three';
import type {Shape3} from './coordinates';
import {EXTERIOR_AIR, ORIGINAL_SKIN, INTACT_FACE} from './exteriorSkinSurface.ts';

/** Test each intersected anatomical fragment against its native skin/air
 * voxel at certified outer faces. This also runs in the light's depth pass,
 * so unsupported deep facets cannot cast a shadow through the intact skin. */
export class ExteriorSkinDomain {
  readonly texture: THREE.Data3DTexture;
  enabled = false;
  constructor(flags: Uint8Array, shape: Shape3) {
    this.texture = new THREE.Data3DTexture(flags, shape[2], shape[1], shape[0]);
    this.texture.format = THREE.RedFormat;
    this.texture.type = THREE.UnsignedByteType;
    this.texture.unpackAlignment = 1;
    this.texture.needsUpdate = true;
  }
  bind(mesh: THREE.Mesh, skin: boolean) {
    const material = mesh.material;
    if (!(material instanceof THREE.MeshStandardMaterial)) return;
    const existing = mesh.userData.exteriorSkinDomain as {domain: ExteriorSkinDomain} | undefined;
    if (existing) {
      existing.domain = this;
      return;
    }
    const state = {domain: this};
    mesh.userData.exteriorSkinDomain = state;
    const install = (m: THREE.Material) => {
      const previous = m.onBeforeCompile,
        key = m.customProgramCacheKey();
      m.onBeforeCompile = (shader, renderer) => {
        previous.call(m, shader, renderer);
        shader.uniforms.uIntactExterior = {
          get value() {
            return state.domain.texture;
          },
        };
        shader.uniforms.uUseExterior = {
          get value() {
            return state.domain.enabled;
          },
        };
        shader.vertexShader =
          'varying vec3 vExteriorVoxel;\n' +
          shader.vertexShader.replace('#include <begin_vertex>', '#include <begin_vertex>\nvExteriorVoxel=position;');
        shader.fragmentShader =
          'varying vec3 vExteriorVoxel;uniform highp sampler3D uIntactExterior;uniform bool uUseExterior;\n' +
          shader.fragmentShader.replace(
            '#include <clipping_planes_fragment>',
            `#include <clipping_planes_fragment>
        if(uUseExterior){
          ivec3 cell=ivec3(floor(vExteriorVoxel.zyx));
          if(all(greaterThanEqual(cell,ivec3(0))) && all(lessThan(cell,textureSize(uIntactExterior,0)-ivec3(1)))){
            int flags=int(round(texelFetch(uIntactExterior,cell,0).r*255.));
            if((flags & ${INTACT_FACE})!=0){
              int nearest=int(round(texelFetch(uIntactExterior,ivec3(floor(vExteriorVoxel.zyx+vec3(.5))),0).r*255.));
              if((nearest & ${skin ? EXTERIOR_AIR : EXTERIOR_AIR | ORIGINAL_SKIN})!=0)discard;
            }
          }
        }`,
          );
      };
      m.customProgramCacheKey = () => key + '/native-exterior/' + skin;
      m.needsUpdate = true;
    };
    install(material);
    const depth = new THREE.MeshDepthMaterial({depthPacking: THREE.RGBADepthPacking});
    mesh.customDepthMaterial = depth;
    install(depth);
    material.addEventListener('dispose', () => depth.dispose());
  }
  dispose() {
    this.enabled = false;
    this.texture.dispose();
  }
}

import * as THREE from 'three';
import type {Shape3} from './coordinates';
import {EXTERIOR_AIR, ORIGINAL_SKIN, INTACT_FACE, OUTSIDE_BRAIN} from './exteriorSkinSurface.ts';

/** Illustrative covering of a segmented tissue where it meets free space in
 * the undeformed head: soft tissue under exterior air is drawn as skin, and
 * white matter under subarachnoid space as cortex. Source labels, contours
 * and picking are unchanged; later cuts are never covered. */
export type SurfaceCover = 'skin' | 'cortex';
export function surfaceCoverFor(kind: string | undefined): SurfaceCover | undefined {
  // Where the scalp estimate has gaps, deeper soft tissue or bone meets the air.
  if (kind === 'galea' || kind === 'extracranial' || kind === 'skull' || kind === 'diploe') return 'skin';
  // Where the cortex estimate is thin, white matter meets CSF or the dura.
  if (kind === 'whiteMatter' || kind === 'tract') return 'cortex';
  return undefined;
}
/** Region kinds outside the brain parenchyma in an undeformed head, including surface vessels. */
export const OUTSIDE_BRAIN_KINDS = new Set(['scalp', 'galea', 'extracranial', 'tissue', 'skull', 'diploe', 'dura', 'artery', 'vein']);
const COVER_BITS: Record<SurfaceCover, [number, number]> = {skin: [EXTERIOR_AIR, 0], cortex: [OUTSIDE_BRAIN, EXTERIOR_AIR]};

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
  /** Flag free space and the given outside-brain owners of the undeformed partition. */
  markOutsideBrain(owner: Uint16Array, outside: ReadonlySet<number>) {
    const flags = this.texture.image.data as Uint8Array;
    if (flags.length !== owner.length) throw Error('Outside-brain flags differ from the source grid');
    for (let i = 0; i < owner.length; i++) if (!owner[i] || outside.has(owner[i])) flags[i] |= OUTSIDE_BRAIN;
    this.texture.needsUpdate = true;
  }
  readonly coverColors: Record<SurfaceCover, {value: THREE.Color}> = {
    skin: {value: new THREE.Color('#bf8d7e')},
    cortex: {value: new THREE.Color('#b38b8b')},
  };
  readonly rasToVoxel = {value: new THREE.Matrix4()};
  /** Requires the illustrative finish on this material; follows it on and off. */
  cover(material: THREE.Material, cover: SurfaceCover, mobile: boolean) {
    if (!(material instanceof THREE.MeshStandardMaterial) || !material.userData.tissueFinish) return false;
    const existing = material.userData.surfaceCover as {domain: ExteriorSkinDomain} | undefined;
    if (existing) {
      existing.domain = this;
      return true;
    }
    const state = {domain: this as ExteriorSkinDomain};
    material.userData.surfaceCover = state;
    const previous = material.onBeforeCompile,
      key = material.customProgramCacheKey(),
      [mask, exclude] = COVER_BITS[cover];
    material.onBeforeCompile = (shader, renderer) => {
      previous.call(material, shader, renderer);
      shader.uniforms.uCoverFlags = {
        get value() {
          return state.domain.texture;
        },
      };
      shader.uniforms.uCoverColor = {
        get value() {
          return state.domain.coverColors[cover].value;
        },
      };
      shader.uniforms.uCoverRasToVoxel = {
        get value() {
          return state.domain.rasToVoxel.value;
        },
      };
      shader.uniforms.uCoverActive = material.userData.tissueFinish;
      shader.vertexShader =
        'varying vec3 vCoverVoxel;uniform mat4 uCoverRasToVoxel;\n' +
        shader.vertexShader.replace(
          '#include <begin_vertex>',
          '#include <begin_vertex>\nvCoverVoxel=' +
            (mobile ? '(uCoverRasToVoxel*vec4(exposureRest,1.)).xyz;' : 'position+finishRestOffset;'),
        );
      shader.fragmentShader =
        'varying vec3 vCoverVoxel;uniform highp sampler3D uCoverFlags;uniform vec3 uCoverColor;uniform bool uCoverActive;\n' +
        shader.fragmentShader.replace(
          '#include <color_fragment>',
          `#include <color_fragment>
        if(uCoverActive){
          ivec3 coverSize=textureSize(uCoverFlags,0).zyx,coverBase=ivec3(floor(vCoverVoxel+1e-4));
          bool covered=false;
          for(int i=0;i<8;i++){
            ivec3 c=coverBase+ivec3(i&1,(i>>1)&1,i>>2);
            if(any(lessThan(c,ivec3(0)))||any(greaterThanEqual(c,coverSize)))continue;
            int f=int(round(texelFetch(uCoverFlags,c.zyx,0).r*255.));
            if((f&${mask})!=0&&(f&${exclude})==0){covered=true;break;}
          }
          if(covered)diffuseColor.rgb=uCoverColor;
        }`,
        );
    };
    material.customProgramCacheKey = () => key + '/surface-cover/' + cover + '/' + mobile;
    material.needsUpdate = true;
    return true;
  }
  dispose() {
    this.enabled = false;
    this.texture.dispose();
  }
}

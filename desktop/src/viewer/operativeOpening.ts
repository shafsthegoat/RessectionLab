import * as THREE from 'three';
import type { OperativeModel } from './operativeFieldModel.ts';
import { operativeVoxelSurface } from './operativeVoxelSurface.ts';
import { operativeSurface } from './operativeSurface.ts';
import { prepareDuralRim } from './operativeDuralRim.ts';
import { estimatedDuralFlap, updateDuralFold } from './operativeDuralFlap.ts';

type Opening = 'bone' | 'dura';
type TissueMesh = THREE.Mesh<THREE.BufferGeometry, THREE.MeshStandardMaterial>;

/** The intact and completely cut surfaces partition each visible fragment at
 * the instrument angle. The same test runs in the directional shadow pass. */
function angularClip(material: THREE.Material, limit: THREE.IUniform<number>, cutSurface: boolean) {
  const previous = material.onBeforeCompile.bind(material);
  const previousKey = material.customProgramCacheKey();
  material.onBeforeCompile = (shader, renderer) => {
    previous(shader, renderer);
    shader.uniforms.uOpeningAngle = limit;
    shader.uniforms.uCutSurface = { value: cutSurface };
    shader.vertexShader =
      'varying vec2 vOpeningXY;\n' +
      shader.vertexShader.replace(
        '#include <begin_vertex>',
        '#include <begin_vertex>\nvOpeningXY = position.xy;'
      );
    shader.fragmentShader =
      'uniform float uOpeningAngle;\nuniform bool uCutSurface;\nvarying vec2 vOpeningXY;\n' +
      shader.fragmentShader.replace(
        '#include <clipping_planes_fragment>',
        `
        #include <clipping_planes_fragment>
        if (uOpeningAngle >= 0.) {
          float theta = vOpeningXY.x == 0. ? sign(vOpeningXY.y) * 1.57079632679 : atan(vOpeningXY.y, vOpeningXY.x);
          if (theta < 0.) theta += 6.28318530718;
          if ((theta < uOpeningAngle) != uCutSurface) discard;
        }`
      );
  };
  material.customProgramCacheKey = () => previousKey + '/operative-opening';
}

type Surfaces = {
  intact: THREE.BufferGeometry;
  cut: THREE.BufferGeometry;
  detached: THREE.BufferGeometry;
  flap: THREE.BufferGeometry | null;
  membrane?: boolean;
  rim?: boolean;
};

export function prepareOperativeOpening(
  kind: Opening,
  model: OperativeModel,
  intact: TissueMesh,
  flap: TissueMesh,
  cutMaterial: THREE.MeshStandardMaterial,
  flapOrigin: THREE.Vector3
) {
  const placeholder = intact.geometry;
  const flapPlaceholder = flap.geometry;
  const cut = new THREE.Mesh(placeholder, cutMaterial);
  cut.name = kind + 'Kerf';
  cut.castShadow = intact.castShadow;
  cut.receiveShadow = intact.receiveShadow;
  intact.parent?.add(cut);
  const limit = { value: -1 };
  for (const mesh of [intact, cut]) {
    mesh.userData.sourceLayer={role:kind,openingLimit:limit,kerf:mesh===cut};
    mesh.customDepthMaterial = new THREE.MeshDepthMaterial({ depthPacking: THREE.RGBADepthPacking });
    angularClip(mesh.material, limit, mesh === cut);
    angularClip(mesh.customDepthMaterial, limit, mesh === cut);
  }
  const sections = new Map<number, Surfaces>();
  const stationary = model.layers[kind];
  const movingKind = kind === 'bone' ? 'boneFlap' : 'duraFlap';
  const moving = model.layers[movingKind];
  const duraSources=model.field.layers?.filter(layer=>layer.role==='dura'&&layer.sampledVoxels>0);
  const estimated=kind==='dura'&&duraSources?.length&&duraSources.every(layer=>layer.model||layer.proxy);
  const initialSide=flap.material.side;
  function prepare(row: number, raw: boolean): Surfaces {
    const surface = (layer:typeof stationary, name:string) => raw
      ? operativeVoxelSurface(model.field, layer.mask, layer.bounds, row)
      : operativeSurface(model.field, layer.mask, layer.bounds, row, name);
    const rim=estimated&&!raw&&row===0?prepareDuralRim(model):null;
    const extract = () => rim?rim.geometry():surface(stationary,kind);
    const original = extract();
    if (model.field.stations[kind] === null)
      return { intact: original, cut: original, detached: original, flap: null };
    try {
      model.applyOpening(kind, 128, false);
      const kerfed = extract();
      model.applyOpening(kind, 128, true);
      const detached = extract();
      const membrane=Boolean(estimated&&!raw);
      const patch = membrane
        ? estimatedDuralFlap(model.field,moving.mask,moving.bounds,row,flapOrigin,model.duralOpening)
        : surface(moving, movingKind).translate(-flapOrigin.x, -flapOrigin.y, -flapOrigin.z);
      return { intact: original, cut: kerfed, detached, flap: patch, membrane, rim:!!rim };
    } finally {
      rim?.dispose();
      model.applyOpening(kind, 0, false);
    }
  }
  function warm(row:number,raw=false) {
    const key = raw ? -1-row : row;
    let surfaces = sections.get(key);
    if (!surfaces) {
      surfaces = prepare(row,raw);
      sections.set(key, surfaces);
    }
    return surfaces;
  }
  function update(row: number, amount: number, raw = false) {
    const surfaces=warm(row,raw);
    if (model.field.stations[kind] === null) amount = 0;
    const sweep = model.openingState(kind, amount);
    const { lift } = sweep;
    const rimSide=surfaces.rim?THREE.DoubleSide:initialSide;
    for(const mesh of [intact,cut])if(mesh.material.side!==rimSide){
      mesh.material.side=rimSide;mesh.material.needsUpdate=true;
    }
    const partial = sweep.fraction > 0 && sweep.fraction < 1;
    limit.value = partial ? sweep.angle : -1;
    intact.geometry = lift > 0 ? surfaces.detached : surfaces.intact;
    cut.geometry = surfaces.cut;
    intact.visible = sweep.fraction < 1 || lift > 0;
    cut.visible = sweep.fraction > 0 && lift === 0;
    flap.visible = lift > 0 && surfaces.flap !== null;
    if (surfaces.flap && flap.geometry!==surfaces.flap) {
      flap.geometry = surfaces.flap;
      flap.morphTargetInfluences=undefined;
      flap.updateMorphTargets();
      const side=surfaces.membrane?THREE.DoubleSide:initialSide;
      if(flap.material.side!==side){flap.material.side=side;flap.material.needsUpdate=true;}
    }
    if(surfaces.membrane)updateDuralFold(flap,lift);
    return sweep;
  }
  function dispose() {
    const resources = new Set([placeholder, flapPlaceholder]);
    for (const surfaces of sections.values()) {
      resources.add(surfaces.intact);
      resources.add(surfaces.cut);
      resources.add(surfaces.detached);
      if (surfaces.flap) resources.add(surfaces.flap);
    }
    for (const mesh of [intact, cut, flap]) {
      mesh.removeFromParent();
      mesh.material.dispose();
      mesh.customDepthMaterial?.dispose();
    }
    for (const geometry of resources) geometry.dispose();
    sections.clear();
  }
  return { update, warm, dispose, kerf:cut };
}

/** Lift the free geometric flap above the sampled skull before translating
 * it clear of the aperture. The parked pose is an illustration. The engine
 * retains its own opening and debris state. */
export function boneFlapTransfer(model:OperativeModel){
  const {field,opening}=model,[nx,ny,nz]=field.shape,s=field.spacingMm;
  const mask=model.layers.boneFlap;
  if(mask.bounds.low[0]>mask.bounds.high[0])return ()=>[0,0,0] as const;
  const xs=(i:number)=>field.origin[0]+i*s;
  const distance=xs(mask.bounds.high[0]+.5)+opening+2*s;
  const minX=Math.max(0,mask.bounds.low[0]-Math.ceil(distance/s)-1);
  const maxX=Math.min(nx-1,mask.bounds.high[0]+1);
  let highest=mask.bounds.high[2];
  for(let x=minX;x<=maxX;x++)for(let y=Math.max(0,mask.bounds.low[1]-1);y<=Math.min(ny-1,mask.bounds.high[1]+1);y++){
    const start=(x*ny+y)*nz;
    for(let z=nz-1;z>highest;z--)if(field.voxels[start+z]&1){highest=z;break;}
  }
  // The top and bottom voxel faces are half a cell from their centres.
  // An extra cell on each side accommodates bounded sheet fairing.
  const height=(highest-mask.bounds.low[2]+3)*s;
  return (lift:number)=>[
    -distance*THREE.MathUtils.smootherstep(lift,.40,1),0,
    height*THREE.MathUtils.smootherstep(lift,0,.40),
  ] as const;
}

import * as THREE from 'three';
import type { OperativeModel } from './operativeFieldModel.ts';
import { operativeVoxelSurface } from './operativeVoxelSurface.ts';
import { operativeSurface } from './operativeSurface.ts';

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
  function prepare(row: number, raw: boolean): Surfaces {
    const surface = (layer:typeof stationary, name:string) => raw
      ? operativeVoxelSurface(model.field, layer.mask, layer.bounds, row)
      : operativeSurface(model.field, layer.mask, layer.bounds, row, name);
    const extract = () => surface(stationary, kind);
    const original = extract();
    if (model.field.stations[kind] === null)
      return { intact: original, cut: original, detached: original, flap: null };
    try {
      model.applyOpening(kind, 128, false);
      const kerfed = extract();
      model.applyOpening(kind, 128, true);
      const detached = extract();
      const patch = surface(moving, movingKind);
      patch.translate(-flapOrigin.x, -flapOrigin.y, -flapOrigin.z);
      return { intact: original, cut: kerfed, detached, flap: patch };
    } finally {
      model.applyOpening(kind, 0, false);
    }
  }
  function update(row: number, amount: number, raw = false) {
    const key = raw ? -1-row : row;
    let surfaces = sections.get(key);
    if (!surfaces) {
      surfaces = prepare(row,raw);
      sections.set(key, surfaces);
    }
    if (model.field.stations[kind] === null) amount = 0;
    const sweep = model.openingState(kind, amount);
    const { lift } = sweep;
    const partial = sweep.fraction > 0 && sweep.fraction < 1;
    limit.value = partial ? sweep.angle : -1;
    intact.geometry = lift > 0 ? surfaces.detached : surfaces.intact;
    cut.geometry = surfaces.cut;
    intact.visible = sweep.fraction < 1 || lift > 0;
    cut.visible = sweep.fraction > 0 && lift === 0;
    flap.visible = lift > 0 && surfaces.flap !== null;
    if (surfaces.flap) flap.geometry = surfaces.flap;
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
  return { update, dispose };
}

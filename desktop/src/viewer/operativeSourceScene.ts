/** Source-resolution surfaces of a host rehearsal, drawn under the engine's state.
 *
 * The asset (`operative-source-surfaces.npz`) carries smooth skin, skull,
 * dura, cortex, white matter, tumour, vessel tube walls and tract streamlines
 * built from the source images, each vertex bound to the engine working cell
 * it lies in. This scene replaces the voxel-meshed drawing of those layers:
 *
 * - scalp, bone and dura vertices inside a body's outline move with that
 *   body's engine cells by the same per-cell rigid transforms the voxel flap
 *   uses (`ExposureBodyTransport`), split along the outline with a narrow kerf
 *   and closed by a wall between the outer and inner surfaces;
 * - removed cells (and CUSA's pending fraction) clip the brain, tumour and
 *   vessel surfaces through a smooth half-cell field, and its 0.5 level is
 *   drawn as the cavity wall, shaded by the scan intensity.
 *
 * Display only: the engine keeps cells, counts and picking; a pick on these
 * surfaces resolves to the engine cell of the hit vertex. */
import * as THREE from 'three';
import type {OperativeSourceAsset, SourceBodyName, SourceOutline, SourceSurface} from '../operative-source-surfaces';
import type {OperativeExposureBody} from '../operative-exposure-types';
import type {SurgicalView} from '../surgical-types';
import type {Affine, Point3, Shape3} from './coordinates';
import {ExposureBodyTransport} from './exposureBodySurface.ts';
import {bodyIsDisplaced, sourceCenter} from './exposureAnatomy.ts';
import {placeInSourceFrame} from './sceneGeometry.ts';
import {tractColor} from './tractDisplay.ts';
import {bindToBody, directionToVoxel, fillSmallHoles, kerfWall, normalsToVoxel, PlanarLoop, partAttribute, rasToVoxel, splitSheet, transportVertices, voxelToRas,
  type BodyBinding, type MeshPart} from './sourceSurfaceGeometry.ts';
import {cavityField, cavitySurface, removalOccupancy, sampleCavity, CAVITY_LEVEL, type CavityField} from './sourceCavityField.ts';

export type SourceRole = 'surface' | 'vessel' | 'cavity' | 'wall' | 'underside';
export type SourceMaterialFactory = (layer: string, kind: string, role: SourceRole) => THREE.MeshStandardMaterial;

/** Vessel colours by the asset's segment kind; unknown kind stays neutral. */
export const SOURCE_VESSEL_COLORS = {artery: '#b33a2e', vein: '#3f4f86', unknown: '#9b8580'} as const;
/** Kerf half-widths (mm) on each side of a body outline once the body has moved. */
export const KERF_HALF_MM: Record<SourceBodyName, number> = {scalp: 0.45, bone: 1.0, dura: 0.3};
const SEAM_COLORS: Record<SourceBodyName, string> = {scalp: '#5d2420', bone: '#5e5040', dura: '#6e3a36'};
const PLANNED_COLOR = '#4b2f78';
const STATIC_SHADOWLESS = new Set(['white', 'skullInner', 'skin', 'skullOuter']);
/** Display-only outward offset of the assumed dura sheet (mm). */
export const DURA_DISPLAY_LIFT_MM = 0.8;

type Piece = {mesh: THREE.Mesh; layer: string; kind: string; cells: Int32Array; group: 'brain' | 'vessel' | 'head'};
type Flap = {
  piece: Piece; restRas: Float32Array; restNormalsRas: Float32Array | null; restVoxel: Float32Array; binding: BodyBinding;
  primary: Int32Array; baseIndex: Uint32Array; goneKey?: string;
};
type BodyDisplay = {
  intact: Piece[]; fixed: Piece[]; flaps: Flap[]; built: boolean;
  transport?: ExposureBodyTransport; members?: string; displaced: boolean;
  lines: THREE.Group; lineKey?: string;
};

/** Install the cavity clip in a surface material: fragments whose rest position
 * lies in the removed region (field above 0.5) are discarded. */
export class CavityClip {
  readonly uniforms = {uCavityOn: {value: false}, uCavity: {value: null as THREE.Data3DTexture | null},
    uCavityOrigin: {value: new THREE.Vector3()}, uCavityStep: {value: 0.5}};
  private installed = new WeakSet<THREE.Material>();
  field: CavityField | null = null;
  set(field: CavityField | null) {
    this.uniforms.uCavity.value?.dispose();
    this.uniforms.uCavity.value = null;
    this.field = field;
    if (field) {
      const data = new Uint8Array(field.values.length);
      for (let i = 0; i < data.length; i++) data[i] = Math.round(255 * Math.max(0, Math.min(1, field.values[i])));
      const texture = new THREE.Data3DTexture(data, field.size[2], field.size[1], field.size[0]);
      texture.format = THREE.RedFormat;
      texture.type = THREE.UnsignedByteType;
      texture.unpackAlignment = 1;
      texture.minFilter = texture.magFilter = THREE.LinearFilter;
      texture.needsUpdate = true;
      this.uniforms.uCavity.value = texture;
      this.uniforms.uCavityOrigin.value.set(...field.origin);
      this.uniforms.uCavityStep.value = field.step;
    }
    this.uniforms.uCavityOn.value = !!field;
  }
  install(material: THREE.Material) {
    if (this.installed.has(material)) return;
    this.installed.add(material);
    const previous = material.onBeforeCompile, key = material.customProgramCacheKey();
    material.onBeforeCompile = (shader, renderer) => {
      previous.call(material, shader, renderer);
      Object.assign(shader.uniforms, this.uniforms);
      shader.vertexShader = 'varying vec3 vCavityRest;\n' + shader.vertexShader.replace('#include <begin_vertex>', '#include <begin_vertex>\nvCavityRest=position;');
      shader.fragmentShader = 'uniform bool uCavityOn;uniform highp sampler3D uCavity;uniform vec3 uCavityOrigin;uniform float uCavityStep;varying vec3 vCavityRest;\n' +
        shader.fragmentShader.replace('#include <clipping_planes_fragment>', `#include <clipping_planes_fragment>
        if(uCavityOn){
          vec3 uvw=((vCavityRest-uCavityOrigin)/uCavityStep+.5).zyx/vec3(textureSize(uCavity,0));
          if(all(greaterThan(uvw,vec3(0.)))&&all(lessThan(uvw,vec3(1.)))&&texture(uCavity,uvw).r>${CAVITY_LEVEL.toFixed(2)})discard;
        }`);
    };
    material.customProgramCacheKey = () => key + '/source-cavity-clip';
    material.needsUpdate = true;
  }
  /** CPU equivalent for picking. */
  removed(voxel: readonly number[]) { return !!this.field && sampleCavity(this.field, voxel) > CAVITY_LEVEL; }
  dispose() { this.set(null); }
}

function geometryOf(positions: Float32Array, normals: Float32Array, indices: Uint32Array, dynamic = false) {
  const g = new THREE.BufferGeometry();
  const p = new THREE.BufferAttribute(positions, 3), n = new THREE.BufferAttribute(normals, 3);
  if (dynamic) { p.setUsage(THREE.DynamicDrawUsage); n.setUsage(THREE.DynamicDrawUsage); }
  g.setAttribute('position', p);
  g.setAttribute('normal', n);
  g.setIndex(new THREE.BufferAttribute(indices, 1));
  g.computeBoundingSphere();
  g.computeBoundingBox();
  return g;
}

function cellsOfPart(part: MeshPart, cells: Int32Array) {
  const out = new Int32Array(part.source.length);
  for (let i = 0; i < out.length; i++) out[i] = part.source[i] >= 0 ? cells[part.source[i]] : cells[part.between[3 * i + 2] < 0.5 ? part.between[3 * i] : part.between[3 * i + 1]];
  return out;
}

function voxelCells(voxel: Float32Array, shape: Shape3) {
  const [nx, ny, nz] = shape, out = new Int32Array(voxel.length / 3);
  for (let i = 0; i < out.length; i++) {
    const x = Math.round(voxel[3 * i]), y = Math.round(voxel[3 * i + 1]), z = Math.round(voxel[3 * i + 2]);
    out[i] = x < 0 || y < 0 || z < 0 || x >= nx || y >= ny || z >= nz ? -1 : (x * ny + y) * nz + z;
  }
  return out;
}

function tube(points: number[][], radius: number, color: string, closed: boolean) {
  const curve = new THREE.CatmullRomCurve3(points.map((p) => new THREE.Vector3(...p)), closed, 'centripetal');
  const mesh = new THREE.Mesh(new THREE.TubeGeometry(curve, Math.max(8, points.length * 2), radius, 6, closed),
    new THREE.MeshStandardMaterial({color, roughness: 0.7, metalness: 0}));
  mesh.raycast = () => {};
  mesh.userData.displayOnly = true;
  return mesh;
}

/** Cells (RAS centres) of the body edges the engine reports cut, as outline point flags. */
export function cutOutlinePoints(outline: SourceOutline, seamEdges: ReadonlyArray<[number, number]>, shape: Shape3, affine: Affine, reachMm = 2.2) {
  const n = outline.hinge.length, flags = new Uint8Array(n);
  if (!seamEdges.length) return flags;
  const mids = seamEdges.map(([a, b]) => {
    const p = sourceCenter(a, shape, affine), q = sourceCenter(b, shape, affine);
    return [(p[0] + q[0]) / 2, (p[1] + q[1]) / 2, (p[2] + q[2]) / 2];
  });
  const cell = reachMm, buckets = new Map<string, number[][]>();
  const key = (p: readonly number[]) => p.map((v) => Math.floor(v / cell)).join(',');
  for (const m of mids) (buckets.get(key(m)) ?? buckets.set(key(m), []).get(key(m))!).push(m);
  for (let i = 0; i < n; i++) {
    if (outline.hinge[i]) continue;
    const p = [outline.points[3 * i], outline.points[3 * i + 1], outline.points[3 * i + 2]], b = p.map((v) => Math.floor(v / cell));
    search: for (let dx = -1; dx <= 1; dx++) for (let dy = -1; dy <= 1; dy++) for (let dz = -1; dz <= 1; dz++)
      for (const m of buckets.get(`${b[0] + dx},${b[1] + dy},${b[2] + dz}`) ?? [])
        if (Math.hypot(m[0] - p[0], m[1] - p[1], m[2] - p[2]) <= reachMm) { flags[i] = 1; break search; }
  }
  return flags;
}

/** Runs of consecutive flagged points of a closed outline. */
function runs(flags: Uint8Array) {
  const n = flags.length, out: number[][] = [];
  if (flags.every(Boolean)) return [[...Array(n).keys(), 0]];
  const start = flags.findIndex((f, i) => f > 0 && !flags[(i - 1 + n) % n]);
  if (start < 0) return out;
  for (let k = 0; k < n; k++) {
    const i = (start + k) % n;
    if (flags[i] && !flags[(i - 1 + n) % n]) out.push([]);
    if (flags[i]) out[out.length - 1].push(i);
  }
  return out.filter((r) => r.length > 1);
}

/** The assumed dura sheet is the inner table where the engine's dura lies within 3 mm,
 * so it has small gaps that read as holes once the bone is off: close loops under 40 mm. */
export function closeDuraGaps(surface: SourceSurface): SourceSurface {
  const filled = fillSmallHoles(surface.positions, surface.normals, surface.triangles, 40);
  if (!filled.loops.length) return surface;
  const extend = <T extends Int32Array | Uint8Array | Float32Array>(values: T | undefined, pick: (loop: number[]) => number): T | undefined => {
    if (!values) return values;
    const out = new (values.constructor as {new (n: number): T})(values.length + filled.loops.length);
    out.set(values);
    filled.loops.forEach((loop, i) => (out[values.length + i] = pick(loop)));
    return out;
  };
  return {...surface, positions: filled.positions, normals: filled.normals, triangles: filled.triangles,
    cells: extend(surface.cells, (loop) => surface.cells[loop[0]])!, ownerCells: extend(surface.ownerCells, (loop) => surface.ownerCells[loop[0]])!,
    bodyCode: extend(surface.bodyCode, (loop) => surface.bodyCode![loop[0]]),
    bodyEdgeMm: extend(surface.bodyEdgeMm, (loop) => loop.reduce((sum, v) => sum + surface.bodyEdgeMm![v], 0) / loop.length)};
}

export class OperativeSourceScene {
  readonly group = new THREE.Group();
  readonly cavity = new CavityClip();
  readonly asset: OperativeSourceAsset;
  /** Every layer name this scene draws (its voxel meshes are not drawn while it is active). */
  readonly replaced = new Set<string>();
  private readonly affine: Affine;
  private readonly shape: Shape3;
  private readonly outwardRas: THREE.Vector3;
  private readonly outwardVoxel: Point3;
  private readonly pieces: Piece[] = [];
  private readonly bodies = new Map<SourceBodyName, BodyDisplay>();
  private readonly voxel = new Map<string, {positions: Float32Array; normals: Float32Array}>();
  /** The asset's surfaces as drawn (the assumed dura with its small gaps closed). */
  private readonly surfaces = new Map<SourceSurface['name'], SourceSurface>();
  private tracts: Array<{lines: THREE.LineSegments<THREE.BufferGeometry, THREE.LineBasicMaterial>; layer: string}> = [];
  private cavityWall?: Piece;
  private cavityKey = '';
  private brainRegions = new Set<number>();
  private owner?: Uint16Array;
  private regionName = new Map<number, string>();
  readonly timings: Record<string, number> = {};
  readonly vesselSummary: {unknownMm: number; arteryMm: number; veinMm: number};

  private readonly material: SourceMaterialFactory;
  constructor(asset: OperativeSourceAsset, layers: ReadonlyArray<{name: string; kind?: string; displayKind?: string}>, outwardRas: Point3,
    material: SourceMaterialFactory) {
    const started = performance.now();
    this.material = material;
    this.asset = asset;
    this.affine = asset.grid.affine;
    this.shape = asset.grid.shape;
    this.group.name = 'operative-source-surfaces';
    this.outwardRas = new THREE.Vector3(...outwardRas).normalize();
    this.outwardVoxel = directionToVoxel(outwardRas, this.affine);
    const byName = new Map(layers.map((l) => [l.name, l]));
    const kindOf = (name: string, fallback: string) => byName.get(name)?.displayKind ?? byName.get(name)?.kind ?? fallback;
    const layerFor = (surface: SourceSurface) => surface.layers.find((name) => byName.has(name)) ?? surface.layers[0] ?? surface.name;
    for (const surface of asset.surfaces.values()) for (const name of surface.layers) if (byName.has(name)) this.replaced.add(name);
    for (const t of asset.tracts) if (byName.has(t.layer)) this.replaced.add(t.layer);
    const vesselLayers = layers.filter((l) => ['artery', 'vein', 'vessel', 'vessels', 'sinus'].includes(l.displayKind ?? l.kind ?? ''));
    if (asset.vessels) for (const l of vesselLayers) this.replaced.add(l.name);
    // Head layers the skin and skull replace (soft tissue between scalp and bone).
    const extracranial = layers.filter((l) => ['galea', 'extracranial', 'tissue'].includes(l.displayKind ?? ''));
    if (asset.surfaces.has('skin') && asset.surfaces.has('skullOuter')) for (const l of extracranial) this.replaced.add(l.name);
    for (const raw of asset.surfaces.values()) this.surfaces.set(raw.name, raw.name === 'dura' ? closeDuraGaps(raw) : raw);
    for (const surface of this.surfaces.values()) {
      // The assumed dura sheet is coarser than the cortex, whose crowns would show through it: draw it a little outward.
      const lift = surface.name === 'dura' ? DURA_DISPLAY_LIFT_MM : 0;
      const raised = lift ? surface.positions.map((v, i) => v + lift * surface.normals[i]) : surface.positions;
      const positions = rasToVoxel(raised, this.affine), normals = normalsToVoxel(surface.normals, this.affine);
      this.voxel.set(surface.name, {positions, normals});
      const layer = layerFor(surface), kind = surface.name === 'skullOuter' || surface.name === 'skullInner' ? 'skull' : kindOf(layer, surface.name);
      const brain = ['pial', 'white', 'tumourRim', 'tumourCore'].includes(surface.name);
      const piece = this.piece(geometryOf(positions, normals, surface.triangles), layer, kind, brain ? 'surface' : 'surface', surface.ownerCells.map((c, i) => (c >= 0 ? c : surface.cells[i])) as Int32Array, brain ? 'brain' : 'head');
      piece.mesh.name = 'source-surface:' + surface.name;
      // Under the near-coaxial lamp the large static sheets cast no visible shadow; leave them out of the shadow pass.
      piece.mesh.castShadow = !STATIC_SHADOWLESS.has(surface.name);
      if (surface.body) {
        const display = this.bodyDisplay(surface.body);
        display.intact.push(piece);
      }
    }
    const v = asset.vessels;
    const mm = {unknownMm: 0, arteryMm: 0, veinMm: 0};
    if (v) {
      const positions = rasToVoxel(v.wallPositions, this.affine), normals = normalsToVoxel(v.wallNormals, this.affine);
      const segments = v.segmentKind.length;
      const kindOfVertex = (i: number) => (v.wallSegment[i] < segments ? v.segmentKind[v.wallSegment[i]] : 0);
      const named = (kind: string) => vesselLayers.find((l) => (l.displayKind ?? l.kind) === kind)?.name;
      const unknownLayer = vesselLayers.find((l) => !['artery', 'vein'].includes(l.displayKind ?? l.kind ?? ''))?.name ?? 'Vessel (source estimate)';
      for (const [code, kind, layer] of [[1, 'artery', named('artery') ?? unknownLayer], [2, 'vein', named('vein') ?? unknownLayer], [0, 'unknown', unknownLayer]] as const) {
        const indices: number[] = [];
        for (let t = 0; t < v.wallTriangles.length; t += 3) if (kindOfVertex(v.wallTriangles[t]) === code) indices.push(v.wallTriangles[t], v.wallTriangles[t + 1], v.wallTriangles[t + 2]);
        if (!indices.length) continue;
        const piece = this.piece(geometryOf(positions, normals, Uint32Array.from(indices)), layer, kind === 'unknown' ? 'vessel' : kind, 'vessel', v.wallCells, 'vessel');
        piece.mesh.name = 'source-vessels:' + kind;
        piece.mesh.userData.vesselKind = kind;
        (piece.mesh.material as THREE.MeshStandardMaterial).color.set(SOURCE_VESSEL_COLORS[kind]);
      }
      for (let s = 0; s < segments; s++) {
        let length = 0;
        for (let i = v.centrelineOffsets[s] + 1; i < v.centrelineOffsets[s + 1]; i++)
          length += Math.hypot(...[0, 1, 2].map((a) => v.centrelinePoints[3 * i + a] - v.centrelinePoints[3 * i - 3 + a]));
        if (v.segmentKind[s] === 1) mm.arteryMm += length; else if (v.segmentKind[s] === 2) mm.veinMm += length; else mm.unknownMm += length;
      }
    }
    this.vesselSummary = mm;
    asset.tracts.forEach((tract, index) => {
      const voxel = rasToVoxel(tract.points, this.affine), segments: number[] = [];
      for (let s = 0; s + 1 < tract.offsets.length; s++)
        for (let i = tract.offsets[s]; i + 1 < tract.offsets[s + 1]; i++) segments.push(...voxel.subarray(3 * i, 3 * i + 6));
      const geometry = new THREE.BufferGeometry();
      geometry.setAttribute('position', new THREE.Float32BufferAttribute(segments, 3));
      const lines = new THREE.LineSegments(geometry, new THREE.LineBasicMaterial({color: tractColor(index), transparent: true, opacity: 0.75, depthWrite: false}));
      lines.name = 'source-tract:' + tract.bundle;
      lines.raycast = () => {};
      placeInSourceFrame(lines, this.affine);
      this.group.add(lines);
      this.tracts.push({lines, layer: tract.layer});
    });
    this.timings.buildMs = performance.now() - started;
  }

  private piece(geometry: THREE.BufferGeometry, layer: string, kind: string, role: SourceRole, cells: Int32Array, group: Piece['group']): Piece {
    const index = geometry.index?.array, count = geometry.getAttribute('position').count;
    let top = -1;
    if (index) for (let i = 0; i < index.length; i++) if (index[i] > top) top = index[i];
    for (const [name, attribute] of Object.entries(geometry.attributes))
      if (attribute.count < count || top >= attribute.count) console.error(`source surface ${layer}/${kind}/${role}: attribute ${name} has ${attribute.count} of ${count} vertices, index reaches ${top}`);
    const material = this.material(layer, kind, role);
    if (group !== 'head' || role === 'cavity') this.cavity.install(material);
    const mesh = new THREE.Mesh(geometry, material);
    placeInSourceFrame(mesh, this.affine);
    mesh.castShadow = true;
    mesh.receiveShadow = true;
    mesh.userData.compartment = layer;
    mesh.userData.sourceSurface = true;
    mesh.userData.sourceCells = cells;
    this.group.add(mesh);
    const piece = {mesh, layer, kind, cells, group};
    this.pieces.push(piece);
    return piece;
  }

  private bodyDisplay(name: SourceBodyName) {
    let display = this.bodies.get(name);
    if (!display) {
      display = {intact: [], fixed: [], flaps: [], built: false, displaced: false, lines: new THREE.Group()};
      display.lines.name = 'source-outline:' + name;
      this.group.add(display.lines);
      this.bodies.set(name, display);
    }
    return display;
  }

  /** Bind the engine's owner grid: region names and which regions are brain tissue (for the cavity wall). */
  setOwner(owner: Uint16Array | undefined, regions: ReadonlyArray<{id: number; name: string; material: string}>) {
    this.owner = owner;
    this.regionName = new Map(regions.map((r) => [r.id, r.name]));
    this.brainRegions = new Set(regions.filter((r) => (r.material === 'soft_tissue' || r.material === 'tumor' || r.material === 'functional_region') &&
      !/scalp|galea|subcutaneous|head tissue|skin/i.test(r.name)).map((r) => r.id));
  }

  /** Build the cut halves of one body's surfaces the first time the body moves. */
  private buildSplit(name: SourceBodyName, display: BodyDisplay, body: OperativeExposureBody, view: SurgicalView) {
    const started = performance.now();
    const outline = this.asset.outlines[name];
    const gap = KERF_HALF_MM[name];
    const surfaces = name === 'scalp' ? ['skin'] : name === 'bone' ? ['skullOuter', 'skullInner'] : ['dura'];
    const innerLoop = outline?.inner ? new PlanarLoop(rasToVoxel(outline.inner.points, this.affine), this.outwardVoxel) : null;
    const outerLoop = outline ? new PlanarLoop(rasToVoxel(outline.outer.points, this.affine), this.outwardVoxel) : null;
    const hingeAt = (p: ArrayLike<number>) => (outline && outerLoop ? outline.outer.hinge[outerLoop.query(p).index] : 0);
    for (const surfaceName of surfaces) {
      const surface = this.surfaces.get(surfaceName as SourceSurface['name']);
      if (!surface?.bodyEdgeMm) continue;
      const {positions, normals} = this.voxel.get(surface.name)!;
      const edge = surface.bodyEdgeMm;
      const hinge = new Uint8Array(edge.length);
      if (outline) for (let i = 0; i < edge.length; i++) if (Math.abs(edge[i]) < 3 * gap + 1.5 && hingeAt(positions.subarray(3 * i, 3 * i + 3))) hinge[i] = 1;
      const {flap, fixed} = splitSheet(positions, normals, surface.triangles, edge, gap, hinge);
      const intact = display.intact.find((p) => p.mesh.name === 'source-surface:' + surface.name)!;
      const fixedPiece = this.piece(geometryOf(fixed.positions, fixed.normals, fixed.indices), intact.layer, intact.kind, 'surface', cellsOfPart(fixed, surface.ownerCells.map((c, i) => (c >= 0 ? c : surface.cells[i])) as Int32Array), 'head');
      fixedPiece.mesh.castShadow = !STATIC_SHADOWLESS.has(surface.name);
      fixedPiece.mesh.name = `source-fixed:${surface.name}`;
      display.fixed.push(fixedPiece);
      const flapPiece = this.flapPiece(flap, partAttribute(flap, edge), intact.layer, intact.kind, 'surface', cellsOfPart(flap, surface.ownerCells.map((c, i) => (c >= 0 ? c : surface.cells[i])) as Int32Array), body, view, (i) => flap.source[i] < 0 || edge[flap.source[i]] > 0);
      flapPiece.piece.mesh.name = `source-flap:${surface.name}`;
      display.flaps.push(flapPiece);
      // Cut faces: a smooth wall from the outer surface's cut edge down to the inner outline.
      const outerSurface = name === 'scalp' ? surface.name === 'skin' : name === 'bone' ? surface.name === 'skullOuter' : false;
      if (outerSurface && innerLoop) {
        const wallLayer = name === 'bone' ? (this.layerOfKind('diploe') ?? intact.layer) : (this.layerOfKind('galea') ?? intact.layer);
        const wallKind = name === 'bone' ? 'diploe' : 'extracranial'; // the cut scalp shows dermis and fat, not galea
        for (const [part, side] of [[fixed, 'fixed'], [flap, 'flap']] as const) {
          if (!part.cutEdges.length) continue;
          const wall = kerfWall(part, innerLoop, this.outwardVoxel);
          const geometry = new THREE.BufferGeometry();
          geometry.setAttribute('position', new THREE.BufferAttribute(wall.positions, 3));
          geometry.setIndex(new THREE.BufferAttribute(wall.indices, 1));
          geometry.computeVertexNormals();
          const normalsWall = geometry.getAttribute('normal').array as Float32Array;
          if (side === 'fixed') {
            const piece = this.piece(geometryOf(wall.positions, normalsWall, wall.indices), wallLayer, wallKind, 'wall', voxelCells(wall.positions, this.shape), 'head');
            piece.mesh.name = `source-kerf-wall:${name}:fixed`;
            display.fixed.push(piece);
          } else {
            const wallPart: MeshPart = {positions: wall.positions, normals: normalsWall, indices: wall.indices, source: new Int32Array(wall.positions.length / 3).fill(-1),
              between: new Float32Array(wall.positions.length), cutEdges: new Uint32Array()};
            const flapWall = this.flapPiece(wallPart, null, wallLayer, wallKind, 'wall', voxelCells(wall.positions, this.shape), body, view, () => true);
            flapWall.piece.mesh.name = `source-kerf-wall:${name}:flap`;
            display.flaps.push(flapWall);
          }
          geometry.dispose();
        }
      }
    }
    // The scalp flap's underside: the outer table under the scalp outline, carried with the scalp.
    if (name === 'scalp' && innerLoop) {
      const skull = this.surfaces.get('skullOuter');
      const voxel = this.voxel.get('skullOuter');
      if (skull && voxel) {
        const bounds = [Infinity, -Infinity, Infinity, -Infinity];
        for (let i = 0; i < innerLoop.u.length; i++) {
          bounds[0] = Math.min(bounds[0], innerLoop.u[i]); bounds[1] = Math.max(bounds[1], innerLoop.u[i]);
          bounds[2] = Math.min(bounds[2], innerLoop.v[i]); bounds[3] = Math.max(bounds[3], innerLoop.v[i]);
        }
        const distance = new Float32Array(voxel.positions.length / 3).fill(-1e3);
        const {e1, e2} = innerLoop.basis;
        for (let i = 0; i < distance.length; i++) {
          const p = voxel.positions.subarray(3 * i, 3 * i + 3);
          const u = p[0] * e1[0] + p[1] * e1[1] + p[2] * e1[2], v = p[0] * e2[0] + p[1] * e2[1] + p[2] * e2[2];
          if (u < bounds[0] - 3 || u > bounds[1] + 3 || v < bounds[2] - 3 || v > bounds[3] + 3) continue;
          distance[i] = innerLoop.query(p).distance;
        }
        const {flap} = splitSheet(voxel.positions, voxel.normals, skull.triangles, distance, gap, undefined,
          (t) => distance[skull.triangles[3 * t]] > -5);
        const layer = this.layerOfKind('galea') ?? 'Subcutaneous tissue and galea';
        // Its normals face the skull; the flap's underside faces away from the skin.
        for (let i = 0; i < flap.normals.length; i++) flap.normals[i] = -flap.normals[i];
        const under = this.flapPiece(flap, null, layer, 'galea', 'underside', cellsOfPart(flap, skull.ownerCells.map((c, i) => (c >= 0 ? c : skull.cells[i])) as Int32Array), body, view, () => true);
        under.piece.mesh.name = 'source-flap:scalp-underside';
        display.flaps.push(under);
      }
    }
    display.built = true;
    this.timings[`split-${name}Ms`] = performance.now() - started;
  }

  private layerOfKind(kind: string) {
    for (const piece of this.pieces) if (piece.kind === kind) return piece.layer;
    return [...this.replaced].find((name) => kind === 'galea' ? /galea|subcutaneous/i.test(name) : kind === 'diploe' ? /diplo/i.test(name) : false);
  }

  private flapPiece(part: MeshPart, _edge: Float32Array | null, layer: string, kind: string, role: SourceRole, cells: Int32Array,
    body: OperativeExposureBody, view: SurgicalView, bound: (i: number) => boolean): Flap {
    const byId = new Map(body.cells.map((id, i) => [id, i]));
    const binding = bindToBody(part.positions, bound, byId, view.scene.shape);
    const restVoxel = part.positions.slice();
    const restRas = voxelToRas(restVoxel, this.affine);
    // Voxel normals back to RAS (affine linear part; spacing is uniform on the working grid).
    const restNormalsRas = new Float32Array(part.normals.length);
    for (let i = 0; i < part.normals.length; i += 3) {
      let len = 0;
      for (let a = 0; a < 3; a++) {
        const value = this.affine[a][0] * part.normals[i] + this.affine[a][1] * part.normals[i + 1] + this.affine[a][2] * part.normals[i + 2];
        restNormalsRas[i + a] = value;
        len += value * value;
      }
      len = Math.sqrt(len) || 1;
      for (let a = 0; a < 3; a++) restNormalsRas[i + a] /= len;
    }
    const geometry = geometryOf(part.positions.slice(), part.normals.slice(), part.indices.slice(), true);
    geometry.setAttribute('finishRestOffset', new THREE.BufferAttribute(new Float32Array(part.positions.length), 3).setUsage(THREE.DynamicDrawUsage));
    const piece = this.piece(geometry, layer, kind, role, cells, 'head');
    // The first (largest) anchor of each vertex; a flap triangle whose anchors were all removed is not drawn.
    const primary = new Int32Array(part.positions.length / 3).fill(-1);
    for (let v = 0; v < primary.length; v++) {
      let best = -1, weight = -1;
      for (let i = binding.offsets[v]; i < binding.offsets[v + 1]; i++) if (binding.weights[i] > weight) { weight = binding.weights[i]; best = binding.anchors[i]; }
      primary[v] = best;
    }
    return {piece, restRas, restNormalsRas, restVoxel, binding, primary, baseIndex: part.indices.slice()};
  }

  /** Apply the engine state: body motion, cut seams, planned outlines and removed tissue. */
  update(view: SurgicalView) {
    const started = performance.now();
    const exposure = view.operativeExposureV1;
    const removed = new Set(view.removed);
    if (exposure) for (const name of ['scalp', 'bone', 'dura'] as const) {
      const body = exposure.bodies[name];
      const display = this.bodies.get(name);
      if (!display || !body) continue;
      const displaced = bodyIsDisplaced(body, view.scene.shape, view.scene.affine);
      display.displaced = displaced;
      if (displaced && !display.built) this.buildSplit(name, display, body, view);
      const members = body.cells.join(',');
      if (display.members !== members) { display.transport = new ExposureBodyTransport(body, view.scene); display.members = members; }
      const changed = display.transport!.update(body.centersMm);
      if (displaced) for (const flap of display.flaps) this.moveFlap(flap, display.transport!, changed, removed, body);
      this.updateLines(name, display, body, view, exposure);
    }
    this.updateCavity(view, removed);
    this.refreshVisibility();
    this.timings.updateMs = performance.now() - started;
  }

  private moveFlap(flap: Flap, transport: ExposureBodyTransport, changed: boolean, removed: ReadonlySet<number>, body: OperativeExposureBody) {
    const geometry = flap.piece.mesh.geometry;
    const position = geometry.getAttribute('position') as THREE.BufferAttribute, normal = geometry.getAttribute('normal') as THREE.BufferAttribute;
    if (changed || !flap.piece.mesh.userData.transported) {
      transportVertices(flap.binding, transport, flap.restRas, flap.restNormalsRas, this.affine, position.array as Float32Array, normal.array as Float32Array);
      const offset = geometry.getAttribute('finishRestOffset') as THREE.BufferAttribute, o = offset.array as Float32Array, p = position.array as Float32Array;
      for (let i = 0; i < o.length; i++) o[i] = flap.restVoxel[i] - p[i];
      position.needsUpdate = normal.needsUpdate = offset.needsUpdate = true;
      geometry.computeBoundingSphere();
      geometry.computeBoundingBox();
      flap.piece.mesh.userData.transported = true;
    }
    // Body cells the engine removed (a kerf, an evacuated flap) take their part of the flap with them.
    const gone = body.cells.filter((id) => removed.has(id));
    const key = gone.length + ':' + (gone[0] ?? '') + ':' + (gone.at(-1) ?? '');
    if (key === flap.goneKey) return;
    flap.goneKey = key;
    const goneSet = new Set(gone.map((id) => body.cells.indexOf(id)));
    const base = flap.baseIndex, kept: number[] = [];
    for (let t = 0; t < base.length; t += 3) {
      const a = flap.primary[base[t]], b = flap.primary[base[t + 1]], c = flap.primary[base[t + 2]];
      if ((a >= 0 && goneSet.has(a)) && (b >= 0 && goneSet.has(b)) && (c >= 0 && goneSet.has(c))) continue;
      kept.push(base[t], base[t + 1], base[t + 2]);
    }
    geometry.setIndex(new THREE.BufferAttribute(Uint32Array.from(kept), 1));
  }

  private updateLines(name: SourceBodyName, display: BodyDisplay, body: OperativeExposureBody, view: SurgicalView,
    exposure: NonNullable<SurgicalView['operativeExposureV1']>) {
    const outline = this.asset.outlines[name]?.outer;
    if (!outline) return;
    // The layer above must be open before this outline is visible at all.
    const above = name === 'scalp' ? null : name === 'bone' ? exposure.bodies.scalp : exposure.bodies.bone;
    const exposed = !above || bodyIsDisplaced(above, view.scene.shape, view.scene.affine);
    const key = JSON.stringify([display.displaced, exposed, body.cutCount, body.seamEdges.length, body.seamEdges[0], body.seamEdges.at(-1)]);
    if (key === display.lineKey) return;
    display.lineKey = key;
    for (const child of [...display.lines.children]) {
      child.removeFromParent();
      const mesh = child as THREE.Mesh;
      mesh.geometry.dispose();
      (mesh.material as THREE.Material).dispose();
    }
    if (display.displaced || !exposed) return;
    const cut = cutOutlinePoints(outline, body.cutCount >= body.edgeCount && body.edgeCount > 0 ? [] : body.seamEdges, view.scene.shape, view.scene.affine);
    if (body.cutCount >= body.edgeCount && body.edgeCount > 0) for (let i = 0; i < cut.length; i++) cut[i] = outline.hinge[i] ? 0 : 1;
    const lift = name === 'scalp' ? 0.2 : 0.15;
    const point = (i: number) => [0, 1, 2].map((a) => outline.points[3 * i + a] + this.outwardRas.getComponent(a) * lift);
    const planned = new Uint8Array(cut.length);
    // The hinge (the flap's uncut base) is not part of the planned cut.
    for (let i = 0; i < cut.length; i++) planned[i] = cut[i] || outline.hinge[i] ? 0 : 1;
    for (const run of runs(cut)) display.lines.add(Object.assign(tube(run.map(point), name === 'scalp' ? 0.22 : 0.3, SEAM_COLORS[name], false), {name: `source-seam:${name}`}));
    if (planned.every(Boolean)) display.lines.add(Object.assign(tube([...Array(cut.length).keys()].map(point), 0.3, PLANNED_COLOR, true), {name: `source-planned:${name}`}));
    else for (const run of runs(planned)) display.lines.add(Object.assign(tube(run.map(point), 0.25, PLANNED_COLOR, false), {name: `source-planned:${name}`}));
  }

  /** Removed brain, tumour and vessel cells: the clip field and the cavity wall. */
  private updateCavity(view: SurgicalView, removed: ReadonlySet<number>) {
    const progress = view.cusaProgressV1 ?? null;
    const owner = this.owner;
    const include = (cell: number) => !owner || this.brainRegions.has(owner[cell]) || view.scene.regions.find((r) => r.id === owner[cell])?.material === 'vessel';
    const occupancy = removalOccupancy(removed, progress, include);
    let key = occupancy.size + ':' + (progress?.cells.length ?? 0);
    if (occupancy.size) {
      let hash = 0;
      for (const [cell, value] of occupancy) hash = (hash * 31 + cell + Math.round(value * 1000)) | 0;
      key += ':' + hash;
    }
    if (key === this.cavityKey) return;
    this.cavityKey = key;
    const started = performance.now();
    const field = cavityField(occupancy, this.shape);
    this.cavity.set(field);
    if (this.cavityWall) {
      this.cavityWall.mesh.removeFromParent();
      this.cavityWall.mesh.geometry.dispose();
      this.pieces.splice(this.pieces.indexOf(this.cavityWall), 1);
      this.cavityWall = undefined;
    }
    if (field) {
      const [nx, ny, nz] = this.shape;
      const tissue = (p: readonly number[]) => {
        const x = Math.round(p[0]), y = Math.round(p[1]), z = Math.round(p[2]);
        if (x < 0 || y < 0 || z < 0 || x >= nx || y >= ny || z >= nz) return false;
        const cell = (x * ny + y) * nz + z;
        return !removed.has(cell) && (!owner || this.brainRegions.has(owner[cell]));
      };
      // Keep the wall only where tissue remains just outside it (not where the cavity opens to the surface).
      const surface = cavitySurface(field, CAVITY_LEVEL, (p, n) => tissue([p[0] + 0.75 * n[0], p[1] + 0.75 * n[1], p[2] + 0.75 * n[2]]));
      if (surface.indices.length) {
        const cells = new Int32Array(surface.positions.length / 3);
        for (let i = 0; i < cells.length; i++) {
          const q = [0, 1, 2].map((a) => Math.round(surface.positions[3 * i + a] + 0.75 * surface.normals[3 * i + a]));
          cells[i] = q.some((v, a) => v < 0 || v >= this.shape[a]) ? -1 : (q[0] * ny + q[1]) * nz + q[2];
        }
        const layer = [...this.replaced].find((n) => /cortex/i.test(n)) ?? 'Cerebral cortex';
        const piece = this.piece(geometryOf(surface.positions, surface.normals, surface.indices), layer, 'grayMatter', 'cavity', cells, 'brain');
        piece.mesh.name = 'source-cavity-wall';
        this.cavityWall = piece;
      }
    }
    this.timings.cavityMs = performance.now() - started;
  }

  /** Meshes of brain tissue that move with a brain-shift field (by layer name), for the scenario display. */
  scenarioSurfaces(): Map<string, THREE.Mesh> {
    const out = new Map<string, THREE.Mesh>();
    for (const piece of this.pieces) if ((piece.group === 'brain' || piece.group === 'vessel') && piece.mesh.name.startsWith('source-') && !out.has(piece.layer) && piece !== this.cavityWall) out.set(piece.layer, piece.mesh);
    return out;
  }
  get cavityMesh() { return this.cavityWall?.mesh; }

  private visibility: (layer: string) => boolean = () => true;
  private tractsShown = false;
  appearance(isVisible: (layer: string) => boolean, apply: (mesh: THREE.Mesh, layer: string, kind: string) => void, tractsShown: boolean, planes: THREE.Plane[]) {
    this.visibility = isVisible;
    this.tractsShown = tractsShown;
    for (const piece of this.pieces) {
      apply(piece.mesh, piece.layer, piece.kind);
      const material = piece.mesh.material as THREE.MeshStandardMaterial;
      const kind = piece.mesh.userData.vesselKind as keyof typeof SOURCE_VESSEL_COLORS | undefined;
      if (kind) material.color.set(SOURCE_VESSEL_COLORS[kind]);
      if (material.clippingPlanes?.length !== planes.length) material.needsUpdate = true;
      material.clippingPlanes = planes;
    }
    for (const {lines} of this.tracts) {
      if (lines.material.clippingPlanes?.length !== planes.length) lines.material.needsUpdate = true;
      lines.material.clippingPlanes = planes;
    }
    for (const display of this.bodies.values()) display.lines.traverse((o) => {
      const mesh = o as THREE.Mesh;
      if (mesh.isMesh) (mesh.material as THREE.Material).clippingPlanes = planes;
    });
    this.refreshVisibility();
  }

  private refreshVisibility() {
    const state = new Map<THREE.Mesh, boolean>();
    for (const display of this.bodies.values()) {
      for (const p of display.intact) state.set(p.mesh, !display.displaced);
      for (const p of display.fixed) state.set(p.mesh, display.displaced);
      for (const f of display.flaps) state.set(f.piece.mesh, display.displaced);
      display.lines.visible = !!this.visibility(display.intact[0]?.layer ?? '');
    }
    for (const piece of this.pieces) piece.mesh.visible = (state.get(piece.mesh) ?? true) && this.visibility(piece.layer);
    for (const {lines, layer} of this.tracts) lines.visible = this.tractsShown && this.visibility(layer);
  }

  /** Pickable meshes (visible surfaces, not lines). */
  get picks() { return this.pieces.map((p) => p.mesh).filter((m) => m.visible); }

  /** The engine cell a hit resolves to: the hit triangle's nearest vertex cell; its current centre and owner layer. */
  resolve(hit: THREE.Intersection, centre: (cell: number) => Point3): {cell: number; point: Point3; layer: string} | null {
    const mesh = hit.object as THREE.Mesh, cells = mesh.userData.sourceCells as Int32Array | undefined;
    if (!cells || !hit.face) return null;
    const position = mesh.geometry.getAttribute('position');
    const local = mesh.worldToLocal(hit.point.clone());
    if (mesh !== this.cavityWall?.mesh && this.cavity.removed(local.toArray())) return null;
    let best = -1, distance = Infinity;
    for (const v of [hit.face.a, hit.face.b, hit.face.c]) {
      const d = local.distanceToSquared(new THREE.Vector3().fromBufferAttribute(position, v));
      if (cells[v] >= 0 && d < distance) { distance = d; best = cells[v]; }
    }
    if (best < 0) return null;
    const region = this.owner?.[best];
    return {cell: best, point: centre(best), layer: (region ? this.regionName.get(region) : undefined) ?? mesh.userData.compartment};
  }

  /** Nearest vessel centreline point to a RAS point (within reach), for wound markers. */
  nearestVesselPoint(p: Point3, reachMm = 4): Point3 | null {
    const v = this.asset.vessels;
    if (!v) return null;
    let best = reachMm * reachMm, out: Point3 | null = null;
    for (let i = 0; i < v.centrelineRadii.length; i++) {
      const d = (v.centrelinePoints[3 * i] - p[0]) ** 2 + (v.centrelinePoints[3 * i + 1] - p[1]) ** 2 + (v.centrelinePoints[3 * i + 2] - p[2]) ** 2;
      if (d < best) { best = d; out = [v.centrelinePoints[3 * i], v.centrelinePoints[3 * i + 1], v.centrelinePoints[3 * i + 2]]; }
    }
    return out;
  }

  summary() {
    const triangles = this.pieces.reduce((s, p) => s + (p.mesh.visible ? (p.mesh.geometry.index?.count ?? 0) / 3 : 0), 0);
    return {asset: this.asset.assetName, loadMs: Math.round(this.asset.loadMs), buildMs: Math.round(this.timings.buildMs ?? 0),
      bodies: Object.fromEntries([...this.bodies].map(([k, d]) => [k, {displaced: d.displaced, built: d.built}])),
      visibleTriangles: triangles, cavityTriangles: (this.cavityWall?.mesh.geometry.index?.count ?? 0) / 3,
      vesselsUnknownMm: Math.round(this.vesselSummary.unknownMm), timings: Object.fromEntries(Object.entries(this.timings).map(([k, v]) => [k, Math.round(v)]))};
  }

  dispose() {
    this.group.traverse((o) => {
      const mesh = o as THREE.Mesh;
      if (mesh.geometry) mesh.geometry.dispose();
      const m = mesh.material as THREE.Material | THREE.Material[] | undefined;
      if (Array.isArray(m)) m.forEach((x) => x.dispose()); else m?.dispose();
    });
    this.cavity.dispose();
    this.group.removeFromParent();
  }
}

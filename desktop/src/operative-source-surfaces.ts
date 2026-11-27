/** Source-resolution display surfaces of a host rehearsal (`operativeSourceSurfaces`).
 *
 * The engine stays authoritative for cells, counts and picking. These are
 * read-only meshes built from the acquired and estimated source images of the
 * same case version: positions and normals in RAS millimetres, and for each
 * vertex the engine working-grid cell it lies in, so the renderer can apply the
 * engine's state to them. Nothing here is an engine input. */
import type {Affine, Shape3} from './viewer/coordinates';

export const SOURCE_SURFACE_NAMES = ['skin', 'skullOuter', 'skullInner', 'dura', 'pial', 'white', 'tumourRim', 'tumourCore'] as const;
export type SourceSurfaceName = (typeof SOURCE_SURFACE_NAMES)[number];
export type SourceBodyName = 'scalp' | 'bone' | 'dura';
export const SOURCE_BODY_CODES: Record<SourceBodyName, number> = {scalp: 1, bone: 2, dura: 3};

export interface SourceSurface {
  name: SourceSurfaceName;
  layers: string[];
  body: SourceBodyName | null;
  origin: string;
  positions: Float32Array;
  normals: Float32Array;
  triangles: Uint32Array;
  cells: Int32Array;
  ownerCells: Int32Array;
  bodyCode?: Uint8Array;
  bodyEdgeMm?: Float32Array;
}
export interface SourceOutline {points: Float32Array; hinge: Uint8Array}
export interface SourceVessels {
  centrelinePoints: Float32Array;
  centrelineRadii: Float32Array;
  centrelineOffsets: Uint32Array;
  segmentKind: Uint8Array;
  segmentInsideBox: Uint8Array;
  wallPositions: Float32Array;
  wallNormals: Float32Array;
  wallTriangles: Uint32Array;
  wallSegment: Uint32Array;
  wallCells: Int32Array;
  origin: string;
  mmByKind: Record<string, number>;
}
export interface SourceTract {bundle: string; layer: string; points: Float32Array; offsets: Uint32Array; origin: string}
export interface OperativeSourceAsset {
  caseHash: string;
  assetName: string;
  scope: string;
  grid: {shape: Shape3; affine: Affine};
  surfaces: Map<SourceSurfaceName, SourceSurface>;
  outlines: Partial<Record<SourceBodyName, {outer: SourceOutline; inner?: SourceOutline}>>;
  vessels: SourceVessels | null;
  tracts: SourceTract[];
  /** Assumed thickness of the dural membrane (mm), when the asset declares one. */
  duraThicknessMm?: number;
  bytes: number;
  loadMs: number;
}
export type SourceSurfaceStatus =
  | {status: 'loading'; caseHash: string}
  | {status: 'available'; caseHash: string; asset: OperativeSourceAsset}
  | {status: 'unavailable'; caseHash: string; code: string; reason: string};

type Descriptor = {assetId?: string; data?: Uint8Array; dtype: string; shape: number[]; byteOrder: string; order: string; byteLength: number; sha256: string};
const ARRAY_TYPES = {float32: Float32Array, int32: Int32Array, uint32: Uint32Array, uint8: Uint8Array} as const;
type ArrayType = keyof typeof ARRAY_TYPES;
type Api = {operativeSourceSurfaces?: (args: {caseHash: string}) => Promise<unknown>; readAsset: (id: string) => Promise<Uint8Array>};
const MAX_SOURCE_BYTES = 512 * 1024 ** 2;

function record(value: unknown, what: string): Record<string, unknown> {
  if (!value || typeof value !== 'object' || Array.isArray(value)) throw Error(`Invalid source-surface ${what}`);
  return value as Record<string, unknown>;
}

/** Check one transfer descriptor against the declared dtype and trailing width. */
export function checkedDescriptor(raw: unknown, key: string, dtype: ArrayType, width: number): Descriptor {
  const d = record(raw, key) as unknown as Descriptor;
  const bytes = ARRAY_TYPES[dtype].BYTES_PER_ELEMENT;
  if (d.dtype !== dtype || d.byteOrder !== 'little' || d.order !== 'C' || !(d.data instanceof Uint8Array || (typeof d.assetId === 'string' && d.assetId)) ||
    !/^(sha256:)?[a-f0-9]{64}$/.test(String(d.sha256)) || !Array.isArray(d.shape) || d.shape.length !== (width === 1 ? 1 : 2) ||
    !d.shape.every((n) => Number.isSafeInteger(n) && n > 0) || (width > 1 && d.shape[1] !== width) ||
    d.byteLength !== d.shape[0] * width * bytes || d.byteLength > MAX_SOURCE_BYTES) throw Error(`Invalid source-surface array ${key}`);
  return d;
}

/** Request and read the source surfaces of an open host-rehearsal case. */
export async function loadOperativeSourceSurfaces(api: Api, caseHash: string): Promise<SourceSurfaceStatus> {
  const started = performance.now();
  if (!api.operativeSourceSurfaces) return {status: 'unavailable', caseHash, code: 'NOT_SUPPORTED', reason: 'this build has no source-surface operation'};
  const reply = record(await api.operativeSourceSurfaces({caseHash}), 'reply');
  if (reply.caseHash !== caseHash) throw Error('Source surfaces belong to another case');
  if (reply.status !== 'available')
    return {status: 'unavailable', caseHash, code: String(reply.code ?? 'UNAVAILABLE'), reason: String(reply.reason ?? 'no source-surface asset')};
  const header = record(reply.header, 'header'), arrays = record(reply.arrays, 'arrays');
  let bytes = 0;
  const queue: Array<() => Promise<void>> = [];
  const read = <T extends ArrayType>(key: string, dtype: T, width: number, sink: (value: InstanceType<(typeof ARRAY_TYPES)[T]>) => void) => {
    const d = checkedDescriptor(arrays[key], key, dtype, width);
    bytes += d.byteLength;
    if (bytes > MAX_SOURCE_BYTES) throw Error('Source-surface asset exceeds the desktop limit');
    queue.push(async () => {
      // Each array is a transfer asset; the main process checks its bytes against the engine's digest.
      const raw = d.data ?? (await api.readAsset(d.assetId!));
      if (raw.byteLength !== d.byteLength) throw Error(`Incomplete source-surface transfer ${key}`);
      const Type = ARRAY_TYPES[dtype];
      sink(new Type(raw.slice().buffer) as InstanceType<(typeof ARRAY_TYPES)[T]>);
    });
  };
  const grid = record(header.workingGrid, 'working grid');
  const shape = grid.shape as Shape3, affine = grid.affine as Affine;
  if (!Array.isArray(shape) || shape.length !== 3 || !shape.every((n) => Number.isSafeInteger(n) && n > 0) ||
    !Array.isArray(affine) || affine.length !== 4 || !affine.every((row) => Array.isArray(row) && row.length === 4 && row.every(Number.isFinite)))
    throw Error('Invalid source-surface working grid');
  const surfaces = new Map<SourceSurfaceName, SourceSurface>();
  let duraThicknessMm: number | undefined;
  for (const raw of (header.surfaces as unknown[]) ?? []) {
    const entry = record(raw, 'surface'), name = entry.name as SourceSurfaceName;
    if (!SOURCE_SURFACE_NAMES.includes(name)) continue;
    const keys = record(entry.arrays, `${name} arrays`) as Record<string, string>;
    const body = entry.body === 'scalp' || entry.body === 'bone' || entry.body === 'dura' ? entry.body : null;
    const provenance = record(entry.provenance, 'provenance');
    const surface = {name, layers: (entry.layers as string[]) ?? [], body, origin: String(provenance.origin ?? 'unknown')} as SourceSurface;
    if (name === 'dura' && typeof provenance.thicknessMm === 'number' && provenance.thicknessMm > 0 && provenance.thicknessMm < 3) duraThicknessMm = provenance.thicknessMm;
    read(keys.positions, 'float32', 3, (v) => (surface.positions = v));
    read(keys.normals, 'float32', 3, (v) => (surface.normals = v));
    read(keys.triangles, 'uint32', 3, (v) => (surface.triangles = v));
    read(keys.cells, 'int32', 1, (v) => (surface.cells = v));
    read(keys.ownerCells, 'int32', 1, (v) => (surface.ownerCells = v));
    if (body) {
      read(keys.body, 'uint8', 1, (v) => (surface.bodyCode = v));
      read(keys.bodyEdgeMm, 'float32', 1, (v) => (surface.bodyEdgeMm = v));
    }
    surfaces.set(name, surface);
  }
  const outlines: OperativeSourceAsset['outlines'] = {};
  const outlineHeader = record(header.outlines ?? {}, 'outlines');
  for (const body of ['scalp', 'bone', 'dura'] as const) {
    const entry = outlineHeader[body] as Record<string, Record<string, string>> | undefined;
    if (!entry?.outer) continue;
    const target: {outer: SourceOutline; inner?: SourceOutline} = {outer: {} as SourceOutline};
    read(entry.outer.points, 'float32', 3, (v) => (target.outer.points = v));
    read(entry.outer.hinge, 'uint8', 1, (v) => (target.outer.hinge = v));
    if (entry.inner) {
      const inner = (target.inner = {} as SourceOutline);
      read(entry.inner.points, 'float32', 3, (v) => (inner.points = v));
      read(entry.inner.hinge, 'uint8', 1, (v) => (inner.hinge = v));
    }
    outlines[body] = target;
  }
  let vessels: SourceVessels | null = null;
  if (arrays['vessels.wallPositions']) {
    const vesselHeader = record(header.vessels ?? {}, 'vessels'), coverage = (vesselHeader.coverage ?? {}) as Record<string, unknown>;
    const v = {origin: String(vesselHeader.origin ?? 'estimated'), mmByKind: (coverage.centrelineMmByKind ?? {}) as Record<string, number>} as SourceVessels;
    read('vessels.centrelinePoints', 'float32', 3, (x) => (v.centrelinePoints = x));
    read('vessels.centrelineRadii', 'float32', 1, (x) => (v.centrelineRadii = x));
    read('vessels.centrelineOffsets', 'uint32', 1, (x) => (v.centrelineOffsets = x));
    read('vessels.segmentKind', 'uint8', 1, (x) => (v.segmentKind = x));
    read('vessels.segmentInsideBox', 'uint8', 1, (x) => (v.segmentInsideBox = x));
    read('vessels.wallPositions', 'float32', 3, (x) => (v.wallPositions = x));
    read('vessels.wallNormals', 'float32', 3, (x) => (v.wallNormals = x));
    read('vessels.wallTriangles', 'uint32', 3, (x) => (v.wallTriangles = x));
    read('vessels.wallSegment', 'uint32', 1, (x) => (v.wallSegment = x));
    read('vessels.wallCells', 'int32', 1, (x) => (v.wallCells = x));
    vessels = v;
  }
  const tracts: SourceTract[] = [];
  for (const raw of (header.tracts as unknown[] | null) ?? []) {
    const entry = record(raw, 'tract'), keys = record(entry.arrays, 'tract arrays') as Record<string, string>;
    const tract = {bundle: String(entry.bundle), layer: String(entry.layer), origin: String(entry.origin ?? 'estimated')} as SourceTract;
    read(keys.points, 'float32', 3, (x) => (tract.points = x));
    read(keys.offsets, 'uint32', 1, (x) => (tract.offsets = x));
    tracts.push(tract);
  }
  // Bounded concurrency: each read is checksummed in the main process.
  let next = 0;
  await Promise.all(Array.from({length: Math.min(8, queue.length)}, async () => {
    while (next < queue.length) await queue[next++]();
  }));
  const asset: OperativeSourceAsset = {caseHash, assetName: String(reply.assetName ?? ''), scope: String(header.scope ?? ''),
    grid: {shape, affine}, surfaces, outlines, vessels, tracts, duraThicknessMm, bytes, loadMs: 0};
  validateSourceAsset(asset);
  asset.loadMs = performance.now() - started;
  return {status: 'available', caseHash, asset};
}

/** Cross-array consistency: indices in range, one cell per vertex, finite values. */
export function validateSourceAsset(asset: OperativeSourceAsset) {
  const cellCount = asset.grid.shape[0] * asset.grid.shape[1] * asset.grid.shape[2];
  const indices = (triangles: Uint32Array, count: number, what: string) => {
    for (let i = 0; i < triangles.length; i++) if (triangles[i] >= count) throw Error(`Source-surface ${what} triangle out of range`);
  };
  const cells = (values: Int32Array, what: string) => {
    for (let i = 0; i < values.length; i++) if (values[i] < -1 || values[i] >= cellCount) throw Error(`Source-surface ${what} cell out of range`);
  };
  for (const s of asset.surfaces.values()) {
    const n = s.positions.length / 3;
    if (s.normals.length !== s.positions.length || s.cells.length !== n || s.ownerCells.length !== n ||
      (s.body && (s.bodyCode?.length !== n || s.bodyEdgeMm?.length !== n))) throw Error(`Source surface ${s.name} arrays disagree`);
    indices(s.triangles, n, s.name);
    cells(s.cells, s.name);
    cells(s.ownerCells, s.name);
  }
  const v = asset.vessels;
  if (v) {
    const n = v.wallPositions.length / 3, segments = v.segmentKind.length;
    if (v.wallNormals.length !== v.wallPositions.length || v.wallSegment.length !== n || v.wallCells.length !== n ||
      v.centrelineOffsets.length !== segments + 1 || v.centrelineRadii.length * 3 !== v.centrelinePoints.length ||
      v.centrelineOffsets[segments] !== v.centrelineRadii.length) throw Error('Source vessel arrays disagree');
    indices(v.wallTriangles, n, 'vessel');
    cells(v.wallCells, 'vessel');
    for (let i = 0; i < n; i++) if (v.wallSegment[i] > segments) throw Error('Source vessel segment out of range');
  }
  for (const t of asset.tracts) if (t.offsets[t.offsets.length - 1] * 3 !== t.points.length) throw Error(`Source tract ${t.bundle} offsets disagree`);
}

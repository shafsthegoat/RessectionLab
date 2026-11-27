import type {Point3} from './coordinates';

/** Soft vessel material tags, data-derived vessel opacity ramps, a CT glass
 * field and a coarse majorant grid for the path-traced volume. Tags choose a
 * material: opacity and normals come from the acquired intensity, except that
 * a confident tag core keeps a floor so a lumen stays connected through
 * partial-volume dips. Source arrays are read, never written. All grids are C
 * order with the last axis fastest. */

export const CINEMATIC_BRICK = 4;
/** Voxels beyond a brick that a cubic B-spline sample inside it can read. */
export const CINEMATIC_SUPPORT = 2;
/** Voxels beyond a brick that a trilinear sample inside it can read. Tags are
 * only ever read trilinearly where they decide extinction, so their brick
 * maxima use this tighter window: a tagged voxel then lies in about 3.4 brick
 * windows on average instead of 8, and thin vessels occupy that much less of
 * the grid. */
export const CINEMATIC_TAG_SUPPORT = 1;
/** Dense bound (per mm) of a CT glass brick the tag gate reaches but where
 * nothing is dense: above the shader's zero test, so the brick is walked and
 * its tags thin the glass; it almost never schedules a collision, and any it
 * schedules is null. */
export const CINEMATIC_GATED_BOUND = 1e-5;
export const CINEMATIC_TAG_SIGMA = .6;
/** Tag value range over which the vessel material switches on. */
export const CINEMATIC_TAG_GATE: [number, number] = [.12, .42];
/** Smoothed tag range over which the lumen opacity floor rises, and its
 * height. A one-voxel-wide tagged line peaks near .44, so its centre is kept. */
export const CINEMATIC_TAG_FLOOR: [number, number, number] = [.3, .42, .75];
/** Ramp half width as a fraction of the core-to-shell contrast. */
export const CINEMATIC_RAMP_FRACTION = .14;
/** Calibrated CT bone ramp (HU). */
export const CINEMATIC_CT_BONE_RAMP: [number, number] = [280, 520];
/** CT gradient magnitude (HU per mm) over which an interface becomes a glass surface. */
export const CINEMATIC_GLASS_GRADIENT: [number, number] = [150, 450];
/** Voxel (or brick) visits between yields of a preparation step: a few
 * milliseconds of work at most, whatever the grid's shape. */
export const CINEMATIC_STEP_WORK = 1 << 17;
const HISTOGRAM_BINS = 4096;

export type CinematicTagClass = 0 | 1;
export type CinematicTagSource =
  | {kind: 'mask'; mask: Uint8Array}
  | {kind: 'labels'; labels: Uint16Array; value: number}
  | {kind: 'probability'; probability: Float32Array}
  /** Prepared soft tags, 0 to 255, used as they are (a layer whose thin branches the engine drew as tubes). */
  | {kind: 'tags'; tags: Uint8Array};
export interface CinematicTagLayer {cls: CinematicTagClass; source: CinematicTagSource}

/** Arteries and unclassified vessels share the first class; veins and sinuses the second. */
export function cinematicTagClass(kind: string | undefined): CinematicTagClass | null {
  if (kind === 'artery' || kind === 'vessel' || kind === 'vessels') return 0;
  if (kind === 'vein' || kind === 'sinus') return 1;
  return null;
}

export interface CinematicClassStats {
  present: boolean;
  members: number;
  coreVoxels: number;
  shellVoxels: number;
  coreMedian: number | null;
  shellMedian: number | null;
  /** Acquired-intensity ramp for tagged voxels of this class. */
  ramp: [number, number];
  /** True when core and shell did not separate; the tag then bounds opacity. */
  tagBounded: boolean;
}

export interface CinematicSignalStats {
  bricks: Point3;
  /** Per-brick maximum intensity over the brick and its filter support. */
  maxValue: Float32Array;
  range: [number, number];
}

export interface CinematicTags {
  /** Interleaved 8-bit tags, one channel per present class; null when no class is present. */
  data: Uint8Array | null;
  channels: 0 | 1 | 2;
  /** Texture channel per class, or -1 when absent. */
  channelOf: [number, number];
  classes: [CinematicClassStats, CinematicClassStats];
  /** Per-brick maximum tag per class (two values per brick) over the trilinear support (CINEMATIC_TAG_SUPPORT). */
  maxTag: Uint8Array;
}

/** CT glass field: RGB is the outward world-space gradient direction scaled by
 * surface strength (127 is zero), A is the bone ramp. Soft voxels within two
 * voxels of bone carry no surface. Brick maxima of both strengths bound faint matter. */
export interface CinematicField {data: Uint8Array; maxSurface: Float32Array; maxBone: Float32Array}

export interface CinematicMaterial {
  vesselSigma: number;
  gate: [number, number];
  ramps: [[number, number], [number, number]];
  floor: [number, number, number];
  /** Dense (opaque) bone; sigma is zero when bone is see-through or hidden. */
  bone: {sigma: number; ramp: [number, number]; dense: boolean};
  /** Faint matter. CT: glass interfaces (soft) and see-through bone (bone),
   * weighted by surface strength and facing. MR: context fog (soft) on a ramp. */
  faint: {glass: boolean; soft: number; bone: number; ramp: [number, number]};
}

export function cinematicRamp(low: number, high: number, value: number) {
  const t = Math.max(0, Math.min(1, (value - low) / Math.max(high - low, 1e-6)));
  return t * t * (3 - 2 * t);
}

function brickShape(shape: Point3): Point3 {
  return shape.map(n => Math.ceil(n / CINEMATIC_BRICK)) as Point3;
}

/** Per-brick maximum of one channel over the voxels a sample in the brick can
 * read: [4b - support, 4b + 3 + support] on each axis, which is [4b - 2, 4b + 5]
 * for the cubic support. Separable; yields every CINEMATIC_STEP_WORK reads.
 * NaN is ignored and an all-NaN window reports -Infinity. */
export function* cinematicWindowMax(values: ArrayLike<number>, shape: Point3, channels = 1, channel = 0,
  support = CINEMATIC_SUPPORT): Generator<void, Float32Array> {
  const [nx, ny, nz] = shape, [bx, by, bz] = brickShape(shape);
  const window = (b: number, n: number) => [Math.max(0, b * CINEMATIC_BRICK - support),
    Math.min(n - 1, b * CINEMATIC_BRICK + CINEMATIC_BRICK - 1 + support)];
  const zw = Array.from({length: bz}, (_, b) => window(b, nz)), yw = Array.from({length: by}, (_, b) => window(b, ny));
  const xw = Array.from({length: bx}, (_, b) => window(b, nx));
  const span = CINEMATIC_BRICK + 2 * support;
  let work = 0;
  const alongZ = new Float32Array(nx * ny * bz);
  for (let x = 0; x < nx; x++) {
    for (let y = 0; y < ny; y++) {
      const row = (x * ny + y) * nz, out = (x * ny + y) * bz;
      for (let b = 0; b < bz; b++) {
        let max = -Infinity;
        for (let z = zw[b][0]; z <= zw[b][1]; z++) { const v = values[(row + z) * channels + channel]; if (v > max) max = v; }
        alongZ[out + b] = max;
      }
      if ((work += span * bz) >= CINEMATIC_STEP_WORK) { work = 0; yield; }
    }
  }
  const alongY = new Float32Array(nx * by * bz);
  for (let x = 0; x < nx; x++) {
    for (let b = 0; b < by; b++) {
      for (let k = 0; k < bz; k++) {
        let max = -Infinity;
        for (let y = yw[b][0]; y <= yw[b][1]; y++) { const v = alongZ[(x * ny + y) * bz + k]; if (v > max) max = v; }
        alongY[(x * by + b) * bz + k] = max;
      }
      if ((work += span * bz) >= CINEMATIC_STEP_WORK) { work = 0; yield; }
    }
  }
  const out = new Float32Array(bx * by * bz);
  for (let b = 0; b < bx; b++) {
    for (let j = 0; j < by; j++) {
      for (let k = 0; k < bz; k++) {
        let max = -Infinity;
        for (let x = xw[b][0]; x <= xw[b][1]; x++) { const v = alongY[(x * by + j) * bz + k]; if (v > max) max = v; }
        out[(b * by + j) * bz + k] = max;
      }
      if ((work += span * bz) >= CINEMATIC_STEP_WORK) { work = 0; yield; }
    }
  }
  return out;
}

/** Per-brick intensity maxima over the filter support and the finite signal range. */
export function* cinematicSignalSteps(signal: Float32Array, shape: Point3): Generator<void, CinematicSignalStats> {
  let low = Infinity, high = -Infinity;
  for (let i = 0; i < signal.length; i++) {
    const value = signal[i];
    if (value < low) low = value;
    if (value > high) high = value;
    if ((i & (8 * CINEMATIC_STEP_WORK - 1)) === 8 * CINEMATIC_STEP_WORK - 1) yield;
  }
  if (low > high) { low = 0; high = 0; }
  const maxValue = yield* cinematicWindowMax(signal, shape);
  for (let i = 0; i < maxValue.length; i++) if (maxValue[i] === -Infinity) maxValue[i] = low;
  return {bricks: brickShape(shape), maxValue, range: [low, high]};
}

/** The CT glass field from central differences of the acquired intensity.
 * worldToVoxel is the row-major 3x3 linear part (voxel index per mm). */
export function* cinematicFieldSteps(signal: Float32Array, shape: Point3, worldToVoxel: ArrayLike<number>): Generator<void, CinematicField> {
  const [nx, ny, nz] = shape, M = worldToVoxel, count = nx * ny * nz, rows = nx * ny;
  const data = new Uint8Array(count * 4), surface = new Uint8Array(count);
  const [g0, g1] = CINEMATIC_GLASS_GRADIENT, [b0, b1] = CINEMATIC_CT_BONE_RAMP;
  const sx = ny * nz, sy = nz;
  // Non-finite intensity carries no data.
  const at = (i: number) => { const v = signal[i]; return v - v === 0 ? v : 0; };
  let work = 0;
  for (let x = 0; x < nx; x++) {
    const xl = x > 0 ? -sx : 0, xh = x < nx - 1 ? sx : 0, dx = (xh - xl) / sx;
    for (let y = 0; y < ny; y++) {
      const yl = y > 0 ? -sy : 0, yh = y < ny - 1 ? sy : 0, dy = (yh - yl) / sy, row = (x * ny + y) * nz;
      for (let z = 0; z < nz; z++) {
        const i = row + z, value = signal[i], q = 4 * i;
        if (!(value - value === 0)) { data[q] = data[q + 1] = data[q + 2] = 127; continue; }
        const zl = z > 0 ? -1 : 0, zh = z < nz - 1 ? 1 : 0;
        const gx = dx ? (at(i + xh) - at(i + xl)) / dx : 0, gy = dy ? (at(i + yh) - at(i + yl)) / dy : 0,
          gz = zh - zl ? (at(i + zh) - at(i + zl)) / (zh - zl) : 0;
        const wx = M[0] * gx + M[3] * gy + M[6] * gz, wy = M[1] * gx + M[4] * gy + M[7] * gz, wz = M[2] * gx + M[5] * gy + M[8] * gz;
        const magnitude = Math.sqrt(wx * wx + wy * wy + wz * wz), s = cinematicRamp(g0, g1, magnitude), scale = magnitude > 0 ? 127 * s / magnitude : 0;
        data[q] = Math.round(127 + wx * scale); data[q + 1] = Math.round(127 + wy * scale); data[q + 2] = Math.round(127 + wz * scale);
        data[q + 3] = Math.round(255 * cinematicRamp(b0, b1, value));
        surface[i] = Math.round(255 * s);
      }
      if ((work += 4 * nz) >= CINEMATIC_STEP_WORK) { work = 0; yield; }
    }
  }
  // Soft glass marks skin and other soft interfaces; within two voxels of
  // bone the interface belongs to the bone, which has its own glass.
  let near = new Uint8Array(count), spare = new Uint8Array(count);
  for (let row = 0; row < rows; row++) {
    for (let i = row * nz, end = i + nz; i < end; i++) near[i] = data[4 * i + 3] > 0 ? 1 : 0;
    if ((work += nz) >= CINEMATIC_STEP_WORK) { work = 0; yield; }
  }
  const strides = [sx, sy, 1];
  for (let axis = 0; axis < 3; axis++) {
    const stride = strides[axis], size = shape[axis], source = near, target = spare;
    for (let x = 0, i = 0; x < nx; x++) {
      for (let y = 0; y < ny; y++) {
        for (let z = 0; z < nz; z++, i++) {
          if (source[i]) { target[i] = 1; continue; }
          const c = axis === 0 ? x : axis === 1 ? y : z;
          target[i] = (c > 0 && source[i - stride]) || (c > 1 && source[i - 2 * stride]) ||
            (c < size - 1 && source[i + stride]) || (c < size - 2 && source[i + 2 * stride]) ? 1 : 0;
        }
        if ((work += nz) >= CINEMATIC_STEP_WORK) { work = 0; yield; }
      }
    }
    near = target; spare = source;
  }
  for (let row = 0; row < rows; row++) {
    for (let i = row * nz, end = i + nz; i < end; i++) if (near[i] && !data[4 * i + 3]) { data[4 * i] = data[4 * i + 1] = data[4 * i + 2] = 127; surface[i] = 0; }
    if ((work += nz) >= CINEMATIC_STEP_WORK) { work = 0; yield; }
  }
  const maxSurface = yield* cinematicWindowMax(surface, shape);
  const maxBone = yield* cinematicWindowMax(data, shape, 4, 3);
  // Byte rounding and interpolation of an encoded vector add at most a level.
  // A zero surface byte is encoded as exactly 127 in each channel, so a brick
  // without any surface has no glass at all.
  for (let i = 0; i < maxSurface.length; i++) { maxSurface[i] = maxSurface[i] > 0 ? Math.min(1, (maxSurface[i] + 2) / 255) : 0; maxBone[i] = maxBone[i] / 255; }
  return {data, maxSurface, maxBone};
}

const gaussian = (() => {
  const w1 = Math.exp(-1 / (2 * CINEMATIC_TAG_SIGMA ** 2)), w2 = Math.exp(-4 / (2 * CINEMATIC_TAG_SIGMA ** 2));
  const sum = 1 + 2 * (w1 + w2);
  return [1 / sum, w1 / sum, w2 / sum] as const;
})();

type Box = {min: Point3; max: Point3};

function memberAt(source: CinematicTagSource, i: number) {
  return source.kind === 'mask' ? source.mask[i] !== 0 : source.kind === 'tags' ? source.tags[i] >= 128 :
    source.kind === 'labels' ? source.labels[i] === source.value : source.probability[i] >= .5;
}

function histogramMedian(histogram: Uint32Array, count: number, range: [number, number]) {
  if (!count) return null;
  const half = count / 2, width = (range[1] - range[0]) / HISTOGRAM_BINS;
  let seen = 0;
  for (let bin = 0; bin < HISTOGRAM_BINS; bin++) {
    seen += histogram[bin];
    if (seen >= half) return range[0] + (bin + .5) * width;
  }
  return range[1];
}

/** Separable Gaussian (sigma CINEMATIC_TAG_SIGMA voxels) in place over one
 * channel, one line at a time. Zero is assumed outside the grid. */
function* smoothChannel(data: Uint8Array, shape: Point3, channels: number, channel: number, box: Box): Generator<void, void> {
  const [, ny, nz] = shape;
  const strides = [ny * nz * channels, nz * channels, channels];
  const line = new Float32Array(Math.max(...shape) + 4);
  const [k0, k1, k2] = gaussian;
  let work = 0;
  for (const axis of [2, 1, 0]) {
    const a = axis === 0 ? 1 : 0, b = axis === 2 ? 1 : 2;
    const lo = box.min[axis], hi = box.max[axis], length = hi - lo + 1, stride = strides[axis];
    for (let i = box.min[a]; i <= box.max[a]; i++) {
      for (let j = box.min[b]; j <= box.max[b]; j++) {
        const position = [0, 0, 0];
        position[a] = i; position[b] = j; position[axis] = lo;
        const start = position[0] * strides[0] + position[1] * strides[1] + position[2] * strides[2] + channel;
        line.fill(0);
        let any = false;
        for (let k = 0; k < length; k++) { const value = data[start + k * stride]; line[k + 2] = value; if (value) any = true; }
        if (any) for (let k = 0; k < length; k++) {
          const c = k + 2;
          data[start + k * stride] = Math.round(k0 * line[c] + k1 * (line[c - 1] + line[c + 1]) + k2 * (line[c - 2] + line[c + 2]));
        }
        if ((work += length) >= CINEMATIC_STEP_WORK) { work = 0; yield; }
      }
    }
  }
}

/** Build soft tags, per-class ramps and per-brick tag maxima. A layer with a
 * float probability or prepared tags uses those values directly; binary masks
 * and categorical labels are smoothed first, on their own, and the rest of
 * their class channel is joined to the result unsmoothed. Ramps are the midpoint of the median intensity in the
 * tag core and in a one-voxel shell outside it, so no unit is assumed. */
export function* cinematicTagSteps(signal: Float32Array, shape: Point3, layers: readonly CinematicTagLayer[],
  range: [number, number]): Generator<void, CinematicTags> {
  const [nx, ny, nz] = shape, count = nx * ny * nz;
  const present: [boolean, boolean] = [false, false];
  for (const layer of layers) present[layer.cls] = true;
  const channels = (Number(present[0]) + Number(present[1])) as 0 | 1 | 2;
  const channelOf: [number, number] = [present[0] ? 0 : -1, present[1] ? (present[0] ? 1 : 0) : -1];
  const bricks = brickShape(shape), brickCount = bricks[0] * bricks[1] * bricks[2];
  const empty = (): CinematicClassStats => ({present: false, members: 0, coreVoxels: 0, shellVoxels: 0,
    coreMedian: null, shellMedian: null, ramp: [range[1] + 1, range[1] + 2], tagBounded: false});
  const classes: [CinematicClassStats, CinematicClassStats] = [empty(), empty()];
  const maxTag = new Uint8Array(brickCount * 2);
  if (!channels) return {data: null, channels, channelOf, classes, maxTag};
  for (const layer of layers) {
    const source = layer.source;
    const length = source.kind === 'mask' ? source.mask.length : source.kind === 'tags' ? source.tags.length :
      source.kind === 'labels' ? source.labels.length : source.probability.length;
    if (length !== count) throw Error('A vessel layer does not match the acquired grid');
  }
  const data = new Uint8Array(count * channels);
  const boxes: Array<Box | null> = [null, null];
  let work = 0;
  // Fill: probability layers write their value; others write a binary 255.
  for (const cls of [0, 1] as const) {
    const channel = channelOf[cls];
    if (channel < 0) continue;
    const members = layers.filter(layer => layer.cls === cls);
    const box: Box = {min: [nx, ny, nz], max: [-1, -1, -1]};
    // Binary sources first, smoothed on their own; continuous sources (a float
    // probability, prepared tags) are joined afterwards and stay as supplied.
    for (const continuous of [false, true]) {
      for (const {source} of members) {
        if ((source.kind === 'probability' || source.kind === 'tags') !== continuous) continue;
        for (let x = 0; x < nx; x++) {
          for (let y = 0; y < ny; y++) {
            const row = (x * ny + y) * nz;
            for (let z = 0; z < nz; z++) {
              const i = row + z;
              let value = 0;
              if (source.kind === 'mask') value = source.mask[i] ? 255 : 0;
              else if (source.kind === 'tags') value = source.tags[i];
              else if (source.kind === 'labels') value = source.labels[i] === source.value ? 255 : 0;
              else {
                const p = source.probability[i];
                value = p > 0 ? Math.round(Math.min(1, p) * 255) : 0;
              }
              if (!value) continue;
              const q = i * channels + channel;
              if (value > data[q]) data[q] = value;
              if (x < box.min[0]) box.min[0] = x; if (x > box.max[0]) box.max[0] = x;
              if (y < box.min[1]) box.min[1] = y; if (y > box.max[1]) box.max[1] = y;
              if (z < box.min[2]) box.min[2] = z; if (z > box.max[2]) box.max[2] = z;
            }
            if ((work += nz) >= CINEMATIC_STEP_WORK) { work = 0; yield; }
          }
        }
      }
      if (box.max[0] < 0) continue;
      // Smoothing spreads two voxels.
      const padded: Box = {min: box.min.map(v => Math.max(0, v - 2)) as Point3,
        max: box.max.map((v, axis) => Math.min(shape[axis] - 1, v + 2)) as Point3};
      boxes[cls] = padded;
      if (!continuous) yield* smoothChannel(data, shape, channels, channel, padded);
    }
  }
  // Classify core and shell voxels.
  const scale = HISTOGRAM_BINS / Math.max(range[1] - range[0], 1e-6);
  const bin = (value: number) => Math.max(0, Math.min(HISTOGRAM_BINS - 1, Math.floor((value - range[0]) * scale)));
  for (const cls of [0, 1] as const) {
    const channel = channelOf[cls], box = boxes[cls];
    if (channel < 0 || !box) continue;
    const own = layers.filter(layer => layer.cls === cls).map(layer => layer.source);
    const other = layers.filter(layer => layer.cls !== cls).map(layer => layer.source);
    const otherChannel = channelOf[1 - cls];
    const core = [new Uint32Array(HISTOGRAM_BINS), new Uint32Array(HISTOGRAM_BINS), new Uint32Array(HISTOGRAM_BINS)];
    const coreCount = [0, 0, 0];
    const shell = new Uint32Array(HISTOGRAM_BINS);
    let shellCount = 0, members = 0;
    for (let x = box.min[0]; x <= box.max[0]; x++) {
      for (let y = box.min[1]; y <= box.max[1]; y++) {
        const row = (x * ny + y) * nz;
        for (let z = box.min[2]; z <= box.max[2]; z++) {
          const i = row + z, tag = data[i * channels + channel];
          if (!tag) continue;
          const value = signal[i];
          if (!(value === value)) continue;
          let member = false;
          for (const source of own) if (memberAt(source, i)) { member = true; break; }
          if (member) {
            members++;
            const b = bin(value);
            core[2][b]++; coreCount[2]++;
            if (tag >= 153) { core[1][b]++; coreCount[1]++; }
            if (tag >= 230) { core[0][b]++; coreCount[0]++; }
          } else if (tag > 5 && tag < 90) {
            let foreign = otherChannel >= 0 && data[i * channels + otherChannel] >= 128;
            if (!foreign) for (const source of other) if (memberAt(source, i)) { foreign = true; break; }
            if (!foreign) { shell[bin(value)]++; shellCount++; }
          }
        }
        if ((work += box.max[2] - box.min[2] + 1) >= CINEMATIC_STEP_WORK) { work = 0; yield; }
      }
    }
    // Thin vessels have no deep core; fall back to the half-tag and then the whole mask.
    const level = coreCount[0] >= 64 ? 0 : coreCount[1] >= 16 ? 1 : 2;
    const coreMedian = histogramMedian(core[level], coreCount[level], range);
    const shellMedian = histogramMedian(shell, shellCount, range);
    const stats = classes[cls];
    Object.assign(stats, {present: members > 0, members, coreVoxels: coreCount[level], shellVoxels: shellCount, coreMedian, shellMedian});
    if (coreMedian !== null && shellMedian !== null && coreMedian - shellMedian > (range[1] - range[0]) * 1e-3) {
      const mid = (coreMedian + shellMedian) / 2, half = CINEMATIC_RAMP_FRACTION * (coreMedian - shellMedian);
      stats.ramp = [mid - half, mid + half];
    } else if (members > 0) {
      stats.ramp = [range[0] - 2, range[0] - 1];
      stats.tagBounded = true;
    }
    const brickMax = yield* cinematicWindowMax(data, shape, channels, channel, CINEMATIC_TAG_SUPPORT);
    for (let brick = 0; brick < brickCount; brick++) maxTag[2 * brick + cls] = Math.max(0, brickMax[brick]);
  }
  return {data, channels, channelOf, classes, maxTag};
}

export const CINEMATIC_SKIP_LIMIT = 8;

/** Chebyshev growth of the occupied set gives each empty brick the largest
 * empty cube centred on it, up to CINEMATIC_SKIP_LIMIT bricks, written to
 * channel 1. Channel 3 receives the largest channel-2 value over that cube
 * (the brick's own value when it is occupied). Each radius is three one-brick
 * passes, one per axis, between two pairs of buffers. */
function* emptyCubes(occupied: Uint8Array, bricks: Point3, data: Float32Array): Generator<void, void> {
  const [bx, by, bz] = bricks, count = occupied.length;
  const strides = [by * bz, bz, 1];
  let grown: Uint8Array = occupied, grownNext: Uint8Array = new Uint8Array(count);
  let spread = new Float32Array(count), spreadNext = new Float32Array(count);
  for (let brick = 0; brick < count; brick++) spread[brick] = data[4 * brick + 3] = data[4 * brick + 2];
  let work = 0;
  for (let radius = 1; radius <= CINEMATIC_SKIP_LIMIT; radius++) {
    for (let axis = 0; axis < 3; axis++) {
      const stride = strides[axis], size = bricks[axis];
      const source = grown, values = spread, next = grownNext, spreadOut = spreadNext;
      for (let x = 0, brick = 0; x < bx; x++) {
        for (let y = 0; y < by; y++) {
          for (let z = 0; z < bz; z++, brick++) {
            const coordinate = axis === 0 ? x : axis === 1 ? y : z;
            let g = source[brick], v = values[brick];
            if (coordinate > 0) { if (source[brick - stride]) g = 1; const w = values[brick - stride]; if (w > v) v = w; }
            if (coordinate < size - 1) { if (source[brick + stride]) g = 1; const w = values[brick + stride]; if (w > v) v = w; }
            next[brick] = g; spreadOut[brick] = v;
          }
          if ((work += bz) >= CINEMATIC_STEP_WORK) { work = 0; yield; }
        }
      }
      grown = next; grownNext = source; spread = spreadOut; spreadNext = values;
    }
    let any = false;
    for (let brick = 0; brick < count; brick++) if (!grown[brick]) { data[4 * brick + 1] = radius; data[4 * brick + 3] = spread[brick]; any = true; }
    if (!any) break;
  }
}

/** Per brick: the dense extinction bound (per mm), the radius in bricks of
 * the cube around an empty brick without dense matter, the faint extinction
 * bound, and the faint bound over that cube. Intensity maxima cover the cubic
 * B-spline support and tag maxima the trilinear one; the material is monotone
 * in intensity and in each tag except bone, which a tag suppresses, so the
 * bound takes the worse of the gate's extremes. Under CT glass a brick inside
 * the tag gate with no extinction keeps a negligible bound, so the walk reads
 * its tags where they thin the glass; MR context never reads tags for faint
 * matter, so such a brick stays empty there. */
export function* cinematicMajorantSteps(signal: CinematicSignalStats, tags: CinematicTags | null, material: CinematicMaterial,
  field: CinematicField | null = null): Generator<void, Float32Array> {
  const count = signal.maxValue.length;
  const data = new Float32Array(count * 4);
  const gate = (tag: number) => cinematicRamp(material.gate[0], material.gate[1], tag / 255);
  const [f0, f1, height] = material.floor;
  const dense = new Uint8Array(count);
  for (let brick = 0; brick < count; brick++) {
    const value = signal.maxValue[brick];
    let vessel = 0, lumen = 0, open = 0;
    if (tags) for (const cls of [0, 1] as const) {
      const tag = tags.maxTag[2 * brick + cls], g = gate(tag);
      if (g <= 0) continue;
      const q = Math.max(cinematicRamp(material.ramps[cls][0], material.ramps[cls][1], value), height * cinematicRamp(f0, f1, tag / 255));
      vessel = Math.max(vessel, g * q); lumen = Math.max(lumen, q); open = Math.max(open, g);
    }
    const bone = material.bone.dense ? material.bone.sigma * cinematicRamp(material.bone.ramp[0], material.bone.ramp[1], value) : 0;
    const v = material.vesselSigma;
    const bound = Math.min(v * vessel + bone, Math.max(bone, open * v * lumen + (1 - open) * bone));
    // Bounds carry 1% headroom against float rounding in filtering.
    const gated = open > 0 && material.faint.glass && field !== null;
    data[4 * brick] = bound > 0 ? 1.01 * bound : gated ? CINEMATIC_GATED_BOUND : 0;
    let faint = 0;
    if (material.faint.glass) {
      if (field) faint = material.faint.soft * field.maxSurface[brick] + material.faint.bone * field.maxBone[brick];
    } else faint = material.faint.soft * cinematicRamp(material.faint.ramp[0], material.faint.ramp[1], value);
    data[4 * brick + 2] = 1.01 * faint;
    dense[brick] = bound > 0 || gated ? 1 : 0;
    if ((brick & (CINEMATIC_STEP_WORK / 8 - 1)) === CINEMATIC_STEP_WORK / 8 - 1) yield;
  }
  yield* emptyCubes(dense, signal.bricks, data);
  return data;
}

export interface CinematicMaterialInput {
  ct: boolean;
  /** Bone extinction scale: one for opaque bone, a layer opacity for see-through bone, zero for none. */
  boneWeight: number;
  /** Display window of the acquisition, used for MR context only. */
  window: [number, number];
}

/** See-through bone at the rim of its glass, per unit bone weight (per mm). */
export const CINEMATIC_GLASS_BONE = 10;
/** CT soft-tissue glass at a strong interface seen edge on (per mm). */
export const CINEMATIC_GLASS_SOFT = .035;

/** Optical constants of the path-traced volume (extinction per mm). The CT
 * bone range is calibrated Hounsfield; vessel ramps always come from tags. */
export function cinematicMaterial(tags: CinematicTags | null, input: CinematicMaterialInput): CinematicMaterial & {gradientScale: number} {
  const ramps = [0, 1].map(cls => tags?.classes[cls as CinematicTagClass].ramp ?? [1e30, 2e30]) as [[number, number], [number, number]];
  const contrasts = tags ? tags.classes.flatMap(c => c.coreMedian !== null && c.shellMedian !== null && !c.tagBounded ? [c.coreMedian - c.shellMedian] : []) : [];
  const weight = Math.max(0, Math.min(1, input.boneWeight)), dense = input.ct && weight >= .5;
  if (input.ct) {
    return {vesselSigma: 3, gate: CINEMATIC_TAG_GATE, ramps, floor: CINEMATIC_TAG_FLOOR,
      bone: {sigma: dense ? 3 * weight : 0, ramp: CINEMATIC_CT_BONE_RAMP, dense},
      faint: {glass: true, soft: CINEMATIC_GLASS_SOFT, bone: dense ? 0 : CINEMATIC_GLASS_BONE * weight, ramp: [1e30, 2e30]},
      gradientScale: 1 / Math.max(120, ...contrasts)};
  }
  // MR context: a faint ramp under the vessel shell signal, or the lower window.
  const shells = tags ? tags.classes.flatMap(c => c.shellMedian !== null ? [c.shellMedian] : []) : [];
  const [low, high] = input.window, span = Math.max(high - low, 1e-6);
  const shell = shells.length ? Math.min(...shells) : low + .35 * span;
  return {vesselSigma: 3, gate: CINEMATIC_TAG_GATE, ramps, floor: CINEMATIC_TAG_FLOOR,
    bone: {sigma: 0, ramp: [1e30, 2e30], dense: false},
    faint: {glass: false, soft: .0008, bone: 0, ramp: [low + .5 * (shell - low), shell]},
    gradientScale: 1 / Math.max(contrasts.length ? Math.max(...contrasts) : .25 * span, 1e-6)};
}

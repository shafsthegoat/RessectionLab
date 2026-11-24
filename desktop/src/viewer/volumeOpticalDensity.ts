import {DataUtils} from 'three';
import type {Affine, Point3} from './coordinates';
import {mriBoundaryExtinction,VOLUME_TRANSFER_WIDTH} from './volumeTransfer.ts';

type OpticalTransfer = {data: Float32Array; range: [number, number]; gradientOpacity: boolean};
export type OpticalSource = {
  signal: Float32Array;
  shape: Point3;
  affine: Affine;
  inverse: Affine;
  support?: Uint8Array;
};

const CACHE_CELL_MM = 4;
const MAX_CELLS_PER_AXIS = 56;

// Uniformly stratified source samples in a cache cell. Sampling density is
// bounded even for a large import. The visible primary ray still reads exact
// full-resolution source voxels, so secondary-ray aliasing cannot erase tissue.
function cellSamples(length: number, cells: number) {
  return Array.from({length: cells}, (_, cell) => {
    const from = Math.floor(cell * length / cells);
    const to = Math.floor((cell + 1) * length / cells);
    const count = Math.min(to - from, 3);
    return Array.from({length: count}, (_, i) => from + Math.floor((i + .5) * (to - from) / count));
  });
}

function* opticalDensitySteps(source: OpticalSource, transfer: OpticalTransfer) {
  const {signal, shape, affine, inverse, support} = source;
  const grid = shape.map((size, axis) => Math.min(size, MAX_CELLS_PER_AXIS,
    Math.max(1, Math.ceil(size * Math.hypot(affine[0][axis], affine[1][axis], affine[2][axis]) / CACHE_CELL_MM)))) as Point3;
  const [nx, ny, nz] = grid;
  const [sx, sy, sz] = shape;
  const xStride = sy * sz;
  const fine = new Float32Array(nx * ny * nz * 2);
  const [xs, ys, zs] = grid.map((cells, axis) => cellSamples(shape[axis], cells));
  const inverseRange = 1 / (transfer.range[1] - transfer.range[0]);
  const toLut = VOLUME_TRANSFER_WIDTH * inverseRange;
  const [a, b, c] = inverse;

  for (let i = 0; i < nx; i++) for (let j = 0; j < ny; j++) {
    for (let k = 0; k < nz; k++) {
      let extinction = 0;
      let shellExtinction = 0;
      for (const x of xs[i]) for (const y of ys[j]) for (const z of zs[k]) {
        const q = x * xStride + y * sz + z;
        if (support && !support[q]) continue;
        if (!Number.isFinite(signal[q])) continue;
        // Transfer before averaging. Air beside dense bone cannot be evaluated
        // as a single synthetic soft-tissue HU at the block-average intensity.
        const lut = Math.max(0, Math.min(VOLUME_TRANSFER_WIDTH - 1, (signal[q] - transfer.range[0]) * toLut - .5));
        const lo = Math.floor(lut);
        const hi = Math.min(lo + 1, VOLUME_TRANSFER_WIDTH - 1);
        const from = transfer.data[4 * lo + 3];
        const density = from + (transfer.data[4 * hi + 3] - from) * (lut - lo);
        extinction += density;
        if (!transfer.gradientOpacity || !density) continue;
        const gx = signal[q + (x < sx - 1 ? xStride : 0)] - signal[q - (x > 0 ? xStride : 0)];
        const gy = signal[q + (y < sy - 1 ? sz : 0)] - signal[q - (y > 0 ? sz : 0)];
        const gz = signal[q + (z < sz - 1 ? 1 : 0)] - signal[q - (z > 0 ? 1 : 0)];
        const dx = a[0] * gx + b[0] * gy + c[0] * gz;
        const dy = a[1] * gx + b[1] * gy + c[1] * gz;
        const dz = a[2] * gx + b[2] * gy + c[2] * gz;
        const gradient = .5 * inverseRange * Math.sqrt(dx * dx + dy * dy + dz * dz);
        shellExtinction += density * mriBoundaryExtinction(Number.isFinite(gradient) ? gradient : 0);
      }
      const sampleCount = xs[i].length * ys[j].length * zs[k].length;
      const index = 2 * ((i * ny + j) * nz + k);
      fine[index] = extinction / sampleCount;
      fine[index + 1] = transfer.gradientOpacity ? shellExtinction / sampleCount : fine[index];
    }
    yield;
  }

  // A separable [1,2,1] blur supplies the wider footprint of a distant cone.
  // Zero outside the sampled field avoids causing shadows from repeated edges.
  const coarse = new Float32Array(fine.length);
  const scratch = new Float32Array(fine.length);
  let previous = fine;
  for (const [axis, stride] of [ny * nz, nz, 1].entries()) {
    const next = axis === 1 ? scratch : coarse;
    for (let i = 0; i < nx; i++) {
      for (let j = 0; j < ny; j++) for (let k = 0; k < nz; k++) {
        const coordinate = axis === 0 ? i : axis === 1 ? j : k;
        const index = 2 * ((i * ny + j) * nz + k);
        const lo = index - 2 * stride;
        const hi = index + 2 * stride;
        for (let channel = 0; channel < 2; channel++) {
          next[index + channel] = .5 * previous[index + channel]
            + (coordinate > 0 ? .25 * previous[lo + channel] : 0)
            + (coordinate < grid[axis] - 1 ? .25 * previous[hi + channel] : 0);
        }
      }
      yield;
    }
    previous = next;
  }
  return {grid, fine, coarse};
}

export function finishOpticalPreparation<T>(steps: Generator<void, T>) {
  let result=steps.next();
  while(!result.done)result=steps.next();
  return result.value;
}

export function buildOpticalDensity(source: OpticalSource, transfer: OpticalTransfer) {
  return finishOpticalPreparation(opticalDensitySteps(source,transfer));
}

// The ray marcher already uses the 16 fragment samplers guaranteed by WebGL2.
// A single RGBA16F texture gives the transfer row and a trilinear density atlas
// two scales and both MRI extinction modes without taking another sampler.
export function* opticalAtlasSteps(transfer: OpticalTransfer, source?: OpticalSource) {
  const density = source ? yield* opticalDensitySteps(source, transfer) : undefined;
  const grid: Point3 = density?.grid ?? [1, 1, 1];
  const [nx, ny, nz] = grid;
  const columns = Math.floor(VOLUME_TRANSFER_WIDTH / nz);
  const height = density ? 1 + Math.ceil(nx / columns) * ny : 1;
  const data = new Uint16Array(VOLUME_TRANSFER_WIDTH * height * 4);
  for (let q = 0; q < transfer.data.length; q++) data[q] = DataUtils.toHalfFloat(transfer.data[q]);
  if (density) {
    const {fine, coarse} = density;
    for (let x = 0; x < nx; x++) {
      for (let y = 0; y < ny; y++) for (let z = 0; z < nz; z++) {
        const from = 2 * ((x * ny + y) * nz + z);
        const row = 1 + Math.floor(x / columns) * ny + y;
        const col = x % columns * nz + z;
        const to = 4 * (row * VOLUME_TRANSFER_WIDTH + col);
        data[to] = DataUtils.toHalfFloat(fine[from]);
        data[to + 1] = DataUtils.toHalfFloat(coarse[from]);
        data[to + 2] = DataUtils.toHalfFloat(fine[from + 1]);
        data[to + 3] = DataUtils.toHalfFloat(coarse[from + 1]);
      }
      yield;
    }
  }
  return {data, width: VOLUME_TRANSFER_WIDTH, height, grid, columns};
}

export function volumeOpticalAtlas(transfer: OpticalTransfer, source?: OpticalSource) {
  return finishOpticalPreparation(opticalAtlasSteps(transfer,source));
}

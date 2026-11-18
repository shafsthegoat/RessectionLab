import * as THREE from 'three';
import type { OperativeField } from '../operative-field';

type XYZ = [number, number, number];
export type VoxelBounds = { low: XYZ; high: XYZ };
export const emptyVoxelBounds = (): VoxelBounds => ({low: [Infinity, Infinity, Infinity], high: [-1, -1, -1]});
export function extendVoxelBounds(bounds: VoxelBounds, x: number, y: number, z: number) {
  bounds.low[0] = Math.min(bounds.low[0], x); bounds.high[0] = Math.max(bounds.high[0], x);
  bounds.low[1] = Math.min(bounds.low[1], y); bounds.high[1] = Math.max(bounds.high[1], y);
  bounds.low[2] = Math.min(bounds.low[2], z); bounds.high[2] = Math.max(bounds.high[2], z);
}

// Binary surface nets. Each mixed cell contributes one vertex at the mean
// of its edge crossings. The 256 possible offsets are grid independent.
const crossings = Array.from({length: 256}, (_, pattern) => {
  const sum: XYZ = [0, 0, 0]; let count = 0;
  for (let v = 0; v < 8; v++) for (let axis = 0; axis < 3; axis++) {
    const end = v ^ (1 << axis);
    if (v >= end || ((pattern >> v) & 1) === ((pattern >> end) & 1)) continue;
    for (let k = 0; k < 3; k++) sum[k] += (((v >> k) & 1) + ((end >> k) & 1)) / 2;
    count++;
  }
  return sum.map(n => n / (count || 1));
});

/** Extract a closed source-label surface. Clipping happens in the sampled
 * volume so a cutaway exposes a filled section through each remaining label. */
export function operativeSurface(field: Pick<OperativeField, 'shape' | 'spacingMm' | 'origin'>, mask: Uint8Array, bounds: VoxelBounds, firstY = 0) {
  const geometry = new THREE.BufferGeometry();
  const [nx, ny, nz] = field.shape, yz = ny * nz;
  const [lowX, lowY, lowZ] = bounds.low, [highX, highY, highZ] = bounds.high;
  const fromY = Math.max(firstY, lowY);
  if (highX < lowX || highY < fromY) return geometry;
  const width = highZ - lowZ + 2, slab = width * (highY - fromY + 2);
  const cells = new Uint32Array(2 * slab), positions: number[] = [], indices: number[] = [];
  const value = (x: number, y: number, z: number) => x >= 0 && y >= firstY && z >= 0 && x < nx && y < ny && z < nz ? mask[x * yz + y * nz + z] : 0;
  const strides = [slab, width, 1];
  for (let x = lowX - 1; x <= highX; x++) {
    const base = ((x - lowX + 1) % 2) * slab;
    strides[0] = base ? slab : -slab;
    for (let y = fromY - 1; y <= highY; y++) {
      let lower = value(x,y,lowZ-1) | value(x+1,y,lowZ-1)<<1 | value(x,y+1,lowZ-1)<<2 | value(x+1,y+1,lowZ-1)<<3;
      let cell = base + (y - fromY + 1) * width;
      for (let z = lowZ - 1; z <= highZ; z++, cell++) {
        const upper = value(x,y,z+1) | value(x+1,y,z+1)<<1 | value(x,y+1,z+1)<<2 | value(x+1,y+1,z+1)<<3;
        const pattern = lower | upper << 4; lower = upper;
        if (!pattern || pattern === 255) continue;
        const offset = crossings[pattern], vertex = positions.length / 3;
        positions.push(field.origin[0]+(x+offset[0])*field.spacingMm, field.origin[1]+(y+offset[1])*field.spacingMm, field.origin[2]+(z+offset[2])*field.spacingMm);
        cells[cell] = vertex;
        for (let axis = 0; axis < 3; axis++) {
          if ((pattern & 1) === ((pattern >> (1 << axis)) & 1)) continue;
          const u = strides[(axis+1)%3], v = strides[(axis+2)%3];
          const a = cells[cell-u], b = cells[cell-u-v], c = cells[cell-v];
          if (pattern & 1) indices.push(vertex,a,b,vertex,b,c);
          else indices.push(vertex,c,b,vertex,b,a);
        }
      }
    }
  }
  geometry.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
  geometry.setIndex(indices); geometry.computeVertexNormals();
  return geometry;
}

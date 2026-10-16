import { inverseAffine, rasAffine } from "./coordinates.ts";
import type { Affine } from "./coordinates";

/** A rendering approximation, never a replacement annotation or measurement.
 * Six Taubin pairs reduce stair-step shading with a negative pass to limit
 * shrinkage. Only existing mesh neighbours interact; no components/holes are
 * removed or joined, and the original triangle buffer is retained separately.
 */
export function smoothSurface(
  positions: Float32Array,
  affine: Affine,
): Float32Array {
  if (positions.length % 9 !== 0 || !positions.every(Number.isFinite))
    throw new Error("Display smoothing requires finite complete triangles.");
  const linear = rasAffine(affine, "RAS+").slice(0, 3).map(row => row.slice(0, 3));
  const inverse = inverseAffine(affine);
  // Half the smallest voxel altitude, including anisotropic/sheared grids.
  const limit = 0.5 / Math.max(...inverse.slice(0, 3).map(row => Math.hypot(...row.slice(0, 3))));
  if (!positions.length) return positions.slice();

  const low = [Infinity, Infinity, Infinity], high = [-Infinity, -Infinity, -Infinity];
  let halfGrid = true;
  for (let i = 0; i < positions.length; i++) {
    const value = positions[i] * 2, axis = i % 3;
    halfGrid &&= Number.isSafeInteger(value);
    low[axis] = Math.min(low[axis], value);
    high[axis] = Math.max(high[axis], value);
  }
  const span = high.map((value, axis) => value - low[axis] + 1);
  const packed = halfGrid && Number.isSafeInteger(span[0] * span[1] * span[2]);
  const key = (i: number) => packed
    ? (positions[i] * 2 - low[0]) * span[1] * span[2] + (positions[i + 1] * 2 - low[1]) * span[2] + positions[i + 2] * 2 - low[2]
    : `${positions[i]},${positions[i + 1]},${positions[i + 2]}`;
  const vertices: number[] = [], ids = new Map<number | string, number>();
  const triangles = new Uint32Array(positions.length / 3);
  for (let i = 0; i < positions.length; i += 3) {
    const id = key(i);
    let index = ids.get(id);
    if (index === undefined) {
      index = vertices.length / 3;
      ids.set(id, index);
      vertices.push(positions[i], positions[i + 1], positions[i + 2]);
    }
    triangles[i / 3] = index;
  }
  const count = vertices.length / 3;
  const original = new Float64Array(vertices.length);
  for (let i = 0; i < vertices.length; i += 3)
    for (let axis = 0; axis < 3; axis++)
      original[i + axis] = linear[axis][0] * vertices[i] + linear[axis][1] * vertices[i + 1] + linear[axis][2] * vertices[i + 2];

  // Compressed adjacency. Each closed manifold edge occurs twice, so these
  // repeated entries give the same uniform weights without per-vertex Sets.
  const offsets = new Uint32Array(count + 1);
  for (const id of triangles) offsets[id + 1] += 2;
  for (let i = 1; i <= count; i++) offsets[i] += offsets[i - 1];
  const neighbours = new Uint32Array(offsets[count]), cursor = offsets.slice();
  const edges = new Map<number, number>(), pinned = new Uint8Array(count);
  for (let i = 0; i < triangles.length; i += 3) {
    for (let corner = 0; corner < 3; corner++) {
      const a = triangles[i + corner], b = triangles[i + (corner + 1) % 3], c = triangles[i + (corner + 2) % 3];
      neighbours[cursor[a]++] = b;
      neighbours[cursor[a]++] = c;
      const edge = Math.min(a, b) * count + Math.max(a, b);
      edges.set(edge, (edges.get(edge) ?? 0) + 1);
    }
  }
  // Preserve open boundaries/non-manifold edges exactly instead of rounding
  // them off. Source binary surfaces are normally closed, including padding.
  for (const [edge, uses] of edges) if (uses !== 2) {
    pinned[Math.floor(edge / count)] = 1;
    pinned[edge % count] = 1;
  }
  let current = original.slice(), next = new Float64Array(original.length);
  for (let pair = 0; pair < 6; pair++) for (const weight of [0.5, -0.53]) {
    for (let vertex = 0; vertex < count; vertex++) {
      const i = vertex * 3, degree = offsets[vertex + 1] - offsets[vertex];
      for (let axis = 0; axis < 3; axis++) {
        let sum = 0;
        for (let edge = offsets[vertex]; edge < offsets[vertex + 1]; edge++)
          sum += current[neighbours[edge] * 3 + axis];
        next[i + axis] = pinned[vertex] || !degree ? original[i + axis]
          : current[i + axis] + weight * (sum / degree - current[i + axis]);
      }
      const distance = Math.hypot(next[i] - original[i], next[i + 1] - original[i + 1], next[i + 2] - original[i + 2]);
      if (distance > limit) for (let axis = 0; axis < 3; axis++)
        next[i + axis] = original[i + axis] + (next[i + axis] - original[i + axis]) * limit / distance;
    }
    [current, next] = [next, current];
  }
  const output = new Float32Array(positions.length);
  const normal = (buffer: Float32Array, i: number) => {
    const ab = [0, 1, 2].map(a => buffer[i + 3 + a] - buffer[i + a]);
    const ac = [0, 1, 2].map(a => buffer[i + 6 + a] - buffer[i + a]);
    return [ab[1] * ac[2] - ab[2] * ac[1], ab[2] * ac[0] - ab[0] * ac[2], ab[0] * ac[1] - ab[1] * ac[0]];
  };
  // Check the actual float32 output, including thin branches. Back off the
  // displacement globally if any face flips or loses >90% of its area.
  for (let factor = 1; factor >= 1 / 128; factor /= 2) {
    const adjusted = new Float32Array(vertices.length);
    for (let i = 0; i < vertices.length; i += 3)
      for (let axis = 0; axis < 3; axis++)
        adjusted[i + axis] = vertices[i + axis] + factor * (
          inverse[axis][0] * (current[i] - original[i]) +
          inverse[axis][1] * (current[i + 1] - original[i + 1]) +
          inverse[axis][2] * (current[i + 2] - original[i + 2]));
    for (let i = 0; i < triangles.length; i++)
      output.set(adjusted.subarray(triangles[i] * 3, triangles[i] * 3 + 3), i * 3);
    let valid = true;
    for (let i = 0; i < output.length; i += 9) {
      const a = normal(positions, i), b = normal(output, i);
      if (a.reduce((sum, value, axis) => sum + value * b[axis], 0) <= 0 || Math.hypot(...b) < 0.1 * Math.hypot(...a)) {
        valid = false;
        break;
      }
    }
    if (valid) return output;
  }
  return positions.slice();
}

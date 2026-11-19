import type { OperativeField } from '../operative-field';
import { aspirationSeedIndex, eligibleTumor } from './operativeSeed.ts';

type XYZ = [number, number, number];
export type AspirationPattern = 'overlapping' | 'spiral';
const TAU = 2 * Math.PI;

export function pointAt(field: OperativeField, index: number): XYZ {
  const [, ny, nz] = field.shape,
    s = field.spacingMm;
  return [
    field.origin[0] + Math.floor(index / (ny * nz)) * s,
    field.origin[1] + (Math.floor(index / nz) % ny) * s,
    field.origin[2] + (index % nz) * s,
  ];
}
// Traverse alternating chords of a shallow disc before descending to the
// next depth. Adjacent spherical contact footprints overlap in both axes.
function overlappingPasses(
  center: XYZ,
  reach: number,
  depth: number,
  tipRadius: number,
  spacing: number,
  advance: (x: number, y: number, z: number) => void,
) {
  const pitch = Math.max(spacing, tipRadius);
  const levels = Math.ceil(depth / pitch);
  let previous = center;
  const visit = (goal: XYZ) => {
    const delta = goal.map((v, axis) => v - previous[axis]);
    const samples = Math.max(1, Math.ceil(Math.hypot(...delta) / (spacing * 0.45)));
    for (let i = 1; i <= samples; i++) {
      const t = i / samples;
      advance(previous[0] + t * delta[0], previous[1] + t * delta[1], previous[2] + t * delta[2]);
    }
    previous = goal;
  };
  for (let level = 0; level <= levels; level++) {
    const t = level / levels;
    const radius = 0.8 * reach * Math.sqrt(1 - 0.85 * t * t);
    const z = center[2] - depth * t;
    const rows = Math.max(2, Math.ceil((2 * radius) / pitch));
    let sign = 1;
    for (let row = 0; row <= rows; row++) {
      const y = ((2 * row) / rows - 1) * radius * (level % 2 ? -1 : 1);
      const x = Math.sqrt(Math.max(0, radius * radius - y * y));
      visit([center[0] - sign * x, center[1] + y, z]);
      visit([center[0] + sign * x, center[1] + y, z]);
      sign *= -1;
    }
  }
}

/** Plan a continuous illustrative sweep through a local connected tumor label.
 * Adjacent centres share a voxel face. Protected cells neither admit the tip
 * nor receive a removal time. A brush affects only eligible source labels. */
export function aspiration(field: OperativeField, tipRadius: number, pattern: AspirationPattern) {
  const {
      shape: [nx, ny, nz],
      voxels,
      spacingMm: s,
    } = field,
    yz = ny * nz;
  const firstHit = new Map<number, number>(),
    path: number[] = [];
  const allowed = (i: number) => eligibleTumor(voxels[i]);
  const seed = aspirationSeedIndex(field);
  if (seed === null) return { path, firstHit, seed: null };
  const center = pointAt(field, seed),
    reach = Math.max(tipRadius * 2.2, Math.min(field.radiusMm * 0.44, field.routeRadiusMm * 1.6));
  const depth = Math.max(3 * s, reach * 1.8);
  const neighbors = [
    [1, 0, 0],
    [-1, 0, 0],
    [0, 1, 0],
    [0, -1, 0],
    [0, 0, -1],
    [0, 0, 1],
  ];
  const radiusSquared = (tipRadius / s) ** 2;
  const visitedCenters = new Set<number>();
  const record = (i: number) => {
    path.push(i);
    if (visitedCenters.has(i)) return;
    visitedCenters.add(i);
    const z0 = i % nz,
      y0 = Math.floor(i / nz) % ny,
      x0 = Math.floor(i / yz);
    // Flood within this tip footprint. A nearby island cannot be reached
    // through empty tissue or a vessel, even if it lies inside the tip radius.
    const queue = [i],
      seen = new Set(queue);
    for (let cursor = 0; cursor < queue.length; cursor++) {
      const j = queue[cursor],
        x = Math.floor(j / yz),
        y = Math.floor(j / nz) % ny,
        z = j % nz;
      if (!firstHit.has(j)) firstHit.set(j, path.length);
      for (const [dx, dy, dz] of neighbors) {
        const xx = x + dx,
          yy = y + dy,
          zz = z + dz;
        if (xx < 0 || xx >= nx || yy < 0 || yy >= ny || zz < 0 || zz >= nz) continue;
        if ((xx - x0) ** 2 + (yy - y0) ** 2 + (zz - z0) ** 2 > radiusSquared) continue;
        const id = xx * yz + yy * nz + zz;
        if (allowed(id) && !seen.has(id)) {
          seen.add(id);
          queue.push(id);
        }
      }
    }
  };
  let current = seed;
  record(seed);
  // Each visit advances at most one eligible face. A guide beyond a
  // boundary can only move the tip along accessible tissue.
  function advance(gx: number, gy: number, gz: number) {
    const pos = pointAt(field, current),
      x = Math.floor(current / yz),
      y = Math.floor(current / nz) % ny,
      z = current % nz;
    let closest = (pos[0] - gx) ** 2 + (pos[1] - gy) ** 2 + (pos[2] - gz) ** 2;
    let next = current;
    for (const [dx, dy, dz] of neighbors) {
      const xx = x + dx,
        yy = y + dy,
        zz = z + dz;
      if (xx < 0 || xx >= nx || yy < 0 || yy >= ny || zz < 0 || zz >= nz) continue;
      const j = xx * yz + yy * nz + zz;
      const d = (pos[0] + dx * s - gx) ** 2 + (pos[1] + dy * s - gy) ** 2 + (pos[2] + dz * s - gz) ** 2;
      if (d < closest - 1e-8 && allowed(j)) {
        closest = d;
        next = j;
      }
    }
    if (next !== current) {
      current = next;
      record(current);
    }
  }
  if (pattern === 'spiral') {
    // Preserve the original geometric sequence for replay comparisons.
    const passes = Math.max(2, Math.min(6, Math.ceil(depth / (tipRadius * 2.4))));
    const samples = passes * Math.ceil((14 * reach) / s);
    for (let sample = 1; sample <= samples; sample++) {
      const t = sample / samples,
        phase = t * passes;
      const radius = reach * 0.84 * Math.abs(Math.sin(Math.PI * phase));
      const angle = phase * TAU * 1.22;
      const gx = center[0] + radius * Math.cos(angle),
        gy = center[1] + radius * Math.sin(angle),
        gz = center[2] - depth * t;
      advance(gx, gy, gz);
    }
  } else {
    overlappingPasses(center, reach, depth, tipRadius, s, (gx, gy, gz) => {
      // The target can slope away from its entry station. Follow nearby
      // exposed target rather than skipping over an unvisited surface lip.
      const x = Math.round((gx - field.origin[0]) / s);
      const y = Math.round((gy - field.origin[1]) / s);
      const z0 = Math.round((gz - field.origin[2]) / s);
      if (x < 0 || x >= nx || y < 0 || y >= ny || z0 < 0 || z0 >= nz) return;
      const column = (x * ny + y) * nz;
      const floor = Math.max(0, z0 - Math.ceil((2 * tipRadius) / s));
      for (let z = z0; z >= floor; z--) {
        const bits = voxels[column + z];
        if (eligibleTumor(bits)) {
          advance(gx, gy, Math.min(gz, field.origin[2] + z * s));
          return;
        }
        if (bits & (1 | 2 | 16)) return;
      }
    });
  }
  return { path, firstHit, seed: center };
}

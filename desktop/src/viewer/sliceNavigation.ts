import { PLANE_AXES, transformPoint, volumeBounds } from './coordinates.ts';
import type { Affine, Point3, Shape3, SlicePlane } from './coordinates.ts';

/** The controls traverse patient-frame planes. For an oblique source these are
 * resampled planes, not acquisition slice numbers. The line through the linked
 * cursor is clipped to the source voxel box, rather than its larger RAS box. */
export function patientSliceNavigation(affine: Affine, inverse: Affine, shape: Shape3, cursor: Point3, plane: SlicePlane) {
  const axis = PLANE_AXES[plane][2];
  const bounds = volumeBounds(shape, affine);
  const voxel = transformPoint(inverse, cursor);
  const step = 1 / Math.hypot(inverse[0][axis], inverse[1][axis], inverse[2][axis]);
  let min = bounds[0][axis], max = bounds[1][axis];
  for (let sourceAxis = 0; sourceAxis < 3; sourceAxis++) {
    const slope = inverse[sourceAxis][axis];
    if (Math.abs(slope) < 1e-12) {
      if (voxel[sourceAxis] < -1e-6 || voxel[sourceAxis] > shape[sourceAxis] - 1 + 1e-6) return null;
      continue;
    }
    const a = cursor[axis] - voxel[sourceAxis] / slope;
    const b = cursor[axis] + (shape[sourceAxis] - 1 - voxel[sourceAxis]) / slope;
    min = Math.max(min, Math.min(a, b));
    max = Math.min(max, Math.max(a, b));
  }
  if (min > max + 1e-6) return null;
  max = Math.max(min, max);
  const value = Math.max(min, Math.min(max, cursor[axis]));
  const count = Math.max(1, Math.ceil((bounds[1][axis] - bounds[0][axis]) / step - 1e-6) + 1);
  const span = bounds[1][axis] - bounds[0][axis];
  const index = span > 0 ? Math.max(1, Math.min(count, Math.round((value - bounds[0][axis]) / span * (count - 1)) + 1)) : 1;
  const resampled = !inverse.slice(0, 3).some(row => Math.abs(row[axis]) > 1e-8 && row.slice(0, 3).every((value, i) => i === axis || Math.abs(value) <= 1e-8));
  return { axis, min, max, step, value, index, count, resampled };
}

export function patientSliceCursor(affine: Affine, inverse: Affine, shape: Shape3, cursor: Point3, plane: SlicePlane, value: number): Point3 {
  const navigation = patientSliceNavigation(affine, inverse, shape, cursor, plane);
  if (!navigation || !Number.isFinite(value)) return cursor;
  const next = [...cursor] as Point3;
  next[navigation.axis] = Math.max(navigation.min, Math.min(navigation.max, value));
  return next;
}

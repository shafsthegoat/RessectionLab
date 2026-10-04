import type { Point3, Shape3 } from "./coordinates";

/** Display-only 0.5 isosurface of source binary voxels; no metric uses this mesh.
 * Marching tetrahedra has a fixed cell diagonal, so neighbouring cells agree.
 * Work is confined to the mask bounding box, with one zero-valued padding cell.
 */
export function maskSurface(mask: Uint8Array, shape: Shape3): Float32Array {
  if (mask.length !== shape[0] * shape[1] * shape[2])
    throw new Error("Mask array length does not match its physical grid.");
  const low = [...shape],
    high = [-1, -1, -1],
    yz = shape[1] * shape[2];
  for (let index = 0; index < mask.length; index++)
    if (mask[index]) {
      const x = Math.floor(index / yz),
        y = Math.floor((index - x * yz) / shape[2]),
        z = index % shape[2];
      [x, y, z].forEach((v, a) => {
        low[a] = Math.min(low[a], v);
        high[a] = Math.max(high[a], v);
      });
    }
  if (high[0] < 0) return new Float32Array();
  const offsets: Point3[] = [
    [0, 0, 0],
    [1, 0, 0],
    [0, 1, 0],
    [1, 1, 0],
    [0, 0, 1],
    [1, 0, 1],
    [0, 1, 1],
    [1, 1, 1],
  ];
  const tetrahedra = [
    [0, 1, 3, 7],
    [0, 3, 2, 7],
    [0, 2, 6, 7],
    [0, 6, 4, 7],
    [0, 4, 5, 7],
    [0, 5, 1, 7],
  ];
  const output: number[] = [];
  const value = (p: Point3) =>
    p.some((v, a) => v < 0 || v >= shape[a])
      ? 0
      : mask[(p[0] * shape[1] + p[1]) * shape[2] + p[2]]
        ? 1
        : 0;
  const midpoint = (a: Point3, b: Point3) =>
    a.map((v, i) => (v + b[i]) / 2) as Point3;
  const triangle = (a: Point3, b: Point3, c: Point3, out: Point3) => {
    const ab = b.map((v, i) => v - a[i]),
      ac = c.map((v, i) => v - a[i]);
    const normal = [
      ab[1] * ac[2] - ab[2] * ac[1],
      ab[2] * ac[0] - ab[0] * ac[2],
      ab[0] * ac[1] - ab[1] * ac[0],
    ];
    if (normal.reduce((sum, v, i) => sum + v * out[i], 0) < 0)
      output.push(...a, ...c, ...b);
    else output.push(...a, ...b, ...c);
  };
  for (let x = low[0] - 1; x <= high[0]; x++)
    for (let y = low[1] - 1; y <= high[1]; y++)
      for (let z = low[2] - 1; z <= high[2]; z++) {
        const points = offsets.map(
            (o) => [x + o[0], y + o[1], z + o[2]] as Point3,
          ),
          values = points.map(value);
        if (values.every((v) => v === values[0])) continue;
        for (const t of tetrahedra) {
          const inside = t.filter((i) => values[i]),
            outside = t.filter((i) => !values[i]);
          if (!inside.length || !outside.length) continue;
          const outward = [0, 1, 2].map(
            (axis) =>
              outside.reduce((sum, i) => sum + points[i][axis], 0) /
                outside.length -
              inside.reduce((sum, i) => sum + points[i][axis], 0) /
                inside.length,
          ) as Point3;
          if (inside.length === 1) {
            const a = points[inside[0]];
            triangle(
              midpoint(a, points[outside[0]]),
              midpoint(a, points[outside[1]]),
              midpoint(a, points[outside[2]]),
              outward,
            );
          } else if (outside.length === 1) {
            const a = points[outside[0]];
            triangle(
              midpoint(a, points[inside[0]]),
              midpoint(a, points[inside[1]]),
              midpoint(a, points[inside[2]]),
              outward,
            );
          } else {
            const a = midpoint(points[inside[0]], points[outside[0]]),
              b = midpoint(points[inside[0]], points[outside[1]]);
            const c = midpoint(points[inside[1]], points[outside[1]]),
              d = midpoint(points[inside[1]], points[outside[0]]);
            triangle(a, b, c, outward);
            triangle(a, c, d, outward);
          }
        }
      }
  return new Float32Array(output);
}

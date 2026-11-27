/** Source arrays remain xyz C order (z fastest); all display coordinates are RAS mm. */
export type Point3 = [number, number, number];
export type Shape3 = [number, number, number];
export type Affine = number[][];
export type SlicePlane = "axial" | "coronal" | "sagittal";
export type Bounds3 = [Point3, Point3];

export const PLANE_AXES: Record<SlicePlane, [number, number, number]> = {
  axial: [0, 1, 2],
  coronal: [0, 2, 1],
  sagittal: [1, 2, 0],
};
export const PLANE_LABELS: Record<
  SlicePlane,
  [string, string, string, string]
> = {
  axial: ["L", "R", "A", "P"],
  coronal: ["L", "R", "S", "I"],
  sagittal: ["P", "A", "S", "I"],
};

export function rasAffine(affine: Affine, frame: string): Affine {
  if (!["RAS", "RAS+", "LPS", "LPS+"].includes(frame))
    throw new Error(`Unsupported coordinate frame: ${frame}`);
  if (
    affine.length !== 4 ||
    affine.some(
      (row) => row.length !== 4 || row.some((x) => !Number.isFinite(x)),
    )
  ) {
    throw new Error("The source affine must be a finite 4 × 4 matrix.");
  }
  if (affine[3].some((x, i) => Math.abs(x - (i === 3 ? 1 : 0)) > 1e-10))
    throw new Error("Invalid homogeneous source affine.");
  return affine.map((row, i) =>
    row.map((x) => x * (frame.startsWith("LPS") && i < 2 ? -1 : 1)),
  );
}

export function transformPoint(affine: Affine, point: Point3): Point3 {
  return [0, 1, 2].map(
    (i) =>
      affine[i][0] * point[0] +
      affine[i][1] * point[1] +
      affine[i][2] * point[2] +
      affine[i][3],
  ) as Point3;
}

export function inverseAffine(m: Affine): Affine {
  const [[a, b, c], [d, e, f], [g, h, i]] = m;
  const determinant =
    a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g);
  if (!Number.isFinite(determinant) || Math.abs(determinant) < 1e-12)
    throw new Error("The source affine is singular.");
  const inverse = [
    [e * i - f * h, c * h - b * i, b * f - c * e],
    [f * g - d * i, a * i - c * g, c * d - a * f],
    [d * h - e * g, b * g - a * h, a * e - b * d],
  ].map((row) => row.map((x) => x / determinant));
  return [
    ...inverse.map((row) => [
      ...row,
      -row.reduce((sum, x, j) => sum + x * m[j][3], 0),
    ]),
    [0, 0, 0, 1],
  ];
}

export function volumeBounds(shape: Shape3, affine: Affine, voxelEdges=false): Bounds3 {
  const low: Point3 = [Infinity, Infinity, Infinity],
    high: Point3 = [-Infinity, -Infinity, -Infinity];
  const edge = voxelEdges ? .5 : 0;
  for (const x of [-edge, shape[0] - 1 + edge])
    for (const y of [-edge, shape[1] - 1 + edge])
      for (const z of [-edge, shape[2] - 1 + edge]) {
        const p = transformPoint(affine, [x, y, z]);
        for (let axis = 0; axis < 3; axis++) {
          low[axis] = Math.min(low[axis], p[axis]);
          high[axis] = Math.max(high[axis], p[axis]);
        }
      }
  return [low, high];
}

export function cIndex(shape: Shape3, voxel: Point3): number {
  return (voxel[0] * shape[1] + voxel[1]) * shape[2] + voxel[2];
}

export function clampCursor(point: Point3, bounds: Bounds3): Point3 {
  return point.map((x, i) =>
    Math.min(bounds[1][i], Math.max(bounds[0][i], x)),
  ) as Point3;
}

export interface SliceRect {
  left: number;
  top: number;
  width: number;
  height: number;
}
/** Equal mm-per-pixel in both axes. Coordinates are in CSS pixels. */
export function sliceRect(
  bounds: Bounds3,
  plane: SlicePlane,
  width: number,
  height: number,
): SliceRect {
  const [a, b] = PLANE_AXES[plane];
  const mmWidth = Math.max(1e-3, bounds[1][a] - bounds[0][a]),
    mmHeight = Math.max(1e-3, bounds[1][b] - bounds[0][b]);
  const availableWidth = Math.max(1, width - 24),
    availableHeight = Math.max(1, height - 54);
  const scale = Math.min(availableWidth / mmWidth, availableHeight / mmHeight);
  const w = mmWidth * scale,
    h = mmHeight * scale;
  return {
    left: (width - w) / 2,
    top: 29 + (availableHeight - h) / 2,
    width: w,
    height: h,
  };
}

export function slicePoint(
  bounds: Bounds3,
  plane: SlicePlane,
  cursor: Point3,
  rect: SliceRect,
  x: number,
  y: number,
): Point3 | null {
  if (
    x < rect.left ||
    y < rect.top ||
    x > rect.left + rect.width ||
    y > rect.top + rect.height
  )
    return null;
  const [a, b] = PLANE_AXES[plane],
    p = [...cursor] as Point3;
  p[a] =
    bounds[0][a] +
    ((x - rect.left) / rect.width) * (bounds[1][a] - bounds[0][a]);
  p[b] =
    bounds[1][b] -
    ((y - rect.top) / rect.height) * (bounds[1][b] - bounds[0][b]);
  return p;
}

/** CPU reference for the GPU's trilinear source sampling, used by analytic tests. */
export function sampleTrilinear(
  data: Float32Array,
  shape: Shape3,
  voxel: Point3,
): number {
  const [nx, ny, nz] = shape;
  let [x, y, z] = voxel;
  if (x < -1e-6 || x > nx - 1 + 1e-6 || y < -1e-6 || y > ny - 1 + 1e-6 ||
      z < -1e-6 || z > nz - 1 + 1e-6) return 0;
  x = Math.max(0, Math.min(nx - 1, x));
  y = Math.max(0, Math.min(ny - 1, y));
  z = Math.max(0, Math.min(nz - 1, z));
  const x0 = Math.floor(x), y0 = Math.floor(y), z0 = Math.floor(z);
  const fx = x - x0, fy = y - y0, fz = z - z0;
  const i = (x0 * ny + y0) * nz + z0;
  const dx = x0 + 1 < nx ? ny * nz : 0;
  const dy = y0 + 1 < ny ? nz : 0;
  const dz = z0 + 1 < nz ? 1 : 0;
  // The physical sampler is also used once per native cortical vertex.
  // Row offsets keep the C-order eight-corner interpolation allocation-free.
  const z00 = data[i] * (1 - fz) + data[i + dz] * fz;
  const z01 = data[i + dy] * (1 - fz) + data[i + dy + dz] * fz;
  const z10 = data[i + dx] * (1 - fz) + data[i + dx + dz] * fz;
  const z11 = data[i + dx + dy] * (1 - fz) + data[i + dx + dy + dz] * fz;
  return (z00 * (1 - fy) + z01 * fy) * (1 - fx) + (z10 * (1 - fy) + z11 * fy) * fx;
}

import type { OperativeField } from '../operative-field';
import type { ConditionedField } from './operativeSurfaceField';
import type { VoxelBounds } from './operativeSurface';

type ColumnRun = { low: number; high: number; lowerShift: number; upperShift: number };
/** Skull and dura are inspected along route-local +z. Fit corresponding
 * upper/lower interfaces across neighbouring columns after occupancy
 * conditioning. A shared translation carries both faces of a thin sheet.
 * Each column is an increasing piecewise-linear map z -> z', including the
 * gaps; blending four such maps preserves ordering for every (x,y). */
export function fitOperativeSheet(
  grid: Pick<OperativeField, 'shape' | 'spacingMm'>,
  field: ConditionedField,
  box: VoxelBounds,
  pos: Float32Array
) {
  const [nx, ny, nz] = grid.shape,
    spacing = grid.spacingMm;
  const sigma = 0.88 / spacing,
    radius = Math.min(8, Math.ceil(2.35 * sigma)),
    maxShift = 0.62 / spacing;
  const x0 = Math.max(0, field.origin[0], box.low[0] - radius - 1),
    x1 = Math.min(nx - 1, field.origin[0] + field.shape[0] - 1, box.high[0] + radius + 2);
  const y0 = Math.max(0, field.origin[1], box.low[1] - radius - 1),
    y1 = Math.min(ny - 1, field.origin[1] + field.shape[1] - 1, box.high[1] + radius + 2);
  const z0 = Math.max(0, field.origin[2] + 1),
    z1 = Math.min(nz - 1, field.origin[2] + field.shape[2] - 2),
    columnCountY = y1 - y0 + 1;
  const columns: ColumnRun[][] = Array.from({ length: (x1 - x0 + 1) * columnCountY }, () => []);
  const column = (x: number, y: number) => columns[(x - x0) * columnCountY + y - y0];
  // The field's conditioning halo contains these neighbouring columns.
  // Crossings of a continuous indicator retain their sub-cell height.
  const crossingFraction = (i: number, j: number) => {
    const a = field.smooth[i],
      b = field.smooth[j];
    if (a === b || field.unresolved[i] || field.unresolved[j]) return 0.5;
    return Math.max(0.08, Math.min(0.92, (127.5 - a) / (b - a)));
  };
  for (let x = x0; x <= x1; x++)
    for (let y = y0; y <= y1; y++) {
      const list = column(x, y),
        base = field.base + x * field.yz + y * field.sz;
      let z = z0;
      while (z <= z1) {
        if (!field.mask[base + z]) {
          z++;
          continue;
        }
        const lo = z;
        while (z + 1 <= z1 && field.mask[base + z + 1]) z++;
        list.push({
          low: lo - 1 + crossingFraction(base + lo - 1, base + lo),
          high: z + crossingFraction(base + z, base + z + 1),
          lowerShift: 0,
          upperShift: 0,
        });
        z++;
      }
    }
  const footprint: number[][] = [];
  for (let dx = -radius; dx <= radius; dx++)
    for (let dy = -radius; dy <= radius; dy++) {
      const d = dx * dx + dy * dy;
      if (d <= radius * radius) footprint.push([dx, dy, Math.exp(-d / (2 * sigma * sigma)), Math.sqrt(d)]);
    }
  const limit = (v: number) => Math.max(-maxShift, Math.min(maxShift, v));
  for (let x = x0; x <= x1; x++)
    for (let y = y0; y <= y1; y++) {
      const list = column(x, y);
      for (const run of list) {
        let lowerShift = 0,
          upperShift = 0,
          lowerWeight = 0,
          upperWeight = 0;
        for (const [dx, dy, w, d] of footprint) {
          const xx = x + dx,
            yy = y + dy;
          if (xx < x0 || xx > x1 || yy < y0 || yy > y1) continue;
          let match: ColumnRun | undefined,
            best = Infinity;
          for (const other of column(xx, yy)) {
            const cost = Math.abs(other.low - run.low) + Math.abs(other.high - run.high);
            if (cost < best) {
              best = cost;
              match = other;
            }
          }
          if (!match) continue;
          const slopeLimit = 1.2 / spacing + 1.25 * d,
            lowerDelta = match.low - run.low,
            upperDelta = match.high - run.high;
          if (Math.abs(lowerDelta) <= slopeLimit) {
            lowerShift += w * lowerDelta;
            lowerWeight += w;
          }
          if (Math.abs(upperDelta) <= slopeLimit) {
            upperShift += w * upperDelta;
            upperWeight += w;
          }
        }
        run.lowerShift = limit(lowerShift / (lowerWeight || 1));
        run.upperShift = limit(upperShift / (upperWeight || 1));
        const thickness = run.high - run.low;
        if (thickness * spacing <= 2.0) {
          run.lowerShift = run.upperShift = (run.lowerShift + run.upperShift) / 2;
        } else if (thickness + run.upperShift - run.lowerShift < thickness * 0.7) {
          const mid = (run.lowerShift + run.upperShift) / 2;
          run.lowerShift = mid + thickness * 0.15;
          run.upperShift = mid - thickness * 0.15;
        }
        if (run.low <= -0.4) run.lowerShift = 0;
        if (run.high >= nz - 0.6) run.upperShift = 0;
      }
      // Prevent different runs or opposite walls from folding over each other.
      // Uniform scaling in a column preserves any shared thin-sheet shift.
      let scale = 1;
      for (let i = 1; i < list.length; i++) {
        const width = list[i].low - list[i - 1].high,
          closing = list[i - 1].upperShift - list[i].lowerShift;
        if (closing > 0) scale = Math.min(scale, (0.65 * width) / closing);
      }
      if (scale < 1)
        for (const run of list) {
          run.lowerShift *= scale;
          run.upperShift *= scale;
        }
    }
  function displacement(x: number, y: number, z: number) {
    const runs = column(Math.max(x0, Math.min(x1, x)), Math.max(y0, Math.min(y1, y)));
    if (!runs.length) return 0;
    let previousHeight = -0.5,
      offset = 0;
    for (const band of runs) {
      if (z < band.low)
        return (
          offset +
          (band.lowerShift - offset) *
            Math.max(0, (z - previousHeight) / Math.max(0.001, band.low - previousHeight))
        );
      if (z <= band.high)
        return (
          band.lowerShift + ((band.upperShift - band.lowerShift) * (z - band.low)) / (band.high - band.low)
        );
      previousHeight = band.high;
      offset = band.upperShift;
    }
    return offset * Math.max(0, 1 - (z - previousHeight) / Math.max(0.001, nz - 0.5 - previousHeight));
  }
  for (let i = 0; i < pos.length; i += 3) {
    const px = pos[i],
      py = pos[i + 1],
      z = pos[i + 2];
    if (z <= -0.5 || z >= nz - 0.5) continue;
    const x = Math.floor(px),
      y = Math.floor(py),
      u = px - x,
      v = py - y;
    const a = displacement(x, y, z) * (1 - u) + displacement(x + 1, y, z) * u;
    const b = displacement(x, y + 1, z) * (1 - u) + displacement(x + 1, y + 1, z) * u;
    pos[i + 2] += a * (1 - v) + b * v;
  }
}

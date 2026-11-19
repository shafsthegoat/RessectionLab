import type { OperativeField } from '../operative-field';

type Grid = Pick<OperativeField, 'shape' | 'spacingMm'> & Partial<Pick<OperativeField, 'voxels'>>;
type Contour = { width: number; x: Float32Array; z: Float32Array };
type Section = {
  row: number;
  owners: Uint8Array;
  reference: Contour;
  nx: number;
  nz: number;
  spacing: number;
  protectedCorners: Uint8Array;
};
const registered = new WeakMap<Uint8Array, { section: Section; owner: number; mutable: boolean }>();

/** Fit one shared planar tessellation. Boundary nodes move along the two
 * connected interface edges; interior nodes follow the same deformation.
 * Source sample ownership and the cap connectivity stay intact. All six
 * incident cap triangles retain positive area through every vertex update. */
function fit(
  owners: Uint8Array,
  nx: number,
  nz: number,
  spacing: number,
  protectedCorners: Uint8Array,
  fixed?: {
    source: Uint8Array;
    owner: number;
    reference: Contour;
    xmin: number;
    xmax: number;
    zmin: number;
    zmax: number;
  }
): Contour {
  const width = nz + 1,
    size = (nx + 1) * width;
  const x = fixed ? fixed.reference.x.slice() : new Float32Array(size);
  const z = fixed ? fixed.reference.z.slice() : new Float32Array(size);
  const near = new Uint8Array(size);
  const previous = new Int32Array(size),
    next = new Int32Array(size);
  previous.fill(-1);
  const nodes: number[][] = [[], [], [], []];
  const radius = Math.min(7, Math.ceil(1.5 / spacing));
  for (let i = 0, k = 0; i <= nx; i++)
    for (let j = 0; j <= nz; j++, k++) {
      if (!fixed) {
        x[k] = i - 0.5;
        z[k] = j - 0.5;
      }
      if (!i || !j || i === nx || j === nz) continue;
      if (
        fixed &&
        (i < fixed.xmin - radius ||
          i > fixed.xmax + 1 + radius ||
          j < fixed.zmin - radius ||
          j > fixed.zmax + 1 + radius)
      )
        continue;
      const a = (i - 1) * nz + j - 1,
        b = i * nz + j - 1;
      const ll = owners[a],
        ul = owners[b],
        uu = owners[b + 1],
        lu = owners[a + 1];
      if (ll === ul && ll === uu && ll === lu) continue;

      const neighbors = [];
      if (ll !== ul) neighbors.push(k - 1);
      if (ul !== uu) neighbors.push(k + width);
      if (uu !== lu) neighbors.push(k + 1);
      if (lu !== ll) neighbors.push(k - width);
      if (neighbors.length !== 2) {
        previous[k] = -2;
        continue;
      }
      previous[k] = neighbors[0];
      next[k] = neighbors[1];
      for (let h = Math.max(1, i - radius); h <= Math.min(nx - 1, i + radius); h++)
        near.fill(1, h * width + Math.max(1, j - radius), h * width + Math.min(nz - 1, j + radius) + 1);
    }
  for (let i = 1; i < nx; i++)
    for (let j = 1; j < nz; j++) {
      const k = i * width + j;
      if (!near[k] || previous[k] === -2) continue;
      if (fixed) {
        const a = (i - 1) * nz + j - 1,
          b = i * nz + j - 1,
          source = fixed.source;
        if (
          source[a] !== fixed.owner ||
          source[a + 1] !== fixed.owner ||
          source[b] !== fixed.owner ||
          source[b + 1] !== fixed.owner
        )
          continue;
      }
      nodes[(i % 2) * 2 + (j % 2)].push(k);
    }
  const maximumShift = 0.8 / spacing;
  const iterations = Math.min(64, Math.max(8, Math.round(4 * (1.1 / spacing) ** 2)));
  const ring = [width, width + 1, 1, -width, -width - 1, -1, width];
  for (let pass = 0; pass < iterations; pass++) {
    for (const group of nodes)
      for (const k of group) {
        const ox = Math.floor(k / width) - 0.5,
          oz = (k % width) - 0.5,
          p = previous[k],
          q = next[k];
        let px: number, pz: number;
        if (p >= 0) {
          px = (x[p] + x[q]) * 0.5;
          pz = (z[p] + z[q]) * 0.5;
        } else {
          px = (x[k - width] + x[k + width] + x[k - 1] + x[k + 1]) * 0.25;
          pz = (z[k - width] + z[k + width] + z[k - 1] + z[k + 1]) * 0.25;
        }
        const length = Math.hypot(px - ox, pz - oz);
        // Keep each protected voxel center inside its original quadrilateral
        // when an eligible target abuts vessels, bone, or dura.
        const limit = protectedCorners[k] ? Math.min(0.42, maximumShift) : maximumShift;
        if (length > limit) {
          px = ox + ((px - ox) * limit) / length;
          pz = oz + ((pz - oz) * limit) / length;
        }
        const dx = px - x[k],
          dz = pz - z[k];
        if (Math.abs(dx) + Math.abs(dz) < 1e-5) continue;
        let step = 0.62;
        for (let t = 0; t < 6; t++) {
          const a = k + ring[t],
            b = k + ring[t + 1];
          const loss = dx * (z[b] - z[a]) - dz * (x[b] - x[a]);
          if (loss <= 0) continue;
          const area = (x[a] - x[k]) * (z[b] - z[k]) - (z[a] - z[k]) * (x[b] - x[k]);
          step = Math.min(step, Math.max(0, (area - 0.12) / loss));
        }
        x[k] += step * dx;
        z[k] += step * dz;
      }
  }
  return { width, x, z };
}

/** Capture the model's already-adjudicated tissue memberships once, before
 * aspiration can mutate the target. Only this 2D inspection plane is cached. */
export function registerOperativeSection(
  grid: Grid,
  masks: Uint8Array[],
  row: number,
  moving: Uint8Array[],
  flap?: { source: Uint8Array; mask: Uint8Array },
  protectedTarget?: Uint8Array
) {
  if (row <= 0 || row >= grid.shape[1]) return;
  const [nx, ny, nz] = grid.shape;
  const owners = new Uint8Array(nx * nz);
  const active = masks.filter((mask) => mask.length);
  active.forEach((mask, index) => {
    for (let x = 0; x < nx; x++) {
      const base = (x * ny + row) * nz,
        out = x * nz;
      for (let z = 0; z < nz; z++) if (mask[base + z]) owners[out + z] = index + 1;
    }
  });
  const protectedCorners = new Uint8Array((nx + 1) * (nz + 1));
  if (grid.voxels && protectedTarget) {
    // A section corner can travel up to .8 mm. Protect the complete source
    // cell around each nearby vessel, dural or osseous center, including
    // thin interfaces whose target was adjacent before aspiration.
    const reach = Math.ceil(0.8 / grid.spacingMm) + 1,
      width = nz + 1;
    for (let x = 0; x < nx; x++)
      for (let z = 0; z < nz; z++) {
        if (!protectedTarget[(x * ny + row) * nz + z]) continue;
        for (let xx = Math.max(0, x - reach); xx <= Math.min(nx - 1, x + reach); xx++)
          for (let zz = Math.max(0, z - reach); zz <= Math.min(nz - 1, z + reach); zz++) {
            if (!(grid.voxels[(xx * ny + row) * nz + zz] & (1 | 2 | 16))) continue;
            const k = xx * width + zz;
            protectedCorners[k] =
              protectedCorners[k + 1] =
              protectedCorners[k + width] =
              protectedCorners[k + width + 1] =
                1;
          }
      }
  }
  const section = {
    row,
    nx,
    nz,
    spacing: grid.spacingMm,
    owners,
    protectedCorners,
    reference: fit(owners, nx, nz, grid.spacingMm, protectedCorners),
  };
  active.forEach((mask, index) =>
    registered.set(mask, { section, owner: index + 1, mutable: moving.includes(mask) })
  );
  if (flap && active.includes(flap.source))
    registered.set(flap.mask, { section, owner: active.indexOf(flap.source) + 1, mutable: true });
}

export function operativeSectionContour(grid: Grid, mask: Uint8Array, row: number): Contour | undefined {
  const entry = registered.get(mask);
  if (!entry || entry.section.row !== row) return;
  const { section, owner, mutable } = entry,
    { nx, nz, owners, reference } = section;
  if (!mutable) return reference;
  const current = owners.slice(),
    ny = grid.shape[1];
  let xmin = nx,
    xmax = -1,
    zmin = nz,
    zmax = -1;
  for (let x = 0; x < nx; x++) {
    const base = (x * ny + row) * nz,
      out = x * nz;
    for (let z = 0; z < nz; z++)
      if (owners[out + z] === owner && !mask[base + z]) {
        current[out + z] = 0;
        xmin = Math.min(xmin, x);
        xmax = Math.max(xmax, x);
        zmin = Math.min(zmin, z);
        zmax = Math.max(zmax, z);
      }
  }
  if (xmax < 0) return reference;
  return fit(current, nx, nz, section.spacing, section.protectedCorners, {
    source: owners,
    owner,
    reference,
    xmin,
    xmax,
    zmin,
    zmax,
  });
}

/** Carry the fitted section into the first millimetre of the connected wall.
 * A smooth taper meets the unaltered mesh beyond that band. A fitted skull
 * edge already has a physical column correction; crossfade to the joint
 * boundary at the plane so the two corrections cannot accumulate. */
export function carrySectionContour(
  contour: Contour,
  row: number,
  spacing: number,
  pos: Float32Array,
  rest: Float32Array,
  sheet: boolean
) {
  const width = contour.width,
    height = contour.x.length / width;
  for (let i = 0; i < pos.length; i += 3) {
    const d = (pos[i + 1] - (row - 0.5)) * spacing,
      t = Math.min(1, Math.max(0, d / 1.2));
    if (t >= 1) continue;
    const w = 1 - t * t * (3 - 2 * t);
    const px = pos[i],
      pz = sheet ? pos[i + 2] * (1 - w) + rest[i + 2] * w : pos[i + 2];
    const xx = Math.max(0, Math.min(height - 1, px + 0.5)),
      zz = Math.max(0, Math.min(width - 1, pz + 0.5));
    const x = Math.min(height - 2, Math.floor(xx)),
      z = Math.min(width - 2, Math.floor(zz));
    const a = xx - x,
      b = zz - z,
      k = x * width + z;
    // Piecewise-linear map with the same diagonal as the cap triangles.
    const v = a > b ? k + width : k + 1,
      u = k + width + 1;
    const w0 = 1 - Math.max(a, b),
      w1 = Math.abs(a - b),
      w2 = Math.min(a, b);
    pos[i] = px + w * (w0 * contour.x[k] + w1 * contour.x[v] + w2 * contour.x[u] - px);
    pos[i + 2] = pz + w * (w0 * contour.z[k] + w1 * contour.z[v] + w2 * contour.z[u] - pz);
  }
}

/** Area-weighted display normals at exactly coincident source-surface vertices.
 * Positions, triangle order, source masks and every quantitative metric remain
 * unchanged. No distance tolerance joins nearby vertices or smooths geometry.
 */
export function surfaceNormals(positions: Float32Array): Float32Array {
  if (positions.length % 9 !== 0)
    throw new Error(
      "Source surface positions must contain complete triangles.",
    );
  // Binary marching-tetrahedra vertices lie exactly on the half-voxel grid.
  // Pack those integer coordinates injectively, without rounding or tolerance.
  // Non-grid callers use exact string identities instead of quantization.
  const low = [Infinity, Infinity, Infinity],
    high = [-Infinity, -Infinity, -Infinity];
  let halfGrid = true;
  for (let i = 0; i < positions.length; i++) {
    const value = positions[i] * 2,
      axis = i % 3;
    if (!Number.isSafeInteger(value)) halfGrid = false;
    low[axis] = Math.min(low[axis], value);
    high[axis] = Math.max(high[axis], value);
  }
  const span = high.map((value, axis) => value - low[axis] + 1);
  const safeGrid =
    halfGrid && Number.isSafeInteger(span[0] * span[1] * span[2]);
  const key: (offset: number) => number | string = safeGrid
    ? (offset) =>
        (positions[offset] * 2 - low[0]) * span[1] * span[2] +
        (positions[offset + 1] * 2 - low[1]) * span[2] +
        positions[offset + 2] * 2 -
        low[2]
    : (offset) =>
        `${positions[offset]},${positions[offset + 1]},${positions[offset + 2]}`;
  const sums = new Map<number | string, [number, number, number]>();
  const faceNormal = (offset: number): [number, number, number] => {
    const abx = positions[offset + 3] - positions[offset],
      aby = positions[offset + 4] - positions[offset + 1],
      abz = positions[offset + 5] - positions[offset + 2],
      acx = positions[offset + 6] - positions[offset],
      acy = positions[offset + 7] - positions[offset + 1],
      acz = positions[offset + 8] - positions[offset + 2];
    return [
      aby * acz - abz * acy,
      abz * acx - abx * acz,
      abx * acy - aby * acx,
    ];
  };
  for (let i = 0; i < positions.length; i += 9) {
    const normal = faceNormal(i);
    if (!normal.every(Number.isFinite) || Math.hypot(...normal) === 0)
      throw new Error("Source surface contains an invalid or degenerate face.");
    for (let corner = 0; corner < 9; corner += 3) {
      const id = key(i + corner),
        sum = sums.get(id) ?? [0, 0, 0];
      for (let axis = 0; axis < 3; axis++) sum[axis] += normal[axis];
      sums.set(id, sum);
    }
  }
  for (const sum of sums.values()) {
    const length = Math.hypot(...sum);
    if (length > 0) for (let axis = 0; axis < 3; axis++) sum[axis] /= length;
  }
  const output = new Float32Array(positions.length);
  for (let i = 0; i < positions.length; i += 3) {
    let normal = sums.get(key(i))!;
    // Opposing faces can meet at one source-grid corner. Keep their own face
    // direction when their sum cancels; do not manufacture a shared direction.
    if (normal[0] === 0 && normal[1] === 0 && normal[2] === 0) {
      normal = faceNormal(Math.floor(i / 9) * 9);
      const length = Math.hypot(...normal);
      for (let axis = 0; axis < 3; axis++) normal[axis] /= length;
    }
    for (let axis = 0; axis < 3; axis++) output[i + axis] = normal[axis];
  }
  return output;
}

/** Area weighting in millimetres, returned as source-frame normal covectors so
 * Three's normal matrix still applies the authoritative affine exactly once. */
export function indexedSurfaceNormals(points:Float32Array,indices:Uint32Array,affine:number[][]):Float32Array {
  const sums=new Float64Array(points.length),linear=affine.slice(0,3).map(r=>r.slice(0,3));
  const determinant=linear[0][0]*(linear[1][1]*linear[2][2]-linear[1][2]*linear[2][1])-linear[0][1]*(linear[1][0]*linear[2][2]-linear[1][2]*linear[2][0])+linear[0][2]*(linear[1][0]*linear[2][1]-linear[1][1]*linear[2][0]);
  const [[xx,xy,xz],[yx,yy,yz],[zx,zy,zz]]=linear,sign=Math.sign(determinant);
  // Keep the retained reduction's addition order, without allocating per-face
  // arrays/closures for the millions of source triangles in a patient surface.
  for(let f=0;f<indices.length;f+=3){const i=indices[f]*3,j=indices[f+1]*3,k=indices[f+2]*3;
    const jx=points[j]-points[i],jy=points[j+1]-points[i+1],jz=points[j+2]-points[i+2],
      kx=points[k]-points[i],ky=points[k+1]-points[i+1],kz=points[k+2]-points[i+2];
    const ax=((0+xx*jx)+xy*jy)+xz*jz,ay=((0+yx*jx)+yy*jy)+yz*jz,az=((0+zx*jx)+zy*jy)+zz*jz,
      bx=((0+xx*kx)+xy*ky)+xz*kz,by=((0+yx*kx)+yy*ky)+yz*kz,bz=((0+zx*kx)+zy*ky)+zz*kz;
    const nx=(ay*bz-az*by)*sign,ny=(az*bx-ax*bz)*sign,nz=(ax*by-ay*bx)*sign;
    sums[i]+=nx;sums[i+1]+=ny;sums[i+2]+=nz;
    sums[j]+=nx;sums[j+1]+=ny;sums[j+2]+=nz;
    sums[k]+=nx;sums[k+1]+=ny;sums[k+2]+=nz;
  }
  const output=new Float32Array(points.length);
  for(let i=0;i<points.length;i+=3){const x=sums[i],y=sums[i+1],z=sums[i+2],
      nx=((0+xx*x)+yx*y)+zx*z,ny=((0+xy*x)+yy*y)+zy*z,nz=((0+xz*x)+yz*y)+zz*z,length=Math.hypot(nx,ny,nz);
    output[i]=length?nx/length:0;output[i+1]=length?ny/length:0;output[i+2]=length?nz/length:0;}
  return output;
}

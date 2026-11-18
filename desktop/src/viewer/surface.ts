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
      low[0]=Math.min(low[0],x);low[1]=Math.min(low[1],y);low[2]=Math.min(low[2],z);
      high[0]=Math.max(high[0],x);high[1]=Math.max(high[1],y);high[2]=Math.max(high[2],z);
    }
  if (high[0] < 0) return new Float32Array();
  return maskSurfaceRegion(mask,shape,low.map(n=>n-1) as Point3,high as Point3).positions;
}

// Each binary cube has one of 256 triangulations. Build the table from the
// original tetrahedra and winding once. Positions are exact half integers.
function cellTemplate(bits:number){
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
  const points=offsets,values=offsets.map((_,i)=>(bits>>i)&1);
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

  return Float32Array.from(output);
}
const cellTemplates=Array.from({length:256},(_,i)=>cellTemplate(i));

/** Inclusive cell-origin core. Reads the one-cell source halo and preserves
 * raster order for global indexing, unchanged normals and incremental edits. */
export function maskSurfaceRegion(mask:Uint8Array,shape:Shape3,low:Point3,high:Point3){
  if(mask.length!==shape[0]*shape[1]*shape[2]||low.some((n,a)=>!Number.isInteger(n)||n< -1||n>high[a])||
    high.some((n,a)=>!Number.isInteger(n)||n>=shape[a]))throw Error('Invalid source surface cell region');
  const [nx,ny,nz]=shape;
  const rowOffsets=new Uint32Array((high[0]-low[0]+1)*(high[1]-low[1]+1)+1);
  const rowBase=(x:number,y:number)=>x>=0&&x<nx&&y>=0&&y<ny?(x*ny+y)*nz:-1;
  const value=(row:number,z:number)=>row>=0&&z>=0&&z<nz&&mask[row+z]?1:0;
  // Two scans allocate the exact transferable size. Avoid a JS array of
  // boxed coordinates whose peak size grows with the full patient surface.
  let positions:Float32Array|undefined;
  for(let pass=0;pass<2;pass++){
    let cursor=0,row=0;
    for(let x=low[0];x<=high[0];x++)for(let y=low[1];y<=high[1];y++){
      rowOffsets[row++]=cursor;
      const a=rowBase(x,y),b=rowBase(x+1,y),c=rowBase(x,y+1),d=rowBase(x+1,y+1);
      let bottom=value(a,low[2])|value(b,low[2])<<1|value(c,low[2])<<2|value(d,low[2])<<3;
      for(let z=low[2];z<=high[2];z++){
        const top=value(a,z+1)|value(b,z+1)<<1|value(c,z+1)<<2|value(d,z+1)<<3;
        const cells=cellTemplates[bottom|(top<<4)];bottom=top;
        if(!positions){cursor+=cells.length;continue;}
        for(let i=0;i<cells.length;i+=3){
          positions[cursor++]=x+cells[i];positions[cursor++]=y+cells[i+1];positions[cursor++]=z+cells[i+2];
        }
      }
    }
    rowOffsets[row]=cursor;
    if(!positions)positions=new Float32Array(cursor);
  }
  return {positions:positions!,rowOffsets};
}

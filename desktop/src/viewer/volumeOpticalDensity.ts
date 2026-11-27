import {DataUtils} from 'three';
import type {Affine, Point3} from './coordinates';
import {mriBoundaryExtinction,VOLUME_TRANSFER_WIDTH} from './volumeTransfer.ts';

type OpticalTransfer = {data: Float32Array; range: [number, number]; gradientOpacity: boolean};
export type OpticalExterior = {offset: Point3; shape: Point3; data: Uint8Array};
export const CT_EXTERIOR_AIR_HU = -1024;
export const CT_EXTERIOR_TEXTURE_LIMIT_BYTES = 128*1024*1024;
export type OpticalSource = {
  signal: Float32Array;
  shape: Point3;
  affine: Affine;
  inverse: Affine;
  support?: Uint8Array;
};
export type OpticalContext = {coverage: Uint8Array};

/** A centered physical footprint for the transparent bone channel only. The
 * one-quarter center weight and six one-eighth face weights cancel odd
 * mixed moments. Store one coverage grid for the eye ray and shadow atlas. */
export function ctOpticalCoverage(mask:Uint8Array, shape:Point3, frame:Affine) {
  const [nx,ny,nz]=shape;
  const coverage=new Uint8Array(mask.length);
  const axis=(n:number,stride:number,i:number)=>{
    const radius=Math.min(1.45,1/Math.hypot(frame[0][i],frame[1][i],frame[2][i]));
    const near=Math.floor(radius),far=near+1,outer=radius-near;
    const offsets=(step:number)=>Int32Array.from({length:n},(_,j)=>Math.max(0,Math.min(n-1,j+step))*stride);
    return {inner:1-outer,outer,nearLow:offsets(-near),nearHigh:offsets(near),farLow:offsets(-far),farHigh:offsets(far)};
  };
  const x=axis(nx,ny*nz,0),y=axis(ny,nz,1),z=axis(nz,1,2);
  const inside=(p:number)=>Number(mask[p]!==0);
  for(let i=0;i<nx;i++)for(let j=0;j<ny;j++){
    const row=(i*ny+j)*nz;
    const xNearLow=x.nearLow[i]+j*nz,xNearHigh=x.nearHigh[i]+j*nz;
    const xFarLow=x.farLow[i]+j*nz,xFarHigh=x.farHigh[i]+j*nz;
    const yNearLow=i*ny*nz+y.nearLow[j],yNearHigh=i*ny*nz+y.nearHigh[j];
    const yFarLow=i*ny*nz+y.farLow[j],yFarHigh=i*ny*nz+y.farHigh[j];
    for(let k=0;k<nz;k++){
      const xFaces=x.inner*(inside(xNearLow+k)+inside(xNearHigh+k))+x.outer*(inside(xFarLow+k)+inside(xFarHigh+k));
      const yFaces=y.inner*(inside(yNearLow+k)+inside(yNearHigh+k))+y.outer*(inside(yFarLow+k)+inside(yFarHigh+k));
      const zFaces=z.inner*(inside(row+z.nearLow[k])+inside(row+z.nearHigh[k]))+
        z.outer*(inside(row+z.farLow[k])+inside(row+z.farHigh[k]));
      coverage[row+k]=Math.round((2*inside(row+k)+xFaces+yFaces+zFaces)*255/8);
    }
  }
  return coverage;
}

/** Estimate typical TOF backdrop brightness from short, fixed native-grid
 * quadratures. Physical ray lengths and gradients account for anisotropy
 * without using vessel segmentation or the camera. */
export function balancedTofDensity(source:OpticalSource, transfer:OpticalTransfer):number|undefined {
  const {signal,shape,affine,inverse,support}=source;
  const stride=[shape[1]*shape[2],shape[2],1];
  const [a,b,c]=inverse;
  const inverseRange=1/(transfer.range[1]-transfer.range[0]);
  const radiance=Float32Array.from({length:VOLUME_TRANSFER_WIDTH},(_,i)=>{
    const q=i*4;
    return (transfer.data[q]*.2126+transfer.data[q+1]*.7152+transfer.data[q+2]*.0722)*transfer.data[q+3];
  });
  const rays:number[]=[];
  for(let axis=0;axis<3;axis++){
    const first=(axis+1)%3, second=(axis+2)%3;
    const length=shape[axis]*Math.hypot(affine[0][axis],affine[1][axis],affine[2][axis]);
    const samples=Math.min(shape[axis],192,Math.max(1,Math.ceil(length/2)));
    const step=length/samples;
    const voxel:Point3=[0,0,0];
    for(let i=0;i<9;i++)for(let j=0;j<9;j++){
      voxel[first]=Math.floor(shape[first]*(.15+.7*(i+.5)/9));
      voxel[second]=Math.floor(shape[second]*(.15+.7*(j+.5)/9));
      let integral=0;
      for(let s=0;s<samples;s++){
        voxel[axis]=Math.floor(shape[axis]*(s+.5)/samples);
        const [x,y,z]=voxel;
        const q=x*stride[0]+y*stride[1]+z;
        const value=signal[q];
        if((support&&!support[q])||!Number.isFinite(value))continue;
        const k=Math.max(0,Math.min(VOLUME_TRANSFER_WIDTH-1,(value-transfer.range[0])*inverseRange*VOLUME_TRANSFER_WIDTH-.5));
        const lo=Math.floor(k),hi=Math.min(lo+1,VOLUME_TRANSFER_WIDTH-1);
        const light=radiance[lo]+(radiance[hi]-radiance[lo])*(k-lo);
        if(light===0)continue;
        const gx=signal[q+(x<shape[0]-1?stride[0]:0)]-signal[q-(x>0?stride[0]:0)];
        const gy=signal[q+(y<shape[1]-1?stride[1]:0)]-signal[q-(y>0?stride[1]:0)];
        const gz=signal[q+(z<shape[2]-1?1:0)]-signal[q-(z>0?1:0)];
        const gradient=.5*inverseRange*Math.hypot(a[0]*gx+b[0]*gy+c[0]*gz,a[1]*gx+b[1]*gy+c[1]*gz,a[2]*gx+b[2]*gy+c[2]*gz);
        integral+=light*mriBoundaryExtinction(Number.isFinite(gradient)?gradient:0)*step;
      }
      rays.push(integral);
    }
  }
  rays.sort((x,y)=>x-y);
  const typical=rays[Math.floor(rays.length/2)];
  // Limit the boost on faint acquisitions so context stays subordinate to fine vessels.
  return typical>.05?Math.max(.008,Math.min(.035,.1224/typical)):undefined;
}

/** A separate display texture preserves the raw acquisition and provides the
 * exact same air-filled values to primary sampling and both shadow scales. */
export function ctExteriorDisplaySignal(signal:Float32Array, shape:Point3, exterior:OpticalExterior) {
  if(signal.byteLength>CT_EXTERIOR_TEXTURE_LIMIT_BYTES)return;
  const {offset,shape:crop,data}=exterior;
  if(signal.length!==shape[0]*shape[1]*shape[2]||data.length!==crop[0]*crop[1]*crop[2]||
    crop.some((size,axis)=>!Number.isInteger(size)||size<1||!Number.isInteger(offset[axis])||
      offset[axis]<0||offset[axis]+size>shape[axis]))return;

  const display=signal.slice();
  const [nx,ny,nz]=crop;
  const [,,z]=shape,stride=shape[1]*z;
  for(let x=0;x<nx;x++)for(let y=0;y<ny;y++){
    const native=(x+offset[0])*stride+(y+offset[1])*z+offset[2];
    const cropped=(x*ny+y)*nz;
    for(let k=0;k<nz;k++)if(data[cropped+k])display[native+k]=CT_EXTERIOR_AIR_HU;
  }
  return display;
}

const CACHE_CELL_MM = 4;
const MAX_CELLS_PER_AXIS = 56;
export const OPTICAL_KEY_LIGHTS: readonly Point3[] = [[-.43,-.35,.83],[-.63,-.26,.73]];

// Uniformly stratified source samples in a cache cell. Sampling density is
// bounded even for a large import. The visible primary ray still reads exact
// full-resolution source voxels, so secondary-ray aliasing cannot erase tissue.
function cellSamples(length: number, cells: number) {
  return Array.from({length: cells}, (_, cell) => {
    const from = Math.floor(cell * length / cells);
    const to = Math.floor((cell + 1) * length / cells);
    const count = Math.min(to - from, 3);
    return Array.from({length: count}, (_, i) => from + Math.floor((i + .5) * (to - from) / count));
  });
}

function* opticalDensitySteps(source: OpticalSource, transfer: OpticalTransfer, context?: OpticalContext) {
  const {signal, shape, affine, inverse, support} = source;
  const coverage = context?.coverage;
  const grid = shape.map((size, axis) => Math.min(size, MAX_CELLS_PER_AXIS,
    Math.max(1, Math.ceil(size * Math.hypot(affine[0][axis], affine[1][axis], affine[2][axis]) / CACHE_CELL_MM)))) as Point3;
  const [nx, ny, nz] = grid;
  const [sx, sy, sz] = shape;
  const xStride = sy * sz;
  const fine = new Float32Array(nx * ny * nz * 2);
  const [xs, ys, zs] = grid.map((cells, axis) => cellSamples(shape[axis], cells));
  const inverseRange = 1 / (transfer.range[1] - transfer.range[0]);
  const toLut = VOLUME_TRANSFER_WIDTH * inverseRange;
  const [a, b, c] = inverse;

  for (let i = 0; i < nx; i++) for (let j = 0; j < ny; j++) {
    for (let k = 0; k < nz; k++) {
      let extinction = 0;
      let secondaryExtinction = 0;
      for (const x of xs[i]) for (const y of ys[j]) for (const z of zs[k]) {
        const q = x * xStride + y * sz + z;
        if (support && !support[q]) continue;
        if (!Number.isFinite(signal[q])) continue;
        // Transfer before averaging. Air beside dense bone cannot be evaluated
        // as a single synthetic soft-tissue HU at the block-average intensity.
        const lut = Math.max(0, Math.min(VOLUME_TRANSFER_WIDTH - 1, (signal[q] - transfer.range[0]) * toLut - .5));
        const lo = Math.floor(lut);
        const hi = Math.min(lo + 1, VOLUME_TRANSFER_WIDTH - 1);
        const from = transfer.data[4 * lo + 3];
        const density = from + (transfer.data[4 * hi + 3] - from) * (lut - lo);
        extinction += density;
        if (!density) continue;
        if (coverage){
          secondaryExtinction += density * coverage[q] / 255;
          continue;
        }
        if (!transfer.gradientOpacity) continue;
        const gx = signal[q + (x < sx - 1 ? xStride : 0)] - signal[q - (x > 0 ? xStride : 0)];
        const gy = signal[q + (y < sy - 1 ? sz : 0)] - signal[q - (y > 0 ? sz : 0)];
        const gz = signal[q + (z < sz - 1 ? 1 : 0)] - signal[q - (z > 0 ? 1 : 0)];
        const dx = a[0] * gx + b[0] * gy + c[0] * gz;
        const dy = a[1] * gx + b[1] * gy + c[1] * gz;
        const dz = a[2] * gx + b[2] * gy + c[2] * gz;
        const gradient = .5 * inverseRange * Math.sqrt(dx * dx + dy * dy + dz * dz);
        secondaryExtinction += density * mriBoundaryExtinction(Number.isFinite(gradient) ? gradient : 0);
      }
      const sampleCount = xs[i].length * ys[j].length * zs[k].length;
      const index = 2 * ((i * ny + j) * nz + k);
      fine[index] = extinction / sampleCount;
      fine[index + 1] = coverage || transfer.gradientOpacity ? secondaryExtinction / sampleCount : fine[index];
    }
    yield;
  }

  // A separable [1,2,1] blur supplies the wider footprint of a distant cone.
  // Zero outside the sampled field avoids causing shadows from repeated edges.
  const coarse = new Float32Array(fine.length);
  const scratch = new Float32Array(fine.length);
  let previous = fine;
  for (const [axis, stride] of [ny * nz, nz, 1].entries()) {
    const next = axis === 1 ? scratch : coarse;
    for (let i = 0; i < nx; i++) {
      for (let j = 0; j < ny; j++) for (let k = 0; k < nz; k++) {
        const coordinate = axis === 0 ? i : axis === 1 ? j : k;
        const index = 2 * ((i * ny + j) * nz + k);
        const lo = index - 2 * stride;
        const hi = index + 2 * stride;
        for (let channel = 0; channel < 2; channel++) {
          next[index + channel] = .5 * previous[index + channel]
            + (coordinate > 0 ? .25 * previous[lo + channel] : 0)
            + (coordinate < grid[axis] - 1 ? .25 * previous[hi + channel] : 0);
        }
      }
      yield;
    }
    previous = next;
  }
  return {grid, fine, coarse};
}

export function finishOpticalPreparation<T>(steps: Generator<void, T>) {
  let result=steps.next();
  while(!result.done)result=steps.next();
  return result.value;
}

export function buildOpticalDensity(source: OpticalSource, transfer: OpticalTransfer, context?: OpticalContext) {
  return finishOpticalPreparation(opticalDensitySteps(source,transfer,context));
}

// Sweep towards the emitter along the major cache-grid axis. At each step
// use the two transverse coordinates to interpolate the next slice's depth.
// The travel distance is in mm even for anisotropic, sheared or mirrored grids.
function* opticalDepthSteps(source: Pick<OpticalSource,'shape'|'inverse'>, grid: Point3,
  extinction: Float32Array, direction: Point3) {
  const magnitude = Math.hypot(...direction);
  const velocity = grid.map((cells, axis) => {
    const row = source.inverse[axis];
    return (row[0]*direction[0]+row[1]*direction[1]+row[2]*direction[2])*cells/(source.shape[axis]*magnitude);
  });
  const major = [0,1,2].reduce((best,axis) => Math.abs(velocity[axis])>Math.abs(velocity[best])?axis:best);
  const [b,c] = [0,1,2].filter(axis => axis!==major);
  const sign = Math.sign(velocity[major]);
  const stepMm = 1/Math.abs(velocity[major]);
  const stride = [2*grid[1]*grid[2],2*grid[2],2];
  const depth = new Float32Array(extinction.length);
  const exit = (axis:number, cell:number) => {
    const speed=velocity[axis];
    if(speed===0)return Infinity;
    return ((speed>0?grid[axis]-.5:-.5)-cell)/speed;
  };
  // Transverse offsets and their weights are shared by all sweep planes.
  const lanes = (axis:number) => Array.from({length:grid[axis]},(_,cell)=>{
    const next=Math.max(0,Math.min(grid[axis]-1,cell+velocity[axis]*stepMm));
    const lo=Math.floor(next);
    return {lo:lo*stride[axis],hi:Math.min(grid[axis]-1,lo+1)*stride[axis],t:next-lo,exit:exit(axis,cell)};
  });
  const bLanes=lanes(b),cLanes=lanes(c);
  for(let plane=0;plane<grid[major];plane++){
    const i=sign>0?grid[major]-1-plane:plane;
    const limit=exit(major,i);
    const base=i*stride[major];
    const nextBase=base+sign*stride[major];
    for(let j=0;j<grid[b];j++)for(let k=0;k<grid[c];k++){
      const u=bLanes[j],v=cLanes[k];
      const index=base+j*stride[b]+k*stride[c];
      const remaining=Math.min(limit,u.exit,v.exit);
      if(remaining<=stepMm){
        // The final partial cell integrates only up to the acquisition edge.
        depth[index]=extinction[index]*remaining;
        depth[index+1]=extinction[index+1]*remaining;
        continue;
      }
      const p00=nextBase+u.lo+v.lo,p01=nextBase+u.lo+v.hi;
      const p10=nextBase+u.hi+v.lo,p11=nextBase+u.hi+v.hi;
      const w00=(1-u.t)*(1-v.t),w01=(1-u.t)*v.t,w10=u.t*(1-v.t),w11=u.t*v.t;
      for(let channel=0;channel<2;channel++){
        const downstream=depth[p00+channel]*w00+depth[p01+channel]*w01+depth[p10+channel]*w10+depth[p11+channel]*w11;
        const edge=extinction[p00+channel]*w00+extinction[p01+channel]*w01+extinction[p10+channel]*w10+extinction[p11+channel]*w11;
        depth[index+channel]=downstream+.5*(extinction[index+channel]+edge)*stepMm;
      }
    }
    yield;
  }
  return depth;
}

export function buildDirectionalOpticalDepth(source: Pick<OpticalSource,'shape'|'inverse'>, grid: Point3,
  extinction: Float32Array, direction: Point3) {
  return finishOpticalPreparation(opticalDepthSteps(source,grid,extinction,direction));
}

// The ray marcher already uses the 16 fragment samplers guaranteed by WebGL2.
// A single RGBA16F texture gives the transfer row and a trilinear density atlas
// two scales and a second extinction channel without taking another sampler.
// The second pair stores MRI boundary extinction or the CT skull contribution.
export function* opticalAtlasSteps(transfer: OpticalTransfer, source?: OpticalSource, context?: OpticalContext) {
  const density = source ? yield* opticalDensitySteps(source, transfer, context) : undefined;
  const grid: Point3 = density?.grid ?? [1, 1, 1];
  const [nx, ny, nz] = grid;
  const columns = Math.floor(VOLUME_TRANSFER_WIDTH / nz);
  const stripRows = density ? Math.ceil(nx / columns) * ny : 0;
  const depthRow = stripRows ? 1 + stripRows : 0;
  const height = 1 + 2*stripRows;
  const data = new Uint16Array(VOLUME_TRANSFER_WIDTH * height * 4);
  for (let q = 0; q < transfer.data.length; q++) data[q] = DataUtils.toHalfFloat(transfer.data[q]);
  if (source && density) {
    const {fine, coarse} = density;
    const first = yield* opticalDepthSteps(source,grid,fine,OPTICAL_KEY_LIGHTS[0]);
    const second = yield* opticalDepthSteps(source,grid,fine,OPTICAL_KEY_LIGHTS[1]);
    for (let x = 0; x < nx; x++) {
      for (let y = 0; y < ny; y++) for (let z = 0; z < nz; z++) {
        const from = 2 * ((x * ny + y) * nz + z);
        const row = 1 + Math.floor(x / columns) * ny + y;
        const col = x % columns * nz + z;
        const to = 4 * (row * VOLUME_TRANSFER_WIDTH + col);
        data[to] = DataUtils.toHalfFloat(fine[from]);
        data[to + 1] = DataUtils.toHalfFloat(coarse[from]);
        data[to + 2] = DataUtils.toHalfFloat(fine[from + 1]);
        data[to + 3] = DataUtils.toHalfFloat(coarse[from + 1]);
        const shadow = to + 4*stripRows*VOLUME_TRANSFER_WIDTH;
        data[shadow] = DataUtils.toHalfFloat(first[from]);
        data[shadow + 1] = DataUtils.toHalfFloat(second[from]);
        data[shadow + 2] = DataUtils.toHalfFloat(first[from + 1]);
        data[shadow + 3] = DataUtils.toHalfFloat(second[from + 1]);
      }
      yield;
    }
  }
  return {data, width: VOLUME_TRANSFER_WIDTH, height, grid, columns, depthRow};
}

export function volumeOpticalAtlas(transfer: OpticalTransfer, source?: OpticalSource) {
  return finishOpticalPreparation(opticalAtlasSteps(transfer,source));
}

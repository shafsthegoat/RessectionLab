import { PLANE_AXES, slicePoint, transformPoint, volumeBounds } from './coordinates.ts';
import type { Affine, Bounds3, Point3, Shape3, SlicePlane, SliceRect } from './coordinates.ts';

export interface SliceFraming { fit: 'image'|'anatomy'; zoom: number }
export interface MprView { bounds: Bounds3; rect: SliceRect; fit: SliceFraming['fit']; zoom: number }
export function sliceFraming(input?: Partial<SliceFraming>): SliceFraming {
  return { fit: input?.fit === 'image' ? 'image' : 'anatomy',
    zoom: typeof input?.zoom === 'number' && Number.isFinite(input.zoom) ? Math.max(.5, Math.min(4, input.zoom)) : 1 };
}

const supportCache = new WeakMap<Float32Array, { key: string; bounds: Bounds3 }>();
/** Display framing only: robust signal extent in RAS, with a safety margin.
 * Sparse bright outliers and low background noise must not determine the fit.
 * This is deliberately not an anatomical segmentation or an acquisition crop.
 * Full image fit always uses the complete source bounds and source texture. */
export function sourceSignalBounds(data: Float32Array, shape: Shape3, affine: Affine): Bounds3 {
  const key = JSON.stringify([shape, affine]), cached = supportCache.get(data);
  if (cached?.key === key) return cached.bounds;
  const full = volumeBounds(shape, affine), samples: number[] = [];
  const stride = Math.max(1, Math.floor(data.length / 32000));
  for (let i = 0; i < data.length; i += stride)
    if (Number.isFinite(data[i]) && data[i] !== 0) samples.push(data[i]);
  samples.sort((a,b) => a-b);
  const low = Math.min(0,samples[Math.floor(samples.length * .005)]), high = samples[Math.min(samples.length-1, Math.floor(samples.length*.995))];
  let result = full;
  if (samples.length && high > low) {
    const threshold = low + .03 * (high-low), bins = 512;
    const counts = [new Uint32Array(bins), new Uint32Array(bins), new Uint32Array(bins)];
    const spans = full[1].map((n,a) => Math.max(1e-6, n-full[0][a]));
    let total = 0;
    const [nx,ny,nz] = shape;
    for (let x=0,index=0; x<nx; x++) for (let y=0; y<ny; y++) for (let z=0; z<nz; z++,index++) {
      if (!Number.isFinite(data[index]) || data[index] <= threshold) continue;
      total++;
      for (let a=0; a<3; a++) {
        const world = affine[a][0]*x + affine[a][1]*y + affine[a][2]*z + affine[a][3];
        const bin = Math.max(0, Math.min(bins-1, Math.floor((world-full[0][a])/spans[a]*bins)));
        counts[a][bin]++;
      }
    }
    if (total >= 16) {
      const fitted: Bounds3 = [[...full[0]], [...full[1]]];
      for (let a=0; a<3; a++) {
        let first=0,last=bins-1,tail=0;
        while (first < last && tail+counts[a][first] <= total*.0005) tail += counts[a][first++];
        tail=0;
        while (last > first && tail+counts[a][last] <= total*.0005) tail += counts[a][last--];
        const lo=full[0][a]+first/bins*spans[a], hi=full[0][a]+(last+1)/bins*spans[a];
        const margin=Math.max(4, (hi-lo)*.04);
        fitted[0][a]=Math.max(full[0][a], lo-margin);
        fitted[1][a]=Math.min(full[1][a], hi+margin);
      }
      // A few hot pixels do not constitute enough support to fit a volume.
      if (fitted[0].every((n,a) => fitted[1][a]-n >= Math.min(10, spans[a]*.1))) result=fitted;
    }
  }
  supportCache.set(data,{key,bounds:result});
  return result;
}

/** Coordinates are local to the image element, whose header is already outside
 * its viewport. Equal physical scale is maintained even when zoom exceeds it. */
export function mprView(full: Bounds3, support: Bounds3, plane: SlicePlane, width: number, height: number, input?: Partial<SliceFraming>): MprView {
  const framing=sliceFraming(input), bounds=framing.fit==='image'?full:support;
  const [a,b]=PLANE_AXES[plane], mmWidth=Math.max(1e-3,bounds[1][a]-bounds[0][a]), mmHeight=Math.max(1e-3,bounds[1][b]-bounds[0][b]);
  const availableWidth=Math.max(1,width-16),availableHeight=Math.max(1,height-26);
  const scale=Math.min(availableWidth/mmWidth,availableHeight/mmHeight)*framing.zoom;
  const w=mmWidth*scale,h=mmHeight*scale;
  return {bounds,rect:{left:(width-w)/2,top:8+(availableHeight-h)/2,width:w,height:h},...framing};
}

export function mprPoint(view: MprView, plane: SlicePlane, cursor: Point3, x: number, y: number, inverse: Affine, shape: Shape3): Point3|null {
  const point=slicePoint(view.bounds,plane,cursor,view.rect,x,y);
  if (!point) return null;
  const voxel=transformPoint(inverse,point);
  // The RAS bounding box of an oblique acquisition includes empty corners.
  return voxel.every((n,a)=>n>=-1e-4 && n<=shape[a]-1+1e-4) ? point : null;
}

export function mprPixel(view: MprView, plane: SlicePlane, point: Point3): [number,number] {
  const [a,b]=PLANE_AXES[plane],{bounds,rect}=view;
  return [rect.left+(point[a]-bounds[0][a])/(bounds[1][a]-bounds[0][a])*rect.width,
    rect.top+(bounds[1][b]-point[b])/(bounds[1][b]-bounds[0][b])*rect.height];
}

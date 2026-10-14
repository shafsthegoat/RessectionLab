import type {ViewerVolume} from './contracts';
import {rasAffine,transformPoint} from './coordinates.ts';
/** Camera framing only. Sampling never modifies the scalar texture, mask or
 * measurements. Include all visible segmentation vertices separately. */
export function scanFitPoints(volume:ViewerVolume,window:[number,number]):Float32Array{
 const points:number[]=[],count=volume.mri.length,stride=Math.max(1,Math.floor(count/50000));
 const threshold=Math.max(0,window[0]+.035*(window[1]-window[0]));
 const affine=rasAffine(volume.affine,volume.frame),[nx,ny,nz]=volume.shape;
 for(let i=0;i<count;i+=stride){if(!Number.isFinite(volume.mri[i])||volume.mri[i]<=threshold)continue;
  const x=Math.floor(i/(ny*nz)),y=Math.floor(i/nz)%ny,z=i%nz;
  points.push(...transformPoint(affine,[x,y,z]));
 }
 return new Float32Array(points);
}

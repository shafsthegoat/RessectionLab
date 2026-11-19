import type {OperativeField,SourceOverview} from '../operative-field';
import type {Point3} from './coordinates';

/** Nearest field-label membership on the same acquired oblique plane as the
 * grayscale inset. The outline is limited to the sampled local field. */
export function fieldTargetOnOverview(field:OperativeField,view:SourceOverview){
  const [height,width]=view.shape,result=new Uint8Array(height*width);
  const project=(direction:Point3)=>field.basisRas.map(row=>row.reduce((sum,n,i)=>sum+n*direction[i],0));
  const right=project(view.rightRas),up=project(view.upRas);
  for(let y=0;y<height;y++)for(let x=0;x<width;x++){
    const dx=(x-view.targetPx[0])*view.spacingMm,dy=(view.targetPx[1]-y)*view.spacingMm;
    const sample=right.map((r,i)=>Math.round((dx*r+dy*up[i]-field.origin[i])/field.spacingMm));
    if(sample.some((n,i)=>n<0||n>=field.shape[i]))continue;
    const [i,j,k]=sample;
    result[y*width+x]=Number((field.voxels[(i*field.shape[1]+j)*field.shape[2]+k]&field.materialBits.tumor)!==0);
  }
  return result;
}

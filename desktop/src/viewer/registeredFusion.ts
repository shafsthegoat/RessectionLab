import { inverseAffine, rasAffine } from './coordinates.ts';
import type { Affine } from './coordinates';
import type { ViewerVolume } from './contracts';

export interface ViewerRegisteredFusion {
  seriesId: string;
  referenceCaseHash: string;
  label: string;
  registration: { status: 'source-provided'|'verified'; overlayPermitted: true; sourceToReferenceRasMm: Affine };
  volume: ViewerVolume;
}
/** Exact physical transform; equal dimensions alone never admit fusion. */
export function checkedFusionTransform(reference: ViewerVolume, fusion: ViewerRegisteredFusion): Affine {
  if(!fusion || fusion.referenceCaseHash!==reference.caseHash ||
    !['source-provided','verified'].includes(fusion.registration?.status) || fusion.registration?.overlayPermitted!==true ||
    typeof fusion.seriesId!=='string' || !fusion.seriesId || typeof fusion.label!=='string' || !fusion.label)
    throw new Error('Image fusion requires same-case source registration with an explicit physical transform.');
  const transform=fusion.registration.sourceToReferenceRasMm;
  if(!Array.isArray(transform)||transform.length!==4||transform.some(row=>!Array.isArray(row)||row.length!==4||row.some(n=>!Number.isFinite(n)))||
    transform[3].some((n,i)=>Math.abs(n-(i===3?1:0))>1e-8))
    throw new Error('The fusion registration transform is invalid.');
  const count=fusion.volume.shape.reduce((a,b)=>a*b,1);
  if(fusion.volume.shape.some(n=>!Number.isInteger(n)||n<2)||count!==fusion.volume.mri.length)
    throw new Error('The fusion source image does not match its declared grid.');
  const referenceToSource=inverseAffine(transform);
  const sourceToVoxel=inverseAffine(rasAffine(fusion.volume.affine,fusion.volume.frame));
  return sourceToVoxel.map(row=>referenceToSource[0].map((_,j)=>row.reduce((sum,n,k)=>sum+n*referenceToSource[k][j],0)));
}

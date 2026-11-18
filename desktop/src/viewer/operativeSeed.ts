import type { OperativeField } from '../operative-field';

type SeedField = Pick<OperativeField,'origin'|'shape'|'spacingMm'|'stations'|'voxels'>;
export const eligibleTumor = (bits:number) => !!(bits&8) && !(bits&(1|2|16));

/** The route-local target endpoint is (0,0,0). The backend returns the
 * uppermost sampled center in its continuous target run up to the entry.
 * A protected or unsupported centerline makes that sweep unavailable. */
export function aspirationSeedIndex(field:SeedField):number|null {
  const station=field.stations.tumor;
  if(station===null)return null;
  const grid=(mm:number,axis:number)=>Math.round((mm-field.origin[axis])/field.spacingMm);
  const x=grid(0,0), y=grid(0,1), endpoint=grid(0,2), entry=grid(station,2);
  const [nx,ny,nz]=field.shape;
  if(x<0||x>=nx||y<0||y>=ny||endpoint<0||entry<endpoint||entry>=nz)return null;
  const centerline=(x*ny+y)*nz;
  for(let z=endpoint;z<=entry;z++)if(!eligibleTumor(field.voxels[centerline+z]))return null;
  return centerline+entry;
}

export function hasAspirationSeed(field:SeedField):boolean {
  return aspirationSeedIndex(field)!==null;
}

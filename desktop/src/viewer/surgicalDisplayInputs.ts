import type {SurgicalView} from '../surgical-types';

export function sameArray<T>(a:readonly T[],b:readonly T[],equals:(left:T,right:T)=>boolean=Object.is):boolean {
  return a===b||(a.length===b.length&&a.every((entry,index)=>equals(entry,b[index])));
}

/** Preserve enumeration order for components whose typed buffers are filled in
 * wire-record order. A fresh response need not retain object references. */
export function sameFieldEntries<T>(a:Record<string,T>|undefined,b:Record<string,T>|undefined,equals:(left:T,right:T)=>boolean=Object.is):boolean {
  if(a===b)return true;
  const left=Object.keys(a??{}),right=Object.keys(b??{});
  return sameArray(left,right)&&left.every(key=>equals(a![key],b![key]));
}

export function sameSurgicalGrid(a:SurgicalView['scene'],b:SurgicalView['scene']):boolean {
  return a===b||(a.sceneId===b.sceneId&&sameArray(a.shape,b.shape)&&sameArray(a.affine,b.affine,(p,q)=>sameArray(p,q)));
}

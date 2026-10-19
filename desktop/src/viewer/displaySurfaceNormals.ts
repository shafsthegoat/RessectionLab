import {inverseAffine} from './coordinates.ts';
import type {Affine} from './coordinates';

/** Lighting normals only: bounded diffusion over existing triangle edges.
 * Work in physical space so anisotropic or mirrored acquisitions receive the
 * same response. No vertex, triangle, source normal or mask is changed, and
 * disconnected (even coincident) vertices never exchange a normal. */
export function displaySurfaceNormals(normals:Float32Array,indices:Uint32Array,affine:Affine,passes=3):Float32Array {
  const n=normals.length/3,inverse=inverseAffine(affine),original=new Float64Array(normals.length);
  for(let i=0;i<normals.length;i+=3){
    for(let a=0;a<3;a++)original[i+a]=inverse[0][a]*normals[i]+inverse[1][a]*normals[i+1]+inverse[2][a]*normals[i+2];
    const length=Math.hypot(original[i],original[i+1],original[i+2]);
    if(length)for(let a=0;a<3;a++)original[i+a]/=length;
  }
  let current=original.slice(),next=new Float64Array(normals.length);
  const weight=new Float64Array(n),minimumCosine=Math.cos(Math.PI/6);
  for(let pass=0;pass<passes;pass++){
    next.fill(0);weight.fill(0);
    // Every contribution follows a supplied face edge. An angular bilateral
    // weight keeps opposing walls and narrow branch tips distinct.
    for(let f=0;f<indices.length;f+=3)for(let edge=0;edge<3;edge++){
      const a=indices[f+edge],b=indices[f+(edge+1)%3],i=a*3,j=b*3;
      const cosine=current[i]*current[j]+current[i+1]*current[j+1]+current[i+2]*current[j+2];
      if(cosine<=.15)continue;
      const w=cosine*cosine;
      for(let k=0;k<3;k++){next[i+k]+=current[j+k]*w;next[j+k]+=current[i+k]*w;}
      weight[a]+=w;weight[b]+=w;
    }
    for(let v=0;v<n;v++){
      const i=v*3;
      for(let k=0;k<3;k++)next[i+k]=weight[v] ? .45*current[i+k]+.55*next[i+k]/weight[v] : original[i+k];
      let length=Math.hypot(next[i],next[i+1],next[i+2]);
      if(length)for(let k=0;k<3;k++)next[i+k]/=length;
      const cosine=next[i]*original[i]+next[i+1]*original[i+1]+next[i+2]*original[i+2];
      if(cosine<minimumCosine){
        const tangent=[0,1,2].map(k=>next[i+k]-cosine*original[i+k]);
        length=Math.hypot(...tangent);
        for(let k=0;k<3;k++)next[i+k]=minimumCosine*original[i+k]+(length ? .5*tangent[k]/length : 0);
      }
    }
    [current,next]=[next,current];
  }
  const output=new Float32Array(normals.length);
  for(let i=0;i<normals.length;i+=3){
    for(let a=0;a<3;a++)output[i+a]=affine[0][a]*current[i]+affine[1][a]*current[i+1]+affine[2][a]*current[i+2];
    const length=Math.hypot(output[i],output[i+1],output[i+2]);
    if(length)for(let a=0;a<3;a++)output[i+a]/=length;
  }
  return output;
}

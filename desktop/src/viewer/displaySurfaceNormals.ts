import {inverseAffine} from './coordinates.ts';
import type {Affine} from './coordinates';
import {MathUtils,Vector3} from 'three';

/** Lighting normals only: bounded diffusion over existing triangle edges.
 * Work in physical space so anisotropic or mirrored acquisitions receive the
 * same response. No vertex, triangle, source normal or mask is changed, and
 * disconnected (even coincident) vertices never exchange a normal. */
export function displaySurfaceNormals(normals:Float32Array,indices:Uint32Array,affine:Affine,passes=3):Float32Array {
  const n=normals.length/3,inverse=inverseAffine(affine),original=new Float32Array(normals.length);
  for(let i=0;i<normals.length;i+=3){
    for(let a=0;a<3;a++)original[i+a]=inverse[0][a]*normals[i]+inverse[1][a]*normals[i+1]+inverse[2][a]*normals[i+2];
    const length=Math.hypot(original[i],original[i+1],original[i+2]);
    if(length)for(let a=0;a<3;a++)original[i+a]/=length;
  }
  let current=original.slice(),next=new Float32Array(normals.length);
  const weight=new Float32Array(n),minimumCosine=Math.cos(Math.PI/6);
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
        const x=next[i]-cosine*original[i],y=next[i+1]-cosine*original[i+1],z=next[i+2]-cosine*original[i+2];
        length=Math.hypot(x,y,z);
        const scale=length ? .5/length : 0;
        next[i]=minimumCosine*original[i]+scale*x;
        next[i+1]=minimumCosine*original[i+1]+scale*y;
        next[i+2]=minimumCosine*original[i+2]+scale*z;
      }
    }
    [current,next]=[next,current];
  }
  const output=next;
  for(let i=0;i<normals.length;i+=3){
    for(let a=0;a<3;a++)output[i+a]=affine[0][a]*current[i]+affine[1][a]*current[i+1]+affine[2][a]*current[i+2];
    const length=Math.hypot(output[i],output[i+1],output[i+2]);
    if(length)for(let a=0;a<3;a++)output[i+a]/=length;
  }
  return output;
}

function transformNormal(target:Vector3,values:Float32Array,i:number,affine:Affine){
  const x=values[i],y=values[i+1],z=values[i+2];
  return target.set(affine[0][0]*x+affine[1][0]*y+affine[2][0]*z,
    affine[0][1]*x+affine[1][1]*y+affine[2][1]*z,
    affine[0][2]*x+affine[1][2]*y+affine[2][2]*z).normalize();
}

function writeNativeNormal(normal:Vector3,physical:Vector3,tangent:Vector3,out:Float32Array,i:number,affine:Affine){
  const cosine=normal.dot(physical),minimum=Math.sqrt(3)/2;
  if(cosine<minimum){
    tangent.copy(normal).addScaledVector(physical,-cosine).normalize();
    normal.copy(physical).multiplyScalar(minimum).addScaledVector(tangent,.5);
  }
  const {x,y,z}=normal;
  tangent.set(affine[0][0]*x+affine[1][0]*y+affine[2][0]*z,
    affine[0][1]*x+affine[1][1]*y+affine[2][1]*z,
    affine[0][2]*x+affine[1][2]*y+affine[2][2]*z).normalize();
  out[i]=tangent.x;out[i+1]=tangent.y;out[i+2]=tangent.z;
}

/** Blend the original score derivative into the unchanged indexed wall normals. */
export function scoreWallLighting(physical:Float32Array,lighting:Float32Array,scores:Float32Array,affine:Affine){
  const inverse=inverseAffine(affine),output=lighting.slice();
  const source=new Vector3(),baseline=new Vector3(),score=new Vector3(),tangent=new Vector3();
  const supported=new Uint8Array(physical.length/3);
  let changed=false;
  for(let i=0,j=0,v=0;i<physical.length;i+=3,j+=4,v++){
    if(!scores[j+3])continue;
    transformNormal(source,physical,i,inverse);
    if(!source.lengthSq())continue;
    transformNormal(baseline,lighting,i,inverse);
    transformNormal(score,scores,j,inverse);
    const confidence=scores[j+3]*MathUtils.smoothstep(score.dot(baseline),.55,.85);
    if(!confidence)continue;
    score.lerp(baseline,1-confidence).normalize();
    writeNativeNormal(score,source,tangent,output,i,affine);
    supported[v]=1;
    changed=true;
  }
  return changed?{normals:output,supported}:undefined;
}

/** Transport the accepted lighting field in the moved physical normal frame.
 * Large local turns use the ordinary lighting of the accepted surface. */
export function advectScoreWallLighting(restPhysical:Float32Array,restLighting:Float32Array,
  currentPhysical:Float32Array,currentLighting:Float32Array,supported:Uint8Array,affine:Affine){
  const inverse=inverseAffine(affine),output=currentLighting.slice();
  const before=new Vector3(),after=new Vector3(),lighting=new Vector3(),baseline=new Vector3(),
    cross=new Vector3(),sideways=new Vector3(),tangent=new Vector3();
  for(let i=0,v=0;i<output.length;i+=3,v++){
    if(!supported[v])continue;
    if(restPhysical[i]===currentPhysical[i]&&restPhysical[i+1]===currentPhysical[i+1]&&restPhysical[i+2]===currentPhysical[i+2]){
      output[i]=restLighting[i];output[i+1]=restLighting[i+1];output[i+2]=restLighting[i+2];
      continue;
    }
    transformNormal(before,restPhysical,i,inverse);
    transformNormal(after,currentPhysical,i,inverse);
    const agreement=before.dot(after);
    if(agreement<=.65)continue;
    transformNormal(lighting,restLighting,i,inverse);
    cross.crossVectors(before,after);
    sideways.crossVectors(cross,lighting);
    tangent.crossVectors(cross,sideways);
    lighting.add(sideways).addScaledVector(tangent,1/(1+agreement));
    transformNormal(baseline,currentLighting,i,inverse);
    lighting.lerp(baseline,1-MathUtils.smoothstep(agreement,.65,.92)).normalize();
    writeNativeNormal(lighting,after,tangent,output,i,affine);
  }
  return output;
}

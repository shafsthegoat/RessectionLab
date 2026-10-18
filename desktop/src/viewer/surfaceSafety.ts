/** BVH audit of nonadjacent triangles in physical millimetres. Existing source
 * contacts are retained. The fairing stage separately checks adjacent faces. */
export function trianglesIntersect(p:Float64Array,q:Float64Array,ids:Uint32Array,a:number,b:number):boolean {
  const ai=[ids[a]*3,ids[a+1]*3,ids[a+2]*3],bi=[ids[b]*3,ids[b+1]*3,ids[b+2]*3];
  const edge=(v:Float64Array,i:number,j:number)=>[v[j]-v[i],v[j+1]-v[i+1],v[j+2]-v[i+2]];
  const cross=(u:number[],v:number[])=>[u[1]*v[2]-u[2]*v[1],u[2]*v[0]-u[0]*v[2],u[0]*v[1]-u[1]*v[0]];
  const ae=ai.map((i,j)=>edge(p,i,ai[(j+1)%3])),be=bi.map((i,j)=>edge(q,i,bi[(j+1)%3]));
  const an=cross(ae[0],ae[1]),bn=cross(be[0],be[1]);
  const separated=(axis:number[])=>{
    const length=Math.hypot(...axis);if(length<1e-15)return false;
    const dot=(v:Float64Array,i:number)=>axis.reduce((s,n,k)=>s+n*(v[i+k]-p[ai[0]+k]),0);
    const av=ai.map(i=>dot(p,i)),bv=bi.map(i=>dot(q,i)),eps=1e-9*length;
    return Math.max(...av)<Math.min(...bv)-eps||Math.max(...bv)<Math.min(...av)-eps;
  };
  if(separated(an)||separated(bn))return false;
  for(const e of ae)for(const f of be)if(separated(cross(e,f)))return false;
  for(const e of ae)if(separated(cross(an,e)))return false;
  for(const e of be)if(separated(cross(bn,e)))return false;
  return true;
}
export function intersectionVertices(source:Float64Array,candidate:Float64Array,indices:Uint32Array):Set<number> {
  const count=indices.length/3,boxes=new Float64Array(count*6),order=Uint32Array.from({length:count},(_,i)=>i);
  for(let f=0;f<count;f++)for(let a=0;a<3;a++){
    const v=[0,1,2].map(c=>candidate[indices[f*3+c]*3+a]);boxes[f*6+a]=Math.min(...v);boxes[f*6+a+3]=Math.max(...v);
  }
  interface Node {box:number[];start:number;end:number;left?:Node;right?:Node}
  const build=(start:number,end:number):Node=>{
    const box=[Infinity,Infinity,Infinity,-Infinity,-Infinity,-Infinity];
    for(let k=start;k<end;k++)for(let a=0;a<3;a++){box[a]=Math.min(box[a],boxes[order[k]*6+a]);box[a+3]=Math.max(box[a+3],boxes[order[k]*6+a+3]);}
    const node:Node={box,start,end};if(end-start<=12)return node;
    let axis=0;for(let a=1;a<3;a++)if(box[a+3]-box[a]>box[axis+3]-box[axis])axis=a;
    const mid=(box[axis]+box[axis+3])*.5;let l=start,r=end-1;
    while(l<=r){const f=order[l];if((boxes[f*6+axis]+boxes[f*6+axis+3])*.5<mid)l++;else{order[l]=order[r];order[r--]=f;}}
    const split=l===start||l===end?(start+end)>>>1:l;
    node.left=build(start,split);node.right=build(split,end);return node;
  };
  const bad=new Set<number>();if(!count)return bad;const root=build(0,count);
  const overlap=(f:number,box:ArrayLike<number>)=>[0,1,2].every(a=>boxes[f*6+a]<=box[a+3]+1e-9&&boxes[f*6+a+3]>=box[a]-1e-9);
  for(let f=0;f<count;f++){
    const stack=[root];while(stack.length){const n=stack.pop()!;if(!overlap(f,n.box))continue;
      if(n.left){stack.push(n.left,n.right!);continue;}
      for(let k=n.start;k<n.end;k++){
        const g=order[k];if(g<=f||!overlap(f,boxes.subarray(g*6,g*6+6)))continue;
        const a=indices.subarray(f*3,f*3+3),b=indices.subarray(g*3,g*3+3);if(a.some(v=>b.includes(v)))continue;
        if(trianglesIntersect(candidate,candidate,indices,f*3,g*3)&&!trianglesIntersect(source,source,indices,f*3,g*3))for(const v of [...a,...b])bad.add(v);
      }
    }
  }
  return bad;
}

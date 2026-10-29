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
interface Node {box:number[];start:number;end:number;left?:Node;right?:Node}
interface SurfaceTree {boxes:Float64Array;order:Uint32Array;root?:Node;nodeCount:number}
interface CompactTree {bounds:Float64Array;children:Int32Array;ranges:Uint32Array;order:Uint32Array}
function candidateTree(candidate:Float64Array,indices:Uint32Array,faces?:Uint32Array):SurfaceTree {
  const count=indices.length/3,boxes=new Float64Array(count*6),order=faces??Uint32Array.from({length:count},(_,i)=>i);
  for(const f of order)for(let a=0;a<3;a++){
    const x=candidate[indices[f*3]*3+a],y=candidate[indices[f*3+1]*3+a],z=candidate[indices[f*3+2]*3+a];
    boxes[f*6+a]=Math.min(x,y,z);boxes[f*6+a+3]=Math.max(x,y,z);
  }
  let nodeCount=0;
  const build=(start:number,end:number):Node=>{
    const box=[Infinity,Infinity,Infinity,-Infinity,-Infinity,-Infinity];
    nodeCount++;
    for(let k=start;k<end;k++){
      for(let a=0;a<3;a++){box[a]=Math.min(box[a],boxes[order[k]*6+a]);box[a+3]=Math.max(box[a+3],boxes[order[k]*6+a+3]);}
    }
    const node:Node={box,start,end};if(end-start<=12)return node;
    let axis=0;for(let a=1;a<3;a++)if(box[a+3]-box[a]>box[axis+3]-box[axis])axis=a;
    const mid=(box[axis]+box[axis+3])*.5;let l=start,r=end-1;
    while(l<=r){const f=order[l];if((boxes[f*6+axis]+boxes[f*6+axis+3])*.5<mid)l++;else{order[l]=order[r];order[r--]=f;}}
    const split=l===start||l===end?(start+end)>>>1:l;
    node.left=build(start,split);node.right=build(split,end);return node;
  };
  const root=order.length?build(0,order.length):undefined;
  return {boxes,order,root,nodeCount};
}
function intersectionAudit(source:Float64Array,candidate:Float64Array,indices:Uint32Array,capture?:(tree:SurfaceTree)=>void):Set<number> {
  const count=indices.length/3,{boxes,order,root,nodeCount}=candidateTree(candidate,indices);
  const bad=new Set<number>();
  if(!root){capture?.({boxes,order,root,nodeCount});return bad;}
  // Keep the exact axis order/epsilon without allocating arrays, callbacks or
  // typed-array views for each of the millions of BVH overlap queries.
  const overlap=(f:number,box:ArrayLike<number>,offset=0)=>
    boxes[f*6]<=box[offset+3]+1e-9&&boxes[f*6+3]>=box[offset]-1e-9&&
    boxes[f*6+1]<=box[offset+4]+1e-9&&boxes[f*6+4]>=box[offset+1]-1e-9&&
    boxes[f*6+2]<=box[offset+5]+1e-9&&boxes[f*6+5]>=box[offset+2]-1e-9;
  const stack:Node[]=[];
  for(let f=0;f<count;f++){
    stack.push(root);while(stack.length){const n=stack.pop()!;if(!overlap(f,n.box))continue;
      if(n.left){stack.push(n.left,n.right!);continue;}
      for(let k=n.start;k<n.end;k++){
        const g=order[k];if(g<=f||!overlap(f,boxes,g*6))continue;
        const a=indices[f*3],b=indices[f*3+1],c=indices[f*3+2],d=indices[g*3],e=indices[g*3+1],h=indices[g*3+2];
        if(a===d||a===e||a===h||b===d||b===e||b===h||c===d||c===e||c===h)continue;
        if(trianglesIntersect(candidate,candidate,indices,f*3,g*3)&&!trianglesIntersect(source,source,indices,f*3,g*3)){
          bad.add(a);bad.add(b);bad.add(c);bad.add(d);bad.add(e);bad.add(h);
        }
      }
    }
  }
  capture?.({boxes,order,root,nodeCount});
  return bad;
}

function compactTree(tree:SurfaceTree):CompactTree {
  const bounds=new Float64Array(tree.nodeCount*6),children=new Int32Array(tree.nodeCount*2).fill(-1),ranges=new Uint32Array(tree.nodeCount*2);
  let next=0;
  const visit=(node:Node):number=>{
    const id=next++;bounds.set(node.box,id*6);ranges[id*2]=node.start;ranges[id*2+1]=node.end;
    if(node.left){children[id*2]=visit(node.left);children[id*2+1]=visit(node.right!);}return id;
  };
  if(tree.root)visit(tree.root);
  return {bounds,children,ranges,order:tree.order};
}

/** Only dirty/current pairs and dirty/certified-clean pairs need testing.
 * SAT always receives current canonical min/max face IDs. Contact vertex
 * resets commute; this private path need not reproduce Set insertion order. */
function incrementalAudit(source:Float64Array,candidate:Float64Array,indices:Uint32Array,dirty:Uint32Array,cleanForOld:Int32Array,prior:CompactTree):Set<number> {
  const bad=new Set<number>(),{boxes,order,root}=candidateTree(candidate,indices,dirty.slice());
  const overlap=(f:number,box:ArrayLike<number>,at=0)=>boxes[f*6]<=box[at+3]+1e-9&&boxes[f*6+3]>=box[at]-1e-9&&
    boxes[f*6+1]<=box[at+4]+1e-9&&boxes[f*6+4]>=box[at+1]-1e-9&&boxes[f*6+2]<=box[at+5]+1e-9&&boxes[f*6+5]>=box[at+2]-1e-9;
  const pair=(f:number,g:number)=>{
    if(f>g)[f,g]=[g,f];
    const a=indices[f*3],b=indices[f*3+1],c=indices[f*3+2],d=indices[g*3],e=indices[g*3+1],h=indices[g*3+2];
    if(a===d||a===e||a===h||b===d||b===e||b===h||c===d||c===e||c===h)return;
    if(trianglesIntersect(candidate,candidate,indices,f*3,g*3)&&!trianglesIntersect(source,source,indices,f*3,g*3)){
      bad.add(a);bad.add(b);bad.add(c);bad.add(d);bad.add(e);bad.add(h);
    }
  };
  const stack:Node[]=[],oldStack:number[]=[],cleanBox=new Float64Array(6);
  for(const f of dirty){
    if(root)stack.push(root);
    while(stack.length){const node=stack.pop()!;if(!overlap(f,node.box))continue;
      if(node.left){stack.push(node.left,node.right!);continue;}
      for(let k=node.start;k<node.end;k++){const g=order[k];if(g>f&&overlap(f,boxes,g*6))pair(f,g);}
    }
    if(prior.bounds.length)oldStack.push(0);
    while(oldStack.length){const id=oldStack.pop()!;if(!overlap(f,prior.bounds,id*6))continue;
      if(prior.children[id*2]>=0){oldStack.push(prior.children[id*2],prior.children[id*2+1]);continue;}
      for(let k=prior.ranges[id*2];k<prior.ranges[id*2+1];k++){
        const g=cleanForOld[prior.order[k]];if(g<0)continue;
        // Preserve the full reference's per-pair AABB rejection as well as
        // SAT's epsilon; node overlap alone is deliberately conservative.
        for(let a=0;a<3;a++){
          const x=candidate[indices[g*3]*3+a],y=candidate[indices[g*3+1]*3+a],z=candidate[indices[g*3+2]*3+a];
          cleanBox[a]=Math.min(x,y,z);cleanBox[a+3]=Math.max(x,y,z);
        }
        if(overlap(f,cleanBox))pair(f,g);
      }
    }
  }
  return bad;
}

/** Full reference audit. No caller-supplied dirty mask can omit its checks. */
export function intersectionVertices(source:Float64Array,candidate:Float64Array,indices:Uint32Array):Set<number> {
  return intersectionAudit(source,candidate,indices);
}

export interface SourceAuditLayout {
  vertexIds:Uint32Array;
  vertexCapacity:number;
  rows:Array<{buffer:ArrayBufferLike;offset:number;begin:number;end:number}>;
}
export interface SurfaceIntersectionAudit {
  run(source:Float64Array,candidate:Float64Array,indices:Uint32Array):Set<number>;
  complete(source:Float64Array,candidate:Float64Array,indices:Uint32Array,contacts:Set<number>):void;
}
interface Certificate {
  source:Float64Array;physical:Float64Array;active:Uint8Array;triangles:Uint32Array;
  ranks:WeakMap<ArrayBufferLike,Uint32Array>;tree:CompactTree;bytes:number;
}

/** A certificate belongs to one worker-owned stable source registry. It is
 * published only after an explicitly empty final audit. Detached source and
 * physical coordinates, ordered triangle IDs and weak chunk-rank bindings
 * prove clean/clean pairs identical to that completed audit. */
export class IncrementalSurfaceIntersectionAudit {
  private certificate?:Certificate;
  private work={fullAudits:0,partialAudits:0,dirtyFaces:0,totalFaces:0};
  get retainedBytes(){return this.certificate?.bytes??0;}
  get statistics(){return {...this.work,certificateBytes:this.retainedBytes};}
  begin(layout:SourceAuditLayout,indices:Uint32Array,maxBytes:number):SurfaceIntersectionAudit {
    this.work={fullAudits:0,partialAudits:0,dirtyFaces:0,totalFaces:indices.length/3};
    if(this.retainedBytes>maxBytes)this.certificate=undefined;
    const prior=this.certificate,n=layout.vertexIds.length,count=indices.length/3;
    let valid=Number.isSafeInteger(layout.vertexCapacity)&&layout.vertexCapacity>=n&&
      Number.isSafeInteger(maxBytes)&&maxBytes>=0&&indices.length%3===0;
    const buffers=new Set(layout.rows.map(row=>row.buffer));
    const bytes=49*layout.vertexCapacity+indices.byteLength+
      [...buffers].reduce((sum,buffer)=>sum+buffer.byteLength/3,0)+buffers.size*256;
    // Unknown/oversized provenance takes the full reference path and retains
    // no new certificate. The existing geometry result is never approximated.
    valid=valid&&Number.isSafeInteger(bytes)&&bytes<=maxBytes;
    const known=new Uint32Array(valid&&prior?count:0);
    if(valid){
      const seen=new Uint8Array(layout.vertexCapacity);
      for(const id of layout.vertexIds){if(id>=seen.length||seen[id]){valid=false;break;}seen[id]=1;}
      let end=0,lastRank=0;
      for(const row of layout.rows){
        if(!Number.isSafeInteger(row.offset)||!Number.isSafeInteger(row.begin)||!Number.isSafeInteger(row.end)||
          row.begin!==end||row.end<row.begin||row.end>count||row.offset<0||row.buffer.byteLength%12||
          row.offset+row.end-row.begin>row.buffer.byteLength/12){valid=false;break;}
        end=row.end;
        const ranks=prior?.ranks.get(row.buffer);
        if(!ranks)continue;
        for(let f=row.begin;f<row.end;f++){
          const rank:number=ranks[row.offset+f-row.begin];if(!rank)continue;
          // SAT is evaluated with the smaller canonical face first. Preserve
          // the relative order as well as each ordered corner triplet.
          if(rank<=lastRank){valid=false;break;}lastRank=rank;
          const old=(rank-1)*3;
          if(prior!.triangles[old]===layout.vertexIds[indices[f*3]]&&
            prior!.triangles[old+1]===layout.vertexIds[indices[f*3+1]]&&
            prior!.triangles[old+2]===layout.vertexIds[indices[f*3+2]])known[f]=rank;
        }
        if(!valid)break;
      }
      valid=valid&&end===count;
    }
    let lastSource:Float64Array|undefined,lastCandidate:Float64Array|undefined,lastIndices:Uint32Array|undefined,lastContacts:Set<number>|undefined,lastTree:SurfaceTree|undefined;
    return {
      run:(source,candidate,currentIndices)=>{
        lastTree=undefined;
        if(valid&&prior&&currentIndices===indices&&source.length===n*3&&candidate.length===n*3&&
          source.every(Number.isFinite)&&candidate.every(Number.isFinite)){
          const unchanged=new Uint8Array(n);
          for(let v=0;v<n;v++){
            const id=layout.vertexIds[v];if(!prior.active[id])continue;
            let same=true;
            for(let a=0;a<3;a++)if(!Object.is(source[v*3+a],prior.source[id*3+a])||
              !Object.is(candidate[v*3+a],prior.physical[id*3+a])){same=false;break;}
            if(same)unchanged[v]=1;
          }
          const dirty=new Uint32Array(count),cleanForOld=new Int32Array(prior.triangles.length/3).fill(-1);let changed=0;
          for(let f=0;f<count;f++){
            if(!known[f]||!unchanged[indices[f*3]]||!unchanged[indices[f*3+1]]||!unchanged[indices[f*3+2]])dirty[changed++]=f;
            else cleanForOld[known[f]-1]=f;
          }
          if(changed<=count*.35){
            this.work.partialAudits++;this.work.dirtyFaces+=changed;
            lastSource=source;lastCandidate=candidate;lastIndices=currentIndices;
            return lastContacts=changed?incrementalAudit(source,candidate,currentIndices,dirty.subarray(0,changed),cleanForOld,prior.tree):new Set();
          }
        }
        this.work.fullAudits++;
        lastSource=source;lastCandidate=candidate;lastIndices=currentIndices;
        return lastContacts=intersectionAudit(source,candidate,currentIndices,tree=>{lastTree=tree;});
      },
      complete:(source,candidate,currentIndices,contacts)=>{
        if(!valid||currentIndices!==indices||lastSource!==source||lastCandidate!==candidate||lastIndices!==indices||
          contacts!==lastContacts||contacts.size||source.length!==n*3||candidate.length!==n*3||
          !source.every(Number.isFinite)||!candidate.every(Number.isFinite)||!lastTree)return;
        const totalBytes=bytes+lastTree.nodeCount*64+lastTree.order.byteLength;
        if(totalBytes>maxBytes)return;
        const next:Certificate={source:new Float64Array(layout.vertexCapacity*3),physical:new Float64Array(layout.vertexCapacity*3),
          active:new Uint8Array(layout.vertexCapacity),triangles:new Uint32Array(indices.length),ranks:new WeakMap(),tree:compactTree(lastTree),bytes:totalBytes};
        for(let v=0;v<n;v++){
          const id=layout.vertexIds[v];next.active[id]=1;
          for(let a=0;a<3;a++){next.source[id*3+a]=source[v*3+a];next.physical[id*3+a]=candidate[v*3+a];}
        }
        for(let i=0;i<indices.length;i++)next.triangles[i]=layout.vertexIds[indices[i]];
        for(const row of layout.rows){
          let ranks=next.ranks.get(row.buffer);
          if(!ranks){ranks=new Uint32Array(row.buffer.byteLength/12);next.ranks.set(row.buffer,ranks);}
          for(let f=row.begin;f<row.end;f++)ranks[row.offset+f-row.begin]=f+1;
        }
        this.certificate=next;
      },
    };
  }
}

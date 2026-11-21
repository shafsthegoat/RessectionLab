import type {Affine} from './coordinates';

export interface CavityRail {inside:number[];outside:number[];anchor:number[];free?:boolean}

/** Fit the local patch to its face neighbors along each material rail.
 * A rail lies in the original extraction simplex; its endpoints are in
 * opposite source half-spaces. Isolated material may also relax tangentially
 * inside the physical error ball. Pin thin or protected interface samples. */
export function fairCavityPatch(points:Float32Array,indices:Uint32Array,rails:Map<number,CavityRail>,affine:Affine,limit:number) {
  if(!rails.size)return 0;
  const links=new Map<number,Set<number>>();
  for(const id of rails.keys())links.set(id,new Set());
  for(let i=0;i<indices.length;i+=3){
    const a=indices[i],b=indices[i+1],c=indices[i+2];
    links.get(a)?.add(b).add(c);links.get(b)?.add(c).add(a);links.get(c)?.add(a).add(b);
  }
  const worldMatrix=affine.slice(0,3);
  const world=(v:number[])=>worldMatrix.map(row=>row[0]*v[0]+row[1]*v[1]+row[2]*v[2]);
  const rows=[...rails].map(([id,r])=>{
    const d=r.inside.map((v,k)=>v-r.outside[k]);
    const dx=world(d),offset=world(r.outside.map((v,k)=>v-r.anchor[k]));
    const dot=(a:number[],b:number[])=>a.reduce((s,v,k)=>s+v*b[k],0);
    const aa=dot(dx,dx),ab=dot(dx,offset),cc=dot(offset,offset)-(.995*limit)**2;
    const disc=Math.sqrt(Math.max(0,ab*ab-aa*cc));
    const lower=Math.max(.025,(-ab-disc)/aa),upper=Math.min(.975,(-ab+disc)/aa);
    const current=Array.from(points.subarray(3*id,3*id+3));
    const axis=d.findIndex(v=>Math.abs(v)>1.e-9),initial=axis<0?0:(current[axis]-r.outside[axis])/d[axis];
    return {id,start:r.outside,anchor:r.anchor,free:!!r.free,d,lower,upper,initial,t:initial,target:current.slice(),original:current,neighbors:[...links.get(id)!]};
  }).filter(row=>row.lower<row.upper&&Number.isFinite(row.lower)&&row.neighbors.length>=3);
  // A positive inverse-edge-length stencil with a weak restoring force
  // damps the discrete wall stairs while retaining the sample support.
  for(let pass=0;pass<14;pass++){
    for(const row of rows){
      const {id,d}=row,center=[0,0,0];let sum=0;
      for(const other of row.neighbors){
        const vx=points[other*3]-points[id*3],vy=points[other*3+1]-points[id*3+1],vz=points[other*3+2]-points[id*3+2];
        const w=1/Math.max(.025,Math.hypot(...worldMatrix.map(r=>r[0]*vx+r[1]*vy+r[2]*vz)));
        center[0]+=w*vx;center[1]+=w*vy;center[2]+=w*vz;sum+=w;
      }
      if(row.free){
        const desired=center.map((c,k)=>points[id*3+k]+.85*c/sum+.03*(row.original[k]-points[id*3+k]));
        const diff=desired.map((p,k)=>p-row.anchor[k]);
        const length=Math.hypot(...worldMatrix.map(r=>r[0]*diff[0]+r[1]*diff[1]+r[2]*diff[2]));
        const scale=Math.min(1,limit*.995/Math.max(length,1.e-12));
        row.target=diff.map((v,k)=>row.anchor[k]+v*scale);
      }else{
        const ray=world(d),target=world(center);
        const length=ray[0]*ray[0]+ray[1]*ray[1]+ray[2]*ray[2];
        const step=(target[0]*ray[0]+target[1]*ray[1]+target[2]*ray[2])/(sum*length);
        row.t=Math.max(row.lower,Math.min(row.upper,row.t+.8*step+.025*(row.initial-row.t)));
        row.target=d.map((v,k)=>row.start[k]+row.t*v);
      }
    }
    for(const row of rows)for(let k=0;k<3;k++)points[3*row.id+k]=row.target[k];
  }
  let max=0;
  for(const row of rows){
    const delta=row.anchor.map((v,k)=>points[3*row.id+k]-v);
    max=Math.max(max,Math.hypot(...worldMatrix.map(r=>r[0]*delta[0]+r[1]*delta[1]+r[2]*delta[2])));
  }
  return max;
}

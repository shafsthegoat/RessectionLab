import * as THREE from 'three';
import type {SurgicalView} from '../surgical-types';
import type {Point3,Shape3} from './coordinates';

const bucketSize=8;
const zero:Point3=[0,0,0];
const coordinates=(i:number,[,ny,nz]:Shape3):Point3=>[Math.floor(i/(ny*nz)),Math.floor(i/nz)%ny,i%nz];
const index=(x:number,y:number,z:number,[,ny,nz]:Shape3)=>(x*ny+y)*nz+z;
const bucket=(x:number,y:number,z:number,shape:Shape3)=>
  (Math.floor(x/bucketSize)*(Math.ceil(shape[1]/bucketSize)+2)+Math.floor(y/bucketSize))*(Math.ceil(shape[2]/bucketSize)+2)+Math.floor(z/bucketSize);
type Cell={shift:Point3;residual:Point3;ambiguous:boolean};

/** Accepted movement is defined on source cells. The displayed mask already
 * includes rounded transport. Carry only its subvoxel remainder onto soft
 * surfaces; reverse the integer part when looking up the registered MRI. */
export class AcceptedSurfaceMotion {
  readonly cells=new Map<number,Cell>();
  readonly buckets=new Set<number>();
  readonly bounds=new THREE.Box3();
  readonly shape:Shape3;
  private hasRasterTransport=false;
  constructor(view:SurgicalView,owner:Uint16Array){
    this.shape=view.scene.shape;
    const shape=view.scene.shape, destinations=new Map(view.moved), cleared=new Set([...view.removed,...destinations.keys()]);
    const incoming=new Map<number,number[]>();
    for(const [source,to] of destinations){
      const list=incoming.get(to)??[]; list.push(source); incoming.set(to,list);
    }
    const soft=new Set(view.scene.regions.filter(r=>['brain','soft_tissue','tumor'].includes(r.material)).map(r=>r.id));
    const voxelToWorld=new THREE.Matrix4().fromArray(view.scene.affine.flat()).transpose();
    const inverse=new THREE.Matrix3().setFromMatrix4(voxelToWorld).invert();
    const vector=new THREE.Vector3();
    const targets=new Set([...incoming.keys(),...Object.keys(view.deformation??{}).map(id=>destinations.get(+id)??+id)]);
    const fixedOwner=(i:number)=>(!cleared.has(i)&&owner[i]!==0&&!soft.has(owner[i]))||
      incoming.get(i)?.some(source=>!soft.has(owner[source]));
    for(const to of targets){
      const sources=[...(incoming.get(to)??[])];
      if(!cleared.has(to)&&owner[to])sources.push(to);
      if(!sources.length)continue;
      const source=sources[0],ambiguous=sources.length!==1;
      // A uniquely occupied stationary vertex with no displacement needs no storage.
      const delta=view.deformation?.[source];
      if(!ambiguous&&source===to&&!delta?.some(Boolean))continue;
      const origin=coordinates(source,shape),p=coordinates(to,shape);
      const shift=p.map((v,j)=>v-origin[j]) as Point3;
      let residual:Point3=zero;
      if(!ambiguous&&delta&&soft.has(owner[source])){
        const d=vector.set(...delta).applyMatrix3(inverse).toArray();
        const remainder=d.map((v,j)=>v-shift[j]) as Point3;
        // Reject inconsistent source/raster pairs. The parent's authoritative
        // mask remains exact even when a model has a different transport rule.
        if(remainder.every(v=>Number.isFinite(v)&&Math.abs(v)<=.500001)){
          let fixed=false;
          for(let x=Math.max(0,p[0]-1);x<=Math.min(shape[0]-1,p[0]+1)&&!fixed;x++)
            for(let y=Math.max(0,p[1]-1);y<=Math.min(shape[1]-1,p[1]+1)&&!fixed;y++)
              for(let z=Math.max(0,p[2]-1);z<=Math.min(shape[2]-1,p[2]+1);z++)
                if(fixedOwner(index(x,y,z,shape))){fixed=true;break;}
          if(!fixed)residual=remainder;
        }
      }
      this.cells.set(to,{shift,residual,ambiguous});
      this.bounds.expandByPoint(vector.set(...p));
      // Rounded transport has a one-cell edge guard in addition to its interpolation stencil.
      const radius=ambiguous||shift.some(Boolean)?2:1;
      this.hasRasterTransport ||= radius===2;
      for(const x of [p[0]-radius,p[0]+radius])for(const y of [p[1]-radius,p[1]+radius])for(const z of [p[2]-radius,p[2]+radius])
        this.buckets.add(bucket(x,y,z,shape));
    }
    this.bounds.expandByScalar(this.hasRasterTransport?2:1);
  }
  /** C1 interpolation of accepted neighboring center remainders. At material
   * boundaries the MRI may be transported only when all contributing owners
   * have the same reverse mapping. Ambiguous triangles get no inferred signal. */
  sample(position:Point3,mask:Uint8Array,registered=true){
    const shape=this.shape,base=position.map(Math.floor);
    const t=position.map((v,j)=>v-base[j]),h=t.map(v=>v*v*(3-2*v)),dh=t.map(v=>6*v*(1-v));
    const shift:Point3=[0,0,0],gradient=new Float64Array(9);
    let mapped:Point3|undefined,ambiguous=false,affected=false;
    for(let dx=0;dx<2;dx++)for(let dy=0;dy<2;dy++)for(let dz=0;dz<2;dz++){
      const x=base[0]+dx,y=base[1]+dy,z=base[2]+dz;
      if(x<0||x>=shape[0]||y<0||y>=shape[1]||z<0||z>=shape[2])continue;
      const wx=dx?h[0]:1-h[0],wy=dy?h[1]:1-h[1],wz=dz?h[2]:1-h[2],weight=wx*wy*wz;
      if(!weight)continue;
      const id=index(x,y,z,shape),cell=this.cells.get(id);
      if(mask[id]){
        const gridShift=cell?.shift??zero;
        if(cell?.ambiguous||mapped?.some((v,j)=>v!==gridShift[j]))ambiguous=true;
        mapped=gridShift;
      }
      if(!cell)continue;
      affected=true;
      for(let j=0;j<3;j++){
        const d=cell.residual[j];shift[j]+=weight*d;
        gradient[3*j]+=d*(dx?dh[0]:-dh[0])*wy*wz;
        gradient[3*j+1]+=d*wx*(dy?dh[1]:-dh[1])*wz;
        gradient[3*j+2]+=d*wx*wy*(dz?dh[2]:-dh[2]);
      }
    }
    // An edge on a tetrahedron can join two otherwise individually valid
    // samples. Reject the one-cell rim between distinct reverse mappings so
    // fragment interpolation cannot sample across an unsupported source seam.
    if(registered&&this.hasRasterTransport&&mapped&&!ambiguous){
      for(let x=Math.max(0,base[0]-1);x<=Math.min(shape[0]-1,Math.ceil(position[0])+1)&&!ambiguous;x++)
        for(let y=Math.max(0,base[1]-1);y<=Math.min(shape[1]-1,Math.ceil(position[1])+1)&&!ambiguous;y++)
          for(let z=Math.max(0,base[2]-1);z<=Math.min(shape[2]-1,Math.ceil(position[2])+1);z++){
            const id=index(x,y,z,shape);
            if(!mask[id])continue;
            const neighbor=this.cells.get(id);
            if(neighbor?.ambiguous||mapped.some((v,j)=>v!==(neighbor?.shift[j]??0))){ambiguous=true;break;}
          }
    }
    return {shift,gradient,sourceShift:mapped??zero,ambiguous:ambiguous||(affected&&!mapped)};
  }
}

type Attribute=THREE.BufferAttribute;
type DisplayState={base:Attribute;normal?:Attribute;positions?:Attribute;normals?:Attribute;offset?:Attribute;unknown?:Attribute;
  bounds:THREE.Box3;sphere:THREE.Sphere;buckets:Map<number,number[]>;changed:number[];field:AcceptedSurfaceMotion|null};
const states=new WeakMap<THREE.BufferGeometry,DisplayState>();
const attr=(geometry:THREE.BufferGeometry,name:string)=>geometry.getAttribute(name) as Attribute|undefined;

/** Private display attributes. Registered mask, raw geometry and worker/cache
 * arrays remain immutable. Per-frame work visits the changed 8-cell buckets. */
export function applyAcceptedSurfaceMotion(geometry:THREE.BufferGeometry,field:AcceptedSurfaceMotion|null,
  mask:Uint8Array,moveGeometry:boolean,registered:boolean){
  let state=states.get(geometry);
  if(!field?.cells.size)field=null;
  if((!state&&!field)||state?.field===field)return;
  if(!state){
    const base=attr(geometry,'position');if(!base)return;
    if(!geometry.boundingBox)geometry.computeBoundingBox();
    if(!field!.bounds.intersectsBox(geometry.boundingBox!))return;
    if(!geometry.boundingSphere)geometry.computeBoundingSphere();
    const buckets=new Map<number,number[]>(),shape=field!.shape;
    for(let i=0;i<base.count;i++){
      const key=bucket(base.getX(i),base.getY(i),base.getZ(i),shape),list=buckets.get(key)??[];
      list.push(i);buckets.set(key,list);
    }
    state={base,normal:attr(geometry,'normal'),bounds:geometry.boundingBox!.clone(),sphere:geometry.boundingSphere!.clone(),buckets,changed:[],field:null};states.set(geometry,state);
  }
  const {base,normal}=state;
  for(const i of state.changed){
    state.positions?.setXYZ(i,base.getX(i),base.getY(i),base.getZ(i));
    if(normal)state.normals?.setXYZ(i,normal.getX(i),normal.getY(i),normal.getZ(i));
    state.offset?.setXYZ(i,0,0,0);state.unknown?.setX(i,0);
  }
  const old=state.changed;state.changed=[];state.field=field;
  if(field){
    const n=new THREE.Vector3(),matrix=new THREE.Matrix3();
    for(const key of field.buckets)for(const i of state.buckets.get(key)??[]){
      const p:Point3=[base.getX(i),base.getY(i),base.getZ(i)];
      const {shift,gradient:g,sourceShift,ambiguous}=field.sample(p,mask,registered);
      const deform=moveGeometry&&(shift.some(Boolean)||g.some(Boolean));
      if(!deform&&!(registered&&(ambiguous||sourceShift.some(Boolean))))continue;
      if(deform){
        if(!state.positions){state.positions=base.clone();geometry.setAttribute('position',state.positions);}
        state.positions.setXYZ(i,p[0]+shift[0],p[1]+shift[1],p[2]+shift[2]);
        if(normal){
          if(!state.normals){state.normals=normal.clone();geometry.setAttribute('normal',state.normals);}
          // Cofactors transform the cross product of tangents, including
          // near-singular gradients, without inverting the deformation.
          const a=1+g[0],b=g[1],c=g[2],d=g[3],e=1+g[4],f=g[5],h=g[6],j=g[7],k=1+g[8];
          matrix.set(e*k-f*j,f*h-d*k,d*j-e*h, c*j-b*k,a*k-c*h,b*h-a*j, b*f-c*e,c*d-a*f,a*e-b*d);
          n.fromBufferAttribute(normal,i).applyMatrix3(matrix);
          if(n.lengthSq()>1e-14){n.normalize();state.normals.setXYZ(i,n.x,n.y,n.z);}
        }
      }
      if(registered){
        if(!state.offset){
          state.offset=new THREE.BufferAttribute(new Float32Array(base.count*3),3);
          state.unknown=new THREE.BufferAttribute(new Float32Array(base.count),1);
          geometry.setAttribute('corticalRestOffset',state.offset);geometry.setAttribute('corticalUnknown',state.unknown);
        }
        state.offset.setXYZ(i,...sourceShift.map((v,j)=>-v-(deform?shift[j]:0)) as Point3);
        state.unknown!.setX(i,Number(ambiguous));
      }
      state.changed.push(i);
    }
  }
  if(old.length||state.changed.length){
    let first=base.count,last=-1;
    for(const indices of [old,state.changed])for(const i of indices){first=Math.min(first,i);last=Math.max(last,i);}
    for(const attribute of [state.positions,state.normals,state.offset,state.unknown])if(attribute){
      const size=attribute.itemSize;
      // Coalesce pending edits if another accepted boundary precedes GPU submission.
      let start=first*size,end=(last+1)*size;
      for(const range of attribute.updateRanges){start=Math.min(start,range.start);end=Math.max(end,range.start+range.count);}
      attribute.clearUpdateRanges();attribute.addUpdateRange(start,end-start);attribute.needsUpdate=true;
    }
    if(state.positions){
      geometry.boundingBox!.copy(state.bounds);geometry.boundingSphere!.copy(state.sphere);
      const p=new THREE.Vector3();
      for(const i of state.changed){
        p.fromBufferAttribute(state.positions,i);
        geometry.boundingBox!.expandByPoint(p);geometry.boundingSphere!.expandByPoint(p);
      }
    }
  }
}

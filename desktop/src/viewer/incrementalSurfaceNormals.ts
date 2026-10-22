import type {Affine} from './coordinates';
/** Worker-local source normals. Only incident vertices of replaced chunk faces
 * are invalidated. Their sums are replayed in complete raster face order;
 * subtracting old sums or grouping by chunk would change floating-point bits. */
export class IncrementalSourceNormals {
  private normals=new Float32Array(0);
  private dirty=new Uint8Array(0);
  private knownVertices=0;
  private affineKey='';
  lastDirtyVertices=0;
  lastRecomputedFaces=0;
  private readonly admit:(bytes:number)=>void;
  constructor(admit:(bytes:number)=>void){this.admit=admit;}
  get retainedBytes(){return this.normals.byteLength+this.dirty.byteLength;}
  mark(ids:Uint32Array){for(const id of ids)if(id<this.dirty.length)this.dirty[id]=1;}
  prepare(points:Float32Array,vertexCount:number,rows:Iterable<{data:Float32Array|Uint32Array}>,affine:Affine){
    if(vertexCount>this.dirty.length){
      let capacity=this.dirty.length||1024;while(capacity<vertexCount)capacity*=2;
      this.admit(capacity*13);
      const normals=new Float32Array(capacity*3),dirty=new Uint8Array(capacity);normals.set(this.normals);dirty.set(this.dirty);this.normals=normals;this.dirty=dirty;
    }
    const key=JSON.stringify(affine);
    if(key!==this.affineKey){this.dirty.fill(1,0,vertexCount);this.affineKey=key;}
    else this.dirty.fill(1,this.knownVertices,vertexCount);
    this.knownVertices=vertexCount;
    let count=0;for(let id=0;id<vertexCount;id++)count+=this.dirty[id];
    this.lastDirtyVertices=count;this.lastRecomputedFaces=0;if(!count)return;
    // Disposable scratch covers stable IDs, while sums cover dirty vertices
    // only. Neither scratch nor transferable output is retained by the cache.
    const dirtyIndex=new Uint32Array(vertexCount),sums=new Float64Array(count*3);
    let offset=0;for(let id=0;id<vertexCount;id++)if(this.dirty[id])dirtyIndex[id]=++offset;
    const [[xx,xy,xz],[yx,yy,yz],[zx,zy,zz]]=affine;
    const determinant=xx*(yy*zz-yz*zy)-xy*(yx*zz-yz*zx)+xz*(yx*zy-yy*zx),sign=Math.sign(determinant);
    for(const row of rows){const indices=row.data;
      for(let f=0;f<indices.length;f+=3){const a=indices[f],b=indices[f+1],c=indices[f+2],da=dirtyIndex[a],db=dirtyIndex[b],dc=dirtyIndex[c];
        if(!da&&!db&&!dc)continue;this.lastRecomputedFaces++;
        const i=a*3,j=b*3,k=c*3,jx=points[j]-points[i],jy=points[j+1]-points[i+1],jz=points[j+2]-points[i+2],kx=points[k]-points[i],ky=points[k+1]-points[i+1],kz=points[k+2]-points[i+2];
        const ax=((0+xx*jx)+xy*jy)+xz*jz,ay=((0+yx*jx)+yy*jy)+yz*jz,az=((0+zx*jx)+zy*jy)+zz*jz,
          bx=((0+xx*kx)+xy*ky)+xz*kz,by=((0+yx*kx)+yy*ky)+yz*kz,bz=((0+zx*kx)+zy*ky)+zz*kz;
        const nx=(ay*bz-az*by)*sign,ny=(az*bx-ax*bz)*sign,nz=(ax*by-ay*bx)*sign;
        if(da){const at=(da-1)*3;sums[at]+=nx;sums[at+1]+=ny;sums[at+2]+=nz;}
        if(db){const at=(db-1)*3;sums[at]+=nx;sums[at+1]+=ny;sums[at+2]+=nz;}
        if(dc){const at=(dc-1)*3;sums[at]+=nx;sums[at+1]+=ny;sums[at+2]+=nz;}
      }
    }
    for(let id=0;id<vertexCount;id++)if(this.dirty[id]){
      const at=(dirtyIndex[id]-1)*3,x=sums[at],y=sums[at+1],z=sums[at+2],nx=((0+xx*x)+yx*y)+zx*z,ny=((0+xy*x)+yy*y)+zy*z,nz=((0+xz*x)+yz*y)+zz*z,length=Math.hypot(nx,ny,nz),out=id*3;
      this.normals[out]=length?nx/length:0;this.normals[out+1]=length?ny/length:0;this.normals[out+2]=length?nz/length:0;this.dirty[id]=0;
    }
  }
  emit(remap:Uint32Array,activeVertices:number){
    const output=new Float32Array(activeVertices*3);
    for(let id=0;id<remap.length;id++)if(remap[id]){
      const to=(remap[id]-1)*3,from=id*3;output[to]=this.normals[from];output[to+1]=this.normals[from+1];output[to+2]=this.normals[from+2];
    }
    return output;
  }
}

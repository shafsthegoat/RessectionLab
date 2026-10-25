import type {Affine,Point3,Shape3} from './coordinates';
type Row={data:Float32Array|Uint32Array;x?:number;y?:number;z?:number};
/** Worker-local source normals. Only incident vertices of replaced chunk faces
 * are invalidated. Their sums are replayed in complete raster face order;
 * subtracting old sums or grouping by chunk would change floating-point bits. */
export class IncrementalSourceNormals {
  private normals=new Float32Array(0);
  private dirty=new Uint8Array(0);
  private knownVertices=0;
  private affineKey='';
  private initialized=false;
  private readonly selection?:{bits:Uint8Array;dimensions:Shape3;chunkSize:number};
  private selectionValid=true;
  private markedChunks=false;
  lastDirtyVertices=0;
  lastRecomputedFaces=0;
  lastScannedFaces=0;
  lastFullScan=true;
  private readonly admit:(bytes:number)=>void;
  constructor(admit:(bytes:number)=>void,source?:{shape:Shape3;chunkSize:number}){
    this.admit=admit;
    if(source){
      const dimensions=source.shape.map(size=>Math.ceil((size+1)/source.chunkSize)) as Shape3,count=dimensions.reduce((a,b)=>a*b,1);
      if(Number.isInteger(source.chunkSize)&&source.chunkSize>=2&&dimensions.every(size=>Number.isSafeInteger(size)&&size>0)&&Number.isSafeInteger(count)){
        const bytes=Math.ceil(count/8);this.admit(bytes);this.selection={bits:new Uint8Array(bytes),dimensions,chunkSize:source.chunkSize};
      }
    }
  }
  get selectionBytes(){return this.selection?.bits.byteLength??0;}
  get retainedBytes(){return this.normals.byteLength+this.dirty.byteLength+this.selectionBytes;}
  mark(ids:Uint32Array,coordinate?:Point3){
    for(const id of ids)if(id<this.dirty.length)this.dirty[id]=1;
    const selection=this.selection;if(!selection)return;
    if(!coordinate||coordinate.some((value,axis)=>!Number.isInteger(value)||value<0||value>=selection.dimensions[axis])){this.selectionValid=false;return;}
    const {bits,dimensions:[nx,ny,nz]}=selection,[cx,cy,cz]=coordinate;
    // Half-grid vertices of this core can be incident only to this chunk or
    // its immediate neighbors. Include deleted old cores as well as new ones.
    for(let x=Math.max(0,cx-1);x<=Math.min(nx-1,cx+1);x++)for(let y=Math.max(0,cy-1);y<=Math.min(ny-1,cy+1);y++)for(let z=Math.max(0,cz-1);z<=Math.min(nz-1,cz+1);z++){
      const index=(x*ny+y)*nz+z;bits[index>>>3]|=1<<(index&7);
    }
    this.markedChunks=true;
  }
  prepare(points:Float32Array,vertexCount:number,rows:Iterable<Row>,affine:Affine){
    if(vertexCount>this.dirty.length){
      let capacity=this.dirty.length||1024;while(capacity<vertexCount)capacity*=2;
      this.admit(capacity*13+this.selectionBytes);
      const normals=new Float32Array(capacity*3),dirty=new Uint8Array(capacity);normals.set(this.normals);dirty.set(this.dirty);this.normals=normals;this.dirty=dirty;
    }
    const key=JSON.stringify(affine);
    const affineChanged=key!==this.affineKey;
    let fullScan=affineChanged||!this.initialized||!this.selection||!this.selectionValid||(vertexCount>this.knownVertices&&!this.markedChunks);
    if(affineChanged){this.dirty.fill(1,0,vertexCount);this.affineKey=key;}
    else this.dirty.fill(1,this.knownVertices,vertexCount);
    this.knownVertices=vertexCount;
    let count=0;for(let id=0;id<vertexCount;id++)count+=this.dirty[id];
    this.lastDirtyVertices=count;this.lastRecomputedFaces=0;this.lastScannedFaces=0;this.lastFullScan=fullScan;
    if(!count){this.selection?.bits.fill(0);this.markedChunks=false;this.selectionValid=true;return;}
    const orderedRows=Array.isArray(rows)?rows:Array.from(rows),selection=this.selection;
    // Unknown row ownership cannot safely skip earlier contributions. Validate
    // before reduction, then preserve the original complete raster row order.
    if(!fullScan)for(const row of orderedRows){
      const coordinates=[row.x,row.y,row.z];
      if(coordinates.some((value,axis)=>!Number.isInteger(value)||Math.floor((value!+1)/selection!.chunkSize)<0||Math.floor((value!+1)/selection!.chunkSize)>=selection!.dimensions[axis])){fullScan=true;break;}
    }
    this.lastFullScan=fullScan;
    // Disposable scratch covers stable IDs, while sums cover dirty vertices
    // only. Neither scratch nor transferable output is retained by the cache.
    const dirtyIndex=new Uint32Array(vertexCount),sums=new Float64Array(count*3);
    let offset=0;for(let id=0;id<vertexCount;id++)if(this.dirty[id])dirtyIndex[id]=++offset;
    const [[xx,xy,xz],[yx,yy,yz],[zx,zy,zz]]=affine;
    const determinant=xx*(yy*zz-yz*zy)-xy*(yx*zz-yz*zx)+xz*(yx*zy-yy*zx),sign=Math.sign(determinant);
    for(const row of orderedRows){
      if(!fullScan){
        const {bits,dimensions:[,ny,nz],chunkSize}=selection!,x=Math.floor((row.x!+1)/chunkSize),y=Math.floor((row.y!+1)/chunkSize),z=Math.floor((row.z!+1)/chunkSize),index=(x*ny+y)*nz+z;
        if(!(bits[index>>>3]&(1<<(index&7))))continue;
      }
      const indices=row.data;this.lastScannedFaces+=indices.length/3;
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
    this.initialized=true;this.selection?.bits.fill(0);this.markedChunks=false;this.selectionValid=true;
  }
  emit(remap:Uint32Array,activeVertices:number){
    const output=new Float32Array(activeVertices*3);
    for(let id=0;id<remap.length;id++)if(remap[id]){
      const to=(remap[id]-1)*3,from=id*3;output[to]=this.normals[from];output[to+1]=this.normals[from+1];output[to+2]=this.normals[from+2];
    }
    return output;
  }
  /** Fresh transferable snapshot in the immutable source registry's ID space. */
  emitStable(vertexHighWater:number){return this.normals.slice(0,vertexHighWater*3);}
}

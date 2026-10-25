import type {Shape3} from './coordinates';

/** Only this cache optimization is abandoned on budget exhaustion. The caller
 * still prepares the exact full surface from its authoritative worker mask. */
export class SurfaceIndexBudgetExceeded extends Error {}

/** Stable worker-local IDs for exact half-grid source vertices. Keys cover the
 * padded source lattice; zero in values denotes an unused hash slot. */
export class SourceVertexIndex {
  private keys=new Uint32Array(0);
  private values=new Uint32Array(0);
  private coordinates=new Float32Array(0);
  private count=0;
  private readonly spanY:number;
  private readonly spanZ:number;
  private readonly spanX:number;
  private readonly admit:(bytes:number)=>void;
  constructor(shape:Shape3,admit:(bytes:number)=>void){
    this.admit=admit;
    // Above 2^23, Float32 source soup rounds half-grid corners. Keep the
    // original soup/indexing path, whose rounded-coordinate semantics differ.
    if(shape.some(n=>n>2**23))throw new SurfaceIndexBudgetExceeded('Source lattice exceeds exact Float32 half-grid coordinates');
    const spans=shape.map(n=>2*n+1);
    if(spans.some(n=>!Number.isSafeInteger(n)||n<1)||spans.reduce((a,b)=>a*b,1)>2**32)throw new SurfaceIndexBudgetExceeded('Source lattice exceeds bounded vertex keys');
    [this.spanX,this.spanY,this.spanZ]=spans;
  }
  get vertexCount(){return this.count;}
  /** Worker-internal immutable coordinates; never transfer this resident buffer. */
  get points(){return this.coordinates;}
  get retainedBytes(){return this.keys.byteLength+this.values.byteLength+this.coordinates.byteLength;}
  private growTable(){
    const capacity=this.keys.length?this.keys.length*2:1024;
    this.admit(capacity*8+this.coordinates.byteLength);
    const keys=new Uint32Array(capacity),values=new Uint32Array(capacity),mask=capacity-1;
    for(let i=0;i<this.values.length;i++)if(this.values[i]){
      const key=this.keys[i];let slot=Math.imul(key,0x9e3779b1)>>>0;slot&=mask;
      while(values[slot])slot=(slot+1)&mask;
      keys[slot]=key;values[slot]=this.values[i];
    }
    this.keys=keys;this.values=values;
  }
  intern(x:number,y:number,z:number){
    const ix=2*x+1,iy=2*y+1,iz=2*z+1;
    if(!Number.isInteger(ix)||!Number.isInteger(iy)||!Number.isInteger(iz)||ix<0||iy<0||iz<0||ix>=this.spanX||iy>=this.spanY||iz>=this.spanZ)throw Error('Invalid half-grid source vertex');
    if(!this.keys.length)this.growTable();
    const key=(ix*this.spanY+iy)*this.spanZ+iz;
    let mask=this.keys.length-1,slot=(Math.imul(key,0x9e3779b1)>>>0)&mask;
    while(this.values[slot]){if(this.keys[slot]===key)return this.values[slot]-1;slot=(slot+1)&mask;}
    if((this.count+1)*10>this.keys.length*7){
      this.growTable();mask=this.keys.length-1;slot=(Math.imul(key,0x9e3779b1)>>>0)&mask;
      while(this.values[slot])slot=(slot+1)&mask;
    }
    if(3*this.count===this.coordinates.length){
      const capacity=this.coordinates.length?this.coordinates.length*2:3072;
      this.admit(this.keys.byteLength+this.values.byteLength+capacity*4);
      const coordinates=new Float32Array(capacity);coordinates.set(this.coordinates);this.coordinates=coordinates;
    }
    const id=this.count++;this.keys[slot]=key;this.values[slot]=id+1;
    this.coordinates[3*id]=x;this.coordinates[3*id+1]=y;this.coordinates[3*id+2]=z;return id;
  }
  copy(id:number,output:Float32Array,offset:number){
    const at=3*id;output[offset]=this.coordinates[at];output[offset+1]=this.coordinates[at+1];output[offset+2]=this.coordinates[at+2];
  }
}

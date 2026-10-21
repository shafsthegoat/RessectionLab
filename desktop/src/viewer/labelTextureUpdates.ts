import * as THREE from 'three';
import type {Shape3} from './coordinates';
export interface LabelPatch {indices:Uint32Array;values:Uint8Array}
export interface LabelUploadRenderer {
  initTexture(texture:THREE.Texture):void;
  copyTextureToTexture?(source:THREE.Texture,destination:THREE.Texture,sourceRegion:THREE.Box3,destinationPosition:THREE.Vector3):void;
}
/** Logical upload bytes and bounded public-API copies, never direct GL state.
 * The source wrapper borrows CPU storage and is deliberately never initialized
 * as a GPU texture. Three then uses its texSubImage3D CPU-copy branch. */
export class LabelTextureUpdates {
  private texture:THREE.Data3DTexture|null=null;
  private uploadedVersion=-1;
  // Submission wall time includes CPU/driver calls, not GPU execution time.
  readonly work={fullUploads:0,fullUploadBytes:0,fullSubmissionMs:0,partialBatches:0,partialRegions:0,partialUploadBytes:0,partialSubmissionMs:0,fallbacks:0,unchanged:0};
  sync(renderer:LabelUploadRenderer,texture:THREE.Data3DTexture){
    if(this.texture===texture&&this.uploadedVersion===texture.version)return;
    const started=performance.now();renderer.initTexture(texture);this.work.fullSubmissionMs+=performance.now()-started;
    this.texture=texture;this.uploadedVersion=texture.version;
    this.work.fullUploads++;this.work.fullUploadBytes+=(texture.image.data as Uint8Array).byteLength;
  }
  update(renderer:LabelUploadRenderer,texture:THREE.Data3DTexture,shape:Shape3,patch:LabelPatch|undefined,full:boolean):'none'|'partial'|'full' {
    const data=texture.image.data as Uint8Array;
    if(!full&&patch&&!patch.indices.length){this.work.unchanged++;return 'none';}
    const valid=data instanceof Uint8Array&&data.length===shape[0]*shape[1]*shape[2]*4&&
      texture.image.width===shape[2]&&texture.image.height===shape[1]&&texture.image.depth===shape[0]&&texture.format===THREE.RGBAFormat&&texture.type===THREE.UnsignedByteType&&texture.unpackAlignment===1&&!texture.flipY;
    if(!full&&patch&&valid&&patch.values.length===patch.indices.length*4&&this.texture===texture&&this.uploadedVersion===texture.version&&renderer.copyTextureToTexture){
      const regions=new Map<string,{low:number[];high:number[]}>();let invalid=false;
      for(const index of patch.indices){
        if(index>=data.length/4){invalid=true;break;}
        // Native memory is z-fastest; GPU (u,v,w) is native (z,y,x).
        const point=[index%shape[2],Math.floor(index/shape[2])%shape[1],Math.floor(index/(shape[1]*shape[2]))],key=point.map(v=>Math.floor(v/32)).join(',');
        const region=regions.get(key);if(region)point.forEach((v,a)=>{region.low[a]=Math.min(region.low[a],v);region.high[a]=Math.max(region.high[a],v+1);});
        else regions.set(key,{low:[...point],high:point.map(v=>v+1)});
      }
      const bytes=[...regions.values()].reduce((sum,r)=>sum+r.high.reduce((n,v,a)=>n*(v-r.low[a]),4),0);
      // Keep command count and overdraw bounded. Box3.max is exclusive in the
      // installed Three copy implementation; each region includes its last cell.
      if(!invalid&&regions.size<=64&&bytes<=data.byteLength/2){
        const source=new THREE.Data3DTexture(data,shape[2],shape[1],shape[0]);source.format=THREE.RGBAFormat;source.type=THREE.UnsignedByteType;source.unpackAlignment=1;
        const started=performance.now();
        try{
          for(const region of regions.values()){
            const low=new THREE.Vector3(...region.low),high=new THREE.Vector3(...region.high);
            renderer.copyTextureToTexture(source,texture,new THREE.Box3(low,high),low);
            this.work.partialRegions++;this.work.partialUploadBytes+=region.high.reduce((n,v,a)=>n*(v-region.low[a]),4);
          }
          this.work.partialBatches++;return 'partial';
        }catch{this.work.fallbacks++;}
        finally{this.work.partialSubmissionMs+=performance.now()-started;source.dispose();}
      }
    }
    texture.needsUpdate=true;return 'full';
  }
}

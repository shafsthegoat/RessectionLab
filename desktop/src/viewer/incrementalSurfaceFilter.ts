import type {SourceAuditLayout} from './surfaceSafety';

/** Produced only by the worker's immutable source registry. Every replaced
 * chunk contributes its old and new incident vertices; generations prevent a
 * missed/failed/fallback frame from hiding an intervening topology change. */
export interface SourceFilterLayout extends SourceAuditLayout {
  registry:object;generation:number;generationValid:boolean;dirtyVertices:Uint32Array;
}
export interface SurfacePolynomialFilter {
  compute(original:Float64Array,offsets:Uint32Array,neighbors:Uint32Array,pinned:Uint8Array,coefficients:Float64Array):Float64Array|undefined;
  capture(original:Float64Array,filtered:Float64Array):void;
  complete():void;
}
interface Field {
  registry:object;generation:number;key:string;source:Float64Array;raw:Float64Array;active:Uint8Array;
}

/** Exact finite-degree graph polynomial. The full implementation remains the
 * fallback/reference; no tolerance, approximate boundary or physical radius
 * determines which outputs can be reused. */
export class IncrementalSurfaceFilter {
  private field?:Field;
  private work={local:false,totalVertices:0,affectedVertices:0,haloVertices:0,reason:'unprepared'};
  get retainedBytes(){return this.field?this.field.source.byteLength+this.field.raw.byteLength+this.field.active.byteLength:0;}
  get statistics(){return {...this.work,retainedBytes:this.retainedBytes};}
  begin(layout:SourceFilterLayout,maxBytes:number):{filter?:SurfacePolynomialFilter;reservedBytes:number} {
    const n=layout.vertexIds.length,bytes=49*layout.vertexCapacity,reserved=bytes+this.retainedBytes;
    this.work={local:false,totalVertices:n,affectedVertices:n,haloVertices:n,reason:'full-reference'};
    if(this.retainedBytes>maxBytes)this.field=undefined;
    if(!n||!Number.isSafeInteger(reserved)||reserved>maxBytes||!Number.isSafeInteger(maxBytes)||maxBytes<0||
      !Number.isSafeInteger(layout.vertexCapacity)||layout.vertexCapacity<n||
      !Number.isSafeInteger(layout.generation)||layout.generation<0||!layout.generationValid||!layout.registry||typeof layout.registry!=='object'){
      this.field=undefined;this.work.reason='unavailable-provenance-or-budget';return {reservedBytes:0};
    }
    const reverse=new Uint32Array(layout.vertexCapacity);
    for(let v=0;v<n;v++){
      const id=layout.vertexIds[v];
      if(id>=reverse.length||reverse[id]){this.field=undefined;this.work.reason='invalid-stable-ids';return {reservedBytes:0};}
      reverse[id]=v+1;
    }
    if(layout.dirtyVertices.some(id=>id>=layout.vertexCapacity)){
      this.field=undefined;this.work.reason='invalid-dirty-ids';return {reservedBytes:0};
    }
    const prior=this.field;
    let key='',pending:Field|undefined,canCapture=false;
    const filter:SurfacePolynomialFilter={
      compute:(original,offsets,neighbors,pinned,coefficients)=>{
        canCapture=original.length===n*3&&original.every(Number.isFinite)&&coefficients.every(Number.isFinite);
        key=JSON.stringify(Array.from(coefficients,value=>Object.is(value,-0)?'-0':value));
        const degree=coefficients.length-1;
        if(!canCapture||degree<1||degree>128||offsets.length!==n+1||pinned.length!==n||offsets[0]!==0||
          offsets[n]!==neighbors.length||neighbors.some(v=>v>=n)){
          canCapture=false;this.work.reason='unsupported-polynomial-input';return;
        }
        for(let v=0;v<n;v++)if(offsets[v]>offsets[v+1]){canCapture=false;this.work.reason='invalid-adjacency';return;}
        if(!prior||prior.registry!==layout.registry||prior.key!==key||
          (prior.generation!==layout.generation&&prior.generation+1!==layout.generation)){
          this.work.reason='new-registry-coefficients-or-generation-gap';return;
        }
        const distance=new Uint16Array(n).fill(65535),queue=new Uint32Array(n);let head=0,tail=0;
        const seed=(v:number)=>{if(distance[v]===65535){distance[v]=0;queue[tail++]=v;}};
        // The source producer guarantees coverage of changed triangle incidence,
        // including deletions. Changed coordinates/new IDs additionally seed D.
        for(const id of layout.dirtyVertices){const v=reverse[id];if(v)seed(v-1);}
        for(let v=0;v<n;v++){
          const id=layout.vertexIds[v];let changed=!prior.active[id];
          for(let a=0;a<3&&!changed;a++)changed=!Object.is(original[v*3+a],prior.source[id*3+a]);
          if(changed)seed(v);
        }
        while(head<tail){
          const v=queue[head++];if(distance[v]>=2*degree)continue;
          for(let e=offsets[v];e<offsets[v+1];e++){
            const other=neighbors[e];if(distance[other]===65535){distance[other]=distance[v]+1;queue[tail++]=other;}
          }
        }
        if(tail>n*.6){this.work.reason='dense-halo';return;}
        const localId=new Uint32Array(n),output=new Float64Array(original.length);
        let affected=0;
        for(let v=0;v<n;v++)for(let a=0;a<3;a++)output[v*3+a]=prior.raw[layout.vertexIds[v]*3+a];
        for(let l=0;l<tail;l++){
          const v=queue[l];localId[v]=l+1;
          if(distance[v]<=degree){affected++;for(let a=0;a<3;a++)output[v*3+a]=coefficients[0]*original[v*3+a];}
        }
        let previous=new Float64Array(tail*3),current=new Float64Array(tail*3),next=new Float64Array(tail*3);
        for(let l=0;l<tail;l++)for(let a=0;a<3;a++)previous[l*3+a]=original[queue[l]*3+a];
        for(let j=1;j<coefficients.length;j++){
          for(let l=0;l<tail;l++){
            const v=queue[l];if(distance[v]>2*degree-j)continue;
            const count=offsets[v+1]-offsets[v];
            for(let a=0;a<3;a++){
              let sum=0;
              for(let e=offsets[v];e<offsets[v+1];e++){
                const at=localId[neighbors[e]];
                if(!at){this.work.reason='incomplete-halo';return;}
                sum+=previous[(at-1)*3+a];
              }
              const i=l*3+a;
              next[i]=pinned[v]||!count?original[v*3+a]:j===1?sum/count:2*sum/count-current[i];
              if(distance[v]<=degree)output[v*3+a]+=coefficients[j]*next[i];
            }
          }
          [current,previous,next]=[previous,next,current];
        }
        this.work={local:true,totalVertices:n,affectedVertices:affected,haloVertices:tail,reason:'exact-graph-halo'};
        return output;
      },
      capture:(original,filtered)=>{
        if(!canCapture||original.length!==n*3||filtered.length!==n*3||!filtered.every(Number.isFinite))return;
        const next:Field={registry:layout.registry,generation:layout.generation,key,source:new Float64Array(layout.vertexCapacity*3),
          raw:new Float64Array(layout.vertexCapacity*3),active:new Uint8Array(layout.vertexCapacity)};
        for(let v=0;v<n;v++){
          const id=layout.vertexIds[v];next.active[id]=1;
          for(let a=0;a<3;a++){next.source[id*3+a]=original[v*3+a];next.raw[id*3+a]=filtered[v*3+a];}
        }
        pending=next;
      },
      complete:()=>{if(pending)this.field=pending;},
    };
    return {filter,reservedBytes:reserved};
  }
}

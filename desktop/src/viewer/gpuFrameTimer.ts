interface TimerExtension {TIME_ELAPSED_EXT:number;GPU_DISJOINT_EXT:number}
export interface GpuElapsedSample {serial:number;measurement:'asynchronous-GPU-elapsed';completed:boolean;milliseconds?:number;reason?:string}
/** Asynchronous elapsed queries measure GPU execution, unlike fence-poll wall
 * time. Outstanding queries and polling lifetime remain bounded independently
 * of the rendering clock. Unsupported/disjoint/context-lost results have no ms. */
export class GpuFrameTimer {
  readonly supported:boolean;
  private readonly extension:TimerExtension|null;
  private readonly pending:Array<{query:WebGLQuery;serial:number;polls:number}>=[];
  private active:{query:WebGLQuery;serial:number;polls:number}|null=null;
  private pollHandle=0;
  private disposed=false;
  private readonly gl:WebGL2RenderingContext;
  private readonly emit:(sample:GpuElapsedSample)=>void;
  private readonly schedule:(callback:FrameRequestCallback)=>number;
  private readonly cancel:(handle:number)=>void;
  constructor(gl:WebGL2RenderingContext,emit:(sample:GpuElapsedSample)=>void,
    schedule:(callback:FrameRequestCallback)=>number=callback=>requestAnimationFrame(callback),
    cancel:(handle:number)=>void=handle=>cancelAnimationFrame(handle)){
    this.gl=gl;this.emit=emit;this.schedule=schedule;this.cancel=cancel;
    this.extension=gl.getExtension('EXT_disjoint_timer_query_webgl2') as TimerExtension|null;
    this.supported=!!this.extension;
  }
  begin(serial:number):void {
    if(this.disposed||!this.extension||this.active||this.pending.length>=3||this.gl.isContextLost())return;
    if(this.gl.getParameter(this.extension.GPU_DISJOINT_EXT)){this.rejectPending('disjoint');return;}
    const query=this.gl.createQuery();if(!query)return;
    this.gl.beginQuery(this.extension.TIME_ELAPSED_EXT,query);this.active={query,serial,polls:0};
  }
  end():void {
    if(!this.active||!this.extension)return;
    this.gl.endQuery(this.extension.TIME_ELAPSED_EXT);this.pending.push(this.active);this.active=null;
    this.schedulePoll();
  }
  private schedulePoll(){if(!this.disposed&&!this.pollHandle&&this.pending.length)this.pollHandle=this.schedule(()=>{this.pollHandle=0;this.poll();});}
  private sample(serial:number,completed:boolean,milliseconds?:number,reason?:string){
    this.emit({serial,measurement:'asynchronous-GPU-elapsed',completed,...(milliseconds===undefined?{}:{milliseconds}),...(reason?{reason}:{})});
  }
  private rejectPending(reason:string){
    for(const item of this.pending){if(!this.gl.isContextLost())this.gl.deleteQuery(item.query);this.sample(item.serial,false,undefined,reason);}
    this.pending.length=0;
  }
  private poll(){
    if(this.disposed||!this.extension)return;
    if(this.gl.isContextLost()){this.rejectPending('context-lost');return;}
    if(this.gl.getParameter(this.extension.GPU_DISJOINT_EXT)){this.rejectPending('disjoint');return;}
    for(let i=this.pending.length-1;i>=0;i--){
      const item=this.pending[i];
      if(this.gl.getQueryParameter(item.query,this.gl.QUERY_RESULT_AVAILABLE)){
        const nanoseconds=this.gl.getQueryParameter(item.query,this.gl.QUERY_RESULT) as number;
        this.gl.deleteQuery(item.query);this.pending.splice(i,1);
        this.sample(item.serial,Number.isFinite(nanoseconds)&&nanoseconds>=0,Number.isFinite(nanoseconds)&&nanoseconds>=0?nanoseconds/1e6:undefined);
      }else if(++item.polls>=120){
        this.gl.deleteQuery(item.query);this.pending.splice(i,1);this.sample(item.serial,false,undefined,'query-timeout');
      }
    }
    this.schedulePoll();
  }
  dispose(reason='disposed'){
    if(this.disposed)return;this.disposed=true;this.cancel(this.pollHandle);this.pollHandle=0;
    if(this.active){if(!this.gl.isContextLost()){
      this.gl.endQuery(this.extension!.TIME_ELAPSED_EXT);this.gl.deleteQuery(this.active.query);
    }this.sample(this.active.serial,false,undefined,reason);this.active=null;}
    this.rejectPending(reason);
  }
}

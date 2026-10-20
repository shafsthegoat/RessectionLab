'use strict';
// Overhead-bearing ownership/byte trace of the same production benchmark.
// Keep this separate from the balanced primary timing runs.
const {app}=require('electron');
const fs=require('node:fs');
const path=require('node:path');
const root=path.resolve(__dirname,'../..');
const output=path.resolve(process.env.INTERACTION_OUTPUT||path.join(root,'testcases/performance/rehearsal-2026-10-19/transport-final'));
if(!output.startsWith(path.join(root,'testcases')+path.sep))throw Error('Keep transport evidence under testcases/');
process.env.INTERACTION_OUTPUT=output;
process.env.INTERACTION_MODE='rehearsal';
const contents=new Set();
function installTrace(){
  if(window.rehearsalTransportTrace)return;
  const trace=window.rehearsalTransportTrace={events:[],measurement:'Unique ArrayBuffer byteLength per worker message; metadata/object headers excluded. Input transfer-list bytes are explicit; received-buffer ownership is verified separately against the worker implementation. This extra tracing is excluded from primary timing comparisons.'};
  const buffers=value=>{
    const seen=new Set(),owned=new Set(),shared=new Set();
    const visit=item=>{
      if(!item||typeof item!=='object'||seen.has(item))return;seen.add(item);
      if(ArrayBuffer.isView(item)){visit(item.buffer);return;}
      if(item instanceof ArrayBuffer){owned.add(item);return;}
      if(typeof SharedArrayBuffer!=='undefined'&&item instanceof SharedArrayBuffer){shared.add(item);return;}
      for(const child of Object.values(item))visit(child);
    };visit(value);
    return {bufferBytes:[...owned].reduce((n,b)=>n+b.byteLength,0),buffers:owned.size,sharedBytes:[...shared].reduce((n,b)=>n+b.byteLength,0)};
  };
  const OriginalWorker=window.Worker;
  window.Worker=class extends OriginalWorker{
    constructor(...args){super(...args);this.traceKind=String(args[0]);
      this.addEventListener('message',event=>trace.events.push({at:performance.now(),direction:'received',worker:this.traceKind,...buffers(event.data)}));
    }
    postMessage(value,options){
      const transfer=Array.isArray(options)?options:options?.transfer??[],event={at:performance.now(),direction:'sent',worker:this.traceKind,...buffers(value),transferListBytes:transfer.reduce((n,b)=>n+(b instanceof ArrayBuffer?b.byteLength:0),0),unknownTransferItems:transfer.filter(b=>!(b instanceof ArrayBuffer)).length,accepted:true};
      try{return super.postMessage(value,options);}catch(error){event.accepted=false;event.error=String(error);throw error;}
      finally{trace.events.push(event);}
    }
  };
}
app.on('web-contents-created',(_event,web)=>{
  contents.add(web);
  web.once('did-finish-load',()=>void web.executeJavaScript('('+installTrace.toString()+')()',true).catch(error=>console.error('Transport trace installation failed:',error)));
});
const originalExit=app.exit.bind(app);let exiting=false;
app.exit=code=>{
  if(exiting)return;exiting=true;
  void (async()=>{
    try{
      const traces=await Promise.all([...contents].filter(web=>!web.isDestroyed()).map(async web=>({webContentsId:web.id,trace:await web.executeJavaScript('window.rehearsalTransportTrace??null',true)})));
      const report=JSON.parse(fs.readFileSync(path.join(output,'report.json'),'utf8'));
      const events=traces.flatMap(t=>t.trace?.events??[]);
      const runs=report.runs.map(run=>({name:run.name,events:events.filter(event=>event.at>=run.inputAt&&event.at<=run.consistentDisplayAt)}));
      fs.writeFileSync(path.join(output,'transport.json'),JSON.stringify({recordedAt:new Date().toISOString(),label:report.label,implementationIdentity:report.implementationIdentity,source:report.source,reportPassed:report.passed,traces,runs},null,2));
    }catch(error){fs.mkdirSync(output,{recursive:true});fs.writeFileSync(path.join(output,'transport-failure.txt'),String(error.stack));console.error(error);code=1;}
    originalExit(code);
  })();
};
require('./benchmark-interactions.cjs');

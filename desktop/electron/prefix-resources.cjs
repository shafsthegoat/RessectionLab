'use strict';
const {execFile}=require('node:child_process');
const {promisify}=require('node:util');
const run=promisify(execFile);
const MAX_RSS=3*1024**3;

function treeRss(text,pids){
  const rows=new Map();
  for(const line of text.trim().split('\n')){
    const fields=line.trim().split(/\s+/).map(Number);
    if(fields.length!==3||fields.some(x=>!Number.isSafeInteger(x)||x<0))throw Error('Invalid process resource sample');
    rows.set(fields[0],{parent:fields[1],bytes:fields[2]*1024});
  }
  const found=new Set(pids);
  if(pids.some(pid=>!rows.has(pid)))throw Error('Prefix worker is no longer running');
  let changed=true;
  while(changed){changed=false;for(const [pid,row]of rows)if(found.has(row.parent)&&!found.has(pid)){found.add(pid);changed=true;}}
  let total=0;for(const pid of found)total+=rows.get(pid)?.bytes||0;
  if(!Number.isSafeInteger(total))throw Error('Invalid process resource total');
  return total;
}

class PrefixResources{
  constructor({pids,onFailure,maxWallMs=7200000,now=()=>performance.now(),sample}={}){
    if(!Array.isArray(pids)||pids.length!==2||new Set(pids).size!==2||pids.some(p=>!Number.isSafeInteger(p)||p<1)
      ||typeof onFailure!=='function'||!Number.isSafeInteger(maxWallMs)||maxWallMs<100||maxWallMs>7200000)throw Error('Invalid two-worker resource budget');
    this.pids=[...pids];this.onFailure=onFailure;this.now=now;this.maxWallMs=maxWallMs;this.began=now();
    this.sample=sample||(()=>run('/bin/ps',['-axo','pid=,ppid=,rss='],{timeout:1500,maxBuffer:1024*1024}).then(r=>r.stdout));
    this.closed=false;this.failed=null;this.peak=0;this.samples=0;this.pending=null;this.timer=null;
  }
  async check(){
    if(this.closed)return;
    if(this.pending)return this.pending;
    this.pending=(async()=>{
      try{
        if(this.now()-this.began>=this.maxWallMs)throw Error('Prefix pipeline reached its wall-time limit');
        const rss=treeRss(await this.sample(),this.pids);
        if(this.closed)return;
        this.samples++;this.peak=Math.max(this.peak,rss);
        if(rss>MAX_RSS)throw Error('Prefix workers exceeded the aggregate 3 GiB RSS limit');
        if(this.now()-this.began>=this.maxWallMs)throw Error('Prefix pipeline reached its wall-time limit');
      }catch(error){
        if(!this.closed&&!this.failed){this.failed=error;this.closed=true;clearTimeout(this.timer);this.onFailure(error);}
        if(!this.closed||this.failed)throw error;
      }
    })();
    try{return await this.pending;}finally{this.pending=null;}
  }
  async start(){
    await this.check();
    const poll=()=>{if(!this.closed)this.timer=setTimeout(()=>{this.check().then(poll,()=>{});},250);};
    poll();
  }
  async close(){this.closed=true;clearTimeout(this.timer);if(this.pending)await this.pending.catch(()=>{});}
  get statistics(){return{workerCount:2,pids:[...this.pids],samples:this.samples,peakAggregatePythonTreeRssBytes:this.peak,
    maxAggregatePythonTreeRssBytes:MAX_RSS,wallMs:this.now()-this.began,maxWallMs:this.maxWallMs,
    failure:this.failed?.message||null,scope:'Two Python process trees including descendants. Electron/renderer/GPU memory is separate; sampled RSS includes transient decode/native allocations.'};}
}
module.exports={PrefixResources,treeRss,MAX_RSS};

'use strict';
const fs = require('node:fs/promises');
const path = require('node:path');
const {execFile} = require('node:child_process');
const {promisify} = require('node:util');
const {randomUUID} = require('node:crypto');
const run = promisify(execFile);

// The last authenticated state at a simultaneous transition owns the following
// interval. A one-second terminal hold makes the final state visible.
function movieIntervals(samples) {
  if (!samples.length) throw Error('No observed engine states to encode.');
  const origin = samples[0].time, result = [];
  for (const [index,sample] of samples.entries()) {
    const milliseconds = Math.round((sample.time-origin)*1000);
    if (!Number.isSafeInteger(milliseconds) || milliseconds < (result.at(-1)?.milliseconds ?? 0))
      throw Error('Engine observation times are out of order.');
    if (milliseconds === result.at(-1)?.milliseconds) result.pop();
    result.push({index,milliseconds});
  }
  return result.map((item,i)=>({...item,duration:(result[i+1]?.milliseconds ?? item.milliseconds+1000)-item.milliseconds}));
}

async function ffmpeg(args,options={}) {
  const search = [process.env.PATH,'/opt/homebrew/bin','/usr/local/bin'].filter(Boolean).join(path.delimiter);
  try { return await run('ffmpeg',args,{...options,env:{...process.env,PATH:search}}); }
  catch(error) { throw Error(error.code==='ENOENT'?'Install FFmpeg with H.264 and subtitle support to export a replay.':String(error.stderr || error.message).slice(-1800)); }
}

function subtitleTime(ms) {return new Date(ms).toISOString().slice(11,23).replace('.',',');}
function caption(value) {return String(value).replace(/[\r\n<>{}\\]/g,' ').slice(0,160);}

// Subtitle rendering has its own image strip, so wrapped text cannot cover
// source pixels. Reserve space for the longest authenticated cue in the movie.
function captionLayout(width,rows) {
  const outputWidth=Math.max(320,Math.ceil(width/2)*2);
  const pixels=outputWidth>=900?17:15;
  const columns=Math.max(8,Math.floor((outputWidth-36)/(pixels*.68)));
  const cues=rows.map(lines=>lines.flatMap(line=>{
    const wrapped=[];let row='',used=0;
    const units=char=>char.codePointAt(0)>=0x2e80?2:1;
    const flush=()=>{if(row)wrapped.push(row);row='';used=0;};
    for(const word of line.trim().split(/\s+/u)){
      const length=Array.from(word).reduce((n,char)=>n+units(char),0);
      if(row&&used+1+length>columns)flush();
      if(row){row+=' ';used++;}
      for(const char of word){
        if(used+units(char)>columns)flush();
        row+=char;used+=units(char);
      }
    }
    flush();
    return wrapped;
  }));
  const lines=cues.reduce((longest,cue)=>Math.max(longest,cue.length),0);
  const height=Math.ceil((lines*pixels*1.45+20)/2)*2;
  return {width:outputWidth,height,fontSize:pixels*288/height,cues};
}

async function encodeStudyReplay(directory,samples,source,rect,signal) {
  const frames=[],rows=[],intervals=movieIntervals(samples);
  const completions=new Map();
  for(const s of samples) if(s.frameIndex===s.frameCount-1){
    const t=Math.round((s.time-samples[0].time)*1000);
    completions.set(t,'Action '+(s.position+1)+' '+s.operation+' '+s.status);
  }
  for(const {index,milliseconds,duration} of intervals) {
    const s=samples[index];
    frames.push("file '"+index+".png'",'option framerate 1000','duration '+duration/1000);
    const quantity=value=>Number.isFinite(value)?value.toFixed(1)+' mm³':'unavailable';
    const boundary=s.endpoint??completions.get(milliseconds);
    rows.push([
      caption(source)+'  |  '+caption(s.branchId)+'  |  '+s.time.toFixed(2)+' simulated s',
      'Action '+(s.position+1)+'/'+s.steps+'  ·  '+caption(s.operation)+'  ·  '+caption(s.kind.replaceAll('_',' '))+
        (boundary?'  ·  '+caption(boundary):''),
      'Target removed '+quantity(s.targetRemoved)+'  ·  Residual '+quantity(s.originalResidual)+'  ·  Other tissue removed '+quantity(s.nonTargetRemoved)]);
  }
  const layout=captionLayout(rect.width,rows);
  const titles=intervals.flatMap(({milliseconds,duration},i)=>[String(i+1),subtitleTime(milliseconds)+' --> '+subtitleTime(milliseconds+duration),...layout.cues[i],'']);
  const last=intervals.at(-1);
  frames.push("file '"+last.index+".png'",'option framerate 1000');
  await fs.writeFile(path.join(directory,'frames.txt'),frames.join('\n')+'\n');
  await fs.writeFile(path.join(directory,'titles.srt'),titles.join('\n'));
  await ffmpeg(['-nostdin','-hide_banner','-loglevel','error','-f','concat','-safe','0','-i','frames.txt',
    '-filter_complex',`[0:v]fps=50:round=up,pad=${layout.width}:ceil(ih/2)*2:0:0:color=0x101619[field];color=c=0x101619:s=${layout.width}x${layout.height}:r=50,subtitles=titles.srt:force_style='FontName=Menlo,FontSize=${layout.fontSize},MarginL=2,MarginR=2,MarginV=10,Outline=0,Shadow=0'[caption];[field][caption]vstack=inputs=2:shortest=1[output]`,'-map','[output]',
    '-t',String((last.milliseconds+last.duration)/1000),'-c:v','libx264','-pix_fmt','yuv420p','-preset','fast','-crf','16','-movflags','+faststart','movie.mp4'],{cwd:directory,signal});
  return {captured:samples.length,timedImages:intervals.length,simulatedSeconds:last.milliseconds/1000,terminalHoldSeconds:1,captionBandPixels:layout.height};
}

class StudyReplayRecorder {
  constructor() {this.session=null;}
  async begin({choose,caseHash,branchId,source}) {
    if(this.session) throw Error('A replay export is already active.');
    await ffmpeg(['-version']);
    const filename=await choose();
    if(!filename) return null;
    if(this.session) throw Error('A replay export is already active.');
    const session={id:randomUUID(),caseHash,branchId,source,filename,samples:[],abort:new AbortController()};
    this.session=session;
    try {session.directory=await fs.mkdtemp(path.join(path.dirname(filename),'.resectionlab-recording-'));}
    catch(error){this.session=null;throw error;}
    return session.id;
  }
  observe(response) {
    const s=this.session;
    if(!s) return;
    const p=response?.playback,v=response?.view;
    s.receipt=null;
    if(response?.caseHash!==s.caseHash || response.branchId!==s.branchId) return;
    s.source??=v?.scene?.source?.caseId??v?.scene?.sceneId?.slice(0,19)??'Recorded engine episode';
    if(response.timeline) s.actions=response.timeline;
    const comparison=response.cranialPlanning;
    if(comparison?.selectedBranchId===s.branchId) {
      const selected=comparison.alternatives.find(row=>row.branchId===s.branchId);
      if(selected) s.endpoint={stateId:selected.finalStateId,reason:selected.stopReason,rootId:selected.rootId};
    }
    s.steps=response.branches?.find(b=>b.id===s.branchId)?.steps??s.steps;
    if(!p?.readOnly || !v) return;
    const action=s.actions?.[p.position];
    if(!action || !s.steps) throw Error('Verified branch action history is unavailable for export.');
    if((p.frameIndex===0 && v.stateId!==action.stateBefore) ||
       (p.frameIndex===p.frameCount-1 && v.stateId!==action.stateAfter))
      throw Error('Observed recording boundary differs from the archived action.');
    s.receipt={position:p.position,frameIndex:p.frameIndex,frameCount:p.frameCount,time:p.observedTimeSeconds,
      stateId:v.stateId,branchId:s.branchId,steps:s.steps,kind:p.frames[p.frameIndex].kind,operation:action.action.operation,
      endpoint:p.frameIndex===p.frameCount-1 && p.position===s.steps-1 && s.endpoint?.stateId===v.stateId?
        s.endpoint.rootId+' · '+s.endpoint.reason.replaceAll('_',' ').toLowerCase():null,
      status:action.status,removedCells:v.removed.length,targetRemoved:v.metrics.targetRemovedMm3,
      nonTargetRemoved:v.metrics.nonTargetRemovedMm3,originalResidual:v.originalTargetResidualMm3??null,
      contactCells:v.contact.length,tools:Object.fromEntries(Object.entries(v.slots).map(([k,t])=>[k,{
        instrument:t.instrument,position:t.position,quaternion:t.quaternion,inserted:t.inserted}]))};
  }
  async command(args,win) {
    const s=this.session;
    if(!s&&args.command==='cancel')return; // Abort and normal cleanup can both release the same export.
    if(!s || args.id!==s.id) throw Error('Replay export session expired.');
    if(args.command==='cancel') {await this.clear(s);return;}
    if(s.busy) throw Error('A replay capture is already in progress.');
    s.busy=true;
    try {
      if(args.command==='finish') {
        const last=s.samples.at(-1);
        if(!last || last.position!==s.steps-1 || last.frameIndex!==last.frameCount-1)
          throw Error('Replay export did not reach the verified endpoint.');
        const media=await encodeStudyReplay(s.directory,s.samples,s.source,s.rect,s.abort.signal);
        if(s!==this.session) throw Error('Export cancelled.');
        await fs.rename(path.join(s.directory,'movie.mp4'),s.filename);
        await this.clear(s);
        return media;
      }
      const r=s.receipt,last=s.samples.at(-1),first=!last || last.frameIndex===last.frameCount-1;
      if(args.command!=='capture' || !r || r.stateId!==args.stateId ||
         r.position!==(last?last.position+Number(first):0) || r.frameIndex!==(first?0:last.frameIndex+1) ||
         (last && r.time<last.time-1e-9))
        throw Error('Displayed observation differs from the expected next engine frame.');
      const rect=args.rect,bounds=win.getContentBounds();
      if(!rect || !['x','y','width','height'].every(key=>Number.isSafeInteger(rect[key])) ||
          rect.width<16 || rect.height<16 || rect.x<0 || rect.y<0 || rect.x+rect.width>bounds.width || rect.y+rect.height>bounds.height)
        throw Error('Keep the operative field visible while exporting.');
      if(s.rect && Object.keys(rect).some(key=>rect[key]!==s.rect[key])) throw Error('The operative field changed size during export.');
      s.rect={...rect};
      const image=await win.webContents.capturePage(rect,{stayHidden:true,stayAwake:true});
      if(image.isEmpty() || s!==this.session || s.receipt!==r) throw Error('The displayed state changed during capture.');
      await fs.writeFile(path.join(s.directory,s.samples.length+'.png'),image.resize({width:rect.width,height:rect.height}).toPNG());
      s.samples.push(r);
    } finally {s.busy=false;}
  }
  async clear(s=this.session) {
    if(!s || this.session!==s) return;
    this.session=null;s.abort.abort();
    if(s.directory) await fs.rm(s.directory,{recursive:true,force:true});
  }
}
module.exports={captionLayout,movieIntervals,StudyReplayRecorder};

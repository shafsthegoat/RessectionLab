// Actual CUA App.getScreenshot({emit:false}) image bytes -> local ffmpeg, no OS capture.
// Importing this module starts no process, screenshot, timer or UI operation.
import {open,mkdir} from 'node:fs/promises';
import {dirname,resolve,basename} from 'node:path';
import {spawn} from 'node:child_process';
import {once} from 'node:events';
import {performance} from 'node:perf_hooks';

const PNG=Buffer.from([137,80,78,71,13,10,26,10]);
function imageInfo(bytes){
  if(!ArrayBuffer.isView(bytes)||bytes.BYTES_PER_ELEMENT!==1||bytes.byteLength<24)throw Error('CUA screenshot must be an encoded Uint8Array');
  const data=Buffer.from(bytes.buffer,bytes.byteOffset,bytes.byteLength);
  if(data.subarray(0,8).equals(PNG))return {width:data.readUInt32BE(16),height:data.readUInt32BE(20),codec:'png'};
  if(data[0]===255&&data[1]===216){
    let offset=2;
    while(offset+4<data.length){
      if(data[offset++]!==255)continue;
      while(data[offset]===255)offset++;
      const marker=data[offset++];
      if(marker===216||marker===217||(marker>=208&&marker<=215))continue;
      const length=data.readUInt16BE(offset);
      if(length<2||offset+length>data.length)break;
      if([192,193,194,195,197,198,199,201,202,203,205,206,207].includes(marker))return {height:data.readUInt16BE(offset+3),width:data.readUInt16BE(offset+5),codec:'mjpeg'};
      offset+=length;
    }
  }
  throw Error('CUA screenshot must be valid PNG or JPEG bytes; extension is not format evidence');
}

export async function startRecorder({app,outputPath,fps=2,ffmpegPath='/opt/homebrew/bin/ffmpeg',engineeringTest=false}){
  if(typeof app?.getScreenshot!=='function')throw Error('Pass the bound CUA app with documented getScreenshot method');
  if(!Number.isFinite(fps)||fps<=0||fps>10)throw Error('fps must be >0 and <=10');
  const path=resolve(outputPath),marker='/testcases/outputs/';
  if(!path.includes(marker)||!path.endsWith('.mkv'))throw Error('Output must be a new .mkv under ignored testcases/outputs/');
  if(engineeringTest&&!basename(path).includes('ENGINEERINGHELPERTEST'))throw Error('Engineering output filename must explicitly contain ENGINEERINGHELPERTEST');
  await mkdir(dirname(path),{recursive:true});
  // Exclusive files are acquired before the first screenshot. Never overwrite
  // an earlier recording, log, source/native, or evidence receipt.
  const movie=await open(path,'wx',0o600);
  let log;
  try{log=await open(path+'.capture.jsonl','wx',0o600);}catch(error){await movie.close();throw error;}
  const t0=performance.now(),utc0=new Date().toISOString();
  const state={outputPath:path,logPath:path+'.capture.jsonl',engineeringTest,startedUtc:utc0,
    requestedFps:fps,captures:0,encodedInputFrames:0,captureErrors:0,encoderErrors:0,stopping:false,stopped:false,
    captureInFlight:false,uiInFlight:false,lastCaptureUtc:null,lastError:null,ffmpegPid:null,ffmpegExitCode:null,stderrTail:''};
  let serial=Promise.resolve(),logs=Promise.resolve(),child,closed,loopPromise,stopPromise,wake,deadline=performance.now();
  const stamp=()=>({utc:new Date().toISOString(),monotonicMs:performance.now()-t0});
  const receipt=(kind,detail={})=>{const row={kind,...stamp(),...detail};logs=logs.then(()=>log.write(JSON.stringify(row)+'\n'));return logs;};
  function locked(fn){const result=serial.then(fn);serial=result.catch(()=>{});return result;}
  async function withUi(fn){
    if(state.stopping||state.stopped)throw Error('Recorder stopping/stopped; do not enqueue new UI work');
    return locked(async()=>{
      state.uiInFlight=true;await receipt('ui_begin');
      try{return await fn();}finally{state.uiInFlight=false;await receipt('ui_end');}
    });
  }
  async function launch(width,height,codec){
    width=Math.ceil(width/2)*2;height=Math.ceil(height/2)*2;
    const args=['-hide_banner','-loglevel','warning','-nostdin','-use_wallclock_as_timestamps','1',
      '-probesize','32','-analyzeduration','0','-f','image2pipe','-framerate','1000','-c:v',codec,'-i','pipe:0',
      '-copyts','-start_at_zero','-an','-vf',`scale=${width}:${height}:force_original_aspect_ratio=decrease,pad=${width}:${height}:(ow-iw)/2:(oh-ih)/2,setsar=1`,
      '-c:v','libx264','-preset','ultrafast','-crf','20','-pix_fmt','yuv420p','-x264-params',`fps=${fps}/1:force-cfr=0`,'-enc_time_base','1:1000','-fps_mode','vfr',
      '-flush_packets','1','-cluster_time_limit','1000','-f','matroska','pipe:3'];
    child=spawn(ffmpegPath,args,{stdio:['pipe','ignore','pipe',movie.fd]});state.ffmpegPid=child.pid;state.inputCodec=codec;
    closed=new Promise(resolveClose=>{
      child.once('error',error=>{state.encoderErrors++;state.lastError=String(error);resolveClose({code:null,error:String(error)});});
      child.once('close',(code,signal)=>{state.ffmpegExitCode=code;resolveClose({code,signal});});
    });
    child.stdin.on('error',error=>{state.encoderErrors++;state.lastError=String(error);});
    child.stderr.on('data',bytes=>{state.stderrTail=(state.stderrTail+bytes.toString()).slice(-6000);});
    await receipt('encoder_start',{ffmpegPath,args,width,height,
      captureSource:engineeringTest?'ENGINEERINGHELPERTEST reused existing PNG; not actual app capture; never handoff':'CUA App.getScreenshot({emit:false}) only',
      timing:'Input demux wall-clock PTS, millisecond timebase, VFR; x264 nominal FPS is a codec hint, no output -r or fixed-rate duplication. PTS is ffmpeg read time; capture start/end are logged separately.',
      recovery:'MKV clusters flushed; graceful stop finalizes; abrupt failure may lose final cluster/footer but prior clusters are retained.'});
  }
  async function writePng(bytes){
    if(child.exitCode!==null||state.encoderErrors)throw Error('ffmpeg is not accepting frames: '+state.stderrTail);
    const began=performance.now();
    if(!child.stdin.write(bytes))await once(child.stdin,'drain');
    state.encodedInputFrames++;
    await receipt('frame_streamed',{captureNumber:state.captures,bytes:bytes.byteLength,backpressureMs:performance.now()-began});
  }
  async function capture(final=false){
    const frame=await locked(async()=>{
      if(state.stopping&&!final)return null;
      state.captureInFlight=true;const begin=stamp();
      try{
        const bytes=await app.getScreenshot({emit:false}),end=stamp(),{width,height,codec}=imageInfo(bytes);
        state.captures++;state.lastCaptureUtc=end.utc;
        await receipt('capture',{captureNumber:state.captures,final,begin,end,width,height,codec,bytes:bytes.byteLength});
        return {bytes,width,height,codec};
      }finally{state.captureInFlight=false;}
    });
    if(!frame)return;
    if(!child)await launch(frame.width,frame.height,frame.codec);
    else if(frame.codec!==state.inputCodec)throw Error('CUA screenshot encoding changed during recording');
    await writePng(frame.bytes);
  }
  function pause(ms){return new Promise(resolveWait=>{const timer=setTimeout(()=>{wake=null;resolveWait();},ms);wake=()=>{clearTimeout(timer);wake=null;resolveWait();};});}
  await receipt('start',{fps,outputPath:path,engineeringTest,processPid:process.pid,
    backgroundLifetime:'Unproven in CUA REPL until root probes status across separate REPL calls. Keep returned recorder and done promise in persistent REPL variables.'});
  try{await capture();}catch(error){
    state.captureErrors++;state.lastError=String(error);await receipt('start_error',{error:String(error)});
    child?.stdin.end();if(closed)await closed;await logs;await movie.close();await log.close();throw error;
  }
  deadline=performance.now()+1000/fps;
  loopPromise=(async()=>{
    while(!state.stopping){
      await pause(Math.max(0,deadline-performance.now()));if(state.stopping)break;
      try{await capture();}catch(error){
        state.captureErrors++;state.lastError=String(error);await receipt('capture_error',{error:String(error)});
        if(state.captureErrors>=3||state.encoderErrors){state.stopping=true;break;}
      }
      // Never burst-capture stale deadlines: capture actual current frames only.
      deadline+=1000/fps;
      if(deadline<=performance.now())deadline=performance.now()+1000/fps;
    }
  })().catch(error=>{state.captureErrors++;state.lastError=String(error);state.stopping=true;});
  async function stop(){
    if(stopPromise)return stopPromise;
    stopPromise=(async()=>{
      state.stopping=true;wake?.();await receipt('stop_requested');await loopPromise;
      // A final actual screenshot, never a synthetic hold/duplicate, represents
      // the final visible state and bounds the recording's ending wall time.
      if(!state.encoderErrors)try{await capture(true);}catch(error){state.captureErrors++;state.lastError=String(error);await receipt('final_capture_error',{error:String(error)});}
      child.stdin.end();const result=await closed;
      state.stopped=true;state.stoppedElapsedWallMs=performance.now()-t0;await receipt('stop',{result,state:{...state},elapsedWallMs:state.stoppedElapsedWallMs});
      await logs;await movie.sync();await movie.close();await log.close();return status();
    })();return stopPromise;
  }
  function status(){return {...state,elapsedWallMs:state.stopped?state.stoppedElapsedWallMs:performance.now()-t0};}
  return {status,stop,withUi,done:loopPromise};
}

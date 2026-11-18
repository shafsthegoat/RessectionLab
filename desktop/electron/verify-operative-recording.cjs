'use strict';
const assert = require('node:assert/strict');
const {spawn} = require('node:child_process');
const {once} = require('node:events');
const path = require('node:path');
const fs = require('node:fs');
const {inputScript} = require('./verify-eden-planning-helpers.cjs');
const width = 1920;
const height = 1080;
const fps = 30;
const stepsPerChapter = 100;

// Sample the field only. Transport and legend updates must not count as motion.
function sampleField(bytes, rect) {
  const stride = 4;
  const sample = Buffer.alloc(Math.ceil(rect.width / stride) * Math.ceil(rect.height / stride) * 3);
  let index = 0;
  for (let y = rect.y; y < rect.y + rect.height; y += stride) {
    for (let x = rect.x; x < rect.x + rect.width; x += stride) {
      const start = (y * width + x) * 4;
      sample[index++] = bytes[start];
      sample[index++] = bytes[start + 1];
      sample[index++] = bytes[start + 2];
    }
  }
  return sample;
}

function hasVisibleMotion(current, previous) {
  let changed = 0;
  for (let channel = 0; channel < current.length; channel += 3) {
    const difference = Math.max(
      Math.abs(current[channel] - previous[channel]),
      Math.abs(current[channel + 1] - previous[channel + 1]),
      Math.abs(current[channel + 2] - previous[channel + 2]),
    );
    if (difference >= 8 && ++changed >= 8) return true;
  }
  return false;
}

async function recordingSize(win,js) {
  win.setContentSize(width,height);
  // A hidden native resize and its renderer layout can arrive on different frames.
  const size = await js(`new Promise(resolve=>{
    const deadline=performance.now()+10000;
    function check() {
      if ((innerWidth===${width} && innerHeight===${height}) || performance.now()>deadline) {
        requestAnimationFrame(()=>requestAnimationFrame(()=>resolve([innerWidth,innerHeight])));
      } else requestAnimationFrame(check);
    }
    check();
  })`);
  assert.deepEqual(size,[width,height],'Recording needs the full hidden 1080p content viewport');
  assert.equal(win.isVisible(),false);
}
async function framePixels(win) {
  assert.equal(win.isVisible(),false);
  const shot=await win.webContents.capturePage(undefined,{stayHidden:true,stayAwake:true});
  const size=shot.getSize();
  assert(size.width>=width && size.height>=height && size.width/width===size.height/height,
    'Compositor supplied an undersized or distorted viewport');
  const image=size.width===width?shot:shot.resize({width,height,quality:'best'});
  const bytes=image.toBitmap();
  assert.equal(bytes.length,width*height*4);
  return bytes;
}
async function recordOperativePlayback(win,js,output,intro) {
  const originalSize=win.getContentSize();
  await recordingSize(win,js);
  const chapters=await js("Array.from(document.querySelectorAll('.operative-chapters button:not(:disabled)'),b=>b.getAttribute('aria-label'))");
  const max=Number(await js(`document.querySelector('[aria-label="Operative timeline"]').max`));
  assert(chapters.length>0 && chapters.length===max);
  const rect=await js(`(()=>{const r=document.querySelector('.operative-field-canvas canvas').getBoundingClientRect();
    return {x:Math.ceil(r.x+r.width*.06),y:Math.ceil(r.y+r.height*.14),width:Math.floor(r.width*.88),height:Math.floor(r.height*.72)};})()`);
  assert(rect.x>=0 && rect.y>=0 && rect.x+rect.width<=width && rect.y+rect.height<=height);
  const filename=path.join(output,'operative-walkthrough.mp4');
  const motionFrames = stepsPerChapter * max + 1;
  const introFrames = intro ? fps * 2 : 0;
  const movieChapters = [];
  if (introFrames) movieChapters.push({title:'Selected approach',start:0,end:introFrames});
  chapters.forEach((title, index) => {
    const start = index === 0 ? introFrames : introFrames + stepsPerChapter * index + 1;
    movieChapters.push({title,start,end:introFrames + stepsPerChapter * (index + 1) + 1});
  });
  const metadata = [';FFMETADATA1', ...movieChapters.flatMap(({title,start,end}) => [
    '[CHAPTER]', 'TIMEBASE=1/'+fps, 'START='+start, 'END='+end, 'title='+title,
  ])].join('\n')+'\n';
  const temporary = fs.mkdtempSync(path.join(output,'.operative-recording-'));
  const partial = path.join(temporary,'operative-walkthrough.mp4');
  const encoder=spawn('ffmpeg',['-hide_banner','-loglevel','error','-n',
    '-f','rawvideo','-pixel_format','bgra','-video_size',width+'x'+height,'-framerate',String(fps),'-i','pipe:0',
    '-f','ffmetadata','-i','pipe:3','-map','0:v:0','-map_metadata','1','-map_chapters','1',
    '-an','-c:v','libx264','-crf','16','-preset','fast','-pix_fmt','yuv420p',
    '-vf','scale=out_color_matrix=bt709','-colorspace','bt709','-color_primaries','bt709','-color_trc','bt709',
    '-movflags','+faststart',partial],{stdio:['pipe','ignore','inherit','pipe']});
  const writing = new AbortController();
  const stopWriting = error => writing.abort(error);
  encoder.once('error',stopWriting);
  encoder.stdin.on('error',stopWriting);
  encoder.stdio[3].on('error',stopWriting);
  const closed = new Promise(resolve => encoder.once('close',(code,signal) => {
    stopWriting(new Error('Movie encoder closed with '+(signal ?? code)));
    resolve(code);
  }));
  async function write(bytes) {
    writing.signal.throwIfAborted();
    if (!encoder.stdin.write(bytes)) {
      try {
        await once(encoder.stdin,'drain',{signal:writing.signal});
      } catch (error) {
        throw writing.signal.reason ?? error;
      }
    }
  }
  let prior=null,still=0,lastChapter=-1;
  const perChapter=Object.fromEntries(chapters.map(name=>[name,{frames:0,changedFrames:0,framesWithVisibleMotion:0,longestStillFrames:0}]));
  const started=performance.now();
  try {
    encoder.stdio[3].end(metadata);
    for(let i=0;i<introFrames;i++) await write(intro);
    for(let i=0;i<motionFrames;i++) {
      const position=i/stepsPerChapter;
      await js(inputScript('Operative timeline',position));
      const state=await js(`new Promise(resolve=>requestAnimationFrame(()=>requestAnimationFrame(()=>{
        const panel=document.querySelector('.operative-playback');
        resolve({phase:panel.dataset.operativePhase,progress:Number(panel.dataset.operativeProgress),
          position:Number(panel.querySelector('input[type=range]').value),
          renderingError:panel.querySelector('.operative-field-canvas [role=alert]')?.textContent});
      })))`);
      assert(!state.renderingError, state.renderingError);
      assert(Math.abs(state.position-position)<.0001,'Unexpected timeline value at frame '+i);
      if(i>0) {
        const chapter=Math.ceil(position)-1;
        assert(Math.abs(state.progress-(position-chapter))<.0011,'Stale stage pose at frame '+i);
        if(chapter!==lastChapter) {
          console.log('Recording '+chapters[chapter]+' at '+width+'×'+height);
          lastChapter=chapter;still=0;
        }
      }
      const bytes=await framePixels(win),sample=sampleField(bytes,rect);
      if(prior && i>0) {
        const stats=perChapter[chapters[lastChapter]];
        stats.frames++;
        if(sample.equals(prior)) {stats.longestStillFrames=Math.max(stats.longestStillFrames,++still);}
        else {
          stats.changedFrames++;still=0;
          if (hasVisibleMotion(sample, prior)) stats.framesWithVisibleMotion++;
        }
      }
      prior=sample;
      await write(bytes);
    }
    encoder.stdin.end();
    const status=await closed;
    assert.equal(status,0,'Movie encoder failed');
    for(const [stage,stats] of Object.entries(perChapter))
      assert(stats.framesWithVisibleMotion>stepsPerChapter/3,'Too few visible operative canvas changes in '+stage);
    fs.renameSync(partial,filename);
    return {filename,width,height,fps,introFrames,motionFrames,movieChapters,sampledRect:rect,perChapter,secondsToRecord:(performance.now()-started)/1000};
  } finally {
    if(encoder.exitCode===null && encoder.signalCode===null) encoder.kill();
    await closed;
    fs.rmSync(temporary,{recursive:true,force:true});
    win.setContentSize(...originalSize);
  }
}
module.exports={recordingSize,framePixels,recordOperativePlayback};

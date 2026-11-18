'use strict';
const fs = require('node:fs');
const path = require('node:path');
const {execFileSync} = require('node:child_process');
const assert = require('node:assert/strict');

function subtitleTime(ms) {
  return new Date(ms).toISOString().slice(11,23).replace('.',',');
}

// The 25 fps output holds each captured compositor image. Captions identify
// recorded action boundaries; the encoder never synthesizes intermediate poses.
function encodeStudyReplay({framesDirectory, filename, report, history}) {
  const {states,binding}=report;
  const frames=[], subtitles=[], sampleTimes=new Map();
  let ms=0;
  for (const [i,state] of states.entries()) {
    const action=history[state.position-1];
    const operation=action?.action.operation;
    const final=i===states.length-1;
    const duration=final||!i?4400:operation==='withdraw'||i===2||i===3?1760:640;
    sampleTimes.set(i,(ms+duration/2)/1000);
    const title=i? (operation==='apply'&&action.reasonCode==='TISSUE_ASPIRATION_IN_PROGRESS'?
      'Aspiration in progress · partial action':state.tools[action.action.slot].instrument.toUpperCase()+' · '+operation+' · '+action.status):
      'Recorded access preparation · opaque microscope view · inspection cutaway off';
    const {dimensions,originalTarget,withdrawalComplete}=state.ledger;
    const number=value=>typeof value==='number'?value.toLocaleString('en-US',{maximumFractionDigits:2}):'unknown';
    const amount=key=>number(dimensions[key].cumulative);
    const result=final&&withdrawalComplete?'Withdrawal complete. ':'';
    frames.push("file '"+i+".png'",'duration '+duration/1000);
    subtitles.push(String(i+1),subtitleTime(ms)+' --> '+subtitleTime(ms+duration-1),
      binding.patientId+'    |    Engine boundary '+i+' / '+history.length+'    |    '+title,
      result+'Target removed '+amount('targetRemovedMm3')+' mm³    ·    Non-target removed '+amount('nonTargetRemovedMm3')+
      ' mm³    ·    Whole target remaining '+number(originalTarget.residualMm3)+' mm³','');
    ms+=duration;
  }
  frames.push("file '"+(states.length-1)+".png'");
  fs.writeFileSync(path.join(framesDirectory,'frames.txt'),frames.join('\n')+'\n');
  fs.writeFileSync(path.join(framesDirectory,'titles.srt'),subtitles.join('\n'));
  execFileSync('ffmpeg',['-nostdin','-hide_banner','-loglevel','error','-n','-f','concat','-i','frames.txt',
    '-vf',"fps=25,pad=iw:ih+100:0:0:color=0x101619,subtitles=titles.srt:force_style='FontName=Arial,FontSize=7,MarginV=5,Outline=0,Shadow=0'",
    '-t',String(ms/1000),'-c:v','libx264','-crf','16','-preset','fast','-pix_fmt','yuv420p','-movflags','+faststart',filename],{cwd:framesDirectory});
  const media=JSON.parse(execFileSync('ffprobe',['-v','error','-select_streams','v:0','-show_entries','stream=codec_name,width,height,duration,nb_frames','-of','json',filename]));
  assert.equal(media.streams[0].codec_name,'h264');
  assert(Math.abs(Number(media.streams[0].duration)-ms/1000)<.12);
  // Compare encoded microscope pixels with the compositor at the same boundary.
  // The central crop excludes text, transport, HUD and source-slice panels.
  const {x,y,width,height}=report.fieldCrop;
  const filter='crop='+[width,height,x,y].join(':')+',scale=100:100,format=rgb24';
  function decode(args) {
    return execFileSync('ffmpeg',['-nostdin','-v','error',...args,'-vf',filter,'-frames:v','1','-f','rawvideo','pipe:1']);
  }
  const fieldEvidence=[];
  const initial=decode(['-i',path.join(framesDirectory,'0.png')]);
  for(const index of [0,Math.floor((states.length-1)/2),states.length-1]){
    const original=decode(['-i',path.join(framesDirectory,index+'.png')]);
    const encoded=decode(['-ss',String(sampleTimes.get(index)),'-i',filename]);
    assert.equal(original.length,encoded.length);
    let error=0,changed=0;
    for(let k=0;k<encoded.length;k++){
      error+=Math.abs(original[k]-encoded[k]);
      if(Math.abs(original[k]-initial[k])>12) changed++;
    }
    const meanAbsoluteError=error/encoded.length;
    assert(meanAbsoluteError<6,'Decoded field differs from captured state '+index);
    fieldEvidence.push({position:index,meanAbsoluteError,changedChannelsFromInitial:changed});
  }
  assert(fieldEvidence[1].changedChannelsFromInitial>20,'Expected visible instrument or tissue change within the operative field');
  return {...media.streams[0],fieldEvidence};
}
module.exports={encodeStudyReplay};

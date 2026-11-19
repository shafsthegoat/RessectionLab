'use strict';
const assert = require('node:assert/strict');
const {createHash} = require('node:crypto');
const {execFileSync} = require('node:child_process');
const path = require('node:path');
const fs = require('node:fs');

const hash = value => createHash('sha256').update(value).digest('hex');

// Read the saved arrays in their native C order, independently of the renderer.
function sourceTarget(episode) {
  const python = path.resolve(__dirname, '../../.venv/bin/python');
  const script = `import io,json,sys,zipfile,numpy as np
with zipfile.ZipFile(sys.argv[1]) as z:
    m=json.loads(z.read('manifest.json'))
    a=np.load(io.BytesIO(z.read('arrays.npz')),allow_pickle=False)
    layers=m['array_index']['compartments']
    mapping=m['artifacts']['surgical_toolbox']['mapping']
    shape=a['mri'].shape
    source=np.zeros(shape,dtype=bool)
    owner=np.zeros(shape,dtype=bool)
    for r in mapping:
        mask=a[layers[r['name']]].astype(bool)
        if r['target']: source |= mask
        owner[mask]=r['target']
    print(json.dumps(dict(shape=shape,affine=a['affine'].tolist(),source=np.flatnonzero(source).tolist(),owned=np.flatnonzero(owner).tolist(),voxelMm3=float(abs(np.linalg.det(a['affine'][:3,:3]))))))`;
  const info = JSON.parse(execFileSync(python,['-c',script,episode],{maxBuffer:32*1024*1024}));
  return {...info,source:new Set(info.source),owned:new Set(info.owned)};
}

function tissueState(source,view) {
  assert.equal(view.moved.length,0,'This comparison requires a removal-only recording');
  const removed = new Set(view.removed);
  assert.equal(removed.size,view.removed.length);
  const residual = [...source.source].filter(i=>!removed.has(i));
  const ownedRemoved = [...removed].filter(i=>source.owned.has(i));
  assert(Math.abs(ownedRemoved.length*source.voxelMm3-view.metrics.targetRemovedMm3)<1e-4);
  assert(Math.abs(residual.length*source.voxelMm3-view.originalTargetResidualMm3)<1e-4);
  return {removed:[...removed].sort((a,b)=>a-b),residual,ownedRemoved:ownedRemoved.length};
}

// Compare the submitted mesh extents with RAS coordinates from the source grid.
function verifyResultBounds(source,removed,remaining,drawn) {
  const [,ny,nz]=source.shape;
  const voxel=i=>[Math.floor(i/(ny*nz)),Math.floor(i/nz)%ny,i%nz];
  const present=new Set(removed);
  const ras=p=>source.affine.slice(0,3).map(row=>row[3]+row[0]*p[0]+row[1]*p[1]+row[2]*p[2]);
  const bounds=positions=>{
    const low=[Infinity,Infinity,Infinity],high=[-Infinity,-Infinity,-Infinity];
    for(const p of positions)ras(p).forEach((n,k)=>{low[k]=Math.min(low[k],n);high[k]=Math.max(high[k],n);});
    return [low,high];
  };
  const boundary=[];
  // Binary marching tetrahedra cross the midpoint of each incident edge.
  for(const i of removed){
    const p=voxel(i);
    for(let mask=1;mask<8;mask++)for(const sign of [-1,1]){
      const step=[0,1,2].map(k=>sign*((mask>>k)&1)),q=p.map((v,k)=>v+step[k]);
      if(q.every((v,k)=>v>=0&&v<source.shape[k])&&present.has((q[0]*ny+q[1])*nz+q[2]))continue;
      boundary.push(p.map((v,k)=>v+step[k]/2));
    }
  }
  const result={actual:drawn};
  for(const [name,points] of [['cavity',boundary],['residual',remaining.map(voxel)]]){
    if(!points.length){assert.equal(drawn[name],null);continue;}
    const expected=bounds(points);
    const error=expected.flatMap((row,i)=>row.map((n,k)=>Math.abs(n-drawn[name][i][k])));
    const limit=name==='cavity'?[0,0,0]:source.affine.slice(0,3).map(row=>row.slice(0,3).reduce((s,n)=>s+Math.abs(n),0));
    assert(error.every((n,k)=>n<=limit[k%3]+1e-5),name+' mesh lies outside its source-frame envelope');
    result[name]={expected,maximumErrorMm:Math.max(...error)};
  }
  return result;
}

function digestCells(cells) {
  const bytes=Buffer.alloc(cells.length*4);
  cells.forEach((cell,i)=>bytes.writeUInt32LE(cell,4*i));
  return {cells:cells.length,sha256:hash(bytes)};
}

function encodeWalkthrough(output,samples,inputs,ledger,inspectionFrames) {
  const {patient,route,acquisitions,target,generator}=inputs;
  const endpoint=ledger.dimensions.targetRemovedMm3.cumulative.toFixed(1)+' mm³ target removed, '+ledger.originalTarget.residualMm3.toFixed(1)+' mm³ remaining';
  const sections=[
    {image:'patient-overview.png',seconds:3.6,title:'Patient reconstruction',note:patient+'  |  '+acquisitions+'  |  '+target},
    {image:'selected-recorded-approach.png',seconds:3.6,title:'Selected approach',note:route+'  |  Planning geometry in patient space'},
    ...samples.sort((a,b)=>a.position-b.position).map(s=>({...s,seconds:0.8,title:'Recorded engine actions',
      note:s.time.toFixed(2)+' simulated seconds   |   '+s.operation+'   |   '+generator})),
    ...inspectionFrames,
    {image:'procedure-final.png',seconds:5.2,title:'Final state inspection',note:endpoint+'  |  '+(ledger.withdrawalComplete?'Withdrawal complete':'Instrument remains in the field')},
  ];
  const temp=fs.mkdtempSync(path.join(output,'.walkthrough-'));
  const file=path.join(output,'patient-engine-walkthrough.mp4');
  const font='/System/Library/Fonts/Supplemental/Arial.ttf';assert(fs.existsSync(font));
  let timestamp=0;
  try {
    const frames=[];
    for(const [i,s] of sections.entries()) {
      const image=String(i)+'.png';
      const escape=t=>String(t).replace(/[\\':%]/g,' ');
      const vf=`scale=1600:900:force_original_aspect_ratio=decrease,pad=1600:1000:(ow-iw)/2:(900-ih)/2:color=0x10191e,setsar=1,drawtext=fontfile='${font}':text='${escape(s.title)}':fontsize=32:fontcolor=white:x=40:y=919,drawtext=fontfile='${font}':text='${escape(s.note)}':fontsize=21:fontcolor=0x9bd5ce:x=40:y=963`;
      execFileSync('ffmpeg',['-nostdin','-v','error','-i',path.join(output,s.image),'-vf',vf,'-frames:v','1','-threads','2',path.join(temp,image)]);
      frames.push("file '"+image+"'",'duration '+s.seconds);
      s.startSeconds=timestamp;timestamp+=s.seconds;
    }
    frames.push("file '"+(sections.length-1)+".png'");fs.writeFileSync(path.join(temp,'frames.txt'),frames.join('\n')+'\n');
    execFileSync('ffmpeg',['-nostdin','-v','error','-y','-f','concat','-safe','1','-i','frames.txt','-vf','fps=25,format=yuv420p','-t',String(timestamp),
      '-c:v','libx264','-threads','4','-crf','17','-preset','fast','-movflags','+faststart',file],{cwd:temp});
  }finally {fs.rmSync(temp,{recursive:true,force:true});}
  const encoded=JSON.parse(execFileSync('ffprobe',['-v','error','-select_streams','v:0','-show_entries','stream=width,height,duration,codec_name','-of','json',file])).streams[0];
  assert(Math.abs(Number(encoded.duration)-timestamp)<0.09);
  const checks=[];
  for(const index of [...new Set([0,2,Math.floor(sections.length/2),sections.length-2,sections.length-1])]) {
    const sample=sections[index],time=sample.startSeconds+sample.seconds/2;
    const rgb=(args,filter)=>execFileSync('ffmpeg',['-nostdin','-v','error',...args,'-vf',filter,'-frames:v','1','-f','rawvideo','pipe:1']);
    const ref=rgb(['-i',path.join(output,sample.image)],'scale=1600:900:force_original_aspect_ratio=decrease,pad=1600:900:(ow-iw)/2:(oh-ih)/2:color=0x10191e,scale=128:72,format=rgb24');
    const decoded=rgb(['-ss',String(time),'-i',file],'crop=1600:900:0:0,scale=128:72,format=rgb24');
    assert.equal(ref.length,decoded.length);const mae=ref.reduce((sum,n,k)=>sum+Math.abs(n-decoded[k]),0)/ref.length;
    assert(mae<8,'Encoded field differs from '+sample.image+' by '+mae);checks.push({time,image:sample.image,mae});
  }
  return {file:path.basename(file),...encoded,checks,sha256:hash(fs.readFileSync(file)),chapters:sections.map(({title,startSeconds,seconds})=>({title,startSeconds,seconds}))};
}
module.exports={verifyResultBounds,digestCells,sourceTarget,tissueState,encodeWalkthrough};

// Read each painted cell at its center. Outlines and projected instrument
// glyphs are separate from the label painted at the source-grid coordinate.
async function verifySection(contents,id,state,source) {
  const section=await contents.executeJavaScript(`(()=>{
    const c=document.querySelector('canvas[data-engine-inspection-state="'+${JSON.stringify(id)}+'"]');if(!c)return null;
    const [w,h]=JSON.parse(c.dataset.sliceSize),dw=c.width/w,dh=c.height/h,ctx=c.getContext('2d');
    const pixels=ctx.getImageData(0,0,c.width,c.height).data,classes=[];
    for(let y=0;y<h;y++)for(let x=0;x<w;x++){
      const i=4*(Math.floor((y+.5)*dh)*c.width+Math.floor((x+.5)*dw)),r=pixels[i],g=pixels[i+1],b=pixels[i+2];
      classes.push(r===85&&g===57&&b===22?2:g-r>60&&b>g?1:0);
    }
    return {axis:+c.dataset.sliceAxis,index:+c.dataset.sliceIndex,origin:JSON.parse(c.dataset.sliceOrigin),size:[w,h],
      caseId:c.dataset.sourceCase,sceneId:c.dataset.sceneId,classes};
  })()`);
  assert(section,'Missing painted section '+id);
  const {axis,index,origin,size,classes}=section,[u,v]=[0,1,2].filter(a=>a!==axis);
  const removed=new Set(state.removed),sectionCavity=[],sectionTarget=[];
  for(let y=0;y<size[1];y++)for(let x=0;x<size[0];x++){
    const p=[0,0,0];p[axis]=index;p[u]=origin[0]+x;p[v]=origin[1]+size[1]-1-y;
    const i=(p[0]*source.shape[1]+p[1])*source.shape[2]+p[2];
    const expected=removed.has(i)?2:source.owned.has(i)?1:0;
    assert.equal(classes[y*size[0]+x],expected,'Rendered cell differs at '+i+' in state '+id);
    if(expected===2)sectionCavity.push(i);
    if(expected===1)sectionTarget.push(i);
  }
  delete section.classes;
  return {...section,cavityOnSlice:digestCells(sectionCavity.sort((a,b)=>a-b)),targetOnSlice:digestCells(sectionTarget.sort((a,b)=>a-b)),
    gridRemoval:digestCells(state.removed),gridResidual:digestCells(state.residual)};
}
module.exports.verifySection=verifySection;

// The application supplies the complete 1× replay. This edit keeps its
// authenticated time and outcome captions visible at the declared viewing speed.
function continuousWalkthrough(output,full,short) {
  const source=path.join(output,'observed-engine-replay.mp4');
  const file=path.join(output,'patient-engine-continuous.mp4');
  const chapters=short.walkthrough.chapters,opening=chapters[2].startSeconds;
  const ending=chapters.find(c=>c.title==='Final result inspection')??chapters.at(-1),recordedSeconds=Number(full.movie.duration);
  const inspectionSeconds=Number(short.walkthrough.duration)-ending.startSeconds;
  const speed=recordedSeconds>90?4:recordedSeconds>45?2:1;
  const engineSeconds=recordedSeconds/speed;
  const imageMovie=path.join(output,short.walkthrough.file);
  const filter=[
    `[0:v]trim=end=${opening},setpts=PTS-STARTPTS,fps=50[a]`,
    `[1:v]setpts=(PTS-STARTPTS)/${speed},fps=50,scale=1600:900:force_original_aspect_ratio=decrease,pad=1600:1000:(ow-iw)/2:(900-ih)/2:color=0x10191e,setsar=1,`+
      "drawtext=fontfile='/System/Library/Fonts/Supplemental/Arial.ttf':text='Recorded engine procedure':fontsize=32:fontcolor=white:x=40:y=919,"+
      "drawtext=fontfile='/System/Library/Fonts/Supplemental/Arial.ttf':text='" + speed + "× simulated time   |   Target and cavity from the observed engine state':fontsize=21:fontcolor=0x9bd5ce:x=40:y=963[b]",
    `[0:v]trim=start=${ending.startSeconds},setpts=PTS-STARTPTS,fps=50[c]`,
    '[a][b][c]concat=n=3:v=1:a=0,format=yuv420p[v]'
  ].join(';');
  execFileSync('ffmpeg',['-nostdin','-v','error','-y','-i',imageMovie,'-i',source,'-filter_complex_threads','2','-filter_complex',filter,'-map','[v]','-c:v','libx264','-threads','4','-crf','17','-preset','fast','-movflags','+faststart',file]);
  const meta=JSON.parse(execFileSync('ffprobe',['-v','error','-select_streams','v:0','-show_entries','stream=width,height,duration,codec_name','-of','json',file])).streams[0];
  assert(Math.abs(Number(meta.duration)-(opening+engineSeconds+inspectionSeconds))<.09);
  const evidence=[];
  const witnessDir=path.join(output,'.movie-checks');
  const frames=full.exportFrames;
  for(const filename of fs.readdirSync(witnessDir)){
    const i=Number(path.basename(filename,'.png')),f=frames[i],next=frames[i+1];if(!next||next.time-f.time<.04)continue;
    const time=(f.time+next.time)/2;
    const rgb=(args,filter)=>execFileSync('ffmpeg',['-nostdin','-v','error',...args,'-vf',filter,'-frames:v','1','-f','rawvideo','pipe:1']);
    const ref=rgb(['-i',path.join(witnessDir,filename)],'scale=128:72,format=rgb24');
    const decoded=rgb(['-ss',String(time),'-i',source],`crop=${f.rect.width}:${f.rect.height}:0:0,scale=128:72,format=rgb24`);
    assert.equal(ref.length,decoded.length);
    const mae=ref.reduce((sum,n,k)=>sum+Math.abs(n-decoded[k]),0)/ref.length;
    assert(mae<8);evidence.push({index:i,time,state:f.id,decodedMae:mae});
  }
  assert(evidence.length>=3&&evidence.some(e=>e.time>frames.at(-1).time*.8),'Check early, intermediate and late decoded engine frames');
  fs.rmSync(witnessDir,{recursive:true,force:true});
  return {file:path.basename(file),...meta,sha256:hash(fs.readFileSync(file)),encodedSource:hash(fs.readFileSync(source)),evidence,
    chapters:[{title:'Patient reconstruction',at:0},{title:'Selected approach',at:chapters[1].startSeconds},
      {title:'Recorded actions at '+speed+'×',at:opening},{title:'Final inspection',at:opening+engineSeconds}]};
}
module.exports.continuousWalkthrough=continuousWalkthrough;

import assert from 'node:assert/strict';
import {test} from 'node:test';
import {CINEMATIC_BRICK,CINEMATIC_SKIP_LIMIT,CINEMATIC_STEP_WORK,CINEMATIC_TAG_FLOOR,cinematicFieldSteps,cinematicMajorantSteps,cinematicMaterial,cinematicRamp,cinematicSignalSteps,cinematicTagClass,cinematicTagSteps} from './cinematicTags.ts';

function finishCinematicSteps(steps){
  let result=steps.next();
  while(!result.done)result=steps.next();
  return result.value;
}
const cinematicMajorants=(...args)=>finishCinematicSteps(cinematicMajorantSteps(...args));

const index=(shape,x,y,z)=>(x*shape[1]+y)*shape[2]+z;
/** Two tubes along the first axis on an anisotropic, non-brick-aligned grid. */
function phantom({artery=400,vein=220,background=40,bone=900,shape=[27,19,11]}={}){
  const count=shape[0]*shape[1]*shape[2];
  const signal=new Float32Array(count).fill(background),arteries=new Uint8Array(count),veins=new Uint8Array(count);
  for(let x=0;x<shape[0];x++)for(let y=0;y<shape[1];y++)for(let z=0;z<shape[2];z++){
    const i=index(shape,x,y,z);
    if((y-5)**2+(z-5)**2<=4){signal[i]=artery;arteries[i]=1;}
    else if((y-13)**2+(z-5)**2<=4){signal[i]=vein;veins[i]=1;}
    else if(y===shape[1]-1)signal[i]=bone;
  }
  return {shape,signal,arteries,veins};
}
const build=(signal,shape,layers)=>{
  const stats=finishCinematicSteps(cinematicSignalSteps(signal,shape));
  return {stats,tags:finishCinematicSteps(cinematicTagSteps(signal,shape,layers,stats.range))};
};

test('vessel kinds map to material classes and other kinds stay untagged',()=>{
  assert.deepEqual(['artery','vessel','vessels','vein','sinus','bone','tumor',undefined].map(cinematicTagClass),[0,0,0,1,1,null,null,null]);
});

test('soft tags are smooth, two-channel and leave every source array unchanged',()=>{
  const {shape,signal,arteries,veins}=phantom();
  const before=[signal.slice(),arteries.slice(),veins.slice()];
  const {stats,tags}=build(signal,shape,[{cls:0,source:{kind:'mask',mask:arteries}},{cls:1,source:{kind:'mask',mask:veins}}]);
  assert.deepEqual([signal,arteries,veins],before);
  assert.equal(tags.channels,2);assert.deepEqual(tags.channelOf,[0,1]);
  assert.equal(tags.data.length,signal.length*2);
  assert.deepEqual(stats.bricks,shape.map(n=>Math.ceil(n/CINEMATIC_BRICK)));
  const at=(x,y,z,c)=>tags.data[index(shape,x,y,z)*2+c];
  // Deep inside is saturated, the wall is intermediate and three voxels out is clear.
  assert.ok(at(13,5,5,0)>=250);
  assert.ok(at(13,7,5,0)>128&&at(13,7,5,0)<255,'wall voxel is soft');
  assert.ok(at(13,8,5,0)>0&&at(13,8,5,0)<128,'first outside voxel is soft');
  assert.equal(at(13,10,5,0),0);
  assert.equal(at(13,5,5,1),0,'artery voxels carry no vein tag');
  // Symmetric in the cross-section despite the anisotropic grid.
  assert.equal(at(13,7,5,0),at(13,3,5,0));assert.equal(at(13,5,7,0),at(13,5,3,0));
});

test('ramps are the core and shell midpoint per class, in the units of the acquisition',()=>{
  const {shape,signal,arteries,veins}=phantom();
  const layers=[{cls:0,source:{kind:'mask',mask:arteries}},{cls:1,source:{kind:'mask',mask:veins}}];
  const {tags}=build(signal,shape,layers);
  const [artery,vein]=tags.classes;
  assert.ok(Math.abs(artery.coreMedian-400)<2&&Math.abs(artery.shellMedian-40)<2);
  assert.ok(Math.abs(vein.coreMedian-220)<2&&Math.abs(vein.shellMedian-40)<2);
  const mid=r=>(r[0]+r[1])/2;
  assert.ok(Math.abs(mid(artery.ramp)-220)<2);assert.ok(Math.abs(mid(vein.ramp)-130)<2);
  assert.ok(artery.ramp[1]-artery.ramp[0]<.35*(400-40),'narrow ramp');
  // A bright-blood MR acquisition in arbitrary units with no bone: same rule.
  const mr=phantom({artery:3.1,vein:2.2,background:1,bone:1});
  const scaled=build(mr.signal,mr.shape,[{cls:0,source:{kind:'mask',mask:mr.arteries}}]).tags.classes[0];
  assert.ok(Math.abs(mid(scaled.ramp)-2.05)<.01);
  assert.equal(scaled.tagBounded,false);
});

test('an absent class is empty and a single class uses one channel',()=>{
  const {shape,signal,veins}=phantom();
  const {tags}=build(signal,shape,[{cls:1,source:{kind:'mask',mask:veins}}]);
  assert.equal(tags.channels,1);assert.deepEqual(tags.channelOf,[-1,0]);
  assert.equal(tags.data.length,signal.length);
  assert.equal(tags.classes[0].present,false);assert.equal(tags.classes[0].coreMedian,null);
  assert.ok(tags.classes[0].ramp[0]>400,'an absent class never becomes opaque');
  const none=build(signal,shape,[]).tags;
  assert.equal(none.channels,0);assert.equal(none.data,null);
  const emptyMask=build(signal,shape,[{cls:0,source:{kind:'mask',mask:new Uint8Array(signal.length)}}]).tags;
  assert.equal(emptyMask.classes[0].present,false);
  assert.equal(emptyMask.data.every(v=>v===0),true);
});

test('a float probability replaces the smoothed mask without being changed',()=>{
  const {shape,signal,arteries}=phantom();
  const probability=Float32Array.from(arteries,(v,i)=>v?.97:signal[i]===40&&i%3===0?.2:0);
  const copy=probability.slice();
  const {tags}=build(signal,shape,[{cls:0,source:{kind:'probability',probability}}]);
  assert.deepEqual(probability,copy);
  for(const i of [index(shape,13,5,5),index(shape,13,8,5),index(shape,2,0,0)])
    assert.equal(tags.data[i],Math.round(Math.min(1,probability[i])*255),'probability is used unsmoothed');
  assert.ok(Math.abs(tags.classes[0].coreMedian-400)<2);
  const labels=Uint16Array.from(arteries,v=>v?7:0);
  const fromLabels=build(signal,shape,[{cls:0,source:{kind:'labels',labels,value:7}}]).tags;
  const fromMask=build(signal,shape,[{cls:0,source:{kind:'mask',mask:arteries}}]).tags;
  assert.deepEqual(fromLabels.data,fromMask.data);
});

test('without core-shell contrast the tag alone bounds vessel opacity',()=>{
  const {shape,arteries}=phantom();
  const flat=new Float32Array(arteries.length).fill(100);
  const {tags}=build(flat,shape,[{cls:0,source:{kind:'mask',mask:arteries}}]);
  assert.equal(tags.classes[0].tagBounded,true);
  assert.equal(cinematicRamp(...tags.classes[0].ramp,100),1);
});

/** Filtered reads as the shader makes them: voxel centres at integers, edges clamped. */
const clampAt=(shape,values,channels,channel)=>(x,y,z)=>values[index(shape,Math.max(0,Math.min(shape[0]-1,x)),Math.max(0,Math.min(shape[1]-1,y)),Math.max(0,Math.min(shape[2]-1,z)))*channels+channel];
function trilinear(at,p){
  const i=p.map(Math.floor),f=p.map((v,a)=>v-i[a]);let sum=0;
  for(const dx of [0,1])for(const dy of [0,1])for(const dz of [0,1])
    sum+=(dx?f[0]:1-f[0])*(dy?f[1]:1-f[1])*(dz?f[2]:1-f[2])*at(i[0]+dx,i[1]+dy,i[2]+dz);
  return sum;
}
const spline=t=>[(1-t)**3/6,(3*t**3-6*t**2+4)/6,(-3*t**3+3*t**2+3*t+1)/6,t**3/6];
function cubic(at,p){
  const i=p.map(Math.floor),w=p.map((v,a)=>spline(v-i[a]));let sum=0;
  for(let a=0;a<4;a++)for(let b=0;b<4;b++)for(let c=0;c<4;c++)sum+=w[0][a]*w[1][b]*w[2][c]*at(i[0]+a-1,i[1]+b-1,i[2]+c-1);
  return sum;
}
/** The dense medium of the shader at filtered intensity v and filtered tags. */
function denseAt(material,v,tags){
  const gate=tags.map(t=>cinematicRamp(...material.gate,t/255));
  const lumen=[0,1].map(c=>Math.max(cinematicRamp(...material.ramps[c],v),material.floor[2]*cinematicRamp(material.floor[0],material.floor[1],tags[c]/255)));
  const vessel=material.vesselSigma*Math.max(gate[0]*lumen[0],gate[1]*lumen[1]);
  const bone=material.bone.dense?material.bone.sigma*cinematicRamp(...material.bone.ramp,v)*(1-Math.max(...gate)):0;
  return vessel+bone;
}
function* positionsIn(shape,brick,count,seed){
  let state=seed;const random=()=>(state=(state*1103515245+12345)%2147483648)/2147483648;
  for(let n=0;n<count;n++)yield brick.map((b,a)=>Math.min(shape[a]-.5,Math.max(-.5,b*CINEMATIC_BRICK-.5+random()*CINEMATIC_BRICK)));
}

test('majorants bound the filtered medium anywhere in a brick for cubic and trilinear reads',()=>{
  const {shape,signal,arteries,veins}=phantom();
  const {stats,tags}=build(signal,shape,[{cls:0,source:{kind:'mask',mask:arteries}},{cls:1,source:{kind:'mask',mask:veins}}]);
  const value=clampAt(shape,signal,1,0),tag=[clampAt(shape,tags.data,2,0),clampAt(shape,tags.data,2,1)];
  for(const [ct,boneWeight] of [[true,1],[true,.03],[false,0]]){
    const material=cinematicMaterial(tags,{ct,boneWeight,window:[0,500]});
    const majorants=cinematicMajorants(stats,tags,material);
    const [bx,by,bz]=stats.bricks;
    assert.equal(majorants.length,bx*by*bz*4);
    let tight=0;
    for(let x=0;x<bx;x++)for(let y=0;y<by;y++)for(let z=0;z<bz;z++){
      const brick=(x*by+y)*bz+z,bound=majorants[4*brick];
      let most=0;
      for(const p of positionsIn(shape,[x,y,z],24,brick+1)){
        const t=tag.map(read=>trilinear(read,p));
        for(const v of [cubic(value,p),trilinear(value,p)]){const d=denseAt(material,v,t);most=Math.max(most,d);assert.ok(d<=bound+1e-6,`brick ${[x,y,z]} ${d} > ${bound}`);}
      }
      if(bound>0&&most>.5*bound)tight++;
    }
    assert.ok(tight>0,'bounds are reached, not just safe');
    if(!ct)assert.equal(material.bone.sigma,0,'MR has no bone');
    assert.equal(material.bone.dense,ct&&boneWeight>=.5);
    // An empty brick's skip radius names a cube of bricks without dense
    // matter, and its faint bound covers every brick in that cube.
    for(let x=0;x<bx;x++)for(let y=0;y<by;y++)for(let z=0;z<bz;z++){
      const brick=(x*by+y)*bz+z,radius=majorants[4*brick+1];
      assert.ok(radius>=0&&radius<=CINEMATIC_SKIP_LIMIT);
      assert.ok(majorants[4*brick+3]>=majorants[4*brick+2]);
      for(let i=Math.max(0,x-radius);i<=Math.min(bx-1,x+radius);i++)for(let j=Math.max(0,y-radius);j<=Math.min(by-1,y+radius);j++)
        for(let k=Math.max(0,z-radius);k<=Math.min(bz-1,z+radius);k++)if(radius){
          const other=(i*by+j)*bz+k;
          assert.equal(majorants[4*other],0);
          assert.ok(majorants[4*brick+3]>=majorants[4*other+2]);
        }
    }
  }
});

test('brick maxima cover exactly the cubic support, two voxels past each face',()=>{
  const shape=[24,8,8],signal=new Float32Array(24*8*8);
  signal[index(shape,10,4,4)]=100;
  const {maxValue,bricks}=finishCinematicSteps(cinematicSignalSteps(signal,shape));
  const along=x=>maxValue[(x*bricks[1]+1)*bricks[2]+1];
  // Voxel 10 lies in brick 2 and within two voxels of brick 3 (voxels 12-15), not of bricks 1 or 4.
  assert.deepEqual([0,1,2,3,4,5].map(along),[0,0,100,100,0,0]);
});

test('the CT glass field points along the world gradient and its brick bounds cover faint matter',()=>{
  const shape=[12,10,8],count=12*10*8,signal=new Float32Array(count);
  // Bone slab beyond x = 6 with a sharp face; voxel spacing .5, .8 and 1.2 mm with the first axis flipped.
  for(let x=0;x<12;x++)for(let y=0;y<10;y++)for(let z=0;z<8;z++)signal[index(shape,x,y,z)]=x>=6?900:20;
  const M=[-2,0,0, 0,1.25,0, 0,0,1/1.2];
  const field=finishCinematicSteps(cinematicFieldSteps(signal,shape,M));
  const decode=i=>[0,1,2].map(c=>(field.data[4*i+c]-127)/127);
  const face=decode(index(shape,6,5,4));
  assert.ok(Math.hypot(...face)>.95,'a 900 HU step is a full glass surface');
  assert.ok(face[0]<-.95&&Math.abs(face[1])<.02&&Math.abs(face[2])<.02,'the world gradient of a +x step is -x under the flipped axis');
  assert.equal(decode(index(shape,2,5,4)).every(c=>c===0),true,'flat matter has no surface');
  assert.equal(field.data[4*index(shape,9,5,4)+3],255);assert.equal(field.data[4*index(shape,2,5,4)+3],0);
  const stats=finishCinematicSteps(cinematicSignalSteps(signal,shape));
  for(const boneWeight of [.03,.3]){
    const material=cinematicMaterial(null,{ct:true,boneWeight,window:[0,500]});
    const majorants=cinematicMajorants(stats,null,material,field);
    const read=c=>clampAt(shape,field.data,4,c),[gx,gy,gz,ga]=[0,1,2,3].map(read);
    const [bx,by,bz]=stats.bricks;
    for(let x=0;x<bx;x++)for(let y=0;y<by;y++)for(let z=0;z<bz;z++){
      const brick=(x*by+y)*bz+z;
      for(const p of positionsIn(shape,[x,y,z],32,brick+7)){
        const s=Math.min(1,Math.hypot(...[gx,gy,gz].map(r=>(trilinear(r,p)-127)/127))),a=trilinear(ga,p)/255;
        // Facing and tags only lower faint matter; the bound must hold edge on.
        const sigma=material.faint.soft*s*(1-a)+material.faint.bone*a*Math.max(s,.04);
        assert.ok(sigma<=majorants[4*brick+2]+1e-6,`faint ${sigma} > ${majorants[4*brick+2]}`);
      }
    }
  }
});

test('a confident tag core keeps a connected lumen where partial volume dims it, and nothing outside tags',()=>{
  // A one-voxel artery whose intensity dips to a third of the contrast midway.
  const shape=[30,9,9],count=30*9*9,signal=new Float32Array(count).fill(40),mask=new Uint8Array(count);
  // The class ramp comes from a thick vessel of the same contrast elsewhere.
  const wide=new Uint8Array(count),thick=new Float32Array(count).fill(40);
  for(let x=0;x<30;x++)for(let y=2;y<7;y++)for(let z=2;z<7;z++){const i=index(shape,x,y,z);wide[i]=1;thick[i]=400;}
  for(let x=0;x<30;x++){const i=index(shape,x,4,4);mask[i]=1;signal[i]=x>=12&&x<18?160:400;}
  const reference=build(thick,shape,[{cls:0,source:{kind:'mask',mask:wide}}]).tags;
  const {tags}=build(signal,shape,[{cls:0,source:{kind:'mask',mask}}]);
  tags.classes[0].ramp=reference.classes[0].ramp;
  const material=cinematicMaterial(tags,{ct:true,boneWeight:0,window:[0,500]});
  const value=clampAt(shape,signal,1,0),tag=clampAt(shape,tags.data,1,0);
  const centre=x=>{const p=[x,4,4];return denseAt(material,cubic(value,p),[trilinear(tag,p),0])/material.vesselSigma;};
  const withoutFloor=x=>{const p=[x,4,4],v=cubic(value,p);return cinematicRamp(...material.gate,trilinear(tag,p)/255)*cinematicRamp(...material.ramps[0],v);};
  assert.ok(withoutFloor(15)<.5,'the intensity ramp alone breaks the lumen');
  for(let x=1;x<29;x++)assert.ok(centre(x)>=.5,`lumen kept at ${x}`);
  assert.ok(CINEMATIC_TAG_FLOOR[2]<1,'the floor stays below the intensity-defined lumen');
  // Two voxels from the tag, with bright intensity, nothing is drawn.
  for(const p of [[15,6,4],[15,4,7],[15,1,1]])assert.equal(denseAt(material,cubic(value,p),[trilinear(tag,p),0]),0);
});

test('bricks far from dense matter receive a skip radius bounded by the nearest occupied brick',()=>{
  const shape=[64,48,40],count=64*48*40,signal=new Float32Array(count).fill(30),mask=new Uint8Array(count);
  for(let x=2;x<6;x++)for(let y=2;y<6;y++)for(let z=2;z<6;z++){const i=index(shape,x,y,z);signal[i]=400;mask[i]=1;}
  const {stats,tags}=build(signal,shape,[{cls:0,source:{kind:'mask',mask}}]);
  const data=cinematicMajorants(stats,tags,cinematicMaterial(tags,{ct:false,boneWeight:0,window:[0,500]}));
  const [,by,bz]=stats.bricks,at=(x,y,z)=>data[4*((x*by+y)*bz+z)+1];
  assert.ok(data[0]>0,'the vessel brick is dense');assert.equal(at(0,0,0),0);
  const faint=Math.max(...data.filter((_,i)=>i%4===2));
  assert.ok(faint>0&&faint<.01,'faint MR context has a small bound');
  assert.equal(at(15,11,9),CINEMATIC_SKIP_LIMIT);
  // The vessel's filter support reaches brick 2; brick 3 is one brick clear of it.
  assert.equal(at(3,0,0),1);assert.equal(at(4,1,1),2);
});

test('a CT brick without any glass surface carries no faint bound',()=>{
  // Uniform air: no gradient anywhere, so every encoded vector is exactly 127.
  const shape=[13,10,9],signal=new Float32Array(13*10*9).fill(-1000);
  const field=finishCinematicSteps(cinematicFieldSteps(signal,shape,[1,0,0,0,1,0,0,0,1]));
  assert.ok(field.data.every((v,i)=>i%4===3?v===0:v===127));
  assert.ok(field.maxSurface.every(v=>v===0),'no surface bound');
  const stats=finishCinematicSteps(cinematicSignalSteps(signal,shape));
  for(const boneWeight of [.03,1]){
    const majorants=cinematicMajorants(stats,null,cinematicMaterial(null,{ct:true,boneWeight,window:[0,500]}),field);
    assert.ok(majorants.every((v,i)=>i%4!==2&&i%4!==3||v===0),'no faint steps in air');
  }
  // A soft interface still bounds its bricks with the rounding margin; bricks out of its reach stay clear.
  for(let x=6;x<13;x++)for(let y=0;y<10;y++)for(let z=0;z<9;z++)signal[index(shape,x,y,z)]=60;
  const one=finishCinematicSteps(cinematicFieldSteps(signal,shape,[1,0,0,0,1,0,0,0,1]));
  assert.ok(one.maxSurface.some(v=>v>0)&&one.maxSurface.some(v=>v===0));
  for(const v of one.maxSurface)if(v>0)assert.ok(v>=2/255);
});

/** Bone dilated by two voxels along each axis in turn, as two one-voxel passes per axis. */
function referenceNear(bone,shape){
  const [nx,ny,nz]=shape;let near=Uint8Array.from(bone);
  for(const [stride,size] of [[ny*nz,nx],[nz,ny],[1,nz]])for(let pass=0;pass<2;pass++){
    const source=near;near=source.slice();
    for(let i=0;i<source.length;i++){
      if(source[i])continue;const c=Math.floor(i/stride)%size;
      if((c>0&&source[i-stride])||(c<size-1&&source[i+stride]))near[i]=1;
    }
  }
  return near;
}

test('soft glass is removed within two voxels of bone, exactly as two one-voxel dilations per axis',()=>{
  const shape=[11,9,13],count=11*9*13,signal=new Float32Array(count);
  let state=7;const random=()=>(state=(state*1103515245+12345)%2147483648)/2147483648;
  for(let i=0;i<count;i++)signal[i]=random()<.003?900:random()<.5?60:-1000;
  const field=finishCinematicSteps(cinematicFieldSteps(signal,shape,[1,0,0,0,1,0,0,0,1]));
  const near=referenceNear(Uint8Array.from({length:count},(_,i)=>field.data[4*i+3]>0?1:0),shape);
  let suppressed=0;
  for(let i=0;i<count;i++)if(near[i]&&!field.data[4*i+3]){
    assert.deepEqual([...field.data.slice(4*i,4*i+3)],[127,127,127]);suppressed++;
  }
  assert.ok(suppressed>0);
  // Away from bone the surface survives somewhere.
  assert.ok([...Array(count).keys()].some(i=>!near[i]&&field.data[4*i]!==127));
});

test('preparation yields in proportion to voxels visited, whatever the grid shape',()=>{
  const count=1<<19;
  const yields=shape=>{
    const signal=new Float32Array(count);
    for(let i=0;i<count;i++)signal[i]=i%97<3?900:i%5?40:-900;
    let n=0;const field=cinematicFieldSteps(signal,shape,[1,0,0,0,1,0,0,0,1]);
    while(!field.next().done)n++;
    let m=0;const stats=cinematicSignalSteps(signal,shape);
    while(!stats.next().done)m++;
    return {field:n,signal:m};
  };
  // One slab, many slabs and an even split: the field visits every voxel in
  // at least nine full passes, so it yields at least that many times per step's work.
  const results=[[1,1024,512],[512,32,32],[64,64,128]].map(yields);
  for(const {field,signal} of results){
    assert.ok(field>=9*count/CINEMATIC_STEP_WORK,`${field} field yields`);
    assert.ok(signal>=count/CINEMATIC_STEP_WORK,`${signal} signal yields`);
  }
  const fields=results.map(r=>r.field);
  assert.ok(Math.max(...fields)<=1.5*Math.min(...fields),`yields depend on shape: ${fields}`);
});

/** The previous synchronous empty-cube growth, kept as a reference. */
function referenceEmptyCubes(occupied,bricks,data){
  const [bx,by,bz]=bricks,count=occupied.length;
  const axes=[[by*bz,bx],[bz,by],[1,bz]];
  let grown=occupied,spread=Float32Array.from({length:count},(_,brick)=>data[4*brick+2]);
  for(let brick=0;brick<count;brick++)data[4*brick+3]=spread[brick];
  for(let radius=1;radius<=CINEMATIC_SKIP_LIMIT;radius++){
    let next=grown;
    for(const [stride,size] of axes){
      const source=next,values=spread;next=source.slice();spread=values.slice();
      for(let brick=0;brick<count;brick++){
        const c=Math.floor(brick/stride)%size;
        if(c>0){if(source[brick-stride])next[brick]=1;if(values[brick-stride]>spread[brick])spread[brick]=values[brick-stride];}
        if(c<size-1){if(source[brick+stride])next[brick]=1;if(values[brick+stride]>spread[brick])spread[brick]=values[brick+stride];}
      }
    }
    grown=next;let any=false;
    for(let brick=0;brick<count;brick++)if(!grown[brick]){data[4*brick+1]=radius;data[4*brick+3]=spread[brick];any=true;}
    if(!any)break;
  }
}

test('time-sliced majorants match the reference empty-cube growth and yield while they are built',()=>{
  const shape=[70,52,61],count=70*52*61,signal=new Float32Array(count).fill(30),mask=new Uint8Array(count);
  for(const [cx,cy,cz] of [[8,9,10],[50,40,12],[30,20,50]])for(let x=cx;x<cx+4;x++)for(let y=cy;y<cy+5;y++)for(let z=cz;z<cz+3;z++){
    const i=index(shape,x,y,z);signal[i]=400;mask[i]=1;
  }
  for(let x=0;x<70;x++)for(let z=0;z<61;z++)signal[index(shape,x,51,z)]=900;
  const {stats,tags}=build(signal,shape,[{cls:0,source:{kind:'mask',mask}}]);
  for(const [ct,boneWeight] of [[false,0],[true,1],[true,.1]]){
    const field=ct?finishCinematicSteps(cinematicFieldSteps(signal,shape,[1,0,0,0,1,0,0,0,1])):null;
    const material=cinematicMaterial(tags,{ct,boneWeight,window:[0,500]});
    const steps=cinematicMajorantSteps(stats,tags,material,field);
    let result=steps.next(),yields=0;
    while(!result.done){yields++;result=steps.next();}
    const data=result.value,expected=data.slice();
    for(let brick=0;brick<expected.length/4;brick++){expected[4*brick+1]=0;expected[4*brick+3]=0;}
    referenceEmptyCubes(Uint8Array.from({length:expected.length/4},(_,brick)=>expected[4*brick]>0?1:0),stats.bricks,expected);
    assert.deepEqual(data,expected);
    assert.ok(data.some((v,i)=>i%4===1&&v>=2),'some empty cubes are skipped');
  }
  // A head-sized brick grid is built in many slices.
  const bricks=[96,96,64],brickCount=96*96*64,maxValue=new Float32Array(brickCount).fill(30);
  for(let brick=0;brick<brickCount;brick+=997)maxValue[brick]=900;
  const steps=cinematicMajorantSteps({bricks,maxValue,range:[0,1000]},null,cinematicMaterial(null,{ct:true,boneWeight:1,window:[0,500]}));
  let yields=0;while(!steps.next().done)yields++;
  assert.ok(yields>=4*brickCount/CINEMATIC_STEP_WORK,`${yields} majorant yields`);
});

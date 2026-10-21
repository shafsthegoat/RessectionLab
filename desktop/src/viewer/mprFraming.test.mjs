import test from 'node:test';
import assert from 'node:assert/strict';
import {cIndex,inverseAffine,sampleTrilinear,sliceRect,transformPoint,volumeBounds} from './coordinates.ts';
import {mprView,mprPoint,mprPixel,sliceFraming,sourceSignalBounds} from './mprFraming.ts';

const identity=[[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]];
const close=(a,b)=>a.forEach((n,i)=>assert.ok(Math.abs(n-b[i])<1e-7,`${a} != ${b}`));

test('Signal fit enlarges padded heads without changing or losing full source access',()=>{
 const shape=[100,100,100],data=new Float32Array(1e6);
 for(let x=28;x<=72;x++)for(let y=20;y<=80;y++)for(let z=25;z<=75;z++)data[cIndex(shape,[x,y,z])]=100;
 data[0]=1000; // A sparse hot pixel cannot shrink the head's display fit.
 const before=data.slice(),full=volumeBounds(shape,identity),support=sourceSignalBounds(data,shape,identity);
 for(let axis=0;axis<3;axis++){assert.ok(support[0][axis]>10);assert.ok(support[1][axis]<90);}
 const previous=sliceRect(full,'axial',330,175),fitted=mprView(full,support,'axial',330,175),image=mprView(full,support,'axial',330,175,{fit:'image',zoom:1});
 assert.ok(fitted.rect.width/(support[1][0]-support[0][0])>previous.width/99*1.65,'Head diameter grows at least 65% from padding and content fit');
 assert.equal(fitted.rect.width/(support[1][0]-support[0][0]),fitted.rect.height/(support[1][1]-support[0][1]));
 assert.equal(image.bounds,full);
 for(const point of [[0,0,50],[99,99,50]]){
  const pixel=mprPixel(image,'axial',point);
  close(mprPoint(image,'axial',[50,50,50],...pixel,identity,shape),point);
 }
 assert.deepEqual(data,before);
 assert.equal(sourceSignalBounds(data,shape,identity),support,'Display recomputations reuse the source extent');
});

test('Empty, nonfinite, sparse and constant negative scans safely fall back to image fit',()=>{
 const shape=[10,10,10],full=volumeBounds(shape,identity);
 for(const data of [new Float32Array(1000),new Float32Array(1000).fill(NaN),new Float32Array(1000).fill(-1000)])assert.deepEqual(sourceSignalBounds(data,shape,identity),full);
 const sparse=new Float32Array(1000);sparse[555]=100;assert.deepEqual(sourceSignalBounds(sparse,shape,identity),full);
 assert.deepEqual(sliceFraming({fit:'invalid',zoom:Infinity}),{fit:'anatomy',zoom:1});
 assert.equal(sliceFraming({zoom:100}).zoom,4);
 assert.equal(sliceFraming({zoom:-2}).zoom,.5);
});

test('Mirrored oblique anisotropic source: pointer, projected overlay and GPU RAS sampling agree in every plane at every zoom',()=>{
 const shape=[13,11,9],a=.42,c=Math.cos(a),s=Math.sin(a);
 const affine=[[-2*c,-3*s,.1,33],[-2*s,3*c,.2,-27],[.25,0,4,12],[0,0,0,1]],inverse=inverseAffine(affine),full=volumeBounds(shape,affine);
 const data=new Float32Array(13*11*9);
 for(let x=0;x<13;x++)for(let y=0;y<11;y++)for(let z=0;z<9;z++)data[cIndex(shape,[x,y,z])]=100*x+10*y+z;
 const native=[6.25,5.1,3.5],world=transformPoint(affine,native);
 for(const plane of ['axial','coronal','sagittal'])for(const zoom of [.5,1,1.75,4]){
  const view=mprView(full,full,plane,520,360,{fit:'image',zoom}),pixel=mprPixel(view,plane,world),picked=mprPoint(view,plane,world,...pixel,inverse,shape);
  close(picked,world);
  const [a,b]=plane==='axial'?[0,1]:plane==='coronal'?[0,2]:[1,2];
  // Exact GLSL mix(uLow,uHigh,(vUv-uRect.xy)/uRect.zw), including GL's bottom origin.
  const shader=[...world],uv=[pixel[0]/520,1-pixel[1]/360],rect=[view.rect.left/520,1-(view.rect.top+view.rect.height)/360,view.rect.width/520,view.rect.height/360];
  shader[a]=full[0][a]+(uv[0]-rect[0])/rect[2]*(full[1][a]-full[0][a]);
  shader[b]=full[0][b]+(uv[1]-rect[1])/rect[3]*(full[1][b]-full[0][b]);
  close(shader,picked);
  assert.ok(Math.abs(sampleTrilinear(data,shape,transformPoint(inverse,shader))-679.5)<1e-7);
 }
 const view=mprView(full,full,'axial',520,360,{fit:'image',zoom:1});
 assert.equal(mprPoint(view,'axial',world,view.rect.left,view.rect.top,inverse,shape),null,'Blank corners of oblique RAS bounding box cannot select outside acquired grid');
});

test('Signal support follows RAS rather than native index axes on mirrored oblique acquisitions',()=>{
 const shape=[40,40,40],data=new Float32Array(64000),angle=.65;
 const affine=[[-2*Math.cos(angle),-3*Math.sin(angle),0,50],[-2*Math.sin(angle),3*Math.cos(angle),0,-30],[0,0,4,10],[0,0,0,1]];
 for(let x=10;x<=25;x++)for(let y=8;y<=28;y++)for(let z=12;z<=26;z++)data[cIndex(shape,[x,y,z])]=100;
 const bounds=sourceSignalBounds(data,shape,affine);
 for(const voxel of [[10,8,12],[25,28,26],[25,8,12],[10,28,26]])transformPoint(affine,voxel).forEach((n,a)=>assert.ok(n>=bounds[0][a]&&n<=bounds[1][a]));
 assert.ok(bounds[0][2]<=58&&bounds[1][2]>=114);
});

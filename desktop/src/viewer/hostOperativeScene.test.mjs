import assert from 'node:assert/strict';
import test from 'node:test';
import * as THREE from 'three';
import {operativeAnatomyTitle,regionProvenance} from './operativeProvenance.ts';
import {operativeSurfaceStyle,operativeColors} from './operativeMaterial.ts';
import {generatedTissueFinish} from './generatedTissueFinish.ts';
import {exposureDisplayKind} from '../modeled-volume.ts';

// Layer rows as a host rehearsal declares them (surgical_host_rehearsal.LAYERS).
const host=[
  ['scalp','estimated'],['galea','estimated'],['tissue','estimated'],['skull','estimated'],['diploe','estimated'],
  ['dura','assumed'],['grayMatter','estimated'],['whiteMatter','estimated'],['tract','estimated'],
  ['tumor','simulated'],['tumor','simulated'],['vein','estimated'],['artery','estimated'],
].map(([kind,provenance])=>({kind,provenance}));

test('a host head names its estimated, assumed and generated layers',()=>{
  assert.equal(operativeAnatomyTitle(host),'Estimated anatomy · assumed dura · generated tumour');
  // Earlier generated phantoms declare nothing and keep their label.
  assert.equal(operativeAnatomyTitle([{kind:'scalp'},{kind:'tumor'}]),'Generated anatomy');
  assert.equal(operativeAnatomyTitle([{kind:'scalp',provenance:'simulated'},{kind:'tumor',provenance:'generated'}]),'Generated anatomy');
  // An undeclared layer is never folded into a declared group.
  assert.match(operativeAnatomyTitle([{kind:'scalp',provenance:'estimated'},{kind:'tumor'}]),/other layers undeclared$/);
});

test('inventory rows lead with provenance and review',()=>{
  assert.equal(regionProvenance({provenance:'assumed',review_status:'assumed'},'membrane',false),'Assumed layer · membrane · context anatomy');
  assert.equal(regionProvenance({provenance:'estimated',review_status:'unreviewed'},'soft_tissue',false),'Model estimate, unreviewed · soft tissue · context anatomy');
  assert.equal(regionProvenance({provenance:'simulated'},'tumor',true),'Generated · tumor · declared target');
  assert.equal(regionProvenance(undefined,'bone',false),'bone · context anatomy');
});

test('host display kinds have operative materials instead of falling back to brain or label colours',()=>{
  const surgeon={operativeView:'surgeon',microscope:{tissueStyle:'operative'}};
  // Extracranial tissue is not brain.
  assert.equal(exposureDisplayKind('tissue'),'extracranial');
  assert.equal(exposureDisplayKind('scalp'),'scalp');
  const extracranial=operativeSurfaceStyle('extracranial',surgeon);
  assert(extracranial&&extracranial.color!==operativeColors.brain);
  // A tract bundle looks like the white matter it belongs to.
  assert.deepEqual(operativeSurfaceStyle('tract',surgeon),operativeSurfaceStyle('whiteMatter',surgeon));
  // Source-label style and the external view keep the source colours.
  assert.equal(operativeSurfaceStyle('tract',{operativeView:'surgeon',microscope:{tissueStyle:'labels'}}),null);
  assert.equal(operativeSurfaceStyle('extracranial',{}),null);
  for(const kind of ['tract','extracranial']){
    const material=new THREE.MeshStandardMaterial();
    assert(generatedTissueFinish(material,kind,false),kind+' has an illustrative finish');
  }
});

test('free space in the undeformed head is flagged once, never by later removal',async()=>{
  const {exteriorSkinSurface,FREE_SPACE,EXTERIOR_AIR}=await import('./exteriorSkinSurface.ts');
  // 11^3: a skin block around white matter with one enclosed free cell and one
  // white-matter cell reaching the exterior through a gap in the skin.
  const n=11,shape=[n,n,n],owner=new Uint16Array(n**3),id=(x,y,z)=>(x*n+y)*n+z;
  for(let x=2;x<9;x++)for(let y=2;y<9;y++)for(let z=2;z<9;z++)owner[id(x,y,z)]=2;
  for(let x=3;x<8;x++)for(let y=3;y<8;y++)for(let z=3;z<8;z++)owner[id(x,y,z)]=3;
  owner[id(5,5,5)]=0;owner[id(5,5,2)]=3;
  const affine=[[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]];
  const surface=exteriorSkinSurface(owner,shape,affine,2,new Set());
  const flags=surface?.flags??new Uint8Array(0);
  assert(flags.length,'exterior skin faces exist');
  assert.equal(flags[id(5,5,5)]&FREE_SPACE,FREE_SPACE,'enclosed free cell');
  assert.equal(flags[id(5,5,5)]&EXTERIOR_AIR,0,'enclosed free space is not exterior air');
  assert.equal(flags[id(0,0,0)]&(FREE_SPACE|EXTERIOR_AIR),FREE_SPACE|EXTERIOR_AIR);
  assert.equal(flags[id(4,4,4)]&FREE_SPACE,0,'owned tissue');
});

test('covers are illustrative: soft tissue under air reads as skin, white matter under CSF as cortex',async()=>{
  const {ExteriorSkinDomain,surfaceCoverFor}=await import('./exteriorSkinDomain.ts');
  const {OUTSIDE_BRAIN}=await import('./exteriorSkinSurface.ts');
  for(const kind of ['galea','extracranial','skull','diploe'])assert.equal(surfaceCoverFor(kind),'skin');
  assert.equal(surfaceCoverFor('whiteMatter'),'cortex');assert.equal(surfaceCoverFor('tract'),'cortex');
  for(const kind of ['scalp','grayMatter','tumor','dura','artery'])assert.equal(surfaceCoverFor(kind),undefined);
  const domain=new ExteriorSkinDomain(new Uint8Array(8),[2,2,2]);
  // Free space and outside-brain owners (here region 3) carry the outside-brain bit; brain (2) does not.
  domain.markOutsideBrain(Uint16Array.of(0,2,3,2,2,2,2,3),new Set([3]));
  assert.deepEqual([...domain.texture.image.data].map(f=>f&OUTSIDE_BRAIN?1:0),[1,0,1,0,0,0,0,1]);
  // Only a material with the illustrative finish can be covered, so source-label colours stay exact.
  assert.equal(domain.cover(new THREE.MeshStandardMaterial(),'skin',false),false);
  const material=new THREE.MeshStandardMaterial();generatedTissueFinish(material,'galea',false);
  assert.equal(domain.cover(material,'skin',false),true);
  const shader={...THREE.ShaderLib.standard,uniforms:{}};
  material.onBeforeCompile(shader,{});
  assert.equal(shader.uniforms.uCoverFlags.value,domain.texture);
  assert.equal(shader.uniforms.uCoverActive,material.userData.tissueFinish,'cover follows the illustrative finish');
  assert.match(shader.fragmentShader,/if\(covered\)diffuseColor\.rgb=uCoverColor;/);
  assert.match(shader.vertexShader,/vCoverVoxel=position\+finishRestOffset;/);
  // A replacement domain is followed without recompiling.
  const next=new ExteriorSkinDomain(new Uint8Array(8),[2,2,2]);
  assert.equal(next.cover(material,'skin',false),true);
  assert.equal(shader.uniforms.uCoverFlags.value,next.texture);
});

test('the external overview looks along the declared exterior normal and fits the working grid',async()=>{
  const {exposureOverviewCamera}=await import('./microscope.ts');
  const affine=[[1,0,0,-50],[0,1,0,-50],[0,0,1,-40],[0,0,0,1]];
  const view={scene:{shape:[101,101,81],affine},operativeExposureV1:{exteriorNormalRas:[-.6,0,.8]}};
  const camera=exposureOverviewCamera(view);
  assert(camera);
  const d=camera.position.map((v,i)=>v-camera.target[i]),length=Math.hypot(...d);
  assert.deepEqual(camera.target.map(v=>Math.round(v)),[0,0,0]);
  // The eye is on the outward side of the opening and above it.
  assert(d[0]*-.6+d[2]*.8>.7*length);
  assert(d[2]>0);
  // The whole grid fits inside the 34 degree field.
  const radius=Math.hypot(50,50,40);
  assert(length*Math.sin(17*Math.PI/180)>=.85*radius);
  assert.equal(exposureOverviewCamera({scene:view.scene,operativeExposureV1:{}}),undefined);
});

test('removal carving lists accepted cells and displaced operative bodies only',async()=>{
  const {displayRemovalCells,maskCells,scanTissueWindow,SCAN_SHADED_KINDS}=await import('./scanRemoval.ts');
  const scene={shape:[4,4,4],affine:[[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]};
  const body=(cells,translationMm=[0,0,0])=>({cells,centersMm:cells.map(id=>[Math.floor(id/16),Math.floor(id/4)%4,id%4].map((v,i)=>v+translationMm[i]))});
  const view={scene,removed:[5,9],operativeExposureV1:{bodies:{scalp:body([60,61],[0,0,4]),bone:body([40]),dura:body([30])}}};
  assert.deepEqual(displayRemovalCells(view).sort((a,b)=>a-b),[5,9,60,61]);
  assert.deepEqual(displayRemovalCells({scene,removed:[3]}),[3]);
  assert.deepEqual([...maskCells(Uint8Array.of(0,1,0,1,1))],[1,3,4]);
  // The tissue window ignores cells outside the brain and lesion masks.
  const signal=Float32Array.from({length:400},(_,i)=>i<200?1000:i/2),mask=Uint8Array.from({length:400},(_,i)=>i>=200?1:0);
  const window=scanTissueWindow(signal,[mask],1);
  assert(window&&window[0]>=100&&window[1]<=200&&window[1]>window[0]);
  assert.equal(scanTissueWindow(new Float32Array(400).fill(3),[mask],1),undefined);
  assert.deepEqual([...SCAN_SHADED_KINDS].sort(),['grayMatter','tract','tumor','whiteMatter']);
});

test('pending CUSA work recedes the displayed cavity wall into partly worked cells only',async()=>{
  const {prepareSurface}=await import('./prepareSurface.ts');
  const shape=[16,12,12],identity=[[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]];
  const index=(x,y,z)=>(x*12+y)*12+z;
  const owner=new Uint16Array(16*12*12);
  for(let x=2;x<14;x++)for(let y=2;y<10;y++)for(let z=2;z<10;z++)owner[index(x,y,z)]=1;
  const removed=[];for(let x=8;x<14;x++)for(let y=2;y<10;y++)for(let z=2;z<10;z++)removed.push(index(x,y,z));
  const mask=Uint8Array.from(owner,v=>v?1:0);for(const i of removed)mask[i]=0;
  const reference={owner,protectedRegions:[]},cells=Uint32Array.from(removed);
  const worked=[index(7,5,5),index(7,5,6),index(7,6,5),index(7,6,6)];
  const plain=prepareSurface(mask,shape,identity,'soft_tissue',true,{...reference,cavity:{region:1,cells}});
  const partly=prepareSurface(mask,shape,identity,'soft_tissue',true,{...reference,cavity:{region:1,cells,
    progress:{cells:Uint32Array.from(worked),fraction:Float32Array.from([.9,.9,.9,.9])}}});
  // Engine geometry and topology are unchanged; only display positions move.
  for(const k of ['positions','normals','indices'])assert.deepEqual(partly[k],plain[k]);
  const shown=s=>{const m=new Map();for(let i=0;i<s.positions.length;i+=3)m.set(Array.from(s.positions.subarray(i,i+3)).join(','),Array.from(s.smoothPositions.subarray(i,i+3)));return m;};
  const a=shown(plain),b=shown(partly);let receded=0,moved=0,far=0,near=0;
  for(const [key,p] of a){
    const q=b.get(key),[x,y,z]=key.split(',').map(Number);
    const d=Math.hypot(q[0]-p[0],q[1]-p[1],q[2]-p[2]);
    if(d>1e-6)moved++;
    assert(d<=.5+1e-6,'within half a cell');
    // A wall vertex beside a worked cell moves into it, toward lower x.
    if(x>7&&x<8&&y>=5&&y<=6&&z>=5&&z<=6){near=Math.max(near,p[0]-q[0]);if(q[0]<p[0]-1e-3)receded++;}
    // Far from the worked cells nothing moves.
    if(Math.abs(y-5.5)>3||Math.abs(z-5.5)>3)far=Math.max(far,d);
  }
  assert(receded>0&&moved>=receded);
  console.log('worked-wall recession',near.toFixed(3),'mm; largest distant change',far.toFixed(3),'mm');
  assert(far<near/4,'the change is local to the worked cells');
});

test('microscope optics stay sharp at the focus plane and fall off gently toward the field edge',async()=>{
  const {defocusRadiusPx,fieldFalloff,MicroscopeOpticsPass}=await import('./microscopeOptics.ts');
  const optics={focusDistanceMm:160,fieldMm:30};
  assert.equal(defocusRadiusPx(160,optics,900),0);
  assert.equal(defocusRadiusPx(164.5,optics,900),0,'within the depth of field');
  const near=defocusRadiusPx(170,optics,900),far=defocusRadiusPx(200,optics,900);
  assert(near>0&&near<far&&far<=6,'radius grows with defocus up to a bound');
  // Higher magnification (smaller field) blurs the same defocus more.
  assert(defocusRadiusPx(170,{...optics,fieldMm:15},900)>near);
  assert.equal(fieldFalloff(0),1);assert.equal(fieldFalloff(.6),1);
  assert(fieldFalloff(1)>.7&&fieldFalloff(1)<1);assert(Math.abs(fieldFalloff(2)-.7)<1e-9);
  const pass=new MicroscopeOpticsPass();
  assert.match(pass.material.fragmentShader,/uOpening/);
  assert.match(pass.material.fragmentShader,/perspectiveDepthToViewZ/);
  assert.match(pass.material.fragmentShader,/#include <colorspace_fragment>/);
  assert.equal(pass.estimatedTargetBytes,0);
  pass.dispose();
});

test('the microscope lamp centres on the bone flap outline and darkens the scalp around it',async()=>{
  const {openingFalloff,openingFromCells}=await import('./microscopeOptics.ts');
  assert.equal(openingFalloff(0,20),1);assert.equal(openingFalloff(21,20),1,'the opening and its rim stay fully lit');
  assert(openingFalloff(30,20)<.85&&openingFalloff(30,20)>.5);assert(Math.abs(openingFalloff(60,20)-.4)<1e-9);
  assert.equal(openingFalloff(60,0),1,'no opening, no lamp spot');
  // A 3 mm thick disc of radius 10 mm in the z=0 plane, outward +z: the radius ignores the thickness.
  const cells=[];for(let x=-10;x<=10;x++)for(let y=-10;y<=10;y++)if(x*x+y*y<=100)for(const z of [0,1,2])cells.push([x+5,y,z]);
  const o=openingFromCells(cells,[0,0,1]);
  assert.deepEqual(o.centerMm.map(v=>+v.toFixed(6)),[5,0,1]);assert(Math.abs(o.radiusMm-10)<1e-9);
  assert.equal(openingFromCells([]),null);
});

test('the wet film is a surgeon-view illustration and never applies to source-label colours',async()=>{
  const {applyOperativeSurface}=await import('./operativeMaterial.ts');
  const {wetFilmStrength}=await import('./generatedTissueFinish.ts');
  const material=new THREE.MeshStandardMaterial();generatedTissueFinish(material,'grayMatter',false);
  applyOperativeSurface(material,'grayMatter',{operativeView:'surgeon',microscope:{tissueStyle:'operative'}});
  assert.equal(material.userData.wetFilm.value,wetFilmStrength.grayMatter);
  applyOperativeSurface(material,'grayMatter',{operativeView:'surgeon',microscope:{tissueStyle:'labels'}});
  assert.equal(material.userData.wetFilm.value,0);
  applyOperativeSurface(material,'grayMatter',{layerOpacity:{a:1}});
  assert.equal(material.userData.wetFilm.value,0,'external views stay dry');
  const shader={...THREE.ShaderLib.standard,uniforms:{}};material.onBeforeCompile(shader,{});
  assert.equal(shader.uniforms.uWetFilm,material.userData.wetFilm);
  assert.match(shader.fragmentShader,/pow\(facing,120\.\)/);
});

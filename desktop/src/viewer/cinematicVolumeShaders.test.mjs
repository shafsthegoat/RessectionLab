import test from 'node:test';
import assert from 'node:assert/strict';
import {registerHooks} from 'node:module';
import * as THREE from 'three';

// The production bundler resolves extensionless viewer imports.
registerHooks({resolve(specifier,context,nextResolve){
  try{return nextResolve(specifier,context)}catch(error){
    if(specifier.startsWith('.')&&!/\.[a-z]+$/i.test(specifier))return nextResolve(specifier+'.ts',context);
    throw error;
  }
}});
const {CinematicVolumePass}=await import('./CinematicVolumePass.ts');
const finishModule=await import('./CinematicVolumePass.ts');

/** Active lines of a GLSL source under the given defines, with #if balance checked. */
function preprocess(source,defines){
  const macros=new Map(Object.entries(defines).map(([k,v])=>[k,String(v)]));
  const stack=[],lines=[];
  const value=expression=>{
    const match=/^(\w+)\s*(==|!=)\s*(\d+)$/.exec(expression.trim());
    if(match){const left=Number(macros.get(match[1])??0);return match[2]==='=='?left===Number(match[3]):left!==Number(match[3]);}
    assert.match(expression.trim(),/^\w+$/,`unsupported #if ${expression}`);
    return Number(macros.get(expression.trim())??expression.trim())!==0;
  };
  for(const raw of source.split('\n')){
    const line=raw.trim(),active=stack.every(frame=>frame.active);
    let match;
    if((match=/^#\s*ifndef\s+(\w+)/.exec(line)))stack.push({active:!macros.has(match[1]),taken:!macros.has(match[1])});
    else if((match=/^#\s*ifdef\s+(\w+)/.exec(line)))stack.push({active:macros.has(match[1]),taken:macros.has(match[1])});
    else if((match=/^#\s*if\s+(.+)$/.exec(line))){const v=value(match[1]);stack.push({active:v,taken:v});}
    else if(/^#\s*else\b/.test(line)){assert.ok(stack.length,'#else without #if');const frame=stack.at(-1);frame.active=!frame.taken;frame.taken=true;}
    else if(/^#\s*endif\b/.test(line)){assert.ok(stack.length,'#endif without #if');stack.pop();}
    else if(!active)continue;
    else if((match=/^#\s*define\s+(\w+)(?:\s+(.*))?$/.exec(line)))macros.set(match[1],match[2]??'1');
    else lines.push(raw);
  }
  assert.equal(stack.length,0,'unbalanced #if');
  return {code:lines.join('\n').replace(/\/\/.*$/gm,''),macros};
}
const uniformsOf=code=>new Set([...code.matchAll(/^\s*uniform\s+\w+\s+(\w+)\s*(?:\[\s*\d+\s*\])?\s*;/gm)].map(m=>m[1]));
function balanced(code,open,close){
  let depth=0;
  for(const c of code){if(c===open)depth++;else if(c===close){depth--;assert.ok(depth>=0,`unbalanced ${close}`);}}
  assert.equal(depth,0,`unbalanced ${open}${close}`);
}
/** Number of top-level comma-separated arguments in a call starting at index. */
function argumentCount(code,index){
  let depth=0,count=1;
  for(let i=code.indexOf('(',index);i<code.length;i++){
    const c=code[i];
    if(c==='(')depth++;else if(c===')'){if(--depth===0)return count;}else if(c===','&&depth===1)count++;
  }
  assert.fail('unterminated call');
}
function checkStructure(name,code,supplied){
  balanced(code,'{','}');balanced(code,'(',')');balanced(code,'[',']');
  assert.doesNotMatch(code,/\b(texture2D|texture3D|gl_FragColor|varying|attribute)\b/,`${name}: GLSL 1 built-ins in a GLSL 3 shader`);
  // Implicit int/float mixing is an error in GLSL ES.
  assert.doesNotMatch(code,/(?<![\w.])\d+\.\d*\s*[-+*/]\s*\d+(?![\d.eE\w])/,`${name}: float literal combined with an int literal`);
  assert.doesNotMatch(code,/(?<![\w.])\d+\s*[-+*/]\s*\d+\.\d*/,`${name}: int literal combined with a float literal`);
  const declared=uniformsOf(code);
  assert.deepEqual([...declared].filter(u=>!supplied.has(u)),[],`${name}: declared uniforms the material does not supply`);
  assert.deepEqual([...supplied].filter(u=>!declared.has(u)),[],`${name}: supplied uniforms the shader does not declare`);
  // Struct constructors name every member.
  for(const [,struct,body] of code.matchAll(/struct\s+(\w+)\s*\{([^}]*)\}/g)){
    const members=body.split(';').filter(member=>member.trim()).flatMap(member=>member.trim().split(/\s+/).slice(1).join('').split(',')).length;
    for(const call of code.matchAll(new RegExp(`(?<![\\w])${struct}\\s*\\(`,'g')))
      assert.equal(argumentCount(code,call.index),members,`${name}: ${struct} constructed with every member`);
  }
}

test('each cinematic kernel assembles to balanced GLSL 3 whose uniforms are exactly those its pass supplies',()=>{
  const texture=new THREE.Data3DTexture(new Float32Array(8),2,2,2);
  const shared={uClipEnabled:{value:false},uClipPlane:{value:new THREE.Vector4()},uExtraClipCount:{value:0},
    uExtraClips:{value:Array.from({length:7},()=>new THREE.Vector4())},uBackground:{value:new THREE.Color()}};
  const pass=new CinematicVolumePass({signal:new Float32Array(8),shape:[2,2,2],texture,worldToVoxel:new THREE.Matrix4(),voxelMm:1,ct:true,floatLinear:false},shared);
  try{
    const kernels=[...pass.rest,...pass.motion];
    assert.deepEqual(kernels.map(k=>[k.defines.CINEMATIC_REST,k.defines.CINEMATIC_PASS]),[[1,0],[1,1],[0,0],[0,1]]);
    for(const kernel of kernels){
      const name=`kernel rest ${kernel.defines.CINEMATIC_REST} pass ${kernel.defines.CINEMATIC_PASS}`;
      const {code,macros}=preprocess(kernel.fragmentShader,kernel.defines);
      assert.equal(kernel.glslVersion,THREE.GLSL3);
      checkStructure(name,code,new Set(Object.keys(kernel.uniforms)));
      assert.ok(Number.isInteger(Number(macros.get('PRIMARY_STEPS')))&&Number.isInteger(Number(macros.get('BOUNCE_STEPS'))),'constant loop bounds');
      // Two colour attachments, as the kernel targets have.
      assert.match(code,/layout\(location=0\)\s*out\s+vec4\s+outColor;/);assert.match(code,/layout\(location=1\)\s*out\s+vec4\s+outGuide;/);
      if(kernel.defines.CINEMATIC_PASS===1)assert.match(code,/isnan\(color\)\)\|\|any\(isinf\(color\)\)\)color=vec3\(0\.0\)/,'a non-finite sample is dropped');
    }
    for(const [name,material] of [['denoise',pass.filter],['surfaces',pass.surfaces],['accumulate',pass.accumulate]]){
      const {code}=preprocess(material.fragmentShader,material.defines??{});
      checkStructure(name,code,new Set(Object.keys(material.uniforms)));
    }
    // The composite reads at most the fragment sampler budget of WebGL 2.
    assert.ok([...preprocess(pass.surfaces.fragmentShader,{}).code.matchAll(/uniform\s+sampler2D/g)].length<=16);
    // Every kernel shares one uniform set, and the composite shares its projection.
    assert.ok(kernels.every(k=>k.uniforms===pass.uniforms));
    assert.equal(pass.surfaces.uniforms.uInverseProjection,pass.uniforms.uInverseProjection);
    assert.equal(pass.accumulation[0].textures[0].magFilter,THREE.NearestFilter,'float colour is not linearly filtered without the extension');
  }finally{pass.dispose();}
  const linear=new CinematicVolumePass({signal:new Float32Array(8),shape:[2,2,2],texture,worldToVoxel:new THREE.Matrix4(),voxelMm:1,ct:true,floatLinear:true},shared);
  try{assert.equal(linear.accumulation[0].textures[0].magFilter,THREE.LinearFilter);}finally{linear.dispose();}
});

test('the structural checks reject the mistakes they look for',()=>{
  assert.throws(()=>balanced('void main(){','{','}'));
  assert.throws(()=>preprocess('#if A == 1\nx;\n',{A:1}));
  assert.throws(()=>checkStructure('t','uniform float uA;\nvoid main(){float x=1.0*2;}',new Set(['uA'])));
  assert.throws(()=>checkStructure('t','uniform float uA;\nvoid main(){}',new Set(['uA','uB'])));
  assert.throws(()=>checkStructure('t','struct H{bool a;vec3 b,c;};\nH h(){return H(true,vec3(0.0));}',new Set()));
  checkStructure('t','struct H{bool a;vec3 b,c;};\nH h(){return H(true,vec3(0.0),vec3(1.0));}\nvoid main(){vec3 v=vec3(0,0,1);float y=2.0*3.0;}',new Set());
});

test('the wet and porcelain finishes add no sampled highlight and keep the firefly clamp',()=>{
  const texture=new THREE.Data3DTexture(new Float32Array(8),2,2,2);
  const shared={uClipEnabled:{value:false},uClipPlane:{value:new THREE.Vector4()},uExtraClipCount:{value:0},
    uExtraClips:{value:Array.from({length:7},()=>new THREE.Vector4())},uBackground:{value:new THREE.Color()}};
  const pass=new CinematicVolumePass({signal:new Float32Array(8),shape:[2,2,2],texture,worldToVoxel:new THREE.Matrix4(),voxelMm:1,ct:true,floatLinear:false},shared);
  try{
    for(const rest of [1,0]){
      const {code}=preprocess(pass.rest[1].fragmentShader,{CINEMATIC_REST:rest,CINEMATIC_PASS:1});
      const key=code.slice(code.indexOf('vec3 keyLight('),code.indexOf('vec3 reflection('));
      // Highlights come from the disc centre: every ggx lobe reads the key direction, never the sampled one.
      const lobes=[...key.matchAll(/ggx\(([^;]*?)uKeyDirection/g)];
      assert.equal(lobes.length,2,'a base lobe and a wet coat lobe');
      assert.doesNotMatch(key,/ggx\([^)]*,l,/);
      assert.match(code,/reflection\(h,view\)/,`rest ${rest}: the studio reflection is shaded`);
      assert.match(code,/if\(luminance>6\.0\)/,'the firefly clamp stays');
    }
    const {CINEMATIC_FINISH:f}=finishModule;
    for(const value of [f.vessel.roughness,f.vessel.coatRoughness,f.bone.roughness])assert.ok(value>.1&&value<.6);
    assert.ok(f.vessel.coatRoughness<f.vessel.roughness&&f.vessel.roughness<f.bone.roughness,'wet lumen is glossier than bone');
    assert.ok(f.bone.shellDiffuse>0&&f.bone.shellDiffuse<1);
  }finally{pass.dispose();}
});

test('vessel albedos are a light vermilion and a blue of matching value',async()=>{
  const {CINEMATIC_VESSEL_ALBEDO}=await import('./CinematicVolumePass.ts');
  const hsv=hex=>{const [r,g,b]=[1,3,5].map(i=>parseInt(hex.slice(i,i+2),16)/255),max=Math.max(r,g,b),min=Math.min(r,g,b),d=max-min;
    const h=d===0?0:max===r?((g-b)/d+6)%6:max===g?(b-r)/d+2:(r-g)/d+4;return {h:h*60,s:d/max,v:max};};
  const artery=hsv(CINEMATIC_VESSEL_ALBEDO.artery),vein=hsv(CINEMATIC_VESSEL_ALBEDO.vein);
  assert.ok(artery.h>=8&&artery.h<=20,`artery hue ${artery.h} is vermilion, not crimson`);
  assert.ok(artery.v>=.7,'and light');
  assert.ok(vein.h>=200&&vein.h<=230);
  assert.ok(Math.abs(vein.v-artery.v)<=.1&&Math.abs(vein.s-artery.s)<=.15,'the vein blue matches the artery in value and saturation');
});

test('silhouette hits are shaded edge on and their secondary rays leave back along the eye ray',()=>{
  const texture=new THREE.Data3DTexture(new Float32Array(8),2,2,2);
  const shared={uClipEnabled:{value:false},uClipPlane:{value:new THREE.Vector4()},uExtraClipCount:{value:0},
    uExtraClips:{value:Array.from({length:7},()=>new THREE.Vector4())},uBackground:{value:new THREE.Color()}};
  const pass=new CinematicVolumePass({signal:new Float32Array(8),shape:[2,2,2],texture,worldToVoxel:new THREE.Matrix4(),voxelMm:1,ct:true,floatLinear:false},shared);
  try{
    const {code}=preprocess(pass.rest[1].fragmentShader,{CINEMATIC_REST:1,CINEMATIC_PASS:1});
    const surface=code.slice(code.indexOf('Hit surfaceAt('),code.indexOf('Hit bounceHit('));
    assert.doesNotMatch(surface,/h\.normal=-h\.normal/,'a back-facing normal is not mirrored into the lumen');
    assert.match(surface,/h\.normal=normalize\(h\.normal-direction\*\(away\+/);
    // Shadow and bounce origins step back toward the viewer as well as off the surface.
    assert.match(code,/vec3 origin=h\.world\+\(h\.normal\*\(0\.6\*h\.surface\)\+view\*0\.4\)\*uVoxelMm;/);
    assert.match(code,/vec3 origin2=world\+\(h\.normal\*\(0\.6\*h\.surface\)\+view\*0\.4\)\*uVoxelMm;/);
  }finally{pass.dispose();}
});

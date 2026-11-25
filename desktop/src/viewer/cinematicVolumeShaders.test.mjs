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
    for(const [name,material] of [['denoise',pass.filter],['surfaces',pass.surfaces]]){
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

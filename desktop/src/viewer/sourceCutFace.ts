/** Tissue layers of the cut faces of the source-surface scene (display only).
 *
 * A kerf wall carries, per vertex, its depth below the cut edge (`cutDepth`)
 * and the wall's full depth there (`cutTotal`), in millimetres of the working
 * grid. These materials draw the layers a surgeon sees on that face from those
 * two numbers: outer table, diploe and inner table on bone; skin, fat, galea
 * and pericranium on scalp; a single pale band on the dura. The layer
 * thicknesses are illustrative proportions of the measured face, not
 * segmented layers. */
import * as THREE from 'three';

export type CutFaceKind = 'bone' | 'scalp' | 'dura';

/** Band edges (mm from the top of the face) for a face `total` mm deep. */
export function cutFaceBands(kind: CutFaceKind, total: number) {
  const d = Math.max(0, total);
  if (kind === 'bone') {
    // Outer table about a third, inner table about a fifth of the vault, within adult ranges.
    let outer = Math.min(2.2, Math.max(0.8, 0.32 * d)), inner = Math.min(1.5, Math.max(0.6, 0.2 * d));
    if (outer + inner > 0.85 * d) { const s = (0.85 * d) / (outer + inner); outer *= s; inner *= s; }
    return {outer, inner: d - inner};
  }
  if (kind === 'scalp') {
    // Skin (epidermis and dermis), subcutaneous fat, galea, loose areolar tissue and pericranium.
    const dermis = Math.min(2.2, Math.max(0.6, 0.3 * d));
    const deep = Math.min(1.6, Math.max(0.5, 0.22 * d));
    return {outer: Math.min(dermis, 0.5 * d), inner: Math.max(d - deep, 0.6 * d)};
  }
  return {outer: d, inner: d};
}

const FINISH = `
float cutHash(vec3 q){q=fract(q*.1031);q+=dot(q,q.yzx+33.33);return fract((q.x+q.y)*q.z);}
float cutNoise(vec3 p){vec3 c=floor(p),f=fract(p);f=f*f*(3.-2.*f);
  return mix(mix(mix(cutHash(c),cutHash(c+vec3(1,0,0)),f.x),mix(cutHash(c+vec3(0,1,0)),cutHash(c+vec3(1,1,0)),f.x),f.y),
    mix(mix(cutHash(c+vec3(0,0,1)),cutHash(c+vec3(1,0,1)),f.x),mix(cutHash(c+vec3(0,1,1)),cutHash(c+vec3(1,1,1)),f.x),f.y),f.z);}`;

/** A flap's rest offset attribute (zero where a geometry has none), declared once. */
const restDeclaration = (vertex: string) => (vertex.includes('attribute vec3 finishRestOffset') ? '' : 'attribute vec3 finishRestOffset;\n');

/** Colours (linear-space hex) of each band, top to bottom. */
const PALETTE: Record<CutFaceKind, string[]> = {
  // cortical outer table, cancellous diploe (red-brown marrow), cortical inner table
  bone: ['#e9dcc3', '#94573f', '#e2d3b8'],
  // epidermis, dermis, subcutaneous fat, galea, areolar tissue and pericranium
  scalp: ['#9c5e4e', '#dcab9b', '#e0bd78', '#eee6dc', '#c79f92'],
  dura: ['#e8ddd8'],
};

/** Install the layered cut-face finish in a kerf-wall material (idempotent). */
export function cutFaceFinish(material: THREE.MeshStandardMaterial, kind: CutFaceKind) {
  if (material.userData.cutFace === kind) return;
  material.userData.cutFace = kind;
  const colours = PALETTE[kind].map((hex) => new THREE.Color(hex));
  const previous = material.onBeforeCompile, key = material.customProgramCacheKey();
  material.onBeforeCompile = (shader, renderer) => {
    previous.call(material, shader, renderer);
    colours.forEach((c, i) => (shader.uniforms[`uCut${i}`] = {value: c}));
    shader.vertexShader = 'attribute float cutDepth;attribute float cutTotal;varying float vCutDepth;varying float vCutTotal;varying vec3 vCutRest;\n' +
      shader.vertexShader.replace('#include <begin_vertex>', '#include <begin_vertex>\nvCutDepth=cutDepth;vCutTotal=cutTotal;vCutRest=position;');
    const uniforms = colours.map((_, i) => `uniform vec3 uCut${i};`).join('');
    let body: string;
    if (kind === 'bone') body = `
      float d=vCutDepth,t=max(vCutTotal,.01);
      float outer=clamp(.32*t,.8,2.2),inner=clamp(.2*t,.6,1.5);
      float s=min(1.,.85*t/(outer+inner));outer*=s;inner*=s;
      float w=.12;
      float diploe=smoothstep(outer-w,outer+w,d)*(1.-smoothstep(t-inner-w,t-inner+w,d));
      // Cancellous bone: marrow spaces and bleeding points; cortical bone: dense, faintly lamellar.
      float pores=cutNoise(vCutRest*3.1),fine=cutNoise(vCutRest*7.3+5.);
      vec3 spongy=uCut1*(.72+.42*pores)*mix(1.,.55,step(.74,fine));
      vec3 cortical=mix(uCut0,uCut2,step(t*.5,d))*(.94+.08*cutNoise(vCutRest*1.7));
      diffuseColor.rgb=mix(cortical,spongy,diploe);`;
    else if (kind === 'scalp') body = `
      float d=vCutDepth,t=max(vCutTotal,.01);
      float dermis=min(clamp(.3*t,.6,2.2),.5*t),deep=max(t-clamp(.22*t,.5,1.6),.6*t);
      float w=.1;
      float lob=cutNoise(vCutRest*1.9),fine=cutNoise(vCutRest*5.7+3.);
      vec3 fat=uCut2*(.86+.22*lob)*mix(1.,.82,smoothstep(.6,.75,fine));
      vec3 c=mix(uCut0,uCut1,smoothstep(.12,.3,d));
      c=mix(c,fat,smoothstep(dermis-w,dermis+w,d));
      float galea=deep+.55*(t-deep);
      c=mix(c,uCut3,smoothstep(deep-w,deep+w,d));
      c=mix(c,uCut4,smoothstep(galea-w,galea+w,d));
      diffuseColor.rgb=c;`;
    else body = `
      diffuseColor.rgb=uCut0*(.96+.06*cutNoise(vCutRest*4.));`;
    shader.fragmentShader = 'varying float vCutDepth;varying float vCutTotal;varying vec3 vCutRest;' + uniforms + '\n' + FINISH + '\n' +
      shader.fragmentShader.replace('#include <color_fragment>', '#include <color_fragment>\n{' + body + '}');
  };
  material.customProgramCacheKey = () => key + '/cut-face-' + kind;
  material.needsUpdate = true;
}

/** The two faces of the assumed dural sheet: the outer face is the dura's
 * outer layer; the back face (seen once the leaf is reflected) its smooth,
 * paler inner surface. Fine collagen grain, travelling with the rest position. */
export function duraSheetFinish(material: THREE.MeshStandardMaterial) {
  if (material.userData.duraSheet) return;
  material.userData.duraSheet = true;
  const inner = new THREE.Color('#e6dedd');
  const previous = material.onBeforeCompile, key = material.customProgramCacheKey();
  material.onBeforeCompile = (shader, renderer) => {
    previous.call(material, shader, renderer);
    shader.uniforms.uDuraInner = {value: inner};
    shader.vertexShader = restDeclaration(shader.vertexShader) + 'varying vec3 vDuraRest;\n' +
      shader.vertexShader.replace('#include <begin_vertex>', '#include <begin_vertex>\nvDuraRest=position+finishRestOffset;');
    shader.fragmentShader = 'varying vec3 vDuraRest;uniform vec3 uDuraInner;\n' + FINISH + '\n' +
      shader.fragmentShader.replace('#include <color_fragment>', `#include <color_fragment>
      {
        // Collagen bundles run in sheets: a fine directional grain, and faint meningeal tone.
        float grain=cutNoise(vDuraRest*vec3(5.5,1.4,5.5))-.5,broad=cutNoise(vDuraRest*.45)-.5;
        diffuseColor.rgb*=1.+.07*grain+.09*broad;
        if(!gl_FrontFacing)diffuseColor.rgb=mix(diffuseColor.rgb,uDuraInner,.55);
      }`);
  };
  material.customProgramCacheKey = () => key + '/dura-sheet';
  material.needsUpdate = true;
}

/** Subtle natural variation of the bone surface: broad mottling between ivory,
 * yellow and grey, fine vascular foramina. Rest coordinates, so a parked flap
 * keeps its pattern. */
export function boneSurfaceFinish(material: THREE.MeshStandardMaterial) {
  if (material.userData.boneSurface) return;
  material.userData.boneSurface = true;
  const previous = material.onBeforeCompile, key = material.customProgramCacheKey();
  material.onBeforeCompile = (shader, renderer) => {
    previous.call(material, shader, renderer);
    shader.vertexShader = restDeclaration(shader.vertexShader) + 'varying vec3 vBoneRest;\n' +
      shader.vertexShader.replace('#include <begin_vertex>', '#include <begin_vertex>\nvBoneRest=position+finishRestOffset;');
    shader.fragmentShader = 'varying vec3 vBoneRest;\n' + FINISH + '\n' +
      shader.fragmentShader.replace('#include <color_fragment>', `#include <color_fragment>
      {
        float broad=cutNoise(vBoneRest*.18)-.5,mid=cutNoise(vBoneRest*.7+9.)-.5,pits=cutNoise(vBoneRest*4.3+2.);
        vec3 tint=vec3(1.+.10*broad,1.+.06*broad-.03*mid,1.-.05*broad-.06*mid);
        diffuseColor.rgb*=tint*(1.+.10*mid)*mix(1.,.72,smoothstep(.86,.95,pits));
      }`);
  };
  material.customProgramCacheKey = () => key + '/bone-surface';
  material.needsUpdate = true;
}

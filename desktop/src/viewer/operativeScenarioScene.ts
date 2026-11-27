/** 3-D display of the engine's opt-in operative scenarios: brain shift,
 * stimulation mapping estimates, vessel wounds and recorded injury.
 * Everything here is drawn from `view` fields; source cells, counts and
 * picking stay with the engine. */
import * as THREE from 'three';
import type {SurgicalView} from '../surgical-types';
import type {Affine} from './coordinates';
import {placeInSourceFrame} from './sceneGeometry.ts';
import {BrainShiftDisplay,analyticShift,sampleRatios,TaperInterpolator,shiftAt,type BrainShiftViewV1} from './brainShiftField.ts';
import {latestMapping,type BleedingWound,type InjuryDisplay,type MappingObservation,type VesselBleedingState} from '../operative-scenarios.ts';

/** Planned (preoperative image) target outline, violet; target where the scenario moved it, amber. */
export const SCENARIO_COLORS={planned:'#a58cff',shifted:'#f0b45a',response:'#f0b45a',noResponse:'#9fb2b8',unknown:'#7d8587',
  arterial:'#d0201c',venous:'#5a0f14'} as const;

/** Heart rate the bleeding model declares by default (assumed research); the arterial emitter pulses at it. */
const PULSE_PER_MIN=70;

type Pt=[number,number,number];
const ras=(affine:Affine,cell:number,shape:[number,number,number]):Pt=>{
  const [,ny,nz]=shape,x=Math.floor(cell/(ny*nz)),y=Math.floor(cell/nz)%ny,z=cell%nz;
  return [0,1,2].map(a=>affine[a][0]*x+affine[a][1]*y+affine[a][2]*z+affine[a][3]) as Pt;
};

/** Mapping sites: amber where the model estimates a response, grey otherwise; the latest carries its reach. */
export function mappingSites(view:SurgicalView|undefined,limit=8){
  const rows=(view?.observations??[]).filter(o=>o.kind==='model-estimate'&&o.category==='mapping') as unknown as MappingObservation[];
  return rows.slice(-limit).map((row,i,all)=>({position:row.position,result:row.result,latest:i===all.length-1,
    reachMm:i===all.length-1&&typeof row.estimatedReachMm==='number'?row.estimatedReachMm:null}));
}

/** Open wounds with display positions (wound cells' centre, carried by the displayed brain shift). */
export function woundEmitters(view:SurgicalView|undefined,displace?:(p:Pt)=>Pt,snapped=false){
  const state=view?.vesselBleedingV1 as VesselBleedingState|undefined;
  if(!view||!state)return [];
  return Object.values(state.wounds??{}).filter((w:BleedingWound)=>w.sealFraction<1&&w.cells?.length).map(w=>{
    const points=w.cells.map(c=>ras(view.scene.affine,c,view.scene.shape));
    const centre=[0,1,2].map(a=>points.reduce((s,p)=>s+p[a],0)/points.length) as Pt;
    const arterial=w.kind==='artery'||w.kind==='uncertain';
    return {id:w.id,position:displace?displace(centre):centre,arterial,kindUnknown:w.kind==='uncertain',rateMm3PerSecond:w.rateMm3PerSecond,calibreMm:w.calibreMm??1,snapped};
  });
}

const outlineVertex=`
uniform bool uApplyShift;uniform bool uShiftOn;uniform highp sampler3D uShiftField;uniform vec3 uShiftOrigin;uniform float uShiftStep;uniform float uShiftScale;
varying vec3 vNormalView;varying vec3 vToEye;
#include <clipping_planes_pars_vertex>
void main(){
  vec3 transformed=position;
  if(uApplyShift&&uShiftOn){
    vec3 uvw=((transformed-uShiftOrigin)/uShiftStep+.5).zyx/vec3(textureSize(uShiftField,0));
    transformed+=(texture(uShiftField,uvw).xyz*2.-1.)*uShiftScale;
  }
  vec4 mvPosition=modelViewMatrix*vec4(transformed,1.);
  vNormalView=normalize(normalMatrix*normal);vToEye=-mvPosition.xyz;
  gl_Position=projectionMatrix*mvPosition;
  #include <clipping_planes_vertex>
}`;
const outlineFragment=`
uniform vec3 uColor;uniform float uOpacity;
varying vec3 vNormalView;varying vec3 vToEye;
#include <clipping_planes_pars_fragment>
void main(){
  #include <clipping_planes_fragment>
  float facing=abs(dot(normalize(vNormalView),normalize(vToEye)));
  float rim=smoothstep(.62,.12,facing);
  gl_FragColor=vec4(uColor,uOpacity*(.05+.95*rim));
}`;

function outlineMaterial(color:string,uniforms:Record<string,THREE.IUniform>,applyShift:boolean){
  return new THREE.ShaderMaterial({vertexShader:outlineVertex,fragmentShader:outlineFragment,transparent:true,depthTest:false,depthWrite:false,
    side:THREE.DoubleSide,clipping:true,uniforms:{...uniforms,uApplyShift:{value:applyShift},uColor:{value:new THREE.Color(color)},uOpacity:{value:.9}}});
}

export class OperativeScenarioScene {
  readonly group=new THREE.Group();
  readonly brainShift=new BrainShiftDisplay();
  private shiftingLayers=new Set<string>();
  private targetLayers:string[]=[];
  private readonly contours=new THREE.Group();
  private contourKey='';
  private shiftKey='';
  private frame:{inward:Pt}|null=null;
  private displace?:(p:Pt)=>Pt;
  private shiftActive=false;
  private readonly sites=new THREE.Group();
  private readonly wounds=new THREE.Group();
  private arterial:THREE.Object3D[]=[];
  readonly marks=new TissueMarks();
  private markedLayers=new Set<string>();
  /** With source vessel tubes, a wound marker sits on the nearest tube centreline point. */
  private woundSnap?:(p:Pt)=>Pt;
  setWoundSnap(snap?:(p:Pt)=>Pt){this.woundSnap=snap;}
  constructor(){this.group.name='scenario:operative';this.sites.name='scenario:mapping-sites';this.wounds.name='scenario:wound-emitters';this.contours.name='scenario:target-contours';this.group.add(this.sites,this.wounds,this.contours);}

  /** Pulse arterial emitters; returns the delay to the next frame while one is open, else null. */
  tick(now:number,visible:boolean):number|null{
    if(!this.arterial.length||!visible)return null;
    const reduced=typeof matchMedia==='function'&&matchMedia('(prefers-reduced-motion: reduce)').matches;
    const phase=(now/1000*PULSE_PER_MIN/60)%1,systole=reduced?.5:Math.exp(-((phase-.12)**2)/.008);
    for(const jet of this.arterial){const base=jet.userData.baseScale as number;jet.scale.set(base*(.75+.35*systole),base*(.55+.9*systole),base*(.75+.35*systole));}
    return reduced?null:110;
  }

  /** Names of the drawn layers the engine's scenario moves. */
  shifts(name:string){return this.shiftActive&&this.shiftingLayers.has(name);}

  update(view:SurgicalView|undefined,affine:Affine,surfaces:ReadonlyMap<string,THREE.Mesh>){
    const shift=view?.brainShiftV1 as BrainShiftViewV1|undefined;
    this.shiftActive=this.brainShift.update(shift??null,affine);
    const ids=new Set(shift?.display?.regionIds??[]);
    this.shiftingLayers=new Set(view?.scene.regions.filter(r=>ids.has(r.id)).map(r=>r.name)??[]);
    this.targetLayers=view?.scene.regions.filter(r=>r.target&&ids.has(r.id)).map(r=>r.name)??[];
    const field=analyticShift(shift);
    const taper=field&&shift?new TaperInterpolator(shift.latticeSamples,sampleRatios(field,shift.latticeSamples)):null;
    const displace=field&&taper?(p:Pt)=>{const d=shiftAt(field,taper,p);return [p[0]+d[0],p[1]+d[1],p[2]+d[2]] as Pt;}:undefined;
    this.displace=field&&taper?(p:Pt)=>shiftAt(field,taper,p) as Pt:undefined;
    this.frame=field?{inward:field.inward as Pt}:null;
    this.shiftKey=JSON.stringify([shift?.state,shift?.currentPeakMm]);
    this.updateSites(view);
    this.updateWounds(view,displace,shift?.resolved.openingInwardRas);
    const injury=view?.surgicalInjuryV1 as InjuryDisplay|undefined;
    this.marks.update(injury,view?.scene.shape);
    this.markedLayers=new Set(injury?view!.scene.regions.filter(r=>['soft_tissue','tumor','vessel','membrane','functional_region'].includes(r.material)).map(r=>r.name):[]);
    this.syncSurfaces(surfaces,affine);
  }

  private updateSites(view:SurgicalView|undefined){
    disposeChildren(this.sites);
    for(const site of mappingSites(view)){
      const color=site.result==='model_estimate_response'?SCENARIO_COLORS.response:site.result==='model_estimate_no_response'?SCENARIO_COLORS.noResponse:SCENARIO_COLORS.unknown;
      // A numbered-tag-like marker at the site; an unknown result is hollow (a ring), never a reassuring colour.
      const tag=site.result==='unknown'?new THREE.Mesh(new THREE.TorusGeometry(.9,.22,8,24),new THREE.MeshBasicMaterial({color,depthTest:false,transparent:true,opacity:.95})):
        new THREE.Mesh(new THREE.SphereGeometry(site.latest?.9:.7,16,12),new THREE.MeshBasicMaterial({color,depthTest:false,transparent:true,opacity:.95}));
      tag.position.set(...site.position);tag.renderOrder=22;tag.raycast=()=>{};tag.name='scenario:mapping-site';
      this.sites.add(tag);
      if(site.reachMm&&site.reachMm>0){
        const shell=new THREE.Mesh(new THREE.SphereGeometry(site.reachMm,32,20),outlineMaterial(color,this.brainShift.uniforms,false));
        (shell.material as THREE.ShaderMaterial).uniforms.uOpacity.value=.55;
        shell.position.set(...site.position);shell.renderOrder=21;shell.raycast=()=>{};shell.name='scenario:stimulation-reach';
        this.sites.add(shell);
      }
    }
  }

  private updateWounds(view:SurgicalView|undefined,displace:((p:Pt)=>Pt)|undefined,inward?:Pt){
    disposeChildren(this.wounds);this.arterial=[];
    const outward=new THREE.Vector3(...(inward??view?.operativeExposureV1?.exteriorNormalRas?.map(v=>-v)??[0,0,-1] as Pt)).negate().normalize();
    const snap=this.woundSnap;
    const place=snap?(p:Pt)=>{const q=snap(p);return displace?displace(q):q;}:displace;
    for(const wound of woundEmitters(view,place,!!snap)){
      const size=Math.max(.8,Math.min(3,wound.calibreMm*1.2));
      if(wound.arterial){
        // A jet from the wound along the opening's outward normal, pulsing at the declared heart rate.
        const jet=new THREE.Mesh(new THREE.ConeGeometry(.45,2.4,12,1,true).translate(0,1.2,0),
          new THREE.MeshStandardMaterial({color:SCENARIO_COLORS.arterial,roughness:.18,metalness:0,emissive:'#3a0303',side:THREE.DoubleSide,depthTest:false,transparent:true,opacity:.95}));
        jet.renderOrder=23;
        jet.quaternion.setFromUnitVectors(new THREE.Vector3(0,1,0),outward);jet.position.set(...wound.position);
        jet.userData.baseScale=size;jet.scale.setScalar(size);jet.name=wound.kindUnknown?'scenario:wound-unknown-kind':'scenario:wound-arterial';
        jet.raycast=()=>{};this.wounds.add(jet);this.arterial.push(jet);
      }
      // Welling blood at every open wound (steady at venous ones).
      const pool=new THREE.Mesh(new THREE.SphereGeometry(.5,16,10),new THREE.MeshStandardMaterial({color:wound.arterial?'#8e1513':SCENARIO_COLORS.venous,roughness:.12,metalness:0,depthTest:false,transparent:true,opacity:.95}));
      // Blood wells up through the opening it makes, so the emitter stays visible over the tissue it came from.
      pool.renderOrder=23;
      pool.scale.set(size*1.4,size*1.4,size*.55);pool.quaternion.setFromUnitVectors(new THREE.Vector3(0,0,1),outward);
      pool.position.set(...wound.position);pool.raycast=()=>{};pool.name=wound.arterial?'scenario:wound-arterial-pool':'scenario:wound-venous';
      this.wounds.add(pool);
    }
  }

  /** Install the field on shifting surfaces and redraw the target contours when their geometry changes. */
  syncSurfaces(surfaces:ReadonlyMap<string,THREE.Mesh>,affine:Affine){
    for(const name of this.shiftingLayers){const mesh=surfaces.get(name);if(mesh)this.brainShift.installMesh(mesh);}
    for(const name of this.markedLayers){const mesh=surfaces.get(name);if(mesh)this.marks.install(mesh.material as THREE.Material);}
    const meshes=this.shiftActive&&this.frame?this.targetLayers.map(name=>surfaces.get(name)).filter((m):m is THREE.Mesh=>!!m):[];
    const key=meshes.length?JSON.stringify([meshes.map(m=>m.geometry.uuid),this.shiftKey]):'';
    if(key===this.contourKey)return;
    this.contourKey=key;disposeChildren(this.contours);
    if(!meshes.length)return;
    const contour=targetContour(meshes.map(m=>m.geometry.getAttribute('position').array as Float32Array),affine,this.frame!.inward);
    if(!contour)return;
    const shifted=contour.loop.map((p,i)=>{const d=this.displace?this.displace(contour.extremes[i]):[0,0,0];
      return [p[0]+d[0],p[1]+d[1],p[2]+d[2]] as Pt;});
    for(const [loop,color,name] of [[contour.loop,SCENARIO_COLORS.planned,'scenario:planned-target-outline'],[shifted,SCENARIO_COLORS.shifted,'scenario:shifted-target-outline']] as const){
      const curve=new THREE.CatmullRomCurve3(loop.map(p=>new THREE.Vector3(...p)),true);
      const tube=new THREE.Mesh(new THREE.TubeGeometry(curve,loop.length*3,.28,6,true),new THREE.MeshBasicMaterial({color,depthTest:false,depthWrite:false,transparent:true,opacity:.92}));
      tube.name=name;tube.renderOrder=20;tube.raycast=()=>{};this.contours.add(tube);
    }
  }

  setClipping(planes:THREE.Plane[]){
    for(const child of this.contours.children)((child as THREE.Mesh).material as THREE.Material).clippingPlanes=planes;
  }

  dispose(){
    disposeChildren(this.sites);disposeChildren(this.wounds);disposeChildren(this.contours);this.arterial=[];this.marks.dispose();
    this.contourKey='';this.brainShift.dispose();this.group.removeFromParent();
  }
}

/** Outline of the target seen along the opening axis: the largest radius in each
 * angular bin around the target's axis, softened around the loop, placed at the
 * target's shallowest depth. `extremes` are the target points that set each bin. */
export function targetContour(positions:readonly Float32Array[],affine:Affine,inward:Pt,bins=72){
  const n=new THREE.Vector3(...inward).normalize();
  const reference=Math.abs(n.z)>.9?new THREE.Vector3(1,0,0):new THREE.Vector3(0,0,1);
  const e1=new THREE.Vector3().crossVectors(reference,n).normalize(),e2=new THREE.Vector3().crossVectors(n,e1);
  const toRas=(x:number,y:number,z:number):Pt=>[0,1,2].map(a=>affine[a][0]*x+affine[a][1]*y+affine[a][2]*z+affine[a][3]) as Pt;
  const centre=[0,0,0];let count=0;
  for(const array of positions)for(let i=0;i<array.length;i+=3){const p=toRas(array[i],array[i+1],array[i+2]);centre[0]+=p[0];centre[1]+=p[1];centre[2]+=p[2];count++;}
  if(count<8)return null;
  const c=new THREE.Vector3(centre[0]/count,centre[1]/count,centre[2]/count);
  const radius=new Float64Array(bins).fill(NaN),extremes:Pt[]=Array(bins);let top=Infinity;
  for(const array of positions)for(let i=0;i<array.length;i+=3){
    const p=toRas(array[i],array[i+1],array[i+2]),d=new THREE.Vector3(...p).sub(c),u=d.dot(e1),v=d.dot(e2),r=Math.hypot(u,v);
    top=Math.min(top,d.dot(n));
    const bin=Math.floor(((Math.atan2(v,u)/(2*Math.PI))+1)%1*bins)%bins;
    if(!(r<=radius[bin])){radius[bin]=r;extremes[bin]=p;}
  }
  const known=[...radius.keys()].filter(i=>Number.isFinite(radius[i]));
  if(known.length<bins/3)return null;
  for(let i=0;i<bins;i++)if(!Number.isFinite(radius[i])){
    let a=i,b=i;while(!Number.isFinite(radius[(a+bins)%bins]))a--;while(!Number.isFinite(radius[b%bins]))b++;
    const t=(i-a)/(b-a);radius[i]=radius[(a+bins)%bins]*(1-t)+radius[b%bins]*t;extremes[i]=extremes[(t<.5?a+bins:b)%bins];
  }
  const smooth=Array.from(radius,(_,i)=>{let sum=0;for(let k=-2;k<=2;k++)sum+=radius[(i+k+bins)%bins];return sum/5;});
  const loop=smooth.map((r,i)=>{const angle=(i+.5)/bins*2*Math.PI;
    return c.clone().addScaledVector(e1,r*Math.cos(angle)).addScaledVector(e2,r*Math.sin(angle)).addScaledVector(n,top).toArray() as Pt;});
  return {loop,extremes,centre:c.toArray() as Pt,topDepthMm:top};
}

function disposeChildren(group:THREE.Group){
  for(const child of [...group.children]){
    child.removeFromParent();
    const mesh=child as THREE.Mesh;mesh.geometry?.dispose();(mesh.material as THREE.Material|undefined)?.dispose();
  }
}

/** Recorded injury on the source grid: contused cells (R) and lacerated, removed cells (G).
 * Surfaces sample it at their rest (source) position, so marks travel with shifted tissue. */
export function injuryMarkGrid(injury:InjuryDisplay|undefined,shape:readonly number[]|undefined){
  if(!injury||!shape||!(injury.contused.length||injury.lacerated.length))return null;
  const [,ny,nz]=shape,cells=[...injury.contused,...injury.lacerated];
  const low=[Infinity,Infinity,Infinity],high=[-Infinity,-Infinity,-Infinity];
  const ijk=(c:number)=>[Math.floor(c/(ny*nz)),Math.floor(c/nz)%ny,c%nz];
  for(const c of cells){const p=ijk(c);for(let a=0;a<3;a++){low[a]=Math.min(low[a],p[a]);high[a]=Math.max(high[a],p[a]);}}
  const origin=low.map(v=>v-2) as Pt,size=high.map((v,a)=>v-origin[a]+3) as Pt;
  if(size.some(n=>n>256))return null;
  const data=new Uint8Array(size[0]*size[1]*size[2]*4);
  const put=(c:number,channel:number)=>{const p=ijk(c).map((v,a)=>v-origin[a]);data[((p[0]*size[1]+p[1])*size[2]+p[2])*4+channel]=255;};
  injury.contused.forEach(c=>put(c,0));injury.lacerated.forEach(c=>put(c,1));
  return {data,origin,size};
}

export class TissueMarks {
  readonly uniforms={uMarksOn:{value:false},uMarks:{value:null as THREE.Data3DTexture|null},uMarksOrigin:{value:new THREE.Vector3()}};
  private readonly installed=new WeakSet<THREE.Material>();
  private key='';
  update(injury:InjuryDisplay|undefined,shape:readonly number[]|undefined){
    const key=injury?`${injury.contused.length}:${injury.lacerated.length}:${injury.contused.at(-1)}:${injury.lacerated.at(-1)}`:'';
    if(key===this.key)return;
    this.key=key;this.uniforms.uMarks.value?.dispose();this.uniforms.uMarks.value=null;
    const grid=injuryMarkGrid(injury,shape);
    if(grid){
      const texture=new THREE.Data3DTexture(grid.data,grid.size[2],grid.size[1],grid.size[0]);
      texture.format=THREE.RGBAFormat;texture.unpackAlignment=1;texture.minFilter=texture.magFilter=THREE.LinearFilter;texture.needsUpdate=true;
      this.uniforms.uMarks.value=texture;this.uniforms.uMarksOrigin.value.set(...grid.origin);
    }
    this.uniforms.uMarksOn.value=!!grid;
  }
  install(material:THREE.Material){
    if(!(material instanceof THREE.MeshStandardMaterial)||this.installed.has(material))return;
    this.installed.add(material);
    const previous=material.onBeforeCompile,key=material.customProgramCacheKey();
    material.onBeforeCompile=(shader,renderer)=>{
      previous.call(material,shader,renderer);
      Object.assign(shader.uniforms,this.uniforms);
      shader.vertexShader='varying vec3 vMarkVoxel;\n'+shader.vertexShader.replace('#include <begin_vertex>','#include <begin_vertex>\nvMarkVoxel=position;');
      shader.fragmentShader='uniform bool uMarksOn;uniform highp sampler3D uMarks;uniform vec3 uMarksOrigin;varying vec3 vMarkVoxel;\n'+shader.fragmentShader.replace('#include <color_fragment>',`#include <color_fragment>
        if(uMarksOn){
          vec3 uvw=(vMarkVoxel-uMarksOrigin+.5).zyx/vec3(textureSize(uMarks,0));
          vec2 mark=texture(uMarks,uvw).rg;
          // Bruise: contused cells stay in place, darkened toward plum.
          float bruise=smoothstep(.08,.45,mark.r);
          diffuseColor.rgb=mix(diffuseColor.rgb,diffuseColor.rgb*vec3(.5,.24,.36),.85*bruise);
          // Torn wall: a ragged dark-red margin where lacerated cells were torn out.
          float grain=fract(sin(dot(floor(vMarkVoxel*2.5),vec3(12.9898,78.233,37.719)))*43758.5453);
          float torn=smoothstep(.08,.4,mark.g)*(.45+.55*grain);
          diffuseColor.rgb=mix(diffuseColor.rgb,vec3(.30,.04,.04),torn);
        }`);
    };
    material.customProgramCacheKey=()=>key+'/injury-marks-v1';material.needsUpdate=true;
  }
  dispose(){this.uniforms.uMarks.value?.dispose();this.uniforms.uMarks.value=null;this.uniforms.uMarksOn.value=false;this.key='';}
}

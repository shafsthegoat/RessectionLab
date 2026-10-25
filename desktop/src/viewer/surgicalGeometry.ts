import * as THREE from 'three';
import type {SurgicalTool,SurgicalSlot} from '../surgical-types';
/** Only the parameters consumed by this display geometry belong in its key.
 * Position, orientation and activation are updated without allocating meshes. */
export function surgicalToolGeometryKey(tool:SurgicalTool,slot:SurgicalSlot,ghost=false):string {
  return JSON.stringify([tool.id,tool.radiusMm,tool.shaftRadiusMm,tool.workingLengthMm,
    slot.parameters.radiusMm??tool.radiusMm,slot.parameters.jawOpeningMm??2,ghost]);
}
export function updateSurgicalToolMesh(group:THREE.Group,slot:SurgicalSlot,ghost=false):void {
  group.position.set(...slot.position);group.quaternion.set(...slot.quaternion);
  group.userData.effectiveActive=!ghost&&slot.active;
  group.userData.recordedRpm=slot.parameters.rpm;
  const accent=group.userData.surgicalAccent as THREE.MeshStandardMaterial|undefined;
  accent?.color.set(ghost?'#95f4ce':slot.active?'#e0b35d':'#41677a');
}
/** Illustrative working-part motion only. The instrument's recorded root pose
 * never changes; paused review restores the commanded jaw geometry. */
export function updateSurgicalToolAnimation(group:THREE.Group,sourceSeconds:number|null,intervalProgress:number|null):void {
  const active=group.userData.effectiveActive&&sourceSeconds!==null;
  const bit=group.getObjectByName('instrument:drill-bit');
  const rpm=group.userData.recordedRpm as number|undefined;
  if(bit)bit.rotation.z=active&&typeof rpm==='number'&&Number.isFinite(rpm)&&rpm>0?(sourceSeconds!*rpm/60*2*Math.PI)%(2*Math.PI):0;
  // One illustrative close/release per observed interval, not a claimed jaw
  // frequency or additional cutting event. No motion on scrub or pause.
  const opening=active&&intervalProgress!==null ? .2+.8*Math.abs(Math.cos(Math.PI*intervalProgress)) : 1;
  for(const child of group.children)if(typeof child.userData.jawBaseX==='number'){
    child.position.x=child.userData.jawBaseX*opening;
    if(typeof child.userData.jawBaseAngle==='number')child.rotation.y=child.userData.jawBaseAngle*opening;
  }
}
/** Generic research geometry, tip at local origin and working axis along +Z. */
export function surgicalToolMesh(tool:SurgicalTool,slot:SurgicalSlot,ghost=false){
  const group=new THREE.Group(),radius=slot.parameters.radiusMm??tool.radiusMm;
  group.name=`instrument:${tool.id}`;
  const metal=new THREE.MeshStandardMaterial({color:ghost?'#95f4ce':'#a6c9d1',metalness:.5,roughness:.35,transparent:ghost,opacity:ghost?.35:1,depthWrite:!ghost});
  const accent=metal.clone();accent.color.set(ghost?'#95f4ce':slot.active?'#e0b35d':'#41677a');
  group.userData.surgicalAccent=accent;
  let parent=group;
  const add=(geometry:THREE.BufferGeometry,x=0,y=0,z=0,material=metal)=>{const mesh=new THREE.Mesh(geometry,material);mesh.position.set(x,y,z);parent.add(mesh);return mesh;};
  const cylinder=(r:number,length:number,x=0,y=0,z=0,material=metal)=>{const mesh=add(new THREE.CylinderGeometry(r,r,length,12),x,y,z,material);mesh.rotation.x=Math.PI/2;return mesh;};
  const box=(w:number,h:number,d:number,x=0,y=0,z=0,material=metal)=>add(new THREE.BoxGeometry(w,h,d),x,y,z,material);
  const sphere=(r:number,x=0,y=0,z=0,material=metal)=>add(new THREE.SphereGeometry(r,16,12),x,y,z,material);
  const tube=(r:number,z:number)=>{const tip=add(new THREE.TorusGeometry(r,Math.max(.12,r*.18),8,20),0,0,z);return tip;};
  const length=tool.workingLengthMm||40;
  cylinder(tool.shaftRadiusMm,length,0,0,-length/2-3);
  cylinder(tool.shaftRadiusMm*2.1,9,0,0,-length+1,accent);
  const jaw=Math.max(.1,slot.parameters.jawOpeningMm??2);
  switch(tool.id){
    case 'cranial_drill':{
      cylinder(radius*.7,4,0,0,-3,accent);
      const bit=new THREE.Group();bit.name='instrument:drill-bit';group.add(bit);parent=bit;
      sphere(radius);for(let i=0;i<4;i++){const ring=tube(radius*.94,0);ring.rotation.y=i*Math.PI/4;}
      // Asymmetric flute makes recorded rotational phase visible.
      box(radius*.18,radius*.35,radius*1.5,radius*.85,0,-radius*.3,accent);parent=group;break;}
    case 'cusa':
      cylinder(radius*.75,7,0,0,-4,accent);tube(radius,-.3);cylinder(radius*.3,3,0,0,-2);break;
    case 'brain_retractor':
      box(radius*2,.5,9,0,0,-4);box(radius*2,.6,2,0,.7,.1).rotation.x=-.55;break;
    case 'scalpel':{
      const outline=new THREE.Shape();outline.moveTo(-radius,-5);outline.lineTo(radius,-3);outline.lineTo(.2,1);outline.lineTo(-radius,-2);outline.closePath();
      const blade=add(new THREE.ExtrudeGeometry(outline,{depth:.25,bevelEnabled:false}));blade.rotation.x=Math.PI/2;break;}
    case 'periosteal_elevator':
      box(radius*2,.45,5,0,0,-2);sphere(radius*.55,0,0,.2).scale.set(1,.25,1);break;
    case 'craniotome':
      cylinder(radius,4,0,0,-2);box(radius*.55,radius*.55,6,radius*1.5,0,-2);box(radius*2.1,radius*.55,.6,radius*.6,0,1);break;
    case 'dural_scissors':
    case 'micro_scissors':{
      const bladeLength=tool.id==='dural_scissors'?6:4;
      for(const sign of [-1,1]){const blade=box(.35,.6,bladeLength,sign*jaw/3,0,-bladeLength/2+1);blade.rotation.y=sign*Math.atan(jaw/bladeLength);blade.userData.jawBaseX=blade.position.x;blade.userData.jawBaseAngle=blade.rotation.y;}
      sphere(.65,0,0,-4);if(tool.id==='dural_scissors')sphere(.7,-jaw/2,0,.4).scale.set(1,.45,1);break;}
    case 'dissector':
      cylinder(radius*.4,4,0,0,-2).rotation.x=Math.PI/2+.35;sphere(radius*.6,0,.6,.2);break;
    case 'frazier_suction':
      cylinder(radius*.85,6,0,0,-3);tube(radius,0);box(.8,.8,3,0,radius,-5,accent);break;
    case 'bipolar_forceps':
    case 'tumor_forceps':
      for(const sign of [-1,1]){
        const jawGroup=new THREE.Group();jawGroup.position.x=sign*jaw/2;jawGroup.userData.jawBaseX=jawGroup.position.x;group.add(jawGroup);parent=jawGroup;
        const arm=box(.45,.65,9,sign*.2,0,-4);arm.rotation.y=sign*-.08;
        if(tool.id==='bipolar_forceps')cylinder(.3,2,0,0,.3,accent);
        else {box(.8,.8,1.8,0,0,.2);for(let i=0;i<3;i++)box(.25,1,.2,0,0,-.5+i*.5,accent);}}
      parent=group;
      break;
    case 'irrigation':
      cylinder(radius*.7,5,0,0,-3);tube(radius*.7,-.4);cylinder(radius*.35,3,0,0,-.5,accent);break;
    case 'cortical_probe':
      for(const sign of [-1,1]){cylinder(.25,4,sign*radius*.65,0,-2);sphere(.45,sign*radius*.65,0,0,accent);}break;
    case 'subcortical_probe':
      cylinder(radius*.4,7,0,0,-3,accent);sphere(radius*.65);break;
    default:
      sphere(radius);break;
  }
  updateSurgicalToolMesh(group,slot,ghost);return group;
}

export type SurgicalEffectKind='contact'|'thermal'|'fluid'|'debris'|'cut'|'transection'|'obstruction';
export const SURGICAL_EFFECT_COLORS:Record<SurgicalEffectKind,string>={contact:'#54ded8',thermal:'#ff924f',fluid:'#479dff',debris:'#a99a82',cut:'#f4d16e',transection:'#f4d16e',obstruction:'#ff586a'};
/** RAS-mm authoritative markers shared by 3-D and MPR; two points denote a segment. */
export interface SurgicalEffectMarker {kind:SurgicalEffectKind;points:import('./coordinates').Point3[];value:number;color:string;stateId:string;radiusMm?:number;surface?:'grid-cut-face'}
/** Equivalent-volume glyph only, not a reconstructed fluid surface or flow law. */
export const fluidMarkerRadiusMm=(volumeMm3:number)=>Number.isFinite(volumeMm3)&&volumeMm3>0?Math.cbrt(volumeMm3*(3/(4*Math.PI))):0;
export const surgicalMarkerRadiusPx=(marker:SurgicalEffectMarker,pixelsPerMm:number)=>marker.radiusMm===undefined?2.5:marker.radiusMm*(Number.isFinite(pixelsPerMm)&&pixelsPerMm>0?pixelsPerMm:0);
export function surgicalEffectMarkers(view:import('../surgical-types').SurgicalView,outcome?:import('../surgical-types').SurgicalOutcome|null):SurgicalEffectMarker[]{
  const result:SurgicalEffectMarker[]=[],[nx,ny,nz]=view.scene.shape,a=view.scene.affine;
  if(![nx,ny,nz].every(n=>Number.isInteger(n)&&n>0)||a.length!==4||a.some(row=>row.length!==4||row.some(n=>!Number.isFinite(n))))return result;
  const count=nx*ny*nz,moved=new Map(view.moved);
  const point=(id:number,material:boolean):import('./coordinates').Point3|null=>{
    if(material)id=moved.get(id)??id;
    if(!Number.isInteger(id)||id<0||id>=count)return null;
    const x=Math.floor(id/(ny*nz)),y=Math.floor(id/nz)%ny,z=id%nz;
    return a.slice(0,3).map(row=>row[0]*x+row[1]*y+row[2]*z+row[3]) as import('./coordinates').Point3;
  };
  const add=(kind:SurgicalEffectKind,points:import('./coordinates').Point3[],value=1,surface?:'grid-cut-face')=>{if(points.length&&points.every(p=>p.length===3&&p.every(Number.isFinite))&&Number.isFinite(value)&&value>0)result.push({kind,points,value,color:SURGICAL_EFFECT_COLORS[kind],stateId:view.stateId,...(kind==='fluid'?{radiusMm:fluidMarkerRadiusMm(value)}:{}),...(surface?{surface}:{})});};
  for(const id of view.contact){const p=point(id,true);if(p)add('contact',[p]);}
  for(const [field,material] of [['thermal',true],['fluid',false],['debris',false]] as const){
    for(const [id,value] of Object.entries(view[field]??{})){const p=point(Number(id),material);if(p)add(field,[p],value);}
  }
  const boneProfile=view.physics?.profile?.bone as {ambientTemperatureC?:number}|undefined;
  const ambient=boneProfile?.ambientTemperatureC;
  if(Number.isFinite(ambient))for(const [id,temperature] of Object.entries(view.drillingV2?.temperatureC??{})){
    const p=point(Number(id),false);if(p&&temperature>(ambient??0)+.01)add('thermal',[p],temperature-(ambient??0));
  }
  const edgeKeys=new Set<string>();
  const indices=(id:number)=>[Math.floor(id/(ny*nz)),Math.floor(id/nz)%ny,id%nz];
  const removed=new Set(view.removed);
  const edges=(kind:'cut'|'transection',items:Array<[number,number]>)=>{for(const [left,right] of items){
    const key=`${kind}:${Math.min(left,right)}:${Math.max(left,right)}`;if(edgeKeys.has(key))continue;edgeKeys.add(key);
    const p=point(left,true),q=point(right,true);if(!p||!q)continue;
    if(kind==='transection'){add(kind,[p,q]);continue;}
    if(![left,right].every(id=>Number.isInteger(id)&&id>=0&&id<count))continue;
    const l=indices(left),r=indices(right),delta=r.map((v,i)=>v-l[i]);
    if(delta.reduce((sum,n)=>sum+Math.abs(n),0)!==1)continue;
    const normal=delta.findIndex(n=>n!==0),tangents=[0,1,2].filter(i=>i!==normal),center=l.map((v,i)=>(v+r[i])/2);
    const offsets=new Set<string>();
    for(const endpoint of [left,right]){
      if(removed.has(endpoint))continue;
      const original=indices(endpoint),destination=indices(moved.get(endpoint)??endpoint),offset=destination.map((v,i)=>v-original[i]);
      const offsetKey=offset.join(',');if(offsets.has(offsetKey))continue;offsets.add(offsetKey);
      const corners=([[-.5,-.5],[.5,-.5],[.5,.5],[-.5,.5],[-.5,-.5]]).map(([u,v])=>{
        const voxel=center.map((n,i)=>n+offset[i]);voxel[tangents[0]]+=u;voxel[tangents[1]]+=v;
        return a.slice(0,3).map(row=>row[0]*voxel[0]+row[1]*voxel[1]+row[2]*voxel[2]+row[3]) as import('./coordinates').Point3;
      });
      add('cut',corners,1,'grid-cut-face');
    }
  }};
  edges('cut',view.cutEdges??[]);edges('transection',view.transectedEdges??[]);
  for(const cut of view.cuts)edges('cut',cut.edges??[]);
  // Mesh edge IDs index retained vertices, not voxels. Their explicit seam points
  // are the only geometry consumed here, avoiding accidental grid reinterpretation.
  for(const seam of view.meshSeams??[])add(seam.kind==='retained_strand_break'?'transection':'cut',seam.points.map(p=>[...p]));
  const obstruction=(outcome?.events??view.history?.at(-1)?.events??[]).find(event=>event.kind==='contact_stop');
  if(obstruction&&Array.isArray(obstruction.position))add('obstruction',[obstruction.position as import('./coordinates').Point3]);
  return result;
}
/** Disposable THREE group for the same marker list used by linked MPR views. */
export function surgicalEffectGroup(view:import('../surgical-types').SurgicalView,preview=false,outcome?:import('../surgical-types').SurgicalOutcome|null):THREE.Group {
  const group=new THREE.Group();group.name=`surgical-effects:${view.stateId}`;group.userData={stateId:view.stateId,preview};
  const markers=surgicalEffectMarkers(view,outcome),kinds=Object.keys(SURGICAL_EFFECT_COLORS) as SurgicalEffectKind[];
  const scale=Math.max(.2,Math.min(...view.scene.resolutionMm));
  for(const kind of kinds){
    const points:number[]=[],diameters:number[]=[],segments:number[]=[],faces:number[]=[];
    for(const marker of markers){if(marker.kind!==kind)continue;
      if(marker.surface==='grid-cut-face'&&marker.points.length===5){for(const index of [0,1,2,0,2,3])faces.push(...marker.points[index]);}
      if(marker.points.length===1){points.push(...marker.points[0]);if(kind==='fluid')diameters.push(2*marker.radiusMm!);}
      else for(let i=1;i<marker.points.length;i++)segments.push(...marker.points[i-1],...marker.points[i]);
    }
    if(faces.length){const geometry=new THREE.BufferGeometry();geometry.setAttribute('position',new THREE.Float32BufferAttribute(faces,3));geometry.computeVertexNormals();
      const object=new THREE.Mesh(geometry,new THREE.MeshBasicMaterial({color:SURGICAL_EFFECT_COLORS[kind],side:THREE.DoubleSide,transparent:true,opacity:preview?.16:.28,depthWrite:false,polygonOffset:true,polygonOffsetFactor:-1,polygonOffsetUnits:-1}));object.name='effect:cut-faces';group.add(object);}
    if(points.length){const geometry=new THREE.BufferGeometry();geometry.setAttribute('position',new THREE.Float32BufferAttribute(points,3));
      const material=new THREE.PointsMaterial({color:SURGICAL_EFFECT_COLORS[kind],size:kind==='fluid'?1:scale,sizeAttenuation:true,transparent:true,opacity:preview?.45:.8,depthWrite:false});
      if(kind==='fluid'){
        geometry.setAttribute('fieldDiameterMm',new THREE.Float32BufferAttribute(diameters,1));
        material.onBeforeCompile=shader=>{shader.vertexShader='attribute float fieldDiameterMm;\n'+shader.vertexShader.replace('gl_PointSize = size;','gl_PointSize = size * fieldDiameterMm * projectionMatrix[1][1];');shader.fragmentShader=shader.fragmentShader.replace('#include <clipping_planes_fragment>','#include <clipping_planes_fragment>\nif (length(gl_PointCoord - vec2(0.5)) > 0.5) discard;');};
        material.customProgramCacheKey=()=> 'fluid-ledger-footprint-v1';
      }
      const object=new THREE.Points(geometry,material);object.name=`effect:${kind}`;group.add(object);}
    if(segments.length){const geometry=new THREE.BufferGeometry();geometry.setAttribute('position',new THREE.Float32BufferAttribute(segments,3));
      const object=new THREE.LineSegments(geometry,new THREE.LineBasicMaterial({color:SURGICAL_EFFECT_COLORS[kind],transparent:true,opacity:preview?.5:1,depthWrite:false}));object.name=`effect:${kind}`;group.add(object);}
  }
  return group;
}

/** Clip in RAS mm before projection, including faces with no vertex in the slab. */
export function clipSurgicalMarkersToSlab(markers:SurgicalEffectMarker[],axis:0|1|2,position:number,halfThickness:number):SurgicalEffectMarker[]{
  if(!Number.isFinite(position)||!Number.isFinite(halfThickness)||halfThickness<0)return [];
  type P=import('./coordinates').Point3;
  const low=position-halfThickness,high=position+halfThickness,out:SurgicalEffectMarker[]=[];
  const project=(p:P)=>p.map((n,i)=>i===axis?position:n) as P;
  const interpolate=(a:P,b:P,t:number)=>a.map((v,i)=>v+(b[i]-v)*t) as P;
  for(const marker of markers){
    if(marker.surface==='grid-cut-face'){
      let polygon=marker.points.slice(0,-1);
      for(const [bound,direction] of [[low,1],[high,-1]]){
        const clipped:P[]=[];
        for(let i=0;i<polygon.length;i++){
          const a=polygon[i],b=polygon[(i+1)%polygon.length],insideA=(a[axis]-bound)*direction>=0,insideB=(b[axis]-bound)*direction>=0;
          if(insideA)clipped.push(a);
          if(insideA!==insideB)clipped.push(interpolate(a,b,(bound-a[axis])/(b[axis]-a[axis])));
        }
        polygon=clipped;
      }
      if(polygon.length)out.push({...marker,points:[...polygon,polygon[0]].map(project)});
    }else if(marker.points.length===1){if(marker.points[0][axis]>=low&&marker.points[0][axis]<=high)out.push({...marker,points:marker.points.map(project)});
    }else for(let i=1;i<marker.points.length;i++){
      const a=marker.points[i-1],b=marker.points[i],delta=b[axis]-a[axis];let start=0,end=1;
      if(Math.abs(delta)<1e-12){if(a[axis]<low||a[axis]>high)continue;}
      else {const first=(low-a[axis])/delta,second=(high-a[axis])/delta;start=Math.max(0,Math.min(first,second));end=Math.min(1,Math.max(first,second));if(start>end)continue;}
      out.push({...marker,points:[project(interpolate(a,b,start)),project(interpolate(a,b,end))]});
    }
  }
  return out;
}

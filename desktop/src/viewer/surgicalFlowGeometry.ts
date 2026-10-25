import * as THREE from 'three';
import type {SurgicalFlowDisplay,SurgicalView} from '../surgical-types';
import {checkedFlowDisplayForView,MAX_FLOW_OPERATIONS} from '../surgical-flow-display.ts';

/** A display gain makes small, ledger-backed operations readable at surgical
 * scale. It is never fed back into a field, quantitative label or source state. */
export const FLOW_DROPLET_RADIUS_GAIN=3;
type FlowAnimation={mesh:THREE.InstancedMesh;time:{value:number};duration:number};
/** Retrospective interval evidence, not a fluid solver. No looping, velocities,
 * tip routes or extrapolation; zero-duration writes remain static glyphs. */
export function updateSurgicalFlowAnimation(group:THREE.Group,elapsedSeconds:number|null):boolean {
  const animation=group.userData.flowAnimation as FlowAnimation|undefined;
  if(!animation)return false;
  const active=elapsedSeconds!==null&&Number.isFinite(elapsedSeconds)&&elapsedSeconds>=0&&elapsedSeconds<animation.duration;
  animation.mesh.visible=active;animation.time.value=active?elapsedSeconds!:-1;
  for(const child of group.children)if(child!==animation.mesh&&child instanceof THREE.Mesh){
    const material=child.material as THREE.MeshBasicMaterial;material.opacity=active?.28:.9;
  }
  return active;
}

/** Fixed-size display glyphs, never fluid surfaces, velocities or mass packets.
 * Each operation uses at most three instances; all buffers are bounded by 256.
 * Straight segments exist only for recorded, adjacent native free-cell edges.
 * Other operations stay local: there is no invented route to/from a tool tip. */
export function surgicalFlowGroup(view:SurgicalView,input:SurgicalFlowDisplay):THREE.Group {
  const flow=checkedFlowDisplayForView(input,view)!;
  const affine=view.scene.affine;
  if(affine.length!==4||affine.some(row=>row.length!==4||row.some(n=>!Number.isFinite(n))))throw Error('Recorded field transfers require a finite scene affine.');
  const group=new THREE.Group();group.name='recorded-field-transfers';
  group.userData={stateId:flow.stateId,sceneId:flow.sceneId,actionIndex:flow.actionIndex,frameIndex:flow.frameIndex,startSeconds:flow.startSeconds,endSeconds:flow.endSeconds,truncated:flow.truncated};
  if(!flow.operations.length)return group;
  const [,,nz]=view.scene.shape,ny=view.scene.shape[1];
  const point=(id:number)=>{
    const xyz=[Math.floor(id/(ny*nz)),Math.floor(id/nz)%ny,id%nz];
    return new THREE.Vector3(...affine.slice(0,3).map(row=>row[0]*xyz[0]+row[1]*xyz[1]+row[2]*xyz[2]+row[3]) as [number,number,number]);
  };
  const spacing=[0,1,2].map(axis=>Math.hypot(affine[0][axis],affine[1][axis],affine[2][axis]));
  if(spacing.some(n=>!Number.isFinite(n)||n<=0))throw Error('Recorded field transfers require nondegenerate cell axes.');
  const scale=Math.min(...spacing),material=()=>new THREE.MeshBasicMaterial({color:'#ffffff',transparent:true,opacity:.9,depthWrite:false});
  const shaft=new THREE.InstancedMesh(new THREE.CylinderGeometry(.055*scale,.055*scale,1,6),material(),MAX_FLOW_OPERATIONS);
  const head=new THREE.InstancedMesh(new THREE.ConeGeometry(.15*scale,1,8),material(),MAX_FLOW_OPERATIONS);
  const local=new THREE.InstancedMesh(new THREE.OctahedronGeometry(.2*scale),material(),MAX_FLOW_OPERATIONS);
  shaft.name='flow:transfer-shafts';head.name='flow:transfer-heads';local.name='flow:local-operations';
  const identity=new THREE.Quaternion(),axis=new THREE.Vector3(0,1,0),matrix=new THREE.Matrix4();let transfers=0,markers=0;
  const color=(op:SurgicalFlowDisplay['operations'][number])=>new THREE.Color(op.field==='debris'?'#a99a82':op.kind==='deposit'&&op.provenance==='wound'?'#c84d5d':op.kind==='deposit'&&op.provenance==='irrigation'?'#66d8e8':'#479dff');
  for(const op of flow.operations){
    // An off-grid irrigation outlet has a measured escaped quantity but no
    // native cell. Retain its numeric record without inventing a location.
    if(op.kind==='escape'&&op.fromCell===null)continue;
    const tint=color(op);
    if(op.kind==='transfer'){
      const start=point(op.fromCell!),end=point(op.toCell!),delta=end.clone().sub(start),length=delta.length();
      const rotation=new THREE.Quaternion().setFromUnitVectors(axis,delta.clone().normalize());
      const tipLength=Math.min(.3*scale,length*.3),bodyLength=length-tipLength;
      matrix.compose(start.clone().addScaledVector(delta,bodyLength/(2*length)),rotation,new THREE.Vector3(1,bodyLength,1));shaft.setMatrixAt(transfers,matrix);shaft.setColorAt(transfers,tint);
      matrix.compose(end.clone().addScaledVector(delta,-tipLength/(2*length)),rotation,new THREE.Vector3(1,tipLength,1));head.setMatrixAt(transfers,matrix);head.setColorAt(transfers++,tint);
    }else {
      matrix.compose(point((op.kind==='deposit'?op.toCell:op.fromCell)!),identity,new THREE.Vector3(1,1,1));local.setMatrixAt(markers,matrix);local.setColorAt(markers++,tint);
    }
  }
  for(const [object,count] of [[shaft,transfers],[head,transfers],[local,markers]] as const){
    if(count){object.count=count;object.instanceMatrix.needsUpdate=true;if(object.instanceColor)object.instanceColor.needsUpdate=true;
      object.frustumCulled=false;object.raycast=()=>{};group.add(object);
    }else {object.geometry.dispose();object.material.dispose();object.dispose();}
  }
  const located=flow.operations.filter(op=>!(op.kind==='escape'&&op.fromCell===null)&&op.endSeconds>op.startSeconds);
  if(located.length){
    const geometry=new THREE.SphereGeometry(1,10,6),times:number[]=[],deltas:number[]=[],radii:number[]=[],kinds:number[]=[];
    // Depth-independent evidence overlay: tiny source-cell operations stay
    // readable behind context anatomy. The ordinary cutaway still clips it.
    const dropletMaterial=new THREE.MeshBasicMaterial({color:'#ffffff',transparent:true,opacity:.92,depthWrite:false,depthTest:false});
    const droplets=new THREE.InstancedMesh(geometry,dropletMaterial,located.length),time={value:-1};
    droplets.name='flow:ledger-droplets';droplets.visible=false;droplets.renderOrder=10;droplets.frustumCulled=false;droplets.raycast=()=>{};
    located.forEach((op,i)=>{
      const start=point((op.kind==='deposit'?op.toCell:op.fromCell)!),end=op.kind==='transfer'?point(op.toCell!):start;
      matrix.compose(start,identity,new THREE.Vector3(1,1,1));droplets.setMatrixAt(i,matrix);droplets.setColorAt(i,color(op));
      times.push(op.startSeconds-flow.startSeconds,op.endSeconds-flow.startSeconds);deltas.push(...end.clone().sub(start).toArray());
      radii.push(FLOW_DROPLET_RADIUS_GAIN*Math.cbrt(3*op.volumeMm3/(4*Math.PI)));kinds.push(op.kind==='deposit'?1:op.kind==='transfer'?0:2);
    });
    geometry.setAttribute('operationTimes',new THREE.InstancedBufferAttribute(new Float32Array(times),2));
    geometry.setAttribute('operationDelta',new THREE.InstancedBufferAttribute(new Float32Array(deltas),3));
    geometry.setAttribute('operationRadius',new THREE.InstancedBufferAttribute(new Float32Array(radii),1));
    geometry.setAttribute('operationKind',new THREE.InstancedBufferAttribute(new Float32Array(kinds),1));
    dropletMaterial.onBeforeCompile=shader=>{
      shader.uniforms.recordedFlowTime=time;
      shader.vertexShader='uniform float recordedFlowTime;\nattribute vec2 operationTimes;\nattribute vec3 operationDelta;\nattribute float operationRadius;\nattribute float operationKind;\n'+shader.vertexShader;
      shader.vertexShader=shader.vertexShader.replace('#include <begin_vertex>',`float p = clamp((recordedFlowTime-operationTimes.x)/(operationTimes.y-operationTimes.x),0.0,1.0);
        float alive = step(operationTimes.x,recordedFlowTime)*(1.0-step(operationTimes.y,recordedFlowTime));
        float fraction = operationKind < 0.5 ? 1.0 : operationKind < 1.5 ? p : 1.0-p;
        vec3 transformed = position*operationRadius*pow(max(0.0,fraction),1.0/3.0)*alive + operationDelta*p;`);
    };
    dropletMaterial.customProgramCacheKey=()=> 'recorded-ledger-droplets-v1';
    droplets.instanceMatrix.needsUpdate=true;if(droplets.instanceColor)droplets.instanceColor.needsUpdate=true;
    group.userData.flowAnimation={mesh:droplets,time,duration:flow.endSeconds-flow.startSeconds} satisfies FlowAnimation;
    group.userData.dropletRadiusGain=FLOW_DROPLET_RADIUS_GAIN;group.add(droplets);
  }
  return group;
}

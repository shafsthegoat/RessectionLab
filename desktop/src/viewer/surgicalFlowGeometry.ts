import * as THREE from 'three';
import type {SurgicalFlowDisplay,SurgicalView} from '../surgical-types';
import {checkedFlowDisplayForView,MAX_FLOW_OPERATIONS} from '../surgical-flow-display.ts';

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
  return group;
}

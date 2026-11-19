import * as THREE from 'three';
import type {OperativeField,OperativePhase} from '../operative-field';
import type {ViewerVolume} from './contracts';
import {sampleTrilinear,type Point3} from './coordinates.ts';

export type OperativeViewChoice='follow'|'exterior'|'section';

/** Only reveal a section automatically when the modeled station lies beneath
 * an overlying source layer. A shallow surface lesion can stay in exterior view. */
export function operativeSectionVisible(field:OperativeField,phase:OperativePhase,choice:OperativeViewChoice){
  if(choice!=='follow')return choice==='section';
  if(phase!=='brain'&&phase!=='tumor')return false;
  const surface=field.stations.bone??field.stations.dura??field.stations.brain;
  const station=field.stations[phase];
  return surface!==null&&station!==null&&surface-station>Math.max(3*field.spacingMm,field.routeRadiusMm);
}

/** Extend the cut face with the acquisition that supplied the operative contrast.
 * No original scan pixels are drawn through the operated field. */
export function operativeSectionContext(volume:ViewerVolume|null,collectionId:string,field:OperativeField,sectionY:number){
  const signal=field.intensity;
  if(!signal||!volume?.displayIdentity||volume.sourceFrameIssue)return null;
  const [collection,,,revision,raw]=JSON.parse(volume.displayIdentity);
  if(collection!==collectionId||revision!==field.snapshotId||raw!==false)return null;
  const source=volume.compartments.find(layer=>
    layer.appearanceSource?.sourceSha256===signal.sourceSha256&&layer.appearanceSource.seriesId===signal.sourceSeries
    &&['reference','reviewed','source-provided'].includes(layer.alignment??''))?.appearanceSource;
  if(!source)return null;
  const basis=field.basisRas.map(axis=>new THREE.Vector3(...axis));
  const target=new THREE.Vector3(...field.targetRasMm);
  const voxelToRas=new THREE.Matrix4().fromArray(source.affine.flat()).transpose();
  const rasToVoxel=voxelToRas.clone().invert();
  const bounds=new THREE.Box2();
  for(const x of [0,source.shape[0]-1])for(const y of [0,source.shape[1]-1])for(const z of [0,source.shape[2]-1]){
    const delta=new THREE.Vector3(x,y,z).applyMatrix4(voxelToRas).sub(target);
    bounds.expandByPoint(new THREE.Vector2(delta.dot(basis[0]),delta.dot(basis[2])));
  }
  const size=bounds.getSize(new THREE.Vector2());
  const pitch=Math.max(.5,size.x/511,size.y/511);
  const width=Math.ceil(size.x/pitch)+1,height=Math.ceil(size.y/pitch)+1;
  const data=new Uint8Array(width*height*4);
  const p=new THREE.Vector3(),voxel:Point3=[0,0,0];
  const [low,high]=signal.window,half=field.spacingMm*.5;
  const minX=field.origin[0]-half,maxX=minX+field.shape[0]*field.spacingMm;
  const minZ=field.origin[2]-half,maxZ=minZ+field.shape[2]*field.spacingMm;
  for(let row=0;row<height;row++)for(let col=0;col<width;col++){
    const x=bounds.min.x+col*pitch,z=bounds.min.y+row*pitch;
    if(x>=minX&&x<=maxX&&z>=minZ&&z<=maxZ)continue;
    p.copy(target).addScaledVector(basis[0],x).addScaledVector(basis[1],sectionY).addScaledVector(basis[2],z).applyMatrix4(rasToVoxel);
    p.toArray(voxel);
    if(voxel.some((n,i)=>n<0||n>source.shape[i]-1))continue;
    const sample=sampleTrilinear(source.data,source.shape,voxel);
    if(!Number.isFinite(sample))continue;
    const gray=Math.round(255*THREE.MathUtils.clamp((sample-low)/(high-low),0,1));
    const i=(row*width+col)*4;data.set([gray,gray,gray,255],i);
  }
  const texture=new THREE.DataTexture(data,width,height);
  texture.colorSpace=THREE.SRGBColorSpace;
  texture.magFilter=texture.minFilter=THREE.LinearFilter;texture.needsUpdate=true;
  const material=new THREE.MeshBasicMaterial({map:texture,alphaTest:.99,side:THREE.DoubleSide,toneMapped:false});
  const mesh=new THREE.Mesh(new THREE.PlaneGeometry(width*pitch,height*pitch),material);
  mesh.rotation.x=Math.PI/2;
  mesh.position.set(bounds.min.x+(width-1)*pitch/2,sectionY,bounds.min.y+(height-1)*pitch/2);
  mesh.name=signal.modality+' · intact source section';
  mesh.userData.sectionSource={series:signal.sourceSeries,sha256:signal.sourceSha256,signalSha256:source.signalSha256,sectionY,pitch,width,height};
  mesh.userData.acquiredSection={minX,maxX,minZ,maxZ};
  return mesh;
}

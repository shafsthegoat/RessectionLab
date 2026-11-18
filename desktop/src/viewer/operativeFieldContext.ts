import * as THREE from 'three';
import type {ViewerVolume} from './contracts';
import type {OperativeField} from '../operative-field';
import {rasAffine} from './coordinates.ts';
import {placeInSourceFrame} from './sceneGeometry.ts';

export function operativeFieldContext(volume:ViewerVolume|null,field:OperativeField,collectionId:string,patchRadius=field.radiusMm){
  const group=new THREE.Group();
  if(!volume?.displayIdentity||volume.sourceFrameIssue)return group;
  const identity=JSON.parse(volume.displayIdentity);
  if(identity[0]!==collectionId||identity[3]!==field.snapshotId||identity[4]!==false)return group;
  const half=field.spacingMm*.5;
  const low=new THREE.Vector3(...field.origin).addScalar(-half);
  const high=low.clone().addScaledVector(new THREE.Vector3(...field.shape),field.spacingMm);
  const toLocal=field.basisRas.map(axis=>[...axis,-axis.reduce((d,n,i)=>d+n*field.targetRasMm[i],0)]);
  placeInSourceFrame(group,[...toLocal,[0,0,0,1]]);
  // The surrounding surface and the operated patch occupy disjoint physical
  // regions. Keep any source-supported discontinuity at the crop boundary.
  for(const role of ['bone','brain'] as const){
    const layers=field.layers.filter(l=>l.role===role&&l.sampledVoxels>0&&!l.detail);
    if(layers.length!==1||layers[0].adjudication)continue;
    const item=layers[0],resident=volume.compartments.find(l=>l.name===item.id);
    if(!resident?.surface?.indices.length||resident.displayFilter||
      (resident.sourceSupport&&resident.sourceSupport.sampledDataSha256!==item.sampledDataSha256))continue;
    const {positions,normals,indices}=resident.surface;
    const geometry=new THREE.BufferGeometry();
    geometry.setAttribute('position',new THREE.BufferAttribute(positions,3));
    geometry.setAttribute('normal',new THREE.BufferAttribute(normals,3));
    geometry.setIndex(new THREE.BufferAttribute(indices,1));
    const material=new THREE.MeshStandardMaterial({color:role==='bone'?'#e5d8bc':'#c68788',roughness:role==='bone'?.76:.6,side:THREE.DoubleSide});
    material.onBeforeCompile=shader=>{
      Object.assign(shader.uniforms,{uLocalLow:{value:low},uLocalHigh:{value:high},uPatchRadius:{value:patchRadius}});
      shader.vertexShader='varying vec3 vOperativeLocal;\n'+shader.vertexShader.replace('#include <begin_vertex>',
        '#include <begin_vertex>\nvOperativeLocal=(modelMatrix*vec4(position,1.0)).xyz;');
      shader.fragmentShader='uniform vec3 uLocalLow,uLocalHigh; uniform float uPatchRadius; varying vec3 vOperativeLocal;\n'+shader.fragmentShader.replace('#include <clipping_planes_fragment>',
        '#include <clipping_planes_fragment>\nif(length(vOperativeLocal.xy)<=uPatchRadius && vOperativeLocal.z>=uLocalLow.z && vOperativeLocal.z<=uLocalHigh.z)discard;');
    };
    const mesh=new THREE.Mesh(geometry,material);
    placeInSourceFrame(mesh,rasAffine(volume.affine,volume.frame));
    mesh.name=item.label;mesh.userData.sourceLayer={directId:item.id,role,patchRadius};
    group.add(mesh);
  }
  return group;
}

/** Limit only the local copy of a layer with a verified wider counterpart.
 * The cutout uses rest coordinates, so the flap remains attached to its scan. */
export function limitToOperativePatch(material:THREE.Material,radius:number,exterior:THREE.IUniform<boolean>){
  const previous=material.onBeforeCompile.bind(material),key=material.customProgramCacheKey();
  material.onBeforeCompile=(shader,renderer)=>{
    previous(shader,renderer);
    Object.assign(shader.uniforms,{uExteriorPatch:exterior,uPatchRadius:{value:radius}});
    shader.vertexShader='varying vec2 vPatchXY;\n'+shader.vertexShader.replace('#include <begin_vertex>',
      '#include <begin_vertex>\nvPatchXY=position.xy;');
    shader.fragmentShader='uniform bool uExteriorPatch; uniform float uPatchRadius; varying vec2 vPatchXY;\n'+shader.fragmentShader.replace('#include <clipping_planes_fragment>',
      '#include <clipping_planes_fragment>\nif(uExteriorPatch && length(vPatchXY)>uPatchRadius)discard;');
  };
  material.customProgramCacheKey=()=>key+'/local-patch';
}

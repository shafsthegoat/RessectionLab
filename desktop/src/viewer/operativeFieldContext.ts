import * as THREE from 'three';
import {matchingOperativeSource,operativeSourceIdentity} from './operativeSourceIdentity.ts';
import type {ViewerVolume} from './contracts';
import type {OperativeField} from '../operative-field';
import {rasAffine,sampleTrilinear,type Point3} from './coordinates.ts';
import {placeInSourceFrame} from './sceneGeometry.ts';

export function operativeFieldContext(volume:ViewerVolume|null,field:OperativeField,collectionId:string,patchRadius=field.radiusMm){
  const group=new THREE.Group();
  const identity=operativeSourceIdentity(volume);
  if(!volume||!identity||identity.collectionId!==collectionId||identity.snapshotId!==field.snapshotId)return group;
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
    if(!resident?.surface?.indices.length||!matchingOperativeSource(item,resident))continue;
    const {positions,normals,indices}=resident.surface;
    const geometry=new THREE.BufferGeometry();
    geometry.setAttribute('position',new THREE.BufferAttribute(positions,3));
    geometry.setAttribute('normal',new THREE.BufferAttribute(normals,3));
    geometry.setIndex(new THREE.BufferAttribute(indices,1));
    const material=new THREE.MeshStandardMaterial({color:role==='bone'?'#e5d8bc':'#c68788',roughness:role==='bone'?.76:.6,side:THREE.DoubleSide});
    // Geometry uses the inspected scan's lattice. Intensities belong to
    // this tissue's acquisition, which can have a different grid and pose.
    const source=resident.appearanceSource;
    const shared=volume.patientContext?.sources.find(s=>s.seriesId===item.sourceSeries&&s.sourceSha256===source?.sourceSha256);
    const local=field.intensity;
    const window=local&&local.sourceSeries===item.sourceSeries&&local.sourceSha256===source?.sourceSha256
      ?local.window:shared?.displayWindow;
    const appearance=source&&source.seriesId===item.sourceSeries&&window?{
      seriesId:source.seriesId,modality:source.modality,sourceSha256:source.sourceSha256,signalSha256:source.signalSha256,
      sourceShape:source.nativeShape,sourceVoxelToPatientRasMm:source.nativeAffine,window
    }:undefined;
    if(source&&appearance){
      const toSource=new THREE.Matrix4().fromArray(source.affine.flat()).transpose().invert()
        .multiply(new THREE.Matrix4().fromArray(rasAffine(volume.affine,volume.frame).flat()).transpose());
      const [lo,hi]=appearance.window;
      const values=new Uint8Array(positions.length/3);
      const p=new THREE.Vector3();
      for(let i=0;i<values.length;i++){
        const voxel=p.fromArray(positions,3*i).applyMatrix4(toSource).toArray() as Point3;
        const covered=voxel.every((n,axis)=>n>=0&&n<=source.shape[axis]-1);
        const normalized=covered?(sampleTrilinear(source.data,source.shape,voxel)-lo)/(hi-lo):.5;
        values[i]=Math.round(255*THREE.MathUtils.clamp(normalized,0,1));
      }
      geometry.setAttribute('scanSignal',new THREE.BufferAttribute(values,1,true));
    }
    material.onBeforeCompile=shader=>{
      if(geometry.hasAttribute('scanSignal')){
        shader.vertexShader='attribute float scanSignal; varying float vScanSignal;\n'+shader.vertexShader.replace('#include <begin_vertex>',
          '#include <begin_vertex>\nvScanSignal=scanSignal;');
        shader.fragmentShader='varying float vScanSignal;\n'+shader.fragmentShader.replace('#include <color_fragment>',
          `#include <color_fragment>\ndiffuseColor.rgb*=exp2((vScanSignal-.5)*${role==='bone'?'1.26':'2.59'});`);
      }
      Object.assign(shader.uniforms,{uLocalLow:{value:low},uLocalHigh:{value:high},uPatchRadius:{value:patchRadius}});
      shader.vertexShader='varying vec3 vOperativeLocal;\n'+shader.vertexShader.replace('#include <begin_vertex>',
        '#include <begin_vertex>\nvOperativeLocal=(modelMatrix*vec4(position,1.0)).xyz;');
      shader.fragmentShader='uniform vec3 uLocalLow,uLocalHigh; uniform float uPatchRadius; varying vec3 vOperativeLocal;\n'+shader.fragmentShader.replace('#include <clipping_planes_fragment>',
        '#include <clipping_planes_fragment>\nif(length(vOperativeLocal.xy)<=uPatchRadius && vOperativeLocal.z>=uLocalLow.z && vOperativeLocal.z<=uLocalHigh.z)discard;');
    };
    material.customProgramCacheKey=()=>`operative-context-${role}-${geometry.hasAttribute('scanSignal')}`;
    const mesh=new THREE.Mesh(geometry,material);
    placeInSourceFrame(mesh,rasAffine(volume.affine,volume.frame));
    mesh.name=item.label;mesh.userData.sourceLayer={directId:item.id,role,patchRadius,sourceSeries:item.sourceSeries,appearance};
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

/** An already visible source-supported external tissue class can establish the
 * planned entry before the registered skull is exposed. */
export function operativeEntryCover(volume:ViewerVolume|null,field:OperativeField,collectionId:string,visible:Readonly<Record<string,boolean>>|undefined){
  const identity=operativeSourceIdentity(volume),signal=field.intensity;
  if(!identity||!volume||!signal||identity.collectionId!==collectionId||identity.snapshotId!==field.snapshotId)return null;
  const candidates=volume.compartments.filter(layer=>layer.displayKind==='outer-head-tissue'&&visible?.[layer.name]===true&&
    layer.sourceSeries===signal.sourceSeries&&layer.sourceSupport?.status==='supported'&&
    layer.sourceSupport.sourceSha256===signal.sourceSha256&&!layer.displayFilter&&
    ['reference','source-provided','reviewed'].includes(layer.alignment??'')&&layer.surface?.indices.length);
  if(candidates.length!==1)return null;
  const layer=candidates[0],surface=layer.surface!;
  const geometry=new THREE.BufferGeometry();
  geometry.setAttribute('position',new THREE.BufferAttribute(surface.positions,3));
  geometry.setAttribute('normal',new THREE.BufferAttribute(surface.normals,3));
  geometry.setIndex(new THREE.BufferAttribute(surface.indices,1));
  const material=new THREE.MeshStandardMaterial({color:'#b9ada4',roughness:.85,side:THREE.DoubleSide,transparent:true});
  const mesh=new THREE.Mesh(geometry,material);
  placeInSourceFrame(mesh,rasAffine(volume.affine,volume.frame));
  mesh.name=layer.label??'Outer head tissue';
  mesh.userData.sourceLayer={directId:layer.name,role:'outer-head-tissue',sourceSeries:layer.sourceSeries,
    contextLayer:{id:layer.name,label:mesh.name,sourceSeries:layer.sourceSeries,evidence:'Source-derived estimate',reviewStatus:layer.qa?.reviewStatus},
    appearance:{seriesId:signal.sourceSeries,modality:signal.modality,sourceSha256:signal.sourceSha256,
      sourceVoxelToPatientRasMm:signal.sourceVoxelToPatientRasMm,sourceShape:signal.sourceShape}};
  return mesh;
}

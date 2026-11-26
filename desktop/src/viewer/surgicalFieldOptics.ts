import * as THREE from 'three';
import {joinedFreeField,type FieldSample} from './freeFieldSurface.ts';
import type {SurgicalView} from '../surgical-types';
import type {Point3} from './coordinates';
import {placeInSourceFrame} from './sceneGeometry.ts';

/** Display the current native free-cell inventories. Reservoirs have no spatial
 * extent and total modeled blood is never apportioned among mixed field cells. */
function samples(field:Record<string,number>|undefined,shape:Point3):FieldSample[] {
  const [nx,ny,nz]=shape,result:FieldSample[]=[];
  for(const [key,amount] of Object.entries(field??{})){
    const cell=Number(key);
    if(Number.isSafeInteger(cell)&&cell>=0&&cell<nx*ny*nz&&Number.isFinite(amount)&&amount>0)
      result.push({cell:[Math.floor(cell/(ny*nz)),Math.floor(cell/nz)%ny,cell%nz],amount});
  }
  return result;
}
function woundOnlyField(view:SurgicalView):boolean{
  if(!view.operativeExposureV1||!view.sources?.length||view.sources.some(s=>s.kind!=='modeled-vessel-wound'))return false;
  const added=view.metrics.fluidAddedMm3,blood=view.metrics.modeledVesselFluidGeneratedMm3;
  // With an initially empty generated field and a single kind of input, all
  // occupied free cells have that origin. Following irrigation the ledger
  // carries a mixture with unknown per-cell proportions.
  return typeof added==='number'&&typeof blood==='number'&&blood>0&&Math.abs(added-blood)<=1e-10*Math.max(1,added);
}
/** Moist lumen wall of a centerline-tube vessel: a broad soft film highlight
 * along the tube with a faint tighter glint from the wall beneath. */
export const wetVesselFinish={roughness:.24,specularIntensity:.42,clearcoat:1,clearcoatRoughness:.3,sheen:0};
/** Tube wall albedo: the rooted display's vessel red (#aa352b) lifted to the volume-lighting
 * vessel albedo, which reads the same once lit. Veins take the same colour. */
export const tubeVesselColors={artery:'#c4553a',vein:'#c4553a'} as const;

export function applyWetVesselFinish(material:THREE.Material){
  if(!(material instanceof THREE.MeshPhysicalMaterial))return false;
  Object.assign(material,wetVesselFinish);
  material.userData.wetVessel=true;
  return true;
}

/** Colour a tube wall by the vein flag of each vertex. Rest and edited walls
 * carry classes aligned with their own vertices; without them the wall is artery-coloured. */
export function applyTubeVesselColor(mesh:THREE.Mesh,classes?:Uint8Array){
  const material=mesh.material as THREE.MeshStandardMaterial,geometry=mesh.geometry;
  const count=geometry.getAttribute('position')?.count??0;
  const veins=classes?classes.reduce((sum,value)=>sum+value,0):0;
  if(classes&&classes.length!==count)classes=undefined;
  const mixed=!!classes&&veins>0&&veins<count;
  if(mixed){
    const artery=new THREE.Color(tubeVesselColors.artery),vein=new THREE.Color(tubeVesselColors.vein),colors=new Float32Array(count*3);
    classes!.forEach((value,i)=>(value?vein:artery).toArray(colors,i*3));
    geometry.setAttribute('color',new THREE.BufferAttribute(colors,3));
    material.color.set('#ffffff');
  }else{
    if(geometry.hasAttribute('color'))geometry.deleteAttribute('color');
    material.color.set(classes&&veins*2>classes.length?tubeVesselColors.vein:tubeVesselColors.artery);
  }
  if(material.vertexColors!==mixed){material.vertexColors=mixed;material.needsUpdate=true;}
}

export function retainedFieldOptics(view:SurgicalView) {
  const fluid=samples(view.fluid,view.scene.shape),debris=samples(view.debris,view.scene.shape);
  if(!fluid.length&&!debris.length)return null;
  const grid=new THREE.Matrix4().fromArray(view.scene.affine.flat()).transpose();
  const capacity=Math.abs(grid.determinant());
  if(!Number.isFinite(capacity)||capacity<1e-12)return null;
  const bloodOnly=woundOnlyField(view);
  const group=new THREE.Group();
  group.name='effect:retained-field';
  const joined=joinedFreeField(fluid,view.scene.shape,capacity);
  const [,ny,nz]=view.scene.shape;
  // Isolated deposits remain volume-scaled. Adjacent free cells share one
  // neutral optical surface. A confined or overfilled cell adds no extra reach.
  for(const [name,entries] of [['fluid',fluid],['debris',debris]] as const){
    if(!entries.length)continue;
    const liquid=name==='fluid';
    const geometry=liquid?new THREE.SphereGeometry(.5,12,8):new THREE.IcosahedronGeometry(.5);
    const material=new THREE.MeshStandardMaterial({color:liquid?(bloodOnly?'#8d3933':'#a8bdc1'):'#bbaa8b',roughness:liquid?.14:.86,
      metalness:0,transparent:liquid,opacity:liquid?(bloodOnly?.80:.57):1,depthWrite:!liquid});
    if(liquid&&joined){
      const pool=new THREE.Mesh(joined.geometry,material.clone());pool.name='field:free-surface';
      placeInSourceFrame(pool,view.scene.affine);
      pool.receiveShadow=true;pool.raycast=()=>{};group.add(pool);
    }
    const mesh=new THREE.InstancedMesh(geometry,material,entries.length);
    mesh.name='field:'+name;mesh.raycast=()=>{};mesh.receiveShadow=true;
    placeInSourceFrame(mesh,view.scene.affine);
    const matrix=new THREE.Matrix4();
    entries.forEach(({cell,amount},index)=>{
      const inPool=liquid&&joined?.joined.has((cell[0]*ny+cell[1])*nz+cell[2]);
      const diameter=inPool?0:Math.cbrt(Math.min(1,amount/capacity));
      matrix.makeScale(diameter,diameter,diameter).setPosition(...cell);
      mesh.setMatrixAt(index,matrix);
    });
    mesh.visible=!liquid||joined?.joined.size!==entries.length;
    mesh.instanceMatrix.needsUpdate=true;group.add(mesh);
  }
  const lower:Point3=[Infinity,Infinity,Infinity],upper:Point3=[-Infinity,-Infinity,-Infinity];
  for(const {cell} of fluid)for(let axis=0;axis<3;axis++){
    lower[axis]=Math.min(lower[axis],cell[axis]);upper[axis]=Math.max(upper[axis],cell[axis]);
  }
  const size=upper.map((v,i)=>v-lower[i]+3) as Point3; // one dry texel border
  const bytes=size[0]*size[1]*size[2];
  let texture:THREE.Data3DTexture|null=null;
  if(fluid.length&&size.every(n=>n<=256)&&bytes<=4*1024*1024){
    const data=new Uint8Array(bytes);
    for(const {cell,amount} of fluid){
      const [x,y,z]=cell.map((n,a)=>n-lower[a]+1);
      data[(x*size[1]+y)*size[2]+z]=Math.round(255*Math.min(1,amount/capacity));
    }
    texture=new THREE.Data3DTexture(data,size[2],size[1],size[0]);
    texture.format=THREE.RedFormat;texture.unpackAlignment=1;
    texture.minFilter=texture.magFilter=THREE.LinearFilter;texture.needsUpdate=true;
  }
  group.userData.fieldSummary={stateId:view.stateId,...(view.operativeExposureV1?{fluidOrigin:bloodOnly?'vascular sources only':'mixed or unknown'}:{}),fluidCells:fluid.length,debrisCells:debris.length,
    fluidMm3:fluid.reduce((sum,item)=>sum+item.amount,0),debrisMm3:debris.reduce((sum,item)=>sum+item.amount,0),
    wetTextureBytes:texture?bytes:0,joinedFluidCells:joined?.joined.size??0};
  return {group,texture,worldToCell:grid.clone().invert(),origin:texture?new THREE.Vector3(...lower).subScalar(1):new THREE.Vector3()};
}
type Field=ReturnType<typeof retainedFieldOptics>;
export class SurgicalWetting {
  private readonly active={value:false};
  private readonly density={value:null as THREE.Data3DTexture|null};
  private readonly worldToCell={value:new THREE.Matrix4()};
  private readonly origin={value:new THREE.Vector3()};
  private readonly installed=new WeakSet<THREE.MeshStandardMaterial>();

  update(field:Field,enabled:boolean){
    if(field?.texture){
      this.worldToCell.value.copy(field.worldToCell);
      this.origin.value.copy(field.origin);
    }
    this.density.value=field?.texture??null;this.active.value=!!this.density.value&&enabled;
  }

  install(material:THREE.MeshStandardMaterial){
    if(this.installed.has(material))return;
    this.installed.add(material);
    const previous=material.onBeforeCompile,key=material.customProgramCacheKey();
    material.onBeforeCompile=(shader,renderer)=>{
      previous.call(material,shader,renderer);
      Object.assign(shader.uniforms,{uWetVisible:this.active,uWetCell:this.worldToCell,uWetDensity:this.density,uWetOrigin:this.origin});
      shader.vertexShader='uniform mat4 uWetCell;varying vec3 vWetCell;\n'+shader.vertexShader.replace(
        '#include <worldpos_vertex>','#include <worldpos_vertex>\nvWetCell=(uWetCell*modelMatrix*vec4(transformed,1.)).xyz;');
      shader.fragmentShader='uniform bool uWetVisible;uniform highp sampler3D uWetDensity;uniform vec3 uWetOrigin;varying vec3 vWetCell;\n'+shader.fragmentShader.replace(
        '#include <roughnessmap_fragment>',`#include <roughnessmap_fragment>
        if (uWetVisible) {
          vec3 uvw=(vWetCell-uWetOrigin+0.5).zyx/vec3(textureSize(uWetDensity,0));
          float wet=texture(uWetDensity,uvw).r;
          float film=smoothstep(.015,.62,wet);
          roughnessFactor=mix(roughnessFactor,min(roughnessFactor,.16),film);
          diffuseColor.rgb*=1.-.20*film;
        }`);
    };
    material.customProgramCacheKey=()=>key+'/retained-field-wetting-v1';material.needsUpdate=true;
  }
}

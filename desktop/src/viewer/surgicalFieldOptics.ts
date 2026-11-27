import * as THREE from 'three';
import {joinedFreeField,type FieldSample} from './freeFieldSurface.ts';
import type {SurgicalView} from '../surgical-types';
import type {Point3} from './coordinates';
import {placeInSourceFrame} from './sceneGeometry.ts';
import {sameSurgicalGrid,sameFieldEntries} from './surgicalDisplayInputs.ts';

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
/** Share of the retained field that is blood. With `vesselBleedingV1` the
 * inventory mixes blood and irrigation without per-cell proportions, so the
 * field takes the overall share of deposited blood in the fluid added. */
export function bloodShare(view:SurgicalView):number{
  if(woundOnlyField(view))return 1;
  const deposited=view.vesselBleedingV1?.depositedMm3,added=view.metrics?.fluidAddedMm3;
  if(typeof deposited!=='number'||!(deposited>0)||typeof added!=='number'||!(added>0))return 0;
  return Math.min(1,deposited/added);
}
/** Pooled blood hides what lies under it until suctioned: opaque once blood dominates the field. */
export function fieldLiquidAppearance(share:number){
  const blood=new THREE.Color('#7e1f1a'),clear=new THREE.Color('#a8bdc1');
  const opaque=share>=.5;
  return {color:clear.lerp(blood,Math.min(1,share*1.6)),opacity:opaque?1:.57+.4*share,opaque};
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

function fieldComponent(scene:SurgicalView['scene'],entries:FieldSample[],capacity:number,liquid:boolean,share=0) {
  if(!entries.length)return null;
  const joined=liquid?joinedFreeField(entries,scene.shape,capacity):null;
  const [,ny,nz]=scene.shape;
  // Isolated deposits remain volume-scaled. Adjacent free cells share one
  // neutral optical surface. A confined or overfilled cell adds no extra reach.
  const geometry=liquid?new THREE.SphereGeometry(.5,12,8):new THREE.IcosahedronGeometry(.5);
  const look=fieldLiquidAppearance(share);
  const material=new THREE.MeshStandardMaterial({color:liquid?look.color:'#bbaa8b',roughness:liquid?.14:.86,
    metalness:0,transparent:liquid&&!look.opaque,opacity:liquid?look.opacity:1,depthWrite:!liquid||look.opaque});
  const objects:Array<THREE.Mesh<THREE.BufferGeometry,THREE.MeshStandardMaterial>>=[];
  if(joined){
    const pool=new THREE.Mesh(joined.geometry,material.clone());pool.name='field:free-surface';
    placeInSourceFrame(pool,scene.affine);
    pool.receiveShadow=true;pool.raycast=()=>{};objects.push(pool);
  }
  const mesh=new THREE.InstancedMesh(geometry,material,entries.length);
  mesh.name=liquid?'field:fluid':'field:debris';mesh.raycast=()=>{};mesh.receiveShadow=true;
  placeInSourceFrame(mesh,scene.affine);
  const matrix=new THREE.Matrix4();
  entries.forEach(({cell,amount},index)=>{
    const inPool=joined?.joined.has((cell[0]*ny+cell[1])*nz+cell[2]);
    const diameter=inPool?0:Math.cbrt(Math.min(1,amount/capacity));
    matrix.makeScale(diameter,diameter,diameter).setPosition(...cell);
    mesh.setMatrixAt(index,matrix);
  });
  mesh.visible=!liquid||joined?.joined.size!==entries.length;
  mesh.instanceMatrix.needsUpdate=true;
  objects.push(mesh);
  return {objects,cells:entries.length,mm3:entries.reduce((sum,item)=>sum+item.amount,0),joinedFluidCells:joined?.joined.size??0};
}

function wetTexture(fluid:FieldSample[],capacity:number) {
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
  return {texture,bytes:texture?bytes:0,origin:texture?new THREE.Vector3(...lower).subScalar(1):new THREE.Vector3()};
}

type FieldComponent=ReturnType<typeof fieldComponent>;
type Field={group:THREE.Group;texture:THREE.Data3DTexture|null;worldToCell:THREE.Matrix4;origin:THREE.Vector3}|null;
function releaseComponent(component:FieldComponent){
  for(const mesh of component?.objects??[]){
    mesh.removeFromParent();mesh.geometry.dispose();mesh.material.dispose();
    if(mesh instanceof THREE.InstancedMesh)mesh.dispose();
  }
}

/** One current component per ledger on the installed source grid. Ownership
 * stays here across effect-marker replacement and temporarily hidden previews. */
export class SurgicalFieldResources {
  private scene?:SurgicalView['scene'];
  private sourceCaseHash?:string;
  private fluidLedger?:SurgicalView['fluid'];
  private debrisLedger?:SurgicalView['debris'];
  private bloodShare=0;
  private fluid:FieldComponent=null;
  private debris:FieldComponent=null;
  private wet?:ReturnType<typeof wetTexture>;
  private current:Field=null;
  readonly work={fluidBuilds:0,debrisBuilds:0,wetTextureBuilds:0,unchanged:0};
  get retained(){return this.current;}

  update(view:SurgicalView,sourceCaseHash?:string):Field {
    const sameGrid=this.scene&&this.sourceCaseHash===sourceCaseHash&&sameSurgicalGrid(this.scene,view.scene);
    // The fluid colour follows the declared origin, which can change with the same ledger.
    const share=bloodShare(view);
    const updateFluid=!sameGrid||Math.abs(this.bloodShare-share)>.02||!sameFieldEntries(this.fluidLedger,view.fluid);
    const updateDebris=!sameGrid||!sameFieldEntries(this.debrisLedger,view.debris);
    if(updateFluid||updateDebris){
      const grid=new THREE.Matrix4().fromArray(view.scene.affine.flat()).transpose();
      const capacity=Math.abs(grid.determinant());
      if(!Number.isFinite(capacity)||capacity<1e-12){this.clear();return null;}
      const fluid=updateFluid?samples(view.fluid,view.scene.shape):null;
      let newFluid:FieldComponent=null,newDebris:FieldComponent=null,wet:ReturnType<typeof wetTexture>|undefined;
      try{
        if(fluid?.length){newFluid=fieldComponent(view.scene,fluid,capacity,true,share);wet=wetTexture(fluid,capacity);}
        if(updateDebris)newDebris=fieldComponent(view.scene,samples(view.debris,view.scene.shape),capacity,false);
      }catch(error){releaseComponent(newFluid);releaseComponent(newDebris);wet?.texture?.dispose();throw error;}
      if(fluid){
        releaseComponent(this.fluid);this.wet?.texture?.dispose();
        this.fluid=newFluid;this.wet=wet;
        if(newFluid)this.work.fluidBuilds++;
        if(wet?.texture)this.work.wetTextureBuilds++;
      }
      if(updateDebris){
        releaseComponent(this.debris);this.debris=newDebris;
        if(newDebris)this.work.debrisBuilds++;
      }
      if(this.fluid||this.debris){
        const group=this.current?.group??new THREE.Group();
        group.name='effect:retained-field';group.clear().add(...(this.fluid?.objects??[]),...(this.debris?.objects??[]));
        this.current={group,texture:this.wet?.texture??null,worldToCell:grid.invert(),origin:this.wet?.origin??new THREE.Vector3()};
      }else {this.current?.group.removeFromParent();this.current=null;}
    }else this.work.unchanged++;
    this.scene=view.scene;this.sourceCaseHash=sourceCaseHash;this.fluidLedger=view.fluid;this.debrisLedger=view.debris;this.bloodShare=share;
    if(this.current)this.current.group.userData.fieldSummary={stateId:view.stateId,
      ...(view.operativeExposureV1?{fluidOrigin:share>=1?'vascular sources only':share>0?`mixed, ${Math.round(100*share)}% blood overall`:'mixed or unknown'}:{}),fluidCells:this.fluid?.cells??0,debrisCells:this.debris?.cells??0,
      fluidMm3:this.fluid?.mm3??0,debrisMm3:this.debris?.mm3??0,wetTextureBytes:this.wet?.bytes??0,joinedFluidCells:this.fluid?.joinedFluidCells??0};
    return this.current;
  }

  clear(){
    releaseComponent(this.fluid);releaseComponent(this.debris);this.wet?.texture?.dispose();this.current?.group.removeFromParent();
    this.fluid=null;this.debris=null;this.wet=undefined;this.current=null;
    this.scene=undefined;this.sourceCaseHash=undefined;this.fluidLedger=undefined;this.debrisLedger=undefined;this.bloodShare=0;
  }
}

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

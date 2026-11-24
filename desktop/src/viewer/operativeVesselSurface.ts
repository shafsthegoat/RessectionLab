import type {SurgicalRegion,SurgicalView} from '../surgical-types';
import type {ViewerVolume} from './contracts';
import type {Affine,Point3,Shape3} from './coordinates';
import {inverseAffine} from './coordinates.ts';
import {indexedSurfaceNormals} from './surfaceNormals.ts';

export interface OperativeVesselSurface {
  mode:'native-probability'|'edited-native-wall'|'source-cell-boundary';
  region:number;
  positions:Float32Array;
  normals:Float32Array;
  indices:Uint32Array;
  sourceCells:Uint32Array;
  cuboidCount?:number;
  geometryIdentity?:string;
}
type RestVessel=OperativeVesselSurface&{sites:number[]};
type EditedWall=NonNullable<NonNullable<SurgicalView['vesselMechanics']>['editedWallsByLabel']>[string];
export function vesselGeometryToken(view:SurgicalView){
  const geometry=view.vesselMechanics?.geometryModeByLabel;
  if(geometry)return [geometry,Object.entries(view.vesselMechanics?.editedWallsByLabel??{}).map(([label,wall])=>[label,wall.identity])];
}
const sitePoint=(id:number,[,ny,nz]:Shape3):Point3=>[Math.floor(id/(ny*nz)),Math.floor(id/nz)%ny,id%nz];
const edgeKey=(a:number,b:number)=>a<b?`${a},${b}`:`${b},${a}`;

/** Admit only score-contour edge crossings on the exact modeled source grid. */
function bindCrossings(positions:Float32Array,mask:Uint8Array,shape:Shape3):Uint32Array|undefined {
  const [nx,ny,nz]=shape,ids=new Uint32Array(positions.length/3);
  for(let i=0;i<ids.length;i++){
    const p=[positions[3*i],positions[3*i+1],positions[3*i+2]].map(x=>Math.abs(x-Math.round(x))<=1e-5?Math.round(x):x);
    const lo=p.map(Math.floor),hi=p.map(Math.ceil);
    if(p.some(x=>!Number.isFinite(x))||lo.filter((x,j)=>x!==hi[j]).length>1)return;
    let best=Infinity,source=-1;
    for(let x=lo[0];x<=hi[0];x++)for(let y=lo[1];y<=hi[1];y++)for(let z=lo[2];z<=hi[2];z++){
      if(x<0||x>=nx||y<0||y>=ny||z<0||z>=nz)continue;
      const id=(x*ny+y)*nz+z,d=(p[0]-x)**2+(p[1]-y)**2+(p[2]-z)**2;
      if(mask[id]&&d<best){source=id;best=d;}
    }
    if(source<0)return;
    ids[i]=source;
  }
  return ids;
}

export function bindNativeVesselSurfaces(source:ViewerVolume,owner:Uint16Array,regions:SurgicalRegion[]){
  const result=new Map<string,RestVessel>();
  for(const {name,id,material} of regions){
    if(material!=='vessel')continue;
    const layer=source.compartments.find(l=>l.name===name),surface=layer?.surface,field=layer?.surfaceDiagnostics?.surfaceField;
    if(!layer||!surface?.indices.length||layer.labelSet||layer.mask.length!==owner.length||
      (field!=='native-probability'&&field!=='filtered-native-probability'))continue;
    const {source:positions,sourceNormals:normals,indices}=surface;
    const sourceCells=layer.vesselSourceCells??bindCrossings(positions,layer.mask,source.shape);
    if(!sourceCells||sourceCells.length*3!==positions.length||sourceCells.some(site=>owner[site]!==id))continue;
    // A surface can enter surgery only if its original mask is the complete
    // original vessel ownership for this region on the simulation grid.
    const sites:number[]=[];
    let bound=true;
    for(let i=0;i<owner.length;i++){
      if(Boolean(layer.mask[i])!==(owner[i]===id)){bound=false;break;}
      if(owner[i]===id)sites.push(i);
    }
    if(bound)result.set(name,{mode:'native-probability',region:id,positions,normals,indices,sourceCells,sites});
  }
  return result;
}

const faces:[number,number,Point3[]][]=[
  [0,-1,[[-.5,-.5,-.5],[-.5,-.5,.5],[-.5,.5,.5],[-.5,.5,-.5]]],
  [0,1,[[.5,-.5,-.5],[.5,.5,-.5],[.5,.5,.5],[.5,-.5,.5]]],
  [1,-1,[[-.5,-.5,-.5],[.5,-.5,-.5],[.5,-.5,.5],[-.5,-.5,.5]]],
  [1,1,[[-.5,.5,-.5],[-.5,.5,.5],[.5,.5,.5],[.5,.5,-.5]]],
  [2,-1,[[-.5,-.5,-.5],[-.5,.5,-.5],[.5,.5,-.5],[.5,-.5,-.5]]],
  [2,1,[[-.5,-.5,.5],[.5,-.5,.5],[.5,.5,.5],[-.5,.5,.5]]],
];

/** Render the same surviving source boxes as the edited-wall collider.
 * Each box keeps six faces so relative cell motion can expose a cut cap. */
function sourceCellBoundary(region:number,sites:number[],shape:Shape3):OperativeVesselSurface {
  const vertices=24*sites.length;
  const positions=new Float32Array(vertices*3),normals=new Float32Array(vertices*3),indices=new Uint32Array(vertices*1.5),sourceCells=new Uint32Array(vertices);
  let vertex=0,index=0;
  for(const site of sites){
    const p=sitePoint(site,shape);
    for(const [axis,side,corners] of faces){
      indices.set([vertex,vertex+1,vertex+2,vertex,vertex+2,vertex+3],index);index+=6;
      for(const corner of corners){
        positions.set([p[0]+corner[0],p[1]+corner[1],p[2]+corner[2]],vertex*3);
        normals[vertex*3+axis]=side;sourceCells[vertex]=site;vertex++;
      }
    }
  }
  return {mode:'source-cell-boundary',region,positions,normals,indices,sourceCells,cuboidCount:sites.length};
}

function editedNativeWall(wall:EditedWall,region:number,owner:Uint16Array,removed:Set<number>,shape:Shape3,affine:Affine):OperativeVesselSurface {
  const {positionsIJK,sourceCells,triangles,cuboidSourceCells=[]}=wall;
  if(positionsIJK.length!==sourceCells.length||positionsIJK.length+24*cuboidSourceCells.length>10_000_000||triangles.length+12*cuboidSourceCells.length>10_000_000)
    throw Error('Edited vessel wall has inconsistent geometry sizes');
  const live=(cell:number)=>Number.isInteger(cell)&&owner[cell]===region&&!removed.has(cell);
  const cubeSet=new Set(cuboidSourceCells);
  if(cubeSet.size!==cuboidSourceCells.length||cuboidSourceCells.some(cell=>!live(cell)))throw Error('Edited vessel boxes do not belong to the accepted wall');
  const cubes=sourceCellBoundary(region,cuboidSourceCells,shape);
  const positions=new Float32Array(positionsIJK.length*3+cubes.positions.length),indices=new Uint32Array(triangles.length*3+cubes.indices.length);
  const cells=new Uint32Array(sourceCells.length+cubes.sourceCells.length);
  for(let i=0;i<sourceCells.length;i++){
    const cell=sourceCells[i],point=positionsIJK[i];
    if(point.length!==3||!point.every(Number.isFinite)||!live(cell)||cubeSet.has(cell))
      throw Error('Edited vessel wall has an invalid surface owner');
    positions.set(point,i*3);cells[i]=cell;
  }
  for(let i=0;i<triangles.length;i++){
    const face=triangles[i];
    if(face.length!==3||face.some(index=>!Number.isInteger(index)||index<0||index>=sourceCells.length))
      throw Error('Edited vessel wall has an invalid triangle');
    indices.set(face,i*3);
  }
  positions.set(cubes.positions,positionsIJK.length*3);cells.set(cubes.sourceCells,sourceCells.length);
  for(let i=0;i<cubes.indices.length;i++)indices[triangles.length*3+i]=sourceCells.length+cubes.indices[i];
  return {mode:'edited-native-wall',region,positions,indices,sourceCells:cells,
    normals:indexedSurfaceNormals(positions,indices,affine),cuboidCount:cuboidSourceCells.length,geometryIdentity:wall.identity};
}

export function acceptedVesselSurfaces(rests:Map<string,RestVessel>,owner:Uint16Array,overrides:Map<number,number>,view:SurgicalView,
  prior:Map<string,{topology:string;surface:OperativeVesselSurface}>){
  const result=new Map<string,OperativeVesselSurface>();
  const moved=new Map(view.moved),edges=[...view.cutEdges??[],...view.transectedEdges??[]];
  const modes=view.vesselMechanics?.geometryModeByLabel;
  for(const [name,rest] of rests){
    const mode=modes?.[rest.region];
    if(mode==='native-probability-wall'){result.set(name,rest);continue;}
    const excluded=new Set(view.removed.filter(id=>owner[id]===rest.region));
    if(mode==='edited-native-wall'){
      const edited=view.vesselMechanics?.editedWallsByLabel?.[rest.region],cached=prior.get(name);
      if(!edited?.identity)throw Error('Accepted edited vessel wall geometry is missing');
      const topology=`edited:${edited.identity}`;
      const surface=cached?.topology===topology?cached.surface:editedNativeWall(edited,rest.region,owner,excluded,view.scene.shape,view.scene.affine);
      prior.set(name,{topology,surface});result.set(name,surface);
      continue;
    }
    if(!mode){
      for(const [source,target] of moved)if(owner[source]===rest.region&&(overrides.get(target)??owner[target])!==rest.region)excluded.add(source);
      for(const [site,region] of overrides)if(!moved.has(site)&&owner[site]===rest.region&&region!==rest.region)excluded.add(site);
    }
    const broken=new Set(edges.filter(([a,b])=>owner[a]===rest.region&&owner[b]===rest.region&&!excluded.has(a)&&!excluded.has(b)).map(([a,b])=>edgeKey(a,b)));
    if(mode==='accepted-source-cuboids'||excluded.size||broken.size){
      const topology=JSON.stringify([...excluded].sort((a,b)=>a-b)),cached=prior.get(name);
      const surface=cached?.topology===topology?cached.surface:sourceCellBoundary(rest.region,rest.sites.filter(site=>!excluded.has(site)),view.scene.shape);
      prior.set(name,{topology,surface});result.set(name,surface);
    }else{
      prior.delete(name);result.set(name,rest);
    }
  }
  if(modes){
    const labels=new Set([...result.values()].map(surface=>String(surface.region)));
    if(Object.keys(modes).some(label=>!labels.has(label)))throw Error('Accepted vessel collision geometry has no matching native display source');
  }
  return result;
}

/** Keep source geometry immutable and advect vertices by accepted RAS motion.
 * The new triangle positions are the piecewise-linear interpolation of their
 * attached source sites, including the zero-shift sites at a supported seam. */
export function deformOperativeVessel(surface:OperativeVesselSurface,view:SurgicalView,affine:Affine){
  const motion=view.vesselMechanics?.surfaceDisplacements;
  const legacy=view.physics?.brain?.assignment;
  const attached=new Set(legacy?.vesselMotionModel==='retained-source-segments-v1'?legacy.vesselAttachments?.map(([site])=>site):[]);
  const moved=new Map(view.moved),inverse=inverseAffine(affine),shifts=new Map<number,Point3>();
  let positions:Float32Array|undefined,movedVertices=0,maxVertexOffsetMm=0,maxVertex=-1;
  for(let i=0;i<surface.sourceCells.length;i++){
    const id=surface.sourceCells[i];
    let shift=shifts.get(id);
    if(!shift){
      const delta=motion?motion[id]:attached.has(id)?view.deformation?.[id]:undefined;
      if(delta)shift=inverse.slice(0,3).map(row=>row[0]*delta[0]+row[1]*delta[1]+row[2]*delta[2]) as Point3;
      else if(!motion&&moved.has(id)){
        const origin=sitePoint(id,view.scene.shape),target=sitePoint(moved.get(id)!,view.scene.shape);
        shift=[target[0]-origin[0],target[1]-origin[1],target[2]-origin[2]];
      }else shift=[0,0,0];
      shifts.set(id,shift);
    }
    if(!shift.some(Boolean))continue;
    positions??=surface.positions.slice();
    const start=i*3;
    for(let j=0;j<3;j++)positions[start+j]=surface.positions[start+j]+shift[j];
    const x=positions[start]-surface.positions[start],y=positions[start+1]-surface.positions[start+1],z=positions[start+2]-surface.positions[start+2];
    if(!x&&!y&&!z)continue;
    movedVertices++;
    const [a,b,c]=affine;
    const mm=Math.hypot(a[0]*x+a[1]*y+a[2]*z,b[0]*x+b[1]*y+b[2]*z,c[0]*x+c[1]*y+c[2]*z);
    if(mm>maxVertexOffsetMm){maxVertexOffsetMm=mm;maxVertex=i;}
  }
  const maxMotionWitness=positions&&maxVertex>=0?{sourceCell:surface.sourceCells[maxVertex],vertex:maxVertex,
    restNativeIJK:Array.from(surface.positions.subarray(3*maxVertex,3*maxVertex+3)),
    displayedNativeIJK:Array.from(positions.subarray(3*maxVertex,3*maxVertex+3))}:undefined;
  return {positions:positions??surface.positions,normals:positions&&surface.mode!=='source-cell-boundary'?indexedSurfaceNormals(positions,surface.indices,affine):surface.normals,
    movedVertices,maxVertexOffsetMm,maxMotionWitness};
}

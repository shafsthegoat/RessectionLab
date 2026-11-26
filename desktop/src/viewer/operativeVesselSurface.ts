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
  scoreLighting?:Float32Array;
  cuboidCount?:number;
  geometryIdentity?:string;
  geometryBinding?:string;
  sourceVolumeBoundary?:boolean;
  /** Tube walls: per vertex, 1 vein and 0 artery or unclassified, aligned with positions. */
  vesselClasses?:Uint8Array;
  /** A centerline-tube wall: its displayed normals are smoothed across small undulations. */
  tube?:boolean;
}
type RestVessel=OperativeVesselSurface&{sites:number[]};
type EditedWall=NonNullable<NonNullable<SurgicalView['vesselMechanics']>['editedWallsByLabel']>[string];
export function vesselGeometryToken(view:SurgicalView){
  const geometry=view.vesselMechanics?.geometryModeByLabel;
  if(geometry){
    const edited=Object.entries(view.vesselMechanics?.editedWallsByLabel??{}).map(([label,wall])=>[label,wall.identity]);
    const vertex=view.vesselMechanics?.vertexDisplacementsByLabel;
    return vertex?[geometry,edited,Object.entries(vertex).map(([label,packet])=>[label,packet.geometryBinding,packet.vertexCount])]:[geometry,edited];
  }
}
const neighbourCache=new WeakMap<Uint32Array,{start:Int32Array;list:Int32Array}>();
function vertexNeighbours(indices:Uint32Array,count:number){
  const cached=neighbourCache.get(indices);
  if(cached&&cached.start.length===count+1)return cached;
  const degree=new Int32Array(count+1);
  for(let i=0;i<indices.length;i++)degree[indices[i]+1]+=2;
  for(let i=0;i<count;i++)degree[i+1]+=degree[i];
  const fill=degree.slice(0,count),list=new Int32Array(degree[count]);
  for(let t=0;t<indices.length;t+=3)for(let k=0;k<3;k++){
    const v=indices[t+k];list[fill[v]++]=indices[t+(k+1)%3];list[fill[v]++]=indices[t+(k+2)%3];
  }
  const result={start:degree,list};neighbourCache.set(indices,result);return result;
}
/** Display-only smoothing of a tube wall's normals: neighbours average in only while their normals
 * stay within crease degrees, so cut caps and junction creases keep their edges while the few-degree
 * undulations of the fitted sections even out into one highlight. Positions are untouched. */
export function smoothTubeNormals(normals:Float32Array,indices:Uint32Array,iterations=20,crease=30):Float32Array{
  const count=normals.length/3,{start,list}=vertexNeighbours(indices,count),limit=Math.cos(crease*Math.PI/180);
  let current=Float32Array.from(normals);
  for(let k=0;k<count;k++){const l=Math.hypot(current[3*k],current[3*k+1],current[3*k+2])||1;for(let j=0;j<3;j++)current[3*k+j]/=l;}
  for(let iteration=0;iteration<iterations;iteration++){
    const next=new Float32Array(current.length);
    for(let v=0;v<count;v++){
      let x=current[3*v],y=current[3*v+1],z=current[3*v+2];
      for(let e=start[v];e<start[v+1];e++){
        const u=list[e],c=current[3*v]*current[3*u]+current[3*v+1]*current[3*u+1]+current[3*v+2]*current[3*u+2];
        const w=Math.max(0,Math.min(1,(c-limit)/(1-limit)))/2;
        x+=w*current[3*u];y+=w*current[3*u+1];z+=w*current[3*u+2];
      }
      const l=Math.hypot(x,y,z)||1;next[3*v]=x/l;next[3*v+1]=y/l;next[3*v+2]=z/l;
    }
    current=next;
  }
  return current;
}
const sitePoint=(id:number,[,ny,nz]:Shape3):Point3=>[Math.floor(id/(ny*nz)),Math.floor(id/nz)%ny,id%nz];
const edgeKey=(a:number,b:number)=>a<b?`${a},${b}`:`${b},${a}`;

/** Probability contours close across padded samples outside the source volume. */
function crossesSourceVolumeBoundary(positions:Float32Array,shape:Shape3){
  return positions.some((value,index)=>value< -1e-5||value>shape[index%3]-1+1e-5);
}

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
    const scoreLighting=field==='native-probability'&&layer.vesselScoreLighting?.length===sourceCells.length*4
      ?layer.vesselScoreLighting:undefined;
    const vesselClasses=layer.vesselClasses?.length===sourceCells.length?layer.vesselClasses:undefined,tube=!!layer.vesselTags;
    if(bound)result.set(name,{mode:'native-probability',region:id,positions,normals:tube?smoothTubeNormals(normals,indices):normals,indices,sourceCells,sites,scoreLighting,vesselClasses,tube,
      geometryBinding:layer.vesselGeometryBinding,
      sourceVolumeBoundary:crossesSourceVolumeBoundary(positions,source.shape)});
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

function retainedScoreLighting(rest:RestVessel,positions:Float32Array,owners:Uint32Array,triangles:EditedWall['triangles'],nativeCount:number){
  const score=rest.scoreLighting;
  if(!score?.length)return;
  const key=(points:Float32Array,cells:Uint32Array,i:number)=>{
    const start=3*i;
    return `${cells[i]},${points[start]},${points[start+1]},${points[start+2]}`;
  };
  const original=new Map<string,number>();
  for(let i=0;i<rest.sourceCells.length;i++){
    const vertex=key(rest.positions,rest.sourceCells,i);
    original.set(vertex,original.has(vertex)?-1:i);
  }
  const match=Int32Array.from({length:nativeCount},(_,i)=>original.get(key(positions,owners,i))??-1);
  const faceKey=(a:number,b:number,c:number)=>a<b&&a<c?`${a},${b},${c}`:b<c?`${b},${c},${a}`:`${c},${a},${b}`;
  const unchanged=new Set<string>();
  for(let i=0;i<rest.indices.length;i+=3)unchanged.add(faceKey(rest.indices[i],rest.indices[i+1],rest.indices[i+2]));
  const changed=new Uint8Array(nativeCount);
  for(const [a,b,c] of triangles){
    const x=match[a],y=match[b],z=match[c];
    if(Math.min(x,y,z)<0||!unchanged.has(faceKey(x,y,z))){changed[a]=1;changed[b]=1;changed[c]=1;}
  }
  // New cap faces and their rim use the accepted geometric normals.
  const lights=new Float32Array(owners.length*4);
  let supported=false;
  for(let i=0;i<nativeCount;i++){
    const j=4*match[i];
    if(j>=0&&!changed[i]&&score[j+3]){lights.set(score.subarray(j,j+4),4*i);supported=true;}
  }
  return supported?lights:undefined;
}

function editedNativeWall(wall:EditedWall,rest:RestVessel,owner:Uint16Array,removed:Set<number>,shape:Shape3,affine:Affine):OperativeVesselSurface {
  const {region}=rest;
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
  // A tube wall's edit packet carries the classes of its own vertices; cap boxes take the artery colour.
  const vesselClasses=wall.vesselClasses&&wall.vesselClasses.length===sourceCells.length
    ?Uint8Array.from({length:cells.length},(_,i)=>i<sourceCells.length&&wall.vesselClasses![i]?1:0):undefined;
  const edgeNormals=indexedSurfaceNormals(positions,indices,affine);
  return {mode:'edited-native-wall',region,positions,indices,sourceCells:cells,vesselClasses,tube:rest.tube,
    normals:rest.tube?smoothTubeNormals(edgeNormals,indices):edgeNormals,cuboidCount:cuboidSourceCells.length,geometryIdentity:wall.identity,geometryBinding:wall.identity,
    sourceVolumeBoundary:crossesSourceVolumeBoundary(positions,shape),
    scoreLighting:retainedScoreLighting(rest,positions,cells,triangles,sourceCells.length)};
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
      const surface=cached?.topology===topology?cached.surface:editedNativeWall(edited,rest,owner,excluded,view.scene.shape,view.scene.affine);
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
  const vertexMotion=view.vesselMechanics?.vertexDisplacementsByLabel;
  if(vertexMotion){
    const byLabel=new Map([...result.values()].map(surface=>[String(surface.region),surface]));
    for(const [label,packet] of Object.entries(vertexMotion)){
      const surface=byLabel.get(label);
      if(!surface||!surface.geometryBinding||packet.geometryBinding!==surface.geometryBinding||packet.vertexCount!==surface.sourceCells.length)
        throw Error('Accepted vessel vertex motion differs from its bound source surface');
    }
  }
  return result;
}

/** Keep source geometry immutable and advect vertices by accepted RAS motion.
 * The new triangle positions are the piecewise-linear interpolation of their
 * attached source sites, including the zero-shift sites at a supported seam. */
export function deformOperativeVessel(surface:OperativeVesselSurface,view:SurgicalView,affine:Affine){
  const motion=view.vesselMechanics?.surfaceDisplacements;
  const vertexMotion=view.vesselMechanics?.vertexDisplacementsByLabel?.[surface.region];
  if(vertexMotion){
    const {geometryBinding,vertexCount,indices,offsetsMm}=vertexMotion;
    if(!surface.geometryBinding||geometryBinding!==surface.geometryBinding||vertexCount!==surface.sourceCells.length||
       !Array.isArray(indices)||!Array.isArray(offsetsMm)||indices.length!==offsetsMm.length||
       indices.some((id,i)=>!Number.isSafeInteger(id)||id<0||id>=vertexCount||(i>0&&id<=indices[i-1]))||
       offsetsMm.some(row=>!Array.isArray(row)||row.length!==3||!row.every(Number.isFinite)))
      throw Error('Accepted vessel vertex offsets do not match their indexed geometry');
  }
  const legacy=view.physics?.brain?.assignment;
  const attached=new Set(legacy?.vesselMotionModel==='retained-source-segments-v1'?legacy.vesselAttachments?.map(([site])=>site):[]);
  const moved=new Map(view.moved),inverse=inverseAffine(affine),shifts=new Map<number,Point3>();
  const toNative=(delta:Point3)=>inverse.slice(0,3).map(row=>row[0]*delta[0]+row[1]*delta[1]+row[2]*delta[2]) as Point3;
  let positions:Float32Array|undefined,movedVertices=0,maxVertexOffsetMm=0,maxVertex=-1;
  let next=0;
  for(let i=0;i<surface.sourceCells.length;i++){
    const id=surface.sourceCells[i];
    let shift:Point3|undefined;
    if(vertexMotion){
      if(i!==vertexMotion.indices[next])continue;
      shift=toNative(vertexMotion.offsetsMm[next++]);
    }else if(!(shift=shifts.get(id))){
      const delta=motion?motion[id]:attached.has(id)?view.deformation?.[id]:undefined;
      if(delta)shift=toNative(delta);
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
  const faceted=positions&&surface.mode!=='source-cell-boundary'?indexedSurfaceNormals(positions,surface.indices,affine):undefined;
  return {positions:positions??surface.positions,normals:faceted?(surface.tube?smoothTubeNormals(faceted,surface.indices):faceted):surface.normals,
    movedVertices,maxVertexOffsetMm,maxMotionWitness};
}

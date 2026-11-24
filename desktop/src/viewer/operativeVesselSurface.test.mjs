import test from 'node:test';
import assert from 'node:assert/strict';
import {modeledVolume} from '../modeled-volume.ts';
import {modeledAnatomy} from './modeledAnatomy.ts';
import {deformOperativeVessel} from './operativeVesselSurface.ts';

const shape=[5,5,5],affine=[[0,-.4,0,30],[.3,0,0,20],[0,0,.5,10],[0,0,0,1]];
const site=(i,j,k)=>(i*5+j)*5+k;
const [a,b,support]=[site(2,2,2),site(3,2,2),site(2,3,2)];
const regions=[{id:1,name:'tissue',material:'brain',color:'#777'},{id:2,name:'lumen',material:'vessel',color:'#733'}];
const mask=ids=>{const out=new Uint8Array(125);ids.forEach(i=>{out[i]=1;});return out;};
const positions=new Float32Array([2,1.8,2, 2,2,2.3, 2,2.37,2, 3,2,2.3]);
const normals=new Float32Array([0,-1,0, 0,0,1, 0,1,0, 0,0,1]);
const source={caseId:'operative vessel',caseHash:'vessel-test',mri:new Float32Array(125),affine,frame:'RAS+',shape,compartments:[
  {name:'tissue',kind:'brain',color:'#777',mask:mask([a,support])},
  {name:'lumen',kind:'vessel',color:'#733',mask:mask([a,b]),surfaceDiagnostics:{surfaceField:'native-probability'},
    surface:{source:positions,positions,sourceNormals:normals,normals,indices:new Uint32Array([0,1,2, 1,3,2])}},
]};
const view={scene:{shape,affine,regions},removed:[],moved:[],deformation:{}};
const wall=(state,volume=source)=>modeledAnatomy(modeledVolume(volume,state)).vesselSurfaces.get('lumen');
const close=(actual,expected)=>assert.ok(Math.abs(actual-expected)<1e-6,`${actual} differs from ${expected}`);

test('native score contour survives mapping, accepted raster movement and support removal',()=>{
  const mesh=wall(view),baseline=deformOperativeVessel(mesh,view,affine);
  assert.equal(mesh.mode,'native-probability');
  assert.equal(baseline.positions,positions);
  assert.equal(baseline.normals,normals);
  assert.deepEqual([...mesh.sourceCells],[a,a,a,b]);
  const moved={...view,moved:[[a,support]],vesselMechanics:{geometryModeByLabel:{2:'native-probability-wall'},surfaceDisplacements:{[a]:[-.6,.15,.25]}}};
  const unsupported={...moved,removed:[support]};
  assert.equal(wall(moved),mesh);
  assert.equal(wall(unsupported),mesh);
  const display=deformOperativeVessel(wall(unsupported),unsupported,affine);
  assert.equal(display.movedVertices,3);
  close(display.maxVertexOffsetMm,Math.hypot(.6,.15,.25));
  const shift=[.5,1.5,.5];
  for(let i=0;i<12;i++)close(display.positions[i],positions[i]+(i<9?shift[i%3]:0));
  // Three wall vertices tied to one cell retain their whole cross-section.
  for(let j=0;j<3;j++)close(display.positions[j]-display.positions[j+6],positions[j]-positions[j+6]);
  assert.equal(deformOperativeVessel(mesh,view,affine).positions,positions);
});

test('an authoritative vessel transection renders closed per-cell faces even with unchanged raster',()=>{
  const intact=wall(view);
  const cut={...view,transectedEdges:[[a,b]],vesselMechanics:{geometryModeByLabel:{2:'accepted-source-cuboids'},surfaceDisplacements:{[b]:[0,.15,0]}}};
  const mesh=wall(cut),display=deformOperativeVessel(mesh,cut,affine);
  assert.equal(mesh.mode,'source-cell-boundary');
  assert.equal(mesh.indices.length/3,24);
  assert.equal(mesh.sourceCells.filter(site=>site===a).length,24);
  assert.equal(mesh.sourceCells.filter(site=>site===b).length,24);
  const splitCaps=Array.from(mesh.sourceCells).flatMap((site,i)=>mesh.positions[3*i]===2.5?[{site,x:display.positions[3*i]}]:[]);
  assert.ok(splitCaps.some(cap=>cap.site===a&&cap.x===2.5));
  assert.ok(splitCaps.some(cap=>cap.site===b&&cap.x===3));
  assert.equal(wall({...cut,removed:[b]}).indices.length/3,12);
  assert.equal(wall(view),intact);
});

test('an edited source score wall preserves its native triangles and limits cuboids to their declared owners',()=>{
  const scoreLighting=new Float32Array(16).fill(.5);
  const lit={...source,compartments:[source.compartments[0],{...source.compartments[1],vesselScoreLighting:scoreLighting}]};
  assert.equal(wall(view,lit).scoreLighting,scoreLighting);
  const edited={identity:'qualified-cut',positionsIJK:[[2,1.8,2],[2,2,2.3],[2,2.37,2]],triangles:[[0,1,2]],
    sourceCells:[a,a,a],cuboidSourceCells:[b]};
  const cut={...view,transectedEdges:[[a,b]],vesselMechanics:{geometryModeByLabel:{2:'edited-native-wall'},
    editedWallsByLabel:{2:edited},surfaceDisplacements:{[b]:[0,.15,0]}}};
  const mesh=wall(cut,lit),display=deformOperativeVessel(mesh,cut,affine);
  assert.equal(mesh.mode,'edited-native-wall');
  assert.equal(mesh.scoreLighting,undefined);
  assert.equal(mesh.cuboidCount,1);
  assert.equal(mesh.indices.length/3,13);
  assert.deepEqual([...mesh.sourceCells.slice(0,3)],[a,a,a]);
  assert.ok(mesh.sourceCells.slice(3).every(cell=>cell===b));
  assert.deepEqual(mesh.positions.slice(0,9),positions.slice(0,9));
  assert.deepEqual(display.positions.slice(0,9),positions.slice(0,9));
  assert.equal(wall({...cut,vesselMechanics:{...cut.vesselMechanics,surfaceDisplacements:{}}},lit),mesh);
  const bad={...edited,identity:'wrong-owner',cuboidSourceCells:[support]};
  assert.throws(()=>wall({...cut,vesselMechanics:{...cut.vesselMechanics,editedWallsByLabel:{2:bad}}}),/accepted wall/);
});

test('score contours with mismatched source ownership or incompatible grid use the engine fallback',()=>{
  const wrongBinding={...source,compartments:[source.compartments[0],{...source.compartments[1],vesselSourceCells:Uint32Array.of(a,a,a,support)}]};
  assert.equal(wall(view,wrongBinding),undefined);
  assert.throws(()=>wall({...view,vesselMechanics:{geometryModeByLabel:{2:'native-probability-wall'}}},wrongBinding),/no matching native display source/);
  const wrongGrid=positions.slice();wrongGrid[2]=2.2;
  const unsupported={...source,compartments:[source.compartments[0],{...source.compartments[1],surface:{...source.compartments[1].surface,source:wrongGrid}}]};
  assert.equal(wall(view,unsupported),undefined);
});

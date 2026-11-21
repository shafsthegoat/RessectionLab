import test from 'node:test';
import assert from 'node:assert/strict';
import {execFileSync} from 'node:child_process';
import {fileURLToPath} from 'node:url';
import * as THREE from 'three';
import {surgicalToolMesh,updateSurgicalToolAnimation} from './surgicalGeometry.ts';

const cwd=fileURLToPath(new URL('../../../',import.meta.url));
const probe=`
import json
from resectionlab.surgical_tools import catalog,operative_tool_parameters
from resectionlab.surgical_motion import envelopes
rows=[]
for tool in catalog(operative=True):
 for jaw in (0.,5.):
  params=operative_tool_parameters(tool["id"],{"jawOpeningMm":jaw})
  cover=envelopes(tool,params,[0,0,0],[0,0,0,1])
  rows.append([tool,[dict(a=a.tolist(),b=b.tolist(),radiusMm=r,part=part) for a,b,r,part in cover],jaw])
print(json.dumps(rows))
`;
test('the 21 physical profile meshes and proposed tools fit every native local contact cover',()=>{
  const rows=JSON.parse(execFileSync(cwd+'.venv/bin/python',['-c',probe],{cwd,encoding:'utf8'}));
  assert.equal(rows.length,42);
  for(const [tool,cover,jaw] of rows){
    for(const ghost of [false,true]){
      const slot={position:[0,0,0],quaternion:[0,0,0,1],parameters:{jawOpeningMm:jaw,rpm:15000},
        contactGeometry:cover,instrument:tool.id,active:true};
      const group=surgicalToolMesh(tool,slot,ghost);
      updateSurgicalToolAnimation(group,.017,.25);
      group.updateMatrixWorld(true);
      assert.equal(group.children.length,cover.length);
      for(const [i,frame] of group.children.entries()){
        const {a,b,radiusMm}=cover[i],segment=new THREE.Line3(new THREE.Vector3(...a),new THREE.Vector3(...b));
        assert.equal(frame.name,'instrument:'+tool.physicalProfile.parts[i]);
        let vertices=0;
        frame.traverse(mesh=>{
          if(!mesh.isMesh)return;
          assert.equal(mesh.material.transparent,ghost);
          const positions=mesh.geometry.getAttribute('position'),vertex=new THREE.Vector3(),nearest=new THREE.Vector3();
          for(let j=0;j<positions.count;j++){
            vertex.fromBufferAttribute(positions,j).applyMatrix4(mesh.matrixWorld);
            if(segment.distanceSq()===0)nearest.copy(segment.start);
            else segment.closestPointToPoint(vertex,true,nearest);
            assert(vertex.distanceTo(nearest)<=radiusMm+2e-5,tool.id+' '+frame.name+' must fit the checked profile');
          }
          vertices+=positions.count;
        });
        assert(vertices>0);
      }
    }
  }
});

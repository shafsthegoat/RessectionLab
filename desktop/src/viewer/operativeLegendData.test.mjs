import assert from 'node:assert/strict';
import test from 'node:test';
import {operativeLegend,originalLabelLocation} from './operativeLegendData.ts';
import {applyOperativeSurface} from './operativeMaterial.ts';
import {MeshStandardMaterial} from 'three';

test('registered label identity and color distinguish the supplied target from the vascular estimate',()=>{
  const registeredAffine=[[-1,0,0,10],[0,2,0,8],[0,0,3,9],[0,0,0,1]];
  const region=(name,material,color,target=false)=>({name,material,color,target});
  const input=(layerId,provenance)=>({layerId,provenance,reviewStatus:'unreviewed',registeredAffine,sourceArray:{shape:[4,5,6]}});
  const scene={
    regions:[region('mapped target','tumor','#9863db',true),region('candidate','vessel','#e54491')],
    source:{adapter:{sourceLayers:{'mapped target':input('enhancing_target',{kind:'observed_source_annotation',method:'Released enhancement mask'}),
      candidate:input('candidate-blood-signal',{model:'experimental-T1CE'})},sourceManifest:{
      series:[{id:'post',modality:'T1CE'}],layers:[{id:'enhancing_target',label:'enhancing target',source_series:'post'},
        {id:'candidate-blood-signal',source_series:'post'}]}}}
  };
  const original=structuredClone(scene);
  const optics={operativeView:'surgeon',microscope:{tissueStyle:'operative'}};
  const [target,vessels]=operativeLegend(scene,optics);
  assert.equal(target.title,'Supplied enhancing target');
  assert.equal(target.series,'T1CE');
  assert.equal(target.sourceColor,target.fieldColor);
  assert.equal(vessels.title,'vessel estimate');
  assert.equal(vessels.model,'experimental-T1CE');
  assert.equal(vessels.fieldColor,'#ac6579');
  assert.equal(vessels.sourceColor,'#e54491');
  for(const [i,row] of [target,vessels].entries()){
    const material=new MeshStandardMaterial({color:row.sourceColor});
    applyOperativeSurface(material,scene.regions[i].material,optics);
    assert.equal('#'+material.color.getHexString(),row.fieldColor);
    material.dispose();
  }
  assert.deepEqual(originalLabelLocation(target,[8,10,18]),{point:[2,1,3],within:true});
  assert.equal(originalLabelLocation(vessels,[100,22,22]).within,false);
  assert.deepEqual(scene,original);
  const sourceStyle=operativeLegend(scene,{operativeView:'surgeon',microscope:{tissueStyle:'labels'}});
  assert.equal(sourceStyle[1].fieldColor,sourceStyle[1].sourceColor);
});
test('unmapped tissue has no attributed patient method, source voxel or provenance',()=>{
  const [row]=operativeLegend({regions:[{name:'phantom insert',material:'tumor',color:'#ffffff',target:true}]},{});
  assert.equal(row.method,undefined);assert.equal(row.layerId,undefined);
  assert.equal(originalLabelLocation(row,[0,0,0]),null);
});

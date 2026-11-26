import assert from 'node:assert/strict';
import {test} from 'node:test';
import {formatStructureVolume,structureDisplayName} from '../src/structure-display-name.ts';

const name=(label,kind,extra={})=>structureDisplayName({label,kind,...extra});

test('source labels become short sentence-case names with provenance underneath',()=>{
  const cases=[
    [['Supplied TopBrain v1 arteries','artery',{provenance:{kind:'supplied_annotation'},volumeMm3:42137,voxelCount:42137}],{name:'Arteries',detail:'42.1 cm³ · Supplied'}],
    [['Supplied TopBrain v1 veins and sinuses','vein',{provenance:{kind:'supplied_annotation'}}],{name:'Veins and sinuses',detail:'Supplied'}],
    [['Estimated brain envelope','brain',{provenance:{kind:'image_derived_estimate'},volumeMm3:1210400}],{name:'Brain',detail:'1210 cm³ · Estimate'}],
    [['Estimated mineralized skull','skull',{provenance:{kind:'image_derived_estimate'}}],{name:'Skull',detail:'Estimate'}],
    [['Simulated tumor contour','tumor',{provenance:{kind:'simulated_observation'},volumeMm3:12345}],{name:'Tumor',detail:'12.3 cm³ · Simulated'}],
    [['Cerebellar gray matter · SIAM MRI estimate','cerebellar-cortex',{provenance:{model:'siam-v2',kind:'image_derived_estimate'}}],{name:'Cerebellar gray matter',detail:'Estimate · SIAM MRI'}],
    [['Left Cerebral Cortex · SuperSynth estimate','brain-region',{provenance:{model:'supersynth'}}],{name:'Left cerebral cortex',detail:'Estimate · SuperSynth'}],
    [['Non-enhancing tumor core / necrosis · source label 1','tumor-core',{provenance:{kind:'observed_source_annotation'},volumeMm3:3300}],{name:'Necrotic core',detail:'3.3 cm³ · Supplied'}],
    [['Peritumoral FLAIR abnormality','edema',{provenance:{kind:'observed_source_annotation'}}],{name:'FLAIR abnormality',detail:'Supplied'}],
    [['TopBrain source arterial labels · partial patient annotation','artery',{provenance:{kind:'supplied_annotation'}}],{name:'Arteries',detail:'Supplied · Partial'}],
    [['Brain extraction (estimated)','brain'],{name:'Brain',detail:'Estimate'}],
  ];
  for(const [[label,kind,extra],expected] of cases){
    const display=name(label,kind,extra);
    assert.deepEqual({name:display.name,detail:display.detail},expected,label);
  }
});

test('the study target role is reported beside the name, not in the detail',()=>{
  const target=name('Simulated tumor contour','tumor',{provenance:{kind:'simulated_observation'},sourceRole:'target'});
  assert.deepEqual([target.name,target.role,target.detail],['Tumor','Study target','Simulated']);
  assert.equal(name('Supplied lesion','tumor',{sourceRole:'context'}).role,undefined);
});

test('legacy layer identifiers read as anatomy names',()=>{
  assert.equal(name('enhancing target','tumor').name,'Enhancing tumor');
  assert.equal(name('FLAIR_abnormality','edema').name,'FLAIR abnormality');
  assert.equal(name('nonenhancing_or_necrotic_core','tumor-core').name,'Necrotic core');
  assert.equal(name('left thalamus','brain-region').name,'Left thalamus');
  assert.equal(name('Motor · functional network','brain-region').name,'Motor · functional network');
});

test('the most cautious provenance wins and is never dropped from the detail',()=>{
  // A supplied record cannot hide an estimate named in the label.
  assert.equal(name('Estimated brain envelope','brain',{provenance:{kind:'supplied_annotation'}}).provenance,'estimate');
  assert.equal(name('Tumor','tumor',{provenance:'simulated'}).detail,'Simulated');
  assert.equal(name('Generated motor','brain-region',{provenance:{kind:'generated_annotation'}}).detail,'Generated');
  assert.equal(name('Vessel estimate','vessels').detail,'Estimate');
  assert.equal(name('Vessels','vessels',{provenance:{model:'vesselfm',kind:'supplied_annotation'}}).provenance,'estimate');
  assert.equal(name('Supplied CST','tract').detail,'Supplied');
  assert.equal(name('Arcuate fasciculus left','tract').detail,'Supplied');
  const unknown=name('Named brain region','brain-region');
  assert.deepEqual([unknown.name,unknown.detail,unknown.provenance],['Named brain region','','unknown']);
});

test('volume appears only from a finite source-cell volume',()=>{
  assert.equal(formatStructureVolume(42137),'42.1 cm³');
  assert.equal(formatStructureVolume(512),'0.51 cm³');
  assert.equal(formatStructureVolume(4),'<0.01 cm³');
  assert.equal(formatStructureVolume(1210400),'1210 cm³');
  for(const missing of [undefined,null,Number.NaN,Number.POSITIVE_INFINITY,-5,0])assert.equal(formatStructureVolume(missing),undefined);
  // Voxels without a physical volume give no number.
  const unitless=name('Supplied lesion','tumor',{voxelCount:900,provenance:{kind:'supplied_annotation'}});
  assert.equal(unitless.quantity,undefined);
  assert.equal(unitless.detail,'Supplied');
  const empty=name('Supplied lesion','tumor',{voxelCount:0,volumeMm3:0,provenance:{kind:'supplied_annotation'}});
  assert.equal(empty.detail,'Empty source mask · Supplied');
  assert.equal(empty.qualifiers,'Supplied');
  // A positive voxel count with zero volume is inconsistent: show no quantity.
  assert.equal(name('Lesion','tumor',{voxelCount:12,volumeMm3:0}).quantity,undefined);
});

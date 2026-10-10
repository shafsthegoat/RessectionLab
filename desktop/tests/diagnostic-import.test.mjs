import assert from 'node:assert/strict';import{test}from'node:test';import{createHash}from'node:crypto';import{createRequire}from'node:module';
import{hydrateDiagnostic}from'../src/scan-diagnostic-data.ts';import{validateDiagnosticResult}from'../src/scan-diagnostic-contract.ts';
const require=createRequire(import.meta.url),{importDiagnosticLayer}=require('../electron/diagnostic-import.cjs');
const hex=c=>c.repeat(64),hash=c=>'sha256:'+hex(c),affine=[[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]],shape=[2,2,2];
function fixture(){
 const raw={state:new Int8Array([-1,0,1,-1,-1,-1,-1,-1]),coverage:new Uint8Array([0,1,1,0,0,0,0,0])};
 const array=(name,dtype)=>({assetId:name,dtype,shape,byteOrder:'little',order:'C',byteLength:8,sha256:createHash('sha256').update(raw[name]).digest('hex')});
 const descriptor={schema:'case4_unreviewed_diagnostic_display_layer_v1',scope:'display_only_atlas_native_grid',output_origin:'model_generated_from_observed_case4_preoperative_scans',model_output_verified:true,shape_xyz:shape,affine_ras_mm:affine,source_sha256:{t1c:hex('a'),flair:hex('b'),support_map:hex('c'),plans:hex('d'),dataset:hex('e')},geometry_sha256:hex('f'),input_receipt_sha256:hex('1'),forward_receipt_sha256:hex('2'),outputs:{state:{path:'state.nii.gz',sha256:hex('3'),dtype:'int8'},coverage:{path:'coverage.nii.gz',sha256:hex('4'),dtype:'uint8'}},state_semantics:{'-1':'unknown','0':'candidate_negative_where_output_covered','1':'candidate_positive_where_output_covered'},predicted_voxels:2,unknown_voxels:6,excluded_positive_voxels:0,anatomical_qc:'unreviewed_inferior_mask_omission',training_overlap_status:'unknown',same_grid_only:true,planning_eligible:false,evaluation_eligible:false,clinical_evidence:false};
 const request={caseHash:hash('5'),seriesId:hash('6')},grid=()=>({shape,affineRAS:affine,sformCode:2});
 const result={...request,descriptor,sourceSha256:hex('a'),stateGrid:grid(),coverageGrid:grid(),loadedStateSha256:hex('3'),loadedCoverageSha256:hex('4'),state:array('state','int8'),coverage:array('coverage','uint8')};
 const source={descriptor:{referenceCaseHash:request.caseHash,seriesId:request.seriesId,volume:{sourceRefs:[{source_id:'structural',sha256:hex('a'),provenance:'observed'}]}},volume:{caseId:'generated-test',caseHash:hash('7'),frame:'RAS+',shape,affine,mri:new Float32Array(8),compartments:[]}};
 return{request,result,source,api:{readAsset:async id=>new Uint8Array(raw[id].buffer).slice()}};
}
test('generated signed state and separate coverage remain display-only',async()=>{const f=fixture(),v=await hydrateDiagnostic(f.result,f.source,f.api);assert.deepEqual(Array.from(v.layer.stateXYZ),[-1,0,1,-1,-1,-1,-1,-1]);assert.equal(v.layer.planningInteractionPermitted,false);assert.equal(f.source.volume.compartments.length,0)});
for(const[name,mutate]of [
 ['wrong source',f=>f.source.descriptor.volume.sourceRefs[0].sha256=hex('b')],['wrong series',f=>f.result.seriesId=hash('8')],
 ['metadata asset injection',f=>f.result.descriptor.private={path:'/private',dtype:'uint8',byteLength:1}],['asset alias',f=>f.result.state.path='/private'],
 ['expanded bytes',f=>f.result.state.byteLength=9],['clinical admission',f=>f.result.descriptor.clinical_evidence=true]
])test(`refuses ${name} before assets`,async()=>{const f=fixture();mutate(f);await assert.rejects(hydrateDiagnostic(f.result,f.source,{readAsset:()=>assert.fail('before assets')}))});
test('changed transfer and decoded affine refuse',async()=>{let f=fixture();await assert.rejects(hydrateDiagnostic(f.result,f.source,{readAsset:async()=>new Uint8Array(8)}),/checksum/);f=fixture();f.result.coverageGrid.affineRAS=structuredClone(affine);f.result.coverageGrid.affineRAS[0][3]=.25;await assert.rejects(hydrateDiagnostic(f.result,f.source,f.api),/sform/)});
test('main supplies picked path; renderer path override and cancelled pick cannot read',async()=>{const f=fixture();let called=0;assert.equal(await importDiagnosticLayer(f.request,{pick:async()=>null,request:()=>assert.fail()}),null);await assert.rejects(importDiagnosticLayer({...f.request,descriptorPath:'/caller'},{pick:()=>assert.fail()}));await importDiagnosticLayer(f.request,{pick:async()=>'/picked/display-layer.json',request:async(op,args)=>{called++;assert.equal(op,'importDiagnosticLayer');assert.deepEqual(args,{...f.request,descriptorPath:'/picked/display-layer.json'})}});assert.equal(called,1)});
test('raw transfer envelope exposes only the two bounded slots',()=>{const f=fixture();for(const name of ['state','coverage']){delete f.result[name].assetId;f.result[name].path='/session/'+name+'.bin'}validateDiagnosticResult(f.result,f.request);f.result.extra=f.result.state;assert.throws(()=>validateDiagnosticResult(f.result,f.request))});
test('actual sidecar validates before any asset publication',async()=>{
 const {Sidecar}=require('../electron/sidecar.cjs');for(const tampered of [false,true]){
  const f=fixture();if(tampered)f.result.descriptor.hidden={path:'/private',dtype:'uint8',byteLength:1};let exposed=0,resolved=0;
  const sidecar=Object.create(Sidecar.prototype);sidecar.pending=new Map([['id',{op:'importDiagnosticLayer',diagnosticArgs:f.request,resolve:()=>resolved++}]]);sidecar.assets={expose:async value=>{exposed++;return value}};sidecar.emit=()=>{};
  const promise=sidecar.handle({id:'id',event:'result',result:f.result});if(tampered)await assert.rejects(promise);else await promise;
  assert.equal(exposed,tampered?0:1);assert.equal(resolved,tampered?0:1);
 }
});

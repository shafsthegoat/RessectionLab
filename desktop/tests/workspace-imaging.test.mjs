import assert from 'node:assert/strict';
import {test} from 'node:test';
import {hydrateDisplaySeries,workspaceDisplay,displaySeriesLabel} from '../src/workspace-imaging-data.ts';
import {sourceFrameDigest} from '../src/source-integrity.ts';
import {createRequire} from 'node:module';
const require=createRequire(import.meta.url);
const {importDisplaySeries}=require('../electron/display-series-import.cjs');
const h='a'.repeat(64),hash=`sha256:${h}`;
function image(identity=hash,affine=[[2,0,0,0],[0,3,0,0],[0,0,4,0],[0,0,0,1]]) {
 return {caseId:'generated',caseHash:identity,frame:'RAS+',shape:[2,2,2],affine,spacingMm:[2,3,4],mri:{assetId:'image',dtype:'float32',shape:[2,2,2],byteOrder:'little',order:'C',sha256:h,byteLength:32},compartments:[],unknowns:['generated fixture'],metadata:{workspace_display_only:true}};
}
async function fixture(){const primary=image();const volume=image(`sha256:${'b'.repeat(64)}`);volume.affine[0][3]=100;
return {primary,attachment:{schema:'workspace-display-series-v1',seriesId:`sha256:${'c'.repeat(64)}`,referenceCaseHash:primary.caseHash,referenceFrameHash:await sourceFrameDigest(primary),sourceFrameHash:await sourceFrameDigest(volume),association:{kind:'explicit_user_selected',samePersonVerified:false,sameTimeVerified:false},registration:{status:'unreviewed',reason:'No transform',sourceToReferenceRasMm:null,registrationHash:null,overlayPermitted:false},scope:'display-only-native-grid',planningEligible:false,evaluationEligible:false,modality:'MRA',modalityOrigin:'user_declared',annotationKind:'none',annotationCoverage:'unknown',acquisitionDatetime:null,expandedBytes:32,volume}}}
test('actual hydration preserves separate native grid and suppresses all primary overlays',async()=>{const {primary,attachment}=await fixture();let reads=0;const api={readAsset:async()=>{reads++;return new Uint8Array(new Float32Array([0,1,2,3,4,5,6,7]).buffer)}};const view=await hydrateDisplaySeries(primary,attachment,api);assert.equal(reads,1);assert.equal(view.volume.affine[0][3],100);const selected=workspaceDisplay({caseHash:primary.caseHash},view);assert.equal(selected.primaryOverlaysPermitted,false);assert.equal(selected.planningInteractionPermitted,false);assert.equal(selected.caseData,view.volume);assert.equal(workspaceDisplay({caseHash:primary.caseHash},null).primaryOverlaysPermitted,true)});
test('mutated frame and stale case refuse before any asset reads',async()=>{for(const mutate of [a=>a.volume.affine[0][3]++,a=>a.referenceFrameHash=hash,a=>a.sourceFrameHash=hash,a=>a.referenceCaseHash='stale',a=>a.registration.overlayPermitted=true,a=>a.planningEligible=true]){const {primary,attachment}=await fixture();mutate(attachment);await assert.rejects(hydrateDisplaySeries(primary,attachment,{readAsset:()=>assert.fail('asset read before binding check')}));}});
test('native selection refuses a stale case reference',()=>assert.throws(()=>workspaceDisplay({caseHash:hash},{descriptor:{referenceCaseHash:'wrong'}})));
async function reviewedFixture(){
 const {primary,attachment}=await fixture();attachment.volume.affine=structuredClone(primary.affine);
 const forward=[[0,-1,0,12],[1,0,0,-4],[0,0,1,7],[0,0,0,1]],pull=[[0,1,0,4],[-1,0,0,12],[0,0,1,-7],[0,0,0,1]];
 const declaration={kind:'independently-reviewed-derived-registration-v1',patient_id:primary.caseId,source_person_id:'IXI:test',group_id:'test-derived',
  source_sha256_by_modality:{T1:'d'.repeat(64),'MRA registered':'e'.repeat(64)},evidence_sha256:'a'.repeat(64),evidence_path:'qc/independent/test.json',reviewer:'generated fixture',
  review_decision:'accepted_for_registered_overlay_inspection_with_alignment_uncertainty',method_report_sha256:'b'.repeat(64),original_moving_image_sha256:'c'.repeat(64),original_annotation_sha256:'f'.repeat(64),
  original_moving_to_reference_ras_mm:forward,reference_to_original_moving_ras_mm:pull,registered_grid:'reference-native-grid'};
 for(const [volume,modality,sha] of [[primary,'T1','d'],[attachment.volume,'MRA registered','e']]){
  volume.metadata={...volume.metadata,selected_modality:modality,data_case:{person_id:'IXI:test'},workspace_registration:structuredClone(declaration)};
  volume.sourceRefs=[{source_id:'structural',uri:'urn:generated:test',sha256:sha.repeat(64),license:'generated regression',native_frame:'RAS+',provenance:'observed'}];
 }
 attachment.sourceFrameHash=await sourceFrameDigest(attachment.volume);attachment.referenceFrameHash=await sourceFrameDigest(primary);
 attachment.association={kind:'source_dataset_identity',samePersonVerified:true,sameTimeVerified:false};attachment.modalityOrigin='source_metadata';
 attachment.registration={status:'verified',reason:'Reviewed for overlays; alignment uncertainty unresolved.',registrationHash:hash,overlayPermitted:true,sourceToReferenceRasMm:[[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]};
 return {primary,attachment};
}
test('reviewed derived image hydrates on reference grid without applying original rigid mapping twice',async()=>{
 const {primary,attachment}=await reviewedFixture();const view=await hydrateDisplaySeries(primary,attachment,{readAsset:async()=>new Uint8Array(new Float32Array([0,1,2,3,4,5,6,7]).buffer)});
 assert.equal(view.descriptor.registration.sourceToReferenceRasMm[0][3],0);
 assert.match(displaySeriesLabel(attachment),/reviewed derived alignment/);assert.equal(view.descriptor.planningEligible,false);
});
test('unreviewed, wrong inverse, stale image or double-transform derived mappings refuse before assets',async()=>{
 for(const fault of ['review','inverse','hash','deformation','double-transform']){
  const {primary,attachment}=await reviewedFixture();const d=attachment.volume.metadata.workspace_registration;
  if(fault==='review')d.review_decision='pending';
  else if(fault==='inverse'){d.reference_to_original_moving_ras_mm[0][3]++;primary.metadata.workspace_registration=structuredClone(d);}
  else if(fault==='hash')attachment.volume.sourceRefs[0].sha256='0'.repeat(64);
  else if(fault==='deformation')primary.metadata.data_case.deformation={unreviewed:true};
  else attachment.registration.sourceToReferenceRasMm=d.original_moving_to_reference_ras_mm;
  await assert.rejects(hydrateDisplaySeries(primary,attachment,{readAsset:()=>assert.fail('invalid mapping read assets')}));
 }
});
test('same-modality source references and model estimates are distinguishable before selection',async()=>{
 const {attachment}=await fixture();
 const source=displaySeriesLabel({...attachment,annotationKind:'source-provided'});
 const estimated=displaySeriesLabel({...attachment,annotationKind:'estimated'});
 assert.notEqual(source,estimated);assert.equal(source,'MRA · source labels · separate frame');
 assert.equal(estimated,'MRA · model estimate · separate frame');
 assert.equal(displaySeriesLabel(attachment),'MRA · image · separate frame');
});
test('specific image captions preserve model and registration status in source choices',async()=>{
 const {attachment}=await fixture();attachment.annotationKind='estimated';
 attachment.volume.metadata.display_label='T1 · estimated tissues';
 assert.equal(displaySeriesLabel(attachment),'T1 · estimated tissues · model estimate · separate frame');
 for(const caption of ['', '  ', 'x'.repeat(81), 'misleading\ncaption', {label:'invalid'}]){
  attachment.volume.metadata.display_label=caption;
  assert.equal(displaySeriesLabel(attachment),'MRA · model estimate · separate frame');
 }
 assert.equal(attachment.planningEligible,false);assert.equal(attachment.registration.overlayPermitted,false);
});
test('host cancellation cannot start image or annotation import',async()=>{for(const annotationKind of ['none','estimated']){let calls=0;const value=await importDisplaySeries({caseHash:hash,modality:'T2',annotationKind},{pick:async()=>null,request:()=>{calls++;assert.fail('cancelled dialog')}});assert.equal(value,null);assert.equal(calls,0)}});
test('host passes only native-selected image/mask paths and declared provenance',async()=>{const choices=['/generated/image.nii.gz','/generated/mask.nii.gz'];let call;await importDisplaySeries({caseHash:hash,modality:'MRA',annotationKind:'source-provided'},{pick:async()=>choices.shift(),request:async(...args)=>{call=args;return {}}});assert.deepEqual(call,['importDisplaySeries',{caseHash:hash,modality:'MRA',annotationKind:'source-provided',imagePath:'/generated/image.nii.gz',annotationPath:'/generated/mask.nii.gz'}]);await assert.rejects(importDisplaySeries({caseHash:hash,modality:'T1',annotationKind:'none',imagePath:'/injected'},{pick:()=>assert.fail(),request:()=>assert.fail()}));});

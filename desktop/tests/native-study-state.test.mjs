import test from 'node:test';
import assert from 'node:assert/strict';
import {studyLayerDisplayKind,nextScanState,scanLabel,studyTimepoint,migrateSkullView,anatomyPresentation,defaultAnatomyPresentation,sourceEvidenceLabel} from '../src/native-study-state.ts';
import {checkedViewerSettings} from '../src/workspace-view-settings.ts';
import {cranialComposition,ctaCoverageOptions,initialNativeLayers,presentationLayers,resolveCtaCoverageVisibility} from '../src/native-study-state.ts';
import {withLayerColor} from '../src/viewer/workspaceSettings.ts';

test('an independent or estimated scan cannot drive the reference scan presentation',()=>{
  const layers=['ct','registered','estimated','independent'].map(source_series=>({source_series}));
  const doc={reference_series:'ct',transforms:{registered:{status:'source-provided'},estimated:{status:'estimated'}},layers};
  assert.deepEqual(presentationLayers(doc,'ct'),layers.slice(0,2));
  assert.deepEqual(presentationLayers(doc,'independent'),[layers[3]]);
  const displayed=['reference','source-provided','estimated','unavailable'].map((alignment,i)=>({alignment,seriesId:layers[i].source_series}));
  assert.deepEqual(presentationLayers(doc,'ct',displayed),layers.slice(0,3));
  assert.deepEqual(presentationLayers(doc,'estimated',displayed),layers.slice(0,3));
  assert.deepEqual(presentationLayers(doc,'independent',displayed),[layers[3]]);
});

test('an acquired angiogram reveals its own artery inside the skull and brain context',()=>{
  const layers=[{id:'cow',kind:'artery',source_series:'cta'},{id:'skull',kind:'skull'},{id:'brain',kind:'brain'}];
  const render=(modality,source)=>anatomyPresentation({},layers,'segmentation',modality,source);
  const view=render('CTA','cta');
  assert.equal(view.vesselShading,'cta-contrast');
  assert.equal(view.skullMode,'transparent');
  assert.equal(view.layerOpacity.cow,1);
  assert.ok(view.layerOpacity.skull<.1&&view.layerOpacity.brain<.2);
  assert.equal(view.volumeRendering.shadows,false);
  const sameSource=layers.map(layer=>layer.id==='skull'?{...layer,source_series:'cta'}:layer);
  assert.equal(anatomyPresentation({},sameSource,'vascular','CTA','cta').volumeRendering.shadows,true);
  assert.equal(anatomyPresentation({},[...sameSource,{id:'trabecular',kind:'bone',source_series:'cta'}],
    'vascular','CTA','cta').volumeRendering.shadows,false);
  assert.equal(anatomyPresentation({},layers,'scan','CTA','cta').volumeRendering.shadows,true);
  assert.equal(anatomyPresentation({},layers,'scan','CTA','cta').vesselShading,undefined);
  assert.equal(anatomyPresentation({vesselShading:'source'},layers,'vascular','CTA','cta').vesselShading,'source');
  assert.equal(render('CT','cta').skullMode,'exterior');
  assert.equal(render('CT','cta').vesselShading,undefined);
  assert.equal(render('MRA','other-scan').skullMode,'exterior');
});

test('a separate angiogram opens its own layers when initial layers describe the reference scan',()=>{
  const study={reference_series:'ct',transforms:{ct:{status:'reference'}},display:{initial_layers:['skull']},
    layers:[{id:'skull',source_series:'ct',kind:'skull'},{id:'artery',source_series:'mra',kind:'artery'},
      {id:'vein',source_series:'mra',kind:'vein'}]};
  assert.deepEqual(initialNativeLayers(study,'mra',true).visible,{skull:false,artery:true,vein:true});
  assert.deepEqual(initialNativeLayers(study,'ct',false).visible,{skull:true,artery:false,vein:false});
  study.display.initial_layers.push('artery');
  assert.deepEqual(initialNativeLayers(study,'mra',true).visible,{skull:false,artery:true,vein:false});
  study.display.initial_layers=[];
  assert.deepEqual(initialNativeLayers(study,'mra',true).visible,{skull:false,artery:false,vein:false});
});

test('CTA coverage controls require the original CTA and every exact annotation revision',()=>{
  const annotation=(id)=>({id,kind:'artery',source_series:'cta',revision:0,label_value:1,data:{sha256:id}});
  const rows=['focused','wider'].map(annotation);
  const combined={...annotation('combined'),provenance:{kind:'derived_annotation_union',source_scan_sha256:'scan',
    input_layers:rows.map(row=>({id:row.id,data_sha256:row.data.sha256,revision:row.revision,label_value:row.label_value}))}};
  const study={series:[{id:'cta',modality:'CTA',original:{sha256:'scan'}}],layers:[...rows,combined],
    display:{cta_arterial_coverage:{series_id:'cta',layer_id:'combined',focus_layers:['focused']}}};
  assert.deepEqual(ctaCoverageOptions(study,'cta'),{combinedId:'combined',focusIds:['focused'],widerIds:['wider'],memberIds:['combined','focused','wider']});
  assert.equal(ctaCoverageOptions(study,'mra'),undefined);
  rows[0].revision++;
  assert.equal(ctaCoverageOptions(study,'cta'),undefined);
  rows[0].revision--;
  study.series[0].original.sha256='different';
  assert.equal(ctaCoverageOptions(study,'cta'),undefined);
});

test('ordinary CTA defaults replace all imported coverage members and retire edited derivatives',()=>{
  const artery=id=>({id,kind:'artery',source_series:'cta',revision:0,label_value:1,data:{sha256:id}});
  const focused=artery('focus'),wider=artery('field'),peripheral=artery('distal');
  const parts=[focused,wider,peripheral];
  const combined={...artery('combined'),provenance:{kind:'derived_annotation_union',source_scan_sha256:'scan',
    input_layers:parts.map(row=>({id:row.id,data_sha256:row.data.sha256,revision:0,label_value:1}))}};
  const option={series_id:'cta',layer_id:'combined',focus_layers:['focus'],wider_layers:['field','distal']};
  const study={reference_series:'cta',transforms:{},series:[{id:'cta',modality:'CTA',original:{sha256:'scan'}}],
    layers:[...parts,combined,{id:'skull',kind:'skull',source_series:'cta'}],display:{cta_arterial_coverage:[option]}};
  assert.deepEqual(initialNativeLayers(study,'cta',false).visible,{focus:false,field:false,distal:false,combined:true,skull:true});
  focused.revision=1;
  assert.deepEqual(resolveCtaCoverageVisibility(study,{combined:true,focus:false,field:false,distal:false,skull:true}),
    {focus:true,field:false,distal:false,skull:true});
  option.retired=true;
  study.layers=study.layers.filter(layer=>layer!==combined);
  assert.deepEqual(initialNativeLayers(study,'cta',false).visible,{focus:true,field:false,distal:false,skull:true});
});

test('a shared frame opens each CTA with one coverage layer',()=>{
  const study={reference_series:'first',transforms:{second:{status:'reviewed'}},series:[],layers:[],display:{cta_arterial_coverage:[]}};
  for(const id of ['first','second']){
    const row=suffix=>({id:`${id}-${suffix}`,kind:'artery',source_series:id,revision:0,label_value:1,data:{sha256:`${id}-${suffix}`}});
    const focus=row('focus'),wider=row('wider'),combined=row('combined');
    study.series.push({id,modality:'CTA',original:{sha256:id}});
    study.layers.push(focus,wider,{...combined,provenance:{kind:'derived_annotation_union',source_scan_sha256:id,
      input_layers:[focus,wider].map(input=>({id:input.id,data_sha256:input.data.sha256,revision:0,label_value:1}))}});
    study.display.cta_arterial_coverage.push({series_id:id,layer_id:combined.id,focus_layers:[focus.id],wider_layers:[wider.id]});
  }
  const shared=initialNativeLayers(study,'first',false).visible;
  assert.deepEqual(Object.keys(shared).filter(id=>shared[id]),['first-combined','second-combined']);
  const independent=initialNativeLayers(study,'second',true).visible;
  assert.deepEqual(Object.keys(independent).filter(id=>independent[id]),['second-combined']);
});

test('scan and anatomy presets preserve the study colors of supplied arterial classes',()=>{
  const branch={id:'left-ica',kind:'artery',source_series:'ct',color:'#278c69',label_value:6,
    provenance:{kind:'supplied_annotation',publisher_class_id:6}};
  const layers=[branch,{...branch,id:'union',provenance:{kind:'supplied_annotation'}},
    {...branch,id:'mismatched',label_value:7},{...branch,id:'estimate',provenance:{kind:'model',publisher_class_id:6}}];
  for(const mode of ['scan','segmentation','vascular']){
    const result=anatomyPresentation({},layers,mode,'CTA','ct');
    assert.deepEqual(result.layerColor,{'left-ica':'#278c69',union:'#aa352b',mismatched:'#aa352b',estimate:'#aa352b'});
  }
});

test('editing a CTA vessel layer color explicitly selects label colors',()=>{
  const settings={vesselShading:'cta-contrast',layerColor:{artery:'#278c69'}};
  assert.deepEqual(withLayerColor(settings,'artery','artery','#d04438'),
    {vesselShading:'source',layerColor:{artery:'#d04438'}});
  assert.equal(withLayerColor(settings,'brain','brain','#a399ad').vesselShading,'cta-contrast');
  assert.equal(settings.layerColor.artery,'#278c69');
  assert.deepEqual(checkedViewerSettings({metadata:{},compartments:[]},{vesselShading:'cta-contrast'}),{vesselShading:'cta-contrast'});
});

test('an unclassified native TOF network gets foreground emphasis without an artery color or a CT bone profile',()=>{
  const layers=[{id:'network',kind:'vessels',source_series:'tof'},{id:'bone',kind:'skull'},{id:'artery',kind:'artery',source_series:'tof'}];
  const view=anatomyPresentation({},layers,'segmentation','3D TOF','tof');
  assert.equal(view.layerOpacity.network,1);
  assert.notEqual(view.layerColor.network,view.layerColor.artery);
  assert.equal(view.skullMode,'transparent');
  assert.ok(view.volumeRendering.density<.03);
  assert.equal(anatomyPresentation({},layers,'segmentation','T1','other').skullMode,'exterior');
});

test('vascular presentation mutes the reviewed peripheral layers and keeps the main network opaque',()=>{
  const group={id:'vesselfm-review-posterior-superficial',kind:'vessels',source_series:'tof',color:'#8d877a',
    provenance:{method:'Review display group of complete native TOF components',evidence_use:'context_only'}};
  const fieldEdge={...group,id:'vesselfm-review-field-edge'};
  const skullExterior={...group,id:'vesselfm-review-skull-exterior'};
  const network={...group,id:'vesselfm-review-intracranial-priority',color:'#b75848'};
  const estimate={...group,provenance:{kind:'model'}};
  const settings=anatomyPresentation({},[group,fieldEdge,skullExterior,network],'vascular','TOF','tof');
  for(const peripheral of [group,fieldEdge,skullExterior]){
    assert.equal(settings.layerColor[peripheral.id],peripheral.color);
    assert.equal(settings.layerOpacity[peripheral.id],.32);
  }
  assert.equal(settings.layerOpacity[network.id],1);
  assert.equal(anatomyPresentation({},[estimate],'vascular','TOF','tof').layerOpacity[group.id],1);
});

test('mixed MR Vessels starts with visible source cortex and vessels, with the native volume available',()=>{
  const cortex={id:'cortex',kind:'brain-region',source_series:'t1',provenance:{model:'supersynth',label_ontology:{value:3}}};
  const network={id:'network',kind:'vessels',source_series:'tof',provenance:{kind:'image_derived_estimate'}};
  const skull={id:'skull',kind:'skull',source_series:'t1',provenance:{kind:'image_derived_estimate'}};
  const lesion={id:'lesion',kind:'tumor',source_series:'t1'};
  const acquiredSkull={...skull,id:'ct-skull',source_series:'cta'};
  const peripheral={...network,id:'vesselfm-review-field-edge',provenance:{
    method:'Review display group of complete native TOF components',evidence_use:'context_only'}};
  const layers=[cortex,network,skull,lesion,acquiredSkull,peripheral];
  const series=[{id:'t1',modality:'T1CE'},{id:'tof',modality:'TOF'},{id:'cta',modality:'CTA'}];
  const visible={cortex:true,network:true,skull:true,lesion:true,'ct-skull':true,[peripheral.id]:true};
  const snapshot=structuredClone({layers,series,visible});
  const present=(mode='vascular',modality='T1CE',admitted=layers,visibility=visible)=>
    anatomyPresentation({},admitted,mode,modality,'t1',{visible:visibility,series});
  const vessels=present();
  assert.equal(vessels.sourcedVesselContext,true);
  assert.equal(vessels.volumeRendering.enabled,false);
  assert.equal(vessels.volumeRendering.density,.02);
  assert.equal(anatomyPresentation({smoothSurfaces:false},layers,'vascular','T1CE','t1',{visible,series}).smoothSurfaces,false);
  assert.equal(vessels.layerOpacity.network,1);
  assert.ok(vessels.layerOpacity.skull<vessels.layerOpacity.cortex);
  assert.notEqual(vessels.layerColor.skull,vessels.layerColor['ct-skull']);
  assert.equal(vessels.volumeRendering.balanceTofContext,undefined);
  const independentTof=anatomyPresentation({},[network],'vascular','TOF','tof',{visible,series});
  assert.equal(independentTof.volumeRendering.balanceTofContext,true);
  assert.equal(independentTof.layerColor.network,'#c16d72');
  assert.equal(anatomyPresentation({smoothSurfaces:false},[network],'vascular','TOF','tof').smoothSurfaces,false);
  assert.equal(anatomyPresentation({},[network],'scan','TOF','tof').volumeRendering.balanceTofContext,undefined);
  assert.ok(vessels.layerOpacity.lesion>0&&vessels.layerOpacity.lesion<1);
  assert.equal(present('segmentation').layerOpacity.lesion,1);
  for(const view of [present('scan'),present('segmentation'),present('vascular','CTA'),
    present('vascular','unknown'),present('vascular','T1CE',[network]),
    present('vascular','T1CE',[cortex,skull]),present('vascular','T1CE',layers,{...visible,cortex:false}),
    present('vascular','T1CE',layers,{...visible,network:false}),
    present('vascular','TOF',[{...cortex,source_series:'tof'},network])]){
    assert.equal(view.sourcedVesselContext,undefined);
    assert.equal(view.volumeRendering.enabled,true);
  }
  assert.deepEqual({layers,series,visible},snapshot);
});

test('cortical presentation requires an identified cortical label and preserves target and source data',()=>{
  const siam=value=>({kind:'brain-region',label_ontology:{value},provenance:{model:'siam-v2'}});
  const cortex={id:'cortex',...siam(1)},deep={id:'deep',...siam(6)};
  const skull={id:'skull',kind:'skull'},tumor={id:'target',kind:'tumor'};
  assert.equal(studyLayerDisplayKind(deep),'brain-region');
  assert.equal(studyLayerDisplayKind({...cortex,provenance:{model:'other'}}),'brain-region');
  for(const value of [3,42])assert.equal(studyLayerDisplayKind({...cortex,provenance:{model:'fastsurfer'},label_ontology:{value}}),'gray-matter');
  const settings={layerOpacity:{skull:1},camera:{position:[-50,20,0]}};
  const before=structuredClone(settings);
  const anatomy=anatomyPresentation(settings,[cortex,deep,skull,tumor],'segmentation','T1CE');
  assert.equal(anatomy.skullMode,'transparent');
  assert.ok(anatomy.layerOpacity.cortex>anatomy.layerOpacity.skull);
  assert.equal(anatomy.layerColor.target,'#65bfd3');
  assert.deepEqual(anatomy.camera,settings.camera);
  assert.deepEqual(settings,before);
  assert.equal(anatomyPresentation(settings,[deep,skull,tumor],'segmentation','CT').skullMode,'exterior');
});

test('structural contrast retains study angiography availability and derived diffusion status',()=>{
  const series=[{modality:'T1'},{modality:'MRA'},{modality:'DTI FA diagnostic'}];
  assert.match(sourceEvidenceLabel(series[0],series),/angiography available in Scans/);
  assert.match(sourceEvidenceLabel(series[1],series),/MR angiography/);
  assert.match(sourceEvidenceLabel(series[2],series),/derived map; planning use excluded/);
  assert.match(sourceEvidenceLabel(series[0],[series[0]]),/no acquired angiography in this study/);
});

test('scan switches retain the physical cursor and anatomy while restoring scan-specific contrast',()=>{
  const current={cursor:[1,2,3],visible:{brain:true,tumor:true,vessels:false},settings:{contrast:2,contextStyle:'shell',sliceFraming:{fit:'anatomy',zoom:2},layerOpacity:{brain:.15},layerColor:{tumor:'#aa00ff'}}};
  const saved={cursor:[4,5,6],visible:{brain:false,tumor:false,vessels:true},settings:{contrast:4,sliceFraming:{fit:'image',zoom:.75},volumeRendering:{preset:'ct-bone'}}};
  const next=nextScanState(current,saved);
  assert.deepEqual(next.visible,current.visible);
  assert.deepEqual(next.settings.layerOpacity,current.settings.layerOpacity);
  assert.equal(next.settings.contrast,4);
  assert.equal(next.settings.volumeRendering.preset,'ct-bone');
  assert.equal(next.settings.contextStyle,'shell');
  assert.deepEqual(next.settings.sliceFraming,{fit:'image',zoom:.75});
  assert.deepEqual(next.cursor,current.cursor);
  next.cursor[0]=99;assert.deepEqual(current.cursor,[1,2,3]);assert.deepEqual(saved.cursor,[4,5,6]);
  next.visible.brain=false;
  assert.equal(current.visible.brain,true);
});
test('study labels use scan names and preserve a distinct postoperative study',()=>{
  assert.equal(scanLabel({id:'ct-brain',modality:'CT'}),'CT BRAIN');
  assert.equal(scanLabel({id:'12345678-abcd-1234-abcd-123456789012',modality:'MRI',label:'T2 FLAIR'}),'T2 FLAIR');
  assert.equal(studyTimepoint({document:{study_timepoint:'postoperative',series:[]}}),'postoperative');
});

test('legacy skull transparency migrates without rewriting explicitly selected modes',()=>{
  const layers=[{id:'skull-ct',kind:'bone'},{id:'vessels',kind:'vessels'}];
  const view={cursor:null,visible:{'skull-ct':true},settings:{layerOpacity:{'skull-ct':.08,vessels:1},clipping:{axis:'z',value:0,enabled:true}}};
  const next=migrateSkullView(view,layers);
  assert.equal(next.settings.skullMode,'exterior');
  assert.equal(next.settings.layerOpacity['skull-ct'],1);
  assert.equal(next.settings.clipping.enabled,false);
  assert.equal(view.settings.layerOpacity['skull-ct'],.08);
  const explicit={...view,settings:{...view.settings,skullMode:'transparent'}};
  assert.equal(migrateSkullView(explicit,layers),explicit);
});

test('legacy vessel filters migrate once; explicit new choices survive save and reopen',async()=>{
  const {vesselScopeForView}=await import('../src/native-study-state.ts');
  assert.equal(vesselScopeForView(),'all');
  assert.equal(vesselScopeForView(null,true),'exterior-air');
  assert.equal(vesselScopeForView({vesselDisplayVersion:2,vesselScope:'all'},true),'all');
  assert.equal(vesselScopeForView({vesselDisplayVersion:2,vesselScope:'exterior-air'},true),'exterior-air');
  for(const vesselScope of ['brain','intracranial','excluded','all',undefined]){
    const legacy={vesselScope,states:{mri:{settings:{contrast:4}}}};
    const before=structuredClone(legacy);
    assert.equal(vesselScopeForView(legacy),'all');
    assert.deepEqual(legacy,before);
    const saved=JSON.parse(JSON.stringify({...legacy,vesselDisplayVersion:2}));
    assert.equal(vesselScopeForView(saved),vesselScope??'all');
  }
});

test('presentation choices preserve source identities, selected anatomy, camera and quantitative masks',()=>{
  const layers=[{id:'vessels',kind:'vessels',source_series:'t1ce',mask:new Uint8Array([1,0,1])},{id:'tumor',kind:'tumor'},{id:'bone',kind:'bone'},{id:'brain',kind:'brain'}];
  const camera={position:[1,2,3],target:[0,0,0],up:[0,0,1]};
  const settings={camera,contrast:1.7,layerOpacity:{custom:.4},layerColor:{custom:'#123456'},clipping:{enabled:true,axis:'z',value:2},volumeRendering:{preset:'ct-angiography',points:[[0,0,0,0,0],[1,1,0,0,1]]}};
  const original=structuredClone({settings,layers});
  for(const mode of ['segmentation','vascular']){
    const next=anatomyPresentation(settings,layers,mode,'T1CE');
    assert.equal(next.camera,camera);
    assert.equal(next.contrast,1.7);
    assert.equal(next.sliceOverlay,'contour');
    assert.equal(next.contextStyle,'shell');
    assert.deepEqual(next.sliceFraming,{fit:'anatomy',zoom:1});
    assert.equal(next.volumeRendering.preset,'auto');
    assert.equal(next.volumeRendering.points,undefined);
    assert.equal(next.clipping.enabled,false);
    assert.equal(next.layerOpacity.custom,.4);
    assert.ok(next.layerOpacity.vessels>0);
    assert.ok(next.layerOpacity.tumor>.9);
    assert.equal(next.layerColor.custom,'#123456');
    assert.deepEqual(JSON.parse(JSON.stringify(next)).presentation,mode);
  }
  assert.deepEqual({settings,layers},original);
});

test('SIAM residual head tissue uses its anatomical context style while other source labels retain their kind',()=>{
  const provenance={model:'siam-v2',label_ontology:{value:16}};
  const tissue={id:'head-other',kind:'context',provenance};
  assert.equal(studyLayerDisplayKind(tissue),'outer-head-tissue');
  assert.equal(studyLayerDisplayKind({...tissue,provenance:{...provenance,model:'different'}}),'context');
  assert.equal(studyLayerDisplayKind({...tissue,label_ontology:{value:3}}),'context');
  assert.equal(anatomyPresentation({},[tissue],'scan','T1').smoothSurfaces,true);
  assert.equal(anatomyPresentation({smoothSurfaces:false},[tissue],'scan','T1').smoothSurfaces,false);
  assert.equal(anatomyPresentation({},[tissue],'segmentation','T1').layerOpacity[tissue.id],.14);
});

test('Scan explicitly restores acquired signal and restrained overlays without overwriting stored settings',()=>{
  const layers=[{id:'brain',kind:'brain'},{id:'cerebellum',kind:'cerebellar-cortex'},{id:'brainstem',kind:'brainstem'},{id:'review',kind:'review-region'},{id:'tumor',kind:'tumor'},{id:'bone',kind:'bone'}];
  const saved={presentation:'segmentation',smoothSurfaces:false,volumeRendering:{enabled:false,density:.025,points:[[0,0,0,0,0],[1,1,1,1,.1]]},layerOpacity:{tumor:1}};
  const original=structuredClone(saved);
  for(const modality of ['T1CE','MRI','T2 FLAIR','CTA','CT','3D TOF MRA']){
    assert.equal(defaultAnatomyPresentation(modality),'scan');
    const view=anatomyPresentation(saved,layers,'scan',modality);
    assert.equal(view.presentation,'scan');
    assert.equal(view.volumeRendering.enabled,true);
    assert.ok(view.volumeRendering.density>.2);
    assert.equal(view.volumeRendering.showCtBoneOverlay,/^CT/.test(modality)?false:undefined);
    assert.equal(view.volumeRendering.points,undefined);
    assert.equal(view.contextStyle,'uniform');
    assert.equal(view.smoothSurfaces,false);
    assert.ok(view.layerOpacity.cerebellum<.05);
    assert.ok(view.layerOpacity.brainstem<.05);
    assert.equal(view.layerOpacity.review,undefined,'Review evidence keeps its authored appearance');
    assert.ok(view.layerOpacity.tumor>view.layerOpacity.brain);
    assert.ok(view.layerOpacity.tumor<.5);
    assert.equal(view.sliceOverlay,'contour');
  }
  const hybrid={...saved,volumeRendering:{...saved.volumeRendering,showCtBoneOverlay:true}};
  const structures=anatomyPresentation(hybrid,layers,'segmentation','CT');
  assert.equal(anatomyPresentation(structures,layers,'scan','CT').volumeRendering.showCtBoneOverlay,true);
  for(const modality of ['segmentation only','analytic signal','generated image signal','DTI FA diagnostic']){
    assert.equal(defaultAnatomyPresentation(modality),'segmentation');
    assert.equal(anatomyPresentation(saved,layers,'scan',modality).presentation,'segmentation');
  }
  assert.deepEqual(saved,original);
});

test('saved context and physical slice framing reopen detached and reject malformed extents',()=>{
  const source={compartments:[],brainMask:null,metadata:{}};
  const savedVessels={presentation:'vascular',sourcedVesselContext:true,volumeRendering:{enabled:true}};
  assert.deepEqual(checkedViewerSettings(source,savedVessels),savedVessels);
  assert.throws(()=>checkedViewerSettings(source,{...savedVessels,sourcedVesselContext:'true'}));
  for(const show of [true,false]){
    const savedCt={volumeRendering:{preset:'ct-angiography',showCtExteriorSignal:show,showCtBoneOverlay:!show}};
    assert.deepEqual(checkedViewerSettings(source,savedCt),savedCt);
    const savedTof={presentation:'vascular',volumeRendering:{preset:'auto',balanceTofContext:show}};
    assert.deepEqual(checkedViewerSettings(source,savedTof),savedTof);
  }
  for(const invalid of ['false',0,null]){
    assert.throws(()=>checkedViewerSettings(source,{volumeRendering:{showCtExteriorSignal:invalid}}));
    assert.throws(()=>checkedViewerSettings(source,{volumeRendering:{showCtBoneOverlay:invalid}}));
    assert.throws(()=>checkedViewerSettings(source,{volumeRendering:{balanceTofContext:invalid}}));
  }
  for(const fit of ['image','anatomy'])for(const zoom of [.5,1,4])for(const contextStyle of ['shell','uniform']){
    const saved={sliceFraming:{fit,zoom},contextStyle};
    const reopened=checkedViewerSettings(source,saved);
    assert.deepEqual(reopened,saved);
    reopened.sliceFraming.zoom=2;
    assert.equal(saved.sliceFraming.zoom,zoom);
  }
  for(const sliceFraming of [null,[],{}, {fit:'brain',zoom:1},{fit:'image',zoom:true},{fit:'image',zoom:.49},{fit:'anatomy',zoom:4.01},{fit:'image',zoom:NaN},{fit:'image',zoom:1,alterSource:true}])
    assert.throws(()=>checkedViewerSettings(source,{sliceFraming}));
  for(const contextStyle of [null,{},[],'unknown'])assert.throws(()=>checkedViewerSettings(source,{contextStyle}));
});

test('a study without a current cursor initializes from the saved scan without aliasing it',()=>{
  const saved={cursor:[4,5,6],visible:{},settings:{contrast:3}},current={cursor:null,visible:{},settings:{}};
  const next=nextScanState(current,saved);assert.deepEqual(next.cursor,saved.cursor);next.cursor[0]=99;assert.deepEqual(saved.cursor,[4,5,6]);
});

test('body orientation guide saves per scan, accepts old settings and rejects non-boolean flags',()=>{
  const source={compartments:[],brainMask:null,metadata:{}};
  assert.equal(checkedViewerSettings(source,{}).showOrientationGuide,undefined);
  for(const enabled of [true,false]){
    const saved=checkedViewerSettings(source,JSON.parse(JSON.stringify({showOrientationGuide:enabled})));
    assert.equal(saved.showOrientationGuide,enabled);
    const current={cursor:null,visible:{},settings:{showOrientationGuide:!enabled}},scan={cursor:null,visible:{},settings:saved};
    assert.equal(nextScanState(current,scan).settings.showOrientationGuide,enabled);
    assert.equal(anatomyPresentation(saved,[],'segmentation','T1').showOrientationGuide,enabled);
  }
  for(const value of [null,0,1,'false',[],{}])assert.throws(()=>checkedViewerSettings(source,{showOrientationGuide:value}));
});

test('linked slices share the patient camera and keep acquisition-specific scalar settings',()=>{
  const current={cursor:[32,15,11],visible:{lesion:true,artery:true},settings:{
    presentation:'segmentation',camera:{position:[0,0,100],target:[4,3,2],up:[0,1,0]},volumeRendering:{density:.12},contrast:1}};
  const saved={cursor:[1,2,3],visible:{},settings:{contrast:2,volumeRendering:{density:.7}}};
  const linked=nextScanState(current,saved);
  assert.deepEqual(linked.cursor,current.cursor);
  assert.deepEqual(linked.settings.camera,current.settings.camera);
  assert.deepEqual(linked.settings.volumeRendering,saved.settings.volumeRendering);
  assert.equal(linked.settings.contrast,2);
  assert.deepEqual(nextScanState(current,saved,false).cursor,saved.cursor);
});


test('cranial views share the source camera and contours and restrict layers to admitted evidence',()=>{
  const layers=[['gray','cerebral-cortex'],['cerebellum','cerebellar-cortex'],['head','head-envelope'],
    ['target','tumor'],['other-target','tumor'],['csf','ventricles'],['veins','vein'],['region','brain-region']].map(
    ([id,kind])=>({id,kind,color:'#6283ac'}));
  const ids=[...layers.map(layer=>layer.id),'tract','independent'];
  const camera={position:[-400,0,0],target:[0,0,0],up:[0,0,1]};
  const initial={cursor:[9,3,4],visible:{tract:true},settings:{camera,sliceOverlay:'fill',sliceFraming:{fit:'image',zoom:1.5},clipping:{enabled:false}}};
  const visible=view=>Object.keys(view.visible).filter(id=>view.visible[id]);
  const plan=cranialComposition(initial,layers,ids,'planning','target','veins');
  assert.deepEqual(visible(plan),['gray','cerebellum','head','target']);
  const inspect=cranialComposition(plan,layers,ids,'inspection','target','region');
  assert.deepEqual(visible(inspect),['gray','cerebellum','head','target','region']);
  assert.ok(inspect.settings.layerOpacity.gray<plan.settings.layerOpacity.gray);
  assert.ok(inspect.settings.layerOpacity.head>plan.settings.layerOpacity.head);
  assert.ok(inspect.settings.layerOpacity.region>inspect.settings.layerOpacity.gray);
  const fallback=cranialComposition(plan,layers,ids,'inspection','target','gray');
  assert.equal(fallback.visible.csf,true);
  for(const view of [plan,inspect]){
    assert.equal(view.settings.camera,camera);assert.equal(view.settings.sliceOverlay,'contour');
    assert.equal(view.settings.sliceFraming,initial.settings.sliceFraming);assert.equal(view.settings.clipping,initial.settings.clipping);
    assert.equal(view.visible.independent,false);
  }
  assert.deepEqual(initial.visible,{tract:true});
});

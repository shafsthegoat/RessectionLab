import assert from "node:assert/strict";
import { test } from "node:test";
import {
  hydrateCase,
  initialCursor,
  validateCaseDescriptor,
  anatomyColor,
  annotationCenter,
} from "../src/case-data.ts";

const hash = "a".repeat(64);
const descriptor = (dtype, assetId) => ({
  assetId,
  dtype,
  shape: [2, 2, 2],
  byteOrder: "little",
  order: "C",
  sha256: hash,
  byteLength: dtype === "float32" ? 32 : 8,
});
function fixture() {
  return {
    caseId: "fixture",
    caseHash: `sha256:${hash}`,
    frame: "RAS+",
    affine: [
      [2, 0, 0, 10],
      [0, 3, 0, -5],
      [0, 0, 4, 7],
      [0, 0, 0, 1],
    ],
    shape: [2, 2, 2],
    spacingMm: [2, 3, 4],
    mri: descriptor("float32", "mri"),
    compartments: [
      {
        name: "enhancing_target",
        volumeMm3: 24,
        array: descriptor("uint8", "mask"),
      },
    ],
    unknowns: ["motor_unknown"],
    metadata: { is_synthetic: true },
  };
}
function api(overrides = {}) {
  const assets = {
    mri: new Uint8Array(new Float32Array([0, 1, 2, 3, 4, 5, 6, 7]).buffer),
    mask: new Uint8Array([0, 0, 0, 0, 0, 0, 0, 1]),
    ...overrides,
  };
  return { readAsset: async (id) => assets[id] };
}

test("full-head80-layer inventory hydrates independently and refuses81", async () => {
  const input=fixture();input.compartments=Array.from({length:80},(_,i)=>({name:`structure_${i}`,volumeMm3:24,array:descriptor('uint8','mask')}));
  const data=await hydrateCase(input,api());
  assert.equal(data.compartments.length,80);
  assert.deepEqual([...data.compartments[79].mask],[0,0,0,0,0,0,0,1]);
  input.compartments.push({name:'excess',volumeMm3:24,array:descriptor('uint8','mask')});
  assert.throws(()=>validateCaseDescriptor(input),/layer inventory/);
});

test("valid anisotropic affine preserves C-order arrays and physical cursor", async () => {
  const data = await hydrateCase(fixture(), api());
  assert.deepEqual([...data.mri], [0, 1, 2, 3, 4, 5, 6, 7]);
  assert.deepEqual(initialCursor(data), [12, -2, 11]);
  assert.deepEqual(data.affine, fixture().affine);
});
test("native vessel score lighting is optional and must match the declared vertices", async () => {
  const input=fixture(), layer=input.compartments[0];
  input.metadata.anatomy_layers={enhancing_target:{kind:'vessel'}};
  const typed=(dtype,assetId,shape)=>({...descriptor(dtype,assetId),shape,byteLength:shape.reduce((a,b)=>a*b)*4});
  layer.vesselSurface={positions:typed('float32','positions',[3,3]),indices:typed('uint32','indices',[1,3]),
    normals:typed('float32','normals',[3,3]),sourceCells:typed('uint32','owners',[3])};
  const score=Float32Array.of(0,0,1,.7,0,1,0,0,1,0,0,.4);
  const bytes=array=>new Uint8Array(array.buffer);
  const supplied=api({positions:bytes(Float32Array.of(1,.8,1,1,1,.8,.8,1,1)),indices:bytes(Uint32Array.of(0,1,2)),
    normals:bytes(Float32Array.of(0,0,1,0,1,0,1,0,0)),owners:bytes(Uint32Array.of(7,7,7)),lighting:bytes(score)});
  assert.equal((await hydrateCase(input,supplied)).compartments[0].vesselScoreLighting,undefined);
  layer.vesselSurface.scoreLighting=typed('float32','lighting',[3,4]);
  assert.deepEqual((await hydrateCase(input,supplied)).compartments[0].vesselScoreLighting,score);
  score[3]=1.1;
  await assert.rejects(hydrateCase(input,supplied),/Invalid vessel score lighting values/);
  layer.vesselSurface.scoreLighting=typed('float32','lighting',[2,4]);
  assert.throws(()=>validateCaseDescriptor(input),/score lighting disagrees/);
});
test("LPS is converted once without resampling voxel data", async () => {
  const input = fixture();
  input.frame = "LPS+";
  const data = await hydrateCase(input, api());
  assert.equal(data.frame, "RAS+");
  assert.deepEqual(initialCursor(data), [-12, 2, 11]);
  assert.deepEqual([...data.compartments[0].mask], [0, 0, 0, 0, 0, 0, 0, 1]);
  assert.equal(input.affine[0][0], 2);
});
test("bound engine brain walls sample the matching acquired reference on the converted grid", async () => {
  const input = fixture();
  input.frame = "LPS+";
  input.compartments.push({...input.compartments[0], name: "brain"});
  input.metadata.anatomy_layers = {brain: {kind: "brain"}, enhancing_target: {kind: "tumor"}};
  input.artifacts = {
    nativeStudyReplay: {caseHash: input.caseHash, patientId: "p", sourceSnapshotId: "revision"},
    surgical_toolbox: {exactGridReplay: {sourceManifest: {
      patient_id: "p", snapshot_id: "revision", reference_series: "t1",
      series: [{id: "t1", modality: "T1", original: {sha256: hash}}],
    }}},
  };
  input.sourceRefs = [{source_id: "structural", provenance: "observed", sha256: hash}];
  const data = await hydrateCase(input, api());
  const signal = data.compartments[1].appearanceSource;
  assert.equal(signal.data, data.mri);
  assert.deepEqual(signal.affine, data.affine);
  assert.equal(data.compartments[0].appearanceSource, undefined);
  input.artifacts.nativeStudyReplay.sourceSnapshotId = "different-revision";
  assert.equal((await hydrateCase(input, api())).compartments[1].appearanceSource, undefined);
  input.artifacts.nativeStudyReplay.sourceSnapshotId = "revision";
  input.sourceRefs[0].provenance = "simulated";
  assert.equal((await hydrateCase(input, api())).compartments[1].appearanceSource, undefined);
});
test("local operative brain signal follows the acquired CTA and the exact native wall binding", async () => {
  const input = fixture();
  const shape = [5, 5, 4], count = 100;
  const volume = (dtype, assetId) => ({...descriptor(dtype, assetId), shape, byteLength:count*(dtype==='float32'?4:1)});
  const typed = (dtype, assetId, dims) => ({...descriptor(dtype, assetId), shape:dims, byteLength:dims.reduce((a,b)=>a*b)*4});
  const bytes = array => new Uint8Array(array.buffer);
  const vessels = {positions:typed('float32','positions',[3,3]), indices:typed('uint32','indices',[1,3]),
    normals:typed('float32','normals',[3,3]), sourceCells:typed('uint32','owners',[3])};
  Object.assign(input,{shape,mri:volume('float32','mri'),compartments:[{name:'brain',array:volume('uint8','brain'),volumeMm3:99*24},
    {name:'vessels',array:volume('uint8','vessels'),vesselSurface:vessels,volumeMm3:24}],
    sourceRefs:[{source_id:'ct',provenance:'observed',sha256:hash}]});
  input.metadata.anatomy_layers={brain:{kind:'brain'},vessels:{kind:'vessels'}};
  input.metadata.surgical_adapter={type:'native-patient-research-derivative',
    nativeWall:{schema:'resectionlab.patient-native-wall.v1',sourceSeries:'ct',sourceShape:[8,8,8],sourceStartIJK:[1,2,3],
      geometryBinding:[hash,hash,hash].join(':'),sourceImageSha256:hash},
    sourceLayers:{brain:{layerId:'brain-source',sampledCoverageCells:count}},
    sourceManifest:{reference_series:'ct',layers:[{id:'brain-source',kind:'brain',source_series:'ct'}],
      series:[{id:'ct',modality:'CTA',affine:[[2,0,0,8],[0,3,0,-11],[0,0,4,-5],[0,0,0,1]],
        data:{shape:[8,8,8]},original:{sha256:hash}}]}};
  const mri=Float32Array.from({length:count},(_,i)=>i===99?850:i);
  const brain=Uint8Array.from({length:count},(_,i)=>Number(i!==99));
  const supplied=api({mri:bytes(mri),brain,vessels:Uint8Array.from(brain,bit=>1-bit),
    positions:bytes(Float32Array.of(4,4,2.8,4,3.8,3,3.8,4,3)), indices:bytes(Uint32Array.of(0,1,2)),
    normals:bytes(Float32Array.of(0,0,1,0,1,0,1,0,0)),owners:bytes(Uint32Array.of(99,99,99))});
  const view=await hydrateCase(input,supplied);
  const signal=view.compartments[0].appearanceSource;
  assert.deepEqual(signal.softTissueWindow,[14,89]);
  assert.equal(signal.nativeWallCrop,true);
  assert.equal(signal.data,view.mri);
  assert.deepEqual(view.mri,mri);
  assert.deepEqual(view.compartments[0].mask,brain);
  assert.equal(view.compartments[1].appearanceSource,undefined);
  for(const mismatch of [
    draft=>{draft.sourceRefs[0].provenance='simulated';},
    draft=>{draft.metadata.surgical_adapter.nativeWall.geometryBinding='different';},
    draft=>{draft.metadata.surgical_adapter.nativeWall.sourceStartIJK=[1,2,2];},
    draft=>{draft.metadata.surgical_adapter.sourceLayers.brain.sampledCoverageCells=99;},
    draft=>{draft.metadata.surgical_adapter.sourceManifest.layers[0].source_series='different';},
  ]){
    const altered=structuredClone(input);
    mismatch(altered);
    assert.equal((await hydrateCase(altered,supplied)).compartments[0].appearanceSource,undefined);
  }

  const paired = structuredClone(input);
  const adapter = paired.metadata.surgical_adapter;
  const manifest = adapter.sourceManifest;
  const cta = manifest.series[0];
  const vascularHash = 'b'.repeat(64);
  const tof = {...structuredClone(cta), id:'mra', modality:'MRA', timepoint:'baseline', original:{sha256:vascularHash},
    affine:[[0,3,0,4],[-2,0,0,6],[0,0,4,-12],[0,0,0,1]]};
  cta.timepoint = 'baseline';
  manifest.patient_id = 'paired-patient';
  manifest.series.push(tof);
  paired.sourceRefs.push({source_id:'mra',provenance:'observed',sha256:vascularHash});
  Object.assign(adapter.nativeWall,{sourceSeries:'mra',sourceImageSha256:vascularHash,layerId:'loaded-wall',centerRasMm:[10,-5,7]});
  const matrix = [[0,-1,0,14],[1,0,0,-15],[0,0,1,7],[0,0,0,1]];
  const sources = Object.fromEntries([cta,tof].map(series=>[series.id,{originalSha256:series.original.sha256,
    nativeAffine:series.affine,nativeArray:series.data,timepoint:series.timepoint}]));
  const review = {schema:'resectionlab.native-wall-alignment-review.v1',digest:'c'.repeat(64),
    binding:{sources,matrix,sourcePatient:manifest.patient_id,referenceSeries:'ct',vascularSeries:'mra',
      brainLayerId:'brain-source',vesselLayerId:'loaded-wall'},
    centerRasMm:[10,-5,7],cropStartIJK:[1,2,3],cropShape:shape};
  manifest.transforms = {ct:{status:'reference'},mra:{status:'unreviewed',matrix:structuredClone(matrix),
    source_sha256:vascularHash,reference_sha256:hash,native_wall_alignment_review:structuredClone(review)}};
  adapter.nativeWallAlignmentReview = review;
  const multimodal = await hydrateCase(paired,supplied);
  const alignedSignal = multimodal.compartments[0].appearanceSource;
  assert.equal(alignedSignal.seriesId,'ct');
  assert.equal(alignedSignal.sourceSha256,hash);
  assert.equal(alignedSignal.data,multimodal.mri);
  assert.deepEqual(alignedSignal.affine,paired.affine);
  assert.deepEqual(alignedSignal.softTissueWindow,signal.softTissueWindow);
  for(const modality of ['T1CE','T2','FLAIR','TOF','SWI','MR DWI']){
    const source=structuredClone(paired);
    source.metadata.surgical_adapter.sourceManifest.series[0].modality=modality;
    const field=await hydrateCase(source,supplied);
    const structuralSignal=field.compartments[0].appearanceSource;
    if(['T1CE','T2','FLAIR'].includes(modality)){
      assert.equal(structuralSignal.nativeWallCrop,true);
      assert.equal(structuralSignal.modality,modality);
      assert.equal(structuralSignal.data,field.mri);
      assert.equal(structuralSignal.sourceSha256,hash);
    }else assert.equal(structuralSignal,undefined);
  }
  for(const mismatch of [
    (a,p)=>{p.sourceRefs.pop();},
    (a)=>{delete a.nativeWallAlignmentReview;},
    (a)=>{a.sourceManifest.transforms.mra.matrix[0][3]+=2;},
    (a)=>{a.nativeWallAlignmentReview.cropStartIJK[0]+=1;},
    (a)=>{a.sourceManifest.series[0].original.sha256=vascularHash;},
    (a,p)=>{p.affine[1][3]+=2;},
  ]){
    const altered=structuredClone(paired);
    mismatch(altered.metadata.surgical_adapter,altered);
    assert.equal((await hydrateCase(altered,supplied)).compartments[0].appearanceSource,undefined);
  }
});
test("annotation centering preserves the recorded source point and converts LPS exactly once", async () => {
  for (const frame of ["RAS+", "LPS+"]) {
    const input = fixture();
    input.frame = frame;
    input.targetCentroidMm = [12, -2, 11];
    const data = await hydrateCase(input, api());
    const beforeMask = data.compartments[0].mask.slice();
    const beforeAffine = structuredClone(data.affine);
    const center = annotationCenter(input, data);
    assert.deepEqual(center, frame === "LPS+" ? [-12, 2, 11] : [12, -2, 11]);
    assert.notEqual(center, input.targetCentroidMm);
    assert.deepEqual(input.targetCentroidMm, [12, -2, 11]);
    assert.deepEqual(data.compartments[0].mask, beforeMask);
    assert.deepEqual(data.affine, beforeAffine);
  }
});
test("annotation centering without a recorded point uses source-grid affine, including oblique LPS", async () => {
  for (const frame of ["RAS+", "LPS+"]) {
    const input = fixture();
    input.frame = frame;
    input.affine = [
      [0, -3, 0, 10],
      [2, 0, 0, -5],
      [0, 0, 4, 7],
      [0, 0, 0, 1],
    ];
    const data = await hydrateCase(input, api());
    assert.deepEqual(
      annotationCenter(input, data),
      frame === "LPS+" ? [-7, 3, 11] : [7, -3, 11],
    );
  }
});
test("annotation centering is unavailable for absent, empty or stale-case annotations", async () => {
  const input = fixture();
  const data = await hydrateCase(input, api());
  assert.equal(annotationCenter(null, data), null);
  assert.equal(annotationCenter(input, null), null);
  assert.equal(
    annotationCenter({ ...input, caseHash: `sha256:${"b".repeat(64)}` }, data),
    null,
  );
  input.compartments[0].volumeMm3 = 0;
  input.targetCentroidMm = [12, -2, 11];
  const empty = await hydrateCase(input, api({ mask: new Uint8Array(8) }));
  assert.equal(annotationCenter(input, empty), null);
  input.compartments = [];
  const absent = await hydrateCase(input, api());
  assert.equal(annotationCenter(input, absent), null);
});
test("generated target centering is not diluted by a much larger brain extraction",async()=>{
  for(const frame of ['RAS+','LPS+']){
    const input=fixture();input.frame=frame;
    input.metadata.anatomy_layers={brain:{kind:'brain',provenance:'estimated'},enhancing_target:{kind:'tumor',provenance:'simulated'}};
    input.compartments.unshift({name:'brain',volumeMm3:192,array:descriptor('uint8','brain')});
    const data=await hydrateCase(input,api({brain:new Uint8Array(8).fill(1)}));
    const before=data.compartments.map(layer=>layer.mask.slice());
    assert.deepEqual(annotationCenter(input,data),frame==='LPS+'?[-12,2,11]:[12,-2,11]);
    assert.deepEqual(data.compartments.map(layer=>layer.mask),before);
  }
});
test("MRI and mask encodings are rejected before transfer", async () => {
  const mutations = [
    (p) => (p.mri.dtype = "float64"),
    (p) => (p.compartments[0].array.dtype = "int8"),
    (p) => (p.mri.byteOrder = "big"),
    (p) => (p.mri.order = "F"),
    (p) => (p.mri.shape = [2, 4, 1]),
    (p) => (p.compartments[0].array.shape = [8]),
    (p) => (p.mri.byteLength = 64),
    (p) => (p.compartments[0].array.byteLength = 7),
  ];
  for (const mutate of mutations) {
    const p = fixture();
    mutate(p);
    let calls = 0;
    await assert.rejects(
      hydrateCase(p, {
        readAsset: async () => {
          calls++;
          return new Uint8Array();
        },
      }),
    );
    assert.equal(calls, 0);
  }
});
test("invalid identities, grids and frames fail closed", () => {
  const mutations = [
    (p) => (p.caseHash = "unknown"),
    (p) => (p.mri.sha256 = "bad"),
    (p) => (p.frame = "RAS"),
    (p) => (p.frame = "LPS-danger"),
    (p) => (p.shape = [2, 0, 2]),
    (p) => (p.shape = [2, 2, 2.5]),
    (p) => (p.shape = [4096, 4096, 4096]),
    (p) => (p.affine[0][0] = NaN),
    (p) => (p.affine[3][0] = 1),
    (p) => (p.affine[2] = [0, 0, 0, 0]),
    (p) => (p.spacingMm[0] = 1),
  ];
  for (const mutate of mutations) {
    const p = fixture();
    mutate(p);
    assert.throws(() => validateCaseDescriptor(p));
  }
});
test("descriptor-only source masks and brain mask must match source grid", () => {
  const p = fixture();
  p.compartments[0].sourceArray = {
    ...descriptor("uint8", "source"),
    shape: [1, 2, 4],
  };
  assert.throws(() => validateCaseDescriptor(p));
  delete p.compartments[0].sourceArray;
  p.brainMask = descriptor("float32", "brain");
  assert.throws(() => validateCaseDescriptor(p));
});
test("truncated buffers and non-finite voxels fail during hydration", async () => {
  await assert.rejects(
    hydrateCase(fixture(), api({ mri: new Uint8Array(31) })),
    /incomplete/,
  );
  await assert.rejects(
    hydrateCase(fixture(), api({ mask: new Uint8Array(7) })),
    /Incomplete/,
  );
  await assert.rejects(
    hydrateCase(
      fixture(),
      api({
        mri: new Uint8Array(
          new Float32Array([NaN, 0, 0, 0, 0, 0, 0, 0]).buffer,
        ),
      }),
    ),
    /non-finite/,
  );
});
test("binary source mask and displayed physical volume reconcile", async () => {
  await assert.rejects(
    hydrateCase(
      fixture(),
      api({ mask: new Uint8Array([0, 0, 0, 0, 0, 0, 0, 2]) }),
    ),
    /binary/,
  );
  const p = fixture();
  p.compartments[0].volumeMm3 = 1e9;
  await assert.rejects(hydrateCase(p, api()), /volume disagrees/);
});
test("semantic colors do not depend on compartment order", () => {
  assert.equal(anatomyColor("enhancing_target"), "#e5a269");
  assert.equal(anatomyColor("nonenhancing_or_necrotic_core"), "#b894d9");
  assert.equal(anatomyColor("FLAIR_abnormality"), "#55bace");
});

test('segmentation-only import retains zero source intensities with a usable display window',async()=>{
 const input=fixture();input.metadata={generated_signal:true};input.intensityRange=[0,0];
 const image=new Uint8Array(new Float32Array(8).buffer);
 const data=await hydrateCase(input,api({mri:image}));
 assert.deepEqual([...data.mri],Array(8).fill(0));assert.deepEqual(data.intensityRange,[-.5,.5]);
 assert.deepEqual(input.intensityRange,[0,0]);
 input.metadata={};assert.throws(()=>validateCaseDescriptor(input),/intensity range/);
});

test("planned inspection hydrates sampled material and rejects a mismatched target or source declaration", async () => {
  const input = fixture();
  input.compartments[0].name = 'tumor';
  const marker = {schema:'resectionlab.planned-derivative-inspection.v1', status:'inspection-only',
    patientTaskAdmission:false, sourceSnapshotId:hash, targetBinding:hash, planRevision:hash, approachId:'route-A'};
  const route = {schema:'resectionlab.native-environment-approach.v1', id:'route-A', targetLayerId:'lesion',
    sourceSnapshotId:hash, targetBinding:hash, planRevision:hash, parameterHash:hash, measurementHash:hash};
  input.metadata = {anatomy_workbench_only:true, anatomy_layers:{tumor:{kind:'tumor'}},
    surgical_properties:{cranialPlanningCandidateV1:marker}, surgical_adapter:{
      type:'native-patient-research-derivative', sourceCollectionHash:hash, targetAffine:input.affine,
      selectedApproach:route, sourceLayers:{tumor:{layerId:'lesion',sampledMaskHash:'sha256:'+hash}}}};
  const volume = await hydrateCase(input, api());
  assert.equal(volume.compartments[0].kind, 'tumor');
  route.targetLayerId = 'other-lesion';
  await assert.rejects(hydrateCase(input, api()), /planned derivative inspection binding/);
  route.targetLayerId = 'lesion';
  marker.schema = 'resectionlab.cranial-planning-candidate.v1';
  await assert.rejects(hydrateCase(input, api()), /Invalid source-owned inspection declaration/);
  delete input.metadata.surgical_properties.cranialPlanningCandidateV1;
  await assert.rejects(hydrateCase(input, api()), /Missing planned derivative inspection binding/);
});

import type {
  ArrayDescriptor,
  CasePayload,
  ResectionApi,
  Vec3,
  ViewerCase,
} from "./types";
import {registerSourceInspection} from './modeled-volume.ts';
import type {SurfaceAppearanceSource} from './viewer/contracts';
import {surfaceKind} from './viewer/surfacePolicy.ts';
import {isStructuralMrModality} from './native-study-state.ts';

function validateVesselSurface(surface:NonNullable<CasePayload['compartments'][number]['vesselSurface']>,shape:readonly number[]) {
  const {positions,normals,indices,sourceCells,scoreLighting,vesselTags,vesselClasses}=surface;
  for(const [name,descriptor,width,dtype] of [['positions',positions,3,'float32'],['normals',normals,3,'float32'],
    ['indices',indices,3,'uint32'],['sourceCells',sourceCells,1,'uint32'],['scoreLighting',scoreLighting,4,'float32']] as const){
    if(name==='scoreLighting'&&descriptor===undefined)continue;
    if(!descriptor||descriptor.dtype!==dtype||descriptor.byteOrder!=='little'||descriptor.order!=='C'||
      !descriptor.assetId||!/^[a-f0-9]{64}$/.test(descriptor.sha256)||descriptor.shape.length!==(width===1?1:2)||
      !Number.isSafeInteger(descriptor.shape[0])||descriptor.shape[0]<=0||(width>1&&descriptor.shape[1]!==width)||
      descriptor.byteLength!==descriptor.shape[0]*width*4||descriptor.byteLength>128*1024**2)throw Error(`Invalid vessel surface ${name}.`);
  }
  if(positions.shape[0]!==normals.shape[0]||positions.shape[0]!==sourceCells.shape[0])throw Error('Vessel surface source cells disagree with its vertices.');
  if(scoreLighting&&scoreLighting.shape[0]!==positions.shape[0])throw Error('Vessel score lighting disagrees with its vertices.');
  const voxels=shape.reduce((a,b)=>a*b,1);
  if(vesselTags&&(vesselTags.dtype!=='float32'||vesselTags.byteOrder!=='little'||vesselTags.order!=='C'||!vesselTags.assetId||
    !/^[a-f0-9]{64}$/.test(vesselTags.sha256)||vesselTags.shape.length!==3||vesselTags.shape.some((n,i)=>n!==shape[i])||
    vesselTags.byteLength!==voxels*4))throw Error('Invalid vessel tags.');
  if(vesselClasses&&(vesselClasses.dtype!=='uint8'||vesselClasses.byteOrder!=='little'||vesselClasses.order!=='C'||!vesselClasses.assetId||
    !/^[a-f0-9]{64}$/.test(vesselClasses.sha256)||vesselClasses.shape.length!==1||vesselClasses.shape[0]!==positions.shape[0]||
    vesselClasses.byteLength!==positions.shape[0]))throw Error('Invalid vessel classes.');
}

async function hydrateVesselSurface(raw:NonNullable<CasePayload['compartments'][number]['vesselSurface']>,api:ResectionApi){
  async function read(d:ArrayDescriptor){
    const bytes=await api.readAsset(d.assetId);if(bytes.byteLength!==d.byteLength)throw Error('Incomplete vessel surface transfer.');
    return bytes.slice().buffer;
  }
  const [p,n,i,c,l,t,k]=await Promise.all([read(raw.positions),read(raw.normals),read(raw.indices),read(raw.sourceCells),raw.scoreLighting&&read(raw.scoreLighting),
    raw.vesselTags&&read(raw.vesselTags),raw.vesselClasses&&read(raw.vesselClasses)]);
  const positions=new Float32Array(p),normals=new Float32Array(n),indices=new Uint32Array(i),vesselSourceCells=new Uint32Array(c);
  if(indices.some(i=>i>=vesselSourceCells.length)||!positions.every(Number.isFinite)||!normals.every(Number.isFinite))throw Error('Invalid vessel surface values.');
  const vesselScoreLighting=l&&new Float32Array(l);
  if(vesselScoreLighting&&!vesselScoreLighting.every((value,index)=>Number.isFinite(value)&&(index%4!==3||value>=0&&value<=1)))
    throw Error('Invalid vessel score lighting values.');
  const vesselTags=t?new Float32Array(t):undefined;
  if(vesselTags&&!vesselTags.every(value=>value>=0&&value<=1))throw Error('Invalid vessel tag values.');
  const vesselClasses=k?new Uint8Array(k):undefined;
  if(vesselClasses&&!vesselClasses.every(value=>value<=1))throw Error('Invalid vessel class values.');
  const vesselGeometryBinding=[raw.positions.sha256,raw.indices.sha256,raw.sourceCells.sha256].join(':');
  return {surface:{source:positions,positions,sourceNormals:normals,normals,indices},vesselSourceCells,vesselScoreLighting,vesselGeometryBinding,
    ...(vesselTags?{vesselTags}:{}),...(vesselClasses?{vesselClasses}:{}),
    surfaceDiagnostics:{policy:'native-source-isosurface-v1',kind:'vessel',maxDisplacementMm:0,restoredComponents:0,
      intersectionCheck:'passed',surfaceField:'native-probability',triangles:indices.length/3,sourceSurfaceOnly:true} as const};
}

export function anatomyColor(name: string): string {
  const key = name.toLowerCase();
  if (key.includes("flair")) return "#55bace";
  if (
    key.includes("nonenhanc") ||
    key.includes("necrot") ||
    key.includes("core")
  )
    return "#b894d9";
  if (key.includes("enhanc")) return "#e5a269";
  return ["#91bdad", "#87aee0", "#c2af80"][
    [...key].reduce((sum, c) => sum + c.charCodeAt(0), 0) % 3
  ];
}

export function readableName(name: string): string {
  return name.replace(/_/g, " ").replace(/\bFLAIR\b/i, "FLAIR");
}

export function rasPoint(point: Vec3, frame: string): Vec3 {
  return frame.startsWith("LPS")
    ? [-point[0], -point[1], point[2]]
    : [...point];
}

/** Prefer visible tumor annotations to broad support anatomy when navigating. */
export function annotationCenter(
  payload: CasePayload | null,
  caseData: ViewerCase | null,
): Vec3 | null {
  if (
    !payload ||
    !caseData ||
    payload.caseHash !== caseData.caseHash ||
    (payload.frame !== "RAS+" && payload.frame !== "LPS+") ||
    !caseData.compartments.some((layer) => layer.volumeMm3 > 0)
  )
    return null;
  if(payload.targetCentroidMm)return rasPoint(payload.targetCentroidMm,payload.frame);
  const anatomy=payload.metadata.anatomy_layers as Record<string,{kind?:string}>|undefined;
  const targets=caseData.compartments.filter(layer=>anatomy?.[layer.name]?.kind==='tumor'&&layer.volumeMm3>0);
  return initialCursor(targets.length?{...caseData,compartments:targets}:caseData);
}

/** Validate the physical/data contract before fetching or allocating any renderer arrays. */
export function validateCaseDescriptor(payload: CasePayload): {
  voxelCount: number;
  voxelVolumeMm3: number;
} {
  if (!payload || typeof payload.caseId !== "string" || !payload.caseId.trim())
    throw new Error("Missing case identity.");
  if (!/^sha256:[a-f0-9]{64}$/.test(payload.caseHash))
    throw new Error("Missing or invalid case fingerprint.");
  if (payload.frame !== "RAS+" && payload.frame !== "LPS+")
    throw new Error("Unsupported physical coordinate frame.");
  if (
    !Array.isArray(payload.shape) ||
    payload.shape.length !== 3 ||
    payload.shape.some(
      (size) => !Number.isSafeInteger(size) || size < 1 || size > 4096,
    )
  )
    throw new Error("Invalid MRI grid shape.");
  const count = payload.shape.reduce((a, b) => a * b, 1);
  if (!Number.isSafeInteger(count) || count * 4 > 1024 ** 3)
    throw new Error("MRI exceeds the local viewer size limit.");
  const matrix = payload.affine;
  if (
    !Array.isArray(matrix) ||
    matrix.length !== 4 ||
    matrix.some(
      (row) =>
        !Array.isArray(row) ||
        row.length !== 4 ||
        row.some((value) => !Number.isFinite(value)),
    )
  )
    throw new Error("Invalid physical affine.");
  if (
    matrix[3].some(
      (value, index) => Math.abs(value - (index === 3 ? 1 : 0)) > 1e-8,
    )
  )
    throw new Error("The affine is not a homogeneous physical transform.");
  const [a, b, c] = matrix;
  const determinant =
    a[0] * (b[1] * c[2] - b[2] * c[1]) -
    a[1] * (b[0] * c[2] - b[2] * c[0]) +
    a[2] * (b[0] * c[1] - b[1] * c[0]);
  const spacing = [0, 1, 2].map((axis) =>
    Math.hypot(a[axis], b[axis], c[axis]),
  );
  if (
    !Number.isFinite(determinant) ||
    Math.abs(determinant) <= 1e-10 * spacing.reduce((x, y) => x * y, 1)
  )
    throw new Error("The physical affine is singular.");
  if (
    !Array.isArray(payload.spacingMm) ||
    payload.spacingMm.length !== 3 ||
    payload.spacingMm.some(
      (value, index) =>
        !Number.isFinite(value) ||
        value <= 0 ||
        Math.abs(value - spacing[index]) > 1e-5 * Math.max(1, value),
    )
  )
    throw new Error("Voxel spacing disagrees with the physical affine.");
  const validateArray = (
    array: ArrayDescriptor,
    dtype: string,
    name: string,
  ) => {
    if (
      !array ||
      array.dtype !== dtype ||
      array.byteOrder !== "little" ||
      array.order !== "C"
    )
      throw new Error(`${name} has an unsupported array encoding.`);
    if (
      !Array.isArray(array.shape) ||
      array.shape.length !== 3 ||
      array.shape.some((value, index) => value !== payload.shape[index])
    )
      throw new Error(`${name} does not match the MRI grid.`);
    if (array.byteLength !== count * (dtype === "float32" ? 4 : 1))
      throw new Error(`${name} declares an invalid byte length.`);
    if (
      typeof array.assetId !== "string" ||
      !array.assetId ||
      !/^[a-f0-9]{64}$/.test(array.sha256)
    )
      throw new Error(`${name} has an invalid transfer identity.`);
  };
  validateArray(payload.mri, "float32", "MRI");
  if (!Array.isArray(payload.compartments) || payload.compartments.length > 80)
    throw new Error("Invalid target layer inventory.");
  const names = new Set<string>();
  payload.compartments.forEach((layer) => {
    if (
      !layer ||
      typeof layer.name !== "string" ||
      !layer.name ||
      names.has(layer.name) ||
      !Number.isFinite(layer.volumeMm3) ||
      layer.volumeMm3 < 0
    )
      throw new Error("Invalid target layer metadata.");
    names.add(layer.name);
    validateArray(layer.array, "uint8", layer.name);
    if (layer.sourceArray)
      validateArray(layer.sourceArray, "uint8", `${layer.name} source`);
    if(layer.vesselSurface)validateVesselSurface(layer.vesselSurface,payload.shape);
  });
  if (payload.brainMask)
    validateArray(payload.brainMask, "uint8", "Brain mask");
  if (
    !Array.isArray(payload.unknowns) ||
    payload.unknowns.some((item) => typeof item !== "string") ||
    !payload.metadata ||
    typeof payload.metadata !== "object"
  )
    throw new Error("Incomplete case evidence record.");
  if (
    payload.intensityRange &&
    (payload.intensityRange.length !== 2 ||
      payload.intensityRange.some((v) => !Number.isFinite(v)) ||
      payload.intensityRange[0] > payload.intensityRange[1] ||
      (payload.intensityRange[0] === payload.intensityRange[1] && payload.metadata.generated_signal !== true))
  )
    throw new Error("Invalid MRI intensity range.");
  if (
    payload.targetCentroidMm &&
    (payload.targetCentroidMm.length !== 3 ||
      payload.targetCentroidMm.some((v) => !Number.isFinite(v)))
  )
    throw new Error("Invalid physical target cursor.");
  return { voxelCount: count, voxelVolumeMm3: Math.abs(determinant) };
}

/** Brain extraction is a display layer, never a target compartment in the engine. */
export function displayBrainMask(payload:CasePayload):boolean {
  const anatomy=payload.metadata.anatomy_layers as Record<string,unknown>|undefined;
  return !!payload.brainMask && !!anatomy?.brain && !payload.compartments.some(layer=>layer.name==='brain');
}

const objectRecord = (value: unknown): value is Record<string, unknown> =>
  !!value && typeof value === "object" && !Array.isArray(value);
const sourceHash = (value: unknown): value is string =>
  typeof value === "string" && /^[a-f0-9]{64}$/.test(value);
const affineMatrix = (value: unknown): value is number[][] =>
  Array.isArray(value) && value.length === 4 && value.every(row =>
    Array.isArray(row) && row.length === 4 && row.every(Number.isFinite));
const sameSourceValue = (a: unknown, b: unknown) => JSON.stringify(a) === JSON.stringify(b);

/** Match the engine's local alignment receipt to the selected source grids.
 * The composed transform positions the native wall crop in reference RAS. */
function reviewedCropPatientAffine(adapter: Record<string, unknown>, manifest: Record<string, unknown>,
  reference: Record<string, unknown>, vascular: Record<string, unknown>, wall: Record<string, unknown>,
  brainLayerId: unknown, shape: number[]): number[][] | undefined {
  const review = adapter.nativeWallAlignmentReview;
  if (!objectRecord(review) || review.schema !== "resectionlab.native-wall-alignment-review.v1" ||
      !sourceHash(review.digest) || !sameSourceValue(review.cropStartIJK, wall.sourceStartIJK) ||
      !sameSourceValue(review.cropShape, shape) || !sameSourceValue(review.centerRasMm, wall.centerRasMm) ||
      !objectRecord(review.binding) || !objectRecord(manifest.transforms) ||
      typeof reference.id !== "string" || typeof vascular.id !== "string") return;
  const {binding} = review;
  const transform = manifest.transforms[vascular.id];
  const referenceTransform = manifest.transforms[reference.id];
  if (!objectRecord(transform) || transform.status !== "unreviewed" ||
      !sameSourceValue(transform.native_wall_alignment_review, review) ||
      !objectRecord(referenceTransform) || referenceTransform.status !== "reference" ||
      binding.sourcePatient !== manifest.patient_id || binding.referenceSeries !== reference.id ||
      binding.vascularSeries !== vascular.id || binding.brainLayerId !== brainLayerId ||
      binding.vesselLayerId !== wall.layerId || !objectRecord(binding.sources) ||
      !affineMatrix(binding.matrix) || !sameSourceValue(binding.matrix, transform.matrix) ||
      !affineMatrix(vascular.affine)) return;
  const sources = binding.sources;
  const selected = [[reference.id, reference, transform.reference_sha256], [vascular.id, vascular, transform.source_sha256]] as const;
  if (selected.some(([id, series, sha]) => {
    const bound = sources[id];
    return !objectRecord(bound) || !objectRecord(series.original) || sha !== series.original.sha256 ||
      bound.originalSha256 !== sha || bound.timepoint !== series.timepoint ||
      !sameSourceValue(bound.nativeAffine, series.affine) || !sameSourceValue(bound.nativeArray, series.data);
  })) return;
  const nativeAffine = vascular.affine;
  return binding.matrix.map(row => nativeAffine.map((_, column) =>
    row.reduce((value, coefficient, index) => value + coefficient * nativeAffine[index][column], 0)));
}

/** Use the acquired reference signal already sampled into the accepted crop. */
function localWallBrainSignal(payload: CasePayload, mri: Float32Array, affine: ViewerCase["affine"],
  mask: Uint8Array): SurfaceAppearanceSource | undefined {
  const adapter = payload.metadata.surgical_adapter;
  if (payload.frame !== "RAS+" || !objectRecord(adapter) || adapter.type !== "native-patient-research-derivative") return;
  const {nativeWall: wall, sourceManifest: manifest, sourceLayers} = adapter;
  if (!objectRecord(wall) || wall.schema !== "resectionlab.patient-native-wall.v1" ||
      !objectRecord(manifest) || !Array.isArray(manifest.series) || !Array.isArray(manifest.layers) ||
      !objectRecord(sourceLayers) || !objectRecord(sourceLayers.brain)) return;
  const brain = sourceLayers.brain;
  const {sourceSeries, sourceShape, sourceStartIJK, sourceImageSha256} = wall;
  const referenceId = manifest.reference_series;
  if (typeof sourceSeries !== "string" || typeof referenceId !== "string" || !sourceHash(sourceImageSha256) ||
      brain.sampledCoverageCells !== mri.length || !manifest.layers.some(row => objectRecord(row) &&
        row.id === brain.layerId && row.kind === "brain" && row.source_series === referenceId)) return;
  const observed = (id: string, sha: string) => payload.sourceRefs?.some(ref =>
    ref.source_id === id && ref.provenance === "observed" && ref.sha256 === sha);
  const reference = manifest.series.find(row => objectRecord(row) && row.id === referenceId);
  const vascular = manifest.series.find(row => objectRecord(row) && row.id === sourceSeries);
  if (!objectRecord(reference) || !objectRecord(reference.original) || !sourceHash(reference.original.sha256) ||
      typeof reference.modality !== 'string' ||
      !(reference.modality === 'CT' || reference.modality === 'CTA' ||
        (isStructuralMrModality(reference.modality) && !/\b(?:SWI|DWI|DTI|ADC|diffusion|susceptibility)\b/i.test(reference.modality))) ||
      !observed(referenceId, reference.original.sha256) ||
      !objectRecord(vascular) || !objectRecord(vascular.original) || vascular.original.sha256 !== sourceImageSha256 ||
      !observed(sourceSeries, sourceImageSha256) || !objectRecord(vascular.data) || !affineMatrix(vascular.affine) ||
      !Array.isArray(sourceShape) || sourceShape.length !== 3 || !Array.isArray(sourceStartIJK) || sourceStartIJK.length !== 3 ||
      !sameSourceValue(vascular.data.shape, sourceShape) || sourceShape.some((size, axis) =>
        !Number.isSafeInteger(size) || !Number.isSafeInteger(sourceStartIJK[axis]) || sourceStartIJK[axis] < 0 ||
        sourceStartIJK[axis] + payload.shape[axis] > size)) return;
  const sourceAffine = sourceSeries === referenceId ? vascular.affine :
    reviewedCropPatientAffine(adapter, manifest, reference, vascular, wall, brain.layerId, payload.shape);
  if (!sourceAffine || affine.some((row, axis) => {
    const native = sourceAffine[axis];
    return row.some((value, column) => Math.abs(value - (column < 3 ? native[column] :
        native[3] + sourceStartIJK.reduce((delta, offset, index) => delta + offset * native[index], 0))) > 1e-6);
  })) return;
  const surface = payload.compartments.find(layer => layer.name === "vessels")?.vesselSurface;
  if (!surface || [surface.positions.sha256, surface.indices.sha256, surface.sourceCells.sha256].join(":") !== wall.geometryBinding) return;

  // Window the acquired signal to the actual sampled brain distribution.
  // The window only controls the albedo of this checked surface.
  const values: number[] = [];
  const stride = Math.max(1, Math.ceil(mri.length / 16000));
  for (let i = 0; i < mri.length; i += stride) if (mask[i]) values.push(mri[i]);
  values.sort((a, b) => a - b);
  if (values.length < 64) return;
  const low = values[Math.floor(values.length * .15)], high = values[Math.floor(values.length * .9)];
  if (!(high > low)) return;
  return {seriesId: referenceId, modality: reference.modality, sourceSha256: reference.original.sha256,
    signalSha256: payload.mri.sha256, shape: payload.shape, affine, data: mri,
    softTissueWindow: [low, high], nativeWallCrop: true};
}

/** The bound engine archive carries reference pixels on its own physical grid.
 * A synthetic lesion can share an acquired background. Its contour is never
 * evidence of an acquired tumor signal. */
function recordedBrainSignal(payload: CasePayload, mri: Float32Array, affine: ViewerCase["affine"]): SurfaceAppearanceSource | undefined {
  const receipt = payload.verifiedStudyReplay ?? payload.artifacts?.nativeStudyReplay;
  if (!objectRecord(receipt) || receipt.caseHash !== payload.caseHash) return;
  const toolbox = payload.artifacts?.surgical_toolbox;
  const exact = objectRecord(toolbox) ? toolbox.exactGridReplay : undefined;
  const adapter = payload.metadata.surgical_adapter;
  const manifest = objectRecord(exact) ? exact.sourceManifest : objectRecord(adapter) ? adapter.sourceManifest : undefined;
  if (!objectRecord(manifest) || !Array.isArray(manifest.series) ||
      manifest.patient_id !== receipt.patientId || manifest.snapshot_id !== receipt.sourceSnapshotId) return;
  const series = manifest.series.find(row => objectRecord(row) && row.id === manifest.reference_series);
  if (!objectRecord(series) || !objectRecord(series.original) || typeof series.id !== "string" ||
      typeof series.modality !== "string" || !["T1", "T1CE", "T2", "FLAIR"].includes(series.modality)) return;
  const sha = series.original.sha256;
  if (typeof sha !== "string" || !/^[a-f0-9]{64}$/.test(sha) ||
      !payload.sourceRefs?.some(source => source.provenance === "observed" && source.sha256 === sha)) return;
  return {seriesId: series.id, modality: series.modality, sourceSha256: sha, signalSha256: payload.mri.sha256,
    shape: payload.shape, affine, data: mri};
}

/** A planned derivative is an ordinary operable scene (owner decision
 * 2026-11-27). Its selected route is provenance and must bind the source; a
 * legacy saved derivative also carries the retired marker, whose copy of that
 * binding must agree. Neither grants nor withholds any action. */
function validatePlannedDerivative(payload: CasePayload, marker: Record<string, unknown> | undefined) {
  const adapter = payload.metadata.surgical_adapter;
  const route = objectRecord(adapter) && adapter.selectedApproach;
  const layers = objectRecord(adapter) && adapter.sourceLayers;
  const digest = (value: unknown) => typeof value === "string" && /^[0-9a-f]{64}$/.test(value);
  if (!objectRecord(adapter) || adapter.type !== "native-patient-research-derivative" ||
      !objectRecord(route) || route.schema !== "resectionlab.native-environment-approach.v1" ||
      !digest(route.sourceSnapshotId) || route.sourceSnapshotId !== adapter.sourceCollectionHash ||
      !digest(route.targetBinding) || !digest(route.planRevision) ||
      typeof route.id !== "string" || !route.id ||
      (marker !== undefined && (marker.sourceSnapshotId !== route.sourceSnapshotId ||
        marker.targetBinding !== route.targetBinding || marker.planRevision !== route.planRevision ||
        marker.approachId !== route.id)) ||
      !digest(route.parameterHash) || !digest(route.measurementHash) ||
      JSON.stringify(adapter.targetAffine) !== JSON.stringify(payload.affine) || !objectRecord(layers) ||
      !objectRecord(layers.tumor) || typeof route.targetLayerId !== "string" || route.targetLayerId !== layers.tumor.layerId ||
      !payload.compartments.some(layer => layer.name === "tumor") ||
      Object.keys(layers).length !== payload.compartments.length ||
      payload.compartments.some(layer => {
        const row = layers[layer.name];
        return !objectRecord(row) || typeof row.sampledMaskHash !== "string" || !/^sha256:[0-9a-f]{64}$/.test(row.sampledMaskHash);
      })) throw Error("Invalid planned derivative route binding");
}

export async function hydrateCase(
  payload: CasePayload,
  api: ResectionApi,
): Promise<ViewerCase> {
  const { voxelVolumeMm3 } = validateCaseDescriptor(payload);
  const layers = [...(displayBrainMask(payload)?[{name:'brain',array:payload.brainMask!,volumeMm3:null}]:[]),
    ...payload.compartments.map(layer=>({...layer,volumeMm3:layer.volumeMm3 as number|null}))];
  const [mriBytes, maskBytes, vesselSurfaces] = await Promise.all([
    api.readAsset(payload.mri.assetId),
    Promise.all(layers.map((layer) => api.readAsset(layer.array.assetId))),
    Promise.all(layers.map(layer=>'vesselSurface' in layer&&layer.vesselSurface?hydrateVesselSurface(layer.vesselSurface,api):undefined)),
  ]);
  if (mriBytes.byteLength !== payload.mri.byteLength)
    throw new Error("MRI transfer is incomplete.");
  const mri = new Float32Array(mriBytes.slice().buffer);
  // Indexed typed-array loops avoid per-voxel iterator/callback overhead on
  // large source grids. The same finite-value and binary-mask gates remain.
  for (let index = 0; index < mri.length; index++) {
    if (!Number.isFinite(mri[index]))
      throw new Error("MRI contains non-finite voxel values.");
  }
  const affine = payload.affine.map((row) => [...row]);
  if (payload.frame === "LPS+") {
    affine[0] = affine[0].map((value) => -value);
    affine[1] = affine[1].map((value) => -value);
  }
  const brainSignal = recordedBrainSignal(payload, mri, affine);
  const result: ViewerCase = {
    caseId: payload.caseId,
    caseHash: payload.caseHash,
    frame: "RAS+",
    affine,
    shape: [...payload.shape],
    spacingMm: [...payload.spacingMm],
    intensityRange: payload.intensityRange?.[0] === payload.intensityRange?.[1] && payload.intensityRange
      ? [payload.intensityRange[0] - .5, payload.intensityRange[1] + .5] : payload.intensityRange,
    mri,
    compartments: layers.map((layer, index) => {
      const mask = maskBytes[index];
      if (mask.byteLength !== layer.array.byteLength)
        throw new Error(`Incomplete ${layer.name} transfer.`);
      let selected = 0;
      for (let index = 0; index < mask.length; index++) {
        const value = mask[index];
        if (value !== 0 && value !== 1)
          throw new Error(`${layer.name} is not a binary mask.`);
        selected += value;
      }
      if (
        layer.volumeMm3!==null && Math.abs(selected * voxelVolumeMm3 - layer.volumeMm3) >
        1e-5 * Math.max(1, layer.volumeMm3)
      )
        throw new Error(`${layer.name} volume disagrees with its source grid.`);
      const declared = (payload.metadata.anatomy_layers as Record<string,{kind?:string;provenance?:unknown}>|undefined)?.[layer.name];
      const kind = declared?.kind;
      const provenance = typeof declared?.provenance === "string" ? declared.provenance : undefined;
      const vessel=vesselSurfaces[index];
      const appearanceSource = kind === "brain" ? localWallBrainSignal(payload, mri, affine, mask) ?? brainSignal : undefined;
      if(vessel&&(surfaceKind(kind)!=='vessel'||vessel.vesselSourceCells.some(i=>!mask[i])))throw Error('Vessel surface differs from the declared native mask.');
      return {
        name: layer.name,
        kind,
        ...(provenance ? {provenance} : {}),
        ...(appearanceSource ? {appearanceSource, displayKind: kind} : {}),
        sourceIdentity: `${payload.caseHash}:${layer.name}:${layer.array.sha256}`,
        qa: layer.qa,
        ...vessel,
        volumeMm3: layer.volumeMm3 ?? selected * voxelVolumeMm3,
        mask,
        color: String((payload.metadata.anatomy_layers as Record<string,{color?:string}>|undefined)?.[layer.name]?.color ?? anatomyColor(layer.name)),
      };
    }),
  };
  const properties=payload.metadata.surgical_properties;
  const adapter = payload.metadata.surgical_adapter;
  const marker = objectRecord(properties) && Object.prototype.hasOwnProperty.call(properties, 'cranialPlanningCandidateV1')
    ? properties.cranialPlanningCandidateV1 : undefined;
  const plannedMarker = objectRecord(marker) && marker.schema === 'resectionlab.planned-derivative-inspection.v1';
  if (objectRecord(adapter) && adapter.type === 'native-patient-research-derivative' && 'selectedApproach' in adapter) {
    validatePlannedDerivative(payload, plannedMarker ? marker : undefined);
  } else if (plannedMarker) {
    throw Error('Invalid planned derivative route binding');
  }
  if (marker !== undefined && !plannedMarker) {
    registerSourceInspection(result, marker, payload.metadata.source_inspection_annotations);
  }
  return result;
}

export function initialCursor(caseData: ViewerCase): Vec3 {
  const [sx, sy, sz] = caseData.shape;
  let count = 0,
    x = 0,
    y = 0,
    z = 0;
  for (const layer of caseData.compartments) {
    for (let index = 0; index < layer.mask.length; index++) {
      if (!layer.mask[index]) continue;
      count++;
      x += Math.floor(index / (sy * sz));
      y += Math.floor(index / sz) % sy;
      z += index % sz;
    }
  }
  const point = count
    ? [x / count, y / count, z / count]
    : [(sx - 1) / 2, (sy - 1) / 2, (sz - 1) / 2];
  return caseData.affine
    .slice(0, 3)
    .map(
      (row) =>
        row[0] * point[0] + row[1] * point[1] + row[2] * point[2] + row[3],
    ) as Vec3;
}

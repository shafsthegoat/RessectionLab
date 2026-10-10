import type {ContactFamilyEpisode} from './contact-family-types.ts';
import type {ViewerCase, Vec3} from './types.ts';
import {arrayDigest, sha256Bytes} from './source-integrity.ts';
import {contactNeed, contactSame} from './public-contact-authority.ts';

/** Rebuild only the declared public initial crop; no model or hidden state.
 * This fixed family exposes analytic structural support and zero compatibility
 * target/motor/language fields. Availability remains distinct from value zero. */
export async function checkedFamilyObservation(e: ContactFamilyEpisode, volume: ViewerCase, support: Uint8Array) {
  const b = e.initialObservationBinding, context = e.taskContract.detachedObservationBinding;
  const shape: Vec3 = [...e.shape], count = shape.reduce((a, n) => a * n, 1);
  const names = ['structural_intensity', 'nominal_tissue', 'nominal_target', 'observed_cavity',
    'nominal_motor', 'nominal_language', 'public_goal', 'committed_probe_contact'];
  const available = [true, true, true, true, false, false, true, true];
  contactNeed(b?.version === 'public-contact-initial-observation-binding-v1' &&
    b.sourceHash === e.sourceHash && b.decisionModelHash === e.decisionModelHash &&
    b.declarationHash === context.declaration_hash && b.objectiveHash === e.publicGoal.objectiveHash &&
    b.goalGridHash === e.publicGoal.goalGridHash && b.cropAffineHash === context.crop_affine_hash &&
    typeof b.observationHash === 'string' && /^sha256:[a-f0-9]{64}$/.test(b.observationHash) &&
    b.frame === 'RAS+' && b.physicalUnits === 'mm' &&
    contactSame(b.cropOriginNative, context.crop_origin_native) && contactSame(b.cropOriginNative, [0, 0, 0]) &&
    contactSame(b.cropShape, shape) && contactSame(b.cropAffine, e.affine) &&
    contactSame(b.channelNames, names) && contactSame(b.channelAvailable, available) &&
    b.channelValueHashes.length === 8 && b.channelCoverageHashes.length === 8 &&
    b.scope === 'raw_public_crop_values_before_policy_normalization_not_full_native_coverage' &&
    b.extraChannelCoverage === 'inherits_required_structural_intensity_coverage',
    'initial family observation/crop/availability changed.');
  const affineBytes = new Uint8Array(new Float64Array(b.cropAffine.flat()).buffer);
  const header = new TextEncoder().encode('{"dtype": "<f8", "shape": [4, 4]}');
  const framed = new Uint8Array(header.length + affineBytes.length);
  framed.set(header); framed.set(affineBytes, header.length);
  contactNeed('sha256:' + await sha256Bytes(framed) === b.cropAffineHash, 'actor crop affine differs from its hash.');
  contactNeed(support.length === count && support.every(value => value === 0 || value === 1) &&
    await arrayDigest(support, shape, '|b1') === e.sourceBinding.support_hash &&
    volume.compartments.length === 1 && volume.compartments[0].mask.every(value => value === 0),
    'family support or zero compatibility target changed.');
  const goal = e.publicGoal.nativeIndex, goalIndex = (goal[0] * shape[1] + goal[1]) * shape[2] + goal[2];
  contactNeed(support[goalIndex] === 1, 'public goal is outside observed source support.');
  const ras = e.affine.slice(0, 3).map(row => row.slice(0, 3).reduce((sum, n, i) => sum + n * goal[i], row[3]));
  contactNeed(contactSame(ras, e.publicGoal.rasMm), 'public goal native/RAS coordinates disagree.');
  const goalGrid = new Uint8Array(count); goalGrid[goalIndex] = 1;
  const zero = new Float32Array(count), zeroBool = new Uint8Array(count), covered = new Uint8Array(count).fill(1);
  const tissue = Float32Array.from(support);
  const values: Array<Float32Array | Uint8Array> = [volume.mri, tissue, zero, zero, zero, zero, goalGrid, zeroBool];
  const hashes = await Promise.all(values.map((value, i) => arrayDigest(
    new Uint8Array(value.buffer, value.byteOffset, value.byteLength), shape, i < 6 ? '<f4' : '|b1')));
  const fullHash = await arrayDigest(covered, shape, '|b1'), emptyHash = await arrayDigest(zeroBool, shape, '|b1');
  contactNeed(contactSame(hashes, b.channelValueHashes) && hashes[6] === e.publicGoal.goalGridHash &&
    contactSame(available.map(value => value ? fullHash : emptyHash), b.channelCoverageHashes),
    'public initial channel values or per-channel coverage differ from source.');
  if (e.learnedAuthorship) contactNeed((e.planning.observation_ids as string[])[0] === b.observationHash,
    'first actor forward used another public observation.');
  return {nativeShape: shape, cropShape: [...b.cropShape] as Vec3, channelAvailable: [...b.channelAvailable],
    coverageScope: 'declared initial public crop only' as const};
}

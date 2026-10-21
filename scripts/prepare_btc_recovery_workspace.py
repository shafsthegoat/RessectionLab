#!/usr/bin/env python3
"""Package recovered PAT22 sources and an actual cached brain estimate for review.

No network, inference, planning or training occurs here. The original scan-only
contract remains intact. This review workspace does not admit a surgical task.
"""
import argparse
import json
from pathlib import Path
import sys

import nibabel as nib
import numpy as np
from scipy.ndimage import label

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from prepare_btc_case import _verify_sources
from resectionlab.core import array_digest, thaw_json
from resectionlab.imaging import load_case, file_sha256, load_fractional_annotation_case
from resectionlab.structural_evidence import StructuralEvidence, structural_frame_hash
from resectionlab.workspace_bundle import OriginalFile, save_workspace, load_workspace, view_state

WEIGHT_HASH = '37417f802196186441aae3e7f385d94f8a98c64a88acaeaa2723af995c653e33'
CAP = 250 * 1024**2


def verify_source_arrays(source, image_path, annotation_path):
    """A metadata receipt alone cannot attest the prepared arrays' contents."""
    expected = load_fractional_annotation_case(image_path, annotation_path, threshold=.5,
        annotation_interpretation='Pinned source annotation; independent recovery-array check',
        compartment_name='reference_threshold_target',case_id='recovery-reference')
    if (not np.array_equal(source.mri, expected.mri) or
            not np.array_equal(source.affine, expected.affine)):
        raise ValueError('Prepared MRI/frame differs from independently verified original bytes')
    # Reuse the existing exact integer axis-reindex contract. PAT22's annotation
    # reverses the first voxel axis; requiring raw array equality would be wrong.
    # The importer rejects warps, coverage drift or fractional resampling.
    if (len(source.compartments) != 1 or
            not np.array_equal(next(iter(source.compartments.values())),
                               expected.compartments['reference_threshold_target'])):
        raise ValueError('Prepared tumor differs from the pinned native threshold annotation')


def resident_bytes(root):
    """Conservative logical/allocated bytes of unique resident files."""
    visited = set(); logical = allocated = 0
    for path in root.rglob('*'):
        if path.is_symlink() or not path.is_file(): continue
        stat = path.stat(); identity = stat.st_dev, stat.st_ino
        if identity in visited: continue
        visited.add(identity); logical += stat.st_size; allocated += stat.st_blocks * 512
    return max(logical, allocated)


def prepare(root, inference_path, output):
    root, inference_path, output = map(lambda p: Path(p).resolve(), (root, inference_path, output))
    if (not root.is_relative_to((ROOT/'testcases').resolve()) or
            not inference_path.is_relative_to(root) or not output.is_relative_to(root)):
        raise ValueError('Recovery inputs/outputs must remain in the approved testcases root')
    if output.exists(): raise ValueError('Preserve existing workspaces; choose a new output')
    manifest_path = ROOT/'artifacts/btc-spatial-acquisition-v1/sub-PAT22-source-manifest.json'
    manifest, paths = _verify_sources(manifest_path, root/'source')
    if manifest['subject'] != 'sub-PAT22' or manifest['development_role'] != 'population_training':
        raise ValueError('This bounded recovery admits only the prospectively assigned PAT22 TRAIN source')
    source = load_case(root/'source-case.ressectionlab')
    if (source.case_id != 'BTC-ds001226-sub-PAT22-preop' or
            source.metadata.get('acquisition_manifest_sha256') != file_sha256(manifest_path) or
            source.metadata.get('split',{}).get('development_role') != 'population_training' or
            source.metadata.get('target_policy_input_allowed') is not False):
        raise ValueError('Prepared source identity, prospective TRAIN role or old information boundary differs')
    record = json.loads(inference_path.read_text())
    image = paths['sub-PAT22/ses-preop/anat/sub-PAT22_ses-preop_T1w.nii.gz']
    verify_source_arrays(source, image, paths['derivatives/tumor_masks/sub-PAT22/anat/sub-PAT22_space_T1_label-tumor.nii'])
    if (record.get('status') != 'succeeded' or record.get('model_id') != 'synthstrip-main' or
            record.get('exit_code') != 0 or record.get('failure') is not None or
            record['input']['sha256'] != file_sha256(image) or
            record['input']['shape'] != list(source.mri.shape) or
            not np.allclose(record['input']['affine_ras_mm'], source.affine, atol=1e-5, rtol=0) or
            [row['sha256'] for row in record['weights']] != [WEIGHT_HASH]):
        raise ValueError('Inference source/model identity differs from the recovered patient')
    masks = record['outputs']
    if len(masks) != 1 or masks[0]['name'] != 'brain_mask.nii.gz':
        raise ValueError('Exactly one explicit saved brain estimate is required')
    mask_path = Path(masks[0]['path']).resolve()
    if not mask_path.is_relative_to(inference_path.parent) or file_sha256(mask_path) != masks[0]['sha256']:
        raise ValueError('Inference mask location or bytes changed')
    image_mask = nib.load(mask_path); values = np.asanyarray(image_mask.dataobj)
    if (values.shape != source.mri.shape or not np.allclose(image_mask.affine, source.affine, atol=1e-5, rtol=0) or
            not np.isin(values, [0,1]).all() or not values.any()):
        raise ValueError('Brain estimate must be nonempty binary data on the exact native source grid')
    brain = values.astype(bool)
    target_name, target = next(iter(source.compartments.items()))
    if np.any(target & ~brain):
        raise ValueError('Target extends outside extraction; do not repair by silently unioning the masks')
    _, components = label(brain)
    if components != 1: raise ValueError('Brain estimate has disconnected components; retain for explicit QC rather than silently repair')
    evidence = StructuralEvidence(evidence_id='pat22-synthstrip-main', mask=brain,
        source_image_hash=array_digest(source.mri), source_frame_hash=structural_frame_hash(source),
        source_file_sha256=file_sha256(image), model_sha256=WEIGHT_HASH,
        run_sha256=file_sha256(inference_path), method=record['model']['name']+' '+record['model']['version'],
        metadata={'inference':record, 'source_patient':'BTC:sub-PAT22',
                  'target_inclusion_fraction':1., 'components':int(components),
                  'scope':'Unreviewed brain estimate for anatomical inspection; no cortical or surgical admission'})
    metadata = thaw_json(source.metadata)
    # Preserve the old target_policy_input_allowed=False and scan-only benchmark
    # fields. An annotation-assisted surgical task needs a DIFFERENT admission.
    metadata.update(anatomy_workbench_only=True, patient_id='BTC:sub-PAT22', patient_name='BTC PAT22 · TRAIN source review',
        selected_modality='T1', missing_anatomy=['skull','vessels','ventricles','functional cortex and tracts'],
        missing_modalities=['T1CE','T2','FLAIR','CT','MRA','DTI'],
        segmentation_cache_policy='saved_masks_only; no inference on open',
        recovered_source_ancestry={'case_hash':source.semantic_hash,'manifest_sha256':file_sha256(manifest_path)},
        research_episode_admission={'status':'refused_pending_distinct_post_exposure_contract',
            'role':'TRAIN', 'patient_group':'BTC:sub-PAT22', 'original_scan_only_contract_unchanged':True,
            'reason':'No enforced observed-domain/exposure corridor or complete critical-structure coverage'},
        anatomy_layers={
            'brain':{'kind':'brain','label':'Brain · SynthStrip estimate','color':'#b9c5c9','provenance':'estimated',
                     'review_status':'review_required','model':record['model'], 'inference':record,
                     'coverage':'Brain boundary estimate; anatomy and tumor-distorted margin unreviewed'},
            target_name:{'kind':'tumor','label':'Tumor · supplied fractional annotation ≥0.5','color':'#9863db',
                'provenance':'derived from supplied source annotation', 'review_status':'local_review_pending',
                'coverage':'Source-derived threshold scenario; not a clinician-approved resection boundary',
                'threshold':.5,'source_annotation_hash':next(row['sha256'] for row in manifest['files']
                    if row['path'].endswith('label-tumor.nii'))}})
    case = source.revised(brain_mask=brain, structural_evidence={evidence.evidence_id:evidence},
        metadata=metadata, unknowns=source.unknowns+('brain_estimate_unreviewed','no_surgical_task_admission'))
    if array_digest(case.mri) != array_digest(source.mri) or array_digest(case.affine) != array_digest(source.affine):
        raise ValueError('Source image/frame changed while packaging')
    originals = [OriginalFile.from_path(manifest_path,role='metadata')]
    for name,path in paths.items():
        role = 'annotation' if name.endswith('label-tumor.nii') else 'imaging' if name.endswith('.nii.gz') else 'metadata'
        originals.append(OriginalFile.from_path(path,role=role))
    originals.extend([OriginalFile.from_path(mask_path,role='model-result'),OriginalFile.from_path(inference_path,role='metadata')])
    # Worst-case uncompressed arrays plus originals and a temporary staged ZIP;
    # reserve before writing instead of discovering a cap breach afterward.
    reserve = 2*(source.mri.nbytes + 5*brain.nbytes + sum(len(row.data) for row in originals) + 1024**2)
    before = resident_bytes(root)
    if before+reserve > CAP: raise ValueError('Conservative workspace write reservation exceeds approved250MiB cap')
    imaging = view_state(case,[])
    save_workspace(case, output, sources=[], imaging=imaging, episode=None, selection=None, artifacts={}, originals=originals)
    reopened, _, workspace = load_workspace(output)
    if reopened.semantic_hash != case.semantic_hash or len(workspace['originals']) != len(originals):
        raise ValueError('Native workspace identity/original-file persistence differs')
    if resident_bytes(root) > CAP: raise ValueError('Approved recovery growth cap exceeded')
    return {'schema':'resectionlab.pat22-recovery-review.v1','sourceCaseHash':source.semantic_hash,
        'workspaceCaseHash':case.semantic_hash,'workspace':str(output),'patientGroup':'BTC:sub-PAT22',
        'developmentRole':'population_training','role':'TRAIN','sourceFilesVerified':len(paths),
        'brainCells':int(brain.sum()),'brainComponents':int(components),'targetCells':int(target.sum()),
        'targetInclusionFraction':1.,'nativeAffinePreserved':True,'sourceArraysUnchanged':True,
        'originalFilesEmbedded':len(originals),'freshNativeReopen':True,'brainReview':'review_required',
        'planningAdmission':False,'oldScanOnlyTargetPolicyAllowed':metadata['target_policy_input_allowed'],
        'optimizerUpdates':0,'residentBytes':resident_bytes(root),'approvedCapBytes':CAP,
        'missingAnatomy':metadata['missing_anatomy'],'clinicalValidation':'none'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--inference',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    report=prepare(args.root,args.inference,args.output)
    path=args.output.with_suffix('.qc.json')
    with path.open('x') as handle: json.dump(report,handle,indent=2);handle.write('\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__': main()

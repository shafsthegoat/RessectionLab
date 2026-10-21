#!/usr/bin/env python3
"""Build one cached healthy IXI265 study and a canonical research simulator case.

All generated patient data live in ignored testcases/. Source arrays and original
headers are preserved. Registration is measured but never promoted to reviewed.
"""
import argparse
import copy
import json
import os
from pathlib import Path

import nibabel as nib
import numpy as np

from resectionlab.patient_anatomy import PatientImagingCollection, file_hash, resample
from resectionlab.anatomy_alignment import load_grid, estimate_pair, input_key
from resectionlab.ixi_multimodal import SUBJECT, diffusion_volumes, tensor_scalars, require_healthy

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / 'outputs/poc-ixi265-mra-fidelity-v1'
DEFAULT = ROOT / 'testcases/environments/IXI265-multimodal'
DIFFUSION = ROOT / 'testcases/downloaded-datasets/IXI265-Guys-0845/diffusion'
HASHES = {'T1': '317faf612b8e8a4e319fe3ec67404561b928ea9090fc23d1da20ce0900257950',
          'MRA': 'a87b381b882242b7d61a1930a8332d44200fdb3c56c68fc09c24a057eea86dc4'}


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def image_file(path, data, reference):
    image = nib.Nifti1Image(np.asarray(data, np.float32), reference.affine, reference.header.copy())
    image.set_data_dtype(np.float32)
    image.set_qform(reference.get_qform(), int(reference.header['qform_code']))
    image.set_sform(reference.get_sform(), int(reference.header['sform_code']))
    nib.save(image, path)


def reuse_cached_files(c):
    """Hard-link exact immutable content; edits always create new content hashes."""
    for cached_root in (CACHE/'native-t1', CACHE/'native-mra'):
        for source in cached_root.rglob('*'):
            if not source.is_file() or not source.relative_to(cached_root).parts[0] in ('arrays', 'originals'):
                continue
            target = c.root/source.relative_to(cached_root)
            if not target.is_file() or os.path.samefile(source, target):
                continue
            if source.stat().st_size == target.stat().st_size and file_hash(source) == file_hash(target):
                temporary = target.with_name(target.name+'.reuse')
                os.link(source, temporary)
                os.replace(temporary, target)


def align(c, sid):
    if sid in c.document['transforms']:
        c.registration_matrix(sid)  # Validate cached source binding before reuse.
        transform = c.document['transforms'][sid]
        if transform.get('quality') and 'input_key' not in transform:
            transform['input_key'] = input_key(c, c.series(sid), c.series('t1'))
            c.save('Bind measured registration inputs to desktop alignment cache: ' + sid)
        return
    print('Rigid registration:', sid, flush=True)
    matrix, quality = estimate_pair(load_grid(c, c.series('t1')), load_grid(c, c.series(sid)))
    c.set_registration(sid, matrix, status='unreviewed', method='rigid-quality-v2.4 Mattes MI; two starts and reverse fit')
    c.document['transforms'][sid]['quality'] = quality
    c.document['transforms'][sid]['input_key'] = input_key(c, c.series(sid), c.series('t1'))
    print(sid, quality['metrics'], flush=True)
    c.save('Measured rigid registration estimate: ' + sid)


def structural(output):
    output.mkdir(parents=True, exist_ok=True)
    native = output / 'native'
    c = PatientImagingCollection(native) if (native/'patient-anatomy.json').exists() else PatientImagingCollection.create(native, SUBJECT)
    role_file = ROOT/'manifests/experiments/ixi-component-person-cohort-v1.json'
    member = next(r for r in json.loads(role_file.read_text())['members'] if r['person_group'] == 'IXI:265')
    if member['role'] != 'TRAIN':
        raise ValueError('Frozen IXI265 role changed; do not use a reserved patient')
    c.document.update(dataset='IXI', source='https://brain-development.org/ixi-dataset/',
                      license='CC-BY-SA-3.0', patient_group='IXI:265', role=member['role'], training_admitted=False,
                      cohort_sha256=file_hash(role_file), scientific_scope='Publisher-described healthy subject; no tumor or resection objective',
                      acquisition_time='unknown; source filenames identify the same subject/site/acquisition code',
                      study_timepoint='source-session-unknown')
    c.document['cached_source_acquisition_receipt'] = c.original(CACHE/'acquisition.json')
    for sid, modality in [('t1', 'T1'), ('mra', 'MRA')]:
        path = CACHE/'originals'/f'IXI265-{modality}.nii.gz'
        if file_hash(path) != HASHES[modality]:
            raise ValueError('Cached acquired source hash differs')
        if sid not in {s['id'] for s in c.document['series']}:
            c.add_series(path, series_id=sid, modality=modality, timepoint='source-session-unknown', patient_id=SUBJECT)
        elif c.series(sid)['original']['sha256'] != HASHES[modality]:
            raise ValueError('Existing collection belongs to a different source')
    records = list((ROOT/'testcases/reconstruction-runs').glob('*/synthstrip-main/*/inference.json'))
    successful = [json.loads(p.read_text()) for p in records if json.loads(p.read_text()).get('status') == 'succeeded']
    record = next((r for r in successful if r.get('identity', {}).get('image_sha256') == HASHES['T1']), None)
    if record is None:
        from resectionlab.poc_models import run
        record = run(CACHE/'originals/IXI265-T1.nii.gz', SUBJECT+'-multimodal', 'synthstrip-main', 'T1')
    if 'brain' not in {r['id'] for r in c.document['layers']}:
        item = record['outputs'][0]
        if file_hash(item['path']) != item['sha256']:
            raise ValueError('Brain inference output changed')
        im = nib.load(item['path'])
        c.add_layer(np.asarray(im.dataobj, dtype=np.uint8), im.affine, layer_id='brain', source_series='t1',
                    kind='brain', role='context', color='#b8a8b6', label='Brain · SynthStrip estimate', provenance={'inference': record})
    if 'vessels' not in {r['id'] for r in c.document['layers']}:
        original = PatientImagingCollection(CACHE/'native-mra')
        layer = original.layer('vessels')
        provenance = copy.deepcopy(layer['provenance'])
        provenance['source_modality'] = original.series('mra')['modality']
        source = original.path(provenance['source_original']['path'])
        provenance['source_original'] = c.original(source)
        c.add_layer(original.array(layer['data']), layer['affine'], layer_id='vessels', source_series='mra',
                    kind='vessels', role='critical', color='#ad4730', label='MRA vessels · published partial annotation', provenance=provenance)
        c.layer('vessels')['coverage'] = 'Published partial vessel annotation within acquired MRA slab; unlabeled cells are unknown, not vessel-free.'
    c.layer('vessels')['provenance']['source_modality'] = c.series('mra')['modality']
    c.document['missing'] = ['Healthy IXI subject: no acquired tumor and no removal target', 'Skull unsupported: no CT',
                             'SWI and venous-specific imaging unavailable', 'Vessel annotation completeness and artery/vein identity unknown',
                             'MRA and diffusion registration estimates require anatomical review',
                             'Motor/language function and named white-matter pathways unknown',
                             'Diffusion motion/distortion correction and physical gradient frame unverified']
    reuse_cached_files(c)
    c.save('Same-subject cached structural and angiographic source assembly')
    align(c, 'mra')
    return c


def diffusion(c, output, directory):
    reference, signal, bvals, bvecs, sources = diffusion_volumes(directory)
    receipt = json.loads((directory/'acquisition.json').read_text())
    if {r['name']: r['sha256'] for r in receipt['members']} != {r['name']: r['sha256'] for r in sources}:
        raise ValueError('Diffusion source receipt mismatch')
    sid = 'dwi-b0'
    if sid not in {r['id'] for r in c.document['series']}:
        row = c.add_series(directory/f'{SUBJECT}-DTI-00.nii.gz', series_id=sid, modality='DWI b0',
                           timepoint='source-session-unknown', patient_id=SUBJECT)
        row['label'] = 'Diffusion b0 · acquired'
    c.document['diffusion'] = {'source_members': sources, 'source_originals': [c.original(directory/r['name']) for r in sources],
                               'gradients': {name: c.original(directory/name) for name in ('bvals.txt', 'bvecs.txt')},
                               'acquisition_receipt': c.original(directory/'acquisition.json'), 'source_member_order': [r['name'] for r in sources],
                               'bvalues_s_per_mm2': bvals.tolist(), 'gradient_frame': 'unresolved',
                               'motion_corrected': False, 'distortion_corrected': False, 'planning_use': 'excluded'}
    c.save('Preserve all sixteen acquired diffusion volumes and published gradients')
    align(c, sid)
    brain = c.mask(c.layer('brain'))
    mask = resample(brain.astype(np.uint8), np.asarray(c.series('t1')['affine']), reference.shape,
                    c.registration_matrix(sid) @ reference.affine, order=0) > 0
    mask &= signal[..., 0] > max(1., float(np.percentile(signal[..., 0], 98)) * .05)
    derivatives = output/'diffusion-diagnostics'
    derivatives.mkdir(exist_ok=True)
    if not (derivatives/'tensor-report.json').exists():
        fit = tensor_scalars(signal, bvals, bvecs, mask)
        for name in ('fa', 'md', 'normalized_rmse', 'fit_coverage'):
            image_file(derivatives/(name+'.nii.gz'), fit[name], reference)
        report = {**fit['report'], 'source_members': sources,
                  'source_gradients': {k: file_hash(directory/k) for k in ('bvals.txt', 'bvecs.txt')},
                  'mask_definition': 'T1 SynthStrip mask sampled through estimated b0-to-T1 rigid transform; positive baseline threshold',
                  'registration': copy.deepcopy(c.document['transforms'][sid]),
                  'derivatives': {name: file_hash(derivatives/(name+'.nii.gz')) for name in ('fa', 'md', 'normalized_rmse', 'fit_coverage')}}
        write(derivatives/'tensor-report.json', report)
    report = json.loads((derivatives/'tensor-report.json').read_text())
    if report['source_members'] != sources or report['registration'] != c.document['transforms'][sid]:
        raise ValueError('Stale tensor diagnostic cache')
    for name, digest in report['derivatives'].items():
        if file_hash(derivatives/(name+'.nii.gz')) != digest:
            raise ValueError('Tensor diagnostic checksum mismatch')
    if 'dwi-fa' not in {s['id'] for s in c.document['series']}:
        row = c.add_series(derivatives/'fa.nii.gz', series_id='dwi-fa', modality='DTI FA diagnostic',
                           timepoint='source-session-unknown', patient_id=SUBJECT)
        row.update(label='FA · uncorrected tensor diagnostic', provenance={'report': c.original(derivatives/'tensor-report.json'),
                   'not_acquired_contrast': True, 'unknown_outside_fit_coverage': True})
        c.set_registration('dwi-fa', c.registration_matrix(sid), status='unreviewed',
                           method='Same native grid as b0; inherited estimated b0-to-T1 transform')
        c.document['transforms']['dwi-fa']['quality'] = copy.deepcopy(c.document['transforms'][sid]['quality'])
        c.document['transforms']['dwi-fa']['quality']['notes'] = ['Inherits b0 rigid estimate on the exactly preserved native DWI grid; no independent FA-to-T1 fit.']
        c.document['transforms']['dwi-fa']['input_key'] = input_key(c, row, c.series('t1'))
        c.document['transforms']['dwi-fa']['inherited_from'] = sid
    c.document['diffusion']['tensor_diagnostics'] = report
    c.document['diffusion']['diagnostic_originals'] = {name: c.original(derivatives/(name+'.nii.gz'))
        for name in ('fa', 'md', 'normalized_rmse', 'fit_coverage')}
    c.save('Diagnostic diffusion scalars; no tract or functional anatomy generated')


def export(c, output):
    from resectionlab.surgical_patient import prepare_surgical_environment
    from resectionlab.imaging import load_case, save_case, read_case_artifacts
    from resectionlab.surgical_scene import from_case
    from resectionlab.surgical_environment import standardize
    from resectionlab.surgical_engine import SurgicalEngine
    from resectionlab.surgical_rl_protection import ProtectionAnnotation, ProtectionSet, GuardConfig
    destination = output/'simulator'
    report = prepare_surgical_environment(c.root, destination, layer_ids={'brain': 'brain', 'vessels': 'vessels'},
                                         resolution_mm=2., allow_unreviewed_registration=True)
    case = load_case(report['casePath'])
    require_healthy(case)
    scene = standardize(from_case(case))
    mask = case.compartments['vessels']
    annotation = ProtectionAnnotation.from_mask(scene, 'partial_mra_vessels', mask,
        provenance='Same-patient published MRA vessel annotation; unreviewed rigid transform; incomplete coverage',
        display_name='Partial MRA vessel exclusion')
    protection = ProtectionSet(scene, [annotation], clearance_mm=1., required=('brainstem',))
    engine = SurgicalEngine(scene, protection=protection, guard_config=GuardConfig(require_critical_coverage=False))
    artifacts = read_case_artifacts(report['casePath'])
    artifacts['surgical_toolbox']['branches']['main'] = engine.snapshot()
    save_case(case, report['casePath'], artifacts=artifacts)
    # Preserve acquired source views in the canonical simulator workspace too.
    # These native arrays stay display-only; the unreviewed transforms cannot
    # silently authorize source fusion or functional policy observations.
    from resectionlab.core import CaseData, SourceRef
    from resectionlab.workspace_bundle import OriginalFile, save_workspace, load_workspace, view_state
    from resectionlab.workspace_imaging import DisplaySource
    sources = []
    for sid in ('t1', 'mra', 'dwi-b0'):
        if sid not in {s['id'] for s in c.document['series']}:
            continue
        row = c.series(sid)
        original = c.path(row['original']['path'])
        source_case = CaseData(case_id=SUBJECT, mri=c.array(row['data']), compartments={},
            affine=np.asarray(row['affine']), source_refs=(SourceRef('structural', original.as_uri(), row['original']['sha256'],
                        license='CC-BY-SA-3.0', native_frame='RAS+', provenance='observed'),),
            metadata={'display_label': row.get('label', row['modality'] + ' · acquired'),
                      'same_patient_source_identity': SUBJECT, 'source_filename_evidence': copy.deepcopy(c.document.get('diffusion', {}).get('source_members', [])),
                      'registration_estimate': copy.deepcopy(c.document['transforms'].get(sid)),
                      'policy_use': 'display_only; diffusion and functional coverage remain unknown'})
        sources.append(DisplaySource(source_case, row['modality'] if sid != 'dwi-b0' else 'other-3D-scalar', 'none'))
    originals = {}
    for row in c.document['series']:
        if row['id'] == 'dwi-fa':
            continue  # A derived scalar is not represented as an acquired source.
        p = c.path(row['original']['path'])
        originals[file_hash(p)] = OriginalFile.from_path(p, role='imaging')
    for row in c.document.get('diffusion', {}).get('source_originals', []):
        p = c.path(row['path'])
        originals[file_hash(p)] = OriginalFile.from_path(p, role='imaging')
    for row in c.document.get('diffusion', {}).get('gradients', {}).values():
        p = c.path(row['path'])
        originals[file_hash(p)] = OriginalFile.from_path(p, role='metadata')
    session_hash = save_workspace(case, report['casePath'], sources=sources, imaging=view_state(case, sources),
        episode=None, selection=None, artifacts=artifacts, originals=list(originals.values()))
    restored, restored_artifacts, saved = load_workspace(report['casePath'])
    if (restored.semantic_hash != case.semantic_hash or saved['sessionHash'] != session_hash
            or len(saved['sources']) != len(sources) or len(saved['originals']) != len(originals)
            or restored_artifacts != artifacts):
        raise AssertionError('Canonical healthy workspace failed source/guard save-reopen verification')
    from resectionlab.surgical_rl_env import SurgicalRLEnv, RLConfig
    try:
        SurgicalRLEnv(scene, config=RLConfig(require_critical_coverage=False))
    except ValueError as error:
        if 'No unprotected explicitly designated tumor' not in str(error):
            raise
        refusal = str(error)
    else:
        raise AssertionError('Healthy case unexpectedly admitted to tumor-removal RL')
    # A bounded source-supported vessel test uses an inactive instrument. No
    # destructive stroke, invented target or trained policy is needed.
    ids = np.flatnonzero(mask)
    point = scene.points(ids[len(ids)//2:len(ids)//2+1])[0]
    action = {'slot': 'left', 'operation': 'move', 'duration': .1,
              'pose': {'position': point.tolist(), 'quaternion': [0., 0., 0., 1.]}}
    probe = engine.clone()
    # Explicit retrospective local initialization, not a claim of a navigable
    # access route through intact healthy brain. The saved scene stays intact.
    probe.state['slots']['left']['position'] = (point - [0., 0., 2.]).tolist()
    local_initial_pose = copy.deepcopy(probe.state['slots']['left'])
    guarded = probe.step(action)
    if guarded.get('reasonCode') != 'PROTECTED_STRUCTURE':
        raise AssertionError('Vessel exclusion failed to block vessel-intersecting instrument')
    write(output/'simulator-verification.json', {'case_path': report['casePath'], 'scene_id': scene.identity,
        'no_tumor': True, 'target_volume_mm3': scene.descriptor()['environment']['target']['initialVolumeMm3'],
        'rl_removal_refusal': refusal, 'engine_observation': engine.observe(),
        'vessel_probe': {'point_ras_mm': point.tolist(), 'action': action, 'outcome': guarded,
                         'initial_pose': local_initial_pose,
                         'scope': 'Hypothetical local geometry probe; no operative access route or resection objective',
                         'capsule_source_cell_hits': len(protection.capsule_hits(point, point, 1.))},
        'source_collection_unchanged_during_export': PatientImagingCollection(c.root).identity() == c.identity(),
        'protection_persisted': True, 'training': False, 'workspace_session_hash': session_hash,
        'native_source_views': len(sources), 'archived_original_files': len(originals),
        'diffusion_policy_role': 'Source acquisition preserved; diagnostic diffusion excluded from functional constraints'})
    print('Healthy simulator:', report['casePath'], flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=DEFAULT)
    parser.add_argument('--diffusion', type=Path, default=DIFFUSION)
    parser.add_argument('--stage', choices=['structural', 'all'], default='all')
    args = parser.parse_args()
    import SimpleITK as sitk
    sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(2)
    from resectionlab.anatomy_workspace import read_view, save_view
    old_path = args.output.resolve()/'native/patient-anatomy.json'
    saved_view = read_view(PatientImagingCollection(old_path.parent)) if old_path.exists() else None
    c = structural(args.output.resolve())
    if args.stage == 'all':
        diffusion(c, args.output.resolve(), args.diffusion.resolve())
        export(c, args.output.resolve())
        write(args.output.resolve()/'modality-coverage.json', {
            'subject': SUBJECT, 'population': 'IXI publisher-described healthy subject; individual clinical status not independently reviewed',
            'same_subject_evidence': 'Cached T1/MRA archive member names and all sixteen DTI filenames share IXI265-Guys-0845; checksums and physical headers preserved',
            'T1': 'acquired; native original retained', 'MRA': 'acquired; partial slab; native original retained',
            'diffusion': 'all sixteen published volumes and gradients acquired; native originals retained',
            'FA_MD': 'diagnostic OLS estimates, uncorrected diffusion; no functional interpretation',
            'brain': 'SynthStrip estimate; unreviewed', 'vessels': 'Published partial semi-automatic/manual annotation; local review pending',
            'tumor': 'absent; no synthetic lesion, target or removal objective', 'skull': 'unsupported; no CT',
            'SWI': 'unavailable', 'venous_coverage': 'unknown; no dedicated venography',
            'tractography': 'not generated; physical gradient convention and preprocessing unresolved',
            'functional_coverage': 'unknown', 'same_acquisition_time_verified': False,
            'registration_review': 'measured rigid estimates; no expert or landmark validation',
            'training': False, 'frozen_corpus_role': c.document['role'],
            'full_multimodal_glioma_milestone': False})
    if saved_view is not None:
        saved_view['snapshotId'] = c.identity()
        save_view(c, saved_view)
    print(c.root/'patient-anatomy.json')


if __name__ == '__main__':
    main()

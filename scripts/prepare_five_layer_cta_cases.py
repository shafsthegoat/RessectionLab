#!/usr/bin/env python3
"""Assemble source-bound CTA inspection cases with an explicitly geometric dural proxy."""
from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path
from shutil import copyfile
from tempfile import TemporaryDirectory

import nibabel as nib
import numpy as np

from build_cta_organic_gliomas import OBSERVATION as LEGACY_TARGET, read_json
from build_new_cta_gliomas import OBSERVATION as NATIVE_TARGET
from data_integrated_hosts import ORDER, TARGET as INTEGRATED_TARGET, UUID, binding, checked_file
from resectionlab.anatomy_dural import reconstruct_dura
from resectionlab.anatomy_package import open_package, referenced_files, save_package
from resectionlab.anatomy_planning import native_target_options
from resectionlab.core import array_digest
from resectionlab.data_root import atomic_json, checked_root, safe_path, sha256_file
from resectionlab.patient_anatomy import PatientImagingCollection, world
from resectionlab.workspace_bundle import load_workspace

OUTPUT = 'prepared/five-layer-cta-inspection-v1'
TOPCOW = 'prepared/cta-organic-gliomas-v1'
ISLES = 'prepared/isles0146-cta-gliomas-v1'
TARGET = 'prior-simulated-tumor'
DURA = 'dural-envelope-proxy'
INTEGRATED = ('004', '018')
SUBJECTS = ('006', '012', '014', '0146', *INTEGRATED)


def parent_for(root, subject):
    if subject == '0146':
        index = safe_path(root, f'{ISLES}/collection.json')
        return checked_file(root, read_json(index)['showcase'])
    if subject == '014':
        parent = f'prepared/topcow014-publisher-branches-v1/TopCoW-{subject}-CTA-PUBLISHER-BRANCHES-v1'
    else:
        parent = f'prepared/cta-published-cow-v1/TopCoW-{subject}-CTA-MRA-PUBLISHED-COW-v1'
    receipt = read_json(safe_path(root, f'{parent}/receipt.json'))
    return checked_file(root, receipt['manifest'])


def prior_observation(root, row, subject, scan, work):
    path = checked_file(root, row['native_case'])
    if subject == '0146':
        donor = PatientImagingCollection(open_package(path, work).parent)
        layer = donor.layer(NATIVE_TARGET)
        donor_scan = donor.series(donor.document['reference_series'])
        if (donor.identity() != row['snapshot_id'] or layer['data']['sha256'] != row['target']['array_sha256']
                or donor_scan['original'] != scan['original']):
            raise ValueError('ISLES contour must bind the same verified source study')
        data, affine = donor.mask(layer), layer['affine']
    else:
        case, artifacts, workspace = load_workspace(path)
        source = next(s for s in case.source_refs if s.source_id == 'structural')
        receipt = read_json(path.with_name('receipt.json'))
        data, affine = case.compartments[LEGACY_TARGET], case.affine
        if (case.case_id != row['case_id'] or case.frame != 'RAS+' or artifacts or not workspace
                or workspace['episode'] is not None or source.sha256 != scan['original']['sha256']
                or array_digest(data) != receipt['observation']['array_hash']
                or not np.array_equal(data, case.source_compartments[LEGACY_TARGET])):
            raise ValueError('TopCoW contour must bind the intact archived observation and source CTA')
    if not np.array_equal(affine, scan['affine']) or list(data.shape) != scan['data']['shape'] or not data.any():
        raise ValueError('Archived contour differs from the acquired reference lattice')
    return data


def stage_parent(path, destination):
    source = PatientImagingCollection(path.parent)
    document = deepcopy(source.document)
    document.pop('snapshot_id', None)
    document['layers'] = [row for row in document['layers'] if row['kind'] not in {'tumor', 'synthetic-target'}]
    destination.mkdir()
    study = PatientImagingCollection(destination, document)
    for name, checksum in set(referenced_files(document)):
        output = study.path(name)
        output.parent.mkdir(parents=True, exist_ok=True)
        copyfile(source.path(name), output)
        if sha256_file(output) != checksum:
            raise ValueError(f'Changed source member {name}')
    return attach_dural_proxy(study)


def attach_dural_proxy(study):
    document = study.document
    scan = study.series(document['reference_series'])
    brain = next(row for row in document['layers'] if row['kind'] == 'brain' and row['source_series'] == scan['id'])
    record = reconstruct_dura(study, scan, cranial_layer_id=None, brain_layer_id=brain['id'], cancelled=lambda: False)
    if record['availability'] != 'available':
        raise ValueError('CT did not support an enclosed dural proxy')
    proxy_path = Path(record['path'])
    proxy = nib.load(proxy_path)
    flags = {key: value for key, value in record.items() if key != 'path'}
    flags['derived_original'] = study.original(proxy_path)
    layer = study.add_layer(np.asanyarray(proxy.dataobj), proxy.affine, layer_id=DURA, source_series=scan['id'],
                            kind='dura', role='context', label='Dural envelope · geometric proxy', color='#c5a5df',
                            provenance=flags)
    layer.update(coverage=record['coverage_limitation'], units_status='mm')
    return study, scan, layer


def integrated_source(root, subject, destination):
    folder = 'integrated-v1/anatomy-hosts-v1'
    index = read_json(safe_path(root, f'{folder}/candidate-index.json'))
    host = next(row for row in index['cases'] if row['subject'] == subject)
    host_report = read_json(checked_file(root, {'path': host['report_path'], 'sha256': host['report_sha256']}))
    evidence = host_report['source']
    parent = evidence['reviewed_anatomy_parent']
    donor, _, state = load_workspace(checked_file(root, parent))
    ct_ref, label_ref = evidence['source_CTA_and_v1_labels']
    ct, labels = (checked_file(root, row) for row in (ct_ref, label_ref))
    observed = next(row for row in donor.source_refs if row.source_id == 'structural')
    if (host_report['family_id'] != f'TopCoW:{subject}' or donor.case_id != parent['case_id']
            or donor.frame != 'RAS+' or not state or state['episode'] is not None
            or observed.sha256 != ct_ref['sha256'] or set(donor.compartments) != {*ORDER, INTEGRATED_TARGET}
            or any(not np.array_equal(mask, donor.source_compartments[name])
                   for name, mask in donor.compartments.items())):
        raise ValueError('Integrated source does not preserve its archived geometry memberships')
    timepoint = 'source-session-unknown'
    study = PatientImagingCollection.create(destination, host['case_id'], timepoint=timepoint)
    scan = study.add_series(ct, series_id='ct', modality='CTA', timepoint=timepoint)
    if (scan['original']['sha256'] != ct_ref['sha256'] or not np.array_equal(scan['affine'], donor.affine)
            or not np.array_equal(study.array(scan['data']), donor.mri)):
        raise ValueError('Integrated native import changed the original CTA')
    scan.update(label='TopBrain CTA · source before the simulated contour',
                publisher_dataset='https://zenodo.org/records/21972006', source_license=observed.license)
    label_image = nib.load(labels)
    categories = np.asanyarray(label_image.dataobj)
    if (not np.array_equal(label_image.affine, donor.affine)
            or not np.array_equal((categories >= 1) & (categories <= 34), donor.compartments[ORDER[2]])
            or not np.array_equal(categories >= 35, donor.compartments[ORDER[3]])):
        raise ValueError('CTA native labels disagree with published artery or vein categories')
    original = study.original(labels)
    masks = dict(donor.compartments)
    skull_source = evidence['anatomy_masks']['observed_skull']
    if subject == '018':
        broad = read_json(safe_path(root, 'integrated-v1/skull-display-CTA018-v1/report.json'))
        if broad['source_ct'] != ct_ref or broad['primary_host'] != host:
            raise ValueError('Wider skull estimate must bind this exact acquired CTA')
        skull_source = broad['estimated_mask']
        image = nib.load(checked_file(root, skull_source))
        extended = np.asanyarray(image.dataobj) > 0
        if (not np.array_equal(image.affine, donor.affine) or np.any(masks[ORDER[1]] & ~extended)
                or any(np.any(extended & masks[name]) for name in (*ORDER[2:], INTEGRATED_TARGET))):
            raise ValueError('Wider skull is incompatible with the reviewed source memberships')
        masks[ORDER[1]] = extended
    for name, kind, label, color in zip(ORDER, ('brain','skull','artery','vein'),
                                 ('Estimated brain envelope','Estimated mineralized skull',
                                  'Supplied TopBrain arteries','Supplied TopBrain veins and sinuses'),
                                 ('#d09bb6','#ddc5a9','#f06455','#71b7ef')):
        if kind in {'artery', 'vein'}:
            provenance = {'kind':'supplied_annotation','source':original}
        else:
            origin = skull_source if kind == 'skull' else evidence['anatomy_masks']['brain_estimate']
            checked_file(root, origin)
            provenance = {'kind':'image_derived_estimate','source_sha256':origin['sha256']}
        layer = study.add_layer(masks[name].astype(np.uint8), donor.affine, layer_id=name,
                                source_series='ct', kind=kind, label=label, role='context', color=color,
                                provenance=provenance)
        if kind in {'artery', 'vein'}:
            layer['units_status'] = label_image.header.get_xyzt_units()[0]
    study.document['data_case'] = {'source_family_id':host_report['family_id'], 'role':'DEVELOPMENT',
                                   'source_family_role':host_report['role'], 'clinical_validation':False}
    if not masks[INTEGRATED_TARGET].any():
        raise ValueError('Archived integrated contour is empty')
    case = {'case_id':donor.case_id,'native_case':{key:parent[key] for key in ('path','sha256')}}
    return attach_dural_proxy(study), case, masks[INTEGRATED_TARGET]


def configure_inspection(variant, required, cursor, axial_fraction):
    document = variant.document
    anatomy = {row['id'] for row in document['layers']}
    visible = [name for name in document.get('display', {}).get('initial_layers', [])
               if name in anatomy and name != 'mra-publisher-artery']
    # Retain the publisher branch visibility on CTA 014. The whole mesh and
    # wider arteries remain separately selectable.
    if not set(visible).intersection(required['vasculature']):
        visible += required['vasculature']
    for key in ('skull', 'brain', 'dura', 'tumor'):
        visible += [name for name in required[key] if name not in visible]
    document['display'] = {'initial_layers': visible, 'cursor_ras_mm': cursor}
    document['five_layer_inspection'] = {'scope': 'anatomical and synthetic-contour inspection',
                                         'anatomy_layers': required, 'measured_patient_dura': False,
                                         'acquired_tumor': False, 'target_axial_proxy_supported_fraction': axial_fraction}


def publish_archive(variant, data, archive, work):
    with TemporaryDirectory(dir=work, prefix='case-verify-') as tmp:
        temp_archive = Path(tmp) / archive.name
        save_package(variant, temp_archive)
        recovered = PatientImagingCollection(open_package(temp_archive, Path(tmp) / 'fresh').parent)
        if recovered.identity() != variant.identity() or native_target_options(recovered):
            raise ValueError('Cold reopen altered the study or inspection authority')
        display = recovered.document['display']
        groups = recovered.document['five_layer_inspection']['anatomy_layers']
        if not all(set(names).intersection(display['initial_layers']) for names in groups.values()):
            raise ValueError('Cold reopen did not expose all five anatomical groups')
        if not np.array_equal(recovered.mask(recovered.layer(TARGET)), data):
            raise ValueError('Cold reopen changed the previously archived tumor contour')
        pending = archive.with_name('.' + archive.name)
        try:
            copyfile(temp_archive, pending)
            if sha256_file(temp_archive) != sha256_file(pending):
                raise ValueError('Portable transfer changed the verified archive')
            pending.replace(archive)
        finally:
            pending.unlink(missing_ok=True)


def build_variant(root, study, scan, envelope, case, subject, directory, work, observation=None):
    identifier = case['case_id'].removesuffix('-v1') + '-FIVE-LAYER-v1'
    archive = directory / f'{identifier}.rsworkspace'
    receipt_path = directory / f'{identifier}.json'
    if archive.exists():
        result = read_json(receipt_path)
        if checked_file(root, result['archive']) != archive or result['original_scenario'] != case['native_case']:
            raise ValueError('Existing five-layer archive does not match its evidence')
        return result

    data = prior_observation(root, case, subject, scan, work / 'donor') if observation is None else observation
    document = deepcopy(study.document)
    document.pop('snapshot_id', None)
    document.update(patient_id=identifier, revision=0, history=[])
    for row in document['series'] + document['layers']:
        row['patient_id'] = identifier
    variant_root = work / identifier
    variant_root.mkdir()
    variant = PatientImagingCollection(variant_root, document)
    for name, _ in set(referenced_files(document)):
        output = variant.path(name)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.hardlink_to(study.path(name))
    variant.add_layer(data.astype(np.uint8), scan['affine'], layer_id=TARGET, source_series=scan['id'], kind='tumor',
                      role='context', label='Simulated tumor · retained contour', color='#f297c8',
                      provenance={'kind': 'simulated_observation', 'prior_scenario_sha256': case['native_case']['sha256'],
                                  'evidence_use': 'context_only', 'planning_target_candidate': False})
    anatomy = variant.document['layers']
    required = {
        'skull': [row['id'] for row in anatomy if row['kind'] in {'bone', 'skull'}],
        'brain': [row['id'] for row in anatomy if row['kind'] == 'brain'],
        'dura': [envelope['id']],
        'vasculature': [row['id'] for row in anatomy if row['kind'] in {'artery', 'vein', 'sinus'}
                         and document['transforms'].get(row['source_series'], {}).get('status')
                         in {'reference', 'reviewed', 'source-provided'}],
        'tumor': [TARGET],
    }
    # On reference frames the reference transform is always present. Gate the
    # exact supplied positive masks and preserve all wider branches in the study.
    if not all(required.values()) or native_target_options(variant):
        raise ValueError('Five-layer companion must provide all groups with inspection-only scope')
    for name in required['skull'] + required['brain'] + required['dura'] + required['vasculature']:
        if not variant.mask(variant.layer(name)).any():
            raise ValueError(f'Source anatomy is empty: {name}')
    points = world(np.asarray(scan['affine']), np.argwhere(data))
    proxy_z = np.rint(world(np.linalg.inv(envelope['affine']), points)[:, 2]).astype(int)
    supported = envelope['provenance']['supported_axial_indices']
    axial_fraction = float(np.mean(np.isin(proxy_z, supported)))
    configure_inspection(variant, required, points.mean(0).tolist(), axial_fraction)
    variant.save('Assemble the source-linked anatomy and geometric dural context')
    publish_archive(variant, data, archive, work)
    result = {'case_id': identifier, 'original_scenario': case['native_case'], 'archive': binding(root, archive),
              'snapshot_id': variant.identity(), 'layers': required,
              'dural_proxy_slices': len(supported), 'target_axial_proxy_supported_fraction': axial_fraction,
              'cold_reopen_passed': True}
    atomic_json(receipt_path, result)
    return result


def build(root, subjects, scratch, library):
    output = safe_path(root, OUTPUT)
    output.mkdir(exist_ok=True)
    result = []
    def register_case(row):
        if library:
            archive = safe_path(root, row['archive']['path'])
            alias = library / archive.name
            if not alias.exists() and not alias.is_symlink():
                alias.symlink_to(archive)
            if not alias.is_symlink() or alias.resolve() != archive:
                raise ValueError(f'Cases alias has an unexpected target: {alias}')
        result.append(row)
        print(row['case_id'], row['dural_proxy_slices'], row['target_axial_proxy_supported_fraction'], flush=True)

    for subject in subjects:
        print(f'Preparing family {subject}', flush=True)
        if subject in INTEGRATED:
            with TemporaryDirectory(dir=scratch, prefix=f'five-layer-{subject}-') as tmp:
                work = Path(tmp)
                (study, scan, dura), case, mask = integrated_source(root, subject, work / 'study')
                register_case(build_variant(root, study, scan, dura, case, subject, output, work, mask))
            continue
        collection = ISLES if subject == '0146' else TOPCOW
        rows = read_json(safe_path(root, f'{collection}/collection.json'))['cases']
        selected = [row for row in rows if subject == '0146' or row['person_id'] == f'TopCoW:{subject}']
        if len(selected) != 4:
            raise ValueError('Expected the four previously sealed scenarios for this source family')
        with TemporaryDirectory(dir=scratch, prefix=f'five-layer-{subject}-') as tmp:
            work = Path(tmp)
            study, scan, dura = stage_parent(parent_for(root, subject), work / 'study')
            for case in selected:
                register_case(build_variant(root, study, scan, dura, case, subject, output, work))
    if set(subjects) == set(SUBJECTS):
        atomic_json(output / 'collection.json', {'schema': 'five-layer-cta-inspection-v1', 'source_people': len(subjects),
                    'generated_targets_reused': len(result), 'source_acquired_dura': 0,
                    'scope': 'Research anatomy inspection with explicitly geometric dura and simulated tumor',
                    'cases': result})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--scratch', type=Path, required=True, help='Ignored local test directory')
    parser.add_argument('--register', type=Path)
    parser.add_argument('--subject', choices=SUBJECTS, action='append')
    args = parser.parse_args()
    root = checked_root(args.root, expected_uuid=UUID, required_bytes=8 * 1024**3)
    build(root, args.subject or SUBJECTS, args.scratch, args.register)


if __name__ == '__main__':
    main()

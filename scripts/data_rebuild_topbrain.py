#!/usr/bin/env python3
"""Inspect and selectively extract the TopBrain V3 source without grid fusion."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import re
import resource
import shutil
import stat
import struct
import sys
import time
import urllib.request
import zipfile
import zlib

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO / 'src'), str(REPO / 'scripts')]
from resectionlab.data_root import checked_root, atomic_json, safe_path, sha256_file
from data_rebuild_acquire import DEFAULT_ROOT, UUID, header_qc

ARCHIVE = 'sources/topbrain/TopBrain_Data_Release_Batches1n2nTA36_081726.zip'
MD5 = 'aab94bed377ad71bfbde3f7e26d652c5'
CATALOG = 'catalog/topbrain'


def guard(root, required=0):
    return checked_root(root, expected_uuid=UUID, required_bytes=required)


def group_v1_labels(raw, dictionary, modality):
    """Interpret the released CT/MR V1 dictionaries independently."""
    import numpy as np
    if modality not in {'ct', 'mr'}:
        raise ValueError('Unknown TopBrain V1 modality')
    if (not np.isfinite(raw).all() or not np.equal(raw, np.floor(raw)).all()
            or not set(np.unique(raw)).issubset(dictionary.values())):
        raise ValueError('Source vascular labels disagree with actual modality/version dictionary')
    venous_names = {'VoG', 'StS', 'ICVs', 'R-BVR', 'L-BVR', 'SSS'}
    actual_venous = {value for name, value in dictionary.items() if name in venous_names}
    if (modality == 'ct' and actual_venous != set(range(35, 41))) or (modality == 'mr' and actual_venous):
        raise ValueError('Source V1 vascular semantics differ from the qualified release')
    grouped = np.zeros(raw.shape, dtype=np.uint8)
    grouped[raw > 0] = 1
    if modality == 'ct':
        grouped[np.isin(raw, list(actual_venous))] = 2
    return grouped


def qualified_mask_units(mask_image, paired_image):
    """Permit a missing unit declaration only with a matching source grid."""
    import numpy as np
    if mask_image.shape != paired_image.shape or not np.allclose(mask_image.affine, paired_image.affine, atol=1e-5, rtol=0):
        raise ValueError('Native vascular annotation does not share the corresponding image grid')
    image_units = paired_image.header.get_xyzt_units()[0]
    mask_units = mask_image.header.get_xyzt_units()[0]
    if image_units != 'mm' or mask_units not in {'mm', 'unknown'}:
        raise ValueError('Unresolved source vascular image/annotation unit convention')
    return mask_units


def package_array_hashes(path):
    """Compare actual NPY byte payloads without co-loading two native cases."""
    with zipfile.ZipFile(path) as outer:
        raw = outer.read('arrays.npz')
    result = {}
    with zipfile.ZipFile(io.BytesIO(raw)) as arrays:
        for entry in arrays.infolist():
            with arrays.open(entry) as stream:
                result[entry.filename] = {'bytes': entry.file_size,
                                          'sha256': hashlib.file_digest(stream, 'sha256').hexdigest()}
    return result


def archive_checked(root):
    guard(root)
    path = safe_path(root, ARCHIVE)
    if path.stat().st_size != 1963719117:
        raise ValueError('Incomplete official TopBrain V3 archive')
    with path.open('rb') as handle:
        if hashlib.file_digest(handle, 'md5').hexdigest() != MD5:
            raise ValueError('TopBrain archive publisher MD5 mismatch')
    return path


def inspect(root):
    archive = archive_checked(root)
    destination = safe_path(root, CATALOG)
    destination.mkdir(parents=True, exist_ok=True)
    members, pairs, metadata = [], {}, {}
    with zipfile.ZipFile(archive) as source:
        names = set()
        for info in source.infolist():
            path = PurePosixPath(info.filename)
            mode = info.external_attr >> 16
            if (path.is_absolute() or '..' in path.parts or '\\' in info.filename
                    or stat.S_ISLNK(mode) or info.filename in names
                    or info.flag_bits & 1 or info.file_size > 2 * 1024**3):
                raise ValueError(f'Unsafe source ZIP member: {info.filename}')
            names.add(info.filename)
            members.append({'name': info.filename, 'bytes': info.file_size,
                            'compressed_bytes': info.compress_size, 'crc32': f'{info.CRC:08x}'})
            match = re.fullmatch(r'topcow_(ct|mr)_(\d+)_0000\.nii\.gz', path.name)
            if match and 'imagesTr' in info.filename:
                modality, subject = match.groups()
                if modality in pairs.setdefault(subject, {}):
                    raise ValueError('Repeated subject/modality image source')
                pairs[subject][modality] = info.filename
            if (not info.is_dir() and info.file_size < 1024**2
                    and (path.suffix.lower() in {'.txt', '.json', '.md'})):
                payload = source.read(info)
                metadata[info.filename] = {'sha256': hashlib.sha256(payload).hexdigest(),
                                           'text': payload.decode('utf-8-sig')}
    if sum(item['bytes'] for item in members) > 12 * 1024**3:
        raise ValueError('Archive expansion exceeds bounded release expectation')
    if len(pairs) != 25 or any(set(value) != {'ct', 'mr'} for value in pairs.values()):
        raise ValueError('Actual archive does not establish 25 complete CT/MR pairs')
    # Exact source family shared by TopCoW, TopBrain and future annotation versions.
    roles = [{'subject': subject, 'family_id': f'TopCoW:{subject}',
              'role': 'DEVELOPMENT' if subject == '001' else 'TRAIN',
              'prior_exposure': 'existing native vascular development protocols v1-v4' if subject == '001' else 'new development; no held-out claim',
              'images': value, 'role_frozen_before_new_image_decode': True}
             for subject, value in sorted(pairs.items())]
    result = {'schema': 'ressectionlab-topbrain-source-plan-v1', 'declared_utc': datetime.now(timezone.utc).isoformat(),
              'source_record': 'https://zenodo.org/records/21972006', 'source_version': 'v3',
              'archive_path': ARCHIVE, 'archive_sha256': sha256_file(archive), 'archive_md5': MD5,
              'license': 'Noncommercial use with source attribution; commercial permission not sought',
              'population': 'clinical vascular cohort, not healthy; tumor status unknown',
              'members': members, 'metadata': metadata, 'subject_families': roles,
              'known_overlap': '25 TopCoW people; never count TopBrain as independent cohort',
              'registration': 'CTA/MRA native source frames retained separately; no fusion or assumed temporal equivalence',
              'accepted': False, 'new_subject_image_decoded': False}
    output = destination / 'source-plan.json'
    if output.exists():
        old = json.loads(output.read_text())
        if old['archive_sha256'] != result['archive_sha256'] or old['subject_families'] != roles:
            raise ValueError('Existing source plan differs; preserve frozen roles')
        return old
    guard(root)
    atomic_json(output, result)
    return result


def extract(root, subject):
    plan_path = safe_path(root, CATALOG + '/source-plan.json')
    plan = json.loads(plan_path.read_text())
    family = next(row for row in plan['subject_families'] if row['subject'] == subject)
    if family['role'] not in {'TRAIN', 'DEVELOPMENT'}:
        raise ValueError('Protected family unavailable for preparation')
    archive = archive_checked(root)
    if sha256_file(archive) != plan['archive_sha256']:
        raise ValueError('Frozen archive identity changed')
    suffix = re.compile(rf'topcow_(?:ct|mr)_{re.escape(subject)}(?:_0000)?\.nii\.gz$')
    selected = [m for m in plan['members'] if suffix.fullmatch(PurePosixPath(m['name']).name)]
    if not selected or sum(m['bytes'] for m in selected) > 1024**3:
        raise ValueError('Subject extraction missing or exceeds 1GiB bound')
    guard(root, sum(m['bytes'] for m in selected) * 2)
    receipts = []
    with zipfile.ZipFile(archive) as source:
        for item in selected:
            path = safe_path(root, f'prepared/topbrain/{subject}/native/' + item['name'])
            path.parent.mkdir(parents=True, exist_ok=True)
            if not path.exists():
                temporary = path.with_name(path.name + '.part')
                guard(root, item['bytes'])
                with source.open(item['name']) as incoming, temporary.open('wb') as outgoing:
                    shutil.copyfileobj(incoming, outgoing, 1024**2)
                if temporary.stat().st_size != item['bytes']:
                    raise ValueError('Truncated source member')
                temporary.replace(path)
            receipts.append({'source_member': item['name'], 'path': str(path.relative_to(root)),
                             'sha256': sha256_file(path), 'bytes': path.stat().st_size,
                             'native_header_qc': header_qc(path)})
    result = {'schema': 'ressectionlab-topbrain-native-extraction-v1', 'family': family,
              'source_plan_sha256': sha256_file(plan_path), 'files': receipts,
              'source_intensity_unchanged': True, 'registration_performed': False,
              'source_annotations_fused': False, 'accepted': False,
              'status': 'native_source_extracted_awaiting_independent_review_and_export'}
    guard(root)
    atomic_json(safe_path(root, CATALOG + f'/{subject}-native-extraction.json'), result)
    return result


def remote_plan(root):
    """Freeze pairing/roles from already acquired tiny official ZIP metadata."""
    guard(root)
    metadata_path = safe_path(root, 'catalog/source-research-topbrain-remote.json')
    record = json.loads(metadata_path.read_text())
    if record['source_archive_md5'] != MD5 or record['source_archive_bytes'] != 1963719117:
        raise ValueError('Unexpected remote source version')
    pairs = {}
    for member in record['members']:
        path = PurePosixPath(member['name'])
        if path.is_absolute() or '..' in path.parts or '\\' in member['name']:
            raise ValueError('Unsafe source ZIP member')
        match = re.fullmatch(r'topcow_(ct|mr)_(\d+)_0000\.nii\.gz', path.name)
        if match and 'imagesTr_topbrain/' in member['name']:
            modality, person = match.groups()
            if modality in pairs.setdefault(person, {}):
                raise ValueError('Ambiguous source subject image')
            pairs[person][modality] = member['name']
    if len(pairs) != 25 or any(set(v) != {'ct', 'mr'} for v in pairs.values()):
        raise ValueError('Remote central directory does not prove25 file pairs')
    result = {'schema': 'ressectionlab-topbrain-selective-source-plan-v1',
              'source_record': 'https://zenodo.org/records/21972006', 'source_version': 'v3',
              'metadata_path': str(metadata_path.relative_to(root)), 'metadata_sha256': sha256_file(metadata_path),
              'whole_archive_md5_verified': False, 'integrity_scope': 'official TLS exact ranges; ZIP CRC32; local SHA256, not publisher member cryptographic hashes',
              'declared_utc': datetime.now(timezone.utc).isoformat(),
              'subject_families': [{'subject': subject, 'family_id': f'TopCoW:{subject}',
                                    'role': 'DEVELOPMENT' if subject == '001' else 'TRAIN',
                                    'role_frozen_before_new_image_decode': True,
                                    'images': files} for subject, files in sorted(pairs.items())],
              'prior_exposure': 'TopCoW001 existing vascular development; remaining source people are new development, not independent evaluation',
              'accepted': False}
    path = safe_path(root, CATALOG + '/selective-source-plan.json')
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        old = json.loads(path.read_text())
        if old['metadata_sha256'] != result['metadata_sha256'] or old['subject_families'] != result['subject_families']:
            raise ValueError('Existing selective family declaration differs')
        return old
    atomic_json(path, result)
    return result


def remote_extract(root, subject):
    started = time.monotonic()
    plan_path = safe_path(root, CATALOG + '/selective-source-plan.json')
    plan = json.loads(plan_path.read_text())
    family = next(row for row in plan['subject_families'] if row['subject'] == subject)
    if family['role'] not in {'TRAIN', 'DEVELOPMENT'}:
        raise ValueError('Protected family cannot be prepared')
    metadata_path = safe_path(root, plan['metadata_path'])
    if sha256_file(metadata_path) != plan['metadata_sha256']:
        raise ValueError('Frozen ZIP metadata changed')
    prior_path = safe_path(root, CATALOG + f'/{subject}-selective-extraction.json')
    if prior_path.exists():
        prior = json.loads(prior_path.read_text())
        if prior['source_plan_sha256'] != sha256_file(plan_path) or prior['family'] != family:
            raise ValueError('Existing selective source receipt differs from frozen family plan')
        for entry in prior['files']:
            if sha256_file(safe_path(root, entry['path'])) != entry['sha256']:
                raise ValueError('Existing original differs from its immutable acquisition receipt')
        return prior
    source = json.loads(metadata_path.read_text())
    url, total = source['source_url'], source['source_archive_bytes']
    # The retained archive prefix is a source byte cache, never a full MD5 claim.
    archive_path = safe_path(root, ARCHIVE)
    partial = safe_path(root, 'staging/healthy-acquisition/' + hashlib.sha256(str(archive_path).encode()).hexdigest() + '.part')
    range_receipts = []

    def read_range(first, length):
        guard(root, length)
        if first < 0 or first + length > total or length > 8 * 1024**2:
            raise ValueError('Invalid bounded ZIP member range')
        if partial.exists() and partial.stat().st_size >= first + length:
            with partial.open('rb') as incoming:
                incoming.seek(first)
                data = incoming.read(length)
            origin = 'retained_original_archive_prefix'
        else:
            req = urllib.request.Request(url, headers={'Range': f'bytes={first}-{first+length-1}', 'Accept-Encoding': 'identity'})
            with urllib.request.urlopen(req, timeout=60) as response:
                if response.status != 206 or response.headers.get('Content-Range') != f'bytes {first}-{first+length-1}/{total}':
                    raise ValueError('Official server refused exact ZIP range')
                data = response.read(length + 1)
            origin = 'official_https_206_range'
        if len(data) != length:
            raise ValueError('Truncated source ZIP range')
        range_receipts.append({'first': first, 'bytes': length, 'origin': origin, 'sha256': hashlib.sha256(data).hexdigest()})
        return data

    suffix = re.compile(rf'topcow_(?:ct|mr)_{re.escape(subject)}(?:_0000)?\.nii\.gz$')
    selected = [m for m in source['members'] if suffix.fullmatch(PurePosixPath(m['name']).name)]
    if len(selected) != 6 or sum(m['bytes'] for m in selected) > 1024**3:
        raise ValueError('Expected two acquired images plus four versioned annotation files')
    guard(root, sum(m['bytes'] for m in selected) * 2)
    receipts = []
    for member in selected:
        destination = safe_path(root, f'sources/topbrain/selective/{subject}/' + member['name'])
        destination.parent.mkdir(parents=True, exist_ok=True)
        header = read_range(member['header_offset'], 30)
        signature, _version, flags, method, _time, _date, _crc, _compressed, _size, namelen, extralen = struct.unpack('<4s5H3I2H', header)
        if signature != b'PK\x03\x04' or flags & 1 or method not in {0, 8}:
            raise ValueError('Unsupported source ZIP member encoding')
        name = read_range(member['header_offset'] + 30, namelen).decode('utf-8')
        if name != member['name']:
            raise ValueError('Source local header differs from frozen central directory')
        first = member['header_offset'] + 30 + namelen + extralen
        compressed = safe_path(root, 'staging/topbrain-members/' + hashlib.sha256(member['name'].encode()).hexdigest() + '.part')
        if not destination.exists():
            compressed.parent.mkdir(parents=True, exist_ok=True)
            offset = compressed.stat().st_size if compressed.exists() else 0
            if offset > member['compressed_bytes']:
                raise ValueError('Oversized retained compressed member')
            with compressed.open('ab') as out:
                while offset < member['compressed_bytes']:
                    chunk = read_range(first + offset, min(8 * 1024**2, member['compressed_bytes'] - offset))
                    out.write(chunk)
                    offset += len(chunk)
            decoder = zlib.decompressobj(-15) if method == 8 else None
            temporary = destination.with_name(destination.name + '.part')
            crc, written = 0, 0
            with compressed.open('rb') as incoming, temporary.open('wb') as outgoing:
                for chunk in iter(lambda: incoming.read(1024**2), b''):
                    data = decoder.decompress(chunk) if decoder else chunk
                    written += len(data)
                    if written > member['bytes']:
                        raise ValueError('ZIP expansion exceeds declared source size')
                    outgoing.write(data)
                    crc = zlib.crc32(data, crc)
                if decoder and (not decoder.eof or decoder.unused_data):
                    raise ValueError('Invalid ZIP deflate stream')
            if written != member['bytes'] or f'{crc:08x}' != member['crc32']:
                raise ValueError('ZIP member CRC32/size mismatch')
            guard(root)
            temporary.replace(destination)
        # Recheck existing source content against central-directory CRC, too.
        crc = 0
        with destination.open('rb') as incoming:
            for chunk in iter(lambda: incoming.read(1024**2), b''):
                crc = zlib.crc32(chunk, crc)
        if destination.stat().st_size != member['bytes'] or f'{crc:08x}' != member['crc32']:
            raise ValueError('Existing native source CRC/size differs')
        receipts.append({'source_member': member['name'], 'path': str(destination.relative_to(root)),
                         'sha256': sha256_file(destination), 'zip_crc32': member['crc32'], 'bytes': member['bytes'],
                         'native_header_qc': header_qc(destination)})
        guard(root)
        atomic_json(safe_path(root, CATALOG + f'/{subject}-selective-progress.json'), {'family': family, 'files': receipts, 'range_receipts': range_receipts})
        # A completed deflate transport cache is redundant after preserving the
        # exact NIfTI member, CRC/SHA receipt and full gzip/NIfTI validation.
        if compressed.exists():
            guard(root)
            compressed.unlink()
    result = {'schema': 'ressectionlab-topbrain-selective-source-v1', 'family': family,
              'files': receipts, 'range_receipts': range_receipts,
              'source_plan_sha256': sha256_file(plan_path), 'whole_archive_md5_verified': False,
              'source_intensities_unchanged': True, 'registration_performed': False,
              'source_annotations_fused': False, 'accepted': False,
              'acquisition_elapsed_seconds': time.monotonic() - started,
              'status': 'native_source_extracted_awaiting_independent_review_and_export'}
    atomic_json(safe_path(root, CATALOG + f'/{subject}-selective-extraction.json'), result)
    return result


def batch(root, review_path, count):
    """Expand reviewed preparation while retaining every case as a candidate."""
    if review_path is None:
        raise ValueError('Independent pilot source/native preparation review is required')
    review = json.loads(review_path.read_text())
    if (review.get('person') != 'TopCoW:001' or review.get('independent') is not True
            or not str(review.get('decision', '')).startswith('accepted')
            or review.get('native_preparation_scaling_allowed') is not True):
        raise ValueError('Independent review does not release TopBrain candidate preparation scaling')
    plan = json.loads(safe_path(root, CATALOG + '/selective-source-plan.json').read_text())
    results = []
    for family in plan['subject_families'][:count]:
        started = time.monotonic()
        subject = family['subject']
        guard(root, 2 * 1024**3)
        try:
            remote_extract(root, subject)
            result = export_native(root, subject)
            result['source_and_preparation_elapsed_seconds'] = time.monotonic() - started
            results.append(result)
            atomic_json(safe_path(root, CATALOG + '/batch-candidates.json'), {
                'independent_review_path': str(review_path), 'independent_review_sha256': sha256_file(review_path),
                'completed_people': len(results), 'people': results, 'accepted': False})
            print(json.dumps({'subject': subject, 'candidate_packages': len(result['candidates']), 'accepted': False}), flush=True)
        except Exception as exc:
            atomic_json(safe_path(root, CATALOG + '/batch-last-failure.json'), {
                'subject': subject, 'error': f'{type(exc).__name__}: {exc}', 'partials_retained': True,
                'completed_people': len(results), 'accepted': False})
            raise
    return {'people_prepared': len(results), 'accepted': False, 'per_case_review': 'pending'}


def render_contact(root, subject, display, image_path, grouped_path):
    """Actual source planes and transparent annotations; no image registration."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    import nibabel as nib
    import numpy as np

    image = nib.as_closest_canonical(nib.load(image_path))
    mask = nib.as_closest_canonical(nib.load(grouped_path))
    values = np.asanyarray(image.dataobj)
    labels = np.asanyarray(mask.dataobj)
    spacing = image.header.get_zooms()
    center = []
    for axis in range(3):
        weights = np.count_nonzero(labels, axis=tuple(a for a in range(3) if a != axis))
        center.append(int(round(np.dot(np.arange(len(weights)), weights) / weights.sum()))
                      if weights.sum() else values.shape[axis] // 2)
    low, high = (-200, 900) if display == 'CTA' else (0, float(np.percentile(values[::4, ::4, ::2], 99.8)))
    rows = 3 if display == 'CTA' else 2
    fig, axes = plt.subplots(rows, 3, figsize=(15, 4.6 * rows), constrained_layout=True)
    names, directions = ['Sagittal voxel plane', 'Coronal voxel plane', 'Axial voxel plane'], ['R', 'A', 'S']
    for axis in range(3):
        plane, overlay = np.take(values, center[axis], axis=axis).T, np.take(labels, center[axis], axis=axis).T
        remaining = [a for a in range(3) if a != axis]
        extent = (0, values.shape[remaining[0]] * spacing[remaining[0]], 0, values.shape[remaining[1]] * spacing[remaining[1]])
        for row in range(rows):
            ax = axes[row, axis]
            ax.imshow(plane, cmap='gray', origin='lower', extent=extent,
                      vmin=0 if row == 2 else low, vmax=2000 if row == 2 else high)
            if row == 1:
                rgba = np.zeros((*overlay.shape, 4), dtype=np.float32)
                rgba[overlay == 1] = (1, .15, .08, .65)
                rgba[overlay == 2] = (.1, .5, 1, .7)
                ax.imshow(rgba, origin='lower', extent=extent)
            ax.set_xlabel(f'{directions[remaining[0]]}+ native distance (mm)')
            ax.set_ylabel(f'{directions[remaining[1]]}+ native distance (mm)')
            ax.set_title(f'{names[axis]} at index {center[axis]}\n' +
                         ('Source image' if row == 0 else 'Source + supplied union' if row == 1 else 'CT bone window; no skull segmentation'))
    legend = [Patch(color='#ff2a14', label='Supplied arterial annotation union')]
    if display == 'CTA':
        legend.append(Patch(color='#1980ff', label='Supplied veins/sinuses union'))
    fig.legend(handles=legend, loc='outside lower center', ncol=len(legend))
    fig.suptitle(f'TopCoW {subject} — {display} — source/native candidate review\n'
                 'Native voxel axes reoriented to RAS for review; original obliquity retained. '
                 'Background is unannotated, not proven vessel-free. No cross-modality registration.', fontsize=12)
    output = safe_path(root, f'qc/topbrain/{subject}-{display}-source-contact.png')
    output.parent.mkdir(parents=True, exist_ok=True)
    guard(root)
    fig.savefig(output, dpi=130)
    plt.close(fig)
    return str(output.relative_to(root))


def export_native(root, subject):
    """Use the real loader/exporter; each modality keeps its own native grid."""
    import dataclasses
    import gc
    import nibabel as nib
    import numpy as np
    from resectionlab.core import thaw_json
    from resectionlab.imaging import load_nifti_case
    from resectionlab.workspace_bundle import OriginalFile, load_workspace, save_workspace, view_state

    receipt_path = safe_path(root, CATALOG + f'/{subject}-selective-extraction.json')
    receipt = json.loads(receipt_path.read_text())
    plan = json.loads(safe_path(root, CATALOG + '/selective-source-plan.json').read_text())
    source = json.loads(safe_path(root, plan['metadata_path']).read_text())
    guard(root, 2 * 1024**3)
    outputs = []
    for modality, display in [('ct', 'CTA'), ('mr', 'TOF-MRA')]:
        started = time.monotonic()
        image_entry = next(f for f in receipt['files'] if f'/imagesTr_topbrain/topcow_{modality}_' in f['source_member'])
        mask_entry = next(f for f in receipt['files'] if f'/labelsTr_topbrain_v1_{modality}/' in f['source_member'])
        image_path, mask_path = safe_path(root, image_entry['path']), safe_path(root, mask_entry['path'])
        for path, entry in [(image_path, image_entry), (mask_path, mask_entry)]:
            if sha256_file(path) != entry['sha256']:
                raise ValueError('Native source differs from acquired receipt')
        mapping_text = next(value for key, value in source['metadata'].items() if key.endswith(f'/labels_topbrain_v1_{modality}.json'))
        dictionary = json.loads(mapping_text)['labels']
        mask_image = nib.load(mask_path)
        paired_image = nib.load(image_path)
        mask_units = qualified_mask_units(mask_image, paired_image)
        raw = np.asarray(mask_image.dataobj)
        grouped = group_v1_labels(raw, dictionary, modality)
        labels = {1: 'supplied_arterial_annotation_union'}
        if modality == 'ct':
            labels[2] = 'supplied_venous_sinus_annotation_union'
        prepared = safe_path(root, f'prepared/topbrain/{subject}/{modality}')
        prepared.mkdir(parents=True, exist_ok=True)
        grouped_path = prepared / 'supplied-v1-annotation-groups.nii.gz'
        grouped_image = nib.Nifti1Image(grouped, mask_image.affine, mask_image.header.copy())
        grouped_image.set_data_dtype(np.uint8)
        # Original supplied masks omit units. Assign units only to this derived
        # display union, after exact matching to its documented paired image.
        grouped_image.header.set_xyzt_units('mm')
        nib.save(grouped_image, grouped_path)
        grouping = {'source_mask_sha256': mask_entry['sha256'], 'source_dictionary': dictionary,
                    'source_native_grid_unchanged': True, 'grouped_mask_sha256': sha256_file(grouped_path),
                    'source_mask_spatial_units': mask_units, 'derived_spatial_units': 'mm',
                    'unit_assignment_evidence': 'Official same-modality subject label association plus exact source image/mask shape and affine coefficients within1e-5; image explicitly mm. Original mask unchanged.',
                    'label_groups': {'1': [name for name, value in dictionary.items() if value > 0 and (modality == 'mr' or value <= 34)],
                                     '2': [name for name, value in dictionary.items() if modality == 'ct' and value > 34]},
                    'meaning': 'Unions of supplied annotations for bounded display. Background is unannotated, not proven vessel-free. Exact original fine labels remain embedded.'}
        atomic_json(prepared / 'annotation-grouping.json', grouping)
        case_id = f'TopCoW-{subject}-{display}-TopBrain-v3'
        contract = {'schema': 'ressectionlab-data-case-v1', 'case_id': case_id,
                    'family_id': receipt['family']['family_id'], 'original_patient': f'TopCoW:{subject}',
                    'case_type': 'real', 'population': 'clinical_vascular_tumor_status_unestablished',
                    'diagnosis': 'clinical vascular cohort; individual diagnosis unavailable',
                    'source': 'TopBrain/TopCoW', 'source_version': 'TopBrain-v3-2026-08-17',
                    'source_doi': '10.5281/zenodo.21972006', 'source_license': plan.get('license', 'Noncommercial with source attribution; commercial permission required'),
                    'source_url': 'https://zenodo.org/records/21972006',
                    'role': receipt['family']['role'], 'split': receipt['family']['role'],
                    'source_files': receipt['files'], 'source_integrity': plan['integrity_scope'],
                    'whole_archive_md5_verified': False, 'source_frame': 'Each original NIfTI affine retained in its own RAS+ physical grid',
                    'source_registration': 'No CTA/MRA cross-modality registration assumed or performed; separate native packages',
                    'components': {display: {'provenance': 'acquired_image', 'original_intensities_preserved': True},
                                   'arteries': {'provenance': 'supplied_annotation', 'processing': 'explicit source class union', 'coverage': 'annotated landmark arteries only'},
                                   'veins_sinuses': {'provenance': 'supplied_annotation' if modality == 'ct' else 'missing', 'coverage': 'selected supplied veins/sinuses only' if modality == 'ct' else 'not supplied by TOF-MRA v1 labels'}},
                    'missing_anatomy': ['tumor_annotation', 'brain_segmentation', 'validated_skull_segmentation', 'tractography', 'functional_mapping'],
                    'capabilities': ['native_angiographic_image_inspection', 'supplied_vascular_annotation_inspection'],
                    'limitations': ['Cropped/defaced source; field of view must be individually reviewed',
                                    'No tumor-present structural MRI or same-stage full surgical anatomy established',
                                    'Vascular annotation cannot establish complete small-vessel/venous coverage or clinical risk',
                                    'No brain mask; simulation/resection eligibility withheld'],
                    'annotation_grouping': grouping, 'clinical_use_status': 'research_only',
                    'accepted': False, 'status': 'candidate_pending_independent_review'}
        if modality == 'mr':
            contract['display_policy'] = {
                'revision': 'display-v2', 'volume_rendering_default_enabled': False,
                'default': 'Structures 3D and full-resolution native MPR',
                'volume_rendering_user_opt_in': True, 'anatomical_resampling': False,
                'source_dataset_version_changed': False,
                'reason': 'High-resolution MRA pilot lost graphics context on hidden application reopen with 3D volume raycasting; source arrays remain unchanged.'}
        case = load_nifti_case(image_path, grouped_path, case_id=case_id, label_map=labels,
                               source_url=contract['source_url'], license=contract['source_license'],
                               metadata={'selected_modality': display, 'patient_id': f'TopCoW:{subject}',
                                         'source_collection': 'TopBrain-v3/TopCoW',
                                         'source_frame_declaration': contract['source_registration'],
                                         'source_annotation_role': 'vascular reference, not resection target',
                                         'data_case': contract})
        source_scalars = nib.load(image_path).get_fdata(dtype=np.float32)
        if not np.array_equal(case.mri, source_scalars):
            raise ValueError('Production scalar importer changed acquired source values')
        del source_scalars
        metadata = thaw_json(case.metadata)
        metadata.update(benchmark_track='vascular_annotation_inspection',
                        simulation_admission='withheld_missing_brain_and_tumor; source inspection only',
                        annotation_role='vascular_reference_not_tumor_or_resection_target')
        case = dataclasses.replace(case, metadata=metadata)
        # Retain useful high attenuation as an explicitly unvalidated proposal;
        # contrast, calcification and bone overlap, so it is not accepted skull.
        if modality == 'ct':
            high = np.asarray(nib.load(image_path).dataobj) >= 600
            high_path = prepared / 'ct-high-attenuation-proposal-not-accepted-skull.nii.gz'
            ni = nib.Nifti1Image(high.astype(np.uint8), nib.load(image_path).affine, nib.load(image_path).header.copy())
            ni.set_data_dtype(np.uint8)
            nib.save(ni, high_path)
            atomic_json(prepared / 'ct-high-attenuation-proposal.json', {
                'method': 'source scaled CT >=600, engineering candidate only; original DICOM HU calibration unavailable', 'source_sha256': image_entry['sha256'],
                'proposal_sha256': sha256_file(high_path), 'evidence': 'estimated_structure_candidate',
                'accepted_skull': False, 'limitations': ['May include contrast-filled vessels/calcification/artifact', 'Calvarial continuity, skull base and FOV unreviewed'],
                'used_in_native_case': False})
            del high, ni
        originals = [OriginalFile.from_path(safe_path(root, entry['path']),
                     role='imaging' if '/imagesTr_' in entry['source_member'] else 'annotation',
                     expected_sha256=entry['sha256']) for entry in receipt['files']]
        suffix = '-display-v2' if modality == 'mr' else ''
        output = safe_path(root, f'staging/candidates/clinical/TopCoW-{subject}-{display}-v3{suffix}.ressectionlab')
        output.parent.mkdir(parents=True, exist_ok=True)
        guard(root, 1024**3)
        expected_semantic, expected_planning = case.semantic_hash, case.planning_hash
        expected_scalar_hash = hashlib.sha256(np.ascontiguousarray(case.mri)).hexdigest()
        if output.exists():
            disposition = 'existing_package_revalidated'
        else:
            initial_view = None
            if modality == 'mr':
                initial_view = view_state(case, [])
                initial_view['states']['primary']['viewer'] = {'volumeRendering': {'enabled': False, 'preset': 'mri'}}
            save_workspace(case, output, sources=[], imaging=initial_view, episode=None, selection=None, artifacts={}, originals=originals)
            disposition = 'new_package_saved'
        # A native source and its reopened copy need not coexist in RAM.
        del raw, grouped, grouped_image, mask_image, paired_image, case, originals
        gc.collect()
        reopened, _, workspace = load_workspace(output)
        if thaw_json(reopened.metadata.get('data_case', {})) != contract:
            raise ValueError('Existing native candidate differs; preserve distinct recipe/version')
        if reopened.semantic_hash != expected_semantic or reopened.planning_hash != expected_planning:
            raise ValueError('Production native save/reopen identity differs')
        # The production scalar importer must preserve acquired scaled CT values.
        if hashlib.sha256(np.ascontiguousarray(reopened.mri)).hexdigest() != expected_scalar_hash:
            raise ValueError('Production reopen changed source scalar values')
        embedded_count = len(workspace['originals'])
        if modality == 'mr' and workspace['imagingState']['states']['primary'].get('viewer', {}).get('volumeRendering', {}).get('enabled') is not False:
            raise ValueError('MRA display policy did not survive production save/reopen')
        del reopened, workspace
        gc.collect()
        display_proof = None
        previous = safe_path(root, f'staging/candidates/clinical/TopCoW-{subject}-{display}-v3.ressectionlab')
        if modality == 'mr' and previous.exists():
            previous_arrays, current_arrays = package_array_hashes(previous), package_array_hashes(output)
            if previous_arrays != current_arrays:
                raise ValueError('Display-only revision changed actual native array payloads')
            display_proof = {'previous_path': str(previous.relative_to(root)), 'previous_sha256': sha256_file(previous),
                             'current_path': str(output.relative_to(root)), 'current_sha256': sha256_file(output),
                             'all_array_payloads_byte_identical': True, 'arrays': current_arrays,
                             'source_dataset_version_changed': False, 'anatomical_resampling': False,
                             'viewer_volume_rendering_enabled': False}
            atomic_json(safe_path(root, CATALOG + f'/{subject}-display-v2-array-proof.json'), display_proof)
        contact = render_contact(root, subject, display, image_path, grouped_path)
        outputs.append({'path': str(output.relative_to(root)), 'sha256': sha256_file(output),
                        'bytes': output.stat().st_size, 'case_id': case_id, 'family_id': contract['family_id'],
                        'role': contract['role'], 'modality': display, 'accepted': False,
                        'status': 'production_save_reopen_pass_pending_independent_review',
                        'embedded_originals': embedded_count, 'source_intensities_preserved': True,
                        'contact_sheet': contact, 'preparation_disposition': disposition,
                        'elapsed_seconds_export_or_revalidation': time.monotonic() - started,
                        'peak_process_rss_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                        'acquisition_stage': 'unknown clinical imaging; no surgical stage established',
                        'CTA_MRA_acquisition_interval': 'unknown; source patient identity pairing only'})
        if display_proof:
            outputs[-1]['display_revision_array_proof'] = str(safe_path(root, CATALOG + f'/{subject}-display-v2-array-proof.json').relative_to(root))
        gc.collect()
    result = {'schema': 'ressectionlab-topbrain-native-candidates-v1', 'subject': subject,
              'family': receipt['family'], 'candidates': outputs, 'accepted': False}
    atomic_json(safe_path(root, CATALOG + f'/{subject}-native-candidates.json'), result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['inspect', 'extract', 'remote-plan', 'remote-extract', 'export', 'batch'])
    parser.add_argument('--data-root', type=Path, default=DEFAULT_ROOT)
    parser.add_argument('--subject', default='001')
    parser.add_argument('--review-receipt', type=Path)
    parser.add_argument('--count', type=int, default=25)
    args = parser.parse_args()
    result = (inspect(args.data_root) if args.command == 'inspect' else
              extract(args.data_root, args.subject) if args.command == 'extract' else
              remote_plan(args.data_root) if args.command == 'remote-plan' else
              remote_extract(args.data_root, args.subject) if args.command == 'remote-extract' else
              export_native(args.data_root, args.subject) if args.command == 'export' else batch(args.data_root, args.review_receipt, args.count))
    print(json.dumps({k: v for k, v in result.items() if k not in {'members', 'metadata'}}, indent=2))

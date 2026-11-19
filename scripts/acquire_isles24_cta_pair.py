#!/usr/bin/env python3
"""Acquire one publisher-paired acute NCCT, registered CTA and CoW from ISLES'24 v7."""

from __future__ import annotations

import argparse
from contextlib import ExitStack
from dataclasses import dataclass, replace
import fcntl
import hashlib
import json
import lzma
import os
from pathlib import Path
import zlib

from data_rebuild_acquire import DEFAULT_ROOT, UUID, request_range
from resectionlab.data_root import atomic_json, checked_root, safe_path, sha256_file


PUBLISHER = 'https://zenodo.org/records/17652035'
ARCHIVE = 'https://zenodo.org/api/records/17652035/files/train.7z/content'
ARCHIVE_SIZE = 99_022_114_670
ARCHIVE_MD5 = '4959a5dd2438d53e3c86d6858484e781'
METADATA_RANGES = ((0, 32), (99_022_114_629, 41), (99_022_073_125, 41_504))
METADATA_SHA256 = '6b7391ad331978451c9254fbc639e3f4e18b80e61a13e608a93e5dd610c7baaf'
LICENSE = 'CC-BY-NC-SA-4.0'
SUBJECT = 'sub-stroke0146'
ACUTE = 'ses-01'
MIB = 1024**2
HARD_CAP = 700 * MIB
REQUEST_CHUNK = 16 * MIB
IO_CHUNK = MIB
DESTINATION = Path('sources/isles24-v7/acute-pair-selective') / SUBJECT
STAGING = Path('staging/isles24-v7-acute-pair-selective') / SUBJECT


@dataclass(frozen=True)
class Member:
    role: str
    archive_tree: str
    suffix: str
    offset: int
    size: int
    crc32: int

    @property
    def name(self):
        return Path('train') / self.archive_tree / SUBJECT / ACUTE / f'{SUBJECT}_{ACUTE}_{self.suffix}.nii.gz'


@dataclass(frozen=True)
class Stream:
    index: int
    archive_offset: int
    max_prefix: int
    members: tuple[Member, ...]

    @property
    def selected_end(self):
        return max(member.offset + member.size for member in self.members)


# All offsets, lengths, CRCs and the LZMA2 16 MiB dictionary are read from the
# exact publisher archive index, independently verified by METADATA_SHA256.
STREAMS = (
    Stream(24, 94_568_961_211, 139 * MIB, (
        Member('acquired_ncct', 'raw_data', 'ncct', 57_074_611, 15_675_115, 0x8DA5FF8D),
    )),
    Stream(16, 62_132_192_109, 560 * MIB, (
        Member('publisher_automated_cow', 'derivatives', 'space-ncct_cow-msk', 452_530_037, 101_761, 0x9D9D6100),
        Member('publisher_registered_cta', 'derivatives', 'space-ncct_cta', 452_631_798, 16_572_970, 0xEB48EC52),
    )),
)
NATIVE_CTA = Member('acquired_native_cta', 'raw_data', 'cta', 4_210_756_388, 39_565_983, 0x3503900D)
NATIVE_CTA_STREAM = 23
NATIVE_CTA_STREAM_END = 94_568_961_211  # Exclusive packed endpoint, from index.
NATIVE_CTA_SUFFIX = 96 * MIB
RESET_SPACING = 64 * MIB


class TransferBudget:
    """Keep every requested byte reserved across process restarts and retries."""

    def __init__(self, path, *, prior_bytes=0, core_sha256=None):
        self.path = path
        self.data = json.loads(path.read_text()) if path.exists() else {
            'archive_metadata_sha256': METADATA_SHA256, 'requested_bytes': prior_bytes,
            'metadata_verified': bool(core_sha256), 'streams': {},
            'base_acquisition_sha256': core_sha256,
        }
        if (self.data['archive_metadata_sha256'] != METADATA_SHA256
                or self.data.get('base_acquisition_sha256') != core_sha256
                or not prior_bytes <= self.data['requested_bytes'] <= HARD_CAP):
            raise ValueError('ISLES transfer state differs from the pinned release or budget')

    def save(self):
        atomic_json(self.path, self.data)

    def reserve(self, length):
        if not 0 < length <= REQUEST_CHUNK or length > self.remaining:
            raise ValueError('Next range would exceed the remaining ISLES transport budget')
        self.data['requested_bytes'] += length
        self.save()

    @property
    def remaining(self):
        return HARD_CAP - self.data['requested_bytes']

    def open_range(self, first, length):
        self.reserve(length)
        response = request_range(ARCHIVE, first, first + length - 1, ARCHIVE_SIZE)
        if (response.headers.get('Content-Encoding', 'identity') != 'identity'
                or int(response.headers.get('Content-Length', length)) != length):
            response.close()
            raise ValueError('Server returned a transformed or wrong-length ISLES archive range')
        return response


def verify_archive_index(budget):
    if budget.data['metadata_verified']:
        return
    sha = hashlib.sha256()
    for first, size in METADATA_RANGES:
        with budget.open_range(first, size) as response:
            data = response.read(size)
        if len(data) != size:
            raise ValueError('Archive metadata response differs from its bounded range')
        sha.update(data)
    if sha.hexdigest() != METADATA_SHA256:
        raise ValueError('Publisher 7z index differs from the pinned v7 source')
    budget.data['metadata_verified'] = True
    budget.save()


def packed_prefix(root, stream, cache, budget):
    """Replay cached bytes, then append exact HTTPS chunks from the same release."""
    offset = cache.stat().st_size if cache.exists() else 0
    if offset > stream.max_prefix:
        raise ValueError('Retained 7z transfer prefix exceeds its stream budget')
    if offset:
        with cache.open('rb') as source:
            while piece := source.read(IO_CHUNK):
                yield piece
    while offset < stream.max_prefix and budget.remaining:
        size = min(REQUEST_CHUNK, stream.max_prefix - offset, budget.remaining)
        checked_root(root, required_bytes=size, expected_uuid=UUID)
        with budget.open_range(stream.archive_offset + offset, size) as response:
            # Read a complete reserved range before checkpointing it. A failed
            # response keeps prior chunks intact and still counts to the cap.
            piece = response.read(size)
        if len(piece) != size:
            raise ValueError('Truncated or oversized ISLES packed range response')
        checked_root(root, required_bytes=size, expected_uuid=UUID)
        with cache.open('ab') as output:
            output.write(piece)
            output.flush()
            os.fsync(output.fileno())
        offset += size
        yield piece


def selected_bytes(packed, stream, outputs):
    decoder = lzma.LZMADecompressor(format=lzma.FORMAT_RAW, filters=[
        {'id': lzma.FILTER_LZMA2, 'dict_size': 16 * MIB},
    ])
    position = 0
    for piece in packed:
        while True:
            data = decoder.decompress(piece, max_length=IO_CHUNK)
            piece = b''
            stop = position + len(data)
            for member, output in outputs.items():
                first = max(position, member.offset)
                last = min(stop, member.offset + member.size)
                if first < last:
                    output.write(data[first - position:last - position])
            position = stop
            if position >= stream.selected_end:
                return
            if decoder.eof:
                raise ValueError('Solid stream ended before a selected archive member')
            if decoder.needs_input:
                break
    raise ValueError('Selected ISLES members were not reached within the transport budget')


def verify_member(path, member):
    sha = hashlib.sha256()
    crc = 0
    with path.open('rb') as source:
        while piece := source.read(IO_CHUNK):
            sha.update(piece)
            crc = zlib.crc32(piece, crc)
    if path.stat().st_size != member.size or crc != member.crc32:
        raise ValueError(f'ISLES archive member size or CRC32 mismatch: {member.name}')
    return {
        'role': member.role, 'path': str((DESTINATION / member.name)),
        'archive_member': str(member.name), 'archive_member_crc32': f'{crc:08x}',
        'bytes': member.size, 'sha256': sha.hexdigest(),
    }


def acquire_stream(root, stream, budget):
    cache = safe_path(root, STAGING / f'stream-{stream.index}.packed.partial')
    destinations = {member: safe_path(root, DESTINATION / member.name) for member in stream.members}
    missing = [member for member in stream.members if not destinations[member].exists()]
    if missing:
        temporary = {member: safe_path(root, STAGING / f'{member.role}.partial') for member in missing}
        with ExitStack() as stack:
            outputs = {member: stack.enter_context(path.open('wb')) for member, path in temporary.items()}
            selected_bytes(packed_prefix(root, stream, cache, budget), stream, outputs)
        for member, path in temporary.items():
            verify_member(path, member)
        checked_root(root, expected_uuid=UUID)
        for member, path in temporary.items():
            destinations[member].parent.mkdir(parents=True, exist_ok=True)
            os.replace(path, destinations[member])
    records = [verify_member(path, member) for member, path in destinations.items()]
    info = {
        'index': stream.index, 'archive_first_byte': stream.archive_offset,
        'decoded_member_end': stream.selected_end,
        'packed_prefix_transferred_bytes': cache.stat().st_size if cache.exists() else 0,
    }
    budget.data['streams'].setdefault(str(stream.index), info)
    budget.save()
    cache.unlink(missing_ok=True)
    return records


def grid_details(image, reference):
    import nibabel as nib
    import numpy as np

    if (len(image.shape) != 3 or not np.isfinite(image.affine).all()
            or abs(np.linalg.det(image.affine[:3, :3])) < 1e-8):
        raise ValueError('Invalid native NIfTI grid')
    qform, qc = image.get_qform(coded=True)
    sform, sc = image.get_sform(coded=True)
    error = float(np.max(np.abs(image.affine - reference.affine)))
    return {
        'shape': list(image.shape), 'affine_ras': image.affine.tolist(),
        'spacing_mm': nib.affines.voxel_sizes(image.affine).tolist(),
        'axis_codes': list(nib.aff2axcodes(image.affine)),
        'units': list(image.header.get_xyzt_units()), 'qform_code': int(qc), 'sform_code': int(sc),
        'qform_sform_max_coefficient_difference': float(np.max(np.abs(qform - sform))) if qc and sc else None,
        'max_affine_coefficient_difference_from_ncct': error,
        'exact_grid_match_to_ncct': image.shape == reference.shape and error <= 1e-5,
    }


def check_grids(root, records):
    import nibabel as nib
    import numpy as np

    images = {row['role']: nib.load(safe_path(root, row['path'])) for row in records}
    reference = images['acquired_ncct']
    results = {role: grid_details(image, reference) for role, image in images.items()}
    mask = np.asanyarray(images['publisher_automated_cow'].dataobj)
    values, counts = np.unique(mask, return_counts=True)
    # Pin the observed labels for this exact subject without reassigning values
    # or treating other versions of the TopCoW label map as interchangeable.
    if set(values) != {0, 1, 2, 3, 4, 5, 6, 7, 11, 12}:
        raise ValueError('ISLES CoW labels differ from the pinned v7 subject mask')
    results['publisher_automated_cow']['nonzero_label_voxel_counts'] = {
        str(int(value)): int(count) for value, count in zip(values, counts) if value > 0
    }
    return results


def acquire(root):
    retained_capacity = sum(member.size for stream in STREAMS for member in stream.members)
    root = checked_root(root, required_bytes=HARD_CAP + retained_capacity, expected_uuid=UUID)
    dest = safe_path(root, DESTINATION)
    dest.mkdir(parents=True, exist_ok=True)
    staging = safe_path(root, STAGING)
    receipt_path = safe_path(root, DESTINATION / 'acquisition.json')
    state_path = safe_path(root, STAGING / 'transfer-state.json')
    with safe_path(root, DESTINATION / '.acquisition.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if receipt_path.exists():
            receipt = json.loads(receipt_path.read_text())
            if (receipt['archive_metadata_sha256'] != METADATA_SHA256
                    or len(receipt['files']) != sum(len(stream.members) for stream in STREAMS)
                    or any(sha256_file(safe_path(root, row['path'])) != row['sha256'] for row in receipt['files'])):
                raise ValueError('Retained ISLES source differs from its v7 receipt')
            state_path.unlink(missing_ok=True)
            for stream in STREAMS:
                safe_path(root, STAGING / f'stream-{stream.index}.packed.partial').unlink(missing_ok=True)
            return receipt, 0
        staging.mkdir(parents=True, exist_ok=True)
        budget = TransferBudget(state_path)
        before = budget.data['requested_bytes']
        verify_archive_index(budget)
        files = []
        for stream in STREAMS:
            print(f'ISLES {SUBJECT} solid stream {stream.index}, selecting {[member.role for member in stream.members]}', flush=True)
            files.extend(acquire_stream(root, stream, budget))
        grids = check_grids(root, files)
        receipt = {
            'schema': 'ressectionlab-isles24-v7-acute-source-pair-v1',
            'dataset': "ISLES'24 – A Real-World Longitudinal Multimodal Stroke Dataset",
            'publisher_release': PUBLISHER, 'version': 'v7', 'doi': '10.5281/zenodo.17652035',
            'license': LICENSE, 'publisher_archive_url': ARCHIVE,
            'publisher_archive_bytes': ARCHIVE_SIZE, 'publisher_archive_md5': ARCHIVE_MD5,
            'whole_archive_md5_verified': False, 'archive_metadata_sha256': METADATA_SHA256,
            'source_person_family': f'ISLES24-v7:{SUBJECT}', 'role': 'DEVELOPMENT',
            'source_timepoint': 'acute admission ses-01',
            'surgical_timepoint': 'unspecified by publisher', 'defaced': True,
            'registered_cta_origin': 'publisher linear registration to acquired NCCT frame',
            'cow_annotation_origin': 'publisher automated derived Circle of Willis labels from CTA',
            'files': files, 'grids': grids,
            'same_grid': all(row['exact_grid_match_to_ncct'] for row in grids.values()),
            'native_cta': 'outside bounded packed prefix acquisition',
            'transfer_limit_bytes': HARD_CAP, 'requested_transport_bytes': budget.data['requested_bytes'],
            'solid_streams': list(budget.data['streams'].values()),
            'integrity_scope': 'Pinned exact-range 7z index, 7z member CRC32 and local SHA-256',
        }
        checked_root(root, expected_uuid=UUID)
        atomic_json(receipt_path, receipt)
        state_path.unlink()
        if not any(staging.iterdir()):
            staging.rmdir()
        return receipt, budget.data['requested_bytes'] - before


def preceding_frames(data, end):
    """Yield LZMA2 framing candidates ending at this packed boundary."""
    import numpy as np

    first = max(0, end - (65_536 + 6))
    count = end - first - 3
    if count <= 0:
        return
    control = data[first:first + count]
    length = np.arange(end - first, end - first - count, -1, dtype=np.int32)
    unpacked_hi = data[first + 1:first + count + 1].astype(np.int32) << 8
    unpacked = unpacked_hi + data[first + 2:first + count + 2] + 1
    raw = ((control == 1) | (control == 2)) & (length == 3 + unpacked)
    for index in np.flatnonzero(raw):
        yield first + int(index), int(unpacked[index])

    control = control[:-2]
    count = len(control)
    packed_hi = data[first + 3:first + count + 3].astype(np.int32) << 8
    packed = packed_hi + data[first + 4:first + count + 4] + 1
    compressed = (control >= 128) & (length[:count] == 5 + (control >= 192) + packed)
    for index in np.flatnonzero(compressed):
        yield first + int(index), ((int(control[index]) & 31) << 16) + int(unpacked[index])


def plausible_reset_frames(data, stream_size, member_size, *, reset_spacing=RESET_SPACING):
    """Walk from the real stream EOF toward periodic independent dictionary resets."""
    if len(data) < 2 or data[-1] != 0:
        raise ValueError('Packed CTA suffix does not end at the indexed LZMA2 stream boundary')
    pending = [(len(data) - 1, 0)]
    visited = set()
    while pending and len(visited) < 25_000:
        end, suffix_length = pending.pop()
        if (end, suffix_length) in visited:
            continue
        visited.add((end, suffix_length))
        for start, decoded in preceding_frames(data, end):
            length = suffix_length + decoded
            if length > stream_size:
                continue
            control = data[start]
            if ((control == 1 or control >= 224) and length >= member_size
                    and (stream_size - length) % reset_spacing == 0):
                yield start, length
            pending.append((start, length))


def extract_native_cta_suffix(cache, temporary):
    import numpy as np

    data = np.memmap(cache, mode='r', dtype=np.uint8)
    unpacked_stream_length = NATIVE_CTA.offset + NATIVE_CTA.size
    for start, decoded_length in plausible_reset_frames(data, unpacked_stream_length, NATIVE_CTA.size):
        member = replace(NATIVE_CTA, offset=decoded_length - NATIVE_CTA.size)
        stream = Stream(NATIVE_CTA_STREAM, 0, NATIVE_CTA_SUFFIX, (member,))
        with cache.open('rb') as source, temporary.open('wb') as output:
            source.seek(start)
            try:
                selected_bytes(iter(lambda: source.read(IO_CHUNK), b''), stream, {member: output})
            except (lzma.LZMAError, ValueError):
                continue
        # Packet framing can yield false matches inside compressed data. Only
        # the complete publisher member's exact size and CRC can accept a reset.
        try:
            record = verify_member(temporary, NATIVE_CTA)
        except ValueError:
            continue
        return record, {
            'packed_dictionary_reset_byte': NATIVE_CTA_STREAM_END - NATIVE_CTA_SUFFIX + start,
            'decoded_dictionary_reset_offset': unpacked_stream_length - decoded_length,
            'packed_through_stream_end_bytes': NATIVE_CTA_SUFFIX - start,
        }
    raise ValueError('The bounded packed suffix yielded no CRC-valid native CTA member')


def extend_native_cta(root, base):
    import nibabel as nib

    base_path = safe_path(root, DESTINATION / 'acquisition.json')
    base_sha256 = sha256_file(base_path)
    receipt_path = safe_path(root, DESTINATION / 'native-cta-extension.json')
    state_path = safe_path(root, DESTINATION / '.native-cta-attempt.json')
    path = safe_path(root, DESTINATION / NATIVE_CTA.name)
    staging = safe_path(root, STAGING)
    cache = safe_path(root, STAGING / 'native-cta.packed.partial')
    temporary = safe_path(root, STAGING / 'native-cta.nii.gz')
    prior_bytes = base['requested_transport_bytes']
    with safe_path(root, DESTINATION / '.acquisition.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if receipt_path.exists():
            result = json.loads(receipt_path.read_text())
            if (result['base_acquisition_sha256'] != base_sha256
                    or result['native_cta'] != verify_member(path, NATIVE_CTA)):
                raise ValueError('Existing native CTA extension differs from its bound acquisition')
            for scratch in (temporary, cache, state_path):
                scratch.unlink(missing_ok=True)
            if staging.exists() and not any(staging.iterdir()):
                staging.rmdir()
            return result
        budget = TransferBudget(state_path, prior_bytes=prior_bytes, core_sha256=base_sha256)
        if 'attempt_error' in budget.data:
            raise ValueError(f"The previous bounded native CTA attempt failed: {budget.data['attempt_error']}")
        cached = cache.stat().st_size if cache.exists() else 0
        if NATIVE_CTA_SUFFIX - cached > budget.remaining:
            raise ValueError('Native CTA trial would exceed the remaining 700 MiB ceiling')
        checked_root(root, required_bytes=NATIVE_CTA_SUFFIX - cached + NATIVE_CTA.size, expected_uuid=UUID)
        staging.mkdir(parents=True, exist_ok=True)
        packed_start = NATIVE_CTA_STREAM_END - NATIVE_CTA_SUFFIX
        stream = Stream(NATIVE_CTA_STREAM, packed_start, NATIVE_CTA_SUFFIX, (NATIVE_CTA,))
        for _ in packed_prefix(root, stream, cache, budget):
            pass
        if cache.stat().st_size != NATIVE_CTA_SUFFIX:
            raise ValueError('Native CTA suffix transfer did not reach its fixed packed endpoint')
        try:
            record, framing = extract_native_cta_suffix(cache, temporary)
            ncct = next(entry for entry in base['files'] if entry['role'] == 'acquired_ncct')
            grids = grid_details(nib.load(temporary), nib.load(safe_path(root, ncct['path'])))
        except (ValueError, nib.filebasedimages.ImageFileError) as error:
            budget.data['attempt_error'] = str(error)
            budget.save()
            temporary.unlink(missing_ok=True)
            cache.unlink(missing_ok=True)
            raise
        result = {
            'schema': 'ressectionlab-isles24-v7-native-cta-extension-v1',
            'base_acquisition_path': str(base_path.relative_to(root)), 'base_acquisition_sha256': base_sha256,
            'source_person_family': base['source_person_family'], 'source_timepoint': base['source_timepoint'],
            'publisher_release': PUBLISHER, 'license': LICENSE,
            'native_cta': record, 'native_grid': grids, 'publisher_native_unregistered': True,
            'native_cta_packed_suffix': {'first': packed_start, 'bytes': NATIVE_CTA_SUFFIX, **framing},
            'transfer_limit_bytes': HARD_CAP, 'base_requested_transport_bytes': prior_bytes,
            'requested_transport_bytes': budget.data['requested_bytes'],
        }
        checked_root(root, expected_uuid=UUID)
        if path.exists():
            if verify_member(path, NATIVE_CTA) != record:
                raise ValueError('Existing native CTA differs from the verified suffix')
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            os.replace(temporary, path)
        atomic_json(receipt_path, result)
        for scratch in (temporary, cache, state_path):
            scratch.unlink(missing_ok=True)
        if not any(staging.iterdir()):
            staging.rmdir()
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, default=Path(os.environ.get('RESSECTIONLAB_DATA_ROOT') or DEFAULT_ROOT))
    parser.add_argument('--native-cta', action='store_true', help='Try the separate native CTA suffix after the core receipt is verified.')
    args = parser.parse_args()
    receipt, received = acquire(args.data_root)
    location = args.data_root / DESTINATION / 'acquisition.json'
    print(json.dumps({'receipt': str(location), 'new_reserved_transport_bytes': received,
                      'total_reserved_transport_bytes': receipt['requested_transport_bytes'],
                      'same_grid': receipt['same_grid'], 'grids': receipt['grids']}), flush=True)
    if args.native_cta:
        root = checked_root(args.data_root, expected_uuid=UUID)
        print(json.dumps(extend_native_cta(root, receipt)), flush=True)


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""Run the TopBrain 2025 top-three segmentation images natively, without a container runtime.

`unpack` streams one saved image archive once, verifies its MD5 against the
Zenodo record, inventories every layer member and extracts only application
code, weights and package metadata to the data root. Nothing from the images is
copied into the checkout. Later commands load the extracted weights with the
local nnU-Net runtime and keep soft class probabilities.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import os
import re
from pathlib import Path, PurePosixPath
import shutil
import tarfile
import time
import zipfile

import nibabel as nib
import numpy as np
from scipy import ndimage as ndi

from resectionlab.data_root import ROOT_ENV, atomic_json, checked_root, safe_path, sha256_file
from resectionlab.vascular_parity import agreement, betti, dice, floating_components
from resectionlab.vessel_ensemble import (
    COMBINERS, ConnectedAddition, SoftSegmentation, average, combine, named_label_map, require_same_grid,
    vessel_groups, vessel_probability,
)

HDD_ROOT = Path('/Volumes/shafrir/old coding projects/RessectionLabData')
HDD_UUID = '9CC3623C-E38B-3EEE-B6BB-B9A06C193700'
RECORD = 'cache/models/topbrain-2025-dockers-zenodo-20158639'
UNPACKED = 'cache/models/topbrain-2025-native'
ARCHIVES = {
    'UZH-ct-mr': ('Team_UZH_2025_topbrain_segmentation_ct_mr.tar.gz', '7d04086c75bdd459f4a8af44e753be0a'),
    'ARG-ct': ('Team_ARG_2025_topbrain_segmentation_ct.tar.gz', '35d5434f91a274456f72f428fce067e0'),
    'ARG-mr': ('Team_ARG_2025_topbrain_segmentation_mr.tar.gz', 'e1fe74d1707907918b1b91002962f40f'),
    'KDH-ct': ('Team_KDH_2025_topbrain_segmentation_ct.tar.gz', '4b71fe691b99e8a76cd0d83ebcf2da95'),
    'KDH-mr': ('Team_KDH_2025_topbrain_segmentation_mr.tar.gz', '0224662747a594f5bc17932f5c85c313'),
}
SYSTEM = {'bin', 'boot', 'dev', 'etc', 'lib', 'lib32', 'lib64', 'libx32', 'media', 'mnt', 'proc', 'run', 'sbin',
          'srv', 'sys', 'tmp', 'usr', 'var'}
WEIGHTS = ('.pth', '.pt', '.ckpt', '.onnx', '.safetensors', '.bin', '.model', '.pkl', '.npz', '.h5')
PACKAGE_METADATA = ('METADATA', 'LICENSE', 'LICENSE.txt', 'LICENSE.md', 'COPYING', 'NOTICE', 'direct_url.json',
                    'top_level.txt', 'RECORD')
BLOCK = 8 * 1024**2
READ_BUFFER = 64 * 1024**2
SCRATCH = '.network-slices.tmp'   # beside a stage's output while a large scan is predicted


class HashingReader:
    def __init__(self, stream):
        self.stream, self.md5, self.bytes = stream, hashlib.md5(), 0

    def read(self, size=-1):
        block = self.stream.read(size)
        self.md5.update(block)
        self.bytes += len(block)
        return block


def wanted(path, packages):
    """Application files, weights anywhere, package metadata and licences, and named packages in full."""
    parts = PurePosixPath(path).parts
    if not parts or any(part in ('..', '') for part in parts) or parts[-1].startswith('.wh.'):
        return False
    lowered = path.lower()
    in_packages = 'site-packages' in parts or 'dist-packages' in parts
    if in_packages:
        index = parts.index('site-packages' if 'site-packages' in parts else 'dist-packages')
        package = parts[index + 1] if len(parts) > index + 1 else ''
        if package.endswith(('.dist-info', '.egg-info')):
            return parts[-1] in PACKAGE_METADATA or 'licenses' in parts
        if package.endswith('.egg-link') or package.endswith('.pth'):
            return True
        return package in packages or parts[-1].upper().startswith(('LICENSE', 'COPYING', 'NOTICE'))
    if lowered.endswith(WEIGHTS):
        return True
    if parts[0] in SYSTEM or parts[0] == 'opt' and len(parts) > 1 and parts[1] in ('conda', 'nvidia'):
        return False
    return not ('/.cache/' in lowered or '__pycache__' in parts)


def unpack(root, name, packages=()):
    filename, expected = ARCHIVES[name]
    archive = safe_path(root, f'{RECORD}/{filename}')
    target = safe_path(root, f'{UNPACKED}/{name}')
    layers = target / 'layers'
    layers.mkdir(parents=True, exist_ok=True)
    inventory, extracted, manifest, config = [], [], None, {}
    started = time.monotonic()
    # Large sequential reads keep the external disk from seeking between small requests.
    with archive.open('rb', buffering=READ_BUFFER) as raw:
        reader = HashingReader(raw)
        with gzip.GzipFile(fileobj=reader) as unzipped, tarfile.open(fileobj=unzipped, mode='r|', bufsize=BLOCK) as outer:
            for member in outer:
                if not member.isfile():
                    continue
                if member.name == 'manifest.json':
                    manifest = json.load(outer.extractfile(member))
                    continue
                stream = outer.extractfile(member)
                if member.size < 4 * 1024**2:
                    content = stream.read()
                    try:
                        config[member.name] = json.loads(content)
                        continue
                    except (UnicodeDecodeError, ValueError):
                        stream = io.BytesIO(content)
                layer = member.name.replace('/', '_')
                try:
                    inner = tarfile.open(fileobj=stream, mode='r|*')
                except tarfile.ReadError:
                    continue
                with inner:
                    for item in inner:
                        path = item.name[2:] if item.name.startswith('./') else item.name.lstrip('/')
                        inventory.append([layer, path, item.size, item.type.decode() if item.type else '0'])
                        if not item.isfile() or not wanted(path, packages):
                            continue
                        destination = layers / layer / path
                        destination.parent.mkdir(parents=True, exist_ok=True)
                        with inner.extractfile(item) as source, destination.open('wb') as sink:
                            shutil.copyfileobj(source, sink, BLOCK)
                        extracted.append([layer, path, item.size])
                print(f'{name}: layer {layer} done, {len(extracted)} files kept, '
                      f'{reader.bytes / 1024**3:.2f} GB read', flush=True)
        while reader.read(BLOCK):
            pass
    md5 = reader.md5.hexdigest()
    with gzip.open(target / 'inventory.tsv.gz', 'wt') as stream:
        for row in inventory:
            stream.write('\t'.join(str(value) for value in row) + '\n')
    atomic_json(target / 'unpack.json', {
        'archive': f'{RECORD}/{filename}', 'archive_bytes': reader.bytes, 'md5': md5, 'record_md5': expected,
        'md5_matches_record': md5 == expected, 'seconds': round(time.monotonic() - started, 1),
        'manifest': manifest, 'configs': config, 'packages_extracted_in_full': list(packages),
        'members': len(inventory), 'extracted': extracted,
    })
    if md5 != expected:
        raise SystemExit(f'{name}: MD5 {md5} differs from the record {expected}')
    merge(target)
    return target


def verified_rootfs(root, image):
    target = safe_path(root, f'{UNPACKED}/{image}')
    if not json.loads((target / 'unpack.json').read_text()).get('md5_matches_record'):
        raise SystemExit(f'{image}: the archive MD5 does not match the Zenodo record')
    return target / 'rootfs'


def merge(target):
    """Apply the kept layer files in manifest order into rootfs/, honouring whiteouts."""
    record = json.loads((target / 'unpack.json').read_text())
    if not record.get('md5_matches_record'):
        raise SystemExit(f'{target.name}: the archive MD5 does not match the Zenodo record')
    order = [layer.replace('/', '_') for layer in record['manifest'][0]['Layers']]
    deleted = {}
    with gzip.open(target / 'inventory.tsv.gz', 'rt') as stream:
        for line in stream:
            layer, path = line.split('\t')[:2]
            name = PurePosixPath(path)
            if name.name.startswith('.wh.'):
                deleted.setdefault(layer, []).append(str(name.parent / name.name[4:]) if name.name != '.wh..wh..opq'
                                                     else str(name.parent) + '/')
    rootfs = target / 'rootfs'
    by_layer = {}
    for layer, path, _ in record['extracted']:
        by_layer.setdefault(layer, []).append(path)
    for layer in order:
        for gone in deleted.get(layer, []):
            victim = rootfs / gone.rstrip('/')
            if gone.endswith('/') and victim.is_dir():
                for child in victim.iterdir():
                    shutil.rmtree(child, ignore_errors=True) if child.is_dir() else child.unlink(missing_ok=True)
            elif victim.is_dir():
                shutil.rmtree(victim, ignore_errors=True)
            else:
                victim.unlink(missing_ok=True)
        for path in by_layer.get(layer, []):
            source, destination = target / 'layers' / layer / path, rootfs / path
            if source.is_file():
                destination.parent.mkdir(parents=True, exist_ok=True)
                os.replace(source, destination)
    shutil.rmtree(target / 'layers', ignore_errors=True)
    return rootfs



TOPBRAIN = ('sources/topbrain/selective/{case}/TopBrain_Data_Release_Batches1n2nTA36_081726/'
            '{folder}/topcow_{modality}_{case}{suffix}.nii.gz')
EDEN = 'prepared/Eden 2020 ResectionLab Projects (restricted)/{patient}/originals/{digest}/{name}'
CASES = {
    **{f'TopCoW-{case}-{modality}': {
        'modality': modality,
        'image': TOPBRAIN.format(case=case, folder='imagesTr_topbrain', modality=modality, suffix='_0000'),
        'reference': TOPBRAIN.format(case=case, folder=f'labelsTr_topbrain_v1_{modality}', modality=modality, suffix=''),
    } for case in ('001', '014', '018') for modality in ('ct', 'mr')},
    'PTNT_4509-mr': {'modality': 'mr', 'reference': None, 'image': EDEN.format(
        patient='PTNT_4509', digest='f607a41468e1fa01c4a20fbef84ec5dc8b3b6a20a24bf3b31712630ebdabed9b', name='scan.nii.gz')},
    'PTNT_9937-mr': {'modality': 'mr', 'reference': None, 'image': EDEN.format(
        patient='PTNT_9937', digest='5e1add34652f68f8a687f7ff815d51fee6224258ffbccfa8c5cccb62997fdab4',
        name='PTNT_9937_s3DI_MC_HR.nii.gz')},
}
PHASE = 'qc/medivis-parity-20261125/phase2-segmentation'
WORK = f'{PHASE}/work'
# Any other prepared study (registered with --study) keeps its results here, apart from the parity evidence.
STORE = 'cache/vessel-ensembles'
RULE = Path(__file__).resolve().parents[1] / 'manifests' / 'vascular-ensemble-rule.json'


def kept_folder(root, case):
    """Where a case's retained team and ensemble folders live."""
    return safe_path(root, f"{CASES[case]['store']}/cases/{case}" if CASES[case].get('store') else f'{PHASE}/cases/{case}')


def made_from(case):
    """What a result of this case must have been made from: the image path, and for a registered study the
    content hash and series of that file."""
    return {key: CASES[case][key] for key in ('image', 'sha256', 'series') if CASES[case].get(key)}


def require_made_from(record_path, case):
    """Refuse a kept or unfinished result that was made from another file than the case's: a study folder of the
    same name elsewhere, another series of the study, or a study prepared again from a new file. A record is held
    to the hash and series when it has them and to the image path otherwise; an ensemble written before it
    recorded its source is held to the team predictions kept beside it."""
    record_path = Path(record_path)
    record = json.loads(record_path.read_text())
    wanted = made_from(case)

    def same_source(path, made):
        keys = [key for key in ('sha256', 'series') if key in wanted and key in made] or ['image']
        for key in keys:
            if made.get(key) != wanted[key]:
                raise SystemExit(f'{path} was made from another file ({key} {made.get(key)}, from {made.get("image")}) than '
                                 f'{case} now names ({wanted[key]}, {wanted["image"]}). Results are kept by study folder '
                                 'name: move the old folder aside, or rename one study folder, and run again')

    if 'image' not in record:
        teams = [record_path.parents[1] / team / 'predict.json' for team in record.get('teams', [])]
        teams = [path for path in teams if path.is_file()]
        if not teams and CASES[case].get('store'):
            raise SystemExit(f'{record_path} does not say which file it was made from; move it aside and run again')
        for path in teams:
            same_source(path, json.loads(path.read_text()))
        return record
    same_source(record_path, record)
    return record


def work_folder(root, case):
    """Where a case's rebuildable work files live while it runs."""
    return safe_path(root, f"{CASES[case]['store']}/work/{case}" if CASES[case].get('store') else f'{WORK}/{case}')
UZH_MODELS = 'opt/app/trained_models/'
UZH_PARTS = {
    'AM': 'Dataset525_topbrain_AM_case_01_27_wReg/Tr_rot30_DiffClusterSM_TopK_ceW_TopBrainAM_13fgCls_cowPT_LRx0p1__nnUNetPlans__3d_fullres',
    'POS': 'Dataset526_topbrain_POS_case_01_27_wReg/Tr_rot30_DiffClusterSM_TopK_ceW_TopBrainPOS_13fgCls_cowPT_LRx0p1__nnUNetPlans__3d_fullres',
    'CAR-ct': 'Dataset531_topbrain_CAR_8fgCls_150Cases_wReg/Tr_rot30_DiffClusterSM_TopK_ceW_TopBrainCAR_CT_8fgCls_cowPT_LRx0p1__nnUNetPlans__3d_fullres_bigPS',
    'CAR-mr': 'Dataset527_topbrain_CAR_case_01_27_wReg/Tr_rot30_DiffClusterSM_TopK_ceW_TopBrainCAR_16fgCls_cowPT_LRx0p1__nnUNetPlans__3d_fullres',
    'V': 'Dataset528_topbrain_V_case_01_27_wReg/Tr_rot30_DiffClusterSM_TopK_ceW_TopBrainV_6fgCls_cowPT_LRx0p1__nnUNetPlans__3d_fullres',
}
# Each team's pipeline as its image runs it (entry points in the README): UZH merges nnU-Net region models with
# its label maps and drops components of 12 voxels or fewer per label; KDH runs one five-fold nnU-Net (MR: masked by a five-fold binary model); ARG feeds a binary vessel stage into a
# 48-label stage and keeps components above 0.05% of the largest. All tiles step 0.5 with Gaussian weighting.
TEAMS = {
    'UZH': {
        'ct': {'image': 'UZH-ct-mr', 'pipeline': 'regions', 'mapping': 'opt/app/label_jsons/label_mapping_ct.json',
               'regions': (('AM', 'AM'), ('CAR', 'CAR-ct'), ('POS', 'POS'), ('V', 'V')), 'folds': (0, 1),
               'min_component_voxels': 13},
        'mr': {'image': 'UZH-ct-mr', 'pipeline': 'regions', 'mapping': 'opt/app/label_jsons/label_mapping_mr.json',
               'regions': (('AM', 'AM'), ('CAR', 'CAR-mr'), ('POS', 'POS')), 'folds': (0, 1),
               'min_component_voxels': 13},
    },
    'KDH': {
        'ct': {'image': 'KDH-ct', 'pipeline': 'regions', 'folds': (0, 1, 2, 3, 4), 'checkpoint': 'checkpoint_best.pth',
               'regions': (('mulcls', 'opt/app/resources/checkpoint/mulcls'),)},
        'mr': {'image': 'KDH-mr', 'pipeline': 'regions', 'folds': (0, 1, 2, 3, 4), 'checkpoint': 'checkpoint_best.pth',
               'regions': (('mulcls', 'opt/app/resources/checkpoint/mulcls'),),
               'binary_mask': 'opt/app/resources/checkpoint/binary'},
    },
    'ARG': {
        'ct': {'image': 'ARG-ct', 'pipeline': 'two-stage', 'resources': 'opt/app/resources'},
        'mr': {'image': 'ARG-mr', 'pipeline': 'two-stage', 'resources': 'opt/app/resources'},
    },
}
VEIN_LABELS = {'ct': (35, 36, 37, 38, 39, 40), 'mr': ()}
TOPBRAIN_LABELS = {'ct': tuple(range(41)), 'mr': tuple(range(43))}
# ARG's 48 stage-two labels: 1-42 as TopBrain MR, 43-48 the CT veins. Its CT image maps 43-48 onto 35-40 and
# 41-42 to background; its MR image maps 43-48 to background.
ARG_MAPPING = {
    'ct': {**{label: label for label in range(41)}, 41: 0, 42: 0, **{43 + k: 35 + k for k in range(6)}},
    'mr': {**{label: label for label in range(43)}, **{43 + k: 0 for k in range(6)}},
}
# Declared deviation for the soft path that feeds the ensemble: on CT, ARG's 35-42 are external carotid branches
# (ECA, STA, MaxA, MMA), which the CT scheme does not have; the image's own CT map keeps 35-40 and so names them as
# veins. Soft CT probabilities send 35-42 to background instead. The official label map stays as the image makes it.
ARG_SOFT_MAPPING = {
    'ct': {**{label: label for label in range(35)}, **{label: 0 for label in range(35, 43)},
           **{43 + k: 35 + k for k in range(6)}},
    'mr': ARG_MAPPING['mr'],
}


class NpyWriter:
    """Write a channel-first .npy sequentially, one channel at a time (memory maps crawl on exFAT)."""

    def __init__(self, path, shape, dtype=np.float16):
        self.path, self.shape, self.dtype, self.written = Path(path), tuple(shape), np.dtype(dtype), 0
        self.partial = self.path.with_name(self.path.name + '.partial')
        self.stream = self.partial.open('wb', buffering=READ_BUFFER)
        np.lib.format.write_array_header_1_0(self.stream, {'descr': np.lib.format.dtype_to_descr(self.dtype),
                                                           'fortran_order': False, 'shape': self.shape})

    def write(self, channel):
        channel = np.ascontiguousarray(channel, self.dtype)
        if channel.shape != self.shape[1:]:
            raise ValueError('Channel shape differs from the output')
        self.stream.write(channel.tobytes())
        self.written += 1

    def __setitem__(self, channel, values):
        if channel != self.written:
            raise ValueError('Channels must be written in order')
        self.write(values)

    def close(self):
        self.stream.close()
        if self.written != self.shape[0]:
            raise ValueError(f'{self.path.name}: wrote {self.written} of {self.shape[0]} channels')
        os.replace(self.partial, self.path)
        return ChannelArray(self.path)


class ChannelArray:
    """A channel-first .npy read one whole channel per request with large sequential reads; memory maps fault in
    small pages and crawl on the external exFAT disk."""

    def __init__(self, path):
        self.path = Path(path)
        with self.path.open('rb') as stream:
            version = np.lib.format.read_magic(stream)
            read = np.lib.format.read_array_header_1_0 if version == (1, 0) else np.lib.format.read_array_header_2_0
            self.shape, fortran, self.dtype = read(stream)
            self.offset = stream.tell()
        if fortran or len(self.shape) != 4:
            raise ValueError(f'{self.path.name} is not a C-ordered channel-first volume')
        self.ndim, self.channel_items = 4, int(np.prod(self.shape[1:]))

    def __len__(self):
        return self.shape[0]

    def __getitem__(self, channel):
        if not isinstance(channel, (int, np.integer)) or not 0 <= channel < self.shape[0]:
            raise IndexError('Index one channel at a time')
        with self.path.open('rb', buffering=READ_BUFFER) as stream:
            stream.seek(self.offset + int(channel) * self.channel_items * self.dtype.itemsize)
            values = np.fromfile(stream, self.dtype, self.channel_items)
        return values.reshape(self.shape[1:])

    def __iter__(self):
        return (self[channel] for channel in range(self.shape[0]))


class NpzChannels:
    """Channels of a retained compressed `probabilities` array, decompressed one at a time in order (reopened to
    go back), so a 4-9 GB volume never sits in memory."""

    def __init__(self, path, member='probabilities.npy'):
        self.path, self.member, self.stream, self.archive = Path(path), member, None, None
        self._open()

    def _open(self):
        self.close()
        self.archive = zipfile.ZipFile(self.path)
        self.stream = self.archive.open(self.member)
        version = np.lib.format.read_magic(self.stream)
        read = np.lib.format.read_array_header_1_0 if version == (1, 0) else np.lib.format.read_array_header_2_0
        self.shape, fortran, self.dtype = read(self.stream)
        if fortran or len(self.shape) != 4:
            raise ValueError(f'{self.path.name} is not a C-ordered channel-first volume')
        self.ndim, self.position = 4, 0
        self.channel_bytes = int(np.prod(self.shape[1:])) * self.dtype.itemsize

    def close(self):
        for handle in (self.stream, self.archive):
            if handle is not None:
                handle.close()

    def _read(self):
        buffer = bytearray(self.channel_bytes)
        view, done = memoryview(buffer), 0
        while done < self.channel_bytes:
            count = self.stream.readinto(view[done:])
            if not count:
                raise ValueError(f'{self.path.name} ended early')
            done += count
        self.position += 1
        return buffer

    def __getitem__(self, channel):
        if not isinstance(channel, (int, np.integer)) or not 0 <= channel < self.shape[0]:
            raise IndexError('Index one channel at a time')
        if channel < self.position:
            self._open()
        while self.position < channel:
            self._read()
        return np.frombuffer(self._read(), self.dtype).reshape(self.shape[1:])

    def __len__(self):
        return self.shape[0]

    def __iter__(self):
        return (self[channel] for channel in range(self.shape[0]))


def case_folder(root, case, name):
    """A team or ensemble folder: the work copy while it exists, otherwise the retained copy."""
    work = work_folder(root, case) / name
    return work if work.is_dir() else kept_folder(root, case) / name


def source_image(root, case):
    """The acquired image reoriented to LPS storage (as reorient_nii.py would) in SimpleITK's (z, y, x) order, and
    the orientation transform back to the source voxel order. TopBrain images are already LPS; EDEN TOF is not."""
    image = nib.load(safe_path(root, CASES[case]['image']))
    return (image, *lps_view(image))


def lps_view(image):
    start = nib.io_orientation(image.affine)
    lps = image.as_reoriented(nib.orientations.ornt_transform(start, nib.orientations.axcodes2ornt('LPS')))
    values = np.asanyarray(lps.dataobj, dtype=np.float32)
    back = nib.orientations.ornt_transform(nib.orientations.axcodes2ornt('LPS'), start)
    return values.transpose(2, 1, 0), [float(v) for v in lps.header.get_zooms()[:3][::-1]], back


def to_source(channel_zyx, back):
    return nib.orientations.apply_orientation(channel_zyx.transpose(2, 1, 0), back)


def half_precision(network):
    """Float16 weights and activations, as the CUDA images ran under autocast; MPS has no float16 transposed 3D
    convolution, so those layers compute in float32."""
    import torch
    network.half()
    for module in network.modules():
        if isinstance(module, torch.nn.ConvTranspose3d):
            module.float()
            module.register_forward_pre_hook(lambda _, args: (args[0].float(),))
            module.register_forward_hook(lambda _, args, out: out.half())
    network.register_forward_pre_hook(lambda _, args: (args[0].half(),))
    return network


def residual_encoder_large(inputs, outputs):
    """KDH MR plans name the architecture 'ResEncL' without arguments; its trainer builds this fixed residual
    encoder U-Net (the nnssl ResEnc-L layout)."""
    import torch
    from dynamic_network_architectures.architectures.unet import ResidualEncoderUNet
    return ResidualEncoderUNet(
        input_channels=inputs, n_stages=6, features_per_stage=[32, 64, 128, 256, 320, 320], conv_op=torch.nn.Conv3d,
        kernel_sizes=[[3, 3, 3]] * 6, strides=[[1, 1, 1]] + [[2, 2, 2]] * 5, n_blocks_per_stage=[1, 3, 4, 6, 6, 6],
        num_classes=outputs, n_conv_per_stage_decoder=[1, 1, 1, 1, 1], conv_bias=True, norm_op=torch.nn.InstanceNorm3d,
        norm_op_kwargs={'eps': 1e-5, 'affine': True}, nonlin=torch.nn.LeakyReLU, nonlin_kwargs={'inplace': True},
        deep_supervision=False)


def build_predictor(folder, checkpoints, device, step=0.5, mirroring=False, half=True):
    """nnU-Net sliding-window predictor over the given checkpoints, network built from the plans."""
    import torch
    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
    from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
    from nnunetv2.utilities.label_handling.label_handling import determine_num_input_channels
    from nnunetv2.utilities.plans_handling.plans_handler import PlansManager

    folder = Path(folder)
    dataset = json.loads((folder / 'dataset.json').read_text())
    plans = PlansManager(json.loads((folder / 'plans.json').read_text()))
    loaded = []
    for path in checkpoints:
        # Checkpoints are pickles from archives whose MD5 matches the Zenodo record. Keep only what inference needs;
        # optimizer state would otherwise stay resident (about 4 GB for KDH MR).
        checkpoint = torch.load(path, map_location='cpu', weights_only=False)
        loaded.append({key: checkpoint[key] for key in ('network_weights', 'trainer_name', 'init_args',
                                                        'inference_allowed_mirroring_axes') if key in checkpoint})
        del checkpoint
    configuration = plans.get_configuration(loaded[0]['init_args']['configuration'])
    inputs = determine_num_input_channels(plans, configuration, dataset)
    outputs = plans.get_label_manager(dataset).num_segmentation_heads
    if configuration.network_arch_class_name == 'ResEncL':
        network = residual_encoder_large(inputs, outputs)
    else:
        network = nnUNetTrainer.build_network_architecture(
            configuration.network_arch_class_name, configuration.network_arch_init_kwargs,
            configuration.network_arch_init_kwargs_req_import, inputs, outputs, enable_deep_supervision=False)
    for checkpoint in loaded:
        network.load_state_dict(checkpoint['network_weights'], strict=True)
    if half:
        half_precision(network)
    predictor = nnUNetPredictor(tile_step_size=step, use_gaussian=True, use_mirroring=mirroring,
                                perform_everything_on_device=False, device=torch.device(device),
                                verbose=False, verbose_preprocessing=False, allow_tqdm=False)
    predictor.manual_initialization(network, plans, configuration, [c['network_weights'] for c in loaded], dataset,
                                    loaded[0]['trainer_name'], loaded[0].get('inference_allowed_mirroring_axes'))
    labels = sorted(int(v) for v in dataset['labels'].values())
    details = {'folder': str(folder), 'checkpoints': [Path(c).relative_to(folder).as_posix() for c in checkpoints],
               'trainer': loaded[0]['trainer_name'], 'configuration': loaded[0]['init_args']['configuration'],
               'spacing': list(configuration.spacing), 'patch_size': list(configuration.patch_size),
               'tile_step': step, 'gaussian': True, 'mirroring': mirroring, 'labels': labels,
               'network': (f'{configuration.network_arch_class_name} as the team trainer builds it'
                           if configuration.network_arch_class_name == 'ResEncL' else
                           'built by the base nnU-Net trainer from the plans') + '; every checkpoint loaded strictly'}
    return predictor, details


def nnunet_input(predictor, channels_czyx, spacing):
    from nnunetv2.inference.data_iterators import PreprocessAdapterFromNpy
    adapter = PreprocessAdapterFromNpy([channels_czyx], [None], [{'spacing': spacing}], [None], predictor.plans_manager,
                                       predictor.dataset_json, predictor.configuration_manager,
                                       num_threads_in_multithreaded=1)
    batch = next(adapter)
    return batch['data'], batch['data_properties']


def trilinear(values, shape):
    import torch
    import torch.nn.functional as F
    if tuple(values.shape) == tuple(shape):
        return np.asarray(values, np.float32)
    tensor = torch.as_tensor(np.asarray(values, np.float32))[None, None]
    return F.interpolate(tensor, size=[int(v) for v in shape], mode='trilinear', align_corners=False)[0, 0].numpy()


def trilinear_input(predictor, channels_czyx, spacing):
    """ARG's preprocessing: no normalisation in nnU-Net, crop to non-zero, trilinear resampling to the plans spacing."""
    import torch
    from nnunetv2.preprocessing.cropping.cropping import crop_to_nonzero
    from nnunetv2.preprocessing.resampling.default_resampling import compute_new_shape
    plans, configuration = predictor.plans_manager, predictor.configuration_manager
    data = channels_czyx.astype(np.float32).transpose([0, *[i + 1 for i in plans.transpose_forward]])
    original = [spacing[i] for i in plans.transpose_forward]
    properties = {'spacing': spacing, 'shape_before_cropping': data.shape[1:]}
    data, _, bbox = crop_to_nonzero(data, None)
    properties.update(bbox_used_for_cropping=bbox, shape_after_cropping_and_before_resampling=data.shape[1:])
    target = list(configuration.spacing)
    target = [original[0], *target] if len(target) < data.ndim - 1 else target
    shape = compute_new_shape(data.shape[1:], original, target)
    return torch.from_numpy(np.stack([trilinear(channel, shape) for channel in data])), properties


class NetworkProbabilities:
    """Network-space class probabilities held in memory: the logits, softmaxed in place."""

    buffer = 'memory'

    def __init__(self, logits):
        import torch
        for start in range(0, logits.shape[1], 8):
            logits[:, start:start + 8] = torch.softmax(logits[:, start:start + 8].float(), 0).half()
        self.values, self.heads = logits, int(logits.shape[0])

    def channel(self, index):
        return self.values[index].float().numpy()

    def close(self):
        self.values = None


class SliceProbabilities:
    """Network-space class probabilities of a scan too large for memory: a scratch file written one finished
    slice at a time (every class of that slice together) and read back one class at a time, with plain file
    reads because memory maps crawl on the external exFAT disk."""

    buffer = 'drive'

    def __init__(self, path, heads, shape):
        self.path, self.heads, self.shape, self.written = Path(path), int(heads), tuple(int(v) for v in shape), 0
        self.plane = self.shape[1] * self.shape[2] * 2
        self.stream = self.path.open('wb', buffering=READ_BUFFER)

    def write(self, classes):
        """One slice: float16 (classes, y, x)."""
        classes = np.ascontiguousarray(classes, np.float16)
        if classes.shape != (self.heads, *self.shape[1:]):
            raise ValueError('Slice shape differs from the scratch file')
        self.stream.write(classes.tobytes())
        self.written += 1

    def finish(self):
        self.stream.close()
        if self.written != self.shape[0]:
            raise ValueError(f'{self.path.name}: wrote {self.written} of {self.shape[0]} slices')
        return self

    def channel(self, index):
        values = np.empty(self.shape, np.float32)
        with self.path.open('rb', buffering=READ_BUFFER) as stream:
            for position in range(self.shape[0]):
                stream.seek((position * self.heads + index) * self.plane)
                values[position] = np.fromfile(stream, np.float16, self.plane // 2).reshape(self.shape[1:])
        return values

    def close(self):
        if not self.stream.closed:
            self.stream.close()
        self.path.unlink(missing_ok=True)


def source_channels(predictor, probabilities, properties, style):
    """Each network-space class probability resampled to the image, uncropped (background 1) and transposed
    back, in SimpleITK (z, y, x) order. Resampling probabilities rather than logits keeps memory to one
    channel at a time; it can move the argmax only where classes are nearly tied."""
    from acvl_utils.cropping_and_padding.bounding_boxes import bounding_box_to_slice
    plans, configuration = predictor.plans_manager, predictor.configuration_manager
    cropped = tuple(int(v) for v in properties['shape_after_cropping_and_before_resampling'])
    spacing = [properties['spacing'][i] for i in plans.transpose_forward]
    current = configuration.spacing if len(configuration.spacing) == len(cropped) else [spacing[0], *configuration.spacing]
    slicer = bounding_box_to_slice(properties['bbox_used_for_cropping'])
    for channel in range(probabilities.heads):
        values = probabilities.channel(channel)
        if style == 'nnunet':
            values = configuration.resampling_fn_probabilities(values[None], cropped, current, spacing)[0]
        else:
            values = trilinear(values, cropped)
        full = np.full(tuple(properties['shape_before_cropping']), 1.0 if channel == 0 else 0.0, np.float32)
        full[slicer] = values
        yield full.transpose(plans.transpose_backward)


def sliding_window(predictor, data):
    """nnU-Net's Gaussian sliding window with every checkpoint evaluated per tile. Returns the output heads, the
    padded shape, the slices that undo the padding, the tile weight, and the tiles in nnU-Net's order as
    (slicer, float16 mean logits times the weight)."""
    import copy
    import torch
    from acvl_utils.cropping_and_padding.padding import pad_nd_image
    from nnunetv2.inference.sliding_window_prediction import compute_gaussian
    if predictor.use_mirroring:
        raise ValueError('None of the images mirrors at test time')
    networks = []
    for parameters in predictor.list_of_parameters:
        network = copy.deepcopy(predictor.network)
        network.load_state_dict(parameters)
        networks.append(network.to(predictor.device).eval())
    # The float32 parameter copies are not needed once the float16 networks sit on the device.
    predictor.list_of_parameters = []
    patch = tuple(predictor.configuration_manager.patch_size)
    padded, revert = pad_nd_image(data, patch, 'constant', {'value': 0}, True, None)
    gaussian = compute_gaussian(patch, sigma_scale=1. / 8, value_scaling_factor=10, device=torch.device('cpu'))

    def tiles():
        with torch.inference_mode():
            for slicer in predictor._internal_get_sliding_window_slicers(padded.shape[1:]):
                tile = padded[slicer][None].to(predictor.device)
                total = None
                for network in networks:
                    output = network(tile)[0].float()
                    if not torch.isfinite(output).all():
                        raise RuntimeError(f'Non-finite logits from the network at tile {slicer}')
                    total = output if total is None else total + output
                yield slicer, (total / len(networks)).cpu().half() * gaussian

    return predictor.label_manager.num_segmentation_heads, tuple(padded.shape[1:]), revert, gaussian, tiles()


def predict_logits(predictor, data):
    """Every tile's mean over the checkpoints added to one shared float16 buffer in memory. nnU-Net sums
    whole-volume fold predictions instead, which needs a second volume of logits."""
    import torch
    heads, shape, revert, gaussian, tiles = sliding_window(predictor, data)
    logits = torch.zeros((heads, *shape), dtype=torch.half)
    weights = torch.zeros(shape, dtype=torch.half)
    with torch.inference_mode():
        for slicer, weighted in tiles:
            logits[slicer] += weighted
            weights[slicer[1:]] += gaussian
        logits /= weights
        if not torch.isfinite(logits).all():
            raise RuntimeError('Non-finite logits in float16 accumulation')
    return logits[(slice(None), *revert[1:])]


def predict_banded(predictor, data, scratch):
    """The same accumulation for a scan whose logits do not fit in memory. Tiles arrive ordered along the first
    axis, so only one patch depth of logits is held: once the tiles move on, the slices behind them are complete
    and are divided by their weights, checked, softmaxed and written to `scratch`. The arithmetic per voxel, and
    so every value, is the one `predict_logits` and `NetworkProbabilities` give."""
    import torch
    heads, shape, revert, gaussian, tiles = sliding_window(predictor, data)
    depth = gaussian.shape[0]
    kept = [range(*part.indices(size)) for part, size in zip(revert[1:], shape)]
    inside = (slice(None), *revert[2:])
    band = torch.zeros((heads, depth, *shape[1:]), dtype=torch.half)
    weights = torch.zeros((depth, *shape[1:]), dtype=torch.half)
    store = SliceProbabilities(scratch, heads, [len(axis) for axis in kept])
    base = None

    def finish(count):
        """Write the first `count` slices of the band, then move the rest to its start."""
        for start in range(0, count, 8):
            stop = min(start + 8, count)
            part = band[:, start:stop] / weights[start:stop]
            if not torch.isfinite(part).all():
                raise RuntimeError('Non-finite logits in float16 accumulation')
            part = torch.softmax(part.float(), 0).half()
            for offset in range(stop - start):
                if base + start + offset in kept[0]:
                    store.write(part[:, offset][inside].numpy())
        for buffer in (*band, weights):
            buffer[:depth - count] = buffer[count:].clone()
            buffer[depth - count:] = 0

    try:
        with torch.inference_mode():
            for slicer, weighted in tiles:
                first = slicer[1].start
                if first < (base or 0) or (base is None and first) or slicer[1].stop - first != depth:
                    raise RuntimeError('Tiles are not ordered along the first axis')
                base = 0 if base is None else base
                if first > base:
                    finish(first - base)
                    base = first
                local = (slicer[0], slice(0, depth), *slicer[2:])
                band[local] += weighted
                weights[local[1:]] += gaussian
            if base is None or base + depth != shape[0]:
                raise RuntimeError('The tiles do not reach the end of the first axis')
            finish(depth)
        return store.finish()
    except BaseException:
        store.close()
        raise


def memory_buffer_limit():
    """The largest float16 logit buffer kept in memory: two fifths of the machine's physical memory."""
    return int(os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES') * 0.4)


def network_probabilities(predictor, data, scratch, limit=None, band_limit=None):
    """Softmaxed network-space probabilities: in memory when the logits fit under `limit` bytes, banded through
    a scratch file otherwise. Both give the same values. A band that would not fit either is refused
    (`band_limit` raises that bound, for tests)."""
    patch = tuple(predictor.configuration_manager.patch_size)
    shape = [max(size, edge) for size, edge in zip(data.shape[1:], patch)]
    if len(patch) != len(shape):
        raise ValueError('Only 3D configurations are run')
    heads, limit = predictor.label_manager.num_segmentation_heads, memory_buffer_limit() if limit is None else limit
    if 2 * heads * int(np.prod(shape)) <= limit:
        return NetworkProbabilities(predict_logits(predictor, data))
    band = 2 * heads * patch[0] * shape[1] * shape[2]
    if band > max(limit, band_limit or 0):
        raise SystemExit(f'This scan needs {band / 2**30:.1f} GiB for one band of logits ({heads} classes, {patch[0]} x '
                         f'{shape[1]} x {shape[2]}) and this machine allows {limit / 2**30:.1f} GiB: crop the scan in '
                         'plane or use a machine with more memory')
    return predict_banded(predictor, data, scratch)


def run_stage(predictor, data, properties, style, back, output, keep=None):
    """Predict one stage, write its source-order class probabilities, and return any kept (z, y, x) channel."""
    import torch
    started = time.monotonic()
    scratch = Path(output).with_name(Path(output).stem + SCRATCH)
    probabilities = network_probabilities(predictor, data, scratch)
    seconds = time.monotonic() - started
    mps = int(torch.mps.driver_allocated_memory()) if predictor.device.type == 'mps' else None
    del data
    kept, writer = None, None
    try:
        for channel, values in enumerate(source_channels(predictor, probabilities, properties, style)):
            if channel == keep:
                kept = values.copy()
            oriented = to_source(values, back)
            writer = writer or NpyWriter(output, (probabilities.heads, *oriented.shape))
            writer.write(oriented)
    finally:
        probabilities.close()
    if predictor.device.type == 'mps':
        torch.mps.empty_cache()
    return writer.close(), kept, {'network_seconds': round(seconds, 1), 'mps_driver_bytes_after_network': mps,
                                  'logit_buffer': probabilities.buffer}


def argmax_labels(probabilities, labels):
    best = np.full(probabilities.shape[1:], -1, np.float32)
    hard = np.zeros(probabilities.shape[1:], np.uint8)
    for channel, label in enumerate(labels):
        values = np.asarray(probabilities[channel], np.float32)
        higher = values > best
        hard[higher], best[higher] = label, values[higher]
    return hard


def map_channels(probabilities, labels, mapping, whole, output):
    """Soft many-to-one relabelling: each TopBrain label sums the channels the team maps onto it."""
    writer = NpyWriter(output, (len(whole), *probabilities.shape[1:]))
    for target in whole:
        total = np.zeros(probabilities.shape[1:], np.float32)
        for channel, label in enumerate(labels):
            if mapping.get(label) == target:
                total += probabilities[channel]
        writer.write(np.clip(total, 0, 1))
    return writer.close()


def compose_regions(parts, whole, output):
    """Soft TopBrain labels from region models with disjoint label sets: P(background) is the product of the
    parts' background, and each named label takes its part's class probability, renormalised over all named
    labels to 1 - P(background)."""
    shape = parts[0][0].shape[1:]
    background = np.ones(shape, np.float32)
    total = np.zeros(shape, np.float32)
    owner = {}
    for index, (probabilities, labels, mapping) in enumerate(parts):
        background *= probabilities[0]
        for channel, label in enumerate(labels):
            if label:
                if mapping[label] in owner:
                    raise ValueError(f'TopBrain label {mapping[label]} is predicted by two parts')
                owner[mapping[label]] = (index, channel)
                total += probabilities[channel]
    scale = np.divide(1 - background, total, out=np.zeros_like(total), where=total > 0)
    del total
    writer = NpyWriter(output, (len(whole), *shape))
    writer.write(background)
    for label in whole[1:]:
        if label in owner:
            index, channel = owner[label]
            writer.write(parts[index][0][channel] * scale)
        else:
            writer.write(np.zeros(shape, np.float16))
    return writer.close()


def drop_small_per_label(hard, minimum):
    """UZH: remove 26-connected components of fewer than `minimum` voxels, label by label."""
    for label in np.unique(hard)[1:]:
        components, _ = ndi.label(hard == label, np.ones((3, 3, 3), bool))
        sizes = np.bincount(components.ravel())
        hard[(sizes < minimum)[components] & (components > 0)] = 0
    return hard


def keep_large_components(hard):
    """ARG: keep 6-connected vessel components larger than 0.05% of the largest."""
    components, count = ndi.label(hard > 0)
    if not count:
        return hard
    sizes = np.bincount(components.ravel())
    sizes[0] = 0
    return np.where((sizes > sizes.max() * 0.0005)[components], hard, 0).astype(hard.dtype)


def predict_regions(rootfs, spec, values, spacing, back, work, device, record):
    """nnU-Net models in the team's merge order (UZH: four region models; KDH: one 40/42-label model)."""
    parts, merged = [], None
    whole = TOPBRAIN_LABELS[record['modality']]
    for region, folder in spec['regions']:
        model = rootfs / (UZH_MODELS + UZH_PARTS[folder] if folder in UZH_PARTS else folder)
        checkpoint = spec.get('checkpoint', 'checkpoint_final.pth')
        predictor, details = build_predictor(model, [model / f'fold_{f}' / checkpoint for f in spec['folds']], device)
        data, properties = nnunet_input(predictor, values[None], spacing)
        probabilities, _, timing = run_stage(predictor, data, properties, 'nnunet', back, work / f'{region}.npy')
        del predictor
        if 'mapping' in spec:
            mapping = {int(k): int(v) for k, v in json.loads((rootfs / spec['mapping']).read_text())[region].items()}
        elif tuple(details['labels']) == whole:
            mapping = {label: label for label in whole if label}
        else:
            raise ValueError(f'{region} labels differ from TopBrain v1 and no mapping is given')
        parts.append((probabilities, details['labels'], mapping))
        hard = argmax_labels(probabilities, details['labels'])
        if spec.get('min_component_voxels'):
            hard = drop_small_per_label(hard, spec['min_component_voxels'])
        merged = np.zeros(hard.shape, np.uint8) if merged is None else merged
        for label, target in mapping.items():
            merged[hard == label] = target
        record['stages'].append({'name': region, **details, **timing})
        print(f'{record["team"]} {record["case"]} {region}: {timing}', flush=True)
    if 'binary_mask' in spec:
        return mask_with_binary(rootfs, spec, values, spacing, back, work, device, record, parts[0][0], merged)
    if len(parts) == 1:
        os.replace(work / f'{spec["regions"][0][0]}.npy', work / 'probabilities.npy')
        return ChannelArray(work / 'probabilities.npy'), merged
    return compose_regions(parts, whole, work / 'probabilities.npy'), merged


def mask_with_binary(rootfs, spec, values, spacing, back, work, device, record, labelled, merged):
    """KDH MR: a five-fold binary vessel model masks the labelled model (label times binary). Softly, each named
    label becomes P(binary vessel) x P(label), and background takes the rest."""
    model = rootfs / spec['binary_mask']
    predictor, details = build_predictor(model, [model / f'fold_{f}' / spec['checkpoint'] for f in spec['folds']], device)
    data, properties = nnunet_input(predictor, values[None], spacing)
    binary, _, timing = run_stage(predictor, data, properties, 'nnunet', back, work / 'binary.npy')
    del predictor
    record['stages'].append({'name': 'binary', **details, **timing})
    writer = NpyWriter(work / 'probabilities.npy', labelled.shape)
    for channel in binary_masked(binary, labelled):
        writer.write(channel)
    merged[argmax_labels(binary, details['labels']) == 0] = 0
    return writer.close(), merged


def binary_masked(binary, labelled):
    """Channels of P(binary vessel) x P(label), with background taking the rest."""
    vessel = np.asarray(binary[1], np.float32)
    yield 1 - vessel * (1 - np.asarray(labelled[0], np.float32))
    for channel in range(1, labelled.shape[0]):
        yield vessel * labelled[channel]


def predict_two_stage(rootfs, spec, values, spacing, back, work, device, record):
    modality = record['modality']
    resources = rootfs / spec['resources']
    first, details = build_predictor(resources / 'weight_1', sorted((resources / 'weight_1').glob('checkpoint*.pth')),
                                     device)
    standardised = (values - values.mean()) / max(float(values.std()), 1e-8)
    data, properties = trilinear_input(first, standardised[None], spacing)
    del standardised
    _, vessel, timing = run_stage(first, data, properties, 'trilinear', back, work / 'stage1.npy', keep=1)
    record['stages'].append({'name': 'binary vessel stage', **details, **timing})
    del first
    scaled = np.clip(values, 0, 700) if modality == 'mr' else values.copy()
    scaled -= scaled.min()
    scaled /= max(float(scaled.max()), 1e-8)
    second, details = build_predictor(resources / 'weight_2', sorted((resources / 'weight_2').glob('checkpoint*.pth')),
                                      device)
    data, properties = trilinear_input(second, np.stack([scaled, vessel]), spacing)
    del scaled, vessel
    stage2, _, timing = run_stage(second, data, properties, 'trilinear', back, work / 'stage2.npy')
    record['stages'].append({'name': 'labelled stage with vessel probability input', **details, **timing})
    del second
    mapping = ARG_MAPPING[modality]
    hard = keep_large_components(argmax_labels(stage2, details['labels']))
    official = np.zeros(hard.shape, np.uint8)
    for label, target in mapping.items():
        official[hard == label] = target
    record['soft_mapping'] = ('CT: ARG 35-42 (external carotid branches) to background, 43-48 to 35-40; the official '
                              'map keeps 35-40' if modality == 'ct' else 'as the official map')
    return map_channels(stage2, details['labels'], ARG_SOFT_MAPPING[modality], TOPBRAIN_LABELS[modality],
                        work / 'probabilities.npy'), official


def predict(root, team, case, device='mps'):
    """Run one team's native pipeline on one case; keep soft class probabilities and the team's own label map."""
    import resource
    modality = CASES[case]['modality']
    spec = TEAMS[team][modality]
    rootfs = verified_rootfs(root, spec['image'])
    work = work_folder(root, case) / team
    work.mkdir(parents=True, exist_ok=True)
    for stale in ('predict.json', 'labels.nii.gz', 'probabilities.npy'):
        (work / stale).unlink(missing_ok=True)
    for stale in work.glob('*' + SCRATCH):   # left by a run that was killed
        stale.unlink()
    image, values, spacing, back = source_image(root, case)
    record = {'team': team, 'case': case, 'modality': modality, 'device': device, **made_from(case),
              'unpacked_image': spec['image'], 'pipeline': spec['pipeline'], 'source_shape': list(image.shape),
              'affine': image.affine.tolist(), 'labels': list(TOPBRAIN_LABELS[modality]), 'stages': [],
              'precision': 'float16 network, float32 transposed convolutions',
              'probability_resampling': 'softmax in network space, then per-channel resampling to the image',
              'orientation': ('stored ' + ''.join(nib.aff2axcodes(image.affine)) + '; reoriented to LPS before inference '
                              '(axis flips and permutations only) and mapped back, where the images expect LPS input'),
              'deviations': {
                  'ARG': 'stage 2 receives the stage-1 vessel probability resampled after softmax, not before',
                  'KDH': ('MR: soft P(binary) x P(label) at 0.5 is slightly stricter than the hard AND of the two label '
                          'maps (182,697 against 183,143 vessel voxels on TopCoW-014 MR)'),
              }.get(team)}
    started = time.monotonic()
    run = predict_regions if spec['pipeline'] == 'regions' else predict_two_stage
    probabilities, official = run(rootfs, spec, values, spacing, back, work, device, record)
    if probabilities.shape[1:] != image.shape or official.shape != image.shape:
        raise ValueError('Outputs left the source grid')
    nib.save(nib.Nifti1Image(official, image.affine), work / 'labels.nii.gz')
    record.update(seconds=round(time.monotonic() - started, 1),
                  peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                  probabilities=str((work / 'probabilities.npy').relative_to(root)),
                  official_labels=str((work / 'labels.nii.gz').relative_to(root)))
    atomic_json(work / 'predict.json', record)
    return record


VESSELFM = {
    'TopCoW-014-ct': 'cache/poc-model-results/TopCoW-014-annotated-cta-benchmark-v1/vesselfm-base',
    'TopCoW-018-ct': 'cache/poc-model-results/TopCoW-018-annotated-cta-benchmark-v1/vesselfm-base',
    'TopCoW-014-mr': 'cache/poc-model-results/TopCoW-014-annotated-mra-benchmark-v1/vesselfm-base',
    'TopCoW-018-mr': 'cache/poc-model-results/TopCoW-018-annotated-mra-benchmark-v1/vesselfm-base',
    'TopCoW-001-ct': 'cache/poc-model-results/TopCoW-001-annotated-cta-benchmark-v1/vesselfm-base',
    'TopCoW-001-mr': 'cache/poc-model-results/TopCoW-001-annotated-mra-benchmark-v1/vesselfm-base',
    'PTNT_4509-mr': 'cache/poc-model-results/anatomy-b402d83f585ccb6a/vesselfm-base',
    'PTNT_9937-mr': 'cache/poc-model-results/anatomy-028393bf96adc4b5/vesselfm-base',
}
# The current displayed TOF cleanup layer, for label-free comparison only.
CLEANUP = {
    'PTNT_4509-mr': 'qc/eden2020-glioma-5714812/vascular-reconstruction-20261123-v1/studies/PTNT_4509/patient-anatomy.json',
    'PTNT_9937-mr': 'prepared/Eden 2020 ResectionLab Projects (restricted)/PTNT_9937/vascular-review.anatomy.json',
}
# Roots: basilar, internal carotids and vertebrals; CT adds the vein of Galen, straight and superior sagittal
# sinuses; MR adds the external carotids (MR labels 35 and 36 are R-ECA and L-ECA, not veins).
ROOTS = {'ct': (1, 4, 6, 23, 24, 35, 36, 40), 'mr': (1, 4, 6, 23, 24, 35, 36)}


EDEN_COHORT = 'prepared/Eden 2020 ResectionLab Projects (restricted)'


ANGIOGRAPHY = {'CTA': 'ct', 'TOF': 'mr', 'MRA': 'mr'}


def retained_vesselfm(root):
    """Retained vesselFM run folders by the sha256 of the file they ran on."""
    runs = {}
    for record in sorted(safe_path(root, 'cache/poc-model-results').glob('*/vesselfm-base/*/inference.json')):
        if record.name.startswith('._'):
            continue
        document = json.loads(record.read_text())
        if document.get('status', 'succeeded') == 'succeeded' and document.get('probability'):
            gaussian = document.get('model', {}).get('parameters', {}).get('blend') == 'gaussian'
            if gaussian or document['input'].get('sha256') not in runs:   # the folder with a gaussian-blend run wins
                runs[document['input'].get('sha256')] = record.parents[1]
    return runs


def study_case(document, folder_name, series_id=None):
    """``(case name, series row, modality)`` for the angiographic series of a prepared study document."""
    rows = [row for row in document.get('series', []) if row.get('modality') in ANGIOGRAPHY and row.get('original')]
    if series_id is not None:
        rows = [row for row in rows if row['id'] == series_id]
    if len(rows) != 1:
        found = ', '.join(f"{row['id']} ({row['modality']})" for row in rows) or 'none'
        raise SystemExit(f'{folder_name}: need exactly one acquired CTA, TOF or MRA series; found {found}. Name one with --series')
    modality = ANGIOGRAPHY[rows[0]['modality']]
    # A series named on the command line is part of the name, so two series of one study are two cases.
    clean = [re.sub(r'[^A-Za-z0-9_.-]+', '-', part).strip('-.') for part in (folder_name, *([series_id] if series_id else []))]
    if not all(clean):
        raise SystemExit(f'{folder_name!r} gives no usable case name')
    name = '-'.join(clean)
    return f'{name}-{modality}', rows[0], modality


def register_study(root, manifest, series_id=None):
    """Add any prepared study on the drive to the case table: its one acquired CTA, TOF or MRA file and the retained
    vesselFM run of that same file. Results go under ``STORE``. Returns the case name."""
    manifest = Path(manifest)
    manifest = safe_path(root, manifest.resolve().relative_to(Path(root).resolve()) if manifest.is_absolute() else manifest)
    if not manifest.is_file():
        raise SystemExit(f'No study document at {manifest}')
    case, series, modality = study_case(json.loads(manifest.read_text()), manifest.parent.name, series_id)
    image = safe_path(root, manifest.parent.relative_to(root) / series['original']['path'])
    if not image.is_file() or sha256_file(image) != series['original']['sha256']:
        raise SystemExit(f'{case}: {image.name} is missing or is not the file the study document records')
    run = retained_vesselfm(root).get(series['original']['sha256'])
    if run is None:
        raise SystemExit(f'{case}: no retained vesselFM run of {image.name}. Run the vessel model on this series in the app '
                         '(Anatomy, vesselFM) with probabilities retained, then run this again')
    flags = ['--study', str(manifest.relative_to(root))] + (['--series', series_id] if series_id else [])
    entry = {'modality': modality, 'reference': None, 'image': str(image.relative_to(root)), 'sha256': series['original']['sha256'],
             'store': STORE, 'flags': flags, 'manifest': str(manifest.relative_to(root)), 'series': series['id']}
    if case in CASES and CASES[case] != entry:
        raise SystemExit(f'{case} is already the name of another case')
    CASES[case] = entry
    VESSELFM[case] = str(run.relative_to(root))
    if 'gaussian' not in vesselfm_runs(root, case):
        raise SystemExit(f'{case}: the retained vesselFM run is not a gaussian-blend run of this file')
    return case


def register_eden(root, patients):
    """Add EDEN patients (restricted, local only) to the case table: the acquired TOF original named by the
    patient's prepared study, and the retained vesselFM run of that same file."""
    runs = retained_vesselfm(root)
    added = []
    for patient in patients:
        case = f'{patient}-mr'
        if case not in CASES:
            if not re.fullmatch(r'PTNT_\d{4}', patient):
                raise SystemExit(f'{patient!r} is not an EDEN patient id')
            study = json.loads(safe_path(root, f'{EDEN_COHORT}/{patient}/patient-anatomy.json').read_text())
            series = next((row for row in study['series'] if row['modality'] == 'TOF' and row.get('original')), None)
            if series is None:
                raise SystemExit(f'{patient} has no acquired TOF')
            run = runs.get(series['original']['sha256'])
            if run is None:
                raise SystemExit(f'{patient} has no retained vesselFM run of its TOF')
            CASES[case] = {'modality': 'mr', 'reference': None, 'image': f'{EDEN_COHORT}/{patient}/{series["original"]["path"]}'}
            VESSELFM[case] = str(run.relative_to(root))
            if 'gaussian' not in vesselfm_runs(root, case):
                raise SystemExit(f'{patient}: the retained vesselFM run is not a gaussian-blend run of this TOF file')
        added.append(case)
    return added


def run_cases(root, cases, teams=('UZH', 'ARG', 'KDH'), device='mps'):
    """Each case in turn: every team that has not finished (one process each, so memory is returned), the frozen
    combined ensemble with the frozen vesselFM rule, then retain. A case already retained is skipped. A registered
    study uses the committed rule; kept and unfinished results must have been made from the case's own file."""
    import subprocess
    import sys
    studies = [case for case in cases if CASES[case].get('store')]
    rule = study_rule(root) if studies else None
    if studies and tuple(teams) != tuple(rule['teams']):
        raise SystemExit(f'A registered study runs the teams of {RULE.name}: {rule["teams"]}')
    combiner = rule['combiner'] if len(studies) == len(cases) else frozen_combiner(root)
    if combiner is None:
        raise SystemExit('Run select-combiner first')
    name = combined_name(combiner, list(teams))
    done = []
    for case in cases:
        kept = kept_folder(root, case) / name / 'ensemble.json'
        if kept.is_file():
            require_made_from(kept, case)
            print(f'{case}: already retained', flush=True)
            done.append(case)
            continue
        patient = CASES[case].get('flags') or (['--eden', case[:-3]] if case.startswith('PTNT_') else [])
        for team in teams:
            finished = work_folder(root, case) / team / 'predict.json'
            if finished.is_file():
                require_made_from(finished, case)
                continue
            print(f'{case}: {team} starting {time.strftime("%H:%M:%S")}', flush=True)
            subprocess.run([sys.executable, __file__, *patient, 'predict', team, case, '--device', device], check=True,
                           stdout=subprocess.DEVNULL)
        addition = dict(rule['connectedAddition']) if CASES[case].get('store') else None
        folder, record = ensemble(root, case, list(teams), addition, combiner)
        print(f'{case}: ensemble {record["named_voxels"]:,} named voxels, {record["added_voxels"]:,} added', flush=True)
        retain(root, case)
        done.append(case)
    return done


def committed_rule():
    """The frozen ensemble rule as committed with the code (manifests/vascular-ensemble-rule.json)."""
    document = json.loads(RULE.read_text())
    if document.get('schema') != 'vascular-ensemble-frozen-rule-v1':
        raise SystemExit(f'{RULE.name} is not a frozen vessel ensemble rule')
    return document


def frozen_addition(root):
    selection = safe_path(root, f'{PHASE}/addition-selection.json')
    if not selection.is_file():
        raise SystemExit('No frozen vesselFM rule: run select on the development cases first')
    return {key: value for key, value in json.loads(selection.read_text())['selected'].items() if key in ('low', 'high')}


def study_rule(root):
    """The rule a registered study is run with: the committed copy. Where the drive holds the selections it was
    copied from, they must still equal it."""
    rule = committed_rule()
    for name, key, read in (('combiner-selection.json', 'combiner', frozen_combiner),
                            ('addition-selection.json', 'connectedAddition', frozen_addition)):
        if safe_path(root, f'{PHASE}/{name}').is_file() and read(root) != rule[key]:
            raise SystemExit(f'{PHASE}/{name} differs from {RULE.name}; settle which is current before running a study')
    return rule


def vesselfm_runs(root, case):
    """Retained vesselFM probabilities on the case's own grid, keyed by blend."""
    runs = {}
    for record in sorted(safe_path(root, VESSELFM[case]).glob('*/inference.json')):
        document = json.loads(record.read_text())
        if document.get('status', 'succeeded') != 'succeeded' or not document.get('probability'):
            continue
        source = nib.load(safe_path(root, CASES[case]['image']))
        # The same file: by content where the case records its hash (a study may have moved), else by path.
        same = (document['input'].get('sha256') == CASES[case]['sha256'] if CASES[case].get('sha256')
                else Path(document['input']['path']).resolve() == safe_path(root, CASES[case]['image']).resolve())
        if not same:
            continue
        modality = {'CTA': 'ct', 'MRA': 'mr'}.get(document['input']['modality'])
        if modality != CASES[case]['modality']:
            raise ValueError(f'vesselFM ran {case} as {document["input"]["modality"]}')
        image = nib.load(document['probability']['path'])
        require_same_grid(source.shape, source.affine, image.shape, image.affine)
        image.vesselfm_modality = modality
        runs[document['model']['parameters']['blend']] = image
    return runs


def team_probabilities(root, case, team):
    folder = case_folder(root, case, team)
    record = require_made_from(folder / 'predict.json', case)  # Never combine a prediction of another file.
    values = (ChannelArray(folder / 'probabilities.npy') if (folder / 'probabilities.npy').is_file()
              else NpzChannels(folder / 'probabilities.npz'))
    return SoftSegmentation(record['modality'], record['labels'], values, record['affine'])


def save_nifti(values, image, path):
    output = nib.Nifti1Image(values, image.affine)
    output.header.set_xyzt_units('mm')
    nib.save(output, path)


def ensemble(root, case, teams, addition=None, combiner=None):
    """Combine the teams' class probabilities and add vesselFM signal that connects to the named tree, as its own
    layer. Without a combiner this is the plain mean (attempt one)."""
    modality = CASES[case]['modality']
    image = nib.load(safe_path(root, CASES[case]['image']))
    members = [team_probabilities(root, case, team) for team in teams]
    for member in members:
        require_same_grid(image.shape, image.affine, member.shape, member.affine)
    name = f'ensemble-{"-".join(teams)}' if combiner is None else combined_name(combiner, teams)
    folder = work_folder(root, case) / name
    folder.mkdir(parents=True, exist_ok=True)
    writer = NpyWriter(folder / 'probabilities.npy', members[0].probabilities.shape)
    if combiner is None:
        mean = average(members, out=writer)
    else:
        weights, base = combiner_arguments(combiner, teams, modality)
        mean = combine(members, combiner['method'], weights=weights, base=base, out=writer)
    groups = vessel_groups(mean, VEIN_LABELS[modality])
    for name, values in groups.items():
        save_nifti(values.astype(np.float32), image, folder / f'{name}_probability.nii.gz')
    named = named_label_map(mean)
    save_nifti(named, image, folder / 'named_labels.nii.gz')
    record = {'case': case, **made_from(case), 'modality': modality, 'teams': list(teams), 'labels': list(mean.labels),
              'vein_labels': list(VEIN_LABELS[modality]), 'named_voxels': int((named > 0).sum()),
              'combiner': combiner or {'method': 'mean'}}
    record = add_vesselfm(root, case, folder, record, named, groups['vessel'], addition)
    display_additions(root, case, folder)
    return folder, record


# Display candidates: vesselFM components (26-connected) at or above `low` that hold a voxel at `high`. Chosen
# on the development cases by looking at the drawn tree: on the TOF case 0.5 joins a noisy mass at the face and
# skull base to the tree (395,165 voxels shown beyond the named tree against 208,629 at the scored thresholds), so
# MR keeps the scored thresholds; on the CTA case the bone ceiling removes that kind of signal and 0.5 stays clean.
DISPLAY_CANDIDATES = {'ct': {'low': 0.5, 'high': 0.5}, 'mr': {'low': 0.7, 'high': 0.9}}
DISPLAY_FILES = ('vesselfm_candidate_probability.nii.gz', 'fine_branch_score.nii.gz', 'display_additions.json')


def display_additions(root, case, folder):
    """Display-only vessel candidates beside an ensemble; the ensemble and its scored addition are not changed.

    `vesselfm_candidate_probability` is every vesselFM component outside the named tree at the display thresholds
    of its modality (DISPLAY_CANDIDATES), connected to the tree or not, with its score. The scored connected
    addition drops every component that does not touch the tree. Which candidates are shown is decided when a
    study is drawn: those that reach the tree directly or across a signal-supported gap; the rest stay in review.
    On CT, voxels at or above the bone ceiling are left out. On CT, `fine_branch_score` adds the thin branches
    followed outward from the tree and the candidates in the acquired image (resectionlab.vessel_fine_growth).
    Never applied across modalities.
    """
    from resectionlab.vascular_rooting import RootingPolicy
    from resectionlab.vessel_fine_growth import grow_fine_branches
    folder = Path(folder)
    image = nib.load(safe_path(root, CASES[case]['image']))
    modality = CASES[case]['modality']
    run = vesselfm_runs(root, case)['gaussian']
    if run.vesselfm_modality != modality:
        raise ValueError(f'A {run.vesselfm_modality} vesselFM run cannot extend a {modality} tree')
    named = nib.load(folder / 'named_labels.nii.gz')
    require_same_grid(image.shape, image.affine, named.shape, named.affine)
    tree = np.asarray(named.dataobj) > 0
    score = np.asarray(run.dataobj, np.float32)
    rule = DISPLAY_CANDIDATES[modality]
    candidates = (score >= rule['low']) & ~tree
    receipt = {'case': case, 'modality': modality, 'rule': ' '.join(display_additions.__doc__.split()),
               'vesselfm': {**rule, 'blend': 'gaussian'}}
    signal = None
    if modality == 'ct':
        signal = np.asarray(image.dataobj, np.float32)
        ceiling = RootingPolicy().ct_ceiling_hu
        bone = candidates & (signal >= ceiling)
        receipt['vesselfm'].update(boneCeilingHu=ceiling, boneVoxelsExcluded=int(bone.sum()))
        candidates &= ~bone
        del bone
    labels, count = ndi.label(candidates, np.ones((3, 3, 3), bool))
    if rule['high'] > rule['low'] and count:
        strong = np.zeros(count + 1, bool)
        strong[np.unique(labels[candidates & (score >= rule['high'])])] = True
        strong[0] = False
        candidates = strong[labels]
        count = int(strong.sum())
    del labels
    receipt['vesselfm'].update(voxels=int(candidates.sum()), components=int(count))
    save_nifti(np.where(candidates, score, 0).astype(np.float32), image, folder / DISPLAY_FILES[0])
    del score
    if modality == 'ct':
        spacing = np.sqrt((np.asarray(image.affine, float)[:3, :3] ** 2).sum(axis=0))
        fine, _, receipt['fineBranches'] = grow_fine_branches(signal, spacing, tree | candidates, 'CT')
        save_nifti(fine, image, folder / DISPLAY_FILES[1])
    else:
        (folder / DISPLAY_FILES[1]).unlink(missing_ok=True)
    atomic_json(folder / DISPLAY_FILES[2], receipt)
    return receipt


def addition_evidence(image, tree, added):
    """Label-free description of the connected vesselFM layer: added voxels within one voxel (26-neighbourhood) of
    the ensemble vessels are wall thickening, the rest new branch; each group's acquired intensity is given with its
    contrast (group - background) / (lumen - background), where lumen is the ensemble vessels and background is the
    shell two to four voxels outside the ensemble and added vessels."""
    values = np.asanyarray(image.dataobj, dtype=np.float32)
    full = np.ones((3, 3, 3), bool)
    thickening = added & ndi.binary_dilation(tree, full)
    branch = added & ~thickening
    both = tree | added
    background = ndi.binary_dilation(both, full, iterations=4) & ~ndi.binary_dilation(both, full)
    lumen, shell = float(values[tree].mean()), float(values[background].mean())

    def describe(mask):
        if not mask.any():
            return {'voxels': 0}
        mean = float(values[mask].mean())
        return {'voxels': int(mask.sum()), 'intensity_mean': mean, 'intensity_median': float(np.median(values[mask])),
                'contrast': (mean - shell) / (lumen - shell) if lumen != shell else None}
    return {'definition': ' '.join(addition_evidence.__doc__.split()),
            'ensemble_lumen_mean': lumen, 'background_mean': shell,
            'wall_thickening': describe(thickening), 'new_branch': describe(branch)}


def add_vesselfm(root, case, folder, record, named=None, vessel=None, addition=None):
    """Apply the (frozen) connected-addition rule to an ensemble folder and record it."""
    image = nib.load(safe_path(root, CASES[case]['image']))
    modality = CASES[case]['modality']
    named = np.asarray(nib.load(folder / 'named_labels.nii.gz').dataobj) if named is None else named
    vessel = np.asarray(nib.load(folder / 'vessel_probability.nii.gz').dataobj, np.float32) if vessel is None else vessel
    addition = addition or frozen_addition(root)
    run = vesselfm_runs(root, case)['gaussian']
    score = np.asarray(run.dataobj, np.float32).clip(0, 1)
    added, mask = ConnectedAddition(**addition).apply(named > 0, modality, image.affine, score, run.vesselfm_modality,
                                                      run.affine)
    save_nifti(added, image, folder / 'vesselfm_connected_probability.nii.gz')
    (folder / 'vessel_with_vesselfm_probability.nii.gz').unlink(missing_ok=True)
    record = {**record, 'addition': {**addition, 'vesselfm_blend': 'gaussian'}, 'added_voxels': int(mask.sum()),
              'addition_evidence': addition_evidence(image, vessel >= 0.5, mask)}
    atomic_json(folder / 'ensemble.json', record)
    return record


def per_class_dice(named, reference):
    present = sorted(set(np.unique(reference).tolist()) | set(np.unique(named).tolist()) - {0})
    return {str(label): dice(named == label, reference == label) for label in present if label}


def score(root, case, rows):
    """Agreement with the TopBrain v1 annotation (labelled cases) or label-free topology (EDEN TOF)."""
    image = nib.load(safe_path(root, CASES[case]['image']))
    spacing = np.linalg.norm(image.affine[:3, :3], axis=0)
    modality = CASES[case]['modality']
    results = {}
    reference = None
    if CASES[case]['reference']:
        label_image = nib.load(safe_path(root, CASES[case]['reference']))
        require_same_grid(image.shape, image.affine, label_image.shape, label_image.affine)
        reference = np.asarray(label_image.dataobj).astype(np.uint8)
        roots = np.isin(reference, ROOTS[modality])
        results['annotation'] = {'file': CASES[case]['reference'], 'betti': betti(reference > 0),
                                 'floating': floating_components(reference > 0, roots, spacing)}
    for name, produce in rows.items():
        mask, named = produce()
        row = {'voxels': int(mask.sum())}
        if reference is not None:
            row.update(agreement(mask, reference > 0, spacing))
            row['floating'] = floating_components(mask, roots, spacing)
            if named is not None:
                row['per_class_dice'] = per_class_dice(named, reference)
        else:
            row['betti'] = betti(mask)
            row['floating'] = {**floating_components(mask, None, spacing), 'roots': 'largest component'}
        results[name] = row
        print(case, name, {k: row.get(k) for k in ('dice', 'cl_dice')}, row.get('betti', {}).get('b0_error'),
              row['floating']['floating'], flush=True)
    return results



def cleanup_layer(root, case, image):
    manifest = safe_path(root, CLEANUP[case])
    document = json.loads(manifest.read_text())
    layer = next(item for item in document['layers'] if item['id'] == 'vesselfm-cleanup')
    require_same_grid(image.shape, image.affine, layer['data']['shape'], layer['affine'])
    return np.load(manifest.parent / layer['data']['path'], mmap_mode='r') > 0


def evaluate(root, case, teams, ensemble_folders):
    """Rows (a) vesselFM alone, (b) each team, (c) the ensemble and (d) the ensemble with connected vesselFM."""
    image = nib.load(safe_path(root, CASES[case]['image']))
    rows = {}
    runs = vesselfm_runs(root, case)
    for blend, run in runs.items():
        rows[f'vesselfm-{blend}'] = lambda run=run: (np.asarray(run.dataobj, np.float32) >= 0.5, None)
    tuned = (frozen_vesselfm(root) or {}).get(CASES[case]['modality'])
    if tuned and 'gaussian' in runs:
        rows['vesselfm-tuned'] = lambda: (filter_components(np.asarray(runs['gaussian'].dataobj, np.float32)
                                                            >= tuned['threshold'], tuned['min_component_voxels']), None)
    if case in CLEANUP:
        rows['vesselfm-cleanup-layer'] = lambda: (cleanup_layer(root, case, image), None)
    for team in teams:
        work = case_folder(root, case, team)

        def official(work=work):
            labels = np.asarray(nib.load(work / 'labels.nii.gz').dataobj)
            return labels > 0, labels

        def soft(team=team):
            member = team_probabilities(root, case, team)
            named = named_label_map(member)
            return named > 0, named
        rows[f'{team}-official'] = official
        rows[f'{team}-soft'] = soft
    for folder in ensemble_folders:
        def tree(folder=folder):
            named = np.asarray(nib.load(folder / 'named_labels.nii.gz').dataobj)
            return named > 0, named

        def connected(folder=folder):
            named = np.asarray(nib.load(folder / 'named_labels.nii.gz').dataobj)
            added = np.asarray(nib.load(folder / 'vesselfm_connected_probability.nii.gz').dataobj) > 0
            return (named > 0) | added, None
        label = 'ensemble' if folder.name.startswith('ensemble-') else folder.name.split('-')[0] + '-' + folder.name.split('-')[1]
        rows[label] = tree
        rows[f'{label}+vesselfm-connected'] = connected
    return score(root, case, rows)



DEVELOPMENT = ('TopCoW-014-ct', 'TopCoW-014-mr')
ADDITION_GRID = [{'low': low, 'high': high} for low in (0.5, 0.7) for high in sorted({low, 0.9})]


def select_addition(root, ensembles):
    """Choose the connected-addition rule on the development cases only, before any held-out case is scored.

    Each setting is scored against the ensemble alone: mean over the development cases of the change in Dice
    plus the change in clDice, minus the change in Betti b0 and b1 error as a fraction of the annotation's
    b0 + b1. The best setting is frozen whether or not it improves on the ensemble.
    """
    trials, baseline = [], {}
    for case, folder in ensembles.items():
        image = nib.load(safe_path(root, CASES[case]['image']))
        spacing = np.linalg.norm(image.affine[:3, :3], axis=0)
        reference = np.asarray(nib.load(safe_path(root, CASES[case]['reference'])).dataobj) > 0
        named = np.asarray(nib.load(folder / 'named_labels.nii.gz').dataobj)
        run = vesselfm_runs(root, case)['gaussian']
        score = np.asarray(run.dataobj, np.float32).clip(0, 1)
        base = agreement(named > 0, reference, spacing)
        scale = base['betti']['reference']['b0'] + base['betti']['reference']['b1']
        baseline[case] = base
        for setting in ADDITION_GRID:
            _, added = ConnectedAddition(**setting).apply(named > 0, CASES[case]['modality'], image.affine, score,
                                                          run.vesselfm_modality, run.affine)
            result = agreement((named > 0) | added, reference, spacing)
            change = (result['dice'] - base['dice'] + result['cl_dice'] - base['cl_dice']
                      - (result['betti']['b0_error'] - base['betti']['b0_error']
                         + result['betti']['b1_error'] - base['betti']['b1_error']) / scale)
            trials.append({'case': case, **setting, 'added_voxels': int(added.sum()), 'objective_change': change,
                           'dice': result['dice'], 'cl_dice': result['cl_dice'], 'betti': result['betti']})
            print(case, setting, round(change, 4), flush=True)
    means = []
    for setting in ADDITION_GRID:
        rows = [t for t in trials if all(t[k] == v for k, v in setting.items())]
        means.append((float(np.mean([t['objective_change'] for t in rows])), setting))
    best = max(means, key=lambda item: item[0])
    return {'rule': ('mean over development cases of dDice + dclDice - (db0_error + db1_error) / (b0 + b1 of '
                     'the annotation), relative to the ensemble alone; best setting frozen'),
            'development_cases': list(ensembles), 'ensembles': {case: str(folder) for case, folder in ensembles.items()},
            'grid': ADDITION_GRID, 'trials': trials,
            'note': ('Added components must touch the tree (26-neighbourhood); the earlier gap parameter is gone. '
                     'The first frozen rule used gap 1, which is touching, so its outputs were unchanged.'),
            'ensemble_alone': baseline, 'mean_objective_change': [{'setting': s, 'mean': m} for m, s in means],
            'selected': best[1], 'selected_mean_objective_change': best[0]}




HELD_OUT = ('TopCoW-018-ct', 'TopCoW-018-mr')
# Pre-registered 2026-11-26, never scored or tuned on: the lowest-numbered TopBrain case on the HDD other than 012,
# 014 and 018 (CT first, its paired MR second). Each may be scored once.
CONFIRMATION = ('TopCoW-001-ct', 'TopCoW-001-mr')
TEAM_ORDER = ('UZH', 'ARG', 'KDH')
HELDOUT_LOG = f'{PHASE}/heldout-log.json'
VESSELFM_GRID = [{'threshold': threshold, 'min_component_voxels': size}
                 for threshold in (0.3, 0.5, 0.7, 0.9) for size in (0, 50, 200, 'largest')]


def objective_change(result, baseline):
    """The frozen development objective: dDice + dclDice - (db0_error + db1_error) / (annotation b0 + b1)."""
    scale = baseline['betti']['reference']['b0'] + baseline['betti']['reference']['b1']
    return (result['dice'] - baseline['dice'] + result['cl_dice'] - baseline['cl_dice']
            - (result['betti']['b0_error'] - baseline['betti']['b0_error']
               + result['betti']['b1_error'] - baseline['betti']['b1_error']) / scale)


def heldout_log(root):
    path = safe_path(root, HELDOUT_LOG)
    return json.loads(path.read_text()) if path.is_file() else []


def append_heldout(root, entry):
    """The held-out log only grows; every scoring pass of 018 or a confirmation case adds one entry."""
    entries = heldout_log(root)
    entries.append({'when': time.strftime('%Y-%m-%d %H:%M %Z'), **entry})
    atomic_json(safe_path(root, HELDOUT_LOG), entries)


def select_combiner(root, teams=TEAM_ORDER, reselect=False):
    """Attempt two: choose how the teams are pooled on the development cases only, with the frozen objective
    relative to the plain mean. Weights are each team's development clDice on that modality; the base team is the
    one with the highest development clDice on that modality. Weights and base are stored by team name."""
    previous = frozen_combiner(root)
    if previous and not reselect:
        name = combined_name(previous, teams)
        if any(name in entry.get('ensembles', []) for entry in heldout_log(root)):
            raise SystemExit('A held-out case was already scored with the selected combiner; pass --reselect to override')
    if previous and reselect:
        append_heldout(root, {'event': 'combiner reselected after held-out scoring', 'previous': previous})
    weights, base = {}, {}
    for case in DEVELOPMENT:
        rows = json.loads(safe_path(root, f'{PHASE}/scores/{case}.json').read_text())
        modality = CASES[case]['modality']
        weights[modality] = {team: rows[f'{team}-official']['cl_dice'] for team in teams}
        base[modality] = max(teams, key=lambda team: weights[modality][team])
    trials = []
    for case in DEVELOPMENT:
        modality = CASES[case]['modality']
        image = nib.load(safe_path(root, CASES[case]['image']))
        spacing = np.linalg.norm(image.affine[:3, :3], axis=0)
        reference = np.asarray(nib.load(safe_path(root, CASES[case]['reference'])).dataobj) > 0
        members = [team_probabilities(root, case, team) for team in teams]
        results = {}
        for method in COMBINERS:
            vessel = vessel_probability(members, method,
                                        [weights[modality][team] for team in teams] if method == 'weighted' else None,
                                        teams.index(base[modality]) if method == 'base' else None)
            results[method] = agreement(vessel >= 0.5, reference, spacing)
            del vessel
        for method, result in results.items():
            change = objective_change(result, results['mean'])
            trials.append({'case': case, 'method': method, 'objective_change': change, 'dice': result['dice'],
                           'cl_dice': result['cl_dice'], 'betti': result['betti'],
                           'patch_betti': result['patch_betti']})
            print(case, method, round(result['dice'], 4), round(result['cl_dice'], 4), result['betti']['b0_error'],
                  result['betti']['b1_error'], round(change, 4), flush=True)
    means = {method: float(np.mean([t['objective_change'] for t in trials if t['method'] == method]))
             for method in COMBINERS}
    chosen = max(COMBINERS, key=lambda method: means[method])
    return {'rule': ' '.join(select_combiner.__doc__.split()),
            'context': ('Attempt two exists because the plain mean (attempt one) missed the Betti b1 part of the gate on '
                        'held-out TopCoW-018 CT. These candidates were defined after seeing that miss; selection used '
                        'only TopCoW-014, so 018 is not a clean test of attempt two. TopCoW-001 CT and MR are '
                        'pre-registered in heldout-log.json as clean confirmation cases.'),
            'development_cases': list(DEVELOPMENT), 'teams': list(teams), 'trials': trials,
            'mean_objective_change': means,
            'selected': {'method': chosen, 'teams': list(teams), 'weights': weights, 'base': base}}


def frozen_combiner(root):
    selection = safe_path(root, f'{PHASE}/combiner-selection.json')
    return json.loads(selection.read_text())['selected'] if selection.is_file() else None


def combined_name(combiner, teams):
    return f'combined-{combiner["method"]}-{"-".join(teams)}'


def combiner_arguments(combiner, teams, modality):
    """Weights and base looked up by team name, whatever order the teams are given in."""
    if sorted(teams) != sorted(combiner['teams']):
        raise ValueError(f'The combiner was chosen for {combiner["teams"]}, not {list(teams)}')
    weights = [combiner['weights'][modality][team] for team in teams] if combiner['method'] == 'weighted' else None
    base = list(teams).index(combiner['base'][modality]) if combiner['method'] == 'base' else None
    return weights, base


def filter_components(mask, minimum):
    """vesselFM tuning: drop 26-connected components below `minimum` voxels, or keep only the largest."""
    if minimum == 0 or not mask.any():
        return mask
    components, count = ndi.label(mask, np.ones((3, 3, 3), bool))
    sizes = np.bincount(components.ravel())
    sizes[0] = 0
    keep = sizes == sizes.max() if minimum == 'largest' else sizes >= minimum
    return keep[components]


def select_vesselfm(root):
    """A tuned vesselFM row for a fairer baseline: threshold and component filter chosen per modality on that
    modality's development case, with the frozen objective relative to vesselFM at 0.5 without a filter (the fixed
    row "vesselFM alone"). Per modality because one joint choice kept only the skull on CT (Dice 0)."""
    trials = []
    for case in DEVELOPMENT:
        image = nib.load(safe_path(root, CASES[case]['image']))
        spacing = np.linalg.norm(image.affine[:3, :3], axis=0)
        reference = np.asarray(nib.load(safe_path(root, CASES[case]['reference'])).dataobj) > 0
        score = np.asarray(vesselfm_runs(root, case)['gaussian'].dataobj, np.float32)
        results = {}
        for setting in VESSELFM_GRID:
            mask = filter_components(score >= setting['threshold'], setting['min_component_voxels'])
            results[json.dumps(setting)] = (setting, agreement(mask, reference, spacing))
        baseline = results[json.dumps({'threshold': 0.5, 'min_component_voxels': 0})][1]
        for setting, result in results.values():
            trials.append({'case': case, **setting, 'objective_change': objective_change(result, baseline),
                           'dice': result['dice'], 'cl_dice': result['cl_dice'], 'betti': result['betti'],
                           'patch_betti': result['patch_betti']})
            print(case, setting, round(trials[-1]['objective_change'], 4), flush=True)
    selected = {}
    for case in DEVELOPMENT:
        best = max((t for t in trials if t['case'] == case), key=lambda t: t['objective_change'])
        selected[CASES[case]['modality']] = {key: best[key] for key in ('threshold', 'min_component_voxels')}
    return {'rule': ' '.join(select_vesselfm.__doc__.split()), 'development_cases': list(DEVELOPMENT),
            'grid': VESSELFM_GRID, 'trials': trials, 'selected': selected}


def wire_study(root, case, folder):
    """Make a retained ensemble the displayed vessels of the study its case was registered from."""
    import importlib.util
    spec = importlib.util.spec_from_file_location('capture_vascular_parity', Path(__file__).with_name('capture_vascular_parity.py'))
    wiring = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(wiring)
    manifest = safe_path(root, CASES[case]['manifest'])
    collection = wiring.open_anatomy_manifest(manifest)
    receipt = wiring.wire_ensemble(collection, CASES[case]['series'], folder, manifest=manifest,
                                   source=str(folder.relative_to(root)))
    return {key: receipt.get(key) for key in ('changed', 'layers', 'added', 'retired', 'binding')}


def frozen_vesselfm(root):
    selection = safe_path(root, f'{PHASE}/vesselfm-tuning.json')
    return json.loads(selection.read_text())['selected'] if selection.is_file() else None


def arg_ct_check(root, case):
    """How ARG's CT map turns external carotid predictions into veins: voxels the raw 48-label stage assigns to
    35-42, and ARG per-class Dice for 35-40 with the image's map (35-40 kept) and with 35-42 sent to background."""
    folder = case_folder(root, case, 'ARG')
    raw = ChannelArray(folder / 'stage2.npy') if (folder / 'stage2.npy').is_file() else NpzChannels(folder / 'stage2.npz')
    hard = argmax_labels(raw, list(range(raw.shape[0])))
    reference = np.asarray(nib.load(safe_path(root, CASES[case]['reference'])).dataobj)
    kept = hard.copy()
    for label, target in ARG_MAPPING['ct'].items():
        kept[hard == label] = target
    dropped = hard.copy()
    for label, target in ARG_SOFT_MAPPING['ct'].items():
        dropped[hard == label] = target
    return {'case': case, 'raw_voxels': {str(label): int((hard == label).sum()) for label in range(35, 49)},
            'dice_with_35_42_kept': {str(label): dice(kept == label, reference == label) for label in range(35, 41)},
            'dice_with_35_42_dropped': {str(label): dice(dropped == label, reference == label) for label in range(35, 41)},
            'note': 'raw argmax before the component filter; the official label map additionally filters components'}



def precision_check(root):
    """Float16 against float32, both on MPS, for one real centred tile per architecture (first checkpoint). The tile
    is half the patch edge per axis so that the float32 network fits in memory; both precisions see the same tile."""
    import torch
    checks = [
        ('UZH PlainConvUNet (AM)', 'TopCoW-014-ct', 'UZH-ct-mr', UZH_MODELS + UZH_PARTS['AM'], 'fold_0/checkpoint_final.pth', 'nnunet'),
        ('ARG PlainConvUNet (stage 1)', 'TopCoW-014-ct', 'ARG-ct', 'opt/app/resources/weight_1', 'checkpoint_best.pth', 'arg'),
        ('KDH CT network', 'TopCoW-014-ct', 'KDH-ct', 'opt/app/resources/checkpoint/mulcls', 'fold_0/checkpoint_best.pth', 'nnunet'),
        ('KDH MR ResEncL', 'TopCoW-014-mr', 'KDH-mr', 'opt/app/resources/checkpoint/mulcls', 'fold_0/checkpoint_best.pth', 'nnunet'),
    ]
    results = []
    for name, case, image_name, folder, checkpoint, style in checks:
        model = verified_rootfs(root, image_name) / folder
        _, values, spacing, _ = source_image(root, case)
        half, details = build_predictor(model, [model / checkpoint], 'mps')
        full, _ = build_predictor(model, [model / checkpoint], 'mps', half=False)
        if style == 'arg':
            values = (values - values.mean()) / max(float(values.std()), 1e-8)
            data, _ = trilinear_input(half, values[None], spacing)
        else:
            data, _ = nnunet_input(half, values[None], spacing)
        strides = (half.configuration_manager.network_arch_init_kwargs or {}).get('strides') or [[1, 1, 1]] + [[2, 2, 2]] * 5
        divisor = np.prod(np.asarray(strides), axis=0)
        patch = [int(max(d, (edge // 2) // d * d)) for edge, d in zip(details['patch_size'], divisor)]
        start = [max((size - edge) // 2, 0) for size, edge in zip(data.shape[1:], patch)]
        tile = data[(slice(None), *[slice(a, a + edge) for a, edge in zip(start, patch)])][None].float()
        with torch.inference_mode():
            reduced = half.network.to('mps').eval()(tile.to('mps'))[0].float().cpu()
            exact = full.network.to('mps').eval()(tile.to('mps'))[0].float().cpu()
        difference = (reduced - exact).abs()
        results.append({'architecture': name, 'case': case, 'tile': list(tile.shape[2:]), 'tile_start': start,
                        'max_abs_logit_difference': float(difference.max()),
                        'p99_abs_logit_difference': float(torch.quantile(difference.flatten()[::97], 0.99)),
                        'max_abs_logit': float(exact.abs().max()),
                        'argmax_agreement': float((reduced.argmax(0) == exact.argmax(0)).float().mean())})
        print(results[-1], flush=True)
        del half, full
        torch.mps.empty_cache()
    return {'definition': ' '.join(precision_check.__doc__.split()), 'checks': results}


def report(root):
    """Collect predictions, ensembles, the frozen rule and scores into metrics.json and a Markdown table."""
    phase = safe_path(root, PHASE)
    document = {'predictions': {}, 'ensembles': {}, 'scores': {}}
    for record in sorted([*(phase / 'work').glob('*/[!.]*/predict.json'), *(phase / 'cases').glob('*/[!.]*/predict.json')]):
        data = json.loads(record.read_text())
        document['predictions'][f'{data["case"]}/{data["team"]}'] = {
            k: data[k] for k in ('seconds', 'peak_rss_bytes', 'device', 'precision', 'probability_resampling')} | {
            'stages': [{k: stage.get(k) for k in ('name', 'checkpoints', 'trainer', 'configuration', 'spacing',
                                                  'patch_size', 'tile_step', 'gaussian', 'mirroring',
                                                  'network_seconds', 'mps_driver_bytes_after_network')}
                       for stage in data['stages']]}
    for record in sorted([*(phase / 'work').glob('*/ensemble-*/ensemble.json'),
                          *(phase / 'cases').glob('*/ensemble-*/ensemble.json')]):
        document['ensembles'][f'{record.parent.parent.name}/{record.parent.name}'] = json.loads(record.read_text())
    selection = phase / 'addition-selection.json'
    if selection.is_file():
        chosen = json.loads(selection.read_text())
        document['addition_selection'] = {k: chosen[k] for k in ('rule', 'development_cases', 'selected',
                                                                 'selected_mean_objective_change',
                                                                 'mean_objective_change')}
    lines = ['| Case | Row | Dice | clDice | b0 err | b1 err | b0 / b1 predicted | floating |', '|' + ' --- |' * 8]
    for path in sorted(path for path in (phase / 'scores').glob('*.json') if not path.name.startswith('._')):
        scores = json.loads(path.read_text())
        document['scores'][path.stem] = scores
        for name, row in scores.items():
            if name == 'annotation':
                continue
            betti_row = row.get('betti', {})
            predicted = betti_row.get('predicted', betti_row)
            fmt = lambda v: '' if v is None else (f'{v:.3f}' if isinstance(v, float) else str(v))
            lines.append(f'| {path.stem} | {name} | {fmt(row.get("dice"))} | {fmt(row.get("cl_dice"))} | '
                         f'{fmt(betti_row.get("b0_error"))} | {fmt(betti_row.get("b1_error"))} | '
                         f'{predicted.get("b0")} / {predicted.get("b1")} | {row["floating"]["floating"]} |')
    atomic_json(phase / 'metrics.json', document)
    return '\n'.join(lines)



def retain(root, case):
    """Keep compressed soft probabilities, label maps and records under cases/<case>/; drop rebuildable work files."""
    work = work_folder(root, case)
    kept = kept_folder(root, case)
    for folder in sorted(path for path in work.iterdir() if path.is_dir() and not path.name.startswith('._')):
        target = kept / folder.name
        target.mkdir(parents=True, exist_ok=True)
        record = json.loads(next(folder.glob('[pe]*.json')).read_text())
        probabilities = np.load(folder / 'probabilities.npy', mmap_mode='r')
        partial = target / 'probabilities.npz.partial'
        with partial.open('wb') as stream:
            np.savez_compressed(stream, probabilities=probabilities, labels=np.asarray(record['labels'], np.uint8),
                                affine=np.asarray(nib.load(safe_path(root, CASES[case]['image'])).affine))
        os.replace(partial, target / 'probabilities.npz')
        if (folder / 'stage2.npy').is_file() and CASES[case]['modality'] == 'ct':
            with (target / 'stage2.npz.partial').open('wb') as stream:
                np.savez_compressed(stream, probabilities=np.load(folder / 'stage2.npy', mmap_mode='r'))
            os.replace(target / 'stage2.npz.partial', target / 'stage2.npz')
        # The record that marks a folder as kept (ensemble.json, predict.json) is copied last, so an
        # interrupted copy is not mistaken for a kept case.
        items = [item for item in folder.iterdir() if item.suffix in ('.json', '.gz') and not item.name.startswith('._')]
        for item in sorted(items, key=lambda item: (item.name in ('ensemble.json', 'predict.json'), item.name)):
            shutil.copyfile(item, target / item.name)
        print(f'{case}/{folder.name} kept', flush=True)
    shutil.rmtree(work, ignore_errors=True)
    return kept


def main():
    early = argparse.ArgumentParser(add_help=False)
    early.add_argument('--eden', action='append', default=[])
    early.add_argument('--study', action='append', default=[])
    early.add_argument('--series')
    known = early.parse_known_args()[0]
    patients, studies = known.eden, known.study
    root = (checked_root(None if os.environ.get(ROOT_ENV) else HDD_ROOT, expected_uuid=HDD_UUID)
            if patients or studies else None)
    if patients:
        register_eden(root, patients)
    if known.series and len(studies) != 1:
        raise SystemExit('--series names the series of exactly one --study')
    study_cases = [register_study(root, study, known.series) for study in studies]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--eden', action='append', default=[], metavar='PTNT_0000',
                        help='Register this EDEN patient (restricted, local only) as case PTNT_0000-mr; repeatable')
    parser.add_argument('--study', action='append', default=[], metavar='MANIFEST',
                        help='Register any prepared study on the data drive by its patient-anatomy.json or review '
                             'manifest (path under the data root); its case is <study folder>-ct or -mr; repeatable')
    parser.add_argument('--series', help='With one --study that has several angiographic series: the series id to use')
    sub = parser.add_subparsers(dest='command', required=True)
    displaying = sub.add_parser('display-additions', help='Write the display-only vessel candidates beside an ensemble')
    displaying.add_argument('case', nargs='+')
    displaying.add_argument('--ensemble', default='combined-base-UZH-ARG-KDH')
    studying = sub.add_parser('study', help='For each --study: run the three models and the frozen ensemble, write the '
                              'display candidates, and make the result the study\'s displayed vessels')
    studying.add_argument('--device', choices=['mps', 'cpu'], default='mps')
    studying.add_argument('--no-wire', action='store_true', help='Leave the study document unchanged')
    running = sub.add_parser('run', help='Predict every team, build the frozen combined ensemble and retain, per case')
    running.add_argument('case', nargs='+')
    running.add_argument('--device', choices=['mps', 'cpu'], default='mps')
    unpacking = sub.add_parser('unpack', help='Stream one archive once and extract code, weights and metadata')
    unpacking.add_argument('image', choices=list(ARCHIVES))
    unpacking.add_argument('--package', action='append', default=[],
                           help='Site-packages directory to extract in full, such as nnunetv2')
    predicting = sub.add_parser('predict', help='Run one team natively on one case')
    predicting.add_argument('team', choices=list(TEAMS))
    predicting.add_argument('case', choices=list(CASES))
    predicting.add_argument('--device', choices=['mps', 'cpu'], default='mps')
    combining = sub.add_parser('ensemble', help='Average teams on one case and add connected vesselFM signal')
    combining.add_argument('case', choices=list(CASES))
    combining.add_argument('--team', action='append', required=True, choices=list(TEAMS))
    combining.add_argument('--addition', help='JSON overrides of the connected-addition rule (development cases only)')
    combining.add_argument('--readd', action='store_true', help='Only reapply the frozen vesselFM rule to an ensemble')
    combining.add_argument('--combined', action='store_true', help='Use the frozen combiner instead of the plain mean')
    scoring = sub.add_parser('score', help='Score vesselFM, each team, the ensemble and the connected addition')
    scoring.add_argument('case', choices=list(CASES))
    scoring.add_argument('--team', action='append', default=[], choices=list(TEAMS))
    scoring.add_argument('--ensemble', action='append', default=[], help='Ensemble or combined folder name')
    reselecting = sub.add_parser('select-combiner', help='Attempt two: choose the team combiner on the development cases')
    reselecting.add_argument('--reselect', action='store_true',
                             help='Allow a new choice after a held-out case was scored with the current one (logged)')
    sub.add_parser('select-vesselfm', help='Choose the tuned vesselFM row on the development cases')
    checking = sub.add_parser('arg-ct-check', help='Measure ARG external carotid labels turning into CT veins')
    checking.add_argument('case', choices=[case for case in CASES if CASES[case]['modality'] == 'ct'])
    selecting = sub.add_parser('select', help='Freeze the connected-addition rule on the development cases')
    selecting.add_argument('--ensemble', required=True, help='Ensemble folder name, the same for both cases')
    retaining = sub.add_parser('retain', help='Compress and keep one case, then delete its work files')
    retaining.add_argument('case', choices=list(CASES))
    sub.add_parser('report', help='Write metrics.json and print the results table')
    sub.add_parser('precision-check', help='Float16 MPS against float32 CPU on one real tile per architecture')
    merging = sub.add_parser('merge', help='Apply already extracted layers into rootfs/')
    merging.add_argument('image', choices=list(ARCHIVES))
    args = parser.parse_args()
    root = root or checked_root(None if os.environ.get(ROOT_ENV) else HDD_ROOT, expected_uuid=HDD_UUID)
    if args.command == 'study':
        if not study_cases:
            raise SystemExit('study needs at least one --study MANIFEST')
        rule = study_rule(root)
        run_cases(root, study_cases, teams=tuple(rule['teams']), device=args.device)
        name = combined_name(rule['combiner'], rule['teams'])
        for case in study_cases:
            folder = kept_folder(root, case) / name
            require_made_from(folder / 'ensemble.json', case)
            print(case, 'ensemble at', folder.relative_to(root), flush=True)
            if not args.no_wire:
                print(case, json.dumps(wire_study(root, case, folder)), flush=True)
    elif args.command == 'run':
        unknown = [case for case in args.case if case not in CASES]
        if unknown:
            raise SystemExit(f'Unknown cases {unknown}; register EDEN patients with --eden or a study with --study')
        print(run_cases(root, args.case, device=args.device))
    elif args.command == 'display-additions':
        for case in args.case:
            receipt = display_additions(root, case, case_folder(root, case, args.ensemble))
            print(case, json.dumps({key: value for key, value in receipt.items() if key != 'rule'})[:1500], flush=True)
    elif args.command == 'unpack':
        print(unpack(root, args.image, tuple(args.package)))
    elif args.command == 'predict':
        print(json.dumps(predict(root, args.team, args.case, args.device))[:2000])
    elif args.command == 'ensemble':
        if args.addition and args.case not in DEVELOPMENT:
            raise SystemExit('--addition is for development cases only')
        if args.readd:
            name = combined_name(frozen_combiner(root), args.team) if args.combined else f'ensemble-{"-".join(args.team)}'
            folder = case_folder(root, args.case, name)
            record = add_vesselfm(root, args.case, folder, json.loads((folder / 'ensemble.json').read_text()))
        else:
            combiner = frozen_combiner(root) if args.combined else None
            if args.combined and combiner is None:
                raise SystemExit('Run select-combiner first')
            folder, record = ensemble(root, args.case, args.team, json.loads(args.addition) if args.addition else None,
                                      combiner)
        print(folder, json.dumps(record))
    elif args.command == 'score':
        if args.case in CONFIRMATION and any(entry.get('case') == args.case and entry.get('pass')
                                             for entry in heldout_log(root)):
            raise SystemExit(f'{args.case} is a confirmation case and has already been scored once')
        folders = [case_folder(root, args.case, name) for name in args.ensemble]
        result = evaluate(root, args.case, args.team, folders)
        output = safe_path(root, f'{PHASE}/scores')
        output.mkdir(parents=True, exist_ok=True)
        atomic_json(output / f'{args.case}.json', result)
        if args.case in HELD_OUT + CONFIRMATION:
            passes = [entry for entry in heldout_log(root) if entry.get('case') == args.case and entry.get('pass')]
            append_heldout(root, {'case': args.case, 'pass': len(passes) + 1, 'ensembles': list(args.ensemble),
                                  'rows': [name for name in result if name != 'annotation'],
                                  'combiner': frozen_combiner(root), 'addition': frozen_addition(root),
                                  'vesselfm_tuned': frozen_vesselfm(root)})
    elif args.command == 'select-combiner':
        atomic_json(safe_path(root, f'{PHASE}/combiner-selection.json'), select_combiner(root, reselect=args.reselect))
    elif args.command == 'select-vesselfm':
        atomic_json(safe_path(root, f'{PHASE}/vesselfm-tuning.json'), select_vesselfm(root))
    elif args.command == 'arg-ct-check':
        output = safe_path(root, f'{PHASE}/scores/arg-ct-check')
        output.mkdir(parents=True, exist_ok=True)
        atomic_json(output / f'{args.case}.json', arg_ct_check(root, args.case))
    elif args.command == 'select':
        folders = {case: case_folder(root, case, args.ensemble) for case in DEVELOPMENT}
        atomic_json(safe_path(root, f'{PHASE}/addition-selection.json'), select_addition(root, folders))
    elif args.command == 'retain':
        print(retain(root, args.case))
    elif args.command == 'precision-check':
        atomic_json(safe_path(root, f'{PHASE}/precision-check.json'), precision_check(root))
    elif args.command == 'report':
        print(report(root))
    elif args.command == 'merge':
        print(merge(safe_path(root, f'{UNPACKED}/{args.image}')))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

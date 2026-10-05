"""One authorized source-only partition; no destination coordinate conversion."""
from __future__ import annotations
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
SNAPSHOT = ROOT / 'build/resect-source-partition-6766a2a/src'
DATA = ROOT / 'data/mechanics/resect-case4-v1/RESECT/NIFTI/Case4'
TAG = DATA / 'Landmarks/Case4-beforeUS-duringUS-full.tag'
US = DATA / 'US/Case4-US-before.nii.gz'
ACQ = ROOT / 'artifacts/mechanics/resect-case4-acquisition-v1/acquisition-receipt.json'
HEADER = ROOT / 'artifacts/mechanics/resect-case4-baseline-header-qc-v1/header-receipt.json'
FIXED = {
    TAG: '9e2c3c2765bffd2914f8113f8b2e223c06b2f0640e561f03a94049752e8ddec1',
    US: '62f6b6217774653267702acd4a55b16ad5c8cc4980237afaa39e92b102240379',
    ACQ: '54c39954cefd2200935d2225bad457ae4aee191690a65d27f8c92cfdd1cbc38b',
    HEADER: '18d6e6306e38f409a314ba180d7d0a21e9a7ee02677f34a1d1fd28ea1294d692',
}


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        while block := stream.read(1024**2):
            h.update(block)
    return h.hexdigest()


def now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def save(name, value):
    with (HERE/name).open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def verify_inputs():
    files = dict(FIXED)
    release = json.loads((HERE/'root-release.json').read_text())
    files[Path(__file__).resolve()] = release['wrapper_sha256']
    files[HERE/'source-snapshot.json'] = release['source_snapshot_sha256']
    for item in json.loads((HERE/'source-snapshot.json').read_text())['files']:
        files[ROOT/item['path']] = item['sha256']
    result = {str(path.relative_to(ROOT)): sha(path) for path in files}
    if any(result[str(path.relative_to(ROOT))] != expected for path, expected in files.items()):
        raise RuntimeError('PINNED_SOURCE_OR_INPUT_CHANGED')
    return result


def child():
    start = time.monotonic()
    before = verify_inputs()
    sys.path.insert(0, str(SNAPSHOT))
    import numpy as np
    from resectionlab import mechanics_landmarks as lm
    from resectionlab import core
    for module in (lm, core):
        if not Path(module.__file__).resolve().is_relative_to(SNAPSHOT):
            raise RuntimeError('MUTABLE_SOURCE_IMPORT_PROHIBITED')

    def patient_access_guard(event, args):
        if event == 'open' and isinstance(args[0], (str, bytes, os.PathLike)):
            path = Path(os.fsdecode(args[0])).resolve()
            if path.is_relative_to(DATA) and path not in (TAG, US):
                raise RuntimeError('UNAUTHORIZED_PATIENT_FILE_ACCESS')

    sys.addaudithook(patient_access_guard)

    def forbidden(*args, **kwargs):
        raise RuntimeError('DESTINATION_COORDINATE_ACCESS_PROHIBITED')

    lm.build_forward_landmarks = forbidden
    lm.reveal_validation_landmarks = forbidden
    lm.read_registration_landmarks = forbidden
    acquisition = json.loads(ACQ.read_text())
    metadata = {Path(item['source_path']).name: item for item in acquisition['files']}
    payload = TAG.read_bytes()
    if hashlib.md5(payload).hexdigest() != '14259fc0f9ab32dfecbe1f2b8830ae86' or len(payload) != 1304:
        raise RuntimeError('TAG_PROVIDER_IDENTITY_CHANGED')
    binding = lm.LandmarkPairBinding('RESECT:Case4', lm.DISPLACEMENT_ROLE,
        FIXED[TAG], FIXED[US], metadata['Case4-US-during.nii.gz']['sha256'], frame_qc=None)
    sources = lm.parse_tag_sources(payload, binding)
    partition = lm.partition_displacement_sources(sources)
    points = sources.source_world_mm
    header = next(item['header'] for item in json.loads(HEADER.read_text())['files']
                  if item['source_path'].endswith('Case4-US-before.nii.gz'))
    affine = np.asarray(header['selected_affine'], dtype=float)
    shape = np.asarray(header['shape'])
    inv = np.linalg.inv(affine)
    voxels = points @ inv[:3,:3].T + inv[:3,3]
    inside = np.all((voxels >= -.5) & (voxels <= shape-.5), axis=1)
    selected = np.array(partition.boundary_ids)-1
    singular = np.linalg.svd(points[selected]-points[selected].mean(axis=0), compute_uv=False)
    after = verify_inputs()
    if before != after:
        raise RuntimeError('SOURCES_CHANGED_DURING_PARTITION')
    save('partition-receipt.json', {
        'status': 'partition_frozen_source_only_frame_still_unverified',
        'created_at_utc': now(), 'wrapper_sha256': sha(Path(__file__)),
        'release_sha256': sha(HERE/'root-release.json'),
        'source_snapshot_sha256': sha(HERE/'source-snapshot.json'),
        'source_and_input_hashes_before': before, 'source_and_input_hashes_after': after,
        'binding': {'patient_group': binding.patient_group, 'role': binding.role,
                    'tag_sha256': binding.tag_sha256, 'source_image_sha256': binding.source_image_sha256,
                    'destination_image_sha256_from_acquisition_metadata_only': binding.destination_image_sha256,
                    'audit_hash': binding.audit_hash, 'frame_qc': None},
        'source_hash': sources.source_hash, 'source_frame': sources.frame,
        'source_row_ids': list(sources.row_ids), 'source_world_mm': points.tolist(),
        'N': len(points), 'B_ids_in_FPS_order': list(partition.boundary_ids),
        'V_ids_original_order': list(partition.validation_ids),
        'partition_rule': partition.rule, 'partition_hash': partition.partition_hash,
        'B_rank_ratio': partition.rank_ratio, 'B_centered_singular_values_mm': singular.tolist(),
        'source_extent_mm': np.ptp(points, axis=0).tolist(),
        'header_bounds_diagnostic': {
            'interpretation': 'Assuming tag world matches original NIfTI sform world as creator describes; not independent frame acceptance',
            'source_voxel_coordinates': voxels.tolist(), 'full_cell_bounds': [[-.5]*3, (shape-.5).tolist()],
            'inside_count': int(inside.sum()),
            'outside_row_ids': [row for row, ok in zip(sources.row_ids, inside) if not ok],
            'minimum_signed_margin_voxels': float(np.min(np.minimum(voxels+.5, shape-.5-voxels))),
        },
        'source_only_eligibility_passed': True,
        'destination_coordinates_converted_or_displayed': False,
        'during_image_opened': False, 'MRI_before_landmarks_opened': False,
        'patient_image_arrays_opened': False, 'frame_qc_accepted': False,
        'forward_DTO_created': False, 'replacement_or_rule_relaxation': False,
        'runtime': {'python': sys.version, 'numpy': np.__version__,
                    'helper_path': str(Path(lm.__file__).resolve()), 'core_path': str(Path(core.__file__).resolve())},
        'elapsed_seconds': time.monotonic()-start,
        'child_peak_RSS_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
    })
    print(json.dumps({'N': len(points), 'B_ids': partition.boundary_ids,
                      'V_count': len(partition.validation_ids), 'rank_ratio': partition.rank_ratio,
                      'source_points_inside_header_grid': int(inside.sum())}))


if __name__ == '__main__':
    if sys.argv[1:] == ['--child']:
        try:
            child()
        except Exception as error:
            save('failure.json', {'status': 'failed_no_relaxation_or_replacement',
                 'type': type(error).__name__, 'reason': str(error),
                 'wrapper_sha256': sha(Path(__file__)), 'recorded_at_utc': now()})
            raise SystemExit(2)
    else:
        save('root-release.json', {'recorded_at_utc': now(), 'authorizer': '/root',
            'helper_commit': '6766a2a3efd03858bf16634cb6776e2af1f48e38',
            'header_QC_commit': '77f41c8', 'access_contract_commit': 'cb9f71b',
            'allowed': 'First-triplet source coordinates only, fixed sixB partition, header bounds',
            'closed': ['all destination coordinates includingB', 'during image', 'MRI-before pair', 'all image arrays'],
            'wall_seconds': 55, 'sampled_parent_child_RSS_limit_bytes': 1024**3,
            'source_snapshot_sha256': sha(HERE/'source-snapshot.json'),
            'wrapper_sha256': sha(Path(__file__))})
        start, peak, failure = time.monotonic(), 0, None
        with (HERE/'run.log').open('x') as log:
            process = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), '--child'], stdout=log, stderr=log)
            while process.poll() is None:
                result = subprocess.run(['ps','-o','rss=','-p',f'{os.getpid()},{process.pid}'],
                                        capture_output=True,text=True,timeout=2,check=False)
                rss = sum(int(line)*1024 for line in result.stdout.splitlines() if line.strip())
                peak = max(peak, rss)
                if time.monotonic()-start > 55 or rss > 1024**3:
                    failure = 'WALL_OR_RSS_LIMIT'
                    process.kill()
                    break
                time.sleep(.025)
            code = process.wait(timeout=2)
        save('resource-receipt.json', {'status': 'passed' if code == 0 and failure is None else 'failed',
             'exit_code': code, 'failure': failure, 'elapsed_seconds': time.monotonic()-start,
             'maximum_sampled_parent_child_RSS_bytes': peak, 'wall_limit_seconds': 55,
             'RSS_limit_bytes': 1024**3, 'sampling_caveat': 'RSS is sampled, not continuously enforced',
             'wrapper_sha256': sha(Path(__file__)), 'log_sha256': sha(HERE/'run.log'),
             'partition_receipt_sha256': sha(HERE/'partition-receipt.json') if (HERE/'partition-receipt.json').exists() else None})
        print(json.dumps({'exit_code': code, 'failure': failure, 'elapsed_seconds': time.monotonic()-start}))
        raise SystemExit(code if code else int(failure is not None))

"""Open an exact native-wall source study without replacing a live rehearsal."""

from contextlib import ExitStack
from dataclasses import dataclass
import hashlib
from pathlib import Path
import shutil
import tempfile
from urllib.parse import unquote, urlparse
import uuid

import numpy as np

from .anatomy_local_wall_review import native_wall_alignment_review
from .anatomy_package import open_package
from .anatomy_sampling import sample_layer
from .core import thaw_json
from .patient_anatomy import PatientImagingCollection, world
from .surgical_patient import _check_layer_export, _resample, _source_registration, _wall_layer_issue
from .surgical_patient_wall import SCHEMA, native_wall_grid


@dataclass
class SourceInspection:
    key: str
    collection: PatientImagingCollection
    manifest_sha: str
    workspace: Path
    scans: dict


def _manifest_hash(path):
    with path.open('rb') as stream:
        contents = stream.read(16 * 1024**2 + 1)
    if len(contents) > 16 * 1024**2:
        raise ValueError('The selected native study manifest exceeds the metadata limit')
    return hashlib.sha256(contents).hexdigest()


def _check_original(collection, original, check):
    digest = hashlib.sha256()
    with collection.path(original['path']).open('rb') as stream:
        while block := stream.read(4 * 1024**2):
            check()
            digest.update(block)
    if digest.hexdigest() != original['sha256']:
        raise ValueError('The acquired source file differs from the saved rehearsal')


def _source_manifest_hint(ref):
    uri = urlparse(ref.uri)
    path = Path(unquote(uri.path))
    if (uri.scheme == 'file' and uri.netloc in ('', 'localhost') and not uri.query and not uri.fragment
            and path.is_absolute() and path.name == 'patient-anatomy.json'):
        return path
    return None


def _verify_study(case, adapter, wall, refs, collection, check):
    if (collection.identity() != adapter['sourceCollectionHash']
            or collection.document != adapter['sourceManifest']
            or collection.document['patient_id'] != adapter['sourcePatient']):
        raise ValueError('Choose the exact study revision used to prepare this rehearsal')
    reference = collection.series(collection.document['reference_series'])
    rows = {role: collection.layer(saved['layerId']) for role, saved in adapter['sourceLayers'].items()}
    brain, vessels = rows['brain'], rows['vessels']
    if (wall['layerId'], wall['sourceSeries'], wall['sourceMaskSha256'], wall['sourceScoresSha256']) != (
            vessels['id'], vessels['source_series'], vessels['data']['sha256'], vessels.get('probability', {}).get('sha256')):
        raise ValueError('The saved wall differs from its source layer and score field')
    scoped = adapter.get('nativeWallAlignmentReview')
    scans = {}
    frames = {}
    for sid in dict.fromkeys([reference['id'], *(row['source_series'] for row in rows.values())]):
        check()
        scan = collection.series(sid)
        original = scan['original']
        ref = refs.get(sid)
        if not ref or not ref.sha256 == original['sha256'] or ref.provenance != 'observed' or ref.native_frame != 'RAS+':
            raise ValueError('The acquired source binding differs from the saved rehearsal')
        _check_original(collection, original, check)
        transform, unreviewed = _source_registration(collection, scan)
        if unreviewed and (sid != vessels['source_series'] or not scoped):
            raise ValueError('The saved rehearsal has no checked correspondence to this source')
        frames[sid] = transform
        center = np.asarray(wall['centerRasMm'])
        if unreviewed:
            center = world(np.linalg.inv(transform), [center])[0]
        scans[sid] = {'seriesId': sid, 'modality': scan['modality'], 'centerRasMm': center.tolist(),
                      'independent': bool(unreviewed), 'layers': []}
    grid = native_wall_grid(collection, vessels, frames[vessels['source_series']], wall['centerRasMm'])
    if (wall['sourceShape'] != list(grid.source_shape) or wall['sourceStartIJK'] != grid.start.tolist()
            or wall['sourceImageSha256'] != grid.source_sha256 or not np.array_equal(grid.affine, case.affine)
            or not np.array_equal(grid.scores, case.vascular_probabilities.get('vessels'))):
        raise ValueError('The angiography samples differ from the saved rehearsal field')
    if scoped and scoped != native_wall_alignment_review(collection, brain, vessels, grid):
        raise ValueError('The alignment review differs from the saved rehearsal field')
    if not np.array_equal(case.mri, _resample(collection.array(reference['data']),
                          collection.patient_affine(reference), grid.affine, grid.mask.shape)):
        raise ValueError('The acquired reference samples differ from the saved rehearsal field')
    for role, row in rows.items():
        check()
        _check_layer_export(collection, role, row)
        if issue := _wall_layer_issue(role, row):
            raise ValueError(issue)
        saved = adapter['sourceLayers'][role]
        if (saved['sourceArray'], saved['sourceAffine'], saved['revision']) != (row['data'], row['affine'], row.get('revision', 0)):
            raise ValueError('A selected anatomy layer differs from the saved rehearsal')
        source_sid = row['source_series']
        if role == 'vessels':
            mask = grid.mask
        else:
            source_frame = frames[source_sid]
            mask, _ = sample_layer(collection, row, grid.affine, grid.mask.shape, check, {}, frames=(
                source_frame @ np.asarray(row['affine']), source_frame @ np.asarray(collection.series(source_sid)['affine'])))
        if not np.array_equal(mask, case.compartments.get(role)):
            raise ValueError('Anatomy samples differ from the saved rehearsal field')
        scans[source_sid]['layers'].append({'id': row['id'], 'kind': row['kind'], 'label': row['label']})
    return scans


def open_source_inspection(session, case, selected, check):
    adapter = thaw_json(case.metadata.get('surgical_adapter') or {})
    wall = adapter.get('nativeWall') or {}
    refs = {ref.source_id: ref for ref in case.source_refs}
    manifest_ref = refs.get('source-manifest')
    if (adapter.get('type') != 'native-patient-research-derivative' or wall.get('schema') != SCHEMA
            or len(refs) != len(case.source_refs) or not manifest_ref or manifest_ref.provenance != 'observed'
            or not manifest_ref.sha256 or manifest_ref.sha256 != adapter.get('sourceManifestSha256')):
        raise ValueError('This rehearsal has no verified native-wall source binding')
    path = Path(selected) if selected else _source_manifest_hint(manifest_ref)
    if path is None or (not selected and not path.is_file()):
        return None
    if (not path.is_absolute() or not path.is_file()
            or not (path.name == 'patient-anatomy.json' or path.suffix == '.rsworkspace')):
        raise ValueError('Choose the saved patient-anatomy.json or portable study')
    try:
        with ExitStack() as cleanup:
            workspace = Path(tempfile.mkdtemp(prefix='native-wall-source-', dir=session.transfers.root))
            cleanup.callback(shutil.rmtree, workspace)
            manifest = open_package(path, workspace / 'study', check=check) if path.suffix == '.rsworkspace' else path
            if _manifest_hash(manifest) != manifest_ref.sha256:
                raise ValueError('Choose the exact study revision used to prepare this rehearsal')
            collection = PatientImagingCollection(manifest.parent)
            scans = _verify_study(case, adapter, wall, refs, collection, check)
            if _manifest_hash(manifest) != manifest_ref.sha256:
                raise ValueError('The native study changed during source verification')
            check()
            current = getattr(session, 'native_wall_source', None)
            if current is not None:
                close_source_inspection(session, current.key)
            key = str(uuid.uuid4())
            session.native_wall_source = SourceInspection(key, collection, manifest_ref.sha256, workspace, scans)
            cleanup.pop_all()
    except (OSError, ValueError) as error:
        if selected:
            raise ValueError(str(error)) from error
        return None
    return {'collectionId': key, 'snapshotId': collection.identity(), 'caseHash': case.semantic_hash,
            'centerRasMm': wall['centerRasMm'], 'cropStartIJK': wall['sourceStartIJK'], 'cropShape': list(case.mri.shape),
            'locallyReviewed': bool(adapter.get('nativeWallAlignmentReview')), 'scans': list(scans.values())}


def close_source_inspection(session, key):
    inspection = getattr(session, 'native_wall_source', None)
    if inspection and inspection.key == key:
        session.native_wall_source = None
        cache = getattr(session, 'anatomy_view_cache', None)
        if cache and cache['key'][0][0] == key:
            session.anatomy_view_cache = None
            session.anatomy_transfer_paths.clear()
        shutil.rmtree(inspection.workspace)

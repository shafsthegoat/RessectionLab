"""Development-role evidence for older acquired native collections.

This reads trusted repository declarations, never an authority supplied in a
case. Native manifests are not rewritten. Mirror attribution does not establish
official-source equivalence, independent patients, anatomical QC or task access.
The older staged-experiment gate remains unchanged.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

from .cohort import _Groups
from .core import semantic_digest
from .patient_anatomy import digest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
REGISTRY_PATH = PROJECT_ROOT/'manifests/cohort_registry.json'


def _require(value, message):
    if not value: raise ValueError('SOURCE_ROLE: '+message)


def _document(path):
    try:
        data = path.read_bytes(); value = json.loads(data)
    except (OSError, ValueError) as error:
        raise ValueError('SOURCE_ROLE: source declaration is unavailable or invalid: '+str(path)) from error
    _require(type(value) is dict, 'source declaration must be an object')
    return value, hashlib.sha256(data).hexdigest()


def _record(registry, patient):
    _require(registry.get('schema_version') == 1, 'unsupported registry schema')
    records = registry.get('records', []); by_id = {r['record_id']: r for r in records}
    _require(len(by_id) == len(records), 'duplicate registry record')
    matches = [r for r in records if type(r.get('identity')) is str
               and (r['identity'] == patient or r['identity'].partition(':')[2] == patient)]
    _require(len(matches) == 1, 'one exact registered source identity is required')
    selected = matches[0]; identities = _Groups(); groups = _Groups(); seen = {}
    # Even an unconfirmed alias is grouped conservatively, as in cohort.py.
    # Only declarations are inspected here; no clinical or evaluation payloads.
    for link in registry.get('identity_links', []):
        names = link.get('identities', [])
        _require(len(names) >= 2 and all(type(n) is str and ':' in n for n in names), 'invalid identity link')
        for name in names[1:]: identities.join(names[0], name)
    for key, row in by_id.items():
        groups.find(key)
        if row.get('identity'):
            identity = identities.find(row['identity'])
            if identity in seen: groups.join(key, seen[identity])
            seen[identity] = key
        parent = row.get('derivative_of')
        if parent in by_id: groups.join(key, parent)
    members = [r for key, r in by_id.items() if groups.find(key) == groups.find(selected['record_id'])]
    for row in members:
        _require(row.get('outer_split') == 'development' and 'final_scoring' not in row.get('uses', []),
                 'registered person/alias/derivative has a conflicting or reserved role')
        ancestors = set(); current = row
        while current.get('derivative_of') is not None:
            key = current['record_id']
            _require(key not in ancestors and current['derivative_of'] in by_id, 'invalid derivative lineage')
            ancestors.add(key); current = by_id[current['derivative_of']]
    _require({'global_development', 'case_optimization'} <= set(selected.get('uses', [])),
             'source is not declared for development and case optimization')
    _require(selected.get('identity_status') in ('mirror_attributed', 'verified_primary'), 'source identity is unassigned')
    return selected, sorted(r['record_id'] for r in members)


def _registry_role(adapter):
    manifest = adapter['sourceManifest']; patient = adapter['sourcePatient']
    _require(manifest.get('patient_id') == patient and manifest.get('snapshot_id') == adapter.get('sourceCollectionHash')
             == digest({k: v for k, v in manifest.items() if k != 'snapshot_id'}), 'native source identity differs')
    registry, registry_sha = _document(REGISTRY_PATH)
    record, members = _record(registry, patient)
    relative = record.get('source_manifest')
    _require(type(relative) is str and not Path(relative).is_absolute() and '..' not in Path(relative).parts,
             'unsafe acquisition declaration path')
    path = (PROJECT_ROOT/relative).resolve()
    _require(path.is_relative_to(PROJECT_ROOT.resolve()), 'acquisition declaration escapes repository')
    acquired, acquired_sha = _document(path)
    _require(acquired.get('schema_version') == '1.0' and acquired.get('case_id') == patient
             and acquired.get('patient_cluster_id') == patient
             and acquired.get('visit') == record.get('visit_id')
             and acquired.get('source_collection', {}).get('name') == record.get('collection_id'),
             'acquisition person, collection or visit differs')
    _require(acquired.get('split', {}).get('role') == 'development_demo'
             and acquired.get('split', {}).get('outer_test_eligible') is False, 'acquisition role is not exposed development')
    mirror = acquired.get('mirror', {})
    _require(type(mirror.get('revision')) is str and bool(mirror['revision'])
             and type(mirror.get('repository')) is str and bool(mirror['repository']), 'pinned mirror identity is missing')
    files = acquired.get('files', []); originals = []
    def join(original, role, provenance, modality=None):
        _require(type(original) is dict and type(original.get('sha256')) is str
                 and len(original['sha256']) == 64 and all(c in '0123456789abcdef' for c in original['sha256']),
                 'original source identity is missing')
        rows = [r for r in files if r.get('sha256') == original['sha256']]
        _require(len(rows) == 1 and rows[0].get('role') == role and rows[0].get('provenance') == provenance,
                 'native original does not uniquely match acquisition role and hash')
        row = rows[0]
        _require(Path(row.get('source_remote_path', '')).name == original.get('name')
                 and type(row.get('size_bytes')) is int and row['size_bytes'] > 0,
                 'acquired original filename or size declaration differs')
        joined = {'sha256': original['sha256'], 'name': original['name'], 'role': role, 'sizeBytes': row['size_bytes']}
        if modality is not None:
            def canonical(value):
                _require(type(value) is str and bool(value), 'imaging modality is missing')
                return {'T1C': 'T1CE'}.get(value.upper(), value.upper())
            _require(canonical(modality) == canonical(row.get('modality')), 'native/acquisition imaging modality differs')
            joined.update(nativeModality=modality, acquisitionModality=row['modality'])
        originals.append(joined)
    series = manifest.get('series', [])
    _require(bool(series) and all(r.get('patient_id') == patient for r in series), 'native imaging identity differs')
    for row in series:
        _require(type(row.get('modality')) is str, 'native imaging modality is missing')
        join(row.get('original'), 'structural_mri', 'observed_preprocessed_imaging', row['modality'])
    supplied = [r for r in adapter['sourceLayers'].values() if r.get('provenance', {}).get('kind') == 'observed_source_annotation']
    _require(bool(supplied), 'supplied annotation source is required')
    for row in supplied: join(row['provenance'].get('source'), 'supplied_tumor_annotation', 'observed_source_annotation')
    proof = {'schema': 'resectionlab.registry-acquired-source-role.v1', 'registrySha256': registry_sha,
        'recordId': record['record_id'], 'personRecords': members, 'acquisitionSha256': acquired_sha,
        'acquisitionPath': relative, 'sourceCollectionHash': adapter['sourceCollectionHash'],
        'sourceManifestSha256': adapter['sourceManifestSha256'], 'identity': record['identity'],
        'identityStatus': record['identity_status'], 'mirror': {'repository': mirror['repository'], 'revision': mirror['revision']},
        'originals': sorted(originals, key=lambda r: (r['sha256'], r['role'])),
        'primarySourceEquivalence': 'unverified',
        'scope': 'Existing DEVELOPMENT role and source-declaration joins only; native bytes, anatomical review, post-exposure task and patient independence require separate checks.'}
    return {'sourceRole': 'DEVELOPMENT', 'patientId': patient,
            'declarationSha256': semantic_digest(proof), 'roleQualification': proof}


def development_role(adapter, *, declarations_root):
    """Preserve the exact old path when any native mirror revision is declared.

    A broken/conflicting modern declaration never falls back to the registry.
    Older manifests use current trusted registry/acquisition bytes, not a role or
    self-sealed certificate attached by the caller. Nothing is persisted here.
    """
    series = adapter.get('sourceManifest', {}).get('series', [])
    if any(r.get('source_mirror_revision') is not None for r in series):
        from .surgical_staged_planning import require_staged_development
        return require_staged_development(SimpleNamespace(metadata={'surgical_adapter': adapter}), declarations_root)
    try: return _registry_role(adapter)
    except (KeyError, TypeError, AttributeError) as error:
        raise ValueError('SOURCE_ROLE: incomplete source or registry declaration') from error

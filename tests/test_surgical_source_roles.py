"""Metadata-only authority controls; generated identities are not patient data."""
import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from resectionlab import surgical_source_roles as roles
from resectionlab.core import semantic_digest
from resectionlab.patient_anatomy import digest
from resectionlab.surgical_staged_planning import require_staged_development


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True))


def seal(adapter):
    manifest = adapter['sourceManifest']
    manifest['snapshot_id'] = digest({k: v for k, v in manifest.items() if k != 'snapshot_id'})
    adapter['sourceCollectionHash'] = manifest['snapshot_id']
    adapter['sourceManifestSha256'] = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()


@pytest.fixture
def authority(tmp_path, monkeypatch):
    patient = 'generated-person-A'
    registry_path = tmp_path/'manifests/cohort_registry.json'
    acquisition_path = tmp_path/'manifests/acquired.json'
    declaration_root = tmp_path/'downloaded'
    declaration_root.mkdir()
    record = {'record_id': 'source-A', 'collection_id': 'generated-collection',
        'identity': 'generated-collection:'+patient, 'identity_status': 'mirror_attributed',
        'visit_id': 'baseline_preoperative', 'derivative_of': None, 'outer_split': 'development',
        'uses': ['global_development', 'case_optimization'], 'source_manifest': 'manifests/acquired.json',
        'inputs': [{'kind': 'outcome', 'used_by_primary_optimizer': False,
                    'payload_path': 'reserved/do-not-read-outcome.json'}]}
    registry = {'schema_version': 1, 'records': [record], 'identity_links': [],
        'evidence': [{'id': 'not-needed-for-role', 'path': 'reserved/do-not-read-evaluation.json'}]}
    originals = [dict(name='scan-T1.nii.gz', sha256='1'*64), dict(name='scan-T2.nii.gz', sha256='2'*64)]
    supplied = dict(name='source-labels.nii.gz', sha256='3'*64)
    acquisition = {'schema_version': '1.0', 'case_id': patient, 'patient_cluster_id': patient,
        'visit': record['visit_id'], 'source_collection': {'name': record['collection_id']},
        'split': {'role': 'development_demo', 'outer_test_eligible': False},
        'mirror': {'repository': 'generated/source', 'revision': 'pinned-generated-revision'},
        'files': [dict(source_remote_path=patient+'/'+o['name'], sha256=o['sha256'],
                       role='structural_mri', provenance='observed_preprocessed_imaging', size_bytes=100+i,
                       modality=('T1', 'T2')[i]) for i, o in enumerate(originals)] +
                 [dict(source_remote_path=patient+'/'+supplied['name'], sha256=supplied['sha256'],
                       role='supplied_tumor_annotation', provenance='observed_source_annotation', size_bytes=53)]}
    adapter = {'type': 'native-patient-research-derivative', 'sourcePatient': patient,
        'sourceManifest': {'schema': 'patient-anatomy-v1', 'patient_id': patient,
            'study_timepoint': 'preoperative', 'series': [dict(id=('T1', 'T2')[i], modality=('T1', 'T2')[i],
                 patient_id=patient, original=dict(o, path='originals/'+o['sha256']+'/'+o['name']))
                 for i, o in enumerate(originals)], 'layers': []},
        'sourceLayers': {'tumor': {'provenance': {'kind': 'observed_source_annotation',
             'source': dict(supplied, path='originals/'+supplied['sha256']+'/'+supplied['name'])}},
             'brain': {'provenance': {'model': 'generated-model',
                 'inference_receipt': 'private/do-not-read-inference.json'}}}}
    seal(adapter)
    monkeypatch.setattr(roles, 'PROJECT_ROOT', tmp_path)
    monkeypatch.setattr(roles, 'REGISTRY_PATH', registry_path)
    write(registry_path, registry); write(acquisition_path, acquisition)
    return SimpleNamespace(root=tmp_path, registry_path=registry_path, acquisition_path=acquisition_path,
        declarations=declaration_root, registry=registry, acquisition=acquisition, adapter=adapter,
        record=record, patient=patient)


def resolve(a):
    return roles.development_role(a.adapter, declarations_root=a.declarations)


def test_exact_external_authority_joins_without_rewriting_native_or_loading_payloads(authority, monkeypatch):
    a = authority; before = copy.deepcopy(a.adapter); reads = []
    original = Path.read_bytes
    allowed = {a.registry_path, a.acquisition_path}
    def guarded(path):
        assert path in allowed, 'Only declared authority metadata may be read'
        reads.append(path)
        return original(path)
    monkeypatch.setattr(Path, 'read_bytes', guarded)
    result = resolve(a); proof = result['roleQualification']
    assert reads == [a.registry_path, a.acquisition_path]
    assert a.adapter == before
    assert result['sourceRole'] == 'DEVELOPMENT' and result['patientId'] == a.patient
    assert result['declarationSha256'] == semantic_digest(proof)
    assert proof['sourceCollectionHash'] == before['sourceCollectionHash']
    assert proof['sourceManifestSha256'] == before['sourceManifestSha256']
    assert proof['registrySha256'] == hashlib.sha256(original(a.registry_path)).hexdigest()
    assert proof['acquisitionSha256'] == hashlib.sha256(original(a.acquisition_path)).hexdigest()
    assert proof['personRecords'] == ['source-A'] and proof['primarySourceEquivalence'] == 'unverified'
    assert {(r['sha256'], r['role']) for r in proof['originals']} == {
        ('1'*64, 'structural_mri'), ('2'*64, 'structural_mri'), ('3'*64, 'supplied_tumor_annotation')}
    assert 'outcome' not in json.dumps(proof) and 'inference' not in json.dumps(proof)


@pytest.mark.parametrize('relationship', ['alias', 'alias-chain', 'derivative', 'ancestor'])
@pytest.mark.parametrize('conflict', ['final_evaluation', 'excluded', 'unassigned', 'final_scoring'])
def test_reserved_or_conflicting_person_group_cannot_be_requalified(authority, relationship, conflict):
    a = authority
    other = dict(a.record, record_id='related', identity='other:generated-relative', derivative_of=None)
    other['outer_split'] = 'development' if conflict == 'final_scoring' else conflict
    if conflict == 'final_scoring': other['uses'] = ['case_optimization', 'final_scoring']
    a.registry['records'].append(other)
    if relationship.startswith('alias'):
        names = [a.record['identity'], other['identity']]
        if relationship == 'alias-chain':
            a.registry['identity_links'] = [{'identities': [names[0], 'unknown:unconfirmed-alias']},
                                           {'identities': ['unknown:unconfirmed-alias', names[1]]}]
        else: a.registry['identity_links'] = [{'identities': names, 'confirmed': False}]
    elif relationship == 'derivative': other['derivative_of'] = a.record['record_id']
    else: a.record['derivative_of'] = other['record_id']
    write(a.registry_path, a.registry)
    with pytest.raises(ValueError, match='conflicting or reserved'):
        resolve(a)


def test_unrelated_reserved_record_is_not_read_as_a_payload_or_counted_as_same_person(authority):
    a = authority
    a.registry['records'].append(dict(a.record, record_id='unrelated', identity='other:separate-person',
                                    outer_split='final_evaluation', source_manifest='reserved/never-read.json'))
    write(a.registry_path, a.registry)
    assert resolve(a)['roleQualification']['personRecords'] == ['source-A']


@pytest.mark.parametrize('mutation', ['duplicate-record-id', 'same-person-two-records', 'ambiguous-bare-id',
                                      'dangling-parent', 'cycle', 'unknown-identity', 'no-optimization',
                                      'no-global-development', 'unsupported-schema'])
def test_registry_ambiguity_or_broken_authority_refuses(authority, mutation):
    a = authority
    if mutation == 'duplicate-record-id': a.registry['records'].append(copy.deepcopy(a.record))
    elif mutation == 'same-person-two-records': a.registry['records'].append(dict(a.record, record_id='duplicate-person'))
    elif mutation == 'ambiguous-bare-id': a.registry['records'].append(dict(a.record, record_id='other', identity='other:'+a.patient))
    elif mutation == 'dangling-parent': a.record['derivative_of'] = 'missing'
    elif mutation == 'cycle': a.record['derivative_of'] = a.record['record_id']
    elif mutation == 'unknown-identity': a.record['identity_status'] = 'unknown'
    elif mutation == 'no-optimization': a.record['uses'] = ['global_development']
    elif mutation == 'no-global-development': a.record['uses'] = ['case_optimization']
    else: a.registry['schema_version'] = 2
    write(a.registry_path, a.registry)
    with pytest.raises(ValueError): resolve(a)


@pytest.mark.parametrize('mutation', ['image-hash', 'image-filename', 'image-role', 'image-provenance',
    'annotation-hash', 'annotation-filename', 'annotation-role', 'annotation-provenance', 'duplicate-hash',
    'missing-supplied', 'mixed-patient', 'snapshot', 'adapter-patient', 'zero-size', 'bool-size'])
def test_exact_native_acquired_joins_cannot_be_borrowed(authority, mutation):
    a = authority
    if mutation.startswith('image-') or mutation.startswith('annotation-'):
        row = a.acquisition['files'][0 if mutation.startswith('image-') else -1]
        what = mutation.partition('-')[2]
        row[{'hash':'sha256', 'filename':'source_remote_path', 'role':'role', 'provenance':'provenance'}[what]] = {
            'hash':'f'*64, 'filename':'different-file.nii.gz', 'role':'other-role', 'provenance':'model-estimate'}[what]
    elif mutation == 'duplicate-hash': a.acquisition['files'].append(copy.deepcopy(a.acquisition['files'][0]))
    elif mutation == 'missing-supplied': a.adapter['sourceLayers'].pop('tumor')
    elif mutation == 'mixed-patient': a.adapter['sourceManifest']['series'][1]['patient_id'] = 'another-person'
    elif mutation == 'snapshot': a.adapter['sourceCollectionHash'] = '0'*64
    elif mutation == 'adapter-patient': a.adapter['sourcePatient'] = 'another-person'
    elif mutation == 'zero-size': a.acquisition['files'][0]['size_bytes'] = 0
    elif mutation == 'bool-size': a.acquisition['files'][0]['size_bytes'] = True
    if mutation not in ('snapshot', 'adapter-patient'): seal(a.adapter)
    write(a.acquisition_path, a.acquisition)
    with pytest.raises(ValueError): resolve(a)


@pytest.mark.parametrize('native,acquired', [('T1', 'T2'), ('T1CE', 'T1'), ('T2', 'FLAIR'),
    (None, 'T1'), ('', 'T1'), (7, 'T1'), ('T1', None), ('T1', ''), ('T1', False),
    ('T1-weighted', 'T1'), ('T1 CE', 'T1CE'), ('T1C ', 'T1CE')])
def test_missing_mismatched_or_invented_modality_alias_refuses(authority, native, acquired):
    a = authority
    a.adapter['sourceManifest']['series'][0]['modality'] = native
    a.acquisition['files'][0]['modality'] = acquired
    seal(a.adapter); write(a.acquisition_path, a.acquisition)
    with pytest.raises(ValueError, match='modality'):
        resolve(a)


@pytest.mark.parametrize('native,acquired', [('T1', 't1'), ('t2', 'T2'), ('T1CE', 'T1c'),
    ('t1c', 't1ce'), ('T1c', 'T1CE'), ('flair', 'FLAIR')])
def test_case_insensitive_modality_and_only_explicit_t1c_spelling_alias_join(authority, native, acquired):
    a = authority
    a.adapter['sourceManifest']['series'][0]['modality'] = native
    a.acquisition['files'][0]['modality'] = acquired
    seal(a.adapter); write(a.acquisition_path, a.acquisition)
    result = resolve(a)
    row = next(r for r in result['roleQualification']['originals'] if r['sha256'] == '1'*64)
    assert row['nativeModality'] == native and row['acquisitionModality'] == acquired
    annotation = next(r for r in result['roleQualification']['originals'] if r['sha256'] == '3'*64)
    assert 'nativeModality' not in annotation and 'acquisitionModality' not in annotation


@pytest.mark.parametrize('checksum', ['g'*64, 'A'*64, '1'*63, '1'*65, '', None])
def test_matching_invalid_original_checksum_declarations_do_not_authorize(authority, checksum):
    a = authority
    a.adapter['sourceManifest']['series'][0]['original']['sha256'] = checksum
    a.acquisition['files'][0]['sha256'] = checksum
    seal(a.adapter); write(a.acquisition_path, a.acquisition)
    with pytest.raises(ValueError, match='original source identity'):
        resolve(a)


@pytest.mark.parametrize('mutation', ['case', 'cluster', 'visit', 'collection', 'role', 'test-eligible',
                                     'revision', 'repository', 'schema'])
def test_wrong_acquisition_identity_or_role_never_authorizes(authority, mutation):
    a = authority; value = a.acquisition
    if mutation == 'case': value['case_id'] = 'another-person'
    elif mutation == 'cluster': value['patient_cluster_id'] = 'another-person'
    elif mutation == 'visit': value['visit'] = 'postoperative'
    elif mutation == 'collection': value['source_collection']['name'] = 'different-collection'
    elif mutation == 'role': value['split']['role'] = 'evaluation'
    elif mutation == 'test-eligible': value['split']['outer_test_eligible'] = True
    elif mutation == 'revision': value['mirror']['revision'] = ''
    elif mutation == 'repository': value['mirror']['repository'] = None
    else: value['schema_version'] = '2.0'
    write(a.acquisition_path, value)
    with pytest.raises(ValueError): resolve(a)


@pytest.mark.parametrize('path', ['/tmp/not-authority.json', '../outside.json', 'manifests/../../outside.json'])
def test_unsafe_acquisition_path_is_refused_before_open(authority, path, monkeypatch):
    a = authority; a.record['source_manifest'] = path; write(a.registry_path, a.registry)
    original = Path.read_bytes; reads = []
    def guarded(file):
        reads.append(file); assert file == a.registry_path
        return original(file)
    monkeypatch.setattr(Path, 'read_bytes', guarded)
    with pytest.raises(ValueError, match='unsafe acquisition'): resolve(a)
    assert reads == [a.registry_path]


def test_symlink_cannot_escape_trusted_repository(authority, tmp_path_factory):
    a = authority; outside = tmp_path_factory.mktemp('outside-role-authority')/'acquired.json'
    write(outside, a.acquisition)
    link = a.root/'manifests/escape.json'; link.symlink_to(outside)
    a.record['source_manifest'] = 'manifests/escape.json'; write(a.registry_path, a.registry)
    with pytest.raises(ValueError, match='escapes repository'): resolve(a)


def test_authority_is_fresh_and_caller_receipt_never_replaces_it(authority):
    a = authority; first = resolve(a)
    a.adapter['roleQualification'] = copy.deepcopy(first)
    a.adapter['sourceRole'] = 'DEVELOPMENT'
    a.acquisition['declaration_note'] = 'Changed trusted authority bytes'
    write(a.acquisition_path, a.acquisition)
    second = resolve(a)
    assert second['declarationSha256'] != first['declarationSha256']
    assert second['roleQualification']['registrySha256'] == first['roleQualification']['registrySha256']
    assert second['roleQualification']['acquisitionSha256'] != first['roleQualification']['acquisitionSha256']
    a.record['outer_split'] = 'final_evaluation'; write(a.registry_path, a.registry)
    with pytest.raises(ValueError, match='reserved role'): resolve(a)
    a.record['outer_split'] = 'development'; write(a.registry_path, a.registry)
    a.acquisition_path.unlink()
    with pytest.raises(ValueError, match='unavailable'): resolve(a)


@pytest.mark.parametrize('which', ['registry', 'acquisition'])
@pytest.mark.parametrize('content', ['{bad-json', '[]', 'null'])
def test_invalid_authority_cannot_be_replaced_with_adapter_claim(authority, which, content):
    a = authority
    a.adapter['roleQualification'] = {'sourceRole': 'DEVELOPMENT'}
    (a.registry_path if which == 'registry' else a.acquisition_path).write_text(content)
    with pytest.raises(ValueError): resolve(a)


@pytest.mark.parametrize('revisions', [['modern', None], ['modern', 'different'], ['', ''], ['modern', 'modern']])
@pytest.mark.parametrize('declaration', ['missing', 'reserved', 'wrong-revision', 'ambiguous'])
def test_broken_modern_declaration_never_falls_back_to_older_registry(authority, revisions, declaration, monkeypatch):
    a = authority
    for row, revision in zip(a.adapter['sourceManifest']['series'], revisions):
        if revision is not None: row['source_mirror_revision'] = revision
    seal(a.adapter)
    value = {'role': 'DEVELOPMENT', 'cases': [a.patient], 'mirror_revision': 'modern'}
    if declaration == 'reserved': value['role'] = 'EVALUATION'
    if declaration == 'wrong-revision': value['mirror_revision'] = 'other'
    if declaration != 'missing': write(a.declarations/'one/development-declaration.json', value)
    if declaration == 'ambiguous': write(a.declarations/'two/development-declaration.json', value)
    def forbidden(_): raise AssertionError('Modern source attempted older registry fallback')
    monkeypatch.setattr(roles, '_registry_role', forbidden)
    with pytest.raises(ValueError): resolve(a)


def test_valid_modern_path_preserves_exact_staged_gate_result(authority, monkeypatch):
    a = authority
    for row in a.adapter['sourceManifest']['series']: row['source_mirror_revision'] = 'modern'
    seal(a.adapter)
    write(a.declarations/'one/development-declaration.json',
          {'role': 'DEVELOPMENT', 'cases': [a.patient], 'mirror_revision': 'modern'})
    expected = require_staged_development(SimpleNamespace(metadata={'surgical_adapter': a.adapter}), a.declarations)
    def forbidden(_): raise AssertionError('Modern source attempted older registry fallback')
    monkeypatch.setattr(roles, '_registry_role', forbidden)
    assert resolve(a) == expected and 'roleQualification' not in expected


def test_old_manifest_does_not_change_the_staged_experiment_gate(authority):
    a = authority
    assert resolve(a)['sourceRole'] == 'DEVELOPMENT'
    with pytest.raises(ValueError, match='source-bound declared DEVELOPMENT'):
        require_staged_development(SimpleNamespace(metadata={'surgical_adapter': a.adapter}), a.declarations)

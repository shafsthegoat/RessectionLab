"""Generated metadata controls only; no original image/annotation reads."""
import copy
import hashlib
import json
from types import SimpleNamespace

import pytest

from resectionlab import surgical_development_candidates as candidates
from resectionlab import surgical_source_roles
from resectionlab.patient_anatomy import digest


def write(path, value):
    path.write_text(json.dumps(value, sort_keys=True))
    return {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


def seal(a):
    manifest = a.adapter['sourceManifest']
    manifest['snapshot_id'] = digest({k: v for k, v in manifest.items() if k != 'snapshot_id'})
    a.adapter['sourceCollectionHash'] = manifest['snapshot_id']


@pytest.fixture
def authority(tmp_path):
    patient = 'GENERATED-COLLECTION:person-A'
    subject = 'generated/person-A'
    declared = {'sourcePatientGroup': patient, 'subject': subject, 'role': candidates.DECLARED}
    declaration = {'schema': 'meningioma-header-only-prospective-development-candidates-v1',
                   'sourceFamily': 'GENERATED-COLLECTION', 'sourcePublishedTrainSplit': None,
                   'reservedChecks': [{'identity': patient, 'splitReservationsRole': None, 'baselinePatientGroupRole': None}],
                   'declaredUtc': '2026-01-01T00:00:00+00:00', 'beforeNetworkTransferOrMemberDecompression': True,
                   'officialTrainClaim': False, 'candidates': [declared]}
    declaration_ref = write(tmp_path/'declaration.json', declaration)
    sources = [{'series': name, 'member': {'name': subject+'/'+name+'.nii.gz', 'size': 101+i},
                'gzipSHA256': str(i+1)*64, 'completeVerified': True, 'gzipCRCAndTrailerVerified': True,
                'releasedGzipBytes': 101+i} for i, name in enumerate(('CT-BRAIN', 'CT-BONE', 'CE-T1W'))]
    protocol = {'sourcePatientGroup': patient, 'subject': subject, 'role': candidates.DECLARED,
                'officialTrainClaim': False, 'roleDeclarationSHA256': declaration_ref['sha256'],
                'roleDeclarationUtc': declaration['declaredUtc'],
                'sources': [{k: r[k] for k in ('series', 'member')} for r in sources]}
    protocol_ref = write(tmp_path/'protocol.json', protocol)
    receipt = {'protocolSHA256': protocol_ref['sha256'], 'role': candidates.DECLARED,
        'status': 'ALL_THREE_ORIGINAL_MEMBERS_VERIFIED_UNQUALIFIED_NO_SCALAR_REVIEW',
        'officialTrainClaim': False, 'patientTaskAdmission': False,
        'fullOriginalsVerified': 3, 'immutableInputPinsUnchanged': True, 'sources': sources}
    receipt_ref = write(tmp_path/'receipt.json', receipt)
    reservations = {'roles': {}}
    baseline = {'groups': []}
    cohort = {'schema_version': 1, 'records': [], 'identity_links': []}
    current = {k: write(tmp_path/(k+'.json'), d) for k, d in
               [('reservations', reservations), ('baseline', baseline), ('cohort', cohort)]}
    record = {'patientGroup': patient, 'subject': subject, 'role': candidates.DECLARED,
              'declaration': declaration_ref, 'acquisitionProtocol': protocol_ref,
              'acquisitionReceipt': receipt_ref, 'currentEvidence': current}
    catalog = {'schema': 'resectionlab.development-candidates.v1', 'records': [record], 'identity_links': []}
    write(tmp_path/'development-candidates.json', catalog)
    adapter = {'sourcePatient': patient, 'sourceRole': candidates.CANDIDATE,
        'sourceManifest': {'schema': 'patient-anatomy-v1', 'patient_id': patient,
            'series': [{'patient_id': patient, 'id': r['series'],
                        'original': {'name': r['series']+'.nii.gz', 'sha256': r['gzipSHA256']}}
                       for r in sources], 'layers': [{'patient_id': patient, 'id': 'model-target'}]},
        'sourceLayers': {'tumor': {'reviewStatus': 'unreviewed', 'provenance': {'kind': 'model_inference',
            'model': 'generated-model', 'patientTaskAdmission': False}}}}
    a = SimpleNamespace(root=tmp_path, adapter=adapter, catalog=catalog, record=record, patient=patient,
        declaration=declaration, protocol=protocol, receipt=receipt, reservations=reservations,
        baseline=baseline, cohort=cohort)
    seal(a)
    return a


def resolve(a):
    return surgical_source_roles.development_role(a.adapter, declarations_root=a.root/'unused',
                                                 candidate_registry_root=a.root)


def update_catalog(a):
    write(a.root/'development-candidates.json', a.catalog)


def test_exact_metadata_join_never_reads_images_or_promotes_model_annotation(authority, monkeypatch):
    a = authority
    before = copy.deepcopy(a.adapter)
    from pathlib import Path
    original_read = Path.read_bytes
    reads = []
    def checked(path):
        assert path.parent == a.root and path.suffix == '.json'
        reads.append(path.name)
        return original_read(path)
    monkeypatch.setattr(Path, 'read_bytes', checked)
    proof = resolve(a)
    assert proof['sourceRole'] == 'DEVELOPMENT_CANDIDATE'
    assert proof['patientId'] == a.patient
    q = proof['roleQualification']
    assert q['personIdentities'] == [a.patient] and len(q['originals']) == 3
    assert not q['officialTrainClaim'] and not q['annotationAuthority'] and not q['patientTaskAdmission']
    assert a.adapter == before and a.adapter['sourceLayers']['tumor']['reviewStatus'] == 'unreviewed'
    assert set(reads) == {'development-candidates.json', 'declaration.json', 'protocol.json', 'receipt.json',
                          'reservations.json', 'baseline.json', 'cohort.json'}


def test_missing_authority_and_unregistered_candidate_never_fall_back(authority):
    a = authority
    with pytest.raises(ValueError, match='authority root'):
        candidates.development_candidate_role(a.adapter)
    with pytest.raises(ValueError, match='unavailable'):
        candidates.development_candidate_role(a.adapter, registry_root=a.root/'missing')
    a.catalog['records'] = []
    update_catalog(a)
    with pytest.raises(ValueError, match='not registered'):
        resolve(a)


def test_case_metadata_cannot_select_an_authority_root(authority):
    a = authority
    a.adapter['candidate_registry_root'] = str(a.root)
    a.adapter['candidateRegistryPath'] = str(a.root/'development-candidates.json')
    with pytest.raises(ValueError, match='authority root'):
        surgical_source_roles.development_role(a.adapter, declarations_root=a.root)


@pytest.mark.parametrize('document', ['declaration', 'protocol', 'receipt'])
def test_changed_pinned_metadata_is_refused(authority, document):
    a = authority
    path = a.root/(document+'.json')
    path.write_text(path.read_text()+' ')
    with pytest.raises(ValueError, match='changed'):
        resolve(a)


@pytest.mark.parametrize('change', ['missing', 'duplicate', 'wrong-hash', 'wrong-patient', 'foreign-descendant', 'train'])
def test_native_claims_cannot_replace_exact_originals_or_group(authority, change):
    a = authority
    series = a.adapter['sourceManifest']['series']
    if change == 'missing': series.pop()
    elif change == 'duplicate': series[-1] = copy.deepcopy(series[0])
    elif change == 'wrong-hash': series[0]['original']['sha256'] = 'a'*64
    elif change == 'wrong-patient': series[0]['patient_id'] = 'reserved:person'
    elif change == 'foreign-descendant': a.adapter['sourceManifest']['layers'][0]['patient_id'] = 'other:person'
    else: a.adapter['sourceRole'] = 'TRAIN'
    seal(a)
    with pytest.raises(ValueError, match='SOURCE_ROLE'):
        resolve(a)


@pytest.mark.parametrize('where', ['reservations', 'baseline', 'alias', 'alias-chain', 'descendant', 'ancestor', 'collection'])
def test_current_reserved_identity_changes_are_rechecked_not_erased(authority, where):
    a = authority
    reserved = 'RESERVED:person-B'
    if where == 'reservations':
        a.reservations['roles'][a.patient] = 'EVAL'
    elif where == 'baseline':
        a.baseline['groups'].append({'family_id': a.patient, 'role': 'FROZEN_ASSESSMENT'})
    elif where == 'collection':
        a.reservations[a.patient.partition(':')[0]] = 'external final reserved'
    elif where.startswith('alias'):
        a.reservations['roles'][reserved] = 'MEASUREMENT_EVAL'
        names = [a.patient, reserved]
        if where == 'alias-chain':
            a.catalog['identity_links'] = [{'identities': [a.patient, 'UNKNOWN:alias'], 'confirmed': False}]
            names = ['UNKNOWN:alias', reserved]
        a.cohort['identity_links'] = [{'identities': names, 'confirmed': False}]
    else:
        first = {'record_id': 'one', 'identity': a.patient, 'outer_split': 'development', 'uses': [], 'derivative_of': None}
        second = {'record_id': 'two', 'identity': reserved, 'outer_split': 'final_evaluation', 'uses': [], 'derivative_of': None}
        (second if where == 'descendant' else first)['derivative_of'] = 'one' if where == 'descendant' else 'two'
        a.cohort['records'] = [first, second]
    for key in ('reservations', 'baseline', 'cohort'):
        write(a.root/(key+'.json'), getattr(a, key))
    update_catalog(a)
    with pytest.raises(ValueError, match='reserved|conflicting'):
        resolve(a)


def test_current_development_alias_and_descendant_inherit_one_patient_group(authority):
    a = authority
    a.cohort['records'] = [
        {'record_id': 'base', 'identity': a.patient, 'outer_split': 'development', 'derivative_of': None},
        {'record_id': 'derived', 'identity': 'DERIVED:one', 'outer_split': 'development', 'derivative_of': 'base'}]
    write(a.root/'cohort.json', a.cohort)
    assert resolve(a)['roleQualification']['personIdentities'] == ['DERIVED:one', a.patient]


def test_unrelated_legacy_authority_dispatch_is_preserved(authority, monkeypatch):
    a = authority
    other = {'sourcePatient': 'LEGACY:other', 'sourceManifest': {'series': []}}
    monkeypatch.setattr(surgical_source_roles, '_registry_role', lambda adapter: {'legacy': adapter is other})
    assert surgical_source_roles.development_role(other, declarations_root=a.root) == {'legacy': True}
    assert surgical_source_roles.development_role(other, declarations_root=a.root,
                                                  candidate_registry_root=a.root) == {'legacy': True}


@pytest.mark.parametrize('field', ['officialTrainClaim', 'patientTaskAdmission', 'immutableInputPinsUnchanged'])
def test_conflicting_resealed_acquisition_claims_do_not_grant_authority(authority, field):
    a = authority
    a.receipt[field] = field != 'immutableInputPinsUnchanged'
    a.record['acquisitionReceipt'] = write(a.root/'receipt.json', a.receipt)
    update_catalog(a)
    with pytest.raises(ValueError, match='receipt conflicts'):
        resolve(a)


@pytest.mark.parametrize('change', ['after-transfer', 'historically-reserved', 'official-train'])
def test_resealed_historical_declaration_cannot_hide_prior_role_conflict(authority, change):
    a = authority
    if change == 'after-transfer': a.declaration['beforeNetworkTransferOrMemberDecompression'] = False
    elif change == 'historically-reserved': a.declaration['reservedChecks'][0]['splitReservationsRole'] = 'EVAL'
    else: a.declaration['officialTrainClaim'] = True
    a.record['declaration'] = write(a.root/'declaration.json', a.declaration)
    a.protocol['roleDeclarationSHA256'] = a.record['declaration']['sha256']
    a.record['acquisitionProtocol'] = write(a.root/'protocol.json', a.protocol)
    a.receipt['protocolSHA256'] = a.record['acquisitionProtocol']['sha256']
    a.record['acquisitionReceipt'] = write(a.root/'receipt.json', a.receipt)
    update_catalog(a)
    with pytest.raises(ValueError, match='predecode'):
        resolve(a)

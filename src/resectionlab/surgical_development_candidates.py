"""Metadata-only joins for prospectively declared engineering candidates.

The caller configures an authority directory; a case cannot select its catalog
or evidence files. This proves a development-use lineage, not TRAIN membership,
annotation truth, anatomical qualification, or permission to execute a task.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .cohort import _Groups
from .core import semantic_digest
from .patient_anatomy import digest

CANDIDATE = 'DEVELOPMENT_CANDIDATE'
DECLARED = 'UNRESERVED_DEVELOPMENT_CANDIDATE'


def _require(value, message):
    if not value:
        raise ValueError('SOURCE_ROLE: '+message)


def _document(path, expected=None):
    # All paths originate in application-owned configuration/catalogs, never
    # adapter metadata. These documents are metadata, not image payloads.
    path = Path(path)
    try:
        _require(path.suffix == '.json' and path.stat().st_size <= 8*1024**2,
                 'candidate authority must be bounded JSON metadata')
        data = path.read_bytes()
        value = json.loads(data)
    except (OSError, ValueError) as error:
        raise ValueError('SOURCE_ROLE: candidate authority is unavailable or invalid') from error
    sha = hashlib.sha256(data).hexdigest()
    _require(type(value) is dict and (expected is None or sha == expected),
             'candidate authority changed or is invalid')
    return value, sha


def _read_reference(root, reference, *, pinned):
    _require(type(reference) is dict and type(reference.get('path')) is str,
             'candidate authority reference is missing')
    path = Path(reference['path'])
    path = path if path.is_absolute() else root/path
    expected = reference.get('sha256')
    if pinned:
        _require(type(expected) is str and len(expected) == 64
                 and all(c in '0123456789abcdef' for c in expected), 'candidate evidence hash is missing')
    return _document(path, expected if pinned else None)


def _current_person_group(patient, catalog, current):
    """Conservatively join even unconfirmed aliases and derivative records."""
    reservations, baseline, cohort = current
    _require(type(reservations.get('roles')) is dict and type(baseline.get('groups')) is list
             and cohort.get('schema_version') == 1, 'unsupported current reservation evidence')
    groups = _Groups()
    groups.find(patient)
    for link in [*catalog.get('identity_links', []), *cohort.get('identity_links', [])]:
        names = link.get('identities', [])
        _require(len(names) >= 2 and all(type(n) is str and ':' in n for n in names), 'invalid candidate identity link')
        for name in names[1:]:
            groups.join(names[0], name)
    records = cohort.get('records', [])
    by_id = {r['record_id']: r for r in records}
    _require(len(by_id) == len(records), 'duplicate current person record')
    for row in records:
        identity = row.get('identity')
        _require(type(identity) is str and ':' in identity, 'invalid current person identity')
        groups.find(identity)
        parent = row.get('derivative_of')
        if parent is not None:
            _require(parent in by_id, 'unresolved current derivative identity')
            groups.join(identity, by_id[parent]['identity'])
    members = {patient}
    for row in records:
        if groups.find(row['identity']) != groups.find(patient):
            continue
        members.add(row['identity'])
        _require(row.get('outer_split') == 'development'
                 and 'final_scoring' not in row.get('uses', []), 'candidate person/descendant is reserved or conflicting')
        visited = set()
        current_row = row
        while current_row.get('derivative_of') is not None:
            key = current_row['record_id']
            _require(key not in visited, 'cyclic candidate derivative identity')
            visited.add(key)
            current_row = by_id[current_row['derivative_of']]
    # Explicit aliases without a cohort record still participate in reservations.
    names = set(reservations['roles']) | {r.get('family_id') for r in baseline['groups']}
    names |= {n for link in [*catalog.get('identity_links', []), *cohort.get('identity_links', [])]
              for n in link['identities']}
    members |= {n for n in names if type(n) is str and groups.find(n) == groups.find(patient)}
    allowed = {'DEVELOPMENT', CANDIDATE, DECLARED, 'previously_exposed_development'}
    for name in members:
        # Collection-wide reservations also withhold a person in that collection.
        _require(name.partition(':')[0] not in reservations, 'candidate collection is reserved')
        role = reservations['roles'].get(name)
        _require(role is None or role in allowed, 'candidate person/alias has a reserved or conflicting role')
        rows = [r for r in baseline['groups'] if r.get('family_id') == name]
        _require(len(rows) <= 1 and all(r.get('role') in allowed for r in rows),
                 'candidate baseline person/alias is reserved or conflicting')
    return sorted(members)


def development_candidate_role(adapter, *, registry_root=None):
    """Return a candidate proof or None for an unrelated existing adapter.

    A matching catalog entry or explicit candidate claim owns this branch: any
    missing/conflicting evidence raises and cannot fall back to a legacy role.
    No arrays, original image bytes, annotations, or model outputs are opened.
    """
    claim = adapter.get('sourceRole')
    claimed = claim in (CANDIDATE, DECLARED)
    if registry_root is None:
        _require(not claimed, 'explicit candidate authority root is required')
        return None
    root = Path(registry_root).resolve()
    catalog, catalog_sha = _document(root/'development-candidates.json')
    _require(catalog.get('schema') == 'resectionlab.development-candidates.v1', 'unsupported candidate catalog')
    patient = adapter.get('sourcePatient')
    rows = [r for r in catalog.get('records', []) if r.get('patientGroup') == patient]
    _require(len(rows) <= 1, 'ambiguous candidate registration')
    if not rows:
        _require(not claimed, 'candidate patient is not registered')
        return None
    _require(claim in (None, CANDIDATE, DECLARED), 'candidate cannot claim another source role')
    record = rows[0]
    _require(record.get('role') == DECLARED, 'registered candidate role conflicts')
    declaration, declaration_sha = _read_reference(root, record.get('declaration'), pinned=True)
    protocol, protocol_sha = _read_reference(root, record.get('acquisitionProtocol'), pinned=True)
    receipt, receipt_sha = _read_reference(root, record.get('acquisitionReceipt'), pinned=True)
    candidates = [r for r in declaration.get('candidates', []) if r.get('sourcePatientGroup') == patient]
    historical = [r for r in declaration.get('reservedChecks', []) if r.get('identity') == patient]
    _require(declaration.get('schema') == 'meningioma-header-only-prospective-development-candidates-v1'
             and declaration.get('sourceFamily') == patient.partition(':')[0]
             and declaration.get('sourcePublishedTrainSplit') is None
             and declaration.get('beforeNetworkTransferOrMemberDecompression') is True
             and declaration.get('officialTrainClaim') is False and len(candidates) == 1
             and candidates[0].get('role') == DECLARED
             and candidates[0].get('subject') == record.get('subject') and len(historical) == 1
             and historical[0].get('splitReservationsRole') is None
             and historical[0].get('baselinePatientGroupRole') is None, 'predecode candidate declaration conflicts')
    _require(protocol.get('sourcePatientGroup') == patient and protocol.get('subject') == record['subject']
             and protocol.get('role') == DECLARED and protocol.get('officialTrainClaim') is False
             and protocol.get('roleDeclarationSHA256') == declaration_sha
             and protocol.get('roleDeclarationUtc') == declaration.get('declaredUtc'), 'acquisition declaration join differs')
    _require(receipt.get('protocolSHA256') == protocol_sha
             and receipt.get('status') == 'ALL_THREE_ORIGINAL_MEMBERS_VERIFIED_UNQUALIFIED_NO_SCALAR_REVIEW'
             and receipt.get('role') == DECLARED and receipt.get('officialTrainClaim') is False
             and receipt.get('patientTaskAdmission') is False and receipt.get('fullOriginalsVerified') == 3
             and receipt.get('immutableInputPinsUnchanged') is True, 'released acquisition receipt conflicts')
    current_refs = record.get('currentEvidence', {})
    current = [_read_reference(root, current_refs.get(k), pinned=False)
               for k in ('reservations', 'baseline', 'cohort')]
    members = _current_person_group(patient, catalog, [d for d, _ in current])
    manifest = adapter.get('sourceManifest', {})
    _require(manifest.get('schema') == 'patient-anatomy-v1' and manifest.get('patient_id') == patient
             and manifest.get('snapshot_id') == adapter.get('sourceCollectionHash')
             == digest({k: v for k, v in manifest.items() if k != 'snapshot_id'}), 'native candidate source identity differs')
    _require(all(r.get('patient_id') == patient for r in manifest.get('layers', [])),
             'candidate descendants must retain their source patient group')
    declared_sources = protocol.get('sources', [])
    sources = receipt.get('sources', [])
    series = manifest.get('series', [])
    _require(len(sources) == len(declared_sources) == len(series) == 3,
             'all three original candidate series are required')
    originals = []
    for row in series:
        original = row.get('original', {})
        matches = [r for r in sources if r.get('gzipSHA256') == original.get('sha256')]
        _require(row.get('patient_id') == patient and len(matches) == 1, 'candidate original does not uniquely join acquisition')
        source = matches[0]
        declared_sources_matching = [r for r in declared_sources if r.get('series') == source.get('series')]
        _require(len(declared_sources_matching) == 1, 'candidate acquired series is ambiguous')
        declared = declared_sources_matching[0]
        member = source.get('member', {})
        _require(member == declared.get('member') and member.get('name') == record['subject']+'/'+original.get('name', '')
                 and source.get('completeVerified') is True and source.get('gzipCRCAndTrailerVerified') is True
                 and source.get('releasedGzipBytes') == member.get('size')
                 and type(member.get('size')) is int and member['size'] > 0,
                 'candidate original member or integrity differs')
        originals.append({'series': source['series'], 'name': original['name'],
                          'sha256': original['sha256'], 'sizeBytes': member['size']})
    _require(len({r['sha256'] for r in originals}) == 3, 'candidate original series is duplicated')
    proof = {'schema': 'resectionlab.development-candidate-role.v1', 'patientGroup': patient,
        'personIdentities': members, 'catalogSha256': catalog_sha, 'declarationSha256': declaration_sha,
        'acquisitionProtocolSha256': protocol_sha, 'acquisitionReceiptSha256': receipt_sha,
        'currentEvidenceSha256': dict(zip(('reservations', 'baseline', 'cohort'), (sha for _, sha in current))),
        'sourceCollectionHash': adapter['sourceCollectionHash'], 'originals': sorted(originals, key=lambda r: r['series']),
        'officialTrainClaim': False, 'annotationAuthority': False, 'patientTaskAdmission': False,
        'metadataJoinsOnly': True, 'crossDatasetPersonIdentity': 'UNKNOWN; quarantine if later matched to a reserved person',
        'scope': 'Prospective engineering-development lineage only; all descendants remain one patient. Image bytes, anatomy, targets, material, critical coverage and native task requirements need separate qualification.'}
    return {'sourceRole': CANDIDATE, 'patientId': patient, 'officialTrainClaim': False,
            'annotationAuthority': False, 'patientTaskAdmission': False, 'declarationSha256': semantic_digest(proof),
            'roleQualification': proof}

"""Independent TRAIN050 metadata/receipt audit; opens no patient payload."""
import argparse
import hashlib
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
USED = {'ReMIND-002', 'ReMIND-008', 'ReMIND-010', 'ReMIND-015', 'ReMIND-018', 'ReMIND-020', 'ReMIND-025', 'ReMIND-045'}
EXPECTED = {'structural_t1ce': ('080373f0-7522-4002-a511-16f30793b709', '3D_AX_T1_postcontrast', 'MR', 192), 'whole_tumor': ('468104cb-e59b-4ded-9355-3125bad79f90', 'tumor seg - MR ref: 3D_AX_T1_postcontrast', 'SEG', 1), 'cerebrum': ('74529c89-23d3-41a4-b7ec-f6beb90761ec', 'cerebrum seg - MR ref: 3D_AX_T1_postcontrast', 'SEG', 1)}

def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024**2), b''):
            h.update(block)
    return h.hexdigest()

def array_rows(path, key):
    decoder = json.JSONDecoder()
    with path.open() as f:
        buffer = f.read(65536)
        match = re.search('"'+key+r'"\s*:\s*\[', buffer)
        assert match
        buffer = buffer[match.end():]
        while True:
            buffer = buffer.lstrip()
            if buffer.startswith(','):
                buffer = buffer[1:].lstrip()
            if buffer.startswith(']'):
                return
            try:
                row, end = decoder.raw_decode(buffer)
            except json.JSONDecodeError:
                more = f.read(65536)
                assert more and len(buffer) < 1024**2
                buffer += more
                continue
            assert isinstance(row, dict)
            yield row
            buffer = buffer[end:]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--case', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    evidence_path = ROOT/'artifacts/hybrid-transfer-readiness-v1/source-evidence.json'
    evidence = json.loads(evidence_path.read_text())['next_metadata_candidate']
    candidate_path = ROOT/'build/remind-train-public-expansion-candidate-v1/candidate.json'
    assert sha(candidate_path) == 'd436dacfc1ed54a2061cd21bb9c309d0403fdf7b98fc637c3a66c00e546084e6'
    candidate = json.loads(candidate_path.read_text())
    for b in candidate['source_bindings']:
        assert sha(ROOT/b['path']) == b['sha256'], b['path']
    cohort = json.loads((ROOT/'manifests/experiments/remind-component-cohort-v1.json').read_text())
    member, = [m for m in cohort['members'] if m['subject'] == 'ReMIND-050']
    assert member['role'] == 'TRAIN' and member['patient_group'] == 'ReMIND:050'
    assert member['diagnosis_eligibility_stratum'] == 'strict_three_label_glioma'
    remaining = sorted(r['patient_id'] for r in candidate['candidate_summary'] if r['patient_id'] not in USED)
    assert len(remaining) == 17 and remaining[0] == 'ReMIND-050'
    assert evidence['subject'] == remaining[0] and evidence['role'] == 'TRAIN'
    index = json.loads((ROOT/evidence['catalog']).read_text())
    selected = {}
    for kind, (sid, desc, modality, count) in EXPECTED.items():
        s, = [s for s in index['series'] if s['crdc_series_uuid'] == sid]
        assert s['PatientID'] == 'ReMIND-050' and s['StudyDescription'] == 'Preop'
        assert (s['SeriesDescription'], s['Modality'], s['instanceCount']) == (desc, modality, count)
        assert s['license_short_name'] == 'CC BY 4.0' and s['source_DOI'] == '10.7937/3rag-d070'
        assert s['StudyInstanceUID'] == evidence['StudyInstanceUID']
        selected[sid] = (kind, s)
    bound = {}
    for family in ('remind-train-originals-v1', 'remind-train-seg-v1'):
        base = ROOT/'data/acquisition'/family
        for o in array_rows(base/'source/train-exact-object-manifest.json', 'objects'):
            if o['series_uuid'] not in selected:
                continue
            _, s = selected[o['series_uuid']]
            assert o['patient_id'] == 'ReMIND-050' and o['role'] == 'TRAIN'
            assert o['series_instance_uid'] == s['SeriesInstanceUID'] and o['series_description'] == s['SeriesDescription']
            assert o['modality'] == s['Modality'] and o['study_description'] == 'Preop'
            assert o['url'] == 'https://idc-open-data.s3.amazonaws.com/' + o['key']
            assert o['etag'] == o['etag_md5'] and re.fullmatch('[a-f0-9]{32}', o['etag_md5'])
            key = (family, o['patient_id']+'/'+o['key'])
            assert key not in bound
            bound[key] = o
        for r in array_rows(base/'completion.json', 'files'):
            key = (family, r['path'])
            if key not in bound:
                continue
            o = bound[key]
            assert r['status'] in ('byte_verified', 'existing_byte_verified')
            assert r['patient_id'] == 'ReMIND-050' and r['bytes'] == o['bytes']
            assert r['source_checksum_type'] == 'MD5' and r['source_checksum'] == o['etag_md5']
            assert re.fullmatch('[a-f0-9]{64}', r['sha256']) and 'receipt_sha256' not in o
            o['receipt_sha256'] = r['sha256']
            o['absolute_path'] = str(base/'verified'/r['path'])
    assert len(bound) == 194 and sum(o['bytes'] for o in bound.values()) == 27142222
    assert all('receipt_sha256' in o for o in bound.values())
    for sid, (_, s) in selected.items():
        assert sum(o['series_uuid'] == sid for o in bound.values()) == s['instanceCount']
    expected_objects = {o['absolute_path']: {'bytes': o['bytes'], 'sha256': o['receipt_sha256'], 'expected_md5': o['etag_md5'], 'series_uuid': o['series_uuid']} for o in bound.values()}
    case_binding = None
    if args.case:
        case = json.loads(args.case.read_text())
        assert (case['patient_id'], case['patient_group'], case['role']) == ('ReMIND-050', 'ReMIND:050', 'TRAIN')
        assert len(case['series']) == 3 and {s['kind'] for s in case['series']} == set(EXPECTED)
        for s in case['series']:
            sid, desc, modality, count = EXPECTED[s['kind']]
            original = selected[sid][1]
            assert s['series_uuid'] == sid and s['source_description'] == desc
            assert s['Modality'] == modality and len(s['objects']) == count
            assert all(s[k] == original[k] for k in ('SeriesInstanceUID', 'StudyInstanceUID'))
        actual_objects = {o['path']: {**{k:o[k] for k in ('bytes', 'sha256', 'expected_md5')}, 'series_uuid':s['series_uuid']} for s in case['series'] for o in s['objects']}
        assert len(actual_objects) == 194 and actual_objects == expected_objects
        parent = case['parent_bindings']['source_binding']
        parent_path = Path(parent['path'])
        if not parent_path.is_absolute(): parent_path = ROOT/parent_path
        assert sha(parent_path) == parent['sha256']
        flat = json.loads(parent_path.read_text())
        assert (flat['patient_id'], flat['patient_group'], flat['role']) == ('ReMIND-050','ReMIND:050','TRAIN')
        flat_objects = {str(ROOT/o['path']): {k:o[k] for k in ('bytes','sha256','expected_md5','series_uuid')} for o in flat['objects']}
        assert len(flat_objects) == 194 and flat_objects == expected_objects
        case_binding = {'path': str(args.case), 'sha256': sha(args.case), 'parent': parent}
    normalized = json.dumps(expected_objects, sort_keys=True, separators=(',', ':')).encode()
    result = {'status':'PASS_EXACT_TRAIN050_PUBLIC_METADATA_ONLY', 'readiness_source_evidence_sha256':sha(evidence_path), 'candidate_sha256':sha(candidate_path), 'original_cohort_role':member, 'selection': {'rule':'lowest ID among complete source triples after eight root-declared used TRAIN people', 'excluded_used_TRAIN':sorted(USED), 'remaining_17_TRAIN':remaining, 'selected':'ReMIND-050', 'outcome_based_selection':False}, 'series':[{'kind':kind, 'series_uuid':sid, 'description':s['SeriesDescription'], 'series_instance_uid':s['SeriesInstanceUID'], 'study_instance_uid':s['StudyInstanceUID'], 'objects':s['instanceCount']} for sid,(kind,s) in selected.items()], 'objects':194, 'bytes':27142222, 'exact_object_binding_sha256':hashlib.sha256(normalized).hexdigest(), 'all_source_MD5_and_local_SHA256_receipts_match':True, 'case_binding':case_binding, 'payload_reads':0, 'header_or_array_reads':0, 'private_annotation_reads':0, 'planning_or_model_calls':0, 'roles_changed':False, 'qualification':'pending header/source/frame and domain QC; no anatomy or training admission', 'limitations':['Preop study label and same study do not prove annotation availability or registration.', 'Whole-tumor annotation is not a prescribed surgical target; automatic cerebrum is not physical occupancy.', 'Missing vessels and function remain unknown; no missing coverage inferred from unparsed headers.', 'Patient remains TRAIN; this is not untouched outer evaluation. Failed case remains selected; no replacement by outcome.']}
    with args.output.open('x') as f:
        json.dump(result, f, indent=2)
        f.write('\n')
    print(json.dumps({'status':result['status'], 'output':str(args.output), 'sha256':sha(args.output)}))

if __name__ == '__main__':
    main()

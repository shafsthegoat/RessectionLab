"""Independent fixed-release metadata audit; never opens patient payloads."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sqlite3

ROOT = Path(__file__).resolve().parents[1]
BASE = 'outputs/patient-corpus-twelve-execution-v1'
CATALOG_SHA = '5f8b833079e0b4673d2b8d01a36fab474574164755ac731a0972fa8071bcd088'
SEMANTIC = 'sha256:6354e2e955aa4f4a5e0b892d5e3e8b069d9f8cb046196787b08273e7a94b4083'
FROZEN = {
    BASE+'/catalog.json': CATALOG_SHA,
    'outputs/patient-corpus-ixi265-v1/catalog.json': '2a7ba1c26f101261e3a5fecf84ae998b637a70c5a7f66d43d224153c738ee673',
    'manifests/patient-corpus-releases/existing-ten-plus-synthrad-bc-v1.json': 'd1eb5d7642ac01c46ad8d7a2464627408358d438ca0c302bcaa8c71e759bbbd1',
    'artifacts/patient-corpus-twelve-execution-v1/outcome.json': '045ec9bbe30674515e4b8084406acd1883fa3880fb9b91eb242b0938c2dfa792',
    'artifacts/patient-corpus-twelve-recovery-v1/outcome.json': '7fb86e0f9a7e6d77ae03f67f050cd409fbcebed5c18eb7810a9248337705c365',
    'artifacts/patient-corpus-twelve-execution-preparation-v1/receipt.json': '591109a4bb4517e7b191843e0df17c12144eec023a20c014f18dca1177d22d86',
}


def digest(relative):
    path = ROOT/relative
    assert path.resolve() == path and path.stat().st_size < 4*1024**2, relative
    assert path.suffix in {'.json', '.sqlite', '.cjs', '.py', '.gz', '.lock'}, relative
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(relative):
    assert Path(relative).suffix == '.json'
    digest(relative)
    return json.loads((ROOT/relative).read_text())


def semantic(value):
    raw = json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)
    return 'sha256:'+hashlib.sha256(raw.encode()).hexdigest()


def audit():
    for path, pin in FROZEN.items():
        assert digest(path) == pin, path
    catalog = read(BASE+'/catalog.json')
    assert catalog['catalog_hash'] == SEMANTIC == semantic({k:v for k,v in catalog.items() if k != 'catalog_hash'})
    prospective = read('manifests/patient-corpus-releases/existing-ten-plus-synthrad-bc-v1.json')
    candidates = {r['patient_group']:r for r in prospective['base_candidates']+prospective['additional_candidates']}
    entries = {r['patient_group']:r for r in catalog['entries']}
    assert len(entries) == len(catalog['entries']) == 12 and set(entries) == set(candidates)
    old = {r['patient_group']:r for r in read('outputs/patient-corpus-ixi265-v1/catalog.json')['entries']}
    retained = ('case_hash','bundle_sha256','anatomy_present','anatomy_present_on_primary','coverage',
        'imaging','layers','missing_anatomy','originals','registrations','rights','role','support_conflicts','tumor_location')
    verified_receipts = []
    for patient, row in entries.items():
        candidate = candidates[patient]
        assert row['bundle_sha256'] == candidate['bundle_sha256'] and row['role'] == candidate['role']
        assert row['status'] == 'qualified_with_limitations'
        assert all(row[k] is False for k in ('clinical_use_permitted','training_admitted','virtual_descendant','training_admission_established','new_patient'))
        assert all(row['tumor_location'][k] == 'UNKNOWN' for k in ('anatomical_region','hemisphere','lobe'))
        assert all(r['anatomical_registration_accepted'] is False and r['independently_validated_registration'] is False for r in row['registrations'])
        receipt_path = BASE+'/'+row['receipt_path']
        receipt = read(receipt_path)
        assert receipt == {k:v for k,v in row.items() if k != 'receipt_path'}
        assert receipt['receipt_hash'] == semantic({k:v for k,v in receipt.items() if k != 'receipt_hash'})
        assert receipt['input_binding_hash'] == semantic(receipt['input_binding'])
        assert receipt['input_binding']['candidate'] == candidate
        if patient in old:
            for key in retained:
                assert row[key] == old[patient][key], (patient, key)
        verified_receipts.append({'patient_group':patient,'path':receipt_path,'sha256':digest(receipt_path)})
    combinations = {name:sum(set(parts) <= set(r['anatomy_present']) for r in entries.values()) for name,parts in {
        'brain_plus_tumor':['brain','tumor'], 'brain_plus_skull':['brain','skull'],
        'brain_plus_vessels':['brain','vessels'], 'brain_tumor_skull_vessels':['brain','tumor','skull','vessels']}.items()}
    counts = catalog['counts']
    assert combinations == {'brain_plus_tumor':7,'brain_plus_skull':3,'brain_plus_vessels':1,'brain_tumor_skull_vessels':0}
    assert counts['anatomical_presence_combinations'] == counts['same_person_presence_across_saved_grids'] == combinations
    assert all(v == 0 for v in counts['accepted_fused_anatomy_combinations'].values())
    assert counts['verified_embedded_original_files'] == sum(r['originals']['embedded_count'] for r in entries.values()) == 66
    assert counts['people_with_portable_source_originals'] == sum(r['originals']['portable_originals_complete'] for r in entries.values()) == 6
    assert all(counts[k] == 0 for k in ('pending','rejected','newly_acquired_people','anatomical_truth_accepted_people','virtual_descendants_counted_as_people'))
    preserved = read('artifacts/patient-corpus-twelve-execution-preparation-v1/receipt.json')['preserved_files']
    assert len(preserved) == 89
    for path,pin in preserved.items(): assert digest(path) == pin, path
    failure = read('artifacts/patient-corpus-twelve-execution-v1/outcome.json')
    assert failure['qualification_completed'] is failure['complete_catalog_created'] is False
    assert failure['supervision']['failure_code'] == 'RSS_LIMIT' and failure['supervision']['worker_exit_code'] == -9
    for row in failure['completed_receipts']: assert digest(row['receipt_path']) == row['receipt_file_sha256']
    stages = [read(f'outputs/patient-corpus-twelve-recovery-v1/stage-{i}.json') for i in range(3)]
    assert [r['result']['new_qualification_receipts'] for r in stages] == [1,1,0]
    assert [r['result']['reused_qualification_receipts'] for r in stages] == [10,11,12]
    assert stages[1]['result']['catalog_hash'] == stages[2]['result']['catalog_hash'] == SEMANTIC
    # Source-declared UCSF display overlays remain distinct from independent acceptance.
    assert sum(r['overlayPermitted'] for r in entries['UCSF-PDGM:004']['registrations']) == 3
    assert not any(r['overlayPermitted'] for r in entries['IXI:265']['registrations'])
    database = ROOT/BASE/'catalog.sqlite'
    assert digest(str(database.relative_to(ROOT))) == '8117626ec5e8a534f851da0ef4937dc8e1d50e0f92009210ed4bece419256338'
    with sqlite3.connect(database.as_uri()+'?mode=ro', uri=True) as connection:
        assert connection.execute('PRAGMA integrity_check').fetchone() == ('ok',)
    return {'schema':'independent-twelve-catalog-metadata-qa-v1','status':'pass','auditor_sha256':digest(str(Path(__file__).relative_to(ROOT))),
        'pinned_metadata':FROZEN,'source_identities':sorted(entries),'counts':counts,'verified_receipts':verified_receipts,
        'previous_source_metadata_unchanged_people':len(old),'preserved_old_files':len(preserved),'original_completed_receipts_preserved':10,
        'original_resource_failure_retained':True,'recovery_new_receipts':[1,1,0],'recovery_reused_receipts':[10,11,12],
        'sqlite_integrity':'ok','source_arrays_or_bundles_opened':False,'models_or_solvers_executed':False,
        'scope':'Metadata/source-lineage consistency; no new anatomical or numerical-array validation',
        'audit_harness_corrections':['Initial metadata suffix allowlist omitted the preserved empty .run.lock file; allowed .lock before rerun. No product change or source payload access.'],
        'limitations':['Namespaced source identities do not establish cross-cohort biological independence.',
            'Presence across saved grids does not establish accepted fusion; UCSF source-declared overlays are preserved.',
            'Cold qualification failed its RSS cap; recovery completed the two pending cases using prior valid caches.']}


if __name__ == '__main__':
    result = audit()
    output = ROOT/'artifacts/patient-corpus-twelve-independent-v1/metadata-qa.json'
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('x') as stream: json.dump(result,stream,indent=2);stream.write('\n')
    print(json.dumps({'status':result['status'],'sha256':digest(str(output.relative_to(ROOT)))},indent=2))

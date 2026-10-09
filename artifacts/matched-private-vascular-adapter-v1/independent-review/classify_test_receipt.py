"""Interpret a guard false-positive without changing the original test receipt."""
from pathlib import Path
import hashlib,json,sys
OUT=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
raw=json.loads((OUT/'test-review.json').read_text())
expected=str(Path(sys.base_prefix)/'lib'/f'python{sys.version_info.major}{sys.version_info.minor}.zip')
assert raw['pytest_exit_code']==0 and raw['source_files_unchanged_during_tests'] is True
assert raw['source_hashes_before']==raw['source_hashes_after'] and raw['torch_imported'] is False
assert raw['blocked_attempts']==[expected]*3
assert '52 passed in 14.71s' in (OUT/'combined-tests.txt').read_text()
result={'status':'PASS_52_generated_controls_with_classified_guard_false_positive','original_receipt_sha256':sha(OUT/'test-review.json'),'original_receipt_status':raw['status'],'pytest_exit_code':0,'passed_controls':52,'pytest_seconds':14.71,'source_files_unchanged_during_tests':True,'original_guard_was_stricter_than_required':True,'blocked_python_stdlib_archive_lookups':raw['blocked_attempts'],'actual_patient_model_archive_or_network_attempts':0,'torch_imported':False,'explanation':'The guard refused every .zip open, including exactly three standard-library Python archive import lookups. These are runtime library lookups, not patient/model/archive payload requests. They remained blocked; all 52 tests passed. Original receipt and log remain unchanged; no test rerun or source modification was needed.'}
(OUT/'test-classification.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
print(json.dumps({'status':result['status'],'classification_sha256':sha(OUT/'test-classification.json')}))

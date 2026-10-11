"""Static AST equivalence check; no production module import or payload read."""
from pathlib import Path
import ast
import hashlib
import json

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
STAGE = ROOT/'build/hybrid-train050-preparation-v1/stage'

def binding(path):
    raw = path.read_bytes()
    return {'path': str(path.relative_to(ROOT)), 'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}

original = ROOT/'src/resectionlab/remind_planning_qc.py'
staged = STAGE/'src/resectionlab/remind_planning_qc.py'
before = ast.parse(original.read_text())
after = ast.parse(staged.read_text())
assert len(before.body) == len(after.body)
changes = []
for old, new in zip(before.body, after.body):
    if ast.dump(old) == ast.dump(new):
        continue
    assert isinstance(old, ast.Assign) and isinstance(new, ast.Assign)
    assert [x.id for x in old.targets] == [x.id for x in new.targets] == ['PUBLIC_SUBJECTS_BY_ROLE']
    old_value, new_value = ast.literal_eval(old.value), ast.literal_eval(new.value)
    expected = dict(old_value)
    expected['TRAIN'] = (*old_value['TRAIN'], 'ReMIND-050')
    assert new_value == expected
    changes.append('PUBLIC_SUBJECTS_BY_ROLE.TRAIN appends only ReMIND-050')
assert len(changes) == 1
copies = []
for relative in ('src/resectionlab/__init__.py', 'scripts/prepare_remind_planning.py'):
    assert (ROOT/relative).read_bytes() == (STAGE/relative).read_bytes()
    copies.append(binding(STAGE/relative))
result = {'status': 'PASS_HEADER_READER_MINIMAL_SCOPE_EXTENSION_ONLY', 'original': binding(original), 'staged': binding(staged), 'unchanged_copies': copies, 'changes': changes, 'all_other_AST_nodes_equal': True, 'public_three_series_validation_unchanged': True, 'immutable_cohort_and_exact_source_object_validation_unchanged': True, 'private_series_rejection_unchanged': True, 'header_decoder_geometry_and_timing_code_unchanged': True, 'crop_mr_domains_gate_still_excludes_050': True, 'factory_and_admission_unchanged': True, 'no_actual_header_or_pixel_execution': True, 'actual_owned_header_launcher_review': 'pending separate frozen launcher source'}
output = HERE/'reader-source-review.json'
with output.open('x') as f:
    json.dump(result, f, indent=2)
    f.write('\n')
print(json.dumps({'status': result['status'], 'source_sha256': result['staged']['sha256'], 'review': binding(output)}))

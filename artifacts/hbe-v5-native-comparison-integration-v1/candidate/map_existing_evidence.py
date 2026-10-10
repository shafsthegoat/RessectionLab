"""Transparent mapping of existing exact audits; no replay or response statistics."""
from pathlib import Path
import hashlib
import json
import sys
ROOT=Path(__file__).resolve().parents[2];HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
from scripts import mechanics_hbe_v5_remaining_one_shot as remaining
from scripts import mechanics_hbe_v5_n12_admission as admission
from scripts import mechanics_hbe_v5_source_bindings as sources

def raw(path):
    path=ROOT/path
    assert path.suffix in ('.json','.md') and path.stat().st_size < 2*1024**2 and not path.is_symlink()
    return path.read_bytes()
def sha(data):return hashlib.sha256(data).hexdigest()
reports=[
 'build/hbe-v5-n8-result-independent/REPORT.md',
 'artifacts/hbe-v5-n12-saved-replay-v1/INDEPENDENT_REVIEW.md',
 'artifacts/hbe-v5-n16-numerical-v1/INDEPENDENT_REVIEW.md',
 'artifacts/hbe-v5-n24-numerical-v1/INDEPENDENT_REVIEW.md',
 'artifacts/hbe-v5-n32-numerical-v1/INDEPENDENT_REVIEW.md',
 'artifacts/hbe-v5-n36-numerical-v1/INDEPENDENT_REVIEW.md',
 'artifacts/hbe-v5-n36-s120-numerical-v1/INDEPENDENT_REVIEW.md',
 'artifacts/hbe-v5-tension-n8-s60-numerical-v1/INDEPENDENT_REVIEW.md',
 'artifacts/hbe-v5-tension-n12-nocache-v2-result-v1/RESULT.md']
markers=['Every non-provenance readout field reproduced the receipt exactly.',
 'The entire `replay.json` is **byte-for-byte identical** to the original 539,665-byte `readout.json` (not merely equivalent parsed JSON).',
 'A fresh read-only `mechanics_hbe_v5_stream.read_bound_run` over its six bound raw inputs produced **exactly** the saved full `readout.json` object',
 'A fresh read-only `mechanics_hbe_v5_stream.read_bound_run` over the six bound inputs reproduced the **entire** saved `readout.json` object',
 'A fresh read-only `mechanics_hbe_v5_stream.read_bound_run` over those six bound raw inputs reproduced the **entire** saved `readout.json` object.',
 None,None,None,
 'Its decoded result exactly matched the retained `readout.json`']
source_map=json.loads(raw(sources.BINDING_PATH))
rows={}
for index,report in enumerate(reports):
    receipt_path=remaining.receipt_path(index);data=raw(receipt_path);receipt=json.loads(data)
    run_id=remaining.ORDER[index];assert receipt['run_id']==run_id
    text=raw(report).decode()
    passage=markers[index]
    if passage is None:
        lines=[line for line in text.splitlines() if ('entire' in line.lower() or 'byte-identical' in line.lower())
               and ('replay' in line.lower() or 'saved' in line.lower()) and 20<len(line)<2048]
        assert lines,report
        passage=lines[0]
    assert passage in text,report
    readout_sha=sha(data) if index==0 else '960c7a4fdfb37b0823b07f88c19fe280b72583169fc73825a90b0dfe8c86772f' if index==1 else receipt['output_bindings']['readout.json']['sha256']
    src=source_map['source_decks'][source_map['run_source_keys'][run_id]]
    exception=None if index!=1 else {'original_status':'failed_or_incomplete',
         'supplement_receipt_sha256':admission.SUPPLEMENT_SHA,'historical_guard_gap':admission.GAP}
    rows[run_id]={'run_id':run_id,'native_receipt_sha256':sha(data),'release_sha256':receipt['release_sha256'],
        'source_deck_sha256':receipt['old_source_deck_sha256'],'adapted_deck_sha256':receipt['adapted_deck_sha256'],
        'native_mesh_sha256':receipt['native_mesh_sha256'],'readout_sha256':readout_sha,
        'evidence_kind':'archived_prose_exact_replay','comparison_scope':'all_non_provenance_fields' if index==0 else 'entire_readout_json',
        'supporting_passage':passage,'review_binding':{'path':report,'sha256':sha(raw(report))},'exception_policy':exception}
    assert rows[run_id]['source_deck_sha256']==src['sha256']
value={'schema':'hbe-v5-existing-independent-evidence-mapping-v1','status':'existing_evidence_mapped_no_new_review_or_release',
       'rows':rows,'missing_native_rows':list(remaining.ORDER[9:]),
       'original_N8_timing_gap':'Historical N8 readout and preparation wall time were not recorded.',
       'native_or_replay_calls':0,'measured_responses_accessed':False,
       'note':'Exact existing prose reviews are sufficient; this map creates no review, qualification, or permission. No separate readout files or raw primitive files were opened; numerical arrays were not interpreted.'}
(HERE/'existing-evidence-mapping.json').write_text(json.dumps(value,indent=2,sort_keys=True)+'\n')
print(json.dumps({'mapped':len(rows),'missing_native_rows':value['missing_native_rows'],'sha256':sha((HERE/'existing-evidence-mapping.json').read_bytes())}))

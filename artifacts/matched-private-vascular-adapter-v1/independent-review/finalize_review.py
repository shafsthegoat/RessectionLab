"""Bind independent generated-source review; no runtime payload reads."""
from pathlib import Path
import ast,copy,hashlib,json
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent;C=ROOT/'build/matched-private-vascular-adapter-v1'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
expected={'matched_private_vascular.py':'148c30d147c4f88f0e725575debb8b50b774e224e80d4b5bf9b18d90465945db','private_vascular_evaluation.py':'a4ca5a710fa6b8f02251e7d30d5e3fc6bf3602a8f6b2ca9ce7af6e809281856d','test_adapter.py':'82637681b30a9fa5edcc40e1902377f41ceb343a7c6b4ca367200840718eb258','source.patch':'4486189366259ff9c3f9de28230433e92ac2b94bf6d7b78bdec1402f2d5a5587'}
for name,value in expected.items():assert sha(C/name)==value
receipt=json.loads((OUT/'test-review.json').read_text());classification=json.loads((OUT/'test-classification.json').read_text())
assert classification['status']=='PASS_52_generated_controls_with_classified_guard_false_positive'
for path,value in receipt['source_hashes_after'].items():assert sha(ROOT/path)==value
baseline=ast.parse((ROOT/'src/resectionlab/private_vascular_evaluation.py').read_text());candidate=ast.parse((C/'private_vascular_evaluation.py').read_text())
old={n.name:n for n in baseline.body if isinstance(n,(ast.FunctionDef,ast.ClassDef))};new={n.name:n for n in candidate.body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
assert set(new)-set(old)=={'_evaluate_preflighted_private_vessels'} and set(old)<=set(new)
for name in old:
 if name!='evaluate_private_vessels':assert ast.dump(old[name])==ast.dump(new[name]),name

def tail(node):
 index=next(i for i,n in enumerate(node.body) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='directory' for t in n.targets))
 body=copy.deepcopy(node.body[index:])
 for n in body:
  if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='expected_binding' for t in n.targets):n.value=ast.Name(id='REVIEWED_CAPTURED_EXPECTED_BINDING',ctx=ast.Load())
 return ast.dump(ast.Module(body=body,type_ignores=[]))
assert tail(old['evaluate_private_vessels'])==tail(new['_evaluate_preflighted_private_vessels'])
review={'schema':'matched-private-vascular-independent-review-v1','decision':'GO_generated_source_integration_only','candidate_sha256':expected,'baseline_sha256':receipt['source_hashes_after'],'checks':{'generated_controls_passed':52,'author_adapter_controls':19,'existing_standalone_controls':30,'independent_callback_mutation_controls':3,'pytest_seconds':14.71,'all_relevant_sources_unchanged_during_tests':True,'existing_geometry_and_other_evaluator_definitions_AST_unchanged':True,'extracted_scoring_body_AST_unchanged_except_captured_expected_binding':True,'all_complete_seals_and_nominal_preflights_before_first_private_loader':True,'common_reference_identity_and_per_method_hashes_captured_before_callbacks':True,'mapping_replacement_cannot_retarget_loader_or_reference':True,'in_place_binding_mutation_with_recomputed_fingerprint_refused_before_next_loader':True,'late_mutation_of_already_scored_binding_invalidates_batch':True,'failed_abstained_slots_preserved':True,'non_null_prescored_outcomes_rejected':True,'strict_generated_session_counters':True,'GENERATED_T1_not_relabelled':True,'private_errors_do_not_export_error_text':True,'actual_model_patient_archive_or_network_attempts':0,'torch_imported':False},'evidence':{name:sha(OUT/name) for name in ['mapping-mutation-finding.json','reproduce_mapping_mutation.py','test_independent_adapter.py','run_review_tests.py','test-review.json','combined-tests.txt','test-classification.json','classify_test_receipt.py']},'resolved_finding':{'original_adapter_sha256':'c4b76aad5b971195f05e6ad7c0783ab18494f7ebfb16dfb144d32fd28e73c52d','issue':'After common-reference preflight, the first private callback could replace the next method binding and loader in caller-owned mappings; the batch accepted successful methods scored in different reference worlds.','generated_reproduction':'SEARCH positive cells2, IL0, both accepted, batch completed.','repair':'Copy mappings, retain immutable expected per-method and common-world fingerprints before private access, verify each selected binding before loader/scoring and all again at completion; helper receives captured expected hash.','status':'closed_by_source_review_and_regression_controls'},'guard_classification':'All52 tests passed. Original wrapper status FAIL is retained because broad .zip guard blocked exactly3 Python stdlib archive import lookups. Supplemental classification verifies those runtime-library lookups; no patient/model/archive/network attempt occurred. No repeated execution or hidden receipt rewrite.','limits':['Generated interface correctness only; no learned-method efficacy or transfer claim.','Existing GENERATED_T1 comparison records remain incompatible with exact T1 vascular identity; no relabelling permitted.','Only generated person/role/lineage accepted. IXI or other patient data admission remains unimplemented.','Planning/reference array and action/microstep bounds remain unchanged; no streaming, frame registration or geometry algorithm changes.','Wall limits are cooperative per-preflight/per-scoring phase; arbitrary blocking callbacks need external supervision before real use.','Accidental-misuse protection, not hostile-Python isolation or source authenticity.','Private vascular encounters remain geometric annotation surrogates; unknown coverage and clinical injury probabilities remain unresolved/null.','Original suite training/inference accounting remains caller-declared and preserved; top-level pre-private replay counts and per-method reported replay counts describe the same preflight work, not two additional replays.'],'tracked_writes_by_reviewer':False,'root_only_integration':True}
(OUT/'review.json').write_text(json.dumps(review,indent=2,sort_keys=True)+'\n')
text='''# Independent matched private vascular adapter review

GO for generated-only source integration by root. All 52 small controls pass in 14.71 seconds: 19 adapter controls, 30 existing single-method evaluator regressions, and 3 independent callback-mutation controls. Relevant source hashes were unchanged before and after testing. No model forward/training, patient/model/archive payload or network access occurred; Torch was not imported.

The initial adapter had a concrete cross-method reference substitution bug: the SEARCH callback could replace IL's caller-owned binding and loader after common-reference preflight. A generated reproduction accepted SEARCH with 2 positive cells and IL with 0 in the same completed batch. The repair copies the mappings and captures immutable per-method and common-world fingerprints before private access. It checks those expectations before each later loader/scoring and again at batch completion. Recomputed in-place binding fingerprints cannot authorize a change; late mutation of an already-scored binding invalidates the batch.

All four terminal method slots and complete strategy seals are durable before any private callback. All nominal replay and geometry preflight completes first; the scoring loop makes no planning decisions or nominal replays. Failed/abstained methods retain null slots, real STOP histories remain distinguishable, and private failures preserve the fixed denominator without exporting exception text. Pre-scored outcomes and nonzero/non-integer generated-session counters are rejected.

AST comparison confirms the original geometry and other evaluator definitions are unchanged. The extracted scoring body is identical apart from taking its previously captured expected binding hash. The existing single-method API retains its behavior and passes all 30 regressions. The geometry streaming work remains separate.

This is a generated interface control, not a learned-method or patient experiment. Existing GENERATED_T1 records are explicitly rejected rather than relabelled T1. Real-person/source/role admission remains unimplemented. Cooperative phase limits are not a hard supervisor; reference coverage is not biological completeness, and no clinical-injury or generalized repair claim follows. The boundary protects against accidental misuse, not hostile Python code.

The original outer test receipt records FAIL solely because its broad archive guard blocked three Python standard-library python312.zip import lookups; pytest itself passed all 52 controls. The unchanged original receipt and log are retained. test-classification.json records the explicit runtime-library classification, with zero patient/model/archive/network attempts.

Exact SHA-256 bindings:

'''
for name,value in expected.items():text+=f'- `{name}`: `{value}`\n'
text+=f'\nMachine review: `review.json` SHA-256 `{sha(OUT/"review.json")}`. Supporting evidence hashes are included there.\n'
(OUT/'REPORT.md').write_text(text)
print(json.dumps({'decision':review['decision'],'report_sha256':sha(OUT/'REPORT.md'),'review_sha256':sha(OUT/'review.json'),'test_classification_sha256':sha(OUT/'test-classification.json')},indent=2))

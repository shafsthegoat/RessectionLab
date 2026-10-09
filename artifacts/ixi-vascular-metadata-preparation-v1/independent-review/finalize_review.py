"""Finalize bounded metadata review; no payload, model or network calls."""
from pathlib import Path
import ast,hashlib,json
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent;C=ROOT/'build/ixi-vascular-admission-v1'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
expected={'ixi_vascular_admission.py':'7c93cf8faf48ac55b0add1558d47f691ab63dc69672df8cd6fab83fc439ca28d','test_admission.py':'20b44cb5607e3d7efea8048132f7461bf625f9c77903c4cda249491b6381b34b','candidate.patch':'8e4ee0528ff0117068e2b4b861e597e21c49ae523f3111c9edfe61c40f8d970f','next-bridge.json':'1c3763af63cc9555a94ef8a92fa8daf09e7ba5f547a36a489dbf4b30d4c2e1a0'}
for name,value in expected.items():assert sha(C/name)==value
receipt=json.loads((OUT/'test-review.json').read_text())
assert receipt['status']=='PASS' and receipt['pytest_exit_code']==0 and not receipt['blocked_payload_or_network_attempts']
assert not receipt['policy_library_imported'] and not receipt['image_library_imported']
assert receipt['source_hashes_before']==receipt['source_hashes_after']
for path,value in receipt['source_hashes_after'].items():assert sha(ROOT/path)==value
# Confirm promotion patch adds the exact source and preserves all test function bodies/decorators.
patch=(C/'candidate.patch').read_text().splitlines(keepends=True);parts={};target=None
for line in patch:
 if line.startswith('+++ b/'):
  target=line.removeprefix('+++ b/').strip();parts[target]=[]
 elif line.startswith('--- '):target=None
 elif target and line.startswith('+'):parts[target].append(line[1:])
source=''.join(parts['src/resectionlab/ixi_vascular_admission.py'])
assert source==(C/'ixi_vascular_admission.py').read_text()
test_path=next(p for p in parts if p.startswith('tests/'));promoted=''.join(parts[test_path])
def functions(code):return {n.name:ast.dump(n) for n in ast.parse(code).body if isinstance(n,ast.FunctionDef)}
assert functions(promoted)==functions((C/'test_admission.py').read_text())
report={'schema':'ixi-vascular-admission-independent-review-v1','decision':'GO_metadata_preparation_source_integration_only','candidate_sha256':expected,'canonical_cohort_sha256':'b27f28c21089e61f6c4899f172f608cd0f81391588777443939196b1700cb127','tests':{'author_generated_controls':43,'independent_metadata_controls':15,'passed':58,'pytest_seconds':0.25,'source_and_cohort_bytes_unchanged':True,'promotion_source_exact_and_test_functions_unchanged':True,'actual_payload_or_network_attempts':0,'model_library_imported':False,'image_library_imported':False},'evidence_sha256':{n:sha(OUT/n) for n in ['test-review.json','combined-tests.txt','test_independent_admission.py','run_review_tests.py']},'reviewed_boundaries':['Exact frozen cohort bytes and person-to-role/member mapping; no reassignment after missing annotation/pair or failed preparation.','TRAIN/SELECT/MEASUREMENT_EVAL purpose restrictions, with no optimizer authorization granted.','Actor projection contains only T1, T1-derived support/task and public identity; private MRA/annotation/registration/coverage changes do not alter it.','Every receipt carries its exact bounded intended-use QC scope; generic pass/header-only/wrong-scope/missing-scope receipts refuse.','Unknown acquisition/private availability remains unknown; retrospective T1/support/task availability must precede decision cutoff; no preoperative claim is invented.','T1-only ancestry, permitted fit-role assertions, preserved pretrained-exposure uncertainty, fixed healthy waypoint meaning and horizon1.','Directed T1-to-MRA proper rigid registration, bound source/frame/valid-domain identities, annotation-MRA grid agreement and coverage-domain linkage.','Unlabelled coverage means unknown, not vessel-free; positive-outside-domain count requires exact integer0.','Detached immutable actor and evaluator contracts; generated evidence labels remain generated and current execution always returns false.','Existing planner, evaluator and streaming source untouched; no real source/array, model, training or native execution.'],'resolved_finding':{'initial_source_sha256':'9d8f2c6d28be522ce705f95650cef19fbf1f497534074dd61f1ba37db6fe1a74','issue':'Generic qc_status pass plus record hash did not distinguish header-only evidence from anatomy/intended-use, registration or coverage review.','repair':'Seven exact qc_scope names keyed by receipt kind are mandatory; seven author scope controls plus seven independent omitted-scope controls verify refusal.','status':'closed'},'limits':['This checks caller-supplied reviewed metadata assertions and content hashes; it does not authenticate receipts, perform anatomy QC, validate actual arrays/coverage, or prove registration quality.','The reviewed_evidence_sha256 and real_source_receipts label require an external trusted review/provenance process. Consistently relabelled and rehashed fabricated evidence cannot be detected by metadata content binding alone.','No extraction, private loading, model inference, fitting, selection or clinical use is released. execution_admitted and patient_payloads_validated remain false.','Current DEVELOPMENT/tumor/preoperative planner and generated-person evaluator restrictions remain unchanged. New retrospective objective/spec and real-person sealed scoring branches are future separately reviewed work.','Current 16^3 planning/32^3 reference execution caps remain unchanged; metadata shape bounds do not grant real-volume execution.','These preparation/QC requirements do not gate or interrupt ongoing opaque archive byte intake.','No rich-training transfer, calibrated abstention, vessel completeness, glioma, clinical injury or unseen-pretrained claim follows.'],'tracked_edits_by_reviewer':False,'root_owns_integration':True}
(OUT/'review.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
text='''# Independent IXI vascular metadata preparation review

GO for root-only integration of the metadata preparation source. All 58 controls pass in 0.25 seconds: 43 author controls and 15 independent controls. Exact source, test and frozen cohort bytes stayed unchanged. The promotion patch contains the exact reviewed source and preserves all test function bodies. No payload, model, image-library, network, training or native execution occurred.

The helper binds exact frozen person roles and source members; missing or mismatched assets do not cause reassignment. TRAIN/SELECT/MEASUREMENT_EVAL uses remain distinct. Actor metadata contains T1 and its support/task dependencies; MRA, annotation, registration and coverage stay in the evaluator contract. Independently changing private frames, registration and coverage leaves the actor projection unchanged. Both projections are detached and immutable.

The initial generic QC pass/hash design could confuse header-only evidence with intended-use review. The repaired source requires seven exact QC scopes for T1, MRA, derived annotation, support, task, directed registration/domain and coverage. Header-only, wrong-kind and omitted scopes are rejected. This checks explicit reviewed assertions; it does not perform or authenticate QC.

Unknown acquisition and private availability times remain unknown. Actor T1/support/task availability must precede the retrospective cutoff, with no invented preoperative claim. T1-only ancestry, fit-role assertions, source/frame direction, proper rigid registration, annotation-MRA grid linkage and qualified coverage semantics are checked. Unknown coverage does not imply vessel absence.

Execution remains unadmitted and patient payloads unvalidated on every prepared result. Existing DEVELOPMENT/tumor/preoperative and generated-only evaluator gates remain unchanged; real-person execution, a distinct healthy waypoint objective and actual array/registration validation are future work. Caller-provided hashes and evidence-domain labels bind assertions, not provenance authenticity. These preparation requirements do not interrupt or add gates to ongoing opaque byte intake.

Exact SHA-256 bindings:

'''
for name,value in expected.items():text+=f'- `{name}`: `{value}`\n'
text+=f'\nMachine review: `review.json` SHA-256 `{sha(OUT/"review.json")}`. Supporting evidence hashes are recorded there.\n'
(OUT/'REPORT.md').write_text(text)
print(json.dumps({'decision':report['decision'],'report_sha256':sha(OUT/'REPORT.md'),'review_sha256':sha(OUT/'review.json'),'test_review_sha256':sha(OUT/'test-review.json')},indent=2))

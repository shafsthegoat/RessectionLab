"""Bind only the reviewed role/scope freeze deltas; never launch or read payloads."""
from pathlib import Path
import argparse,hashlib,json,subprocess,sys
R=Path(__file__).resolve().parents[2];P=R/'build/ixi-paired-intake-preparation-v1';B=R/'data/acquisition/ixi-t1-mra-vessel-v1'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
C='manifests/experiments/ixi-component-person-cohort-v1.json';S='manifests/experiments/ixi-t1-mra-vessel-byte-intake-v1.json'
EXPECTED={'proposed-cohort.json':'cc6641d25162b7016089e29256c6870d2427642d58abcd4e6e46f44e67fad716','proposed-scope.json':'7a09c828609cc0f7f91ab6f1466c902ac1ca1dfc91da7db307e0942f66e1e586','run-intake.py':'8101d3c3edfea9d9099848468dbbfe24157bcd552bb16854f94b9c2c5a297dc2','prepared-declaration.json':'1d3435db6f164855a654939e72ba90e222b7e6cd9388defea7f1004a1d9b7f0a'}
def main():
 parser=argparse.ArgumentParser();parser.add_argument('--commit',required=True);a=parser.parse_args()
 for n,h in EXPECTED.items():assert sha(P/n)==h,n
 commit=subprocess.check_output(['git','rev-parse',a.commit+'^{commit}'],cwd=R,text=True).strip()
 subprocess.run(['git','merge-base','--is-ancestor',commit,'HEAD'],cwd=R,check=True)
 for rel in [C,S]:assert subprocess.check_output(['git','show',commit+':'+rel],cwd=R)==(R/rel).read_bytes(),'uncommitted freeze'
 oldc=json.loads((P/'proposed-cohort.json').read_text());newc=json.loads((R/C).read_text());expected=dict(oldc,declaration_status='frozen_before_archive_body_intake');assert newc==expected,'role delta outside reviewed freeze'
 olds=json.loads((P/'proposed-scope.json').read_text());news=json.loads((R/S).read_text());expected=json.loads(json.dumps(olds));expected.update(declaration_status='frozen_before_archive_body_intake',cohort_path=C,cohort_sha256=sha(R/C));expected['transport']['execution_released']=True;assert news==expected,'scope delta outside reviewed freeze'
 prepared=P/'prepared-declaration.json';d=json.loads(prepared.read_text())
 for rel,h in d['metadata_files_sha256'].items():assert sha(R/rel)==h,rel
 for n,h in d['source_files_sha256'].items():assert sha(B/'source'/n)==h,n
 assert not (B/'declaration.json').exists(),'preserve released declaration'
 # Preserve the candidate source copies before the sole authorized replacement.
 sys.path.insert(0,str(B/'source'));import real_intake_io as io
 assert Path(io.__file__).resolve()==B/'source/real_intake_io.py'
 for n in ['cohort.json','scope.json']:io.atomic_preserve(B/'prepared-source'/n,(B/'source'/n).read_bytes())
 for n,rel in [('cohort.json',C),('scope.json',S)]:(B/'source'/n).write_bytes((R/rel).read_bytes())
 d.update(canonical_cohort_path=C,canonical_cohort_sha256=sha(R/C),scope_path=S,scope_sha256=sha(R/S),role_commit=commit,scope_commit=commit,execution_released=True,scope_frozen_before_payload_access=True)
 d['source_files_sha256']['cohort.json']=sha(R/C);d['source_files_sha256']['scope.json']=sha(R/S)
 io.atomic_preserve(B/'declaration.json',(json.dumps(d,indent=2,sort_keys=True)+'\n').encode())
 released_sha=sha(B/'declaration.json')
 subprocess.run([sys.executable,str(P/'run-intake.py'),'--declaration',str(B/'declaration.json'),'--declaration-sha256',released_sha,'--check-only'],cwd=R,check=True)
 receipt={'prepared':EXPECTED,'scope_commit':commit,'canonical_cohort_sha256':sha(R/C),'canonical_scope_sha256':sha(R/S),'released_declaration_sha256':released_sha,'permitted_scope_and_role_delta_only':True,'runner_and_helpers_unchanged':True,'payload_bytes_read':0,'network_requests':0,'execution_launched':False}
 io.atomic_preserve(P/'root-release-binding.json',(json.dumps(receipt,indent=2,sort_keys=True)+'\n').encode())
 print(json.dumps(receipt,indent=2))
if __name__=='__main__':main()

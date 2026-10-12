import pathlib,sys,runpy,json
HERE=pathlib.Path(__file__).resolve().parent
sys.argv=[str(HERE.parent/'bootstrap_smoke.py'),'--output',str(HERE/'bootstrap')]
runpy.run_path(str(HERE.parent/'bootstrap_smoke.py'),run_name='__main__')
r=json.loads((HERE/'bootstrap/receipt.json').read_text())
assert r['status']=='passed' and r['payload_open_attempts']==[]
assert r['native_calls']==r['policy_forwards']==r['optimizer_steps']==0
print('Actual isolated worker imports passed; no scientific payload opens.')

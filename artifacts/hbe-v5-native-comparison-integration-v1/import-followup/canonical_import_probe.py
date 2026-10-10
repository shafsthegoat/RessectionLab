"""Exact source copies, actual canonical imports, real guards; no raw-file access."""
from pathlib import Path
import ast,hashlib,inspect,json,os,sys
ROOT=Path(__file__).resolve().parents[2];HERE=Path(__file__).resolve().parent
PREP=ROOT/'build/hbe-v5-native-comparison-preparation-v1'
SOURCE=json.loads((PREP/'comparison-source-inventory.json').read_text())
BASE=HERE/'generated-canonical-root'
sha=lambda raw:hashlib.sha256(raw).hexdigest()
# Audit hook applies even to os.open, and rejects all child processes/network.
forbidden=(ROOT/'data/mechanics',ROOT/'outputs/mechanics')
def audit(event,args):
    if event=='open' and isinstance(args[0],(str,bytes,os.PathLike)):
        p=Path(os.fsdecode(args[0])).resolve()
        if any(p.is_relative_to(folder) for folder in forbidden):raise RuntimeError('Original mechanics input forbidden')
    if event in ('subprocess.Popen','os.system','os.posix_spawn','socket.connect'):
        raise RuntimeError('Process/network forbidden')
sys.addaudithook(audit)
for relative,digest in SOURCE['bindings'].items():
    candidate=PREP/'stage'/relative
    source=candidate if candidate.exists() else ROOT/relative
    raw=source.read_bytes();assert sha(raw)==digest
    target=BASE/relative;target.parent.mkdir(parents=True,exist_ok=True)
    if target.exists():assert target.read_bytes()==raw
    else:target.write_bytes(raw)
sys.path.insert(0,str(BASE))
import scripts
scripts.__path__=[str(BASE/'scripts')]
from scripts import mechanics_hbe_v5_native_comparison as n
assert Path(n.__file__).resolve()==BASE/'scripts/mechanics_hbe_v5_native_comparison.py'
origins=n._verify_comparison_sources(BASE,SOURCE['bindings'])
frozen=tuple(n.remaining.SOURCE_PATHS)
try:n.remaining.audit_imports(root=BASE)
except ValueError as exc:refusal=str(exc)
else:raise AssertionError('Native-launch guard was weakened')
assert frozen==tuple(n.remaining.SOURCE_PATHS) and len(frozen)==20
unsupported={name:path for name,path in origins.items() if name.startswith('scripts.') and path not in frozen}
assert 'scripts.mechanics_hbe_v5_native_comparison' in unsupported
entry=ast.parse(inspect.getsource(n.compare_native_rows));chain=ast.parse(inspect.getsource(n.remaining.validate_prior_chain))
callnames=lambda tree:[ast.unparse(x.func) for x in ast.walk(tree) if isinstance(x,ast.Call)]
assert callnames(entry).count('remaining.validate_prior_chain')==2
assert not any(x.endswith(('audit_imports','validate_release')) for x in callnames(entry)+callnames(chain))
# Bounded canonical import/source guard is exercised. Never call native entry or
# its bulk verifiers: the separate generated tests exercise orchestration seams.
result={'schema':'hbe-v5-canonical-import-boundary-probe-v1','source_bindings_verified':len(SOURCE['bindings']),
    'loaded_canonical_origins':origins,'native_launch_allowlist_size':len(frozen),
    'native_launch_audit_refusal':refusal,'separately_bound_comparison_modules':unsupported,
    'read_only_entry_calls_validate_prior_chain_twice':True,'entry_calls_native_validate_release':False,
    'entry_or_chain_calls_native_import_audit':False,'bulk_verifiers_called':0,
    'native_calls':0,'saved_stream_replays':0,'original_mechanics_inputs_opened':0,
    'source_bindings_unchanged_after':n._verify_comparison_sources(BASE,SOURCE['bindings'])==origins}
(HERE/'canonical-import-result.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
print(json.dumps({'status':'PASS_distinct_boundaries','sources':len(SOURCE['bindings']),
    'loaded_modules':len(origins),'separate_modules':len(unsupported),'native_audit_refusal':refusal}))

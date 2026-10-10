"""Freeze source-only candidate metadata; never invoke native admission."""
from pathlib import Path
import difflib,hashlib,json,sys
ROOT=Path(__file__).resolve().parents[2];HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT));import scripts
scripts.__path__=[str(HERE/'stage/scripts'),*scripts.__path__]
from scripts import mechanics_hbe_v5_native_comparison as native
sha=lambda b:hashlib.sha256(b).hexdigest()
def save(name,value):
    (HERE/name).write_text(json.dumps(value,indent=2,sort_keys=True)+'\n')
# Fix only an overbroad wording: embedded N8 numerical arrays were parsed as part
# of the existing receipt but never interpreted by the mapping utility.
for name in ('map_existing_evidence.py','existing-evidence-mapping.json'):
    p=HERE/name;p.write_text(p.read_text().replace('No readout arrays or raw primitive files were opened.',
        'No separate readout files or raw primitive files were opened; numerical arrays were not interpreted.'))
# Canonical tests differ only in repository/path selection.
test=(HERE/'test_generated.py').read_text()
test=test.replace("ROOT=Path(__file__).resolve().parents[2]\nHERE=Path(__file__).resolve().parent", "ROOT=Path(__file__).resolve().parents[1]")
test=test.replace("scripts.__path__ = [str(HERE/'stage/scripts'), *scripts.__path__]\n",'')
target=HERE/'stage/tests/test_mechanics_hbe_v5_native_comparison.py';target.parent.mkdir(parents=True,exist_ok=True);target.write_text(test)
paths=['scripts/mechanics_hbe_v5_comparison.py','scripts/mechanics_hbe_v5_native_comparison.py','tests/test_mechanics_hbe_v5_native_comparison.py']
patch=[];source_rows={}
for relative in paths:
    old=ROOT/relative;new=HERE/'stage'/relative
    original=old.read_bytes() if old.exists() else b'';candidate=new.read_bytes()
    patch.extend(difflib.unified_diff(original.decode().splitlines(True),candidate.decode().splitlines(True),
        fromfile='a/'+relative if old.exists() else '/dev/null',tofile='b/'+relative))
    source_rows[relative]={'original_sha256':sha(original) if old.exists() else None,'candidate_sha256':sha(candidate),'bytes':len(candidate)}
(HERE/'source.patch').write_text(''.join(patch))
save('promotion-paths.json',source_rows)
# Bind all currently loaded repository project sources; staged candidates map to
# their canonical destination. Native input manifest will copy this binding map.
bindings={};origins={}
for name,module in sorted(sys.modules.items()):
    if not name.startswith(('scripts.','launchers.','resectionlab.')):continue
    p=Path(module.__file__).resolve()
    relative=str(p.relative_to(HERE/'stage')) if p.is_relative_to(HERE/'stage') else str(p.relative_to(ROOT))
    bindings[relative]=sha(p.read_bytes());origins[name]=relative
for relative in set(native.remaining.SOURCE_PATHS) | set(native.supplement.HISTORICAL_REPLAY_SOURCE_PATHS):
    bindings.setdefault(relative,sha((ROOT/relative).read_bytes()))
save('comparison-source-inventory.json',{'schema':'hbe-v5-comparison-source-inventory-v1',
    'bindings':bindings,'loaded_origins':origins,'runtime_binary_or_cache_authentication':False,
    'note':'Source-only inventory for canonical promotion; caller binds a fresh supervised runtime separately.'})
files=['stage/'+p for p in paths]+['test_generated.py','run_controls.py','source.patch','shared-extraction.json',
    'comparison-source-inventory.json','promotion-paths.json','existing-evidence-mapping.json','generated-controls-v9.txt']
save('candidate-pins.json',{name:{'sha256':sha((HERE/name).read_bytes()),'bytes':(HERE/name).stat().st_size} for name in files})
print(json.dumps({'source_paths':source_rows,'source_inventory_count':len(bindings),'pins_sha256':sha((HERE/'candidate-pins.json').read_bytes())},indent=2))

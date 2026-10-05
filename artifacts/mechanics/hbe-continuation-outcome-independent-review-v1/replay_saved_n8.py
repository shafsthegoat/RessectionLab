"""One saved N8 readout replay; no solver invocation or measured archive access."""
import hashlib
import json
from pathlib import Path
import sys

root = Path(sys.argv[1]).resolve()
frozen = Path(sys.argv[2]).resolve()
out = Path(sys.argv[3]).resolve()
sys.path.insert(0, str(frozen))
from scripts import mechanics_hbe_readout as readout
from scripts import mechanics_hbe_access as access
from scripts import mechanics_hbe_outputs as parser
from scripts import mechanics_hbe_physics as physics

sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
modules = {
    readout: '21700558e5cb614a11a7cc62dbd5ec8118fabb11b4fbfc88f89cb8aa02012963',
    parser: 'a372cb89ea9515186d433c74ec650dc3b5cae289619e8b35d481280f6662f473',
    physics: '00b8b5c87696609178a673ed98bbbe706ec524b4a8a09d0f0a55a85f53fb345d',
    access: 'b1e4f90b2ec84c39949ee02c1088bc0404f34904e24c2d4f328d503d96f76fe0',
}
for module, expected in modules.items():
    assert Path(module.__file__).resolve().is_relative_to(frozen)
    assert sha(module.__file__) == expected
summary = json.loads((root/'artifacts/mechanics/hbe-01-03-continuation-v1/medium-readout-summary.json').read_text())
execution = access.verify_binding(root, summary['execution_binding'], read_json=True)
saved = access.verify_binding(root, summary['readout_binding'], maximum_bytes=16*1024**2, read_json=True)
receipt, cache = readout.read_run(root, execution['primitive_bindings'],
    protocol_sha256='ab4385f5ad315d2444ca3aedc87b455db0d36539cea4e2119114d803fba49035',
    expected_branch='compression', expected_mesh_N=8, expected_steps=60,
    expected_mu_Pa=1000., retain_scale_primitives=False)
assert cache is None
receipt['execution_binding'] = summary['execution_binding']
assert access.canonical_json(receipt) == access.canonical_json(saved)
assert receipt['passed'] is True and receipt['frame_count'] == 61
access.verify_run_execution(root, 'compression:N8:S60:reference', receipt, execution['protocol_sha256'])
for module, expected in modules.items():
    assert sha(module.__file__) == expected
for record in execution['primitive_bindings'].values():
    access.verify_binding(root, record)
result = {'status': 'passed_exact_saved_n8_readout_reproduction',
          'saved_readout_binding': summary['readout_binding'],
          'execution_binding': summary['execution_binding'],
          'canonical_full_readout_sha256': hashlib.sha256(access.canonical_json(receipt)).hexdigest(),
          'frame_count': receipt['frame_count'], 'criteria': receipt['criteria'],
          'actual_source_hashes': {str(Path(mod.__file__).relative_to(frozen)): expected for mod, expected in modules.items()},
          'sources_and_primitive_inputs_unchanged': True,
          'new_solver_calls': 0, 'measured_response_archive_access': False}
with out.open('x') as stream:
    json.dump(result, stream, indent=2, sort_keys=True)
    stream.write('\n')
print('Complete saved N8 readout reproduced exactly; 61 states; no new solve.')

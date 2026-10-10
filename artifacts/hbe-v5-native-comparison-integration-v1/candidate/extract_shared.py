"""Factor admission-neutral numerical math without changing its statements."""
import ast
import hashlib
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent
path=ROOT/'scripts/mechanics_hbe_v5_comparison.py'
old=path.read_text()
text=old
start=text.index('    coordinates = readout.get(')
stop=text.index('\n\ndef _signed_pair',start)
numeric=text[start:stop]
text=text[:start]+'''    return _numerical_view(study, prior, run_id, readout)


def _numerical_view(study: dict, prior: dict, run_id: str, readout: dict) -> tuple[dict, bool]:
    """Shared exact numerical contract; provenance admission belongs to callers."""
    spec = v5.run_spec(study, prior, run_id)
    schedule = v5.schedule(study, prior, run_id)
'''+numeric+text[stop:]
start=text.index('    coverage = {run_id: v4.strict_coverage(')
math_body=text[start:]
# Only rename the two provenance-specific result keys and omit release claims;
# every numerical statement and returned diagnostic expression is unchanged.
math_body=math_body.replace("'schema': 'hbe-v5-twelve-row-generated-diagnostic-v1',\n            ", '')
math_body=math_body.replace("'individual_generated_stream_pass': individual", "'individual_stream_pass': individual")
math_body=math_body.replace("'generated_diagnostic_screen_passed': bool(screen),", "'diagnostic_screen_passed': bool(screen),")
finish=math_body.index("            'numerical_qualification_passed': False,")
math_body=math_body[:finish]+"            'continuum_error_bound': False}\n"
text=text[:start]+'''    result = _compare_views(prior, ordered, views, individual, historical)
    result['individual_generated_stream_pass'] = result.pop('individual_stream_pass')
    result['generated_diagnostic_screen_passed'] = result.pop('diagnostic_screen_passed')
    return {**result, 'schema': 'hbe-v5-twelve-row-generated-diagnostic-v1',
            'numerical_qualification_passed': False, 'native_execution_admitted': False,
            'adapted_deck_authenticated': False, 'source_deck_bytes_authenticated': False,
            'physical_validation_pass': None, 'calibration_released': False,
            'measured_response_accessed': False, 'patient_data_accessed': False}


def _compare_views(prior: dict, ordered: list, views: dict, individual: dict, historical: dict) -> dict:
    """Admission-neutral frozen numerical arithmetic, limits and classifications."""
'''+math_body
ast.parse(text)
oldtree,newtree=ast.parse(old),ast.parse(text)
oldnodes={x.name:x for x in oldtree.body if isinstance(x,ast.FunctionDef)}
newnodes={x.name:x for x in newtree.body if isinstance(x,ast.FunctionDef)}
unchanged=[]
for name in oldnodes:
    if name not in ('_view','compare_generated_rows'):
        assert ast.dump(oldnodes[name])==ast.dump(newnodes[name]);unchanged.append(name)
orig=oldnodes['compare_generated_rows'].body
begin=next(i for i,n in enumerate(orig) if isinstance(n,ast.Assign) and getattr(n.targets[0],'id',None)=='coverage')
assert ast.dump(ast.Module(body=orig[begin:-1],type_ignores=[]))==ast.dump(ast.Module(body=newnodes['_compare_views'].body[1:-1],type_ignores=[]))
orig=oldnodes['_view'].body
begin=next(i for i,n in enumerate(orig) if isinstance(n,ast.Assign) and getattr(n.targets[0],'id',None)=='coordinates')
assert ast.dump(ast.Module(body=orig[begin:],type_ignores=[]))==ast.dump(ast.Module(body=newnodes['_numerical_view'].body[3:],type_ignores=[]))
out=HERE/'stage/scripts/mechanics_hbe_v5_comparison.py';out.write_text(text)
record={'original_sha256':hashlib.sha256(old.encode()).hexdigest(),'candidate_sha256':hashlib.sha256(text.encode()).hexdigest(),
        'unchanged_functions':unchanged,'comparison_numerical_statements_ast_equal':True,
        'row_numeric_contract_ast_equal':True,'generated_public_output_contract_preserved':True}
(HERE/'shared-extraction.json').write_text(json.dumps(record,indent=2,sort_keys=True)+'\n')
print(json.dumps(record))

"""Saved-metadata-only projection of the pure proposed group-order function."""
import ast
from collections import Counter
import hashlib,json
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
OLD=ROOT/'build/select013-residual-blocker-v1/attempt-01'
SOURCE=HERE/'stage/resectionlab/native_proposals.py'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text())
def main():
 need=OLD/'result.json'
 if sha(need)!='532581e440afdf656c8a3f24ed730e30beba45470588f9e5a337932e97ffb12c':raise ValueError('Prior actual result changed')
 tree=ast.parse(SOURCE.read_text());fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_target_ray_group_order')
 ns={};exec(compile(ast.Module(body=[fn],type_ignores=[]),str(SOURCE),'exec'),ns)
 catalog=read(OLD/'prospective/prospective-catalog.json')
 rows=[read(p) for p in sorted((OLD/'prospective').glob('base-ray-*.json'))]
 cells={tuple(r['blocker_voxel']) for r in catalog['ledger'] if r['family']=='obstruction_opening'}
 eligible={tuple(r['voxel']) for r in catalog['accounting']['static_aperture_pairs'] if not r['excluded']}
 tools=list(dict.fromkeys(r['proposal']['tool_id'] for r in rows));groups=[];counts=Counter()
 for row in rows:
  # These52 declared nominal-family endpoints are positive by the preserved
  # original proposer contract. No unrecorded array membership is invented.
  if row['proposal']['family'] in ('proximal_nominal','distal_nominal') and row['sidecar']:
   blockers=tuple(map(tuple,row['sidecar']['blocked_indices_native']))
   groups.append(((row['proposal']['column_index'],tools.index(row['proposal']['tool_id'])),blockers));counts.update(blockers)
 nominal={tuple(r['blocker_voxel']) for r in catalog['ledger'] if r['family']=='obstruction_opening' and r['public_nominal_priority']}
 order,coverage=ns['_target_ray_group_order'](cells,groups,eligible,nominal,16)
 prior=sorted({tuple(r['blocker_voxel']) for r in catalog['ledger'] if r['family']=='obstruction_opening' and r['reason']=='PROPOSED_UNCERTIFIED'})
 frequency=sorted(eligible,key=lambda c:(-counts[c],c));w=(7,92,85)
 result={'source_sha256':sha(SOURCE),'result_sha256':sha(need),'metadata_only':True,'native_queries':0,
  'array_or_weight_access':False,'tool_order':tools,'all35_retained_cells':sorted(cells),'aperture_eligible27':sorted(eligible),
  'public_nominal_member_cells':sorted(nominal),'old_selected16':prior,'new_selected16':order[:16],
  'all_cell_order':order,'groups':coverage,'newly_included':sorted(set(order[:16])-set(prior)),
  'newly_omitted':sorted(set(prior)-set(order[:16])),
  'witness':{'cell':w,'old_selected':w in prior,'frequency_rank':frequency.index(w)+1,'new_rank':order.index(w)+1},
  'unqueried_new_tool_pairs':True,'scope':'ordering projection from52 declared public nominal-endpoint rays and saved aperture tests; no native feasibility or complete-route prediction',
  'runtime_alias_rule':'runtime uses permitted endpoint membership, so coincident goal aliases do not depend on family name; saved projection uses positive membership established by nominal-family contract'}
 path=HERE/'saved-order-projection.json'
 with path.open('x') as f:json.dump(result,f,indent=2,sort_keys=True);f.write('\n')
 print(json.dumps({'output_sha256':sha(path),'witness':result['witness'],'newly_included':result['newly_included'],'newly_omitted':result['newly_omitted']}))
if __name__=='__main__':main()

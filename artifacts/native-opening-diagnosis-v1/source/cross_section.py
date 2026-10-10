"""Optimistic axis-cylinder upper bounds. No NumPy, patient arrays or native calls."""
import ast
import hashlib
import json
import math
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_assignment(path, name):
    tree = ast.parse(path.read_text())
    return next(n.value for n in tree.body if isinstance(n, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == name for t in n.targets))


def distances(cell, column, spacing):
    delta = [abs(a-b)*s for a,b,s in zip(cell,column,spacing)]
    return (math.hypot(*(max(0.,d-s/2) for d,s in zip(delta,spacing))),
            math.hypot(*(d+s/2 for d,s in zip(delta,spacing))))


def analyze(name, spacing, columns, tools):
    # Complete local enumeration: beyond this square no selected shaft or tip
    # reaches a cell. Both exact cell min/max transverse distances are retained.
    sy,sz = spacing[1:]
    margin = math.ceil(max(t['shaft_radius_mm'] for t in tools)/min(sy,sz))+3
    lo = min(min(c) for c in columns)-margin
    hi = max(max(c) for c in columns)+margin
    cells = [(y,z) for y in range(lo,hi+1) for z in range(lo,hi+1)]
    footprints = {t['tool_id']: {cell for cell in cells if any(
        distances(cell,c,(sy,sz))[1] <= t['tip_radius_mm']-1e-10 for c in columns)} for t in tools}
    union = set().union(*footprints.values())
    rows=[]
    for tool in tools:
        for column in columns:
            shaft = {cell for cell in cells if distances(cell,column,(sy,sz))[0] <= tool['shaft_radius_mm']+1e-9}
            blockers = sorted(shaft-union)
            witness = min(blockers,key=lambda cell: distances(cell,column,(sy,sz))[0]) if blockers else None
            minimum = None if witness is None else distances(witness,column,(sy,sz))[0]
            # In a solid half-space x>=0, an unremovable transverse cell
            # intersects the forward shaft cap when d reaches this limit.
            cap = None if minimum is None else tool['tip_length_mm']-math.sqrt(max(0.,tool['shaft_radius_mm']**2-minimum**2))
            rows.append({'tool_id':tool['tool_id'],'column':column,
                'shaft_cells':len(shaft),'all_tip_union_blockers':len(blockers),
                'witness':witness,'witness_min_transverse_mm':minimum,
                'solid_slab_tip_depth_upper_bound_mm':cap,
                'own_tool_union_blockers':len(shaft-footprints[tool['tool_id']])})
    return {'name':name,'spacing_xyz_mm':spacing,'columns':columns,
        'optimistic_infinite_cylinder_union_cells':len(union),'columns_by_tool':rows,
        'small_shaft_columns_without_static_all_tool_blocker':[r['column'] for r in rows if r['tool_id']==tools[0]['tool_id'] and not r['all_tip_union_blockers']],
        'large_shaft_columns_without_static_all_tool_blocker':[r['column'] for r in rows if r['tool_id']==tools[1]['tool_id'] and not r['all_tip_union_blockers']]}


def main():
    started=time.perf_counter()
    proposals=ROOT/'src/resectionlab/native_proposals.py'
    geometry=ROOT/'src/resectionlab/geometry.py'
    columns=ast.literal_eval(load_assignment(proposals,'DEFAULT_COLUMN_OFFSETS'))
    nodes=load_assignment(geometry,'GENERIC_TOOLS').elts
    tools=[]
    for node in nodes:
        assert isinstance(node,ast.Call) and isinstance(node.func,ast.Name) and node.func.id=='ToolGeometry'
        values=[ast.literal_eval(x) for x in node.args]
        keywords={x.arg:ast.literal_eval(x.value) for x in node.keywords}
        tools.append(dict(zip(('tool_id','tip_radius_mm','shaft_radius_mm','working_length_mm'),values),**keywords))
    assert [(t['tip_radius_mm'],t['shaft_radius_mm'],t['tip_length_mm']) for t in tools]==[(1.,1.4,3.),(1.7,2.7,4.)]
    paths=[ROOT/'build/remind-planning-qc-v1/ReMIND-008-public-inputs-v2.json']+[
        ROOT/f'build/remind-planning-qc-v1/public-cohort-v1/ReMIND-{s}-public-inputs-v2.json' for s in ('010','020','025')]
    metadata=[(json.loads(p.read_text())['patient_id'],json.loads(p.read_text())['reindex_policy']['derived_grid']['spacing_mm']) for p in paths]
    grids=[('isotropic-0.5',[.5]*3),('isotropic-1.0',[1.]*3)]+metadata
    original=[analyze(name,spacing,columns,tools) for name,spacing in grids]
    # Geometry-only comparison, never an action configuration change.
    plus_cardinal=columns+((-1,0),(1,0),(0,-1),(0,1))
    augmented=[analyze(name,spacing,plus_cardinal,tools) for name,spacing in grids]
    # Exact hand checks: 1 mm diagonal cell is not in the all-tool union,
    # but intersects the centre suction shaft; a 0.5 mm central shaft has no
    # static all-tool blocker. These do not assert sequential reachability.
    central=lambda report,tool: next(r for r in report['columns_by_tool'] if r['column']==(0,0) and r['tool_id']==tool)
    assert central(original[1],'generic_suction')['all_tip_union_blockers']>0
    assert central(original[0],'generic_suction')['all_tip_union_blockers']==0
    assert central(augmented[1],'generic_suction')['all_tip_union_blockers']==0
    result={'scope':'pure_geometric_solid_slab_optimistic_upper_bound_not_patient_reachability',
        'assumptions':['axis0 inward; orthogonal grid; slab x>=0 initially occupied with no pre-existing holes',
          'all declared columns available to both tools, even if actual aperture disallows them',
          'arbitrary stroke depths/horizon allowed; full infinite-cylinder containment is an optimistic superset of finite capsule removal',
          'one cell must fit one capsule; union of partially contacted slivers cannot remove it',
          'static union cover is necessary, not sufficient: prior-cavity clearance and connectivity still apply'],
        'tools':tools,'original':original,'comparison_plus_cardinal_one_only':augmented,
        'native_trajectories':0,'patient_array_reads':0,'model_calls':0,
        'source_sha256':{str(p.relative_to(ROOT)):sha(p) for p in (proposals,geometry,ROOT/'src/resectionlab/native_resection.py',Path(__file__),*paths)},
        'elapsed_seconds':time.perf_counter()-started}
    with (HERE/'result.json').open('x') as f:json.dump(result,f,indent=2,allow_nan=False);f.write('\n')
    for record in original:
        a,b=(central(record,t['tool_id']) for t in tools)
        print(record['name'], 'central small/large blockers',a['all_tip_union_blockers'],b['all_tip_union_blockers'],
              'central tip bounds',a['solid_slab_tip_depth_upper_bound_mm'],b['solid_slab_tip_depth_upper_bound_mm'],
              'unblocked small/large columns',len(record['small_shaft_columns_without_static_all_tool_blocker']),len(record['large_shaft_columns_without_static_all_tool_blocker']))
    print('elapsed',result['elapsed_seconds'])


if __name__=='__main__':main()

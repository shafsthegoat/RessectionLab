"""Saved JSON coordinate arithmetic only; no arrays, project imports or simulator."""
import hashlib
import itertools
import json
import math
from pathlib import Path

HERE=Path(__file__).resolve().parent
RUN=HERE/'attempt-01'
ROOT=HERE.parents[1]
refs={}
def load(path):
    raw=path.read_bytes();refs[str(path.relative_to(ROOT))]=hashlib.sha256(raw).hexdigest()
    return json.loads(raw)
def canonical(x):return json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False)
def semantic(x):return 'sha256:'+hashlib.sha256(canonical(x).encode()).hexdigest()
def add(a,b):return [x+y for x,y in zip(a,b)]
def sub(a,b):return [x-y for x,y in zip(a,b)]
def dot(a,b):return sum(x*y for x,y in zip(a,b))
def mul(a,s):return [x*s for x in a]
def norm(a):return math.sqrt(dot(a,a))

def main():
    result=load(RUN/'result.json');receipt=load(HERE/'attempt-01.supervision/receipt.json')
    assert result['status']=='complete_fixed_ray_witness' and receipt['status']=='complete'
    assert receipt['result_sha256']==refs[str((RUN/'result.json').relative_to(ROOT))]
    inventory=load(RUN/'initial-inventory.json');world=load(RUN/'world.json')
    base=load(RUN/'branch-00/fixed-ray-preview.json');B={tuple(x) for x in base['obstruction_diagnostic']['blocked_indices_native']}
    assert B=={(121,44,72)}
    byid={r['action_id']:r for r in inventory['emitted']};rows=[];all_removed=set();frame=None
    baseline_geometry={k:base['obstruction_diagnostic'][k] for k in ('failure_interval_index','planned_microsteps',
        'failure_tip_mm','previous_tip_mm','shaft_sweep_start_mm','shaft_sweep_end_mm','blocked_indices_hash')}
    for i,row in enumerate(result['branches']):
        d=load(RUN/f'branch-{i:02d}/fixed-ray-preview.json');b=d['obstruction_diagnostic']
        wrapper=load(RUN/f'branch-{i:02d}/plan.json');p=wrapper['plan'];assert semantic(p)==wrapper['plan_seal']
        assert {k:b[k] for k in baseline_geometry}==baseline_geometry
        assert b['complete_first_failure_set'] and b['blocked_indices_native']==[[121,44,72]]
        assert b['prior_temporary_removed_count']==0 and d['temporary_removals_committed'] is False
        assert d['blocker_occupancy']['in_added_T_minus_S']==1 and d['blocker_occupancy']['in_raw_S']==0
        removed={tuple(c) for h in p['history'] for c in h.get('removed_indices_native',[])}
        contacts={tuple(c) for h in p['history'] for c in h.get('contact_indices_native',[])}
        all_removed|=removed;prep=None if i==0 else byid[row['preparation_action']]
        if i:frame=p['history'][0]['native_affine']
        successor=None if i==0 else load(RUN/f'branch-{i:02d}/successor-inventory.json')
        rows.append({'ordinal':i,'tool':None if prep is None else prep['tool_id'],
            'family':None if prep is None else prep['family'],'column':None if prep is None else prep['column_index'],
            'endpoint_voxel':None if prep is None else prep['voxel'],
            'removed_cells':sorted(removed),'removed_blockers':sorted(removed&B),'contacted_blockers':sorted(contacts&B),
            'fixed_ray_failure_interval':b['failure_interval_index'],'temporary_ray_removed_cells':0,
            'target_removed_mm3':row['outcomes']['target_removed_mm3'],'total_return':row['outcomes']['total_reward'],
            'successor_omissions':None if successor is None else successor['omitted_count']})
    assert len(rows)==16 and all(not r['removed_blockers'] and r['target_removed_mm3']==0 for r in rows)
    assert all(r['successor_omissions']==0 for r in rows[1:])
    assert result['selected_ordinal']==0 and all(r['total_return']<0 for r in rows[1:])
    public=world['common_public_world'];normal=public['access']['normal_inward'];normal=mul(normal,1/norm(normal))
    def point(q):return [sum(frame[i][j]*q[j] for j in range(3))+frame[i][3] for i in range(3)]
    blocker=next(iter(B));center=point(blocker)
    corners=[point(add(blocker,o)) for o in itertools.product((-.5,.5),repeat=3)]
    def radial(x,entry):
        delta=sub(x,entry);return norm(sub(delta,mul(normal,dot(delta,normal))))
    columns={}
    for row in inventory['emitted']:columns.setdefault(row['column_index'],row)
    assert len(columns)==13
    radial_bounds=[{'column':i,'offset_source_voxels':r['offset_source_voxels'],
        'transverse_native':r['voxel'][1:],'blocker_center_distance_mm':radial(center,r['entry_mm']),
        'max_corner_distance_to_infinite_axis_mm':max(radial(x,r['entry_mm']) for x in corners)}
        for i,r in sorted(columns.items())]
    best=min(radial_bounds,key=lambda r:r['max_corner_distance_to_infinite_axis_mm'])
    max_tip=max(t['tip_radius_mm'] for t in public['tools'])
    assert best['max_corner_distance_to_infinite_axis_mm']>max_tip
    depth=dot(sub(center,public['access']['center_mm']),normal);entry=sub(center,mul(normal,depth))
    suction=next(t for t in public['tools'] if t['tool_id']=='generic_suction')
    margin=public['access']['radius_mm']-radial(entry,public['access']['center_mm'])-max(suction['tip_radius_mm'],suction['shaft_radius_mm'])
    proposed={'status':'proposed_only_not_previewed_or_executable_inventory_action','tool_id':suction['tool_id'],
        'axis_rule':'source-normal line through the recorded public first blocker cell center',
        'endpoint_voxel':list(blocker),'offset_source_voxels':[-1,4],'entry_mm':entry,'tip_mm':center,
        'insertion_depth_mm':depth,'normal_aperture_envelope_margin_mm':margin,
        'max_corner_radial_distance_mm':max(radial(x,entry) for x in corners),
        'necessary_checks_only':True,'full_shaft_temporal_exposure_and_connected_removal':'unmeasured; unchanged native preview required',
        'next_experiment':'One root-released diagnostic preview of this single missing clearance axis in the same union025 root state. If feasible, separately version an opt-in shared proposal rule before any commit; then test clearing this cell followed by the original ray with all costs and replay. Do not silently inject the diagnostic ray.',
        'source_rule_followup':'A reusable rule must derive uncovered clearance axes from permitted occupancy/target geometry, preserve existing proposals, and bind a new decision identity; do not hardcode this patient cell.'}
    source_refs={}
    for name in ('native_proposals.py','native_resection.py'):
        p=ROOT/'src/resectionlab'/name;source_refs[str(p.relative_to(ROOT))]=hashlib.sha256(p.read_bytes()).hexdigest()
    output={'version':'union025-saved-blocker-diagnosis-v1','scope':'saved JSON and source bytes only; stdlib coordinate arithmetic; zero patient-array reads/native/model calls',
        'finding':'All15 preparations leave the same singleton first blocker and zero temporary fixed-ray cuts; no current fixed-axis tool capsule can fully contain that cell at any endpoint depth.',
        'blocker':{'native_index':list(blocker),'center_ras_mm':center,'public_occupancy_class':'added T minus raw S',
            'first_rejected_interval_index':12,'planned_microsteps':15,
            'previous_tip_depth_mm':dot(sub(base['obstruction_diagnostic']['previous_tip_mm'],base['entry_mm']),normal),
            'rejected_tip_depth_mm':dot(sub(base['failure_tip_mm'],base['entry_mm']),normal),
            'shaft_front_depth_mm':dot(sub(base['obstruction_diagnostic']['shaft_sweep_end_mm'],base['entry_mm']),normal),
            'fixed_ray_tip_radius_mm':suction['tip_radius_mm'],'fixed_ray_shaft_radius_mm':suction['shaft_radius_mm'],
            'first_prior_microstep_contacting_cell':next(i for i,m in enumerate(base['prior_completed_microsteps']) if list(blocker) in m['contact_indices_native']),
            'temporary_cells_removed_before_failure':0,'failure_pose_and_blockers_identical_all16':True},
        'branches':rows,'distinct_cells_removed_across_preparations':len(all_removed),
        'axis_containment':{'method':'Transform the saved native cell eight corners with recorded affine. For each of13 saved source-normal lines, take maximum corner radial distance to the infinite line. Distance to any finite active capsule segment is at least this distance.',
            'bounds':radial_bounds,'minimum_max_corner_distance_mm':best['max_corner_distance_to_infinite_axis_mm'],
            'nearest_column':best['column'],'maximum_declared_active_tip_radius_mm':max_tip,
            'minimum_excess_mm':best['max_corner_distance_to_infinite_axis_mm']-max_tip,
            'consequence':'Under these fixed13 lines, unchanged tips and all-corners whole-cell removal, this blocker cannot be removed by additional endpoint depths or longer histories. This is a specific fixed-ray obstruction proof, not global target unreachability.',
            'source_basis':'NominalCavityProposalProvider freezes _columns; all five enabled endpoint families use these same transverse columns. contained_capsule_cells requires all eight corners inside radius minus inward epsilon.'},
        'recommended_single_diagnostic':proposed,
        'not_explanations_here':['No cap omissions at root or15 recorded successor states','No missing fixed-target ray: it remains emitted but blocked','No reward/training issue explains this particular geometric refusal'],
        'limits':['No new axis was previewed. Aperture/radial necessary checks do not certify full-tool feasibility or connected removal.',
            'A successful first-cell clearance could expose another blocker; no positive-return completion is yet known.',
            'S union T remains an explicit unvalidated material assumption, with full target denominator77974.35860665982mm3.'],
        'costs':{'owned_seconds':receipt['elapsed_seconds'],'peak_sampled_rss_bytes':receipt['sampled_peak_rss_bytes'],
            'previews':3252,'rollout_transitions':31,'replay_transitions':31,'model_work':0},
        'saved_input_sha256':refs,'source_sha256':source_refs}
    target=HERE/'saved-blocker-diagnosis.json';target.write_text(json.dumps(output,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'path':str(target),'sha256':hashlib.sha256(target.read_bytes()).hexdigest(),
        'same_blocker_all16':True,'best_radial_bound_mm':best['max_corner_distance_to_infinite_axis_mm'],
        'largest_tip_mm':max_tip,'proposed_suction_aperture_margin_mm':margin,'diagnosis_only':True},indent=2))

if __name__=='__main__':main()

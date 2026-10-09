"""Pure-stdlib saved JSON/source audit; forbids subprocess/network/payload reads."""
from pathlib import Path
import hashlib
import json
import math
import os
import stat
import sys

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
PREP = 'build/private-vascular-streaming-preparation-v1/'
OUT = 'build/private-vascular-streaming-profile-v1/'
REVIEW = 'build/private-vascular-streaming-launch-independent-v1/REPORT.md'
REVIEW_SHA = 'cf47706a02860411205dacf6f830a4da09081c10e07ef1c7eb6c01df76375af3'
OBSERVED = {}
def guard(event,args):
    if event in ('subprocess.Popen','os.system','os.exec','socket.connect'):
        raise AssertionError('Saved audit cannot run children or network operations')
    if event=='open' and isinstance(args[0],(str,bytes)):
        assert not os.fsdecode(args[0]).lower().endswith(('.nii','.nii.gz','.mat','.pt','.ckpt','.npy','.npz','.dcm','.bin','.safetensors','.pkl','.h5'))
sys.addaudithook(guard)

def digest(raw): return hashlib.sha256(raw).hexdigest()
def raw(name):
    fd = os.open(ROOT/name,os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK)
    with os.fdopen(fd,'rb') as stream:
        st=os.fstat(stream.fileno()); assert stat.S_ISREG(st.st_mode) and st.st_size<=4*1024**2
        value=stream.read(4*1024**2+1); assert len(value)<=4*1024**2
    OBSERVED[name]={'bytes':len(value),'sha256':digest(value)}
    return value
def obj(name): return json.loads(raw(name))
def identity(value): return digest(json.dumps(value,sort_keys=True,allow_nan=False,separators=(',',':')).encode())
def same(a,b): return math.isclose(a,b,rel_tol=1e-12,abs_tol=1e-9)

receipt_bytes=raw(OUT+'receipt.json'); receipt=json.loads(receipt_bytes)
profile_bytes=raw(OUT+'worker/profile.json'); profile=json.loads(profile_bytes)
attempt=obj(OUT+'worker/attempt.json'); child=obj(OUT+'worker-release.json')
console=obj(OUT+'readout-console.txt')
release=obj(PREP+'root-supervisor-release.json'); disabled=obj(PREP+'supervisor-disabled-release.json')
preflight=obj(PREP+'root-preflight.json')
assert digest(raw(REVIEW))==REVIEW_SHA==preflight['review_sha256']
assert preflight['only_release_delta']=='execution_released=true' and preflight['source_bindings_verified'] is True
assert preflight['source_head']=='abbc3e31ef2dd9123e470a31b3701eb375f85424'
assert disabled['execution_released'] is False and release==dict(disabled,execution_released=True)
assert OBSERVED[PREP+'root-supervisor-release.json']['sha256']==receipt['root_release_sha256']==preflight['release_sha256']
assert OBSERVED[OUT+'worker-release.json']['sha256']==receipt['child_release_sha256']==release['child_release_sha256']==attempt['release_sha256']
assert receipt_bytes==(json.dumps(receipt,sort_keys=True,indent=2,allow_nan=False)+'\n').encode()
assert receipt['source_bindings']==release['source_bindings']
for name,expected in receipt['source_bindings'].items(): assert digest(raw(name))==expected
assert digest(raw(PREP+'launch_profile.py'))==receipt['launcher_sha256']==release['launcher_sha256']
inventory=obj(PREP+'repository-source-pins.json')
assert len(inventory)==77==attempt['repository_sources_verified']==profile['repository_sources_verified']
assert set(inventory)=={str(p.relative_to(ROOT)) for p in (ROOT/'src/resectionlab').rglob('*.py')}
for name,expected in inventory.items(): assert digest(raw(name))==expected
assert profile['source_bytes_unchanged'] is True and profile['prohibited_actions']==[]
assert profile['source_hashes']==attempt['source_hashes']==child['source_hashes']
for name,expected in profile['source_hashes'].items(): assert digest(raw(PREP+name))==expected
assert child=={'scope':'one_generated_256x256x192_streaming_contact_profile','root_release':True,'source_hashes':profile['source_hashes']}
assert profile['scope']==attempt['scope']==child['scope']
assert profile['root_external_supervision_required'] is True and attempt['root_external_supervision_required'] is True
assert release['patient_or_model_access_permitted'] is False and receipt['patient_or_model_access_permitted'] is False
assert attempt['patient_access'] is False and attempt['release'] is True and attempt['status']=='reserved_generated_profile'
assert attempt['coverage']=='partial_x_half' and attempt['repeated_action'] is True and attempt['spacing_mm']==.7
assert attempt['cooperative_wall_seconds']==20 and attempt['distinct_actions']==3 and attempt['capsules']==6
assert {p.name for p in (ROOT/OUT).iterdir()}=={'receipt.json','worker-release.json','readout-console.txt','worker'}
assert not (ROOT/OUT/'worker').is_symlink()
assert {p.name for p in (ROOT/OUT/'worker').iterdir()}=={'attempt.json','profile.json'}
assert set(receipt['output_bindings'])=={'readout-console.txt','worker-release.json','worker/attempt.json','worker/profile.json'}
for name,binding in receipt['output_bindings'].items(): assert OBSERVED[OUT+name]==binding
assert receipt['profile_sha256']==digest(profile_bytes) and receipt['profile_bytes']==len(profile_bytes)
actual_output_bytes=sum(OBSERVED[OUT+name]['bytes'] for name in ('receipt.json',*receipt['output_bindings']))
assert actual_output_bytes==15401

caps={'attempts':1,'worker_wall_seconds':35,'cleanup_reserve_seconds':10,'lifecycle_wall_seconds':45,
      'sampled_process_group_rss_bytes':536870912,'aggregate_output_bytes':4194304,'profile_json_bytes':1048576,'numerical_threads':1}
assert receipt['caps']==release['caps']==caps and actual_output_bytes<caps['aggregate_output_bytes']
assert receipt['status']=='generated_streaming_profile_complete' and receipt['readout_calls_attempted']==1
assert receipt['cleanup']=={'contained':True,'direct_child_reaped':True,'errors':[],'exit_code':0,'fallback_used':False,'remaining_members':[]}
stage=receipt['readout_stage']
assert stage['status']=='completed_within_caps' and stage['exit_code']==0 and stage['kill_reason'] is None
assert stage['wall_cap_seconds']==35 and stage['sampled_process_group_rss_cap_bytes']==536870912 and stage['active_output_cap_bytes']==4194304
assert stage['peak_sampled_active_output_bytes']<=actual_output_bytes and stage['peak_sampled_process_group_rss_bytes']<536870912
worker_cache=ROOT/OUT/'worker/unused-pycache'
assert stage['command']==[str(ROOT/'.venv/bin/python'),'-B','-X','pycache_prefix='+str(worker_cache),str(ROOT/PREP/'profile_generated.py'),
                          '--release',str(ROOT/OUT/'worker-release.json'),'--output-directory',str(ROOT/OUT/'worker')]
assert attempt['source_cache_prefix']==str(worker_cache)
for path in (worker_cache,ROOT/PREP/'launcher-unused-pycache'):
    assert not path.exists() and not path.is_symlink()
assert 0<profile['profile_seconds']<=stage['elapsed_seconds']<=receipt['lifecycle_seconds']<45 and stage['elapsed_seconds']<35
assert 0<=profile['tracemalloc_current_bytes']<=profile['tracemalloc_peak_bytes']<536870912
assert profile['process_peak_rss_bytes']<536870912
assert console=={name:profile[name] for name in ('process_peak_rss_bytes','profile_seconds','tracemalloc_peak_bytes')}|{'status':'complete_generated_contact_profile'}
expected_origins={'resectionlab':'src/resectionlab/__init__.py','resectionlab.independent_geometry_batch':'src/resectionlab/independent_geometry_batch.py'}
assert profile['loaded_repository_origins_before']==profile['loaded_repository_origins_after']==expected_origins

result=profile['result']; work=result['work']; budget=result['budget']
assert result['schema']=='generated-tiled-vascular-contact-v1' and result['status']=='complete_generated_contact_profile'
assert result['patient_admission'] is False and result['strategy_replay_or_admission_performed'] is False
assert result['removed_overlap']=={'existing_congruence_requirement_unchanged':True,'outcomes':None,'status':'not_evaluated_by_contact_kernel'}
assert result['grid_shape']==attempt['grid_shape']==profile['grid']['shape']==[256,256,192]
assert result['grid_voxels']==attempt['grid_voxels']==math.prod(result['grid_shape'])==12582912
assert result['capsule_count']==6 and result['action_count']==3 and result['contact_tolerance_squared_mm2']==1e-10
assert 0<result['elapsed_seconds']<=profile['profile_seconds']<20
assert result['reference_grid_identity']==identity(profile['grid'])
assert result['capsules_identity']==identity(profile['capsule_records'])
assert result['reference_identity']==identity({'domain':'generated_fixture_only','grid':result['reference_grid_identity'],
                                               'pattern':'lattice','coverage':'partial_x_half','version':result['schema']})
angle=.37; rotation=((math.cos(angle),-math.sin(angle),0.),(math.sin(angle),math.cos(angle),0.),(0.,0.,1.))
translation=(-60.,-90.,-67.2)
for i in range(3):
    for j in range(3): assert same(profile['grid']['affine_ras_mm'][i][j],rotation[i][j]*.7)
    assert profile['grid']['affine_ras_mm'][i][3]==translation[i]
assert profile['grid']['affine_ras_mm'][3]==[0.,0.,0.,1.]
expected_caps=[]
for action in ('first','repeated'):
    expected_caps.extend(((action,'shaft',(10.,10.,-10.),(140.,140.,103.),1.5),(action,'tip',(50.,50.,25.),(140.,140.,103.),2.)))
expected_caps.extend((('crossing','shaft',(160.,10.,-10.),(30.,140.,103.),1.5),('crossing','tip',(120.,50.,25.),(30.,140.,103.),2.)))
outside_by_part={'shaft':False,'tip':False}; outside_by_action={}; legacy_counts=[]
for record,(action,part,start,end,radius) in zip(profile['capsule_records'],expected_caps,strict=True):
    assert record['action_id']==action and record['part']==part and record['radius_mm']==radius
    for name,point in (('start_ras_mm',start),('end_ras_mm',end)):
        assert all(same(record[name][i],sum(rotation[i][j]*point[j] for j in range(3))+translation[i]) for i in range(3))
    outside=any(min(start[i],end[i])-radius<-.35 or max(start[i],end[i])+radius>(result['grid_shape'][i]-.5)*.7 for i in range(3))
    outside_by_part[part]|=outside; outside_by_action[action]=outside_by_action.get(action,False)|outside
    lo=[max(math.floor((min(start[i],end[i])-radius)/.7-.5),0) for i in range(3)]
    hi=[min(math.ceil((max(start[i],end[i])+radius)/.7+.5),result['grid_shape'][i]-1) for i in range(3)]
    legacy_counts.append(math.prod(max(hi[i]-lo[i]+1,0) for i in range(3)))
assert max(legacy_counts)==profile['hypothetical_legacy_largest_bounding_box_cells']==5624599
assert max(legacy_counts)*96==profile['hypothetical_legacy_four_coordinate_arrays_bytes']==539961504
assert same(result['tile_pruning_padding_mm'],179.2e-9)
assert budget=={'tile_edge':16,'coarse_batch':256,'max_tile_capsule_pairs':2000000,'max_cell_capsule_pairs':4000000,'max_sampled_cells':2000000,'wall_seconds':20.}
assert work=={'cell_capsule_pairs':1130496,'maximum_action_masks':3,'maximum_cell_geometry_batch':256,
              'maximum_coarse_geometry_batch':256,'maximum_geometry_batch':256,'maximum_tile_cells':4096,
              'reference_sample_calls':105,'reference_sampled_cells':21314,'tile_capsule_pairs':18432,
              'tiles_evaluated':105,'tiles_pruned':2967,'tiles_scanned':3072}
assert work['tiles_scanned']==math.prod(n//16 for n in result['grid_shape'])
assert work['tiles_pruned']+work['tiles_evaluated']==work['tiles_scanned'] and work['tile_capsule_pairs']==6*work['tiles_scanned']
assert work['cell_capsule_pairs']%4096==0 and work['cell_capsule_pairs']<budget['max_cell_capsule_pairs']
assert work['reference_sample_calls']==work['tiles_evaluated'] and work['reference_sampled_cells']==result['whole_tool']['touched_reference_cells']
assert set(result['per_action'])=={'first','repeated','crossing'} and result['per_action']['first']==result['per_action']['repeated']
rows={'shaft':result['shaft'],'tip':result['tip'],'whole_tool':result['whole_tool'],**result['per_action']}
for name,row in rows.items():
    touched,positive,unknown=[row[key] for key in ('touched_reference_cells','positive_reference_cells','unknown_reference_cells')]
    assert all(type(n) is int and 0<=n<=touched for n in (touched,positive,unknown)) and positive+unknown<=touched<=12582912
    outside=(any(outside_by_part.values()) if name=='whole_tool' else outside_by_part[name] if name in outside_by_part else outside_by_action[name])
    assert row['outside_reference_fov'] is outside
    assert row['annotated_positive_encounter'] is (True if positive else None if unknown or outside else False)
    assert row['annotation_coverage_complete_for_sweep'] is (not unknown and not outside)
    assert row['biological_vessel_free'] is None and row['clinical_injury_probability'] is None
    assert same(row['positive_cell_volume_upper_bound_mm3'],positive*.7**3) and same(row['unknown_in_grid_cell_volume_mm3'],unknown*.7**3)
for key in ('touched_reference_cells','positive_reference_cells','unknown_reference_cells'):
    assert max(result['shaft'][key],result['tip'][key])<=result['whole_tool'][key]<=result['shaft'][key]+result['tip'][key]
    assert max(result['per_action']['first'][key],result['per_action']['crossing'][key])<=result['whole_tool'][key]<=result['per_action']['first'][key]+result['per_action']['crossing'][key]
assert receipt['semantics']=={'process_peak_rss_bytes':profile['process_peak_rss_bytes'],'profile_seconds':profile['profile_seconds'],
                             'status':'generated_streaming_profile_complete','tracemalloc_peak_bytes':profile['tracemalloc_peak_bytes'],'work':work}
before=dict(OBSERVED)
for name,binding in before.items(): assert digest(raw(name))==binding['sha256']
assert OBSERVED==before
summary={'schema':'independent-saved-streaming-profile-audit-v1','status':'passed_saved_metadata_audit',
         'saved_only':True,'new_grid_evaluations':0,'new_child_or_model_or_native_runs':0,'source_and_evidence_file_count':len(before),
         'snapshots_equal':True,'output_bytes':actual_output_bytes,'receipt_sha256':digest(receipt_bytes),'profile_sha256':digest(profile_bytes),
         'profile_seconds':profile['profile_seconds'],'supervised_stage_seconds':stage['elapsed_seconds'],'recorded_lifecycle_seconds':receipt['lifecycle_seconds'],
         'sampled_group_peak_rss_bytes':stage['peak_sampled_process_group_rss_bytes'],'process_peak_rss_bytes':profile['process_peak_rss_bytes'],
         'process_peak_rss_MiB':profile['process_peak_rss_bytes']/1024**2,'traced_kernel_peak_bytes':profile['tracemalloc_peak_bytes'],
         'traced_kernel_peak_MiB':profile['tracemalloc_peak_bytes']/1024**2,'tile_pruning_fraction':work['tiles_pruned']/work['tiles_scanned'],
         'work':work,'whole_tool':result['whole_tool'],'hypothetical_legacy_arrays_bytes':max(legacy_counts)*96,
         'hypothetical_legacy_arrays_MiB':max(legacy_counts)*96/1024**2,
         'measured_baseline_or_speedup_available':False,'clinical_or_patient_admission':False}
(HERE/'source-and-evidence-snapshot.json').write_text(json.dumps(before,sort_keys=True,indent=2)+'\n')
(HERE/'audit-result.json').write_text(json.dumps(summary,sort_keys=True,indent=2)+'\n')
print(json.dumps(summary,sort_keys=True,indent=2))

"""Opt-in saved-fit metadata/matrix audit. Never opens tags or image arrays.

Run --audit-saved-fit after explicit ROOT release. No registration imports or
optimizer calls. Compressed image bytes are streamed only for SHA/MD5 fixity.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import re

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
DATA='outputs/poc-resect-case3-image-registration-v1/'
ARTIFACT='artifacts/resect-case3-registration-result-independent-v1/'
PRE='artifacts/resect-case3-image-registration-preparation-v1/'
RELEASE='artifacts/poc-resect-case3-image-registration-v1/'
PINS={
    DATA+'result.json':'ea8e57322989e6b5f580c67181201deead9235a4829315781bc7aad0f598b12b',
    DATA+'frozen-transform.json':'e6648339d69841985f8fa644ae5d69168422ce12ca5dadc4659b275a8b9cd8af',
    DATA+'candidate.json':'17a9599de644b9d7d4cd3fd9b356630adb3e38180496b9501f914520056949ef',
    DATA+'supervision.json':'e675d52669dfef2442e31d019e98c8d14eb705a470692510f0bc37a1cf2c9fe9',
    DATA+'started.json':'360b23b64d11d468a989340f1c825067632196ecfae999144badfc7b408ca0f5',
    DATA+'worker.log':'e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855',
    RELEASE+'root-fit-release.json':'bc196c57fb4701c161f50bcb4b3edae03b5988b864308a0c217e9c83ac8962e0',
    RELEASE+'unused-initial-root-fit-release.json':'e4bc86bdbf5127959eb9a090d70c77cfb4f9d2091753fa348f0f8061c8ec774d',
    PRE+'header-preflight-rss-v2.json':'ef9e0cd6029e869a6364de6c946f0284bc0a7ca1f700695a8bb75b52e60fd7ba',
    PRE+'prospective-protocol.json':'b1b27118ae79b6d6ab576c97298749ba5f126d8fac985788c147e42961e14bd7',
    'artifacts/resect-case3-registration-preflight-independent-v1/independent-qa.json':'280aff3692c0673a164c113bb16a908183a5d7c04350a8a2d0c087dabb8665c5',
    'scripts/poc_resect_case3_image_registration.py':'e17d755eeadb6a3d1fd365b7a3b4dc31f9e27c4f93afdb6045d8f6e27ef9612f',
}


def sha(path):
    assert Path(path).suffix!='.tag','tag bytes are outside this release'
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024**2),b''):digest.update(block)
    return digest.hexdigest()


def read(relative):
    path=ROOT/relative
    assert path.suffix=='.json' and path.stat().st_size<1024**2
    return json.loads(path.read_text())


def finite(value,minimum=0,maximum=math.inf):
    assert isinstance(value,(int,float)) and not isinstance(value,bool)
    assert math.isfinite(value) and minimum<=value<=maximum


def rigid_stats(forward,inverse):
    matrix=np.asarray(forward,dtype=float);reverse=np.asarray(inverse,dtype=float)
    for item in (matrix,reverse):
        assert item.shape==(4,4) and np.isfinite(item).all()
        np.testing.assert_allclose(item[3],[0,0,0,1],rtol=0,atol=1e-12)
        np.testing.assert_allclose(item[:3,:3].T@item[:3,:3],np.eye(3),rtol=0,atol=1e-5)
        assert abs(np.linalg.det(item[:3,:3])-1)<1e-5
    error=float(max(np.abs(matrix@reverse-np.eye(4)).max(),np.abs(reverse@matrix-np.eye(4)).max()))
    assert error<1e-9
    angle=float(np.degrees(np.arccos(np.clip((np.trace(matrix[:3,:3])-1)/2,-1,1))))
    return {'forward_rotation_determinant':float(np.linalg.det(matrix[:3,:3])),
        'forward_rotation_orthogonality_max_error':float(np.abs(matrix[:3,:3].T@matrix[:3,:3]-np.eye(3)).max()),
        'two_sided_inverse_max_error':error,'rotation_degrees_recomputed':angle}


def validate(record):
    c,f,r,s,b,release,started=(record[k] for k in ('candidate','frozen','result','supervision','preflight','release','started'))
    assert c['schema']=='resect-case3-image-registration-v1-candidate'
    assert f['schema']=='resect-case3-image-registration-v1-frozen' and r['schema']=='resect-case3-image-registration-v1-result'
    assert all(value['case_id']=='RESECT:Case3' for value in (c,f,r,b,release))
    assert b['role']=='TRAIN'
    assert r['status']==s['status']=='completed' and r['exit_code']==s['exit_code']==0
    assert r['evaluation_run'] is r['anatomical_alignment_accepted'] is f['anatomical_alignment_accepted'] is False
    assert f['evaluation_requires_separate_root_release'] is True
    assert r['frozen_transform_sha256']==PINS[DATA+'frozen-transform.json']
    assert f['candidate_sha256']==PINS[DATA+'candidate.json']
    assert r['supervision_sha256']==f['supervision_sha256']==PINS[DATA+'supervision.json']
    for value in (c,f,r,started):
        assert value['preflight_sha256']==PINS[PRE+'header-preflight-rss-v2.json']
        assert value['fit_release_sha256']==PINS[RELEASE+'root-fit-release.json']
    extras={'candidate_sha256','supervision_sha256','evaluation_requires_separate_root_release','anatomical_alignment_accepted'}
    assert {k:v for k,v in f.items() if k not in extras|{'schema'}}=={k:v for k,v in c.items() if k!='schema'}
    assert c['runner_sha256']==b['runner_sha256']==release['runner_sha256']==PINS['scripts/poc_resect_case3_image_registration.py']
    assert c['runtime']==b['runtime'] and c['sources_rechecked_after_each_decode'] is True
    assert c['source_fixity_after_fit']=={name:item['sha256'] for name,item in b['inputs'].items()}
    assert release['decision']=='release_one_fit' and release['registration_parameter_fitting_released'] is True
    assert all(release[k] is False for k in ('landmark_access_permitted','evaluation_released','clinical_or_anatomical_acceptance','neural_training','new_acquisition'))
    assert release['limits']['automatic_retry'] is False
    assert release['prepared_commit']=='fa44c9e529b17ca298f5bf0ea88d6ee6b33a5ba6'
    report=c['report'];config=b['configuration'];limits=b['limits']
    assert report['configuration']==started['configuration']==config and started['limits']==limits
    assert started['pid']==s['pid'] and s['worker_reaped'] is s['process_group_cleanup_attempted'] is True
    assert s['schema']=='resect-case3-image-registration-v1-supervision'
    assert s['memory_control']=='sampled_stop_enforcement' and s['os_hard_memory_limit'] is False
    assert s['samples']>=1 and isinstance(s['samples'],int)
    assert s['sampling_interval_seconds']==limits['memory_sampling_interval_seconds']==.2
    assert s['sampled_family_rss_stop_bytes']==limits['sampled_family_rss_stop_bytes']==release['limits']['sampled_family_rss_stop_bytes']
    finite(s['peak_family_rss_bytes_sampled'],maximum=limits['sampled_family_rss_stop_bytes'])
    finite(s['peak_worker_rss_bytes_sampled'],maximum=s['peak_family_rss_bytes_sampled'])
    finite(s['minimum_available_memory_bytes_sampled'],minimum=1)
    finite(s['elapsed_seconds'],maximum=limits['worker_wall_seconds'])
    finite(report['elapsed_registration_seconds'],maximum=min(config['optimizer_seconds'],s['elapsed_seconds']))
    assert report['fixed']=='before_resection_US' and report['moving']=='during_resection_US'
    assert report['sampling_direction']=='before-RAS output physical point -> during-RAS input physical point'
    assert report['identity_baseline_before_RAS_to_during_RAS']==np.eye(4).tolist()
    assert report['landmarks_used'] is report['registration_reviewed'] is report['anatomical_registration_accepted'] is report['metric_improvement_is_anatomical_acceptance'] is False
    assert report['classification']=='software_valid_candidate_anatomical_review_pending'
    stats=rigid_stats(report['before_RAS_mm_to_during_RAS_mm'],report['during_RAS_mm_to_before_RAS_mm'])
    assert abs(stats['rotation_degrees_recomputed']-report['rotation_degrees'])<1e-8
    assert report['simpleitk_version']==b['runtime']['SimpleITK']
    assert report['metric_support']==b['metric_support']
    finite(report['center_displacement_mm'])
    finite(report['native_fixed_positive_signal_support_voxels'],minimum=1,maximum=np.prod(b['headers']['before']['shape']))
    objective=report['matched_objective'];assert objective['metric']=='fixed-bin histogram NMI32, same paired voxels'
    finite(objective['before']);finite(objective['after']);finite(objective['overlap_retained_fraction'],maximum=1)
    finite(objective['voxels'],minimum=100)
    for window in report['normalization_windows'].values():
        assert len(window)==2;finite(window[0]);finite(window[1]);assert window[1]>window[0]
    flags=[]
    if objective['after']<objective['before']:flags.append('matched_NMI_decreased')
    if report['center_displacement_mm']>20 or stats['rotation_degrees_recomputed']>15:flags.append('large_pose_change')
    if objective['overlap_retained_fraction']<.9:flags.append('overlap_loss')
    assert report['flags']==flags
    history=report['optimizer_history'];expected_levels=list(range(len(config['shrink_levels'])))
    assert [item['level'] for item in history]==sorted(item['level'] for item in history)
    assert sorted(set(item['level'] for item in history))==expected_levels
    counts=[]
    for level in expected_levels:
        events=[item for item in history if item['level']==level]
        assert [item['iteration'] for item in events]==list(range(len(events)))
        assert 1<=len(events)<=config['iterations_per_level']
        for item in events:finite(item['sampled_negative_mattes_mi'],minimum=-math.inf)
        counts.append(len(events))
    stop=re.search(r'Step too small after (\d+) iterations',report['optimizer_stop_reason'])
    assert stop and int(stop.group(1))==counts[-1]
    return {'matrix_checks':stats,'optimizer_events_per_level':counts,'optimizer_events_total':len(history),
        'stop_reason':report['optimizer_stop_reason'],'flags_recomputed_from_recorded_metrics':flags,
        'image_objective_reported':objective,'center_displacement_mm_reported_only':report['center_displacement_mm'],
        'elapsed_registration_seconds_reported':report['elapsed_registration_seconds']}


def controls(record):
    names=[]
    def reject(name,edit,mirror=False):
        value=deepcopy(record);edit(value)
        if mirror:
            for key,item in value['candidate'].items():
                if key!='schema':value['frozen'][key]=deepcopy(item)
        try:validate(value)
        except (AssertionError,ValueError,TypeError,KeyError):names.append(name)
        else:raise AssertionError('accepted altered '+name)
    reject('wrong_frozen_hash',lambda v:v['result'].__setitem__('frozen_transform_sha256','0'*64))
    reject('wrong_release_hash',lambda v:v['candidate'].__setitem__('fit_release_sha256','0'*64),True)
    reject('failed_result',lambda v:v['result'].__setitem__('status','worker_failed'))
    reject('evaluation_claim',lambda v:v['result'].__setitem__('evaluation_run',True))
    reject('anatomical_claim',lambda v:v['result'].__setitem__('anatomical_alignment_accepted',True))
    reject('runtime_changed',lambda v:v['candidate']['runtime'].__setitem__('SimpleITK','different'),True)
    reject('source_changed',lambda v:v['candidate']['source_fixity_after_fit'].__setitem__('before','0'*64),True)
    reject('inverse_wrong',lambda v:v['candidate']['report'].__setitem__('during_RAS_mm_to_before_RAS_mm',np.eye(4).tolist()),True)
    reject('direction_wrong',lambda v:v['candidate']['report'].__setitem__('sampling_direction','reverse'),True)
    reject('identity_wrong',lambda v:v['candidate']['report']['identity_baseline_before_RAS_to_during_RAS'][0].__setitem__(3,1),True)
    for key in ('landmarks_used','registration_reviewed','anatomical_registration_accepted','metric_improvement_is_anatomical_acceptance'):
        reject(key,lambda v,k=key:v['candidate']['report'].__setitem__(k,True),True)
    for key,value in [('peak_family_rss_bytes_sampled',5*1024**3),('elapsed_seconds',241),('samples',0),('worker_reaped',False),('exit_code',1)]:
        reject('supervision_'+key,lambda v,k=key,x=value:v['supervision'].__setitem__(k,x))
    reject('optimizer_time',lambda v:v['candidate']['report'].__setitem__('elapsed_registration_seconds',181),True)
    reject('optimizer_duplicate_iteration',lambda v:v['candidate']['report']['optimizer_history'][1].__setitem__('iteration',0),True)
    reject('flag_inconsistent',lambda v:v['candidate']['report'].__setitem__('flags',['overlap_loss']),True)
    reject('release_allows_tags',lambda v:v['release'].__setitem__('landmark_access_permitted',True))
    # Independent generated arithmetic with known 90-degree RAS rotation.
    matrix=np.array([[0.,-1.,0.,3.],[1.,0.,0.,-2.],[0.,0.,1.,1.],[0.,0.,0.,1.]])
    points=np.array([[1.,0.,0.],[0.,2.,3.],[-2.,1.,4.]])
    expected=np.array([[3.,-1.,1.],[1.,-2.,4.],[2.,-4.,5.]])
    moved=points@matrix[:3,:3].T+matrix[:3,3]
    np.testing.assert_array_equal(moved,expected)
    stats=rigid_stats(matrix,np.linalg.inv(matrix));assert stats['rotation_degrees_recomputed']==90
    np.testing.assert_allclose((moved-matrix[:3,3])@matrix[:3,:3],points,rtol=0,atol=1e-12)
    errors=np.sqrt(np.sum((moved-(expected+np.array([[3.,4.,0.],[0.,0.,0.],[0.,0.,12.]])))**2,axis=1))
    np.testing.assert_array_equal(errors,[5.,0.,12.])
    assert errors.mean()==17/3 and np.median(errors)==5 and math.sqrt(np.mean(errors**2))==math.sqrt(169/3)
    names.append('generated_RAS_direction_inverse_and_all_point_distance_arithmetic')
    return names


def audit():
    pins=dict(PINS)
    for path,pin in pins.items():assert sha(ROOT/path)==pin,path
    record={name:read(path) for name,path in {'candidate':DATA+'candidate.json','frozen':DATA+'frozen-transform.json',
        'result':DATA+'result.json','supervision':DATA+'supervision.json','started':DATA+'started.json',
        'preflight':PRE+'header-preflight-rss-v2.json','release':RELEASE+'root-fit-release.json'}.items()}
    facts=validate(record);checks=controls(record)
    release=record['release'];old=read(RELEASE+'unused-initial-root-fit-release.json')
    assert set(old)-set(release)=={'acquisition_or_model_fitting'}
    assert set(release)-set(old)=={'new_acquisition','neural_training','registration_parameter_fitting_released','unused_initial_release'}
    assert all(old[k]==release[k] for k in set(old)&set(release))
    assert release['unused_initial_release']['sha256']==pins[RELEASE+'unused-initial-root-fit-release.json']
    assert release['unused_initial_release']['consumed_by_worker'] is False
    assert record['started']['fit_release_sha256']!=pins[RELEASE+'unused-initial-root-fit-release.json']
    source_checks=[]
    for name,entry in record['preflight']['inputs'].items():
        path=ROOT/entry['path'];digest=hashlib.sha256();md5=hashlib.md5()
        assert path.suffix=='.gz' and path.stat().st_size==entry['bytes']
        with path.open('rb') as stream:
            for block in iter(lambda:stream.read(1024**2),b''):digest.update(block);md5.update(block)
        assert digest.hexdigest()==entry['sha256'] and md5.hexdigest()==entry['expected_md5']
        source_checks.append({'phase':name,'sha256':digest.hexdigest(),'md5':md5.hexdigest(),'compressed_bytes':entry['bytes'],
            'decoded':False,'post_fit_source_byte_fixity_verified':True})
    for path,pin in record['preflight']['pins'].items():assert sha(ROOT/path)==pin;pins[path]=pin
    runtime={}
    for name,item in record['preflight']['runtime']['binary_and_wrapper_bindings'].items():
        assert sha(item['path'])==item['sha256'];runtime[name]=item['sha256']
    protocol=read(PRE+'prospective-protocol.json')
    assert protocol['configuration']==record['candidate']['report']['configuration']
    assert protocol['inputs']==record['preflight']['inputs']
    for path,pin in pins.items():assert sha(ROOT/path)==pin,path
    result={'schema':'independent-resect-case3-saved-fit-audit-v1','status':'pass_saved_software_candidate_unaccepted',
        'auditor_sha256':sha(__file__),'pinned_metadata_unchanged_before_after':pins,
        'source_byte_continuity':source_checks,'runtime_binary_hashes_rechecked':runtime,**facts,
        'supervision':record['supervision'],'independent_generated_and_tamper_controls':checks,
        'attempt_accounting':{'recorded_started_workers':1,'same_started_and_supervised_pid':True,
            'successful_frozen_candidates':1,'unused_initial_release_retained':True,'unused_initial_release_consumed':False,
            'scientific_settings_changed_between_release_records':False,'evaluation_runs':0},
        'source_arrays_decoded':False,'actual_tag_coordinates_read':False,'registration_optimizer_reexecuted':False,
        'anatomical_alignment_accepted':False,'full_transform_in_tracked_report':False,
        'limitations':['The image objective, support count and support-center displacement are reported values; without image arrays they are not independently recalculated here.',
            'Rotation, inverse, identity, event counts, resource bounds, hashes and lineage are independently checked; these do not establish anatomical accuracy.',
            'Actual all-point tag parsing and TRE evaluation remain separately unreleased. Generated distance controls do not evaluate Case3.',
            'Recorded one-worker accounting is based on the supplied release/start/supervision chain, not a claim of exhaustive system-wide process history.',
            'Sampled worker-family RSS is not an OS hard cap or exact transient peak.']}
    target=ROOT/ARTIFACT/'independent-fit-qa.json';target.parent.mkdir(parents=True,exist_ok=True)
    with target.open('x') as stream:json.dump(result,stream,indent=2);stream.write('\n')
    print(json.dumps({'status':result['status'],'checks':len(checks),'report':str(target.relative_to(ROOT)),
                      'sha256':sha(target),'optimizer_events':facts['optimizer_events_per_level']},indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--audit-saved-fit',action='store_true',required=True)
    parser.parse_args();audit()

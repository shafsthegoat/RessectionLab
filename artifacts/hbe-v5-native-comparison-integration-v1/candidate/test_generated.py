"""Synthetic metadata/contracts only; no native output or measured responses."""
from copy import deepcopy
from pathlib import Path
import ast
import hashlib
import importlib.util
import json
import os
import sys
import tempfile

import pytest
ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent
import scripts
scripts.__path__ = [str(HERE/'stage/scripts'), *scripts.__path__]
from scripts import mechanics_hbe_v5_comparison as c
from scripts import mechanics_hbe_v5_native_comparison as n
from scripts import mechanics_hbe_branch_calibration_v5 as v5
from scripts import mechanics_hbe_v5_source_bindings as sources
sys.path.insert(0,str(ROOT/'tests'))
import test_mechanics_hbe_v5_comparison as old_tests
study,prior=v5.validate_preparation(ROOT)
source_map=json.loads((ROOT/sources.BINDING_PATH).read_text())

@pytest.fixture(autouse=True)
def no_forbidden_actions(monkeypatch):
    import subprocess, zipfile
    monkeypatch.setattr(subprocess,'Popen',lambda *a,**k:pytest.fail('No subprocess'))
    monkeypatch.setattr(zipfile,'ZipFile',lambda *a,**k:pytest.fail('No measured archive'))
    old=Path.open
    def check(path,*a,**k):
        resolved=path.resolve()
        if resolved.is_relative_to(ROOT/'data/mechanics') or resolved.is_relative_to(ROOT/'outputs/mechanics'):
            pytest.fail('No original numerical/measured payload access')
        return old(path,*a,**k)
    monkeypatch.setattr(Path,'open',check)


def fixture(index=5):
    run_id=n.remaining.ORDER[index];row=deepcopy(old_tests.generated_rows()[run_id])
    src=source_map['source_decks'][source_map['run_source_keys'][run_id]];mesh=source_map['meshes'][src['mesh']]
    receipt_sha='1'*64;release_sha='2'*64;adapted='3'*64;readout_sha='4'*64;commit='5'*40
    if index==0:
        receipt_sha=n.remaining.N8_SHA;release_sha=n.remaining.N8_RELEASE_SHA
        adapted=n.remaining.N8_DECK_SHA;readout_sha=receipt_sha;commit=n.remaining.N8_SOURCE_COMMIT
    if index==1:receipt_sha=n.remaining.N12_FAILED_RECEIPT_SHA
    records={name:{'path':'outputs/mechanics/generated-fixture/'+name,'sha256':hashlib.sha256(name.encode()).hexdigest(),'bytes':10}
             for name in ('specimen.feb','nodes.log','elements.log','solver.log','console.txt','readout.json')}
    records['specimen.feb']['sha256']=adapted;records['readout.json']['sha256']=readout_sha
    primitives={'source_deck':{k:src[k] for k in ('path','sha256')},'mesh':mesh,
                **{key:{field:records[name][field] for field in ('path','sha256')}
                   for key,name in (('adapted_deck','specimen.feb'),('nodes','nodes.log'),('elements','elements.log'),('solver','solver.log'))}}
    row['provenance'].update(adapted_deck_sha256=adapted,source_binding_checked=True,
        output_origin='unverified_saved_stream',native_output_observed=None,generated_fixture_only=None,
        primitive_bindings=primitives,reconstruction_provenance='reflected_native_half_not_native_full'
            if row['representation']=='reconstructed_full' else 'full_native')
    receipt={'run_id':run_id,'status':'passed_numerical_software_only','release_sha256':release_sha,
             'native_calls_attempted':1,'no_retry':True,'source_commit':commit,'adapted_deck_sha256':adapted,
             'old_source_deck_sha256':src['sha256'],'native_mesh_sha256':mesh['sha256'],
             'saved_numerical_readout':{'readout_sha256':readout_sha},'output_bindings':records,
             'native_output_bindings':{key:records[key] for key in ('specimen.feb','nodes.log','elements.log','solver.log','console.txt')},
             'readout_token_sha256':'6'*64}
    release={'source_commit':commit,'adapted_deck_sha256':adapted}
    work={'run_id':run_id,'source_commit':commit,'release_sha256':release_sha,'readout_token_sha256':'6'*64,'bindings':primitives}
    exact=None;exception=None
    if index==0:
        row['provenance'].update(output_origin='one_supervised_native_FEBio_attempt',native_output_observed=True,generated_fixture_only=False)
        receipt['saved_numerical_readout']=row;work=None
    if index==1:
        receipt['status']='failed_or_incomplete'
        exact={'original_receipt_sha256':receipt_sha,'supplement_receipt_sha256':n.supplement.SUPPLEMENT_SHA,
               'historical_guard_gap':n.supplement.GAP,'readout':row,'original_file_bindings':records}
        exception={'original_status':'failed_or_incomplete','supplement_receipt_sha256':n.supplement.SUPPLEMENT_SHA,
                   'historical_guard_gap':n.supplement.GAP}
    independent={'run_id':run_id,'native_receipt_sha256':receipt_sha,'release_sha256':release_sha,
        'readout_sha256':readout_sha,'source_deck_sha256':src['sha256'],'adapted_deck_sha256':adapted,
        'native_mesh_sha256':mesh['sha256'],'exception_policy':exception,'evidence_kind':'archived_prose_exact_replay',
        'comparison_scope':'all_non_provenance_fields' if index==0 else 'entire_readout_json',
        'supporting_passage':'All numerical fields reproduced exactly from saved streams.',
        'review_binding':{'path':'build/generated-review.md','sha256':'7'*64}}
    return {'readout':row,'receipt':receipt,'receipt_sha':receipt_sha,'release':release,'release_sha':release_sha,
            'work_order':work,'independent':independent,'source_map':source_map,'readout_sha':readout_sha,'exact_supplement':exact}


def validate(data,index=5):return n.validate_native_row(study,prior,n.remaining.ORDER[index],**data)


@pytest.mark.parametrize('index',range(12))
def test_every_declared_native_shape_admits_compact_generated_contract_without_relabel(index):
    data=fixture(index);before=deepcopy(data)
    view,passed=validate(data,index)
    assert passed and view['frame_count']==data['readout']['frame_count']
    assert data==before
    # Generated-only public admission must continue to reject saved-stream origin.
    source=data['readout']['provenance']['source_deck_sha256']
    with pytest.raises(ValueError):c._view(study,prior,n.remaining.ORDER[index],data['readout'],source)


@pytest.mark.parametrize('field',['run_id','source_deck_sha256','adapted_deck_sha256','readout_sha256','native_receipt_sha256','release_sha256','native_mesh_sha256','comparison_scope'])
def test_independent_mapping_swaps_refuse(field):
    data=fixture();data['independent'][field]='wrong'
    with pytest.raises(ValueError):validate(data)


@pytest.mark.parametrize('mutation',['generated_flag','native_flag','source','adapted','primitive','work_token','release','status','measured','representation','frame_count','schedule','nonfinite','probe_shape','flag_contradiction'])
def test_native_metadata_and_numeric_tamper_refuses(mutation):
    data=fixture();row=data['readout']
    if mutation=='generated_flag':row['provenance']['generated_fixture_only']=True
    elif mutation=='native_flag':row['provenance']['native_output_observed']=True
    elif mutation=='source':row['provenance']['source_deck_sha256']='f'*64
    elif mutation=='adapted':row['provenance']['adapted_deck_sha256']='f'*64
    elif mutation=='primitive':data['work_order']=deepcopy(data['work_order']);data['work_order']['bindings']['nodes']['sha256']='f'*64
    elif mutation=='work_token':data['work_order']['readout_token_sha256']='f'*64
    elif mutation=='release':data['release']['source_commit']='f'*40
    elif mutation=='status':data['receipt']['status']='failed_or_incomplete'
    elif mutation=='measured':row['provenance']['measured_response_accessed']=True
    elif mutation=='representation':row['representation']='full_native_fixture'
    elif mutation=='frame_count':row['frame_count']=60
    elif mutation=='schedule':row['load_coordinate_full_m'][-1]+=1e-15
    elif mutation=='nonfinite':row['applied_force_N'][-1]=float('nan')
    elif mutation=='probe_shape':row['probe_displacements_m'][-1].pop()
    else:row['numerical_passed']=False
    with pytest.raises(ValueError):validate(data)


@pytest.mark.parametrize('change',['status','receipt','supplement','gap'])
def test_original_n12_failure_is_not_reclassified_or_generalized(change):
    data=fixture(1)
    if change=='status':data['receipt']['status']='passed_numerical_software_only'
    elif change=='receipt':data['receipt_sha']='f'*64
    elif change=='supplement':data['exact_supplement']['supplement_receipt_sha256']='f'*64
    else:data['exact_supplement']['historical_guard_gap']='repaired retroactively'
    with pytest.raises(ValueError):validate(data,1)


def test_full_native_numeric_diagnostics_equal_generated_math_without_provenance_relabel():
    views,individual={},{}
    for i,run_id in enumerate(n.remaining.ORDER):views[run_id],individual[run_id]=validate(fixture(i),i)
    numeric=c._compare_views(prior,list(n.remaining.ORDER),views,individual,c._historical_n32(ROOT,prior))
    generated=c.compare_generated_rows(ROOT,old_tests.generated_rows())
    assert numeric['diagnostic_screen_passed'] is generated.pop('generated_diagnostic_screen_passed')
    assert numeric['individual_stream_pass']==generated.pop('individual_generated_stream_pass')
    for key,value in numeric.items():
        if key not in ('diagnostic_screen_passed','individual_stream_pass'):assert value==generated[key],key


def write_bound(root,path,value):
    raw=json.dumps(value,allow_nan=False).encode() if not isinstance(value,bytes) else value
    target=root/path;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(raw)
    return {'path':path,'sha256':hashlib.sha256(raw).hexdigest()}


def test_small_reader_refuses_tamper_duplicate_traversal_fifo_and_oversize(tmp_path):
    binding=write_bound(tmp_path,'build/ok.json',{'safe':1});assert n._bound_json(tmp_path,binding)=={'safe':1}
    for raw in (b'{"a":1,"a":2}',b'{"a":NaN}'):
        malformed=write_bound(tmp_path,'build/bad.json',raw)
        with pytest.raises(ValueError):n._bound_json(tmp_path,malformed)
    with pytest.raises(ValueError):n._bound_json(tmp_path,{**binding,'sha256':'a'*64})
    with pytest.raises(ValueError):n._bound_json(tmp_path,{**binding,'path':'build/../outside.json'})
    with pytest.raises(ValueError):n._bound_json(tmp_path,binding,maximum=1)
    link=tmp_path/'build/link.json';link.symlink_to(tmp_path/'build/ok.json')
    with pytest.raises(ValueError):n._bound_json(tmp_path,{**binding,'path':'build/link.json'})
    fifo=tmp_path/'build/fifo.json';os.mkfifo(fifo)
    with pytest.raises(ValueError):n._bound_json(tmp_path,{**binding,'path':'build/fifo.json'})


def test_existing_prose_authenticates_without_new_review_certificate(tmp_path):
    mapping=fixture()['independent'];binding=write_bound(tmp_path,'build/existing-report.md',mapping['supporting_passage'].encode())
    mapping['review_binding']=binding;n._authenticate_review(tmp_path,mapping)
    mapping['supporting_passage']='This statement never appeared in the archived review.'
    with pytest.raises(ValueError):n._authenticate_review(tmp_path,mapping)


def test_missing_row_refuses_before_full_raw_chain_or_any_release(tmp_path,monkeypatch):
    manifest={'schema':'hbe-v5-native-comparison-inputs-v1','ordered_run_ids':list(n.remaining.ORDER),'rows':{}}
    binding=write_bound(tmp_path,'build/incomplete.json',manifest)
    monkeypatch.setattr(n.remaining,'validate_prior_chain',lambda *a,**k:pytest.fail('No raw read for incomplete input'))
    with pytest.raises(ValueError,match='exact_twelve_native_rows_required'):n.compare_native_rows(tmp_path,binding)


def extension_fixture():
    # Saved compact receipt metadata only; no numerical readout or raw streams.
    folder=ROOT/'artifacts/hbe-v5-tension-n12-nocache-v2-result-v1'
    side=json.loads((folder/'v2-sidecar-receipt.json').read_text())
    envelope=json.loads((folder/'outer-envelope.json').read_text())
    native=json.loads((folder/'native-receipt.json').read_text())
    descriptor={'schema':'hbe-v5-nocache-policy-sidecar-v2',
                'status':'passed_numerical_software_only_with_declared_io_policy_v2',
                'envelope':{'sha256':'2fc24ff3c33923ba33e16ae0842ceb97c1d89b6ede74b2b0e733b0127f74feb7'}}
    kwargs={'descriptor':descriptor,'native_sha':'67f7b82dca721d6d36b16e80e77c5d126b0ea9f9f510f8c2dd54b64716206a8e',
            'closed_native_bytes':49873201,'prior_surcharge_seconds':0.,'prior_surcharge_bytes':0}
    return side,envelope,native,kwargs


def test_existing_v2_metadata_resource_delta_matches_reviewed_value():
    side,envelope,native,kwargs=extension_fixture()
    value=n.extension_charge(side,envelope,native,**kwargs)
    assert value['incremental_prep_seconds']==pytest.approx(1.1413705407176167)
    assert value['incremental_output_bytes']==1048576


@pytest.mark.parametrize('mutation',['native','source','cleanup','host','hints','prep','bytes','ledger','double_charge','missing_prior_surcharge'])
def test_policy_metadata_tamper_refuses_without_new_native_run(mutation):
    side,envelope,native,kwargs=extension_fixture()
    if mutation=='native':side['native_receipt_sha256']='f'*64
    elif mutation=='source':side['extension_sources_after']['source_hashes']={}
    elif mutation=='cleanup':side['owned_stage_cleanup']['native']['direct_child_reaped']=False
    elif mutation=='host':side['host_before_validation']['available_percent']=53
    elif mutation=='hints':side['final_hint_audit']['eligible_bytes']-=1
    elif mutation=='prep':side['extension_resource_charge']['launcher_inclusive_prep_seconds_with_finalization_reserve']=151.
    elif mutation=='bytes':side['extension_resource_charge']['charged_current_output_bytes']-=1
    elif mutation=='ledger':side['extension_resource_charge']['aggregate_with_extension']['aggregate_prep_wall_seconds']-=1
    elif mutation=='double_charge':side['extension_resource_charge']['aggregate_with_extension']['aggregate_prep_wall_seconds']+=1
    else:kwargs['prior_surcharge_seconds']=1.
    with pytest.raises(ValueError):n.extension_charge(side,envelope,native,**kwargs)


def test_following_continuation_carries_prior_surcharge_exactly_once():
    side,envelope,native,kwargs=extension_fixture()
    previous=deepcopy(side['old_validation_identity']['previous'])
    previous['prep_seconds']+=2.;previous['combined_wall_seconds']+=2.
    previous['output_bytes']+=1048576;previous['combined_output_bytes']+=1048576
    side['charged_previous']=previous
    charge=side['extension_resource_charge']
    charge['aggregate_with_extension']=n.remaining._check_aggregate(previous,native['native_stage']['elapsed_seconds'],
        native['readout_stage']['elapsed_seconds'],charge['launcher_inclusive_prep_seconds_with_finalization_reserve'],
        charge['charged_current_output_bytes'],1)
    kwargs.update(prior_surcharge_seconds=2.,prior_surcharge_bytes=1048576)
    result=n.extension_charge(side,envelope,native,**kwargs)
    assert result['incremental_output_bytes']==1048576
    assert result['incremental_prep_seconds']==pytest.approx(1.1413705407176167)
    side['charged_previous']['prep_seconds']+=2.
    with pytest.raises(ValueError):n.extension_charge(side,envelope,native,**kwargs)


def test_sidecar_reservation_is_separate_from_native_row_cap_and_charged_to_aggregate():
    side,envelope,native,kwargs=extension_fixture()
    closed=native['caps']['active_output_bytes']-512*1024
    kwargs['closed_native_bytes']=closed
    charge=side['extension_resource_charge'];charge['native_closed_output_bytes_at_check']=closed
    charge['charged_current_output_bytes']=closed+charge['sidecar_reserved_output_bytes']
    charge['aggregate_with_extension']=n.remaining._check_aggregate(side['old_validation_identity']['previous'],
        native['native_stage']['elapsed_seconds'],native['readout_stage']['elapsed_seconds'],
        charge['launcher_inclusive_prep_seconds_with_finalization_reserve'],charge['charged_current_output_bytes'],1)
    assert n.extension_charge(side,envelope,native,**kwargs)['incremental_output_bytes']==1048576
    native['caps']['active_output_bytes']=closed-1
    with pytest.raises(ValueError,match='extension_charge_identity'):
        n.extension_charge(side,envelope,native,**kwargs)


def orchestration_fixture(tmp_path,monkeypatch):
    """Real generated JSON files, frozen-ID constants scoped to synthetic values.

    Only raw/Git validators, preparation location and executing-origin location
    are replaced. Row contracts, byte ledger, source map, math, sidecar accounting
    and both-pass orchestration execute. No original outputs or native program.
    """
    events=[];data=[fixture(i) for i in range(12)]
    manifest={'schema':'hbe-v5-native-comparison-inputs-v1','ordered_run_ids':list(n.remaining.ORDER),
        'rows':{},'native_receipts':[],'measured_response_access':False,'patient_data_access':False,
        'native_execution_release':False,'comparison_source_bindings':{'scripts/generated.py':'0'*64},
        'backend_profile':{},'runtime_identity':{}}
    srcbind=write_bound(tmp_path,'build/source-map.json',source_map)
    monkeypatch.setattr(n.sources,'BINDING_PATH',srcbind['path']);monkeypatch.setattr(n.sources,'BINDING_SHA256',srcbind['sha256'])
    previous={'sha256':[],'native_seconds':0.,'readout_seconds':0.,'prep_seconds':0.,'output_bytes':0,
              'combined_wall_seconds':0.,'combined_output_bytes':0}
    for i,d in enumerate(data):
        run=n.remaining.ORDER[i];r=d['receipt'];base=f'build/rows/{i}'
        release=write_bound(tmp_path,base+'/release.json',d['release']);d['release_sha']=release['sha256']
        r['release_sha256']=release['sha256']
        if i==0:monkeypatch.setattr(n.remaining,'N8_RELEASE_SHA',release['sha256'])
        rowbind=None
        if i:
            rowbind=write_bound(tmp_path,base+'/readout.json',d['readout']);d['readout_sha']=rowbind['sha256']
            r['output_bindings']['readout.json'].update(sha256=rowbind['sha256'],bytes=(tmp_path/rowbind['path']).stat().st_size)
            r['saved_numerical_readout']['readout_sha256']=rowbind['sha256']
            d['work_order']['release_sha256']=release['sha256']
        r.update(prep_elapsed_seconds=2.,native_stage={'elapsed_seconds':3.},readout_stage={'elapsed_seconds':1.},
                 caps=n.remaining.caps(i or 1),prior_receipt_sha256=list(previous['sha256']))
        for key,field in (('native_seconds','prior_native_wall_seconds'),('readout_seconds','prior_readout_wall_seconds'),
            ('prep_seconds','prior_prep_wall_seconds'),('output_bytes','prior_active_output_bytes'),
            ('combined_wall_seconds','prior_combined_wall_seconds'),('combined_output_bytes','prior_combined_output_bytes')):
            r[field]=previous[key]
        rec=write_bound(tmp_path,base+'/receipt.json',r);d['receipt_sha']=rec['sha256']
        if i==0:
            monkeypatch.setattr(n.remaining,'N8_SHA',rec['sha256']);d['readout_sha']=rec['sha256']
        if i==1:monkeypatch.setattr(n.remaining,'N12_FAILED_RECEIPT_SHA',rec['sha256'])
        manifest['native_receipts'].append({'run_id':run,**rec})
        entry={'release':release,'policy_extension':None}
        if i:entry.update(readout=rowbind,work_order=write_bound(tmp_path,base+'/work-order.json',d['work_order']))
        manifest['rows'][run]=entry
        previous['sha256'].append(rec['sha256'])
        previous['native_seconds']+=3.;previous['readout_seconds']+=1.;previous['prep_seconds']+=2.
        previous['combined_wall_seconds']+=6.
        closed=(tmp_path/rec['path']).stat().st_size+sum(x['bytes'] for x in r['output_bindings'].values())
        previous['output_bytes']+=closed;previous['combined_output_bytes']+=closed
    sup=write_bound(tmp_path,'build/supplement/receipt.json',{'original_file_bindings':data[1]['receipt']['output_bindings']})
    monkeypatch.setattr(n.supplement,'SUPPLEMENT_SHA',sup['sha256']);monkeypatch.setattr(n.supplement.replay,'OUTPUT','build/supplement')
    exact={**data[1]['exact_supplement'],'original_receipt_sha256':data[1]['receipt_sha'],
        'supplement_receipt_sha256':sup['sha256'],'original_accounting':{'native_calls':1},'supplement_overhead':{'native_calls':0}}
    for i,d in enumerate(data):
        mapping=d['independent'];mapping.update(native_receipt_sha256=d['receipt_sha'],release_sha256=d['release_sha'],readout_sha256=d['readout_sha'])
        if i==1:mapping['exception_policy']['supplement_receipt_sha256']=sup['sha256']
        mapping['review_binding']=write_bound(tmp_path,f'build/reviews/{i}.md',mapping['supporting_passage'].encode())
        manifest['rows'][n.remaining.ORDER[i]]['independent_mapping']=mapping
    cumulative_time,cumulative_bytes=0.,0
    for i in range(8,12):
        run=n.remaining.ORDER[i];native=data[i]['receipt'];entry=manifest['rows'][run]
        side,outer,_,_=extension_fixture();side['source_commit']=outer['source_commit']=native['source_commit']
        outer['inner_release']=entry['release'];side['inner_release_sha256']=native['release_sha256']
        side['native_receipt_sha256']=data[i]['receipt_sha']
        for field in ('extension_sources_before','extension_sources_after'):side[field]['source_commit']=native['source_commit']
        prev={'sha256':native['prior_receipt_sha256'],'native_calls':i,'supplement_readout_seconds':0.,'supplement_prep_seconds':0.,'supplement_replay_calls':1,'supplement_output_bytes':0}
        for key,field in (('native_seconds','prior_native_wall_seconds'),('readout_seconds','prior_readout_wall_seconds'),
            ('prep_seconds','prior_prep_wall_seconds'),('output_bytes','prior_active_output_bytes'),
            ('combined_wall_seconds','prior_combined_wall_seconds'),('combined_output_bytes','prior_combined_output_bytes')):prev[key]=native[field]
        side['old_validation_identity']['previous']=deepcopy(prev)
        for key in ('prep_seconds','combined_wall_seconds'):prev[key]+=cumulative_time
        for key in ('output_bytes','combined_output_bytes'):prev[key]+=cumulative_bytes
        if i>8:side['charged_previous']=prev
        closed=(tmp_path/manifest['native_receipts'][i]['path']).stat().st_size+sum(x['bytes'] for x in native['output_bindings'].values())
        charge=side['extension_resource_charge'];charge.update(launcher_inclusive_prep_seconds_with_finalization_reserve=2.5,
            native_closed_output_bytes_at_check=closed,charged_current_output_bytes=closed+1048576,native_stage_seconds=3.,readout_stage_seconds=1.,launcher_elapsed_seconds_at_check=5.5)
        side['launcher_elapsed_seconds_terminal']=5.6
        charge['aggregate_with_extension']=n.remaining._check_aggregate(prev,3.,1.,2.5,closed+1048576,1)
        outerbind=write_bound(tmp_path,f'build/policy/{i}/envelope.json',outer);side['envelope_sha256']=outerbind['sha256']
        sidebind=write_bound(tmp_path,f'build/policy/{i}/sidecar.json',side)
        entry['policy_extension']={'schema':side['schema'],'status':side['status'],'sidecar':sidebind,'envelope':outerbind}
        if i==8:monkeypatch.setattr(n,'V2_SIDECAR_SHA',sidebind['sha256'])
        cumulative_time+=.5;cumulative_bytes+=1048576
    hist=c._historical_n32(ROOT,prior)
    monkeypatch.setattr(n.v5,'validate_preparation',lambda root:(study,prior))
    monkeypatch.setattr(n.comparison,'_historical_n32',lambda root,prior:hist)
    def chain(*a,**kw):events.append('chain');assert a[1]==12;return deepcopy(previous)
    def verify(**kw):events.append('supplement');return deepcopy(exact)
    def source(*a,**kw):events.append('sources');return {'scripts.generated':'scripts/generated.py'}
    monkeypatch.setattr(n.remaining,'validate_prior_chain',chain)
    monkeypatch.setattr(n.supplement,'verify_exact',verify)
    monkeypatch.setattr(n,'_verify_comparison_sources',source)
    monkeypatch.setattr(n.supplement.replay,'committed_source',lambda *a,**kw:events.append('historical-source'))
    binding=write_bound(tmp_path,'build/native-inputs.json',manifest)
    return manifest,binding,events


def test_full_generated_public_orchestration_compares_without_raw_io_or_permission(tmp_path,monkeypatch):
    manifest,binding,events=orchestration_fixture(tmp_path,monkeypatch)
    result=n.compare_native_rows(tmp_path,binding)
    assert result['native_output_admitted'] and result['numerical_qualification_passed']
    assert result['native_execution_released'] is False and result['calibration_released'] is False
    assert result['physical_validation_pass'] is None and result['measured_response_accessed'] is False
    assert result['extension_policy_accounting']['incremental_prep_seconds']==2.
    assert result['extension_policy_accounting']['incremental_output_bytes']==4*1048576
    assert events.count('chain')==events.count('supplement')==events.count('sources')==2
    assert events.count('historical-source')==4 and result['small_evidence_rechecked']>=65
    assert result['exact_N12_exception']['historical_guard_gap']==n.supplement.GAP


@pytest.mark.parametrize('target',['manifest','readout','work_order','review','sidecar','envelope','supplement','source_map'])
def test_late_bound_metadata_mutation_refuses_full_result(tmp_path,monkeypatch,target):
    manifest,binding,events=orchestration_fixture(tmp_path,monkeypatch)
    entry=manifest['rows'][n.remaining.ORDER[8]]
    selected={'manifest':binding,'readout':entry['readout'],'work_order':entry['work_order'],
        'review':entry['independent_mapping']['review_binding'],'sidecar':entry['policy_extension']['sidecar'],
        'envelope':entry['policy_extension']['envelope'],'supplement':{'path':'build/supplement/receipt.json'},
        'source_map':{'path':'build/source-map.json'}}[target]
    compare=n.comparison._compare_views
    def mutate(*a,**kw):
        result=compare(*a,**kw);path=tmp_path/selected['path'];path.write_bytes(path.read_bytes()+b' ');return result
    monkeypatch.setattr(n.comparison,'_compare_views',mutate)
    with pytest.raises(ValueError,match='bound_metadata_bytes_changed'):n.compare_native_rows(tmp_path,binding)


def test_source_inventory_guards_origin_missing_and_late_bytes(tmp_path,monkeypatch):
    from types import SimpleNamespace
    names=(n.__name__,c.__name__,v5.__name__,sources.__name__,n.remaining.__name__,n.supplement.__name__,'scripts.transitive_math')
    modules={};bindings={}
    for name in names:
        path=name.replace('.','/')+'.py';target=tmp_path/path;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(b'# generated\n')
        bindings[path]=hashlib.sha256(target.read_bytes()).hexdigest();modules[name]=SimpleNamespace(__file__=str(target))
    monkeypatch.setattr(n.sys,'modules',modules)
    assert len(n._verify_comparison_sources(tmp_path,bindings))==len(names)
    shadow=modules[n.__name__].__file__;modules[n.__name__].__file__='/outside/shadow.py'
    with pytest.raises(ValueError,match='shadow_project_module'):n._verify_comparison_sources(tmp_path,bindings)
    modules[n.__name__].__file__=shadow
    missing=bindings.copy();missing.pop('scripts/transitive_math.py')
    with pytest.raises(ValueError,match='unbound_loaded_project_source'):n._verify_comparison_sources(tmp_path,missing)
    (tmp_path/'scripts/transitive_math.py').write_bytes(b'# changed\n')
    with pytest.raises(ValueError,match='bound_metadata_bytes_changed'):n._verify_comparison_sources(tmp_path,bindings)


@pytest.mark.parametrize('mutation',['elapsed','terminal','prep_limit','native_calls','readout_calls'])
def test_extension_recorded_clock_and_call_contradictions_refuse(mutation):
    side,envelope,native,kwargs=extension_fixture();charge=side['extension_resource_charge']
    if mutation=='elapsed':charge['launcher_elapsed_seconds_at_check']+=10.
    elif mutation=='terminal':side['launcher_elapsed_seconds_terminal']+=10.
    elif mutation=='prep_limit':charge['per_row_prep_wall_seconds']+=1
    elif mutation=='native_calls':side['native_calls_attempted']=2
    else:side['hbe_readout_calls_attempted']=0
    with pytest.raises(ValueError):n.extension_charge(side,envelope,native,**kwargs)


@pytest.mark.parametrize('mutation',['missing_policy','late_chain','late_source'])
def test_full_entry_policy_and_terminal_guards(tmp_path,monkeypatch,mutation):
    manifest,binding,events=orchestration_fixture(tmp_path,monkeypatch)
    if mutation=='missing_policy':
        manifest['rows'][n.remaining.ORDER[9]]['policy_extension']=None
        binding=write_bound(tmp_path,binding['path'],manifest)
    elif mutation=='late_chain':
        before=n.remaining.validate_prior_chain
        def changed(*a,**kw):
            result=before(*a,**kw)
            if events.count('chain')==2:result['prep_seconds']+=1
            return result
        monkeypatch.setattr(n.remaining,'validate_prior_chain',changed)
    else:
        before=n._verify_comparison_sources
        def changed(*a,**kw):
            result=before(*a,**kw)
            return {} if events.count('sources')==2 else result
        monkeypatch.setattr(n,'_verify_comparison_sources',changed)
    with pytest.raises(ValueError):n.compare_native_rows(tmp_path,binding)

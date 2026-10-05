"""Small synthetic XML/receipt fixtures only; no build, solver or tissue data."""
import copy
import hashlib
import json
from pathlib import Path

import pytest

from scripts import mechanics_hbe_access as access
from scripts import mechanics_hbe_backend as b
from scripts import mechanics_hbe_experiment as x


def save(root, relative, value):
    path=root/relative;path.parent.mkdir(parents=True,exist_ok=True)
    data=value if isinstance(value,bytes) else access.canonical_json(value)
    path.write_bytes(data)
    return {'path':relative,'sha256':hashlib.sha256(data).hexdigest()}


def xml():
    return ('<?xml version="1.0"?><febio_spec><Control><solver><dtol>1e-8</dtol>'
            +b.SKYLINE_XML+'</solver></Control><Material><c1>2000</c1></Material>'
            '<Mesh><elem>1,2,3,4,5,6,7,8</elem></Mesh><LoadData><pt>0,0</pt><pt>1,1</pt></LoadData></febio_spec>')


def test_exact_solver_subtree_is_only_change():
    original=xml();adapted=b.transform_deck(original)
    b.verify_deck(original.encode(),adapted.encode())
    assert adapted.replace(b.ACCELERATE_XML,b.SKYLINE_XML)==original
    assert '<iterative>0</iterative>' in adapted and '<factorization>4</factorization>' in adapted


@pytest.mark.parametrize('before,after',[
    ('<c1>2000</c1>','<c1>2100</c1>'),('<dtol>1e-8</dtol>','<dtol>1e-6</dtol>'),
    ('1,2,3,4,5,6,7,8','2,1,3,4,5,6,7,8'),('<pt>0,0</pt><pt>1,1</pt>','<pt>1,1</pt><pt>0,0</pt>'),
    ('<iterative>0</iterative>','<iterative>1</iterative>'),('<factorization>4</factorization>','<factorization>-1</factorization>')])
def test_backend_change_rejects_scientific_or_solver_setting_drift(before,after):
    adapted=b.transform_deck(xml());assert before in adapted
    with pytest.raises(ValueError):b.verify_deck(xml(),adapted.replace(before,after))


@pytest.mark.parametrize('source',[xml().replace(b.SKYLINE_XML,''),xml().replace(b.SKYLINE_XML,b.SKYLINE_XML*2),xml().replace(b.SKYLINE_XML,'<linear_solver type="skyline"><iterative>0</iterative></linear_solver>')])
def test_deck_transform_rejects_missing_duplicate_or_noncanonical_source(source):
    with pytest.raises(ValueError):b.transform_deck(source)


def profile_fixture(root,monkeypatch):
    """Constructed accepted-build metadata, never compiled/executed binaries."""
    prefix=root/b.PREFIX
    libraries={}
    for name in sorted(b.INSTALLED_FILES):
        record=save(root,b.PREFIX+'/'+name,('synthetic non-executable '+name).encode())
        libraries[name]=record['sha256']
    omp=save(root,b.PREFIX+'/openmp/lib/libomp.dylib',b'synthetic omp')
    patched=save(root,b.PREFIX+'/source/NumCore/AccelerateSparseSolver.cpp',b'synthetic patched source')
    monkeypatch.setattr(b,'PATCHED_SOURCE_SHA',patched['sha256'])
    patch_file=save(root,'constructed-combined.patch',b'synthetic combined patch, not applied')
    monkeypatch.setattr(b,'PATCH_SHA',patch_file['sha256'])
    adapter_controls={key:save(root,'adapter-controls/'+key+'.json',{'synthetic_fixture_only':True})
                      for key in ('declaration','acceptance','result','supervision','independent_review')}
    def absolute(record):return {'path':str(root/record['path']),'sha256':record['sha256']}
    directory='build-fixture/build-01/'
    source=save(root,'build-fixture/configure-01/source-inventory.json',
                {'NumCore/AccelerateSparseSolver.cpp':{'sha256':patched['sha256']}})
    installed=save(root,directory+'installed-inventory.json',
        {name.removeprefix('install/'):{'sha256':digest} for name,digest in libraries.items()})
    linkage=save(root,directory+'linkage.json',{'status':'static_linkage_passed',
        'files':[{'path':name,'sha256':digest} for name,digest in libraries.items()]})
    candidate={'schema_version':2,'status':'candidate_pending_parent_build_acceptance',
        'source_commit':b.UPSTREAM_COMMIT,'runtime_version':'4.13.0','architecture':'arm64',
        'source_patch_sha256':b.PATCH_SHA,'patched_source_sha256':patched['sha256'],
        'executable':str(prefix/'install/bin/febio4'),'executable_sha256':libraries['install/bin/febio4'],
        'libraries':libraries,'private_openmp':{'path':str(root/omp['path']),'sha256':omp['sha256']},
        'driver_sha256':'1'*64,'declaration_sha256':'2'*64,
        'source_inventory':absolute(source),'source_inventory_sha256':source['sha256'],
        'source_inventory_path':str(root/source['path']),'installed_inventory':absolute(installed),
        'linkage':absolute(linkage),'adapter_controls':adapter_controls,'synthetic_fixture_only':True}
    preservation={'synthetic_original_preserved':True}
    records={'installed-inventory.json':installed,'linkage.json':linkage,
        'runtime-identity-candidate.json':save(root,directory+'runtime-identity-candidate.json',candidate),
        'result.json':save(root,directory+'result.json',{'status':'completed','stage':'build','driver_sha256':'1'*64,
            'declaration_sha256':'2'*64,'original_before':preservation,'original_after':preservation}),
        'supervision.json':save(root,directory+'supervision.json',{'status':'completed','exit_code':0,'kill_reason':None}),
        'commands.json':save(root,directory+'commands.json',{'synthetic_fixture_only':True})}
    artifacts={key:record['sha256'] for key,record in records.items()}
    accepted=save(root,directory+'acceptance.json',{'status':'completed','stage':'build','driver_sha256':'1'*64,
        'declaration_sha256':'2'*64,'artifacts':artifacts,'parent_original_after':preservation})
    runtime={**candidate,'status':'isolated_patched_runtime_built_pending_numerical_controls',
        'build_acceptance':absolute(accepted),'build_acceptance_sha256':accepted['sha256'],
        'build_acceptance_path':str(root/accepted['path']),
        'receipts_sha256':{'build-01/'+key:value for key,value in artifacts.items()}}
    runtime_binding=save(root,'runtime.json',runtime)
    controls={}
    for key,cases,status,field in [('hex8_controls',b.HEX_CASES,'passed_all_five_fixed_patch_controls','rows'),('tet10_mpc_controls',b.TET_CASES,'three_actual_fixed_software_controls_passed','case_rows')]:
        value={'status':status,'solver_backend':'accelerate','runtime_identity_sha256':runtime_binding['sha256'],
               'solver_invocations':len(cases),field:[{'case':c,'passed':True} for c in sorted(cases)],
               'stiffness_scaling':{'passed':True},'synthetic_fixture_only':True}
        controls[key]=save(root,key+'.json',value)
    patch=save(root,'patch.json',{'one_source_file_only':'NumCore/AccelerateSparseSolver.cpp',
                               'patch':patch_file,'patched_source':patched})
    monkeypatch.setattr(b,'PATCH_IDENTITY',patch)
    profile={'schema':'hbe-solver-backend-v1','profile_id':b.PROFILE_ID,'runtime_identity':runtime_binding,
             **controls,'patch_identity':patch}
    return profile,save(root,'profile.json',profile)


def test_profile_binding_collection_with_control_replay_explicitly_isolated(tmp_path,monkeypatch):
    profile,record=profile_fixture(tmp_path,monkeypatch)
    monkeypatch.setattr(b,'_verify_controls',lambda *args:None)
    context=b.verify_profile(tmp_path,record)
    assert context['runtime_identity']==profile['runtime_identity'] and context['prefix']==str(tmp_path/b.PREFIX)
    assert len(context['inputs'])>=20 and str(tmp_path/b.PREFIX/'install/bin/febio4') in context['inputs']


def test_committed_combined_patch_identity_and_bytes_are_bound():
    root=Path(__file__).resolve().parents[1]
    seen={}
    def bound(record,*,json_value=True):
        value=access.verify_binding(root,record,read_json=json_value)
        seen[record['path']]=record['sha256']
        return value
    assert b.PATCH_SHA=='67a1858f2d55046796d7ccb46758ca06e5bfb673e75a539d158b1a17652f3340'
    assert b.PATCHED_SOURCE_SHA=='476ac8471ea681a99c298352a50aba2a7a5baa71635e1678155a7f58b72ba04a'
    b._verify_patch_identity(b.PATCH_IDENTITY,bound)
    assert len(seen)==3 and b.PATCH_SHA in seen.values() and b.PATCHED_SOURCE_SHA in seen.values()


@pytest.mark.parametrize('change',['old_identity','changed_patch','changed_source','missing_adapter_receipt'])
def test_combined_profile_rejects_old_or_stale_repair_dependencies(tmp_path,monkeypatch,change):
    profile,record=profile_fixture(tmp_path,monkeypatch)
    monkeypatch.setattr(b,'_verify_controls',lambda *args:None)
    if change=='old_identity':
        profile['patch_identity']={'path':'artifacts/febio-accelerate-csc-repair-v1/patch-identity.json','sha256':'0'*64}
        record=save(tmp_path,'profile.json',profile)
    elif change=='changed_patch':(tmp_path/'constructed-combined.patch').write_bytes(b'changed')
    elif change=='changed_source':(tmp_path/b.PREFIX/'source/NumCore/AccelerateSparseSolver.cpp').write_bytes(b'changed')
    else:(tmp_path/'adapter-controls/independent_review.json').unlink()
    with pytest.raises((ValueError,FileNotFoundError)):b.verify_profile(tmp_path,record)


@pytest.mark.parametrize('bad',['missing','skyline','wrong_runtime','missing_case','failed_scale','failed_mpc','source_changed','missing_binary'])
def test_profile_rejects_missing_stale_or_borrowed_evidence(tmp_path,monkeypatch,bad):
    profile,record=profile_fixture(tmp_path,monkeypatch)
    if bad=='missing':profile.pop('tet10_mpc_controls')
    elif bad=='source_changed':(tmp_path/b.PREFIX/'source/NumCore/AccelerateSparseSolver.cpp').write_bytes(b'changed')
    elif bad=='missing_binary':(tmp_path/b.PREFIX/'install/bin/febio4').unlink()
    else:
        key='tet10_mpc_controls' if bad=='failed_mpc' else 'hex8_controls'
        value=json.loads((tmp_path/profile[key]['path']).read_text())
        if bad=='skyline':value['solver_backend']='skyline'
        if bad=='wrong_runtime':value['runtime_identity_sha256']='0'*64
        if bad=='missing_case':value['rows'].pop()
        if bad=='failed_scale':value['stiffness_scaling']['passed']=False
        if bad=='failed_mpc':value['case_rows'][0]['passed']=False
        profile[key]=save(tmp_path,key+'.json',value)
    record=save(tmp_path,'profile.json',profile)
    with pytest.raises((ValueError,FileNotFoundError)):b.verify_profile(tmp_path,record)


def prepared_fixture(root):
    mesh=save(root,'original/mesh.json',{'synthetic_mesh_only':True})
    deck=save(root,'original/specimen.feb',xml().encode())
    loading=save(root,'original/loading.json',{'deck_sha256':deck['sha256'],'preserved_loading_marker':[1,2,3]})
    return {key:{'mesh':mesh,'deck':deck,'loading':loading} for key,row in access.expected_runs().items() if row[-1]!='fitted'}


def test_eighteen_prepared_decks_preserve_original_bytes_mesh_and_loading(tmp_path):
    original=prepared_fixture(tmp_path)
    before={row['path']:(tmp_path/row['path']).read_bytes() for row in next(iter(original.values())).values()}
    record=b.prepare_decks(tmp_path,original,tmp_path/'adapted')
    cases,sources,inputs=b.verify_prepared_decks(tmp_path,record,original)
    assert len(cases)==len(sources)==18
    assert all(row['mesh']==next(iter(original.values()))['mesh'] for row in cases.values())
    assert all((tmp_path/path).read_bytes()==data for path,data in before.items())
    with pytest.raises(FileExistsError):b.prepare_decks(tmp_path,original,tmp_path/'adapted')


@pytest.mark.parametrize('change',['xml','loading','mesh','dropped_case'])
def test_prepared_inventory_rejects_rehashed_scientific_changes(tmp_path,change):
    original=prepared_fixture(tmp_path);record=b.prepare_decks(tmp_path,original,tmp_path/'adapted')
    manifest=json.loads((tmp_path/record['path']).read_text());key=next(iter(manifest['cases']))
    row=manifest['cases'][key]['adapted']
    if change=='xml':
        text=(tmp_path/row['deck']['path']).read_text().replace('<c1>2000</c1>','<c1>2100</c1>')
        row['deck']=save(tmp_path,row['deck']['path'],text.encode())
    if change=='loading':
        value=json.loads((tmp_path/row['loading']['path']).read_text());value['preserved_loading_marker'].reverse()
        row['loading']=save(tmp_path,row['loading']['path'],value)
    if change=='mesh':row['mesh']=save(tmp_path,'different-mesh.json',{'different':True})
    if change=='dropped_case':manifest['cases'].pop(key)
    record=save(tmp_path,record['path'],manifest)
    with pytest.raises(ValueError):b.verify_prepared_decks(tmp_path,record,original)


@pytest.mark.parametrize('opt_in,declared',[(False,True),(True,False)])
def test_backend_execution_requires_both_release_profile_and_optin(tmp_path,monkeypatch,opt_in,declared):
    plan={'backend_profile':{}} if declared else {}
    monkeypatch.setattr(x,'preflight',lambda *args:plan)
    with pytest.raises(ValueError,match='Backend comparison'):
        x.launch(tmp_path,{}, {},explicit_backend=opt_in)
    assert list(tmp_path.iterdir())==[]


def test_run_rejects_scientific_deck_change_before_solver_count(tmp_path,monkeypatch):
    import time
    from types import SimpleNamespace
    original=save(tmp_path,'source.feb',xml().encode())
    wrong=b.transform_deck(xml()).replace('<c1>2000</c1>','<c1>2100</c1>')
    inputs={'deck':save(tmp_path,'adapted.feb',wrong.encode()),
            'loading':save(tmp_path,'loading.json',{}),'mesh':save(tmp_path,'mesh.json',{})}
    directory=tmp_path/'experiment';directory.mkdir()
    plan={'experiment':str(directory),'backend_profile':{'path':'profile.json','sha256':'0'*64},
          'backend_source_decks':{x.REUSED_RUN:original},'inputs':{},
          'protocol':{'budgets':{'each_solver_seconds':90}}}
    state={'solver_invocations':0,'runs':{x.REUSED_RUN:{'status':'not_executed'}}}
    monkeypatch.setattr(x,'solve',lambda *args:pytest.fail('Invalid deck reached a solver'))
    with pytest.raises(ValueError,match='outside the exact solver'):
        x.run_case(tmp_path,plan,x.REUSED_RUN,inputs,1000.,state,SimpleNamespace(active=None),time.monotonic()+5)
    assert state['solver_invocations']==0 and state['runs'][x.REUSED_RUN]['status']=='failed'


def test_fitted_backend_pipeline_preserves_original_scaled_decks(tmp_path,monkeypatch):
    from tests.test_mechanics_hbe_experiment import fixture,scalar_deck
    real_check=x.check_fitted_deck
    plan,events=fixture(tmp_path,monkeypatch)
    plan['backend_profile']={'path':'profile.json','sha256':'0'*64}
    monkeypatch.setattr(x,'check_fitted_deck',real_check)
    def model_deck(mu):return scalar_deck(mu).replace('<dtol>',b.SKYLINE_XML+'<dtol>')
    monkeypatch.setattr(x.meshing,'specimen_deck',lambda mesh,mode,steps,mu,protocol:
        (model_deck(mu),{'deck_sha256':hashlib.sha256(model_deck(mu).encode()).hexdigest()}))
    for mode in ('compression','tension'):
        reference=save(tmp_path,f'{mode}-reference.feb',b.transform_deck(model_deck(1000.)).encode())
        plan['cases'][f'{mode}:N12:S120:reference']['deck']=reference
    fake_case=x.run_case;fitted=[]
    def checking_case(root,plan,run_id,inputs,mu,state,watch,deadline,**kwargs):
        if run_id.endswith(':fitted'):
            source=kwargs['backend_source_deck']
            source_bytes=(root/source['path']).read_bytes()
            assert source_bytes==model_deck(mu).encode()
            b.verify_deck(source_bytes,(root/inputs['deck']['path']).read_bytes())
            loading=json.loads((root/inputs['loading']['path']).read_text())
            assert loading['deck_sha256']==inputs['deck']['sha256']
            fitted.append(run_id)
        return fake_case(root,plan,run_id,inputs,mu,state,watch,deadline)
    monkeypatch.setattr(x,'run_case',checking_case)
    result=x.experiment_worker(tmp_path,plan)
    assert result['status']=='completed_descriptive_one_specimen_comparison',result.get('error')
    assert result['solver_invocations']==20 and fitted==['compression:N12:S120:fitted','tension:N12:S120:fitted']
    assert events.index('compare20')<events.index('freeze')<events.index('holdout')


def test_runtime_identity_check_does_not_replace_eight_control_gate(tmp_path,monkeypatch):
    profile,record=profile_fixture(tmp_path,monkeypatch)
    (tmp_path/profile['hex8_controls']['path']).unlink()
    (tmp_path/profile['tet10_mpc_controls']['path']).unlink()
    context=b.verify_runtime_binding(tmp_path,profile['runtime_identity'])
    assert context['runtime_identity']==profile['runtime_identity']
    assert 'hex8_controls' not in context
    with pytest.raises(FileNotFoundError):b.verify_profile(tmp_path,record)


def test_summary_booleans_do_not_substitute_for_actual_control_evidence(tmp_path,monkeypatch):
    profile,record=profile_fixture(tmp_path,monkeypatch)
    with pytest.raises(ValueError,match='actual eight-case attempt'):
        b.verify_profile(tmp_path,record)


@pytest.mark.parametrize('change',['candidate','missing_library','acceptance_missing','acceptance_failed','stale_installed_inventory'])
def test_runtime_requires_complete_parent_accepted_build(tmp_path,monkeypatch,change):
    profile,record=profile_fixture(tmp_path,monkeypatch)
    identity=json.loads((tmp_path/profile['runtime_identity']['path']).read_text())
    if change=='candidate':identity['status']='candidate_pending_parent_build_acceptance'
    if change=='missing_library':identity['libraries'].pop('install/lib/libnumcore.dylib')
    if change=='acceptance_missing':identity.pop('build_acceptance')
    if change=='acceptance_failed':
        path=Path(identity['build_acceptance']['path']);value=json.loads(path.read_text());value['status']='failed_or_incomplete'
        path.write_bytes(access.canonical_json(value));digest=hashlib.sha256(path.read_bytes()).hexdigest()
        identity['build_acceptance']['sha256']=digest;identity['build_acceptance_sha256']=digest
    if change=='stale_installed_inventory':Path(identity['installed_inventory']['path']).write_text('{}')
    record=save(tmp_path,'runtime.json',identity)
    with pytest.raises((ValueError,KeyError)):b.verify_runtime_binding(tmp_path,record)

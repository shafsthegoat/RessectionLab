"""Temporary analytical ZIP/receipt fixtures only; real HBE members stay sealed."""
import hashlib
import json
from pathlib import Path
import zipfile

import pytest

from scripts import mechanics_hbe_access as a
from scripts import mechanics_hbe_evaluation as e


def save(root,path,value):
    target=root/path;target.parent.mkdir(parents=True,exist_ok=True)
    content=a.canonical_json(value);target.write_bytes(content)
    return {'path':path,'sha256':a.digest(content)}


def schemas(branches):
    return {b:{'delimiter':',','header':['x','y'],'coordinate_column':0,'response_column':1,
               'coordinate_unit':'rad' if b.startswith('torsion') else 'm',
               'response_unit':'Nm' if b.startswith('torsion') else 'N'} for b in branches}


def study_fixture(root):
    path=root/'analytical.zip'
    contents={'compression':'x,y\n0,0\n-0.0005,-0.01\n','tension':'x,y\n0,0\n0.0005,0.01\n',
              'torsion_neg':'x,y\n0,0\n-0.1,-0.0002\n','torsion_pos':'x,y\n0,0\n0.1,0.0002\n'}
    with zipfile.ZipFile(path,'w') as archive:
        for mode,value in contents.items():archive.writestr(f'fixture/{mode}.csv',value)
        archive.writestr('other_donor_sealed.csv','unopened analytical sentinel')
    with zipfile.ZipFile(path) as archive:
        members={mode:{'path':f'fixture/{mode}.csv','bytes':archive.getinfo(f'fixture/{mode}.csv').file_size,
                       'crc32':f'{archive.getinfo(f"fixture/{mode}.csv").CRC:08x}'} for mode in contents}
    source={'archive_path':'analytical.zip','archive_bytes':path.stat().st_size,'archive_sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
    roles=save(root,'roles.json',{'source':source,'calibration':{'members':[members[x] for x in ('compression','tension')]},
                                'held_out_validation':{'members':[members[x] for x in ('torsion_neg','torsion_pos')]}})
    protocol=save(root,'protocol.json',{'roles':roles})
    prerequisites=save(root,'test-only-prerequisite.json',{'test_only_analytical_contract':True})
    release=save(root,'release.json',{'schema':'hbe-calibration-release-v1','source_commit':'1'*40,'source_archive_sha256':'2'*64,
                                     'protocol_sha256':protocol['sha256'],'roles_sha256':roles['sha256'],'archive_sha256':source['archive_sha256'],
                                     'permitted_members':[members[x]['path'] for x in ('compression','tension')],
                                     'prerequisite_evidence':{k:prerequisites for k in ('runtime','analytic_verification','mesh_validation')},
                                     'execution':{'access_ledger_path':'access.jsonl','csv_schemas':schemas(a.BRANCHES)}})
    return a.ReleasedStudy(root,protocol,release,ledger_path='access.jsonl')


def analytical_prerequisite_stubs(monkeypatch):
    """Lifecycle isolation only; explicitly does not prove actual20-run fidelity.

    Authentic provenance/replay rejection has separate unmocked negative tests.
    Physics, raw parsing and readout have independent analytical test modules.
    """
    monkeypatch.setattr(a,'verify_release_provenance',lambda *args,**kwargs:None)
    def accepted_analytical_report(root,binding,*,protocol_sha256,fitted_mu_Pa,expected_backend_profile=None):
        assert expected_backend_profile is None  # This fixture isolates legacy lifecycle only.
        report=a.verify_binding(root,binding,read_json=True)
        return a._check_report(report,protocol_sha256=protocol_sha256,fitted_mu_Pa=fitted_mu_Pa,require_fitted=True)
    monkeypatch.setattr(a,'validate_numerical_evidence',accepted_analytical_report)


def numerical_fixture(root,study,mu):
    dummy=save(root,'test-only-primitive.json',{'analytical_contract_fixture_not_a_solver_result':True})
    metric={'actual':0.,'limit':1.,'units':'declared-normalized-error'}
    report={'schema':'hbe-numerical-evidence-v1','protocol_sha256':study.protocol_binding['sha256'],
            'source_bindings':{k:dummy for k in ('physics','primitive_parser','mesh_deck','curve_evaluator','run_readout')},'runs':{}}
    for key,(branch,N,steps,role) in a.expected_runs().items():
        endpoint=.15*.00489159/(.004 if branch.startswith('torsion') else 1)
        if branch in ('compression','torsion_neg'):endpoint=-endpoint
        x=[endpoint*i/steps for i in range(steps+1)]
        scale={'reference':1.,'double_mu':2.,'fitted':mu/1000}[role]
        criteria={k:dict(metric) for k in ('solver_residual','force_balance','moment_balance','prescribed_motion','work_energy','primitive_consistency')}
        criteria['minimum_sampled_J']={'actual':.9,'limit':0,'units':'dimensionless'}
        report['runs'][key]={'branch':branch,'mesh_N':N,'steps':steps,'mu_Pa':1000*scale,'frame_count':steps+1,
                            'load_coordinate':x,'applied_force_N':[10*v*scale for v in x],
                            'applied_torque_Nm':[.001*v*scale for v in x],
                            'primitive_bindings':{k:dummy for k in ('nodes','elements','solver','mesh','deck','loading')},'criteria':criteria}
        report['runs'][key]['execution_binding']=dummy
    for kind in ('mesh','step','scale','fitted_confirmation'):
        modes={'scale':('compression','torsion_pos'),'fitted_confirmation':('compression','tension')}.get(kind,a.BRANCHES)
        names={'reaction','motion','torque'} if kind in ('scale','fitted_confirmation') else {'reaction','motion'}
        if kind=='mesh':names|={'reaction_trend','motion_trend'}
        report[kind]={mode:{name:dict(metric) for name in names} for mode in modes}
    return report


def frozen_fixture(root,study):
    calibration=study.read_calibration(schemas(('compression','tension')))
    numerical=numerical_fixture(root,study,2000.)
    refs={mode:e.curve(mode,numerical['runs'][f'{mode}:N12:S120:reference']['load_coordinate'],
                      numerical['runs'][f'{mode}:N12:S120:reference']['applied_force_N']) for mode in ('compression','tension')}
    fit=e.fit_scale(calibration,refs)
    assert fit['mu_Pa']==pytest.approx(2000)
    numerical=numerical_fixture(root,study,fit['mu_Pa'])
    prediction={'schema':'hbe-torsion-prediction-v1','reference':{},'fitted':{}}
    for mode in ('torsion_neg','torsion_pos'):
        row=numerical['runs'][f'{mode}:N12:S120:reference']
        prediction['reference'][mode]={'coordinate':row['load_coordinate'],'response':row['applied_torque_Nm']}
        scaled=e.scale_prediction(e.curve(mode,row['load_coordinate'],row['applied_torque_Nm']),fit['scale'])
        prediction['fitted'][mode]={'coordinate':list(scaled.coordinate),'response':list(scaled.response),'response_unit':'Nm'}
    fit_binding=save(root,'fit.json',fit);prediction_binding=save(root,'prediction.json',prediction)
    numerical_binding=save(root,'numerical.json',numerical)
    freeze=study.freeze_predictions(fit_binding=fit_binding,prediction_binding=prediction_binding,
                                    numerical_binding=numerical_binding,output_path='freeze.json')
    return freeze,fit_binding,prediction_binding,numerical_binding


def test_only_exact_calibration_members_read_after_release(tmp_path,monkeypatch):
    analytical_prerequisite_stubs(monkeypatch)
    study=study_fixture(tmp_path);opened=[];original=zipfile.ZipFile.read
    def observed(archive,name,*args,**kwargs):
        opened.append(name.filename if isinstance(name,zipfile.ZipInfo) else name)
        return original(archive,name,*args,**kwargs)
    monkeypatch.setattr(zipfile.ZipFile,'read',observed)
    result=study.read_calibration(schemas(('compression','tension')))
    assert set(result)=={'compression','tension'}
    assert opened==['fixture/compression.csv','fixture/tension.csv']
    assert [json.loads(x)['phase'] for x in (tmp_path/'access.jsonl').read_text().splitlines()]==['calibration_attempt','calibration_completed']


def test_missing_release_or_changed_archive_does_not_open_member(tmp_path,monkeypatch):
    study=study_fixture(tmp_path)
    def reject(*args,**kwargs):raise AssertionError('No member may be opened')
    monkeypatch.setattr(zipfile.ZipFile,'read',reject)
    (tmp_path/'release.json').write_text('{}')
    with pytest.raises(ValueError,match='hash'):study.read_calibration(schemas(('compression','tension')))


def test_durable_freeze_then_only_declared_torque_and_no_overwrite(tmp_path,monkeypatch):
    analytical_prerequisite_stubs(monkeypatch)
    study=study_fixture(tmp_path);freeze,*_=frozen_fixture(tmp_path,study)
    opened=[];original=zipfile.ZipFile.read
    def observed(archive,name,*args,**kwargs):
        opened.append(name.filename);return original(archive,name,*args,**kwargs)
    monkeypatch.setattr(zipfile.ZipFile,'read',observed)
    report=study.evaluate_held_out(freeze_binding=freeze,schemas=schemas(('torsion_neg','torsion_pos')))
    assert report['equal_branch_RMSE']<1e-18
    assert report['physical_validation_pass'] is None
    assert opened==['fixture/torsion_neg.csv','fixture/torsion_pos.csv']
    with pytest.raises(FileExistsError):a.exclusive_json(tmp_path/'freeze.json',{})


@pytest.mark.parametrize('target',['fit.json','prediction.json','numerical.json','test-only-primitive.json'])
def test_changed_frozen_artifact_blocks_holdout_before_read(tmp_path,monkeypatch,target):
    analytical_prerequisite_stubs(monkeypatch)
    study=study_fixture(tmp_path);freeze,*_=frozen_fixture(tmp_path,study)
    (tmp_path/target).write_text('{}')
    def reject(*args,**kwargs):raise AssertionError('Sealed torque must not be read')
    monkeypatch.setattr(zipfile.ZipFile,'read',reject)
    with pytest.raises(ValueError,match='hash'):study.evaluate_held_out(freeze_binding=freeze,schemas=schemas(('torsion_neg','torsion_pos')))


def test_numerical_bundle_rejects_bare_pass_infinite_metric_and_missing_run(tmp_path):
    study=study_fixture(tmp_path);report=numerical_fixture(tmp_path,study,2000)
    key=next(iter(report['runs']))
    report['runs'][key]['criteria']['force_balance']={'passed':True}
    binding=save(tmp_path,'numerical.json',report)
    with pytest.raises(ValueError,match='numerical criterion'):a.check_numeric_report(report,fitted_mu_Pa=2000)
    with pytest.raises(ValueError):a._metric_record({'actual':0,'limit':float('inf'),'units':'N'})
    report=numerical_fixture(tmp_path,study,2000);del report['runs'][key]
    binding=save(tmp_path,'numerical.json',report)
    with pytest.raises(ValueError,match='twenty'):a.check_numeric_report(report,fitted_mu_Pa=2000)


def test_dummy_source_prerequisites_and_fake_numerical_proof_are_rejected(tmp_path,monkeypatch):
    study=study_fixture(tmp_path)
    def reject(*args,**kwargs):raise AssertionError('No actual member access is authorized')
    monkeypatch.setattr(zipfile.ZipFile,'read',reject)
    with pytest.raises(ValueError,match='source inventory'):study.read_calibration(schemas(('compression','tension')))
    binding=save(tmp_path,'fake-numerical.json',numerical_fixture(tmp_path,study,2000))
    with pytest.raises(ValueError,match='source binding'):
        a.validate_numerical_evidence(tmp_path,binding,protocol_sha256=study.protocol_binding['sha256'],fitted_mu_Pa=2000)


def test_rehashed_freeze_fresh_ledger_and_refitting_after_reveal_rejected(tmp_path,monkeypatch):
    analytical_prerequisite_stubs(monkeypatch)
    study=study_fixture(tmp_path);freeze,*_=frozen_fixture(tmp_path,study)
    changed=json.loads((tmp_path/'freeze.json').read_text());changed['mu_Pa']=999
    forged=save(tmp_path,'forged-freeze.json',changed)
    with pytest.raises(ValueError,match='durably created'):
        study.evaluate_held_out(freeze_binding=forged,schemas=schemas(('torsion_neg','torsion_pos')))
    study.evaluate_held_out(freeze_binding=freeze,schemas=schemas(('torsion_neg','torsion_pos')))
    fresh=a.ReleasedStudy(tmp_path,study.protocol_binding,study.release_binding,ledger_path='different.jsonl')
    with pytest.raises(ValueError,match='ledger location'):
        fresh.read_calibration(schemas(('compression','tension')))
    reopened=a.ReleasedStudy(tmp_path,study.protocol_binding,study.release_binding,ledger_path='access.jsonl')
    with pytest.raises(ValueError,match='seals subsequent'):
        reopened.read_calibration(schemas(('compression','tension')))


def test_explicit_schema_units_headers_and_sign_cannot_be_guessed():
    spec=schemas(('compression',))['compression']
    with pytest.raises(ValueError,match='header'):a.parse_member_csv(b'a,b\n0,0\n-1,-2\n','compression',spec)
    spec['coordinate_unit']='mm'
    with pytest.raises(ValueError,match='units'):a.parse_member_csv(b'x,y\n0,0\n-1,-2\n','compression',spec)

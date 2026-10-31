"""Generated falsification controls for the separate held-program auditor."""
import copy
import importlib.util
from pathlib import Path
import shutil
import tempfile

import pytest

from resectionlab.imaging import save_case
from test_surgical_preparation_workspace import intact, controls, completed, study, installed, source_record, reseal

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('preparation_comparison_auditor',ROOT/'scripts/verify_preparation_comparison.py')
auditor=importlib.util.module_from_spec(spec);spec.loader.exec_module(auditor)


@pytest.fixture
def audit_case(installed):
    source, _, record, _, _, case=installed
    # Existing auditors deliberately accept only ignored testcases inputs.
    root=Path(tempfile.mkdtemp(prefix='generated-preparation-audit-',dir=ROOT/'testcases'))
    native=source.scene.source['adapter']['sourceManifest']
    assert native['patient_id']=='generated-exposure-person'
    from resectionlab.surgical_environment import SCHEMA as ENVIRONMENT_SCHEMA
    saved={'environmentSchema':ENVIRONMENT_SCHEMA,'physicsProfile':copy.deepcopy(source.scene.properties['tissuePhysicsV2']),
        'mapping':source.scene.regions,**copy.deepcopy(record)}
    path=root/'source.ressectionlab';save_case(case,path,artifacts={'surgical_toolbox':saved})
    try:yield root,path,case,saved
    finally:shutil.rmtree(root)


def test_actual_held_native_preparation_and_installed_branch_audit(audit_case):
    root,path,_,_=audit_case;before=path.read_bytes()
    report=auditor.verify(path,root/'proofs')
    assert report['passed'] and report['sourceUnchanged'] and path.read_bytes()==before
    revision=report['revisions'][0]
    assert revision['rows'][0]['preparation']['microsteps']>0
    assert revision['rows'][0]['geometry']['passed'] and revision['applications']
    assert (root/'proofs/audit.json').exists()
    assert (root/'proofs/revision-1-width-0-preparation.json').exists()
    with pytest.raises(ValueError,match='preserve earlier evidence'):auditor.verify(path,root/'proofs')


@pytest.mark.parametrize('change',['schema','source','setup','width','ray','cost','admission','policy','suffix-cost','app-revision','app-receipt','app-snapshot','app-parent','cycle'])
def test_resealed_source_row_policy_and_application_forgeries_refuse(audit_case,change):
    root,path,case,record=audit_case;record=copy.deepcopy(record)
    item=record['preparationComparisons'][0];report=item['report'];row=report['rows'][0];app=record['preparationApplications'][0]
    if change=='schema':report['schema']+='-forged'
    elif change=='source':report['sourceSnapshotHash']='0'*64
    elif change=='setup':report['configuredSnapshot']=record['branches']['main']
    elif change=='width':row['widthMm']=1.
    elif change=='ray':report['controls']['sourceRay']['surfaceMm'][0]+=.5
    elif change=='cost':
        row['preparation']['publicPlan']['attempts'][0]['summary']['additionalNonTargetMm3']=0.
        reseal(row['preparation']['publicPlan'],'publicSeal');reseal(row['preparation'],'preparationSeal')
    elif change=='admission':
        row['preparedAdmission']['task']['entry']['radiusMm']+=.1;reseal(row['preparedAdmission'],'admissionSeal')
    elif change=='policy':report['policies']['completePlanAware']['selectedPreparationId']='width-0'
    elif change=='suffix-cost':
        row['comparison']['publicPlan']['alternatives'][0]['summary']['elapsedSeconds']=1.
        reseal(row['comparison']['publicPlan'],'publicPlanSeal');reseal(row['comparison'],'reportSeal')
    elif change=='app-revision':app['revision']='1'
    elif change=='app-receipt':app['receipt']['preparationActions']+=1
    elif change=='app-snapshot':record['branches'][app['branchId']]=record['branches']['main']
    elif change=='app-parent':record['parents'][app['preparationBranchId']]['position']=1
    else:record['parents']['main']={'branchId':app['branchId'],'position':0}
    reseal(report,'reportSeal');item['sourceSnapshotHash']=report['sourceSnapshotHash'];reseal(item,'workspaceSeal')
    app['reportSeal']=report['reportSeal'];app['receipt']['reportSeal']=report['reportSeal'];reseal(app,'applicationSeal')
    save_case(case,path,artifacts={'surgical_toolbox':record})
    with pytest.raises(ValueError):auditor.verify(path,root/'proofs')


def test_auditor_has_no_preparation_producer_imports():
    source=(ROOT/'scripts/verify_preparation_comparison.py').read_text()
    assert 'surgical_preparation_planning' not in source and 'surgical_preparation_workspace' not in source
    assert 'surgical_exposure' not in source and 'surgical_two_tool' not in source


@pytest.mark.parametrize('kind',['missing-functional','empty-preparation-family'])
def test_empty_rows_state_their_limited_independent_scope(audit_case,study,kind):
    from resectionlab import surgical_preparation_planning as planning
    from resectionlab import surgical_preparation_workspace as workspace
    root,path,case,saved=audit_case;engine,native,controls,_,_=study;controls=copy.deepcopy(controls)
    if kind=='missing-functional':controls['requiredEvidence'].append('functional')
    else:controls['maxNonTargetMm3']=.1
    report=planning.generate(engine,controls,native.root)
    record={k:v for k,v in saved.items() if k not in ('preparationComparisons','preparationApplications')}
    record.update(source_record(engine));workspace.install_report(engine,record,case.semantic_hash,report)
    save_case(case,path,artifacts={'surgical_toolbox':record})
    audited=auditor.verify(path,root/'empty-proofs');row=audited['revisions'][0]['rows'][0]
    assert audited['passed'] and report['rows'][0]['preparedSnapshot'] is None
    if kind=='missing-functional':assert row['missingEvidenceVerified']
    else:
        assert row['preparation']['microsteps']==0
        assert 'not independently qualified' in audited['scope']

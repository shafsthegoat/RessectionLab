"""Tiny saved-record contract checks; no patient arrays or inference."""
from copy import deepcopy
import importlib.util
from pathlib import Path
import sys

import pytest


SCRIPTS = Path(__file__).resolve().parents[1] / 'scripts'
SPEC = importlib.util.spec_from_file_location('audit_pat16_pat20_extraction', SCRIPTS / 'audit_pat16_pat20_extraction.py')
audit = importlib.util.module_from_spec(SPEC)
sys.path.insert(0, str(SCRIPTS))
try:
    SPEC.loader.exec_module(audit)
finally:
    sys.path.pop(0)


@pytest.fixture
def saved_contract():
    versions = dict(python='3.12.14', numpy='2.2.6', torch='2.14.1', surfa='0.6.3', scipy='1.18.1', nibabel='5.4.2')
    config = dict(device='mps', cpu_threads=2, border_mm=1, timeout_seconds=300, maximum_rss_bytes=6442450944,
                  annotation_threshold=0.5)
    assets = {name: {'sha256':name+'-fixture-hash', 'bytes':42} for name in
              ('mri_synthstrip','LICENSE_FreeSurfer.txt','synthstrip.nocsf.1.pt','synthstrip.1.pt')}
    declaration = dict(wrapper={'sha256':'wrapper'}, inference_runtime=versions, frozen_configuration=config,
                       expected_executed_MPS_runner_sha256='runner',model_code_commit='pinned',weights_version=1,model_assets=assets)
    subject = dict(subject='sub-PAT16',source_t1_sha256='source',source_annotation_sha256='annotation')
    run = dict(run_id='PAT16-r1',subject='sub-PAT16',argv=['frozen'],output_directory='fixed',variant_order=['nocsf','main'])
    attempt = dict(run_id='PAT16-r1',subject='sub-PAT16',argv=['frozen'],output_directory='fixed',status='completed',exit_code=0,
                   runtime_metadata_preflight=versions.copy(),children={},started_at='2026-10-04T12:00:00+00:00',finished_at='2026-10-04T12:01:00+00:00')
    report = dict(brain_reviewed=False,cortical_access_permitted=False,implementation_sha256='wrapper',source_t1_sha256='source',
                  source_annotation={'sha256':'annotation','geometry_and_threshold':{'threshold':0.5}},variants={},created_at='2026-10-04T12:00:59+00:00')
    for name in ('nocsf','main'):
        weight = 'synthstrip.nocsf.1.pt' if name=='nocsf' else 'synthstrip.1.pt'
        inference = dict(exit_code=0,failure=None,input_sha256='source',executed_runner_sha256='runner',
                         model={'variant':name,'upstream_code_commit':'pinned','weights_version':1,
                                'files':{f:assets[f] for f in ('mri_synthstrip','LICENSE_FreeSurfer.txt',weight)}},
                         configuration={k:v for k,v in config.items() if k!='annotation_threshold'},
                         runtime_versions={k:versions[k] for k in ('python','numpy','torch','surfa')},
                         brain_reviewed=False,cortical_access_permitted=False,artifact_hashes={'mask':'fixture'})
        inference['configuration']['weights_only_load']=True
        qc=dict(brain_reviewed=False,cortex_localized=False,cortical_access_permitted=False,clinical_deficit_probability=None,
                source_annotation_outside_voxels=7)
        report['variants'][name]={'inference':inference,'qc':qc}
        attempt['children'][name]={'artifact_hashes':{'mask':'fixture'},'source_annotation_outside_voxels':7}
    return declaration,run,attempt,report,subject


def test_consistent_metadata_contract_is_not_mutated(saved_contract):
    before=deepcopy(saved_contract)
    audit.check_run_contract(*saved_contract)
    assert saved_contract==before


@pytest.mark.parametrize('mutation',['runtime','source','runner','threshold','omission','approval','model','time','extra_attempt_scope'])
def test_inconsistent_saved_contract_fails_closed(saved_contract,mutation):
    declaration,run,attempt,report,subject=saved_contract
    if mutation=='runtime': attempt['runtime_metadata_preflight']['numpy']='different'
    elif mutation=='source': report['source_t1_sha256']='other'
    elif mutation=='runner': report['variants']['main']['inference']['executed_runner_sha256']='changed'
    elif mutation=='threshold': report['source_annotation']['geometry_and_threshold']['threshold']=0.25
    elif mutation=='omission': attempt['children']['main']['source_annotation_outside_voxels']=0
    elif mutation=='approval': report['variants']['main']['qc']['brain_reviewed']=True
    elif mutation=='model': report['variants']['main']['inference']['model']['files'].pop('synthstrip.1.pt')
    elif mutation=='time': report['created_at']='2026-10-04T11:59:00+00:00'
    else: attempt['argv']=['different-scope']
    with pytest.raises(ValueError):
        audit.check_run_contract(*saved_contract)

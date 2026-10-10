"""Generated-only bridge checks; evaluator owns private reference and replay."""
from pathlib import Path
import copy
import hashlib
import json
import sys

import pytest
from resectionlab.desktop_bridge import BridgeRuntime

class Client:
    def __init__(self, path):
        self.events=[];self.runtime=BridgeRuntime(path,self.events.append);self.serial=0
    def call(self, op, args, *, error=False):
        self.serial+=1; key=str(self.serial)
        self.runtime.submit({'id':key,'op':op,'args':args})
        assert self.runtime.wait_idle(20)
        result=[row for row in self.events if row['id']==key][-1]
        assert result['event']==('error' if error else 'result'), result
        return result if error else result['result']
    def close(self): self.runtime.close()

@pytest.fixture
def current(tmp_path):
    client=Client(tmp_path/'bridge')
    executed=client.call('executeDevelopmentEpisode',{'fixture':'generated-sequential-v1','selector':'scripted'})
    args={'caseHash':executed['case']['caseHash'],'episodeId':executed['episode']['episodeId']}
    yield client,executed,args
    client.close()


def test_actual_postseal_result_preserves_episode_case_and_transfers(current):
    client,executed,args=current
    entry=client.runtime.session.cases[args['caseHash']]
    envelope=copy.deepcopy(entry.episode);planning_hash=entry.case.planning_hash
    transfers=copy.deepcopy(client.runtime.session.transfers._entries)
    out=client.call('evaluateDevelopmentEpisodeVascular',args)
    report=out['evaluation']
    assert set(out)=={'caseHash','evaluation','evaluationCanonicalJson'}
    assert out['caseHash']==args['caseHash'] and report['episodeId']==args['episodeId']
    assert 'sha256:'+hashlib.sha256(out['evaluationCanonicalJson'].encode()).hexdigest()==report['evaluationId']
    assert json.loads(report['physicalHistoryCanonicalJson'])==executed['episode']['history']
    assert report['actionIds']==[row['action_id'] for row in executed['episode']['history']]
    assert len(report['perAction'])==len(report['actionIds'])
    assert report['perAction'][-1]['actionId']=='STOP' and report['perAction'][-1]['wholeTool'] is None
    assert report['patientAdmission'] is False and report['clinicalInjuryProbability'] is None
    assert entry.episode==envelope and entry.case.planning_hash==planning_hash
    assert client.runtime.session.transfers._entries==transfers
    assert not hasattr(entry,'vascular_evaluation')


def test_reopened_backend_episode_is_evaluable_but_report_is_not_saved(current,tmp_path):
    client,executed,args=current
    path=tmp_path/'generated.ressectionlab'
    client.call('saveCase',{'caseHash':args['caseHash'],'path':str(path)})
    from zipfile import ZipFile
    with ZipFile(path) as archive: original_workspace=json.loads(archive.read('workspace.json'))
    second=Client(tmp_path/'second')
    try:
        loaded=second.call('loadCase',{'path':str(path)})
        assert loaded['workspaceSession']['episodeReplay']['episode']['episodeId']==args['episodeId']
        out=second.call('evaluateDevelopmentEpisodeVascular',args)
        assert out['evaluation']['episodeId']==args['episodeId']
        saved=tmp_path/'after.ressectionlab'
        second.call('saveCase',{'caseHash':args['caseHash'],'path':str(saved)})
        from zipfile import ZipFile
        with ZipFile(saved) as archive:
            workspace=json.loads(archive.read('workspace.json'))
            assert workspace==original_workspace
            assert out['evaluation']['evaluationId'] not in json.dumps(workspace)
    finally: second.close()


@pytest.mark.parametrize('bad',[
    {'episode':{}}, {'referencePath':'forbidden'}, {'output_directory':'forbidden'},
    {'episodeId':'sha256:'+'0'*64}, {'caseHash':'sha256:'+'0'*64},
])
def test_renderer_override_or_stale_identity_refuses_before_reference(current,monkeypatch,bad):
    from resectionlab import shared_vascular_evaluation as evaluator
    def forbidden(**kwargs):pytest.fail('invalid request reached reference evaluator')
    monkeypatch.setattr(evaluator,'evaluate_development_episode_vascular',forbidden)
    client,_,args=current
    client.call('evaluateDevelopmentEpisodeVascular',{**args,**bad},error=True)


def test_missing_or_mutated_cached_episode_refuses_before_reference(current,monkeypatch):
    from resectionlab import shared_vascular_evaluation as evaluator
    monkeypatch.setattr(evaluator,'evaluate_development_episode_vascular',lambda **kwargs:pytest.fail('reached evaluator'))
    client,_,args=current;entry=client.runtime.session.cases[args['caseHash']]
    old=entry.episode;entry.episode=None
    assert client.call('evaluateDevelopmentEpisodeVascular',args,error=True)['error']['code']=='EPISODE_UNAVAILABLE'
    entry.episode=copy.deepcopy(old);entry.episode['episode']['history'][0]['action_id']='changed'
    assert client.call('evaluateDevelopmentEpisodeVascular',args,error=True)['error']['code']=='EPISODE_BINDING_MISMATCH'


def test_evaluator_failure_cleans_temporary_output_and_preserves_owned_envelope(current,monkeypatch):
    from resectionlab import shared_vascular_evaluation as evaluator
    paths=[]
    def broken(*,episode,output_directory,cancelled):
        paths.append(output_directory);output_directory.mkdir();(output_directory/'negative.json').write_text('{}')
        episode['history'][0]['action_id']='mutated_copy'
        raise ValueError('generated negative')
    monkeypatch.setattr(evaluator,'evaluate_development_episode_vascular',broken)
    client,_,args=current;entry=client.runtime.session.cases[args['caseHash']];old=copy.deepcopy(entry.episode)
    client.call('evaluateDevelopmentEpisodeVascular',args,error=True)
    assert entry.episode==old and len(paths)==1 and not paths[0].parent.exists()


@pytest.mark.parametrize('tamper',['private_payload','source','history','digest','order','sweep_count','coverage','encounter'])
def test_malformed_evaluator_report_never_publishes(current,monkeypatch,tamper):
    from resectionlab import shared_vascular_evaluation as evaluator
    original=evaluator.evaluate_development_episode_vascular
    def malformed(**kwargs):
        report=original(**kwargs)
        if tamper=='private_payload':report['referenceArray']=[1,2,3]
        elif tamper=='source':report['sourceHash']='sha256:'+'0'*64
        elif tamper=='history':report['physicalHistoryCanonicalJson']='[]'
        elif tamper=='order':report['perAction'][0]['actionIndex']=1
        elif tamper=='sweep_count':report['perAction'][0]['sweepCount']=2
        elif tamper=='coverage':report['wholeTool']['annotation_coverage_complete_for_sweep']=not report['wholeTool']['annotation_coverage_complete_for_sweep']
        elif tamper=='encounter':report['wholeTool']['annotated_positive_encounter']=not report['wholeTool']['annotated_positive_encounter']
        else:report['evaluationId']='sha256:'+'0'*64
        if tamper!='digest':
            canonical=json.dumps({k:v for k,v in report.items() if k!='evaluationId'},sort_keys=True,separators=(',',':'),allow_nan=False)
            report['evaluationId']='sha256:'+hashlib.sha256(canonical.encode()).hexdigest()
        return report
    monkeypatch.setattr(evaluator,'evaluate_development_episode_vascular',malformed)
    client,_,args=current
    assert client.call('evaluateDevelopmentEpisodeVascular',args,error=True)['error']['code']=='VASCULAR_RESULT_MISMATCH'


def test_cancellation_after_evaluation_prevents_publication_and_cleans_output(current,monkeypatch):
    from resectionlab import shared_vascular_evaluation as evaluator
    from resectionlab.desktop_bridge import _Request,BridgeError
    original=evaluator.evaluate_development_episode_vascular;paths=[];request=_Request('cancel-control')
    def cancel_after(**kwargs):
        paths.append(kwargs['output_directory']);report=original(**kwargs)
        request.cancelled.set()
        return report
    monkeypatch.setattr(evaluator,'evaluate_development_episode_vascular',cancel_after)
    client,_,args=current
    with pytest.raises(BridgeError) as raised:
        client.runtime.session.execute('evaluateDevelopmentEpisodeVascular',args,request,lambda *args:None)
    assert raised.value.code=='CANCELLED'
    assert len(paths)==1 and not paths[0].parent.exists()


@pytest.mark.parametrize('change',['replace','in_place'])
def test_ownership_change_during_evaluation_refuses_stale_publication(current,monkeypatch,change):
    from dataclasses import replace
    from resectionlab import shared_vascular_evaluation as evaluator
    original=evaluator.evaluate_development_episode_vascular;paths=[]
    client,_,args=current;session=client.runtime.session;entry=session.cases[args['caseHash']]
    def changed(**kwargs):
        paths.append(kwargs['output_directory']);report=original(**kwargs)
        if change=='replace':session.cases[args['caseHash']]=replace(entry)
        else:entry.episode['episode']['history'][0]['action_id']='changed_after_evaluation'
        return report
    monkeypatch.setattr(evaluator,'evaluate_development_episode_vascular',changed)
    assert client.call('evaluateDevelopmentEpisodeVascular',args,error=True)['error']['code']=='EPISODE_VERSION_MISMATCH'
    assert len(paths)==1 and not paths[0].parent.exists()

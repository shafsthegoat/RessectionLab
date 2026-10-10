from pathlib import Path
import hashlib,json,sys
root=Path(__file__).resolve().parent
from resectionlab.desktop_bridge import BridgeSession,_Request,BridgeError,OPERATIONS
from resectionlab.data_policy import DataPolicyError
import pytest
@pytest.fixture
def session(tmp_path):
 s=BridgeSession(tmp_path)
 yield s
 s.transfers.close()
@pytest.mark.parametrize('selector',['scripted','SEARCH'])
def test_real_generated_execution_and_serialization(session,selector):
 result=session.execute('executeDevelopmentEpisode',{'fixture':'generated-sequential-v1','selector':selector},_Request('episode'),lambda *a:None)
 episode=result['episode'];canonical=result['episodeCanonicalJson']
 assert episode['episodeId']=='sha256:'+hashlib.sha256(canonical.encode()).hexdigest()
 assert json.loads(canonical)=={k:v for k,v in episode.items() if k!='episodeId'}
 assert result['case']['caseHash']==episode['caseHash']
 assert episode['patientAdmission'] is False and episode['geometryAudit']['feasible'] is True
 assert episode['history'][-1]['interaction_mode']=='stop'
def test_legacy_guard_unchanged(session):
 with pytest.raises(DataPolicyError): session.execute('createSyntheticCase',{},_Request('legacy'),lambda *a:None)
@pytest.mark.parametrize('args',[{}, {'fixture':'patient','selector':'scripted'},{'fixture':'generated-sequential-v1','selector':'learned'},{'fixture':'generated-sequential-v1','selector':'scripted','path':'x'}])
def test_bad_request_no_install(session,args):
 with pytest.raises(BridgeError):session.execute('executeDevelopmentEpisode',args,_Request('invalid'),lambda *a:None)
 assert not session.cases
def test_cancel_before_execution(session):
 request=_Request('cancelled');request.cancelled.set()
 with pytest.raises(BridgeError):session.execute('executeDevelopmentEpisode',{'fixture':'generated-sequential-v1','selector':'scripted'},request,lambda *a:None)
 assert not session.cases
def test_both_operations_advertised():assert {'executeDevelopmentEpisode','importDisplaySeries'}<=OPERATIONS

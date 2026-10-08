"""Read only the already released six-B field; no raw Tag, V, or image access."""
import copy
from pathlib import Path
import pytest
from resectionlab.desktop_bridge import BridgeRuntime, BridgeSession, _Request, BridgeError
from resectionlab.observed_landmark_replay import ObservedLandmarkReplay, STUDY_ID, FIELD_PATH
from resectionlab.core import semantic_digest
ROOT=Path(__file__).resolve().parents[2]

def test_only_source_open_and_deep_immutability(monkeypatch):
    opened=[]; orig=Path.open
    def guard(p,*a,**k):
        opened.append(p); assert p==ROOT/FIELD_PATH
        return orig(p,*a,**k)
    with monkeypatch.context() as ctx:
        ctx.setattr(Path,'open',guard)
        replay=ObservedLandmarkReplay.load(ROOT)
        for phase in ('before','during'):
            for row in (1,14,8,17,19,7):
                view=replay.advance(phase,row); f=view.frame()
                assert view.reopen(f['snapshot']).frame()==f
                f['snapshot']['binding']['patientGroup']='WRONG'
                f['state']['selectedInspectionPoint']['positionRasMm'][0]=None
                assert view.frame()['state']['selectedInspectionPoint']['positionRasMm'][0] is not None
    assert opened==[ROOT/FIELD_PATH]
    with pytest.raises(TypeError): replay._field['baseline']['observations']['source_ras_mm'][0][0]=None

@pytest.mark.parametrize('action',['open','advance','reopen'])
def test_cancellation_at_actual_commit_boundary(action,tmp_path):
    session=BridgeSession(tmp_path/'transfer',observed_source_root=ROOT)
    try:
        before=session.execute('inspectObservedLandmarkUpdate',{'action':'open','studyId':STUDY_ID},_Request('open'),lambda *_:None)
        current=session.observed_replay
        req=_Request('cancelled'); original=req.begin_commit
        def cancel_then_commit():
            assert req.cancel() is True
            original()
        req.begin_commit=cancel_then_commit
        args={'action':action,'studyId':STUDY_ID}
        if action=='advance':args.update(expectedStateHash=before['stateHash'],phase='during',landmarkId=14)
        if action=='reopen':args.update(expectedStateHash=before['stateHash'],snapshot=current.advance('during',14).snapshot())
        with pytest.raises(BridgeError) as error:session.execute('inspectObservedLandmarkUpdate',args,req,lambda *_:None)
        assert error.value.code=='CANCELLED'
        assert session.observed_replay is current
        assert session.observed_replay.frame()==before
    finally:session.close()

def test_cancellation_after_commit_begins_does_not_claim_cancelled(tmp_path):
    session=BridgeSession(tmp_path/'transfer',observed_source_root=ROOT)
    try:
        req=_Request('open');orig=req.begin_commit
        def boundary():
            orig(); assert req.cancel() is False
        req.begin_commit=boundary
        result=session.execute('inspectObservedLandmarkUpdate',{'action':'open','studyId':STUDY_ID},req,lambda *_:None)
        assert session.observed_replay.frame()==result and not req.cancelled.is_set()
    finally:session.close()

def test_missing_packaged_source_is_operation_failure_not_engine_failure(tmp_path):
    events=[];runtime=BridgeRuntime(tmp_path/'transfer',events.append,observed_source_root=None)
    try:
        runtime.submit({'id':'observe','op':'inspectObservedLandmarkUpdate','args':{'action':'open','studyId':STUDY_ID}})
        assert runtime.wait_idle(5)
        event=[x for x in events if x['id']=='observe'][-1]
        assert event['error']['code']=='OBSERVED_SOURCE_UNAVAILABLE'
        assert runtime.session.observed_replay is None
        runtime.submit({'id':'ping','op':'ping'})
        assert events[-1]['event']=='result'
    finally:runtime.close()

def test_stale_snapshot_does_not_change_current_session(tmp_path):
    session=BridgeSession(tmp_path/'transfer',observed_source_root=ROOT)
    try:
        current=ObservedLandmarkReplay.load(ROOT).advance('during',7);session.observed_replay=current
        snap=current.snapshot();snap['binding']['sourceImageSha256']='sha256:'+'0'*64
        with pytest.raises(ValueError):session.execute('inspectObservedLandmarkUpdate',{'action':'reopen','studyId':STUDY_ID,'expectedStateHash':current.frame()['stateHash'],'snapshot':snap},_Request('reopen'),lambda *_:None)
        assert session.observed_replay is current
    finally:session.close()

"""Generated counter/spool tests only. No patient loading or physics."""
import json
import os
import subprocess
import sys
from pathlib import Path
import tempfile
import unittest
import zlib
from types import MappingProxyType, SimpleNamespace

from vascular_playback_monitor import (BRANCH, FrameSpool, VascularPlaybackMonitor,
    ForbiddenPhysicsStep, MAX_RSS, ROOT, canonical_json, observed_activation, checked_field_tokens)


FAKE_SCENE=object()


class FakeEngine:
    def __init__(self):
        self.actions=[]
        self.scene=FAKE_SCENE
        self._protection_contract=None
        self.executed=0
    def step(self, action):
        self.executed+=1
        self.actions.append(action)


class MonitorTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(dir=ROOT/'testcases/outputs/rl-readiness-20261022')
        self.reference=[{'action':{'operation':'inspect','slot':'left','fixtureOrdinal':i}} for i in range(73)]
        self.now=[0.];self.rss=[0]
        self.guard=VascularPlaybackMonitor(FakeEngine, self.temp.name, reference=self.reference,clock=lambda:self.now[0],rss=lambda:self.rss[0])
        self.guard.case_hash='generated-fixture';self.guard.prefix_started=True
        self.guard.owner=SimpleNamespace(_scene=FAKE_SCENE,_template=SimpleNamespace(_protection_contract=None))
    def tearDown(self):
        self.guard.spool.close();self.temp.cleanup()
    def context(self,i,**extra):
        return {'operation':'surgicalRehearsal','caseHash':'generated-fixture','command':'seekPrefixFrame','branchId':BRANCH,'position':i,'frameIndex':0,**extra}
    def test_actual_monitor_permits_exact73_generated_entries_and_refuses74_before_body(self):
        engine=FakeEngine()
        with self.assertRaises(ForbiddenPhysicsStep):
            with self.guard:
                for i in range(73):
                    self.guard.context=self.context(i);engine.step(self.reference[i]['action'])
                self.guard.context=self.context(73);engine.step({'operation':'inspect'})
        self.assertEqual(engine.executed,73);self.assertEqual(len(self.guard.calls),73);self.assertEqual(len(self.guard.forbidden),1)
    def test_actual_monitor_refuses_altered_action_branch_cursor_and_warm_physics_before_body(self):
        for ordinal,(extra,action) in enumerate([({'branchId':'other'},self.reference[0]['action']),({'frameIndex':1},self.reference[0]['action']),({}, {'operation':'inspect','slot':'left','fixtureOrdinal':False})]):
            with self.subTest(extra=extra,action=action):
                engine=FakeEngine();self.guard.context=self.context(0,**extra)
                with self.assertRaises(ForbiddenPhysicsStep):
                    with self.guard:engine.step(action)
                self.assertEqual(engine.executed,0)
                # One monitor is intentionally not reusable; create a fresh one.
                self.guard.spool.close()
                sub=Path(self.temp.name)/str(ordinal);sub.mkdir(exist_ok=True)
                self.guard=VascularPlaybackMonitor(FakeEngine,sub,reference=self.reference,clock=lambda:self.now[0],rss=lambda:self.rss[0]);self.guard.case_hash='generated-fixture';self.guard.prefix_started=True
                self.guard.owner=SimpleNamespace(_scene=FAKE_SCENE,_template=SimpleNamespace(_protection_contract=None))
    def test_scene_and_protection_owner_substitution_fail_before_step(self):
        engine=FakeEngine();engine.scene=object();self.guard.context=self.context(0)
        with self.assertRaises(ForbiddenPhysicsStep):
            with self.guard:engine.step(self.reference[0]['action'])
        self.assertEqual(engine.executed,0)
    def test_execute_wrapper_rejects_unexpected_command_before_delegate(self):
        class Session:
            def __init__(self):self.executed=0
            def execute(self,*args):self.executed+=1;return {}
            def close(self):pass
        self.guard.install(Session);session=Session()
        with self.assertRaises(ValueError):session.execute('surgicalRehearsal',{'caseHash':'generated-fixture','command':'inspect'},None,None)
        self.assertEqual(session.executed,0);self.assertTrue(self.guard.failure['beforeDispatch'])
    def test_periodic_watchdog_stops_a_long_generated_body(self):
        # Child exits fail-closed from the watchdog; no engine/patient method runs.
        output=Path(self.temp.name)/'watchdog'
        code="""
import time,threading
from types import SimpleNamespace
from vascular_playback_monitor import VascularPlaybackMonitor,MAX_RSS
class Fixture:
    def step(self,action):pass
rss=[0]
g=VascularPlaybackMonitor(Fixture,OUTPUT,reference=[{'action':{}}]*73,rss=lambda:rss[0])
def grow():
    time.sleep(.05);rss[0]=MAX_RSS+1
threading.Thread(target=grow,daemon=True).start()
with g:time.sleep(2)
raise RuntimeError('watchdog failed to stop body')
""".replace('OUTPUT',repr(str(output)))
        child=subprocess.run([sys.executable,'-c',code],cwd=ROOT,env={**os.environ,'PYTHONPATH':str(ROOT/'desktop/electron')+os.pathsep+str(ROOT/'src')},capture_output=True,timeout=4)
        self.assertEqual(child.returncode,70,child.stderr.decode())
        receipt=json.loads((output/'physics-guard.json').read_text());self.assertTrue(receipt['failure']['watchdog']);self.assertGreater(receipt['processPeakResidentBytes'],MAX_RSS)
    def test_watchdog_still_hard_exits_when_receipt_io_fails(self):
        output=Path(self.temp.name)/'failed-receipt'
        code="""
import time,threading
import vascular_playback_monitor as module
class Fixture:
    def step(self,action):pass
rss=[0]
g=module.VascularPlaybackMonitor(Fixture,OUTPUT,reference=[{'action':{}}]*73,rss=lambda:rss[0])
def broken(*args):raise OSError('generated disk failure')
module.atomic_json=broken
def grow():time.sleep(.05);rss[0]=module.MAX_RSS+1
threading.Thread(target=grow,daemon=True).start()
with g:time.sleep(2)
raise RuntimeError('failed receipt disabled watchdog')
""".replace('OUTPUT',repr(str(output)))
        child=subprocess.run([sys.executable,'-c',code],cwd=ROOT,env={**os.environ,'PYTHONPATH':str(ROOT/'desktop/electron')+os.pathsep+str(ROOT/'src')},capture_output=True,timeout=4)
        self.assertEqual(child.returncode,70,child.stderr.decode())
    def test_effective_activation_includes_short_apply_and_parked_other_hand(self):
        state={'slots':{'left':{'active':False,'instrument':'drill'},'right':{'active':True,'instrument':'probe'}}};tools={'drill':{'effect':'cut'},'probe':{'effect':'probe'}}
        self.assertEqual(observed_activation(state,{'slot':'left','operation':'apply'},tools),{'left':True,'right':True})
        self.assertEqual(observed_activation(state,{'slot':'left','operation':'inspect'},tools),{'left':False,'right':True})
        self.assertEqual(observed_activation(state,{'slot':'right','operation':'inspect'},tools),{'left':False,'right':True})
    def test_field_tokens_are_independently_bound_to_decoded_heavy_field(self):
        import hashlib
        state={'flapComponents':[{'cells':[1,2],'offset':[0.,0.,0.]}]}
        digest=hashlib.sha256(json.dumps(state['flapComponents'],sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
        checked_field_tokens(state,{'flapComponents':digest});checked_field_tokens({}, {})
        for tokens in ({},{'flapComponents':'0'*64},{'other':digest},None):
            with self.subTest(tokens=tokens),self.assertRaises(ValueError):checked_field_tokens(state,tokens)
        with self.assertRaises(ValueError):checked_field_tokens({}, {'flapComponents':digest})
    def test_dispatch_guard_rejects_canonical_training_branch_and_skipped_cursor(self):
        for operation,args in [('trainPatient',{}),('surgicalRehearsal',{'command':'inspect','caseHash':'generated-fixture'}),('surgicalRehearsal',{'command':'seekPrefixFrame','caseHash':'generated-fixture','branchId':'other','position':0,'frameIndex':0}),('surgicalRehearsal',{'command':'seekPrefixFrame','caseHash':'generated-fixture','branchId':BRANCH,'position':1,'frameIndex':0})]:
            with self.subTest(args=args),self.assertRaises(ValueError):self.guard.admit_command(operation,args)
        self.guard.admit_command('surgicalRehearsal',{'command':'seekPrefixFrame','caseHash':'generated-fixture','branchId':BRANCH,'position':0,'frameIndex':0})
    def test_wall_and_whole_rss_fail_before_next_step(self):
        engine=FakeEngine();self.guard.context=self.context(0);self.now[0]=600
        with self.assertRaises(ValueError):self.guard.admit_step(engine,self.reference[0]['action'])
        self.now[0]=0;self.rss[0]=MAX_RSS+1
        with self.assertRaises(ValueError):self.guard.admit_step(engine,self.reference[0]['action'])
        self.assertEqual(engine.executed,0);self.assertEqual(self.guard.calls,[])
    def test_frozen_action_json_identity_preserves_type_distinctions(self):
        self.assertEqual(canonical_json(MappingProxyType({'path':((1.,2.,3.),)})),canonical_json({'path':[[1.,2.,3.]]}))
        self.assertNotEqual(canonical_json({'x':False}),canonical_json({'x':0}))


class SpoolTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(dir=ROOT/'testcases/outputs/rl-readiness-20261022')
    def tearDown(self):self.temp.cleanup()
    def frame(self,state):return {'state':zlib.compress(json.dumps(state).encode()),'stateId':'generated','kind':'action_start','observedTimeSeconds':0.,'effectiveActivation':{'left':False},'fieldTokens':{}}
    def test_within_run_dedup_offsets_and_full_unprojected_receipt(self):
        spool=FrameSpool(self.temp.name);frame=self.frame({'generated':True});receipt={'action':{'path':[[1.,2.,3.]]},'events':[{'kind':'drill_work','inputEnergyJ':1.}], 'deltas':{'removedCells':[3]}}
        spool.append(0,[frame,frame],receipt);spool.append(1,[frame],receipt);spool.close()
        rows=[json.loads(line) for line in (Path(self.temp.name)/'observed-actions.jsonl').read_text().splitlines()]
        self.assertEqual(spool.size,len(frame['state']));self.assertEqual(spool.frames,3);self.assertEqual(rows[0]['receipt'],receipt)
        self.assertEqual(rows[0]['frames'][0]['blob'],rows[1]['frames'][0]['blob'])
    def test_both_caps_admitted_before_any_write_and_failure_keeps_prior_commit(self):
        spool=FrameSpool(self.temp.name,max_bytes=1024,max_manifest=1)
        with self.assertRaises(ValueError):spool.append(0,[self.frame({'large':'x'*100})],{})
        self.assertEqual(spool.data.tell(),0);self.assertEqual(spool.manifest.tell(),0);spool.close()
        other=Path(self.temp.name)/'other';spool=FrameSpool(other,max_bytes=1)
        with self.assertRaises(ValueError):spool.append(0,[self.frame({})],{})
        self.assertEqual(spool.data.tell(),0);self.assertEqual(spool.actions,0);spool.close()


if __name__=='__main__':unittest.main()

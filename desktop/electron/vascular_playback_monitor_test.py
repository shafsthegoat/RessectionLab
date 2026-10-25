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
    NativeJsonAdmissionCounter, ForbiddenPhysicsStep, MAX_RSS, ROOT, canonical_json,
    observed_activation, checked_field_tokens, initialize_guard_inventory, checked_prepared_progress)


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
    def test_mixed_mutable_and_frozen_action_ancestry_canonicalizes_recursively(self):
        mixed=[MappingProxyType({'operation':'apply','path':[MappingProxyType({'position':(1.,2.,3.),'active':False})]}),{'nested':[MappingProxyType({'flag':True,'count':0})]}]
        plain=[{'operation':'apply','path':[{'position':[1.,2.,3.],'active':False}]},{'nested':[{'flag':True,'count':0}]}]
        self.assertEqual(canonical_json(mixed),canonical_json(plain))
        altered=[{'operation':'apply','path':[{'position':[1.,2.,3.],'active':0}]},{'nested':[{'flag':True,'count':0}]}]
        self.assertNotEqual(canonical_json(mixed),canonical_json(altered))
    def test_actual_preimport_initialization_keeps_inventory_strict(self):
        from resectionlab.surgical_engine import SurgicalEngine
        modules=initialize_guard_inventory()
        self.assertIn('resectionlab.surgical_archive_view',[module.__name__ for module in modules])
        guard=VascularPlaybackMonitor(SurgicalEngine,Path(self.temp.name)/'actual-imports',reference=self.reference,rss=lambda:0)
        try:
            with guard:
                initialize_guard_inventory()
                guard.check()
            late=VascularPlaybackMonitor(SurgicalEngine,Path(self.temp.name)/'late-class',reference=self.reference,rss=lambda:0)
            try:
                with self.assertRaisesRegex(RuntimeError,'inventory changed'):
                    with late:
                        class GeneratedLateSubclass(SurgicalEngine):pass
            finally:late.spool.close()
        finally:guard.spool.close()
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


class PreparedOnlyMonitorTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(dir=ROOT/'testcases/outputs/rl-readiness-20261022')
        def try_encode(value,limit):return value
        def try_encode_canonical(value,limit):return value
        self.adapter=SimpleNamespace(try_encode=try_encode,try_encode_canonical=try_encode_canonical)
        self.guard=VascularPlaybackMonitor(FakeEngine,self.temp.name,reference=[{'action':{}}]*73,
            rss=lambda:0,prepared_only=True,_json_adapter=self.adapter)
        self.guard.case_hash='generated';self.guard.prefix_started=True
    def tearDown(self):
        self.guard.spool.close();self.temp.cleanup()
    def args(self,position=0,frame_index=0):
        return {'caseHash':'generated','command':'seekPrefixFrame','branchId':BRANCH,'position':position,'frameIndex':frame_index}
    def test_prepared_only_hard_denies_first_step_before_body_and_cleans_all_three_tools(self):
        engine=FakeEngine()
        with self.assertRaisesRegex(ForbiddenPhysicsStep,'Prepared-only'):
            with self.guard:
                tools=[self.guard.tool,self.guard.json_counter.tool,self.guard.canonical_json_counter.tool]
                self.adapter.try_encode('insert',20);self.adapter.try_encode_canonical('canonical',20)
                self.guard.context={'operation':'surgicalRehearsal',**self.args()}
                engine.step({})
        self.assertEqual(engine.executed,0);self.assertEqual(self.guard.calls,[])
        self.assertEqual(len(self.guard.forbidden),1);self.assertEqual(self.guard.attempts['preparedOnlyPublicStep'],1)
        for tool in tools:self.assertIsNone(sys.monitoring.get_tool(tool))
        receipt=self.guard.receipt();self.assertEqual(receipt['qualificationMode'],'prepared-only')
        self.assertEqual(receipt['nativeJson']['total']['admittedStrings'],1)
        self.assertEqual(receipt['nativeCanonicalJson']['total']['admittedStrings'],1)
    def test_prepared_only_resume_is_always_forbidden(self):
        with self.assertRaises(ForbiddenPhysicsStep):self.guard.admit_step(FakeEngine(),{},resumed=True)
        self.assertEqual(self.guard.forbidden[0]['event'],'resume');self.assertEqual(self.guard.calls,[])
    def test_prepared_start_requires_true_flag_and_frames_cannot_change_mode(self):
        start={'caseHash':'generated','command':'seekPrefix','branchId':BRANCH,'position':0}
        for value in (None,False,1,'true'):
            self.guard.prefix_started=False
            with self.subTest(value=value),self.assertRaises(ValueError):
                self.guard.admit_command('surgicalRehearsal',{**start,**({} if value is None else {'cacheOnly':value})})
            self.assertFalse(self.guard.prefix_started)
        self.guard.admit_command('surgicalRehearsal',{**start,'cacheOnly':True})
        for value in (True,False):
            with self.assertRaises(ValueError):self.guard.admit_command('surgicalRehearsal',{**self.args(),'cacheOnly':value})
    def test_loaded_action_cursor_advances_only_contiguously_at_frame_zero_without_calls(self):
        self.guard.admit_command('surgicalRehearsal',self.args())
        for position,index in ((0,1),(1,0)):
            with self.assertRaises(ValueError):self.guard.admit_command('surgicalRehearsal',self.args(position,index))
        self.guard.spool.actions=1  # Scripted committed package, no engine/state.
        self.guard.admit_command('surgicalRehearsal',self.args(0,1))
        self.guard.admit_command('surgicalRehearsal',self.args(1,0))
        for position,index in ((1,1),(2,0)):
            with self.assertRaises(ValueError):self.guard.admit_command('surgicalRehearsal',self.args(position,index))
        self.assertEqual(self.guard.calls,[])
    def test_prepared_progress_rejects_authority_counts_misses_and_publications(self):
        cache={'mode':'only','loadedActions':1,'publishedActions':0,'misses':0,'lastSkip':None}
        self.assertEqual(checked_prepared_progress({'preparedCache':cache},1,True),cache)
        for patch in ({'mode':'prepare'},{'loadedActions':0},{'loadedActions':True},{'publishedActions':1},{'misses':1},{'lastSkip':'unavailable'}):
            with self.subTest(patch=patch),self.assertRaises(ValueError):checked_prepared_progress({'preparedCache':{**cache,**patch}},1,True)
        fresh={**cache,'mode':'prepare','loadedActions':0,'publishedActions':1}
        checked_prepared_progress({'preparedCache':fresh},1,False)
        with self.assertRaises(ValueError):checked_prepared_progress({'preparedCache':{**fresh,'loadedActions':1}},1,False)
    def test_default_fresh_start_still_rejects_prepared_mode_flag(self):
        self.guard.prepared_only=False;self.guard.prefix_started=False
        with self.assertRaises(ValueError):self.guard.admit_command('surgicalRehearsal',{'caseHash':'generated','command':'seekPrefix','branchId':BRANCH,'position':0,'cacheOnly':True})
        self.assertFalse(self.guard.prefix_started)
        self.guard.admit_command('surgicalRehearsal',{'caseHash':'generated','command':'seekPrefix','branchId':BRANCH,'position':0})
    def test_nonboolean_monitor_mode_is_rejected_before_output_creation(self):
        path=Path(self.temp.name)/'bad-mode'
        with self.assertRaises(ValueError):VascularPlaybackMonitor(FakeEngine,path,prepared_only=1)
        self.assertFalse(path.exists())


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


class NativeJsonCounterTests(unittest.TestCase):
    def adapter(self):
        def try_encode(value, raw_limit):
            return value
        def try_encode_canonical(value, raw_limit):
            return value
        return SimpleNamespace(try_encode=try_encode,try_encode_canonical=try_encode_canonical)

    def test_exact_return_counts_are_attributed_without_saving_payloads_or_calling_encoder(self):
        adapter=self.adapter();counter=NativeJsonAdmissionCounter(adapter)
        with counter:
            adapter.try_encode('outside',20)
            previous=counter.begin_command()
            adapter.try_encode('large payload content',30);adapter.try_encode(None,30)
            first=counter.end_command(previous)
            previous=counter.begin_command();adapter.try_encode('',1);second=counter.end_command(previous)
        self.assertEqual(first,{'normalReturns':2,'admittedStrings':1,'fallbackNone':1,'returnedCharacters':21,'maxReturnedCharacters':21})
        self.assertEqual(second['admittedStrings'],1)
        receipt=counter.receipt()
        self.assertEqual(receipt['total']['normalReturns'],4);self.assertEqual(receipt['outsideCommand']['admittedStrings'],1)
        self.assertNotIn('large payload content',json.dumps(receipt));self.assertIsNone(counter.current)
        self.assertIsNone(counter.tool);self.assertFalse(receipt['active'])

    def test_exceptional_exit_is_not_counted_as_none_fallback(self):
        def try_encode(value, raw_limit):
            raise ValueError('generated encoder failure')
        counter=NativeJsonAdmissionCounter(SimpleNamespace(try_encode=try_encode))
        with counter:
            with self.assertRaisesRegex(ValueError,'generated encoder failure'):try_encode(None,10)
        self.assertEqual(counter.total['normalReturns'],0)
        self.assertEqual(counter.total['fallbackNone'],0)

    def test_unexpected_return_type_fails_closed_and_cleans_up(self):
        adapter=self.adapter();counter=NativeJsonAdmissionCounter(adapter)
        with self.assertRaisesRegex(ValueError,'previously failed'):
            with counter:
                with self.assertRaisesRegex(ValueError,'Unexpected native JSON'):adapter.try_encode(b'bytes',20)
        self.assertIn('Unexpected',counter.failure['message']);self.assertIsNone(counter.tool)

    def test_local_event_and_callback_drift_are_refused_and_cleaned(self):
        for kind in ('events','callback'):
            with self.subTest(kind=kind):
                counter=NativeJsonAdmissionCounter(self.adapter())
                with self.assertRaises(ValueError):
                    with counter:
                        tool=counter.tool
                        if kind=='events':sys.monitoring.set_local_events(tool,counter.code,0)
                        else:sys.monitoring.register_callback(tool,sys.monitoring.events.PY_RETURN,lambda *_:None)
                        counter.check()
                self.assertIsNone(sys.monitoring.get_tool(tool));self.assertIsNotNone(counter.failure)

    def test_function_and_loaded_code_identity_drift_are_refused(self):
        for kind in ('function','code'):
            with self.subTest(kind=kind):
                adapter=self.adapter();counter=NativeJsonAdmissionCounter(adapter)
                def replacement(value, raw_limit):return None
                with self.assertRaisesRegex(ValueError,'function/code changed'):
                    with counter:
                        if kind=='function':adapter.try_encode=replacement
                        else:adapter.try_encode.__code__=replacement.__code__
                        counter.check()
                self.assertIsNone(counter.tool)

    def test_event_constant_drift_is_refused_with_original_event_cleanup(self):
        counter=NativeJsonAdmissionCounter(self.adapter());events=sys.monitoring.events;original=events.PY_RETURN
        try:
            with self.assertRaisesRegex(ValueError,'event constants changed'):
                with counter:
                    tool=counter.tool;events.PY_RETURN=0;counter.check()
            self.assertIsNone(counter.tool);self.assertIsNone(sys.monitoring.get_tool(tool))
        finally:events.PY_RETURN=original

    def test_counter_ownership_drift_does_not_clobber_replacement_owner(self):
        counter=NativeJsonAdmissionCounter(self.adapter());tool=None
        try:
            with self.assertRaisesRegex(ValueError,'ownership/events changed'):
                with counter:
                    tool=counter.tool;sys.monitoring.set_local_events(tool,counter.code,0)
                    sys.monitoring.register_callback(tool,sys.monitoring.events.PY_RETURN,None)
                    sys.monitoring.free_tool_id(tool);sys.monitoring.use_tool_id(tool,'generated-replacement-owner')
                    counter.check()
            self.assertEqual(sys.monitoring.get_tool(tool),'generated-replacement-owner')
        finally:
            if tool is not None and sys.monitoring.get_tool(tool)=='generated-replacement-owner':sys.monitoring.free_tool_id(tool)

    def test_startup_exhaustion_fails_without_leaking_a_slot(self):
        acquired=[]
        try:
            for tool in range(6):
                if sys.monitoring.get_tool(tool) is None:
                    sys.monitoring.use_tool_id(tool,'generated-occupied-'+str(tool));acquired.append(tool)
            counter=NativeJsonAdmissionCounter(self.adapter())
            with self.assertRaisesRegex(ValueError,'No native JSON monitoring tool'):counter.__enter__()
            self.assertIsNone(counter.tool);self.assertIsNotNone(counter.failure)
        finally:
            for tool in acquired:sys.monitoring.free_tool_id(tool)

    def test_global_step_guard_stays_intact_with_local_return_counter(self):
        from vascular_playback_monitor import ReadOnlyStepMonitor
        class GeneratedEngine:
            def step(self, action):raise AssertionError('forbidden body entered')
        adapter=self.adapter();counter=NativeJsonAdmissionCounter(adapter)
        guard=ReadOnlyStepMonitor(GeneratedEngine)
        with self.assertRaises(ForbiddenPhysicsStep):
            with guard:
                with counter:
                    self.assertNotEqual(guard.tool,counter.tool)
                    self.assertEqual(sys.monitoring.get_events(guard.tool),guard.flags)
                    adapter.try_encode('observed',20);guard.check();counter.check()
                    GeneratedEngine().step({})
        self.assertEqual(counter.total['admittedStrings'],1);self.assertIsNone(counter.tool);self.assertIsNone(guard.tool)

    def test_guard_report_records_counter_drift_as_failure(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'testcases/outputs/rl-readiness-20261022') as directory:
            guard=VascularPlaybackMonitor(FakeEngine,directory,reference=[{'action':{}}]*73,rss=lambda:0,_json_adapter=self.adapter())
            try:
                with self.assertRaises(ValueError):
                    with guard:
                        sys.monitoring.set_local_events(guard.json_counter.tool,guard.json_counter.code,0)
                        guard.resources()
                self.assertTrue(guard.failure['nativeJsonCounter']);self.assertIsNotNone(guard.receipt()['nativeJson']['failure'])
                self.assertIsNone(guard.tool);self.assertIsNone(guard.json_counter.tool)
            finally:guard.spool.close()

    def test_guard_exit_preserves_body_error_after_recording_counter_failure(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'testcases/outputs/rl-readiness-20261022') as directory:
            guard=VascularPlaybackMonitor(FakeEngine,directory,reference=[{'action':{}}]*73,rss=lambda:0,_json_adapter=self.adapter())
            original=RuntimeError('original command failure')
            try:
                with self.assertRaises(RuntimeError) as caught:
                    with guard:
                        sys.monitoring.set_local_events(guard.json_counter.tool,guard.json_counter.code,0)
                        raise original
                self.assertIs(caught.exception,original)
                self.assertTrue(guard.failure['nativeJsonCounter'])
                self.assertIsNotNone(guard.json_counter.failure)
                self.assertIsNone(guard.tool);self.assertIsNone(guard.json_counter.tool)
            finally:guard.spool.close()

    def test_guard_exit_without_body_error_still_refuses_counter_failure(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'testcases/outputs/rl-readiness-20261022') as directory:
            guard=VascularPlaybackMonitor(FakeEngine,directory,reference=[{'action':{}}]*73,rss=lambda:0,_json_adapter=self.adapter())
            try:
                with self.assertRaisesRegex(ValueError,'ownership/events changed'):
                    with guard:
                        sys.monitoring.set_local_events(guard.json_counter.tool,guard.json_counter.code,0)
                self.assertTrue(guard.failure['nativeJsonCounter'])
                self.assertIsNone(guard.tool);self.assertIsNone(guard.json_counter.tool)
            finally:guard.spool.close()

    def test_execute_wrapper_records_per_command_admission_and_fallback(self):
        adapter=self.adapter()
        class Session:
            def execute(self, operation, args, request, progress):
                adapter.try_encode('native encoded',30);adapter.try_encode(None,30)
                adapter.try_encode_canonical('canonical',30)
                return {'caseHash':'generated'}
            def close(self):pass
        with tempfile.TemporaryDirectory(dir=ROOT/'testcases/outputs/rl-readiness-20261022') as directory:
            guard=VascularPlaybackMonitor(FakeEngine,directory,reference=[{'action':{}}]*73,rss=lambda:0,_json_adapter=adapter)
            guard.install(Session)
            try:
                with guard:Session().execute('ping',{},None,None)
                self.assertEqual(guard.commands[0]['nativeJson']['admittedStrings'],1)
                self.assertEqual(guard.commands[0]['nativeJson']['fallbackNone'],1)
                self.assertEqual(guard.receipt()['nativeJson']['total']['normalReturns'],2)
                self.assertEqual(guard.commands[0]['nativeCanonicalJson']['admittedStrings'],1)
                self.assertEqual(guard.receipt()['nativeCanonicalJson']['total']['returnedCharacters'],9)
            finally:guard.spool.close()

    def test_explicit_entrypoints_have_independent_local_counts_and_binding(self):
        adapter=self.adapter()
        insertion=NativeJsonAdmissionCounter(adapter)
        canonical=NativeJsonAdmissionCounter(adapter,'try_encode_canonical')
        with insertion,canonical:
            adapter.try_encode('insert',10)
            adapter.try_encode_canonical('sorted',10);adapter.try_encode_canonical(None,10)
            self.assertNotEqual(insertion.tool,canonical.tool)
            insertion.check();canonical.check()
        self.assertEqual(insertion.total['normalReturns'],1)
        self.assertEqual(canonical.total['normalReturns'],2)
        self.assertEqual(canonical.total['fallbackNone'],1)
        self.assertEqual(canonical.receipt()['entrypoint'],'try_encode_canonical')
        with self.assertRaisesRegex(ValueError,'Unsupported'):
            NativeJsonAdmissionCounter(adapter,'other')

    def test_warm_command_canonical_hash_is_not_counted_as_insertion_encoding(self):
        adapter=self.adapter()
        class Session:
            def execute(self,operation,args,request,progress):
                if args.get('cold'):adapter.try_encode('insertion',30)
                adapter.try_encode_canonical('canonical hash input',30)
                return {'caseHash':'generated'}
            def close(self):pass
        with tempfile.TemporaryDirectory(dir=ROOT/'testcases/outputs/rl-readiness-20261022') as directory:
            guard=VascularPlaybackMonitor(FakeEngine,directory,reference=[{'action':{}}]*73,rss=lambda:0,_json_adapter=adapter)
            guard.install(Session)
            try:
                with guard:
                    session=Session()
                    session.execute('ping',{'cold':True},None,None)
                    session.execute('ping',{'cold':False},None,None)
                cold,warm=guard.commands
                self.assertEqual(cold['nativeJson']['admittedStrings'],1)
                self.assertEqual(warm['nativeJson']['normalReturns'],0)
                self.assertEqual(warm['nativeCanonicalJson']['admittedStrings'],1)
                self.assertEqual(guard.receipt()['nativeCanonicalJson']['total']['admittedStrings'],2)
            finally:guard.spool.close()

    def test_either_entrypoint_drift_is_recorded_and_all_guard_slots_are_released(self):
        for name in ('try_encode','try_encode_canonical'):
            with self.subTest(entrypoint=name),tempfile.TemporaryDirectory(dir=ROOT/'testcases/outputs/rl-readiness-20261022') as directory:
                adapter=self.adapter()
                guard=VascularPlaybackMonitor(FakeEngine,directory,reference=[{'action':{}}]*73,rss=lambda:0,_json_adapter=adapter)
                try:
                    with self.assertRaisesRegex(ValueError,'function/code changed'):
                        with guard:
                            tools=[guard.tool,guard.json_counter.tool,guard.canonical_json_counter.tool]
                            setattr(adapter,name,lambda *_:None)
                            guard.resources()
                    self.assertEqual(guard.failure['entrypoint'],name)
                    for tool in tools:self.assertIsNone(sys.monitoring.get_tool(tool))
                    other=guard.canonical_json_counter if name=='try_encode' else guard.json_counter
                    self.assertIsNone(other.failure)
                finally:guard.spool.close()

    def test_second_counter_slot_exhaustion_releases_first_and_global_guard(self):
        acquired=[]
        try:
            for tool in range(6):
                if sys.monitoring.get_tool(tool) is None:
                    sys.monitoring.use_tool_id(tool,'generated-occupied-'+str(tool));acquired.append(tool)
            # Leave exactly two slots: global guard and insertion counter.
            for tool in acquired[-2:]:sys.monitoring.free_tool_id(tool)
            held=acquired[:-2]
            with tempfile.TemporaryDirectory(dir=ROOT/'testcases/outputs/rl-readiness-20261022') as directory:
                guard=VascularPlaybackMonitor(FakeEngine,directory,reference=[{'action':{}}]*73,rss=lambda:0,_json_adapter=self.adapter())
                try:
                    with self.assertRaisesRegex(ValueError,'No native JSON monitoring tool'):guard.__enter__()
                    self.assertEqual(guard.failure['entrypoint'],'try_encode_canonical')
                    self.assertIsNone(guard.tool);self.assertIsNone(guard.json_counter.tool);self.assertIsNone(guard.canonical_json_counter.tool)
                    for tool in acquired[-2:]:self.assertIsNone(sys.monitoring.get_tool(tool))
                    for tool in held:self.assertEqual(sys.monitoring.get_tool(tool),'generated-occupied-'+str(tool))
                finally:guard.spool.close()
        finally:
            for tool in acquired:
                if sys.monitoring.get_tool(tool)=='generated-occupied-'+str(tool):sys.monitoring.free_tool_id(tool)

    def test_both_counters_coexist_with_unchanged_global_step_rejection(self):
        from vascular_playback_monitor import ReadOnlyStepMonitor
        class GeneratedEngine:
            def step(self,action):raise AssertionError('forbidden body entered')
        adapter=self.adapter();insertion=NativeJsonAdmissionCounter(adapter);canonical=NativeJsonAdmissionCounter(adapter,'try_encode_canonical')
        guard=ReadOnlyStepMonitor(GeneratedEngine)
        with self.assertRaises(ForbiddenPhysicsStep):
            with guard,insertion,canonical:
                self.assertEqual(len({guard.tool,insertion.tool,canonical.tool}),3)
                self.assertEqual(sys.monitoring.get_events(guard.tool),guard.flags)
                adapter.try_encode('insert',10);adapter.try_encode_canonical('sorted',10)
                guard.check();insertion.check();canonical.check()
                GeneratedEngine().step({})
        self.assertEqual(insertion.total['admittedStrings'],1);self.assertEqual(canonical.total['admittedStrings'],1)
        self.assertIsNone(guard.tool);self.assertIsNone(insertion.tool);self.assertIsNone(canonical.tool)

    def test_cleanup_failure_does_not_leak_other_slots_or_mask_body_error(self):
        for body_error in (False,True):
            with self.subTest(bodyError=body_error),tempfile.TemporaryDirectory(dir=ROOT/'testcases/outputs/rl-readiness-20261022') as directory:
                guard=VascularPlaybackMonitor(FakeEngine,directory,reference=[{'action':{}}]*73,rss=lambda:0,_json_adapter=self.adapter())
                original=RuntimeError('original body failure')
                try:
                    with self.assertRaisesRegex(RuntimeError,'original body failure' if body_error else 'generated cleanup failure') as caught:
                        with guard:
                            tools=[guard.tool,guard.json_counter.tool,guard.canonical_json_counter.tool]
                            close=guard.canonical_json_counter.close
                            def failing_close():
                                close()
                                raise RuntimeError('generated cleanup failure')
                            guard.canonical_json_counter.close=failing_close
                            if body_error:raise original
                    if body_error:self.assertIs(caught.exception,original)
                    self.assertEqual(guard.failure['entrypoint'],'try_encode_canonical')
                    self.assertTrue(guard.failure['cleanup'])
                    for tool in tools:self.assertIsNone(sys.monitoring.get_tool(tool))
                finally:guard.spool.close()


if __name__=='__main__':unittest.main()

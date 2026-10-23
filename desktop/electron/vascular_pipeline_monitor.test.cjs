'use strict';
const test = require('node:test'), assert = require('node:assert/strict');
const { spawnSync } = require('node:child_process');
const fs = require('node:fs'), path = require('node:path');
const root = path.resolve(__dirname, '../..'), python = path.join(root, '.venv/bin/python');

test('pipeline vascular role guards and metadata-only proof reject generated tampering without patient physics',
  { skip: !fs.existsSync(python), timeout: 65000 }, () => {
    const probe = spawnSync(python, ['-c', String.raw`
import copy, hashlib, json, sys, tempfile, unittest
from pathlib import Path
from types import MappingProxyType, SimpleNamespace
sys.path[:0] = [${JSON.stringify(__dirname)}, ${JSON.stringify(path.join(root, 'src'))}]
import vascular_pipeline_monitor as p
import vascular_playback_monitor as b

D = 'sha256:' + 'a'*64
BINDING = dict(schema='resectionlab.prepared-prefix-binding.v1', sceneId=D,
    originSnapshotHash=D, actionLogHash=D, protectionContractHash=D,
    implementation='b'*64, prefixImplementation='c'*64)
class Engine:
    def __init__(self): self.actions=[]; self.executed=0
    def step(self, action): self.executed+=1; self.actions.append(action)
def owner(origin='fresh-public-actions', current=True):
    value = {'sceneId':D,'protectionContractHash':D,'expectedEndpointStateId':D,
        'verification':{k:BINDING[k] for k in ('originSnapshotHash','actionLogHash','implementation','prefixImplementation')},
        'observationOrigin':origin,'preparedCacheDirectory':'/new','preparedCacheNamespace':'/new/namespace',
        'sourceArchiveSha256':b.NATIVE_SHA,'canonicalReferenceSha256':'d'*64}
    if current: value['preparedBinding']=dict(BINDING)
    else:
        value['verification']['implementation']='e'*64
        value['verification']['prefixImplementation']='f'*64
        value['preparedCacheDirectory']='/old';value['preparedCacheNamespace']='/old/namespace'
    return value
def frames():
    return [{'state':('committed-byte-fixture-'+str(i)).encode(), 'stateId':D,
        'kind':'action_start' if i==0 else 'action_boundary' if i==2 else 'microstep',
        'observedTimeSeconds':float(i),'effectiveActivation':{'left':False},
        'fieldTokens':{'flapComponents':'a'*64},
        'flowDisplay':{'fixture':'complete metadata', 'operations':[{'amount':i,'sample':True}]}}
        for i in range(3)]
def manifests(root):
    source=frames(); producer=p.DigestFrameSpool(root/'producer');producer.owner_metadata=owner()
    receipt=MappingProxyType({'action':{'fixture':1},'stateBefore':D,'stateAfter':D,
                              'nested':{'types':[True,1,1.0]},'events':[]})
    producer.append(0,source,receipt);producer.close()
    record=json.loads((root/'producer/observed-actions.jsonl').read_text())
    consumer=copy.deepcopy(record);consumer['owner']=owner('prepared-cache')
    old=copy.deepcopy(record);old['owner']=owner(current=False)
    offset=0
    for c,o in zip(consumer['frames'],old['frames']):
        c['blob']['offset']=offset;o['blob']['offset']=offset;offset+=c['blob']['bytes']
    for folder,row in (('consumer',consumer),('old',old)):
        (root/folder).mkdir();(root/folder/'observed-actions.jsonl').write_bytes(p._json(row)+b'\n')
    return record,consumer,old
def omission_manifests(root, count=200):
    record,consumer,old=manifests(root)
    expanded=[]
    for i in range(count):
        frame=copy.deepcopy(record['frames'][1]);frame['observedTimeSeconds']=float(i)
        frame['flowDisplay']={'schema':'resectionlab.recorded-field-transfers.v1','sceneId':D,
            'actionIndex':0,'frameIndex':i,'beforeStateId':D,'stateId':D,
            'startSeconds':float(max(0,i-1)),'endSeconds':float(i),
            'operations':[],'truncated':False}
        expanded.append(frame)
    record['frames']=expanded;consumer['frames']=copy.deepcopy(expanded);old['frames']=copy.deepcopy(expanded)
    for c,o in zip(consumer['frames'],old['frames']):
        c['blob']['offset']=0;o['blob']['offset']=0;del c['flowDisplay']
    for folder,row in (('producer',record),('consumer',consumer),('old',old)):
        (root/folder/'observed-actions.jsonl').write_bytes(p._json(row)+b'\n')
    return record,consumer,old
def parent_fixture():
    proof={'schema':'resectionlab.vascular-pipeline-parent-proof.v1','passed':True,
        'qualificationMode':'fresh-pipeline','caseHash':D,'branchId':b.BRANCH,'mainPid':99,
        'sourceArchiveSha256':b.NATIVE_SHA,'binding':dict(BINDING),
        'workers':[{'role':role,'pid':101+i,'exited':True,'exitCode':0,'signal':None}
                   for i,role in enumerate(('normal','producer','consumer'))],
        'resources':{'closed':True,'failure':None,'maxAlive':2,'peakAggregatePythonTreeRssBytes':100,
                     'maxWallMs':600000,'wallMs':1000},'leases':[]}
    guards={}
    for i,role in enumerate(('normal','producer','consumer')):
        consumer=role=='consumer';producer=role=='producer';verified=int(role!='normal')
        cache={'mode':'only' if consumer else 'prepare','loadedActions':int(consumer),
               'publishedActions':int(producer),'misses':0,'lastSkip':None}
        guard={'pipelineRole':role,'pipelineSchema':'resectionlab.vascular-pipeline-role.v1','workerPid':101+i,'parentPid':99,'caseHash':D,
            'qualificationMode':'prepared-only' if consumer else 'fresh','failure':None,'forbiddenSteps':[],
            'publicStepAttempts':{},'preparedBinding':dict(BINDING),'processPeakResidentBytes':100,'wallSeconds':1,
            'nativeIdentities':{'idle':{'mode':'native','fixture':'same'},'prefixJson':{'mode':'native'}},
            'ownership':{k:False for k in ('canonicalSessionCreated','replayCacheCreated','frameCacheCreated','frameAdmissionsCreated')},
            'actualStore':{'directory':'/shared','namespaceDirectory':'/shared/namespace','maxDiskBytes':512*p.MIB,
                'maxActionBytes':64*p.MIB,'hits':int(consumer),'publications':int(producer),'misses':0,'lastSkip':None},
            'permittedSteps':[{'operation':'surgicalRehearsal','command':'seekPrefixFrame','branchId':b.BRANCH,
                'caseHash':D,'position':0,'frameIndex':0,'priorActions':0,'event':'entry'}] if producer else [],
            'spool':{'actions':verified,'frames':verified,'manifestBytes':100 if verified else 0,'binaryBytes':100 if consumer else 0}}
        guard['ownership']['prefixStatistics']={'verifiedThrough':verified,'totalActions':1,'preparedCache':cache}
        for key in ('nativeJson','nativeCanonicalJson'):
            guard[key]={'started':True,'active':False,'failure':None,'cleanupFailure':None,'total':{'admittedStrings':1}}
        guards[role]=guard
    for role,start,end in (('producer',10,20),('consumer',21,30)):
        producer=role=='producer'
        proof['leases'].append({'role':role,'index':0,'requestId':role+'-id','start':start,'end':end,
            'verifiedThrough':1,'actionFrameBytes':4096,
            'cache':copy.deepcopy(guards[role]['ownership']['prefixStatistics']['preparedCache']),
            'store':{key:guards[role]['actualStore'][key] for key in ('hits','publications','misses','lastSkip')}})
    identity={'branchId':b.BRANCH,'position':0,'frameIndex':0,'stateId':D}
    ipc={**identity,'command':'seekPrefixFrame'}
    receipt={**ipc,'notificationSerial':1,'receivedAt':101}
    shown={**identity,'receiptNotificationSerial':1,'rendererReceivedAt':101,'firstSubmittedAt':102}
    report={'qualificationMode':'fresh-pipeline','playStarted':1,'playEnded':40,
        'playStartedRenderer':100,'playEndedRenderer':103,'allCompletedPlaybackIpc':[ipc],
        'rendererReceipts':[receipt],'submittedPlayback':[shown]}
    projection=[[[row[key] for key in ('branchId','position','frameIndex','stateId')] for row in rows]
                for rows in (report['allCompletedPlaybackIpc'],report['rendererReceipts'],report['submittedPlayback'])]
    proof['rendererTupleSha256']=hashlib.sha256(json.dumps(projection,separators=(',',':')).encode()).hexdigest()
    return proof,guards,report

class Tests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(dir=b.ROOT/'testcases/outputs/rl-readiness-20261022')
        self.root=Path(self.temp.name)
    def tearDown(self): self.temp.cleanup()
    def test_digest_matches_complete_metadata_without_binary_or_retained_state(self):
        record,_,_=manifests(self.root)
        self.assertFalse((self.root/'producer/observed-states.bin').exists())
        self.assertEqual(record['frames'][1]['blob']['sha256'],hashlib.sha256(frames()[1]['state']).hexdigest())
        result=p.compare_manifests(*(self.root/folder/'observed-actions.jsonl' for folder in ('producer','consumer','old')),_frame_counts=(3,))
        self.assertTrue(result['allCompleteReceiptsDescriptorsFlowAndCompressedShaLengthsExact'])
        self.assertEqual(result['stateDecodes'],0);self.assertEqual(result['observations'],3)
    def test_each_complete_descriptor_receipt_digest_length_and_json_type_is_compared(self):
        _,consumer,_=manifests(self.root);target=self.root/'consumer/observed-actions.jsonl'
        for mutate in (
            lambda row:row['frames'][1]['blob'].__setitem__('sha256','0'*64),
            lambda row:row['frames'][1]['blob'].__setitem__('bytes',99),
            lambda row:row['frames'][1]['fieldTokens'].__setitem__('flapComponents','0'*64),
            lambda row:row['frames'][1]['flowDisplay']['operations'][0].__setitem__('sample',1),
            lambda row:row['receipt']['nested']['types'].__setitem__(0,1),
            lambda row:row['owner'].__setitem__('sourceArchiveSha256','0'*64)):
            row=copy.deepcopy(consumer);mutate(row);target.write_bytes(p._json(row)+b'\n')
            with self.assertRaises(ValueError):
                p.compare_manifests(*(self.root/folder/'observed-actions.jsonl' for folder in ('producer','consumer','old')),_frame_counts=(3,))
    def test_digest_budget_admitted_before_append_and_no_ordinary_spool(self):
        spool=p.DigestFrameSpool(self.root/'tiny',max_manifest=8)
        with self.assertRaises(ValueError):spool.append(0,frames(),{})
        self.assertEqual(spool.statistics['actions'],0);spool.close()
        self.assertEqual((self.root/'tiny/observed-actions.jsonl').stat().st_size,0)
        ordinary=p._NoActionSpool(self.root/'normal')
        with self.assertRaises(ValueError):ordinary.append(0,frames(),{})
        self.assertFalse((self.root/'normal').exists())
    def test_only_proven_empty_whole_action_budget_fallback_is_admitted_and_disclosed(self):
        record,consumer,old=omission_manifests(self.root)
        result=p.compare_manifests(*(self.root/folder/'observed-actions.jsonl' for folder in ('producer','consumer','old')),_frame_counts=(200,))
        self.assertFalse(result['allCompleteReceiptsDescriptorsFlowAndCompressedShaLengthsExact'])
        self.assertFalse(result['rawProducerConsumerFullDescriptorsExact'])
        self.assertTrue(result['allCompleteReceiptsPhysicalDescriptorsFlowOperationsAndCompressedShaLengthsExact'])
        self.assertEqual(result['preparedEmptyEnvelopeOmissions']['count'],200)
        self.assertEqual(result['preparedEmptyEnvelopeOmissions']['intervals'][1]['frameIndex'],1)
        for change in ('truncated','operation','partial','context','physical','digest','length','retained-extra'):
            rows=copy.deepcopy((record,consumer,old));q,c,o=rows
            if change=='truncated':q['frames'][1]['flowDisplay']['truncated']=True
            elif change=='operation':q['frames'][1]['flowDisplay']['operations']=[{'unexpected':True}]
            elif change=='partial':c['frames'][1]['flowDisplay']=copy.deepcopy(q['frames'][1]['flowDisplay'])
            elif change=='context':q['frames'][1]['flowDisplay']['beforeStateId']='sha256:'+'0'*64
            elif change=='physical':c['frames'][1]['effectiveActivation']['left']=True
            elif change=='digest':c['frames'][1]['blob']['sha256']='0'*64
            elif change=='length':c['frames'][1]['blob']['bytes']=99
            elif change=='retained-extra':c['frames'][1]['newField']=False
            for folder,row in zip(('producer','consumer','old'),rows):
                (self.root/folder/'observed-actions.jsonl').write_bytes(p._json(row)+b'\n')
            with self.subTest(change=change),self.assertRaises(ValueError):
                p.compare_manifests(*(self.root/folder/'observed-actions.jsonl' for folder in ('producer','consumer','old')),_frame_counts=(200,))
    def test_empty_envelope_omission_without_budget_exhaustion_is_rejected(self):
        omission_manifests(self.root,count=3)
        with self.assertRaisesRegex(ValueError,'without publisher metadata budget exhaustion'):
            p.compare_manifests(*(self.root/folder/'observed-actions.jsonl' for folder in ('producer','consumer','old')),_frame_counts=(3,))
    def test_base_default_and_fixed_role_factories_keep_exact_caps(self):
        reference=[{'action':{'fixture':i}} for i in range(73)]
        ordinary=b.VascularPlaybackMonitor(Engine,self.root/'old',reference=reference)
        self.assertIsInstance(ordinary.spool,b.FrameSpool);ordinary.spool.close()
        for role in ('normal','producer','consumer'):
            guard=p.PipelineVascularMonitor(Engine,self.root,role=role,reference=reference)
            self.assertEqual(guard.prepared_only,role=='consumer')
            if role=='consumer':self.assertEqual(guard.spool.max_manifest,6*p.MIB)
            self.assertEqual((self.root/'roles'/role/'observed-states.bin').exists(),role=='consumer')
            guard.spool.close()
        self.assertEqual(sum(p.CONTROL_BYTES.values())+p.PARENT_BYTES+2*p.ACTION_MANIFEST_BYTES,16*p.MIB)
    def test_normal_and_consumer_step_guards_refuse_before_generated_body(self):
        reference=[{'action':{'fixture':i}} for i in range(73)]
        for role in ('normal','consumer'):
            engine=Engine();guard=p.PipelineVascularMonitor(Engine,self.root,role=role,reference=reference)
            try:
                with self.assertRaises(b.ForbiddenPhysicsStep):
                    with guard:engine.step(reference[0]['action'])
                self.assertEqual(engine.executed,0)
            finally:guard.spool.close()
    def test_strict_cursor_and_new_write_not_accepted_existing_publication(self):
        reference=[{'action':{'fixture':i}} for i in range(73)]
        guard=p.PipelineVascularMonitor(Engine,self.root,role='producer',reference=reference)
        guard.case_hash=D;guard.prefix_started=True
        with self.assertRaises(ValueError):guard.admit_command('surgicalRehearsal',dict(caseHash=D,command='seekPrefixFrame',branchId=b.BRANCH,position=0,frameIndex=1))
        stored={'diskBytes':1,'maxDiskBytes':512*p.MIB,'maxActionBytes':64*p.MIB,
                'hits':0,'misses':0,'publications':0,'lastSkip':None,'directory':'/shared','namespaceDirectory':'/shared/ns'}
        fake=SimpleNamespace(_prepared_binding=dict(BINDING),_scene=SimpleNamespace(identity=D),
            _origin_hash=D,_action_hash=D,_protection_hash=D,_implementation='b'*64,_prefix_implementation='c'*64,
            _prepared_store=SimpleNamespace(statistics=stored),_history=[{}])
        with self.assertRaisesRegex(ValueError,'fresh writes'):
            guard.capture(SimpleNamespace(surgical_prefix_session=(None,None,b.BRANCH,fake)))
        self.assertEqual(guard.spool.actions,0);guard.spool.close()
    def test_parent_positive_and_exit_native_counter_clock_tuple_tamper(self):
        proof,guards,report=parent_fixture()
        self.assertEqual(p.check_parent_proof(proof,guards,report,_frame_counts=(1,))['consumerPublicSteps'],0)
        for change in ('exit','native','store','overlap','startup','tuple','bool','rss','parent','main-bool'):
            q,g,r=copy.deepcopy((proof,guards,report))
            if change=='exit':q['workers'][0]['exited']=False
            elif change=='native':g['consumer']['nativeIdentities']['idle']['fixture']='other'
            elif change=='store':g['producer']['actualStore']['publications']=0
            elif change=='overlap':q['leases'][1]['start']=19
            elif change=='startup':r['playStarted']=11
            elif change=='tuple':q['rendererTupleSha256']='0'*64
            elif change=='bool':r['submittedPlayback'][0]['position']=False
            elif change=='rss':q['resources']['peakAggregatePythonTreeRssBytes']=b.MAX_RSS+1
            elif change=='parent':g['consumer']['parentPid']=100
            elif change=='main-bool':q['mainPid']=True
            with self.subTest(change=change),self.assertRaises(ValueError):
                p.check_parent_proof(q,g,r,_frame_counts=(1,))

suite=unittest.defaultTestLoader.loadTestsFromTestCase(Tests)
result=unittest.TextTestRunner(verbosity=2).run(suite)
if not result.wasSuccessful():sys.exit(1)
print('generated pipeline monitor checks passed; no patient actions or state decode')
`], { cwd: root, encoding: 'utf8', timeout: 60000, maxBuffer: 8 * 1024 * 1024,
        env: { ...process.env, PYTHONDONTWRITEBYTECODE: '1' } });
    assert.equal(probe.status, 0, probe.stderr || probe.stdout || String(probe.error));
    assert.match(probe.stdout, /generated pipeline monitor checks passed/);
  });

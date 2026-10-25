'use strict';
const test=require('node:test'),assert=require('node:assert/strict');
const {spawnSync}=require('node:child_process');
const path=require('node:path'),fs=require('node:fs');
const root=path.resolve(__dirname,'../..'),python=path.join(root,'.venv/bin/python');
const monitor=path.join(root,'testcases/outputs/full-case-rehearsal-20261020/review');

test('prefix audit permits only three ordered preparations and forbids warm/fourth steps before entry',
  {skip:!fs.existsSync(python)||!fs.existsSync(monitor)},()=>{
  const probe=spawnSync(python,['-c',`
import sys
sys.path[:0]=[${JSON.stringify(__dirname)},${JSON.stringify(monitor)}]
from prefix_playback_monitor import PrefixPlaybackMonitor
from readonly_step_monitor import ForbiddenPhysicsStep
class Engine:
    def __init__(self): self.actions=[]
    def step(self): self.actions.append('generated')
class Session:
    def __init__(self): self.engine=Engine()
    def execute(self,op,args,request,progress):
        for _ in range(args.get('steps',1)): self.engine.step()
    def close(self): pass
guard=PrefixPlaybackMonitor(Engine)
guard.install(Session)
s=Session()
with guard:
    s.execute('surgicalRehearsal',{'command':'seekPrefix','position':2,'steps':2},None,None)
    s.execute('surgicalRehearsal',{'command':'seekPrefixFrame','position':2,'frameIndex':0},None,None)
    s.execute('surgicalRehearsal',{'command':'seekPrefixFrame','position':2,'frameIndex':1,'steps':0},None,None)
    s.close()
assert len(guard.calls)==3 and len(s.engine.actions)==3
assert guard.receipt()['ownership']['beforeClose']
assert not guard.receipt()['ownership']['canonicalSessionCreated']
for args in ({'command':'seekPrefixFrame','position':2,'frameIndex':1},
             {'command':'seekPrefixFrame','position':2,'frameIndex':0}):
    # A new monitored owner with a full three-call audit represents a warm lane.
    g=PrefixPlaybackMonitor(Engine);g.calls=list(guard.calls);g.install(Session)
    try:
        with g: s.execute('surgicalRehearsal',args,None,None)
    except ForbiddenPhysicsStep: pass
    else: raise AssertionError('unexpected physics admitted')
    assert len(s.engine.actions)==3 and len(g.forbidden)==1
print('generated audit checks passed')
`],{cwd:root,encoding:'utf8'});
  assert.equal(probe.status,0,probe.stderr||probe.stdout);
  assert.match(probe.stdout,/generated audit checks passed/);
});

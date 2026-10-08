"""Independent pure/negative controls; no model output generation or native work."""
from pathlib import Path
from copy import deepcopy
import ast, hashlib, json, subprocess, sys, xml.etree.ElementTree as ET
import pytest
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from scripts import mechanics_hbe_halfheight_tension_n24_temporal as case
from scripts import mechanics_hbe_halfheight_temporal_common as common
from scripts import mechanics_hbe_halfheight_tension_n24_temporal_experiment as runner

@pytest.fixture(autouse=True)
def block_native(monkeypatch):
    def no(*a,**k):raise AssertionError('Native work forbidden in review')
    for obj,name in [(runner.old,'solve'),(runner.runtime,'supervise'),(runner,'launch'),(runner,'worker'),(runner,'prepare'),(common,'read_frames'),(case.spatial.BASE,'generate_mesh')]:
        if hasattr(obj,name):monkeypatch.setattr(obj,name,no)
    monkeypatch.setattr(common.subprocess,'Popen',no)

@pytest.fixture(scope='module')
def study():return json.loads((ROOT/case.DECLARATION_PATH).read_bytes())

@pytest.fixture(scope='module')
def compact(study):
    return {key:json.loads((ROOT/study['baseline'][key]['path']).read_bytes()) for key in
        ('independent_review','result','state','execution','baseline_readout','comparison')}

@pytest.mark.parametrize('record,path,value',[
 ('result',('status',),'completed_numerical_diagnostic_only'),
 ('comparison',('passed',),True),
 ('independent_review',('accepted_spatial_convergence',),True),
 ('independent_review',('failed_criteria',),[]),
 ('baseline_readout',('passed',),False),
 ('execution',('runtime_identity',),{'path':'different','sha256':'0'*64}),
 ('result',('supervision','cleanup_error'),'not-reaped'),
])
def test_preserved_failure_and_individual_origin_refuse(study,compact,record,path,value):
    values=deepcopy(compact)
    target=values[record]
    for key in path[:-1]:target=target[key]
    target[path[-1]]=value
    def bound(binding,**kwargs):
        for key,row in values.items():
            if binding==study['baseline'][key]:return row
        raise AssertionError('Refusal must precede further reads')
    with pytest.raises(ValueError):runner.baseline_metadata(ROOT,study,{'runtime_identity':study['baseline']['runtime_identity']},bound)

@pytest.fixture(scope='module')
def decks(study):
    protocol,extraction,_,_=case.geometry(ROOT,study)
    new=case.contents(ROOT,study,protocol,extraction)[1]
    old=(ROOT/study['baseline']['primitive_bindings']['deck']['path']).read_bytes()
    return old,new

@pytest.mark.parametrize('mutation',['bodyforce','sidecontact','flip_midplane','endpoint','odd_point','extra_controller'])
def test_exact_schedule_gate_refuses_physical_changes(decks,mutation):
    old,new=decks;t=ET.fromstring(new)
    if mutation in ('bodyforce','sidecontact'):ET.SubElement(t,'Loads' if mutation=='bodyforce' else 'Contact')
    elif mutation=='flip_midplane':
        bc=next(b for b in t.findall('Boundary/bc') if b.attrib['node_set'].startswith('top_node_'))
        bc.find('value').text='-0.5'
    elif mutation=='endpoint':
        curve=t.findall('LoadData/load_controller')[-1].find('points')
        time,value=map(float,curve[-1].text.split(','));curve[-1].text=f'{time},{value+1e-6}'
    elif mutation=='odd_point':
        curve=t.findall('LoadData/load_controller')[-1].find('points')
        time,value=map(float,curve[1].text.split(','));curve[1].text=f'{time},{value+1e-6}'
    else:ET.SubElement(t.find('LoadData'),'load_controller',{'id':'99','type':'loadcurve'})
    with pytest.raises(ValueError):common.verify_schedule_only_change(old,ET.tostring(t))

class Proc:
    pid=99
    def __init__(self,timeout=False):self.timeout=timeout;self.killed=False;self.waits=[]
    def wait(self,timeout=None):
        self.waits.append(timeout)
        if timeout is not None and self.timeout:raise subprocess.TimeoutExpired(['pure-child'],timeout)
        return 0
    def poll(self):return None if self.timeout and not self.killed else 0
    def kill(self):self.killed=True

@pytest.mark.parametrize('failure',['timeout','publication'])
def test_pure_child_reaps_and_charges_own_receipt(tmp_path,monkeypatch,failure):
    clock=[0.];p=Proc(timeout=failure=='timeout');calls={};saved=[]
    def popen(*args,**kw):calls.update(kw);return p
    def save(*args):
        saved.append(deepcopy(args[-1]))
        if failure=='publication':clock[0]=61.
    monkeypatch.setattr(common.time,'monotonic',lambda:clock[0])
    monkeypatch.setattr(common.subprocess,'Popen',popen)
    monkeypatch.setattr(common.receipts,'durable_json',save)
    with pytest.raises((TimeoutError,subprocess.TimeoutExpired)):
        common.pure_child(['pure-child'],tmp_path,60.,60.)
    assert calls.get('start_new_session',False) is False
    assert p.waits[-1] is None and len(p.waits)==2
    assert p.killed is (failure=='timeout')
    assert saved and saved[0]['solver_calls']==saved[0]['gmsh_generation_calls']==0


def test_declared_source_inventory_passes_exact_archive_request(monkeypatch,study):
    sources={k:{'path':'scripts/'+v,'sha256':study['inherited_source_sha256'][k]} for k,v in study['inherited_source_filenames'].items()}
    for k,mod in dict(runner.NEW,**{runner.RUNNER_KEY:runner}).items():
        p=Path(mod.__file__).resolve();sources[k]={'path':str(p.relative_to(ROOT)),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
    release={'source_bindings':sources,'study':{'path':case.DECLARATION_PATH,'sha256':case.DECLARATION_SHA256},
        'source_archive':{'path':'not-executed.tar','sha256':'0'*64},'source_commit':'a'*40}
    captured={}
    def check(root,archive,commit,wanted,count):captured.update(root=root,archive=archive,commit=commit,wanted=wanted,count=count)
    def bound(b,**kw):
        if b!=release['source_archive']:assert hashlib.sha256((ROOT/b['path']).read_bytes()).hexdigest()==b['sha256']
    monkeypatch.setattr(runner.receipts,'verify_archive',check)
    runner.source_inventory(ROOT,release,study,bound)
    assert captured['count']==32 and len(captured['wanted'])==32
    assert len([p for p in captured['wanted'] if p.startswith('scripts/')])==24
    assert captured['commit']==release['source_commit'] and captured['archive']==release['source_archive']
    bad=deepcopy(release);bad['source_bindings'][runner.RUNNER_KEY]['path']='build/wrong/'+Path(runner.__file__).name
    with pytest.raises(ValueError,match='origin'):runner.source_inventory(ROOT,bad,study,bound)


def test_physics_loop_refactor_matches_accepted_reader_exactly():
    # Compare AST of the actual native-state arithmetic against already accepted
    # N36/S120 source. Geometry and frame count are validated by exact adapters.
    source=ROOT/'scripts/mechanics_hbe_halfheight_global_n36_temporal_readout.py'
    ref=ast.parse(source.read_text());new=ast.parse((ROOT/'scripts/mechanics_hbe_halfheight_temporal_common.py').read_text())
    def loop(tree,name):
        f=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==name)
        return next(n for n in ast.walk(f) if isinstance(n,ast.For) and isinstance(n.target,ast.Tuple) and [e.id for e in n.target.elts]==['hn','he'])
    a,b=loop(ref,'read_temporal_run'),loop(new,'read_frames')
    class Normalize(ast.NodeTransformer):
        def visit_Name(self,n):return ast.copy_location(ast.Name(id='branch' if n.id=='expected_branch' else n.id,ctx=n.ctx),n)
    assert ast.dump(Normalize().visit(a),include_attributes=False)==ast.dump(b,include_attributes=False)


def test_original_tension_groups_reproduce_without_global_promotion(study,compact):
    rows={int(k):json.loads((ROOT/v['path']).read_bytes()) for k,v in study['baseline']['older_tension_readouts_for_sensitivity'].items()}
    rows[24]=compact['baseline_readout']
    assert compact['comparison']['passed'] is False
    for first in (8,12):
        actual=case.original_readout._refinement_group(rows[first],rows[16],rows[24],kind='mesh')
        expected=compact['comparison']['groups'][f'tension:N{first}-N16-N24']
        assert case.access.canonical_json(actual)==case.access.canonical_json(expected)
        assert all(case.metric_pass(v) for v in actual.values())

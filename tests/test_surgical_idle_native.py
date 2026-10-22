"""Generated optional-kernel contracts; native checks require an explicit local build."""
import ast
import copy
import json
import math
import os
from pathlib import Path
import shutil
import struct
import subprocess
import sys
import textwrap

import pytest

from resectionlab import surgical_idle_native as adapter
from resectionlab import build_surgical_idle_native as builder


def _child(source, *, receipt=None, path=None):
    environment = dict(os.environ)
    environment.pop('RESSECTIONLAB_IDLE_THERMAL_BUILD', None)
    if receipt is not None:
        environment['RESSECTIONLAB_IDLE_THERMAL_BUILD'] = str(receipt)
    environment['PYTHONPATH'] = str(path or Path(adapter.__file__).resolve().parents[1])
    result = subprocess.run([sys.executable, '-c', textwrap.dedent(source)],
                            env=environment, capture_output=True, text=True, timeout=15)
    assert result.returncode == 0, result.stdout + result.stderr
    return result


def test_unconfigured_fallback_is_stable_detached_and_rejects_environment_changes(tmp_path):
    _child('''
        import os
        from resectionlab import surgical_idle_native as a
        identity = a.strict_live_identity()
        assert identity['mode'] == 'python'
        identity['mode'] = 'mutated'
        assert a.strict_live_identity()['mode'] == 'python'
        assert a.try_idle(None) is False
        os.environ['RESSECTIONLAB_IDLE_THERMAL_BUILD'] = 'new-build'
        try: a.try_idle(None)
        except ValueError as exc: assert 'ENVIRONMENT_CHANGED' in str(exc)
        else: raise AssertionError('selection changed silently')
    ''')


def test_configured_missing_receipt_raises(tmp_path):
    _child('''
        try: from resectionlab import surgical_idle_native
        except ValueError: pass
        else: raise AssertionError('configured missing native became fallback')
    ''', receipt=tmp_path / 'absent.json')


def test_fallback_ignores_new_files_without_configuration(tmp_path):
    _child(f'''
        from pathlib import Path
        from resectionlab import surgical_idle_native as a
        before = a.strict_live_identity()
        Path({str(tmp_path / 'new-build.json')!r}).write_text('{{}}')
        assert a.strict_live_identity() == before
        assert a.try_idle(None) is False
    ''')


def test_replaced_recipe_function_is_not_used():
    _child('''
        from resectionlab import surgical_idle_native as a
        a._recipe.file_digest = lambda path: 'forged'
        try: a.strict_live_identity()
        except ValueError as exc: assert 'RECIPE_FUNCTION_REPLACED' in str(exc)
        else: raise AssertionError('replacement accepted')
    ''')


@pytest.fixture
def native_build(tmp_path):
    supplied = os.environ.get('RESSECTIONLAB_IDLE_THERMAL_TEST_BUILD')
    if not supplied:
        pytest.skip('Explicit optional native build receipt was not supplied')
    original = json.loads(Path(supplied).read_text())
    assert original['schema'] == builder.SCHEMA
    binary = tmp_path / Path(original['binary']).name
    shutil.copyfile(original['binary'], binary)
    receipt = copy.deepcopy(original)
    receipt['binary'] = str(binary)
    receipt['command'] = builder.compile_command(receipt['metadata'], Path(receipt['source']), binary)
    receipt_path = tmp_path / 'build-receipt.json'
    receipt_path.write_text(json.dumps(receipt))
    return receipt_path


_NATIVE_SETUP = '''
import copy, math, struct, sys
from resectionlab import surgical_idle_native as a

def bits(value):
    if type(value) is dict:
        return ('dict', tuple((bits(key), bits(child)) for key, child in value.items()))
    if isinstance(value, float): return (type(value).__name__, struct.pack('!d', value))
    return (type(value).__name__, repr(value))

def state(temps=None, doses=None, cooling=-0.):
    return {'temperatureC': {'1':38.} if temps is None else temps,
            'thermalDoseCSeconds': {'1':-0.} if doses is None else doses,
            'coolingLossJ':cooling}

def scalars(dt=.1, capacity=.002, tau=8., ambient=37., threshold=6.):
    return (dt,capacity,tau,ambient,threshold,math.exp(-dt/tau),-math.expm1(-dt/tau))

def legacy(s, p, args):
    dt,capacity,tau,ambient,threshold,decay,one_minus = args
    for key in sorted(set(s['temperatureC'])|set(p), key=int):
        old=s['temperatureC'].get(key,ambient)-ambient
        power=p.get(key,0.)
        eq=power*tau/capacity
        new=old*decay+eq*one_minus
        s['temperatureC'][key]=ambient+new
        assert dt>=0. and tau>0. and threshold>=0. and power==0. and old<threshold
        s['thermalDoseCSeconds'][key]=s['thermalDoseCSeconds'].get(key,0.)+0.
        s['coolingLossJ']+=max(0.,power*dt-capacity*(new-old))
'''


def test_native_exact_bits_insertion_order_identity_and_discriminating_leftfold(native_build):
    _child(_NATIVE_SETUP + '''
for dt in (.1,0.,-0.,1.):
    original=state({'2':-0.,'10':math.nextafter(43.,-math.inf),'1':-1.e100},
                   {'1':-0.,'2':-0.,'10':0.})
    expected=copy.deepcopy(original);args=scalars(dt)
    legacy(expected,{},args)
    ids=(id(original['temperatureC']),id(original['thermalDoseCSeconds']))
    assert a.try_idle(original,{},*args) is True
    assert bits(original)==bits(expected)
    assert ids==(id(original['temperatureC']),id(original['thermalDoseCSeconds']))
s=state({str(i):38. for i in reversed(range(24))},
        {str(i):-0. for i in range(24)},1.e16)
expected=copy.deepcopy(s);args=scalars(1.,capacity=2.)
legacy(expected,{},args)
loss=2.*(1.-math.exp(-1./8.))
assert 0.<loss<1. and 24*loss>2.
assert 1.e16+sum([loss]*24)!=expected['coolingLossJ']==1.e16
assert a.try_idle(s,{},*args) is True
assert bits(s)==bits(expected)
identity=a.strict_live_identity();assert identity['mode']=='native'
identity['runtime']['version']='mutated'
assert a.strict_live_identity()['runtime']['version']==sys.version
''', receipt=native_build)


def test_native_row_cap_independent_of_keyset_size_mismatch(native_build):
    _child(_NATIVE_SETUP + '''
s=state({str(i):38. for i in range(32769)}, {str(i):0. for i in range(32769)})
before=bits(s)
assert len(s['temperatureC'])==len(s['thermalDoseCSeconds'])==32769
assert a.try_idle(s,{},*scalars()) is False
assert bits(s)==before
''', receipt=native_build)


@pytest.mark.parametrize('case', ['threshold','aliases','subclass','overflow','powers','key','integer','scalar'])
def test_native_declines_before_mutation(native_build, case):
    _child(_NATIVE_SETUP + f'''
class Dict(dict): pass
class Float(float): pass
case={case!r};s=state();p={{}};args=scalars()
if case=='threshold':s['temperatureC']['1']=43.
if case=='aliases':s['thermalDoseCSeconds']=s['temperatureC']
if case=='subclass':s['temperatureC']=Dict(s['temperatureC'])
if case=='overflow':s['temperatureC']['1']=1.e308;args=scalars(ambient=-1.e308,threshold=6.)
if case=='powers':p={{'1':1.}}
if case=='key':s=state({{'01':38.}},{{'01':0.}})
if case=='integer':s['temperatureC']['1']=38
if case=='scalar':args=(Float(.1),*args[1:])
before=bits(s);beforep=bits(p)
assert a.try_idle(s,p,*args) is False
assert bits(s)==before and bits(p)==beforep
''', receipt=native_build)


def test_native_empty_preserves_objects_and_reference_ownership(native_build):
    _child(_NATIVE_SETUP + '''
s=state({},{});cooling=s['coolingLossJ']
assert a.try_idle(s,{},*scalars()) is True
assert s['coolingLossJ'] is cooling and bits(cooling)==bits(-0.)
s=state({'1':38.,'2':43.},{'1':0.,'2':0.})
objects=[s,*s.values(),*s['temperatureC'].keys(),*s['temperatureC'].values(),*s['thermalDoseCSeconds'].values()]
refs=[sys.getrefcount(value) for value in objects]
for _ in range(16):assert a.try_idle(s,{},*scalars()) is False
assert refs==[sys.getrefcount(value) for value in objects]
''', receipt=native_build)


def test_empty_thermal_does_not_use_native_and_first_real_use_rejects_drift(native_build):
    _child(_NATIVE_SETUP + '''
import os
from types import SimpleNamespace
from resectionlab import surgical_drilling_v2 as drill
scene=SimpleNamespace(properties={'tissuePhysicsV2':{'bone':copy.deepcopy(drill.DEFAULT_PROFILE)}},voxel_mm3=1.)
thermal=state({}, {}, -0.)
engine=SimpleNamespace(scene=scene,state={'drillingV2':thermal})
before=bits(thermal)
os.environ.pop('RESSECTIONLAB_IDLE_THERMAL_BUILD')
drill._thermal(engine,.1,{})  # No optional implementation is used.
assert bits(thermal)==before
for call in (a.strict_live_identity,lambda:a.try_idle(thermal,{},*scalars())):
    try:call()
    except ValueError as exc:assert 'ENVIRONMENT_CHANGED' in str(exc)
    else:raise AssertionError('Explicit native access accepted selection drift')
thermal['temperatureC']['1']=38.
thermal['thermalDoseCSeconds']['1']=0.
before=bits(thermal)
try:drill._thermal(engine,.1,{})
except ValueError as exc:assert 'ENVIRONMENT_CHANGED' in str(exc)
else:raise AssertionError('Nonempty thermal update accepted native drift')
assert bits(thermal)==before
''', receipt=native_build)


@pytest.mark.parametrize('replacement', ['callable','module','metadata','environment','recipe','deleted-recipe'])
def test_selected_native_replacements_raise_instead_of_fallback(native_build, replacement):
    _child(_NATIVE_SETUP + f'''
import os
case={replacement!r}
if case=='callable':a._SELECTION[0].try_idle=lambda *args:False
if case=='module':sys.modules[a._recipe.KERNEL_NAME]=object()
if case=='metadata':a._SELECTION[0].BUILD_IDENTITY='changed'
if case=='environment':os.environ.pop('RESSECTIONLAB_IDLE_THERMAL_BUILD')
if case=='recipe':a._recipe.compile_command=lambda *args:[]
if case=='deleted-recipe':del a._recipe.compile_command
for fn in (a.strict_live_identity,lambda:a.try_idle(state(),{{}},*scalars())):
    try:fn()
    except ValueError:pass
    else:raise AssertionError('selected implementation silently drifted')
''', receipt=native_build)


@pytest.mark.parametrize('artifact', ['binary','receipt','source','recipe'])
def test_selected_native_disk_drift_invalidates_without_touching_production(native_build, tmp_path, artifact):
    package=tmp_path/'private-package'/'resectionlab'
    package.mkdir(parents=True)
    (package/'__init__.py').write_text('')
    for name in ('surgical_idle_native.py','surgical_idle_native.c','build_surgical_idle_native.py'):
        shutil.copyfile(Path(adapter.__file__).with_name(name),package/name)
    _child(_NATIVE_SETUP + f'''
from pathlib import Path
import json
artifact={artifact!r}
if artifact=='binary':path=Path(a._SELECTION[3])
elif artifact=='receipt':path=Path(a._SELECTED_ENVIRONMENT)
elif artifact=='source':path=Path(a.__file__).with_suffix('.c')
else:path=Path(a._recipe.__file__)
with path.open('ab') as stream:stream.write(b'\\n')
try:a.try_idle(state(),{{}},*scalars())
except ValueError:pass
else:raise AssertionError('changed loaded bytes trusted as loaded code')
''', receipt=native_build, path=package.parent)


def test_drill_hook_falls_back_to_exact_original_loop_and_errors():
    from resectionlab import surgical_drilling_v2 as drill
    from types import SimpleNamespace
    tree=ast.parse(Path(drill.__file__).read_text())
    function=next(node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name=='_thermal')
    function.body=[node for node in function.body if not (isinstance(node,ast.If)
        and isinstance(node.test,ast.Call) and isinstance(node.test.func,ast.Attribute)
        and isinstance(node.test.func.value,ast.Name) and node.test.func.value.id=='_idle_native')]
    namespace=dict(drill.__dict__)
    exec(compile(ast.Module(body=[function],type_ignores=[]),'original-thermal-loop','exec'),namespace)
    original=namespace['_thermal']
    for temperature in (38.,43.,math.nextafter(43.,math.inf)):
        profile=copy.deepcopy(drill.DEFAULT_PROFILE)
        initial={'drillingV2':{'temperatureC':{'1':temperature},'thermalDoseCSeconds':{'1':-0.},'coolingLossJ':-0.}}
        scene=SimpleNamespace(properties={'tissuePhysicsV2':{'bone':profile}},voxel_mm3=1.)
        expected=SimpleNamespace(state=copy.deepcopy(initial),scene=scene)
        actual=SimpleNamespace(state=copy.deepcopy(initial),scene=scene)
        original(expected,.1,{})
        drill._thermal(actual,.1,{})
        assert json.dumps(expected.state,allow_nan=False)==json.dumps(actual.state,allow_nan=False)


def test_native_drill_fallback_complete_state_and_exception_parity(native_build):
    _child(_NATIVE_SETUP + '''
import ast
from pathlib import Path
from types import SimpleNamespace
from resectionlab import surgical_drilling_v2 as drill
tree=ast.parse(Path(drill.__file__).read_text())
function=next(node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name=='_thermal')
function.body=[node for node in function.body if not (isinstance(node,ast.If)
    and isinstance(node.test,ast.Call) and isinstance(node.test.func,ast.Attribute)
    and isinstance(node.test.func.value,ast.Name) and node.test.func.value.id=='_idle_native')]
namespace=dict(drill.__dict__)
exec(compile(ast.Module(body=[function],type_ignores=[]),'original-thermal-loop','exec'),namespace)
original=namespace['_thermal']
for case in ('threshold','powers','key-error','setup-error','alias'):
    p=copy.deepcopy(drill.DEFAULT_PROFILE)
    s=state({'1':43.},{'1':-0.});powers={}
    if case=='powers':powers={'1':.01}
    if case=='key-error':s=state({'bad':38.},{'bad':0.})
    if case=='setup-error':p['coolingTimeSeconds']=0.
    if case=='alias':s['thermalDoseCSeconds']=s['temperatureC']
    scene=SimpleNamespace(properties={'tissuePhysicsV2':{'bone':p}},voxel_mm3=1.)
    left=SimpleNamespace(state={'drillingV2':copy.deepcopy(s)},scene=scene)
    right=SimpleNamespace(state={'drillingV2':copy.deepcopy(s)},scene=scene)
    def run(fn,e):
        try:fn(e,.1,copy.deepcopy(powers))
        except Exception as exc:return(type(exc),str(exc))
    assert run(original,left)==run(drill._thermal,right)
    assert bits(left.state)==bits(right.state)
''', receipt=native_build)


def test_native_watched_dictionaries_decline_without_callbacks(native_build):
    _child(_NATIVE_SETUP + '''
import ctypes
api=ctypes.pythonapi
if hasattr(api,'PyDict_AddWatcher'):
    events=[]
    callbacktype=ctypes.CFUNCTYPE(ctypes.c_int,ctypes.c_int,ctypes.py_object,ctypes.c_void_p,ctypes.c_void_p)
    @callbacktype
    def callback(event,mapping,key,value):events.append(event);return 0
    api.PyDict_AddWatcher.argtypes=[callbacktype];api.PyDict_AddWatcher.restype=ctypes.c_int
    for name in ('PyDict_Watch','PyDict_Unwatch'):
        getattr(api,name).argtypes=[ctypes.c_int,ctypes.py_object];getattr(api,name).restype=ctypes.c_int
    api.PyDict_ClearWatcher.argtypes=[ctypes.c_int];api.PyDict_ClearWatcher.restype=ctypes.c_int
    watcher=api.PyDict_AddWatcher(callback);assert watcher>=0
    try:
        for field in ('state','temperatureC','thermalDoseCSeconds','powers'):
            s=state();p={};mapping=s if field=='state' else p if field=='powers' else s[field]
            before=bits(s)
            assert api.PyDict_Watch(watcher,mapping)==0
            try:
                assert a.try_idle(s,p,*scalars()) is False
                assert bits(s)==before
            finally:assert api.PyDict_Unwatch(watcher,mapping)==0
        assert events==[]
    finally:assert api.PyDict_ClearWatcher(watcher)==0
''', receipt=native_build)


@pytest.mark.parametrize('problem',['flags','runtime','headers','binary-size','oversized-receipt'])
def test_invalid_explicit_build_rejected_before_native_loading(native_build, problem):
    document=json.loads(native_build.read_text())
    if problem=='flags':document['metadata']['compileFlags'].append('-ffast-math')
    if problem=='runtime':document['metadata']['runtime']['version']='different interpreter'
    if problem=='headers':document['metadata']['headers']['treeSha256']='0'*64
    if problem=='binary-size':document['binaryBytes']+=1
    if problem=='oversized-receipt':native_build.write_text(' '* (1024*1024+1))
    else:native_build.write_text(json.dumps(document))
    _child('''
        import importlib.util
        def forbidden(*args):raise AssertionError('native loading must not start')
        importlib.util.module_from_spec=forbidden
        try:from resectionlab import surgical_idle_native
        except ValueError:pass
        else:raise AssertionError('invalid build accepted')
    ''',receipt=native_build)

"""Generated contracts; native checks require an explicitly built local receipt.

No builder, engine, patient action, or learning code is executed by these tests.
"""
import copy
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import textwrap

import pytest

from resectionlab import build_surgical_prefix_json_native as builder


_ENV = 'RESSECTIONLAB_PREFIX_JSON_BUILD'
_ROOT = Path(builder.__file__).resolve().parents[1]


def _child(source, *, receipt=None, path=None):
    environment = dict(os.environ)
    environment.pop(_ENV, None)
    if receipt is not None:
        environment[_ENV] = str(receipt)
    environment['PYTHONPATH'] = str(path or _ROOT)
    result = subprocess.run([sys.executable, '-c', textwrap.dedent(source)], env=environment,
                            capture_output=True, text=True, timeout=15)
    assert result.returncode == 0, result.stdout + result.stderr
    return result


def test_unconfigured_selection_detached_fixed_and_no_implicit_build(tmp_path):
    _child(f'''
        import os
        from pathlib import Path
        from resectionlab import surgical_prefix_json_native as a
        identity = a.strict_live_identity()
        assert identity == a.strict_live_identity(full=True)
        assert identity['mode'] == 'python'
        identity['mode'] = 'changed'
        assert a.strict_live_identity()['mode'] == 'python'
        assert a.try_encode({{'x':1}}, 1024) is None
        Path({str(tmp_path / 'new-build.json')!r}).write_text('{{}}')
        assert a.strict_live_identity()['mode'] == 'python'
        os.environ['RESSECTIONLAB_PREFIX_JSON_BUILD'] = 'appeared'
        try: a.try_encode(None, 1024)
        except ValueError as e: assert 'ENVIRONMENT_CHANGED' in str(e)
        else: raise AssertionError('selection drift silently accepted')
    ''')


@pytest.mark.parametrize('case', ['missing', 'empty', 'invalid-json', 'oversized'])
def test_configured_invalid_receipt_never_becomes_fallback(tmp_path, case):
    path = tmp_path / 'invalid.json'
    _child(f'''
        from pathlib import Path
        path=Path({str(path)!r});case={case!r}
        if case=='empty':path.write_bytes(b'')
        if case=='invalid-json':path.write_bytes(b'{{}}')
        if case=='oversized':path.write_bytes(b'x'*(1024*1024+1))
        try: from resectionlab import surgical_prefix_json_native
        except ValueError: pass
        else: raise AssertionError('invalid configuration silently fell back')
    ''', receipt=path)


@pytest.mark.parametrize('replacement', ['function', 'code', 'defaults', 'constant', 'deleted'])
def test_python_mode_recipe_drift_is_rejected(replacement):
    _child(f'''
        from resectionlab import surgical_prefix_json_native as a
        case={replacement!r}
        if case=='function': a._recipe.scalar_images=lambda: {{}}
        if case=='code': a._recipe.scalar_images.__code__=(lambda: {{}}).__code__
        if case=='defaults': a._recipe.scalar_images.__defaults__=(1,)
        if case=='constant': a._recipe.MAX_RAW_BYTES+=1
        if case=='deleted': del a._recipe.scalar_images
        try: a.strict_live_identity()
        except ValueError: pass
        else: raise AssertionError('changed recipe accepted')
    ''')


def test_scalar_image_stat_and_full_byte_checks_are_independent(tmp_path):
    _child(f'''
        import hashlib
        from pathlib import Path
        from resectionlab import surgical_prefix_json_native as a
        p=Path({str(tmp_path / 'image')!r});p.write_bytes(b'abcdef')
        digest=hashlib.sha256(p.read_bytes()).hexdigest();stat=a._stat_identity(p)
        images=((p,digest,stat),)
        a._check_images(images,full=False);a._check_images(images,full=True)
        p.write_bytes(b'ABCDEF')
        for full in (False,True):
            try:a._check_images(images,full=full)
            except ValueError:pass
            else:raise AssertionError('changed stat admitted')
        # Model an unchanged stat oracle explicitly to isolate the full byte check.
        # This does not claim ordinary filesystem ctime can be restored.
        a._stat_identity=lambda path:stat
        a._check_images(images,full=False)
        try:a._check_images(images,full=True)
        except ValueError as e:assert 'SCALAR_IMAGE_BYTES_CHANGED' in str(e)
        else:raise AssertionError('full check did not hash bytes')
    ''')


@pytest.fixture
def native_build(tmp_path):
    supplied = os.environ.get('RESSECTIONLAB_PREFIX_JSON_TEST_BUILD')
    if not supplied:
        pytest.skip('Explicit optional prefix JSON build receipt was not supplied')
    original = json.loads(Path(supplied).read_text())
    assert original['schema'] == builder.SCHEMA
    binary = tmp_path / Path(original['binary']).name
    shutil.copyfile(original['binary'], binary)
    receipt = copy.deepcopy(original)
    receipt['binary'] = str(binary)
    receipt['command'] = builder.compile_command(receipt['metadata'], Path(receipt['source']), binary)
    path = tmp_path / 'build-receipt.json'
    path.write_text(json.dumps(receipt))
    return path


_SETUP = '''
import json, math, struct, sys
from resectionlab import surgical_prefix_json_native as a
assert a.strict_live_identity()['mode']=='native'
def stock(value):
    return json.dumps(value,ensure_ascii=True,allow_nan=False,separators=(',',':'))
def bits(value):
    if type(value) is float:return ('float',struct.pack('!d',value))
    if type(value) is dict:return ('dict',tuple((k,bits(v)) for k,v in value.items()))
    if type(value) is list:return ('list',tuple(map(bits,value)))
    return (type(value).__name__,value)
'''


def test_native_exact_bytes_ieee_order_unicode_aliases_and_limits(native_build):
    _child(_SETUP + '''
shared=[-0.,0.,5.e-324,math.nextafter(1.,2.),sys.float_info.max]
v={'z':None,'a':True,'b':False,'integers':[-2**63,2**63-1,0],
   'unicode':{'\\x00\\n\\t\\"\\\\':'é😀\\ud800'},'tuple':(1,2),'left':shared,'right':shared}
ids=(id(v),id(shared));expected=stock(v)
for limit in (16*1024*1024,64*1024*1024,10**100):
    got=a.try_encode(v,limit)
    assert type(got) is str and got.isascii() and got==expected
    assert bits(json.loads(got))==bits(json.loads(expected))
    assert ids==(id(v),id(shared)) and v['left'] is v['right']
identity=a.strict_live_identity();assert identity==a.strict_live_identity(full=True)
identity['runtime']['version']='modified'
assert a.strict_live_identity()['runtime']['version']==sys.version
for limit in (0,-1,1,2):assert a.try_encode(v,limit) is None
assert a.try_encode([],2)=='[]'
''', receipt=native_build)


@pytest.mark.parametrize('case', ['list','tuple','dict','str','int','float','nonfinite','hugeint',
                                  'key','cycle','depth','nodes','budget'])
def test_native_decline_is_untouched_and_has_no_custom_callbacks(native_build, case):
    _child(_SETUP + f'''
case={case!r};calls=[]
class L(list):
    def __iter__(self):calls.append('iter');raise RuntimeError('custom iterator')
class T(tuple):pass
class D(dict):
    def items(self):calls.append('items');raise RuntimeError('custom items')
class S(str):pass
class I(int):pass
class F(float):pass
v=1;limit=16*1024*1024
if case=='list':v=L([1])
if case=='tuple':v=T((1,))
if case=='dict':v=D(x=1)
if case=='str':v=S('x')
if case=='int':v=I(1)
if case=='float':v=F(1.)
if case=='nonfinite':v=[math.inf,-math.inf,math.nan]
if case=='hugeint':v=2**64
if case=='key':v={{1:'coercible'}}
if case=='cycle':v=[];v.append(v)
if case=='depth':
    v=0
    for _ in range(65):v=[v]
if case=='nodes':v=[None]*1000001
if case=='budget':v=['x'*1000];limit=100
before_refs=sys.getrefcount(v);before_len=len(v) if hasattr(v,'__len__') else None
for _ in range(3):assert a.try_encode(v,limit) is None
assert not calls and sys.getrefcount(v)==before_refs
assert (len(v) if hasattr(v,'__len__') else None)==before_len
''', receipt=native_build)


def test_native_invalid_limit_cannot_invoke_index_comparison_or_min(native_build):
    _child(_SETUP + '''
class Limit:
    def __index__(self):raise AssertionError('index called')
    def __lt__(self,other):raise AssertionError('comparison called')
class I(int):
    def __lt__(self,other):raise AssertionError('subclass comparison called')
for limit in (Limit(),I(100),True,1.,'100',None):
    assert a.try_encode({'x':1},limit) is None
    assert a._SELECTION.encode({'x':1},limit) is None
''', receipt=native_build)


def test_native_recursive_context_conservatively_declines(native_build):
    _child(_SETUP + '''
sys.setrecursionlimit(400)
def nested(n):
    if n:return nested(n-1)
    return a.try_encode({'x':1},1024)
assert nested(200) is None
assert a.try_encode({'x':1},1024)=='{"x":1}'
''', receipt=native_build)


@pytest.mark.parametrize('replacement', ['module','callable','metadata','environment','ascii'])
def test_selected_native_live_drift_raises(native_build, replacement):
    _child(_SETUP + f'''
import os
case={replacement!r}
if case=='module':sys.modules[a._recipe.KERNEL_NAME]=None
if case=='callable':a._SELECTION.module.try_encode=lambda *args:None
if case=='metadata':a._SELECTION.module.BUILD_IDENTITY='changed'
if case=='environment':os.environ['RESSECTIONLAB_PREFIX_JSON_BUILD']='changed'
if case=='ascii':a._SELECTION.json_module.encode_basestring_ascii=lambda v:'"x"'
try:a.try_encode({{'x':1}},1024)
except ValueError:pass
else:raise AssertionError('drift silently admitted or fell back')
''', receipt=native_build)


@pytest.mark.parametrize('target', ['binary','receipt','source','recipe','utilities'])
def test_selected_disk_drift_raises_without_mutating_original_files(native_build, tmp_path, target):
    package_root = tmp_path / 'private-src'
    package = package_root / 'resectionlab'
    package.mkdir(parents=True)
    (package / '__init__.py').write_text('')
    for filename in ('surgical_prefix_json_native.py','surgical_prefix_json_native.c',
                     'build_surgical_prefix_json_native.py','build_surgical_idle_native.py'):
        shutil.copyfile(_ROOT / 'resectionlab' / filename, package / filename)
    receipt = json.loads(native_build.read_text())
    targets = {'binary': receipt['binary'], 'receipt': str(native_build),
               'source': str(package / 'surgical_prefix_json_native.c'),
               'recipe': str(package / 'build_surgical_prefix_json_native.py'),
               'utilities': str(package / 'build_surgical_idle_native.py')}
    _child(_SETUP + f'''
from pathlib import Path
p=Path({targets[target]!r});p.write_bytes(p.read_bytes()+b' ')
try:a.strict_live_identity()
except ValueError:pass
else:raise AssertionError('selected disk change admitted')
''', receipt=native_build, path=package_root)


@pytest.mark.parametrize('damage', ['oversized','mismatch','runtime','flags'])
def test_bad_build_rejected_before_native_module_creation(native_build, damage):
    receipt = json.loads(native_build.read_text())
    if damage == 'oversized':
        with Path(receipt['binary']).open('wb') as stream:
            stream.truncate(builder.MAX_BINARY_BYTES + 1)
    if damage == 'mismatch':receipt['binaryBytes'] += 1
    if damage == 'runtime':receipt['metadata']['runtime']['version'] += 'different'
    if damage == 'flags':receipt['metadata']['compileFlags'] += ['-ffast-math']
    native_build.write_text(json.dumps(receipt))
    _child('''
import importlib.util
importlib.util.module_from_spec=lambda spec: (_ for _ in ()).throw(AssertionError('native load attempted'))
try:from resectionlab import surgical_prefix_json_native
except ValueError:pass
else:raise AssertionError('invalid build admitted')
''', receipt=native_build)


def test_native_decline_preserves_legacy_error_and_custom_iterator_behavior(native_build):
    _child(_SETUP + '''
def fallback(value):
    native=a.try_encode(value,16*1024*1024)
    return stock(value) if native is None else native
class Custom(list):
    def __iter__(self):
        yield 7
v=Custom([1])
assert a.try_encode(v,1024) is None and fallback(v)==stock(v)=='[7]'
cycle=[];cycle.append(cycle)
for v in (math.nan,math.inf,cycle,object()):
    errors=[]
    for operation in (stock,fallback):
        try:operation(v)
        except Exception as exc:errors.append((type(exc),str(exc)))
        else:raise AssertionError('legacy error disappeared')
    assert errors[0]==errors[1]
''', receipt=native_build)

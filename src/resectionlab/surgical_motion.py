"""Whole-instrument capsule envelopes including roll-dependent jaws/blades.

Inflated midpoint capsules conservatively enclose translation and rotation over
each deterministic substep. Working geometry never shrinks the physical shaft.
"""
from collections import OrderedDict
import functools as _envelope_functools
import sys
from types import BuiltinFunctionType, CodeType, FunctionType, ModuleType

import numpy as np
from scipy.spatial.transform import Rotation, Slerp
from .geometry import segments_within_distance
from . import geometry as _contact_geometry, native_backend


def _function_state(function):
    keywords = function.__kwdefaults__
    return (function, function.__code__, function.__defaults__,
            tuple((id(key), key, id(value), value) for key, value in keywords.items())
            if type(keywords) is dict else (() if keywords is None else object()))


_ROTATION = Rotation
_SLERP = Slerp
_ROTATION_OBJECT_HOOKS = ('__getattribute__', '__getattr__', '__setattr__', '__delattr__', '__new__', '__del__')


def _capture_rotation_helpers():
    result = []
    for owner, names in ((Rotation, ('__init__', '__len__', '__getitem__', '__mul__', 'from_quat',
                                     'from_rotvec', 'inv', 'magnitude', 'as_quat', 'as_rotvec',
                                     'apply', 'as_matrix')),
                         (Slerp, ('__init__', '__call__'))):
        if type(owner) is not type:
            return ((owner, None, None, None),)
        bases = owner.__bases__
        if (len(bases) != 1 or bases[0] is not object
                or any(name in owner.__dict__ for name in _ROTATION_OBJECT_HOOKS)):
            return ((owner, None, None, None),)
        for name in names:
            raw = owner.__dict__.get(name)
            function = raw
            if name in ('from_quat', 'from_rotvec'):
                function = raw.__func__ if type(raw) in (staticmethod, classmethod) else None
            state = _function_state(function) if type(function) is FunctionType else None
            result.append((owner, name, raw, state))
    return tuple(result)


_ROTATION_HELPERS = _capture_rotation_helpers()


def _source_rotation_helpers_genuine():
    # shortcut: source-qualify Python wrappers; qualify native-only versions if profiling warrants reuse.
    try:
        classes = {}
        for owner, name, raw, state in _ROTATION_HELPERS:
            if state is None or type(owner) is not type or type(owner.__module__) is not str:
                return False
            module = sys.modules[owner.__module__]
            function = state[0]
            if (type(function) is not FunctionType or function.__globals__ is not vars(module)
                    or function.__module__ != module.__name__):
                return False
            if owner not in classes:
                code = module.__spec__.loader.get_code(module.__name__)
                body, = (value for value in code.co_consts
                         if type(value) is CodeType and value.co_name == owner.__name__)
                classes[owner] = {value.co_name: value for value in body.co_consts if type(value) is CodeType}
            if function.__code__ != classes[owner].get(name):
                return False
            if name in ('from_quat', 'from_rotvec'):
                kind = classmethod if classes[owner][name].co_varnames[0] == 'cls' else staticmethod
                if type(raw) is not kind:
                    return False
            defaults, keywords = function.__defaults__, function.__kwdefaults__
            if ((defaults is not None and type(defaults) is not tuple)
                    or (keywords is not None and type(keywords) is not dict)):
                return False
            if keywords is not None and any(type(key) is not str for key in keywords):
                return False
            values = (() if defaults is None else defaults) + (() if keywords is None else tuple(keywords.values()))
            if any(all(type(value) is not atom for atom in (type(None), bool, int, float, str, bytes))
                   for value in values):
                return False
        return True
    except (AttributeError, ImportError, KeyError, OSError, NotImplementedError, TypeError, ValueError):
        return False


_rotation_captures_genuine = _source_rotation_helpers_genuine()
_sweep_rotation_cache = OrderedDict()


def _sweep_rotation(quaternion_bytes):
    cached = _sweep_rotation_cache.get(quaternion_bytes)
    if cached is not None:
        _sweep_rotation_cache.move_to_end(quaternion_bytes)
        return cached
    rotations = Rotation.from_quat(np.frombuffer(quaternion_bytes, dtype=np.float64).reshape(2, 4))
    angle = (rotations[1] * rotations[0].inv()).magnitude()
    q = Slerp([0., 1.], rotations)(.5).as_quat()
    result = angle, q.tobytes()
    _sweep_rotation_cache[quaternion_bytes] = result
    if len(_sweep_rotation_cache) > 64:
        _sweep_rotation_cache.popitem(last=False)
    return result


_SWEEP_BODY = _sweep_rotation
_SWEEP_STATE = _function_state(_SWEEP_BODY)


def _rotation_helpers_unchanged():
    if (not _rotation_captures_genuine or Rotation is not _ROTATION
            or Slerp is not _SLERP or _sweep_rotation is not _SWEEP_BODY):
        return False
    for owner in (_ROTATION, _SLERP):
        bases = owner.__bases__
        if (len(bases) != 1 or bases[0] is not object
                or any(name in owner.__dict__ for name in _ROTATION_OBJECT_HOOKS)):
            return False
    current = _function_state(_SWEEP_BODY)
    if (current[0] is not _SWEEP_STATE[0] or current[1] is not _SWEEP_STATE[1]
            or current[2] is not _SWEEP_STATE[2]
            or current[3] != _SWEEP_STATE[3]):
        return False
    for owner, name, raw, expected in _ROTATION_HELPERS:
        if owner.__dict__.get(name) is not raw:
            return False
        function = raw.__func__ if type(raw) in (staticmethod, classmethod) else raw
        current = _function_state(function)
        if (current[0] is not expected[0] or current[1] is not expected[1]
                or current[2] is not expected[2]
                or current[3] != expected[3]):
            return False
    return _numpy_underflow_ignored()


_NUMPY_GETERR_STATE = _function_state(np.geterr) if type(np.geterr) is FunctionType else None
_NUMPY_ERROR_GETTER = (_NUMPY_GETERR_STATE[0].__globals__.get('_get_extobj_dict')
                       if _NUMPY_GETERR_STATE is not None else None)


def _source_numpy_error_getter_genuine():
    try:
        if _NUMPY_GETERR_STATE is None or type(_NUMPY_ERROR_GETTER) is not BuiltinFunctionType:
            return False
        function = _NUMPY_GETERR_STATE[0]
        namespace = function.__globals__
        module = sys.modules[namespace['__name__']]
        if (module.__name__ not in ('numpy._core._ufunc_config', 'numpy.core._ufunc_config')
                or namespace is not vars(module) or function.__defaults__ is not None
                or function.__kwdefaults__ is not None):
            return False
        code = module.__spec__.loader.get_code(module.__name__)
        original, = (value for value in code.co_consts if type(value) is CodeType and value.co_name == 'geterr')
        getter = _NUMPY_ERROR_GETTER
        owner = sys.modules[getter.__module__]
        return (function.__code__ == original and type(owner) is ModuleType
            and getter.__module__ in ('numpy._core._multiarray_umath', 'numpy.core._multiarray_umath')
            and getter.__self__ is owner and getter.__name__ == '_get_extobj_dict'
            and vars(owner).get('_get_extobj_dict') is getter)
    except (AttributeError, ImportError, KeyError, OSError, NotImplementedError, TypeError, ValueError):
        return False


_numpy_error_captures_genuine = _source_numpy_error_getter_genuine()


def _numpy_underflow_ignored():
    if (not _numpy_error_captures_genuine or np.geterr is not _NUMPY_GETERR_STATE[0]
            or _NUMPY_GETERR_STATE[0].__globals__.get('_get_extobj_dict') is not _NUMPY_ERROR_GETTER
            or vars(_NUMPY_ERROR_GETTER.__self__).get('_get_extobj_dict') is not _NUMPY_ERROR_GETTER):
        return False
    current = _function_state(_NUMPY_GETERR_STATE[0])
    if (current[0] is not _NUMPY_GETERR_STATE[0] or current[1] is not _NUMPY_GETERR_STATE[1]
            or current[2] is not _NUMPY_GETERR_STATE[2]
            or current[3] != _NUMPY_GETERR_STATE[3]):
        return False
    return _NUMPY_GETERR_STATE[0]()['under'] == 'ignore'


def _ordinary_vector(value, length):
    if type(value) is np.ndarray:
        return value.shape == (length,) and value.dtype == np.dtype(np.float64)
    return (type(value) in (list, tuple) and len(value) == length
            and all(type(component) in (int, float) for component in value))


def _envelopes_reference(tool, parameters, position, quaternion, inflation=0.):
    p = np.asarray(position, float); r = Rotation.from_quat(quaternion)
    radius = parameters['radiusMm']; depth = parameters.get('depthMm', 2.)
    shape = tool['geometry']
    def capsule(a, b, width, part):
        return p + r.apply(a), p + r.apply(b), width + inflation, part
    if shape == 'aspirator' and parameters.get('contactGeometryRevision') == 2:
        # V3 aspiration acts at the spherical tip. A widened rear capsule in
        # revision1 demanded removal outside that sphere before further work.
        # Revision2 keeps the declared 1.5-mm tip / .5-mm shaft and work rate,
        # with a continuous checked shaft rather than an unaspirated shoulder.
        return [capsule([0,0,-40],[0,0,0],tool['shaftRadiusMm'],'shaft'),
                capsule([0,0,0],[0,0,0],radius,'working')]
    result = [capsule([0, 0, -40], [0, 0, -max(1., min(depth, 5.))], tool['shaftRadiusMm'], 'shaft')]
    if shape in ('scissors', 'forceps'):
        half = parameters.get('jawOpeningMm', 2.) / 2
        result.extend(capsule([sign * half, 0, -depth], [sign * half, 0, 0], .35, 'working') for sign in (-1, 1))
    elif shape in ('blade', 'elevator', 'kerf'):
        if shape=='blade' and parameters.get('retractorFaceContactRevision')==1:
            # Cover the whole finite pad, including the interior between the
            # historical two crossbars. Rounded-edge contact remains unqualified.
            length=min(depth,5.);count=max(1,int(np.ceil(length/.25)))
            width=float(np.hypot(.4,length/count/2))
            result.extend(capsule([-radius,0,-z],[radius,0,-z],width,'working') for z in np.linspace(0,length,count+1))
        else:
            result.extend(capsule([-radius, 0, -z], [radius, 0, -z], .4, 'working') for z in (0., min(depth, 5.)))
    else:
        result.append(capsule([0, 0, -min(depth, 3.)], [0, 0, 0], radius, 'working'))
    return result


def _capture_envelope_helpers():
    """Qualify the stock scalar apply path that the fused helper replaces."""
    if not _rotation_captures_genuine:
        return None
    try:
        owner = sys.modules[_ROTATION.__module__]
        namespace = vars(owner)
        cython, extra = namespace['cython_backend'], namespace['xpx']
        registry = namespace['backend_registry']
        if type(registry) is not dict or len(registry) != 1:
            return None
        xp, = registry
        if (type(xp) is not ModuleType or registry[xp] is not cython
                or xp.__name__ != 'scipy._external.array_api_compat.numpy'):
            return None
        if (type(cython) is not ModuleType or type(extra) is not ModuleType
                or cython.__name__ != 'scipy.spatial.transform._rotation_cy'):
            return None
        if type(np.asarray) is not BuiltinFunctionType or type(np.asarray.__self__) is not ModuleType:
            return None
        if type(np.atleast_2d) is not vars(np.asarray.__self__).get('_ArrayFunctionDispatcher'):
            return None
        if type(namespace['is_numpy']) is not _envelope_functools._lru_cache_wrapper:
            return None
        scalar_helpers = (namespace['select_backend'], namespace['xp_device'],
                          vars(extra).get('atleast_nd'), vars(xp).get('asarray'))
        if any(type(function) is not FunctionType for function in scalar_helpers):
            return None
        bindings = [(owner, 'select_backend'), (owner, 'xp_device'), (owner, 'is_numpy'), (owner, 'np'),
                    (owner, 'cython_backend'), (owner, 'xpx'), (owner, 'backend_registry'),
                    (cython, 'apply'), (cython, 'as_matrix'), (cython, 'np'), (extra, 'atleast_nd'),
                    (np, 'asarray'), (np, 'matmul'), (np, 'atleast_2d'), (np, 'float64'),
                    (xp, 'asarray'), (xp, 'atleast_2d'), (np, 'array'), (np, 'dtype')]
        functions = [*scalar_helpers, namespace['is_numpy'].__wrapped__, np.atleast_2d.__wrapped__]
        array_helpers = xp.asarray.__globals__['_helpers']
        if type(array_helpers) is not ModuleType:
            return None
        bindings.append((array_helpers, '_check_device'))
        functions.append(vars(array_helpers).get('_check_device'))
        device_namespace = namespace['xp_device'].__globals__
        for name in ('is_numpy_array', '_issubclass_fast', '_compat_module_name'):
            helper = device_namespace[name]
            bindings.append((sys.modules[namespace['xp_device'].__module__], name))
            if type(helper) is FunctionType:
                functions.append(helper)
            elif type(helper) is _envelope_functools._lru_cache_wrapper:
                functions.append(helper.__wrapped__)
            else:
                return None
        bindings.append((sys.modules[extra.atleast_nd.__module__], 'is_numpy_namespace'))
        namespace_helper = extra.atleast_nd.__globals__['is_numpy_namespace']
        if type(namespace_helper) is not _envelope_functools._lru_cache_wrapper:
            return None
        functions.append(namespace_helper.__wrapped__)
        shape_module = sys.modules[np.atleast_2d.__wrapped__.__module__]
        bindings.extend(((shape_module, 'asanyarray'), (shape_module, '_nx'),
                         (sys.modules['builtins'], 'min'), (sys.modules['builtins'], 'max'),
                         (sys.modules['builtins'], 'float')))
        captured = tuple((module, name, vars(module)[name]) for module, name in bindings)
        for function in functions:
            if type(function) is not FunctionType:
                return None
            module = sys.modules[function.__module__]
            if function.__globals__ is not vars(module):
                return None
            body = module.__spec__.loader.get_code(module.__name__)
            original, = (code for code in body.co_consts
                         if type(code) is CodeType and code.co_name == function.__name__)
            if function.__code__ != original:
                return None
        if (Rotation.apply.__defaults__ != (False,) or Rotation.apply.__kwdefaults__ is not None
                or Rotation.as_matrix.__defaults__ is not None or Rotation.as_matrix.__kwdefaults__ is not None
                or type(cython.apply).__name__ != 'cython_function_or_method'
                or cython.apply.__module__ != cython.__name__
                or cython.apply.__name__ != 'apply'
                or cython.as_matrix.__module__ != cython.__name__
                or cython.as_matrix.__name__ != 'as_matrix'
                or type(np.asarray) is not BuiltinFunctionType
                or np.asarray.__name__ != 'asarray'
                or type(np.asarray.__self__) is not ModuleType
                or np.asarray.__self__.__name__ not in ('numpy._core._multiarray_umath', 'numpy.core._multiarray_umath')
                or vars(np.asarray.__self__).get('asarray') is not np.asarray
                or vars(np.asarray.__self__).get('array') is not np.array
                or vars(np.asarray.__self__).get('dtype') is not np.dtype
                or cython.np is not np
                or type(np.matmul) is not np.ufunc or np.matmul.__name__ != 'matmul'
                or vars(np.asarray.__self__).get('matmul') is not np.matmul
                or vars(sys.modules['builtins'])['float'] is not (0.).__class__
                or any(type(vars(sys.modules['builtins'])[name]) is not BuiltinFunctionType
                       or vars(sys.modules['builtins'])[name].__name__ != name
                       or vars(sys.modules['builtins'])[name].__self__ is not sys.modules['builtins']
                       for name in ('min', 'max'))):
            return None
        return captured, tuple(_function_state(function) for function in functions), registry, cython, xp
    except (AttributeError, ImportError, KeyError, OSError, NotImplementedError, TypeError, ValueError):
        return None


_ENVELOPE_HELPERS = _capture_envelope_helpers()
_ENVELOPE_FALLBACK = object()


def _envelope_helpers_unchanged():
    if _ENVELOPE_HELPERS is None or not _rotation_helpers_unchanged():
        return False
    bindings, states, registry, cython, xp = _ENVELOPE_HELPERS
    if any(vars(owner).get(name) is not original for owner, name, original in bindings):
        return False
    if len(registry) != 1 or registry.get(xp) is not cython:
        return False
    return all(_function_state(state[0]) == state for state in states)


def envelopes(tool, parameters, position, quaternion, inflation=0.):
    if (native_backend.cpp_enabled() and _envelope_helpers_unchanged()
            and type(tool) is dict and type(parameters) is dict
            and _ordinary_vector(position, 3) and _ordinary_vector(quaternion, 4)
            and type(inflation) in (float, int, np.float64)):
        result = native_backend.motion_envelopes(tool, parameters, position, quaternion, inflation,
                                                 Rotation, _ENVELOPE_FALLBACK)
        if result is not _ENVELOPE_FALLBACK:
            return result
    return _envelopes_reference(tool, parameters, position, quaternion, inflation)


def _swept_envelopes_reference(tool, parameters, start, end):
    a, b = np.asarray(start['position']), np.asarray(end['position'])
    rotations = Rotation.from_quat([start['quaternion'], end['quaternion']])
    angle = (rotations[1] * rotations[0].inv()).magnitude()
    # Every centerline lies within 42 mm of the tip; shortest quaternion arc.
    inflation = np.linalg.norm(b - a) / 2 + 84 * np.sin(angle / 4)
    q = Slerp([0., 1.], rotations)(.5).as_quat()
    return envelopes(tool, parameters, (a + b) / 2, q, inflation)


def swept_envelopes(tool, parameters, start, end):
    if not native_backend.cpp_enabled() or not _rotation_helpers_unchanged():
        return _swept_envelopes_reference(tool, parameters, start, end)
    if type(start) is not dict or type(end) is not dict:
        return _swept_envelopes_reference(tool, parameters, start, end)
    first, second = start['quaternion'], end['quaternion']
    if (not _ordinary_vector(first, 4) or not _ordinary_vector(second, 4)
            or not _ordinary_vector(start['position'], 3) or not _ordinary_vector(end['position'], 3)):
        return _swept_envelopes_reference(tool, parameters, start, end)
    a, b = np.asarray(start['position']), np.asarray(end['position'])
    # Guards reuse the same orientations; cache only immutable angular work.
    angle, q_bytes = _sweep_rotation(np.asarray([first, second], dtype=np.float64).tobytes())
    inflation = np.linalg.norm(b - a) / 2 + 84 * np.sin(angle / 4)
    q = np.frombuffer(q_bytes, dtype=np.float64).copy()
    return envelopes(tool, parameters, (a + b) / 2, q, inflation)


def _trajectory_quaternion(interpolation, index, count, chunk, rotation_type, slerp_type):
    """Reuse bounded stock SciPy trajectory work; custom dispatch stays scalar."""
    if (rotation_type is not _ROTATION or slerp_type is not _SLERP
            or type(interpolation) is not _SLERP or not native_backend.cpp_enabled()
            or not _rotation_helpers_unchanged()):
        return None, None
    # Unit rotations and (0, 1] times bound arithmetic; underflow remains observable.
    if chunk is None or not chunk[0] <= index < chunk[0] + len(chunk[1]):
        times = [(j + 1) / count for j in range(index, min(index + 32, count))]
        chunk = index, interpolation(times).as_quat()
    return chunk[1][index - chunk[0]].copy(), chunk


def _capture_contact_helpers():
    result = []
    for owner, names in ((_contact_geometry, ('segments_within_distance', 'segment_segment_distance',
                                             'point_segment_distances')),
                         (native_backend, ('try_segment_predicate', 'try_segment_batch', '_check_live'))):
        for name in names:
            function = vars(owner).get(name) if type(owner) is ModuleType else None
            result.append((owner, name, _function_state(function) if type(function) is FunctionType else None))
    return tuple(result)


_CONTACT_HELPERS = _capture_contact_helpers()
_CONTACT_NUMPY_NAMES = ('array', 'float64', 'ndarray', 'dtype')
_CONTACT_NUMPY = tuple(vars(np).get(name) for name in _CONTACT_NUMPY_NAMES)
_CONTACT_EPS = _contact_geometry._EPS
_CONTACT_ANY = any
_CONTACT_MATH = sys.modules.get('math')
_CONTACT_FINITE = vars(_CONTACT_MATH).get('isfinite') if type(_CONTACT_MATH) is ModuleType else None
_CONTACT_ADAPTER_BUILTINS = (('type', ().__class__.__class__), ('float', (0.).__class__), ('len', len))


def _source_contact_helpers_genuine():
    try:
        for owner, name, state in _CONTACT_HELPERS:
            if state is None:
                return None
            function, code, defaults, keywords = state
            if (function.__globals__ is not vars(owner) or function.__module__ != owner.__name__
                    or defaults is not None or function.__kwdefaults__ is not None):
                return None
            body = owner.__spec__.loader.get_code(owner.__name__)
            original, = (value for value in body.co_consts if type(value) is CodeType and value.co_name == name)
            if code != original:
                return None
        adapter_builtins = _CONTACT_HELPERS[4][2][0].__builtins__
        if type(adapter_builtins) is not dict:
            return None
        for name, original in _CONTACT_ADAPTER_BUILTINS:
            if vars(native_backend).get(name, original) is not original or adapter_builtins.get(name) is not original:
                return None
        length = _CONTACT_ADAPTER_BUILTINS[2][1]
        if (type(length) is not BuiltinFunctionType or length.__name__ != 'len'
                or length.__self__ is not sys.modules['builtins']):
            return None
        array, scalar, ndarray, dtype = _CONTACT_NUMPY
        if type(array) is not BuiltinFunctionType or type(array.__self__) is not ModuleType:
            return None
        owner = array.__self__
        if (owner.__name__ not in ('numpy._core._multiarray_umath', 'numpy.core._multiarray_umath')
                or sys.modules.get(owner.__name__) is not owner or array.__name__ != 'array'
                or vars(owner).get('array') is not array or vars(owner).get('ndarray') is not ndarray
                or vars(owner).get('dtype') is not dtype or type(_CONTACT_EPS) is not float
                or _CONTACT_EPS != 1e-9 or type(_CONTACT_ANY) is not BuiltinFunctionType
                or _CONTACT_ANY.__name__ != 'any' or _CONTACT_ANY.__module__ != 'builtins'
                or _CONTACT_ANY.__self__ is not sys.modules['builtins']
                or native_backend.math is not _CONTACT_MATH or type(_CONTACT_FINITE) is not BuiltinFunctionType
                or _CONTACT_FINITE.__name__ != 'isfinite' or _CONTACT_FINITE.__self__ is not _CONTACT_MATH):
            return None
        example = array([], dtype='float64')
        return example.dtype if type(example) is ndarray and example.dtype.type is scalar else None
    except (AttributeError, ImportError, KeyError, OSError, NotImplementedError, TypeError, ValueError):
        return None


_CONTACT_DTYPE = _source_contact_helpers_genuine()


def _contact_helpers_unchanged(predicate):
    if (_CONTACT_DTYPE is None or predicate is not _CONTACT_HELPERS[0][2][0]
            or _contact_geometry._EPS is not _CONTACT_EPS or _contact_geometry.np is not np
            or native_backend.np is not np or _contact_geometry.native_backend is not native_backend
            or native_backend.math is not _CONTACT_MATH or _CONTACT_MATH.isfinite is not _CONTACT_FINITE
            or _contact_geometry._segment_predicate_reference is not _CONTACT_HELPERS[1][2][0]
            or _contact_geometry._segment_predicate_point is not _CONTACT_HELPERS[2][2][0]):
        return False
    codes = _contact_geometry._segment_predicate_codes
    if (type(codes) is not tuple or len(codes) != 2 or codes[0] is not _CONTACT_HELPERS[1][2][1]
            or codes[1] is not _CONTACT_HELPERS[2][2][1]):
        return False
    for owner, name, state in _CONTACT_HELPERS:
        function = state[0]
        if (vars(owner).get(name) is not function or function.__code__ is not state[1]
                or function.__defaults__ is not state[2] or function.__kwdefaults__ is not None):
            return False
    for name, original in zip(_CONTACT_NUMPY_NAMES, _CONTACT_NUMPY):
        if vars(np).get(name) is not original:
            return False
    for name, original in _CONTACT_ADAPTER_BUILTINS:
        if (vars(native_backend).get(name, original) is not original
                or _CONTACT_HELPERS[4][2][0].__builtins__.get(name) is not original):
            return False
    return True


def _try_capsule_contact(first, second, predicate):
    """First ordered contact index, -1 for clear, or None for scalar authority."""
    # Presence is only a hint; the adapter owns both live checks and publication.
    if (native_backend._module is None or type(first) not in (list, tuple)
            or type(second) not in (list, tuple) or not 2 <= len(first) * len(second) <= 64
            or not _contact_helpers_unchanged(predicate)):
        return None
    for capsules in (first, second):
        for capsule in capsules:
            if type(capsule) not in (list, tuple) or len(capsule) != 4:
                return None
            for point in capsule[:2]:
                if (type(point) is not _CONTACT_NUMPY[2] or point.shape != (3,)
                        or type(point.dtype) is not type(_CONTACT_DTYPE) or point.dtype != _CONTACT_DTYPE):
                    return None
            width = capsule[2]
            if type(width) is not float and type(width) is not _CONTACT_NUMPY[1]:
                return None
            # Bounded nonnegative operands make eager typed additions harmless.
            if not 0. <= (0.).__class__(width) <= 1e100:
                return None
    endpoints = np.array([(a, b, c, d) for a, b, _, _ in first for c, d, _, _ in second], dtype=np.float64)
    thresholds = np.array([ar + br + 1e-9 for _, _, ar, _ in first for _, _, br, _ in second], dtype=np.float64)
    result = native_backend.try_segment_batch(endpoints, thresholds, _contact_geometry._EPS)
    return None if result is None or result[1] < 0 else result[0]


def _held_cell_contact(start, end, radius, capsules, key, source_id, grip, predicate):
    if (native_backend._module is None or type(capsules) is not dict or type(key) is not str
            or type(source_id) is not int or type(grip) is not set
            or not all(type(value) is int for value in grip)):
        return None
    for hand, segments in capsules.items():
        if type(hand) is not str or type(segments) not in (list, tuple):
            return None
        for segment in segments:
            if type(segment) not in (list, tuple) or len(segment) != 4 or type(segment[3]) is not str:
                return None
    selected, hands = [], []
    for hand, segments in capsules.items():
        for segment in segments:
            if hand == key and segment[3] != 'shaft' and source_id in grip:
                continue
            selected.append(segment); hands.append(hand)
    result = _try_capsule_contact(((start, end, radius, None),), selected, predicate)
    return None if result is None else (False if result < 0 else hands[result])


def pair_conflict(first, second):
    if any is _CONTACT_ANY:
        result = _try_capsule_contact(first, second, segments_within_distance)
        if result is not None:
            return result >= 0
    return any(segments_within_distance(a, b, c, d, ar + br + 1e-9)
               for a, b, ar, _ in first for c, d, br, _ in second)

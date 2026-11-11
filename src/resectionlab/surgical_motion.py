"""Whole-instrument capsule envelopes including roll-dependent jaws/blades.

Inflated midpoint capsules conservatively enclose translation and rotation over
each deterministic substep. Working geometry never shrinks the physical shaft.
"""
from collections import OrderedDict
import sys
from types import BuiltinFunctionType, CodeType, FunctionType, ModuleType

import numpy as np
from scipy.spatial.transform import Rotation, Slerp
from .geometry import segments_within_distance
from . import native_backend


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
                                     'from_rotvec', 'inv', 'magnitude', 'as_quat', 'as_rotvec')),
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


def envelopes(tool, parameters, position, quaternion, inflation=0.):
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


def pair_conflict(first, second):
    return any(segments_within_distance(a, b, c, d, ar + br + 1e-9)
               for a, b, ar, _ in first for c, d, br, _ in second)

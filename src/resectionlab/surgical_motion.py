"""Whole-instrument capsule envelopes including roll-dependent jaws/blades.

Inflated midpoint capsules conservatively enclose translation and rotation over
each deterministic substep. Working geometry never shrinks the physical shaft.
"""
from collections import OrderedDict
import sys
from types import CodeType, FunctionType

import numpy as np
from scipy.spatial.transform import Rotation, Slerp
from .geometry import segments_within_distance
from . import native_backend


def _function_state(function):
    function = getattr(function, '__func__', function)
    keywords = getattr(function, '__kwdefaults__', None)
    return (function, getattr(function, '__code__', None),
            getattr(function, '__defaults__', None),
            tuple((id(key), key, id(value), value) for key, value in keywords.items())
            if type(keywords) is dict else (() if keywords is None else object()))


_ROTATION = Rotation
_SLERP = Slerp
_ROTATION_HELPERS = tuple(
    (owner, name, _function_state(getattr(owner, name)))
    for owner, names in ((Rotation, ('__init__', '__len__', '__getitem__', '__mul__', 'from_quat',
                                     'from_rotvec', 'inv', 'magnitude', 'as_quat', 'as_rotvec')),
                         (Slerp, ('__init__', '__call__')))
    for name in names)


def _source_rotation_helpers_genuine():
    # shortcut: source-qualify Python wrappers; qualify native-only versions if profiling warrants reuse.
    try:
        classes = {}
        for owner, name, state in _ROTATION_HELPERS:
            if type(owner) is not type or type(owner.__module__) is not str:
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
    for function, expected in ((_SWEEP_BODY, _SWEEP_STATE), *(
            (getattr(owner, name), state) for owner, name, state in _ROTATION_HELPERS)):
        current = _function_state(function)
        if (any(current[index] is not expected[index] for index in range(3))
                or current[3] != expected[3]):
            return False
    return True


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


def pair_conflict(first, second):
    return any(segments_within_distance(a, b, c, d, ar + br + 1e-9)
               for a, b, ar, _ in first for c, d, br, _ in second)

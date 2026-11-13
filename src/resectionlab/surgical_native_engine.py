"""Conversion at the Python boundary of the resident C++ simulator."""
from dataclasses import asdict
import copy
import importlib
import sys
from types import CodeType, FunctionType, ModuleType, BuiltinFunctionType

from . import native_backend

_FIELDS = ('state', 'actions', 'history')
_MODULES = ('surgical_engine', 'geometry', 'surgical_motion', 'surgical_tools',
    'surgical_effects', 'surgical_physics', 'surgical_cusa_v3', 'surgical_clock',
    'surgical_craniotomy', 'surgical_bone_transport', 'surgical_membrane_transport',
    'surgical_cranial_stages', 'surgical_cranial_guards', 'surgical_task_domain')
_stock_capture = None
_stock_bindings = ()
_stock_constants = ()
_stock_constant_json = None
_numeric_capture = None
_read_capture = None
_read_bindings = ()


def _read_current(operation):
    """Watch only helpers omitted by this native read, using existing signatures."""
    from .surgical_proof_cache import _default
    if _read_capture is None:
        return False
    pairs = []
    for group, owner, key, function, signature in _read_capture:
        if group != 'hash' and group != operation and operation != 'all':
            continue
        try:
            current = _default(function, _pairs=pairs)
        except (TypeError, ValueError, RecursionError):
            return False
        if vars(owner).get(key) is not function or current != signature:
            return False
    if any(vars(owner).get(key) is not value for owner, key, value in _read_bindings):
        return False
    from .surgical_engine import _engine_copy_hooks_unchanged
    return (_engine_copy_hooks_unchanged() and native_backend._json_encoder_guard is not None
        and native_backend._module.simulation_json_guard(native_backend._json_encoder_guard))


def _numeric_qualified():
    global _numeric_capture
    if _numeric_capture is None:
        from .surgical_native_numeric_profile import qualified_modules
        qualified, reason = qualified_modules()
        if qualified is None:
            return False
        captures = []
        for module, path, expected in qualified:
            native_backend._files[path] = (native_backend._stat(path), expected)
            captures.append((module.__name__, module, module.__file__))
        _numeric_capture = tuple(captures)
    return all(sys.modules.get(name) is module and module.__file__ == path for name, module, path in _numeric_capture)


def _codes(code):
    yield code
    for item in code.co_consts:
        if type(item) is CodeType:
            yield from _codes(item)


def _stock(engine):
    global _stock_capture, _stock_bindings, _stock_constants, _stock_constant_json
    global _read_capture, _read_bindings
    from .surgical_proof_cache import _proof_dispatch, _default
    if not _proof_dispatch():
        return False
    from .surgical_engine import SurgicalEngine
    from .surgical_scene import SurgicalScene
    from .surgical_rl_protection import GuardConfig, ProtectionSet
    if type(engine) is not SurgicalEngine or any(name in engine.__dict__
            for name, method in vars(SurgicalEngine).items() if type(method) is FunctionType):
        return False
    if (type(engine.scene) is not SurgicalScene or
            engine.protection is not None and type(engine.protection) is not ProtectionSet or
            engine.guard_config is not None and type(engine.guard_config) is not GuardConfig):
        return False
    if type(engine.scene.properties) is dict and any(name in engine.scene.properties
            for name in ('tissuePhysicsV2', 'brainContinuumV1', 'vascularGraphV1',
                         'bipolarThermalV1', 'planarBladeManeuverV1', 'retractorSurfaceContactV1')):
        return False
    if not _numeric_qualified():
        return False
    if _stock_capture is None:
        captures = []
        bindings, constants = [], []
        dispatch = []
        for name in _MODULES:
            module = importlib.import_module('resectionlab.' + name)
            declared = {}
            for code in _codes(module.__spec__.loader.get_code(module.__name__)):
                declared.setdefault(code.co_qualname, []).append(code)
            for key, value in tuple(vars(module).items()):
                if type(value) in (ModuleType, BuiltinFunctionType, FunctionType, type):
                    bindings.append((module, key, value))
                if key.isupper():
                    try:
                        native_backend._module.simulation_json(value)
                    except (TypeError, ValueError, OverflowError):
                        pass
                    else:
                        constants.append((module, key))
                owners = [(module, key, value)]
                if type(value) is type and value.__module__ == module.__name__:
                    owners = [(value, key, method) for key, method in vars(value).items()]
                for owner, key, method in owners:
                    if type(method) in (staticmethod, classmethod):
                        method = method.__func__
                    elif type(method) is property:
                        method = method.fget
                    expected_name = key if owner is module else owner.__qualname__ + '.' + key
                    if expected_name in declared and (type(method) is not FunctionType
                            or method.__code__ not in declared[expected_name]):
                        return False
                    if type(method) is FunctionType and method.__module__ == module.__name__:
                        codes = declared.get(method.__qualname__)
                        if codes is None:
                            continue  # Dataclass-generated methods have no source body.
                        if method.__code__ not in codes:
                            return False
                        try:
                            signature = _default(method, _pairs=dispatch)
                        except (TypeError, ValueError, RecursionError):
                            return False
                        captures.append((owner, key, vars(owner)[key], method, signature))
        _stock_capture = tuple(captures)
        _stock_bindings, _stock_constants = tuple(bindings), tuple(constants)
        _stock_constant_json = native_backend._module.simulation_json(
            [vars(owner)[key] for owner,key in _stock_constants])
        reads = []
        for group, name, keys in (
                ('hash', 'core', ('semantic_digest',)),
                ('hash', 'surgical_engine', ('_owned_digest', '_owned_digest_reference')),
                ('metrics', 'surgical_physics', ('physics_metrics',)),
                ('metrics', 'surgical_physics_v2', ('metrics',)),
                ('metrics', 'surgical_cusa_v3', ('metrics',)),
                ('clone', 'surgical_engine', ('_copy_engine_data', '_copy_engine_data_reference'))):
            owner = importlib.import_module('resectionlab.' + name)
            declared = tuple(_codes(owner.__spec__.loader.get_code(owner.__name__)))
            for key in keys:
                function = vars(owner)[key]
                if type(function) is not FunctionType or function.__code__ not in declared:
                    return False
                reads.append((group, owner, key, function, _default(function, _pairs=[])))
        from . import core
        _read_bindings = tuple((owner, key, vars(owner)[key]) for owner, keys in (
            (core.json, ('dumps',)), (core.hashlib, ('sha256',)),
            (core, ('native_json_copy',))) for key in keys)
        _read_capture = tuple(reads)
    dispatch = []
    for owner, key, raw, method, signature in _stock_capture:
        try:
            current = _default(method, _pairs=dispatch)
        except (TypeError, ValueError, RecursionError):
            return False
        if vars(owner).get(key) is not raw or current != signature:
            return False
    if any(vars(owner).get(key) is not value for owner,key,value in _stock_bindings):
        return False
    try:
        if native_backend._module.simulation_json([vars(owner)[key] for owner,key in _stock_constants]) != _stock_constant_json:
            return False
    except (TypeError, ValueError, OverflowError):
        return False
    from .surgical_motion import _rotation_helpers_unchanged
    return _rotation_helpers_unchanged() and _read_current('all')


def create(engine):
    """Copy a validated source once; all action work then owns native data."""
    if not native_backend.cpp_enabled():
        return None, 'reference-selected'
    scene = engine.scene
    scene.require_cache_source()
    from .surgical_craniotomy import _original_targets, validate_contract
    access = validate_contract(scene) if engine._cranial_access is not None else None
    stages = engine._cranial_stages
    if stages is not None:
        from .surgical_cranial_stages import validate_stage_contract
        stages = validate_stage_contract(engine, stages)
    metadata = {'regions': scene.regions, 'properties': scene.properties,
        'source': scene.source, 'identity': scene.identity,
        'protectionContract': engine._protection_contract,
        'guardConfig': None if engine.guard_config is None else asdict(engine.guard_config),
        'cranialAccess': None if access is None else access.as_dict(),
        'cranialStages': None if stages is None else stages.as_dict(),
        'protectedCells': sorted(engine._protected_ids),
        'originalTargetMemberships': None if scene._original_target_memberships is None else {
            key: value.tolist() for key, value in scene._original_target_memberships.items()},
        'inspectionOnly': engine._source_inspection_only,
        'initialStateId': getattr(engine, '_native_initial_state_id', engine.state_id)}
    if access is not None:
        metadata['originalTargetCells'] = sorted(_original_targets(scene))
    data = {'state': engine.state, 'actions': engine.actions, 'history': engine.history}
    try:
        owner, reason = native_backend._module.simulation_create(scene.labels, scene.affine,
            scene.inverse, scene.spacing, scene.voxel_mm3, metadata, data)
    except native_backend._module.UnsupportedSimulationBoundary:
        return None, 'unsupported-boundary-value'
    if owner is not None:
        engine.__dict__['_resident_source'] = _source_facts(engine)
    return owner, reason


def _source_facts(engine):
    protection, guard = engine.protection, engine.guard_config
    return (engine.scene, protection, guard, engine._protected_ids,
        engine._cranial_access, engine._cranial_stages,
        None if guard is None else (guard.require_critical_coverage, guard.critical_max_displacement_mm),
        None if protection is None else (protection.clearance_mm, protection.cells, protection.annotations, protection.required))


def _source_current(engine):
    before = engine.__dict__.get('_resident_source')
    if before is not None:
        if any(a is not b for a,b in zip(before[:6],(engine.scene,engine.protection,engine.guard_config,
                engine._protected_ids,engine._cranial_access,engine._cranial_stages))):
            return False
        current = _source_facts(engine)
        if any(a is not b for a, b in zip(before[:6], current[:6])) or before[6] != current[6]:
            return False
        if before[7] is not None:
            if before[7][0] != current[7][0] or any(a is not b for a,b in zip(before[7][1:],current[7][1:])):
                return False
    try:
        engine.scene.require_cache_source()
        if engine.protection is not None:
            type(engine.protection)._require_geometry(engine.protection)
        if engine._cranial_stages is not None:
            from .surgical_cranial_stages import validate_stage_contract
            validate_stage_contract(engine, engine._cranial_stages)
    except ValueError:
        return False
    return True


def field(engine, name):
    values = engine.__dict__
    if name not in values and values.get('_resident') is not None:
        native_backend._check_live()
        values[name] = native_backend._module.simulation_data(values['_resident'], name)
    return values[name]


def sync(engine, operation='hash'):
    owner = engine.__dict__.get('_resident')
    if owner is None:
        return False
    native_backend._check_live()
    if not _read_current(operation) or not _source_current(engine):
        detach(engine)
        return False
    fields = {name: engine.__dict__[name] for name in _FIELDS if name in engine.__dict__}
    if fields:
        try:
            reason = native_backend._module.simulation_import(owner, fields)
        except native_backend._module.UnsupportedSimulationBoundary:
            detach(engine)
            return False
        if reason is not None:
            detach(engine)
            return False
    return True


def detach(engine):
    owner = engine.__dict__.get('_resident')
    if owner is None:
        return
    prefix = native_backend._module.simulation_query(owner, 'bonePrefix')
    for name in _FIELDS:
        field(engine, name)
    engine.__dict__['_resident'] = None
    if prefix is not None:
        from . import surgical_cranial_stages as stages
        proof = stages.NativeBonePrefix(**{key: value for key, value in prefix.items() if key != 'opening_summary'})
        stages._prefixes[id(proof)] = proof
        stages._prefix_facts[id(proof)] = tuple((f.name, getattr(proof, f.name)) for f in stages.fields(proof))
        stages.weakref.finalize(proof, stages._prefix_facts.pop, id(proof), None)
        engine._cranial_stage_bone_prefix = proof
        engine._cranial_stage_permission = None


def clone(engine):
    if not sync(engine, 'clone'):
        return None
    result = copy.copy(engine)
    result.__dict__['_resident'] = native_backend._module.simulation_clone(engine.__dict__['_resident'])
    for name in _FIELDS:
        result.__dict__.pop(name, None)
    return result


def prepare(engine, action, check):
    if native_backend._module is None:
        return NotImplemented
    from .surgical_native_control import native_stop_token, translate_native_stop
    token = native_stop_token(check) if check is not None else None
    allowed = ((check is None or token is not None) and not engine.__dict__.get('_native_disabled')
        and not engine.__dict__.get('_restoring_legacy_guard_partial')
        and not engine.__dict__.get('_legacy_guard_partial_actions') and _stock(engine) and _source_current(engine))
    if not allowed:
        detach(engine)
        return NotImplemented
    if not native_backend.cpp_enabled():
        return NotImplemented
    if engine.__dict__.get('_resident') is None:
        owner, reason = create(engine)
        if owner is None:
            engine.__dict__['_native_disabled'] = reason
            return NotImplemented
        engine.__dict__['_resident'] = owner
    if not sync(engine):
        return NotImplemented
    if token is None:
        token = native_backend._module.SimulationStopToken()
    if check is not None:
        check()
    try:
        owner, result = native_backend._module.simulation_prepare(engine.__dict__['_resident'], action, token)
        if result is None:
            detach(engine)
            return NotImplemented
        native_backend._check_live()
        if check is not None:
            check()
    except RuntimeError as error:
        translate_native_stop(check, error)
    candidate = None
    if owner is not None:
        candidate = copy.copy(engine)
        candidate.__dict__['_resident'] = owner
        for name in _FIELDS:
            candidate.__dict__.pop(name, None)
    native_backend._counters['residentActionCalls'] += 1
    return candidate, result


def query(engine, name):
    if not _source_current(engine):
        detach(engine)
        return NotImplemented
    if not sync(engine):
        return NotImplemented
    owner = engine.__dict__['_resident']
    return native_backend._module.simulation_query(owner, name)

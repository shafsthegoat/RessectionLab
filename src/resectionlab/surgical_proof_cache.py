"""Process-local authority for canonical strategy verdicts, never model state.

Content hashes alone cannot authorize reuse after implementation changes. Every
public proof boundary qualifies its original DTO/scene dispatch, binds loaded
physics/planning/source implementation, and rechecks that binding before return.
Unknown/custom inputs keep their original validation path with reuse disabled.
"""
from collections import OrderedDict
from contextlib import contextmanager
from contextvars import ContextVar
import dataclasses
import abc
import enum
import importlib
import math
from operator import is_ as _same_object
from pathlib import Path
import sys
import typing
from threading import RLock
from types import FunctionType, ModuleType, BuiltinFunctionType, MethodDescriptorType, WrapperDescriptorType
import weakref
import builtins as _builtins

# Qualify captures before trusting their identity. These checks use intrinsic
# types and exact dictionaries, never the mutable builtins they are checking.
_intrinsic_type = ().__class__.__class__
_intrinsic_dict = {}.__class__
_intrinsic_function = (lambda: None).__class__
_intrinsic_c_function = [].append.__class__
_builtin_names = ("__build_class__ __import__ abs all any ascii bin breakpoint callable chr "
    "compile delattr dir divmod eval exec format getattr globals hasattr hash hex id input "
    "isinstance issubclass iter aiter len locals max min next anext oct ord pow print repr "
    "round setattr sorted sum vars bool memoryview bytearray bytes classmethod complex dict "
    "enumerate filter float frozenset property int list map object range reversed set slice "
    "staticmethod str super tuple type zip BaseException BaseExceptionGroup Exception "
    "GeneratorExit KeyboardInterrupt SystemExit ArithmeticError AssertionError AttributeError "
    "BufferError EOFError ImportError LookupError MemoryError NameError OSError ReferenceError "
    "RuntimeError StopAsyncIteration StopIteration SyntaxError SystemError TypeError ValueError "
    "Warning FloatingPointError OverflowError ZeroDivisionError BytesWarning DeprecationWarning "
    "EncodingWarning FutureWarning ImportWarning PendingDeprecationWarning ResourceWarning "
    "RuntimeWarning SyntaxWarning UnicodeWarning UserWarning BlockingIOError ChildProcessError "
    "ConnectionError FileExistsError FileNotFoundError InterruptedError IsADirectoryError "
    "NotADirectoryError PermissionError ProcessLookupError TimeoutError IndentationError "
    "IndexError KeyError ModuleNotFoundError NotImplementedError RecursionError UnboundLocalError "
    "UnicodeError BrokenPipeError ConnectionAbortedError ConnectionRefusedError ConnectionResetError "
    "TabError UnicodeDecodeError UnicodeEncodeError UnicodeTranslateError").split()
_builtin_bindings = ().__class__((name, _builtins.__dict__.get(name)) for name in _builtin_names)


def _genuine_builtins():
    for name, value in _builtin_bindings:
        if _intrinsic_type(value) is _intrinsic_c_function:
            if (value.__self__ is not _builtins or value.__name__ != name or
                    value.__module__ != 'builtins'): return False
        elif _intrinsic_type(value) is _intrinsic_type:
            if value.__flags__ & 512 or value.__name__ != name or value.__module__ != 'builtins':
                return False
        else: return False
    return True


_builtin_captures_genuine = _genuine_builtins()


def _dispatch(function, pairs):
    """Check the function's effective lookup maps, including copied builtins."""
    namespace, mapping = function.__globals__, function.__builtins__
    if (_intrinsic_type(namespace) is not _intrinsic_dict or
            _intrinsic_type(mapping) is not _intrinsic_dict): return False
    # Identity pairs avoid repeating the full map check for every class method.
    for previous_globals, previous_builtins in pairs:
        if namespace is previous_globals and mapping is previous_builtins: return True
    module_name = namespace.get('__name__')
    if _intrinsic_type(module_name) is not ''.__class__: return False
    project = module_name.startswith('resectionlab.')
    for name, original in _builtin_bindings:
        if mapping.get(name) is not original: return False
        # Foreign library globals have their own supported numerical aliases.
        # Our validators must resolve builtin names to the qualified originals.
        if project and namespace.get(name, original) is not original: return False
    pairs.append((namespace, mapping))
    return True


def _proof_dispatch():
    if not _builtin_captures_genuine: return False
    pairs = []
    for value in _proof_globals.values():
        if _intrinsic_type(value) is _intrinsic_function:
            if not _dispatch(value, pairs): return False
        elif _intrinsic_type(value) is _intrinsic_type and value.__module__ == __name__:
            for method in value.__dict__.values():
                if _intrinsic_type(method) is _intrinsic_function and not _dispatch(method, pairs):
                    return False
    return True


_proof_globals = _proof_dispatch.__globals__


def _cache_dispatch():
    if _proof_dispatch(): return True
    owner = _ACTIVE.get()
    if owner is not None and not owner.closed and owner.allowed: owner.taint()
    return False

# Fixed semantic domain, imported before capturing an epoch. Imported function
# aliases are included too; lazy source/planning modules cannot launder a lower
# content-only verdict into a new implementation's authority.
_DOMAIN = ('surgical_proof_cache', 'surgical_strategy_workspace', 'surgical_two_tool',
    'surgical_strategy_sensitivity', 'surgical_anatomical_sensitivity',
    'surgical_planning_observation', 'surgical_anatomical_encounter',
    'surgical_task_admission', 'surgical_source_decode', 'surgical_patient',
    'patient_anatomy', 'surgical_interactive_planning', 'surgical_interactive_workspace',
    'surgical_preparation_planning', 'surgical_preparation_workspace', 'worlds')
_ACTIVE = ContextVar('canonical_strategy_proof', default=None)
_LOCK = RLock()
_LAST_SIGNATURE = None
_LAST_FAST = None
_LAST_EPOCH = 0
_CACHES = []
_UNAVAILABLE = object()
_LIBRARY_TYPES = (abc.ABCMeta, enum.EnumType, type(typing.Any), type(typing.Iterable))
_LIBRARY_PREFIXES = ('resectionlab.', 'numpy', 'scipy.', 'builtins', 'pathlib', 'types',
    'collections', 'threading', '_thread', 'dataclasses', 'enum', 'fractions',
    'contextvars', '_contextvars', 'weakref', 'typing', 'abc', 'datetime', 'itertools', 'array')
_LIGHT_MODULES = ('surgical_frame_replay', 'surgical_observed_cache', 'surgical_telemetry')
_BINARIES = ('numpy._core._multiarray_umath', 'scipy.spatial._ckdtree',
    'scipy.spatial.transform._rotation', 'scipy.integrate._odepack',
    'scipy.ndimage._nd_image', 'scipy.linalg._fblas', 'scipy.linalg._flapack')


class _Identity:
    def __init__(self, value): self.value = value
    def __eq__(self, other): return type(other) is _Identity and self.value is other.value


def _closure(value, dispatch):
    # A callable capturing caller data is not an implementation certificate.
    # The empty set used by dataclass repr recursion is harmless bookkeeping.
    kind = type(value)
    if kind is dict or kind is list or (kind is set and value):
        raise TypeError('Mutable closure is not proof authority')
    if kind is tuple: return (tuple, tuple(_closure(v, dispatch) for v in value))
    return _default(value, _pairs=dispatch)


def _default(value, *, _linked=False, _pairs=None):
    kind = type(value)
    if value is None or kind is bool or kind is int or kind is str or kind is bytes:
        return (kind, value)
    if kind is float: return (float, value.hex())
    if kind is FunctionType:
        if _pairs is not None and not _dispatch(value, _pairs):
            raise TypeError('Changed effective builtin dispatch')
        if type(value.__module__) is not str: raise TypeError('Unknown function module')
        if not value.__module__.startswith('resectionlab.'):
            owner = sys.modules.get(value.__module__)
            enum_method = vars(enum.Enum).get(value.__name__)
            if type(enum_method) is staticmethod or type(enum_method) is classmethod: enum_method = enum_method.__func__
            if (type(owner) is not ModuleType or
                    not value.__module__.startswith(('scipy.', 'numpy.', 'dataclasses', 'contextlib',
                        'copy', 'threading', 'urllib.parse', 'enum')) or
                    (not _linked and vars(owner).get(value.__name__) is not value and
                     not (value.__module__ == 'enum' and enum_method is value))):
                raise TypeError('Custom proof function')
        return ('function', value, value.__code__, id(value.__code__),
                _default(value.__defaults__, _pairs=_pairs), _default(value.__kwdefaults__, _pairs=_pairs),
                tuple((_default(cell.cell_contents, _linked=True, _pairs=_pairs)
                    if cell.cell_contents is vars(value).get('__wrapped__') else _closure(cell.cell_contents, _pairs))
                    for cell in (value.__closure__ or ())))
    if kind is tuple or kind is list:
        return (kind, tuple(_default(v, _pairs=_pairs) for v in value))
    if kind is dict and all(type(k) is str for k in value):
        return (dict, tuple((k, _default(v, _pairs=_pairs)) for k, v in sorted(value.items())))
    if kind is set or kind is frozenset:
        return (kind, frozenset(_default(v, _pairs=_pairs) for v in value))
    if kind is type: return ('type', value)
    if kind is BuiltinFunctionType and any(type(value.__self__) is kind for kind in (ModuleType, type)):
        return ('builtin', value)
    if kind is MethodDescriptorType or kind is WrapperDescriptorType:
        return ('descriptor', value)
    if value is dataclasses.MISSING or value is dataclasses._HAS_DEFAULT_FACTORY:
        return ('dataclass-marker', value)
    raise TypeError('Unknown mutable proof default')


def _namespace(module, dispatch):
    result = []
    for name, value in sorted(vars(module).items()):
        if type(value) is FunctionType:
            result.append((name, _default(value, _pairs=dispatch)))
        elif type(value) is BuiltinFunctionType or type(value) is MethodDescriptorType or type(value) is WrapperDescriptorType:
            result.append((name, _default(value)))
        elif type(value) is type or any(type(value) is kind for kind in _LIBRARY_TYPES):
            origin = getattr(value, '__module__', '')
            if type(origin) is not str or not origin.startswith(_LIBRARY_PREFIXES): raise TypeError('Custom proof class')
            members = []
            if origin.startswith('resectionlab.') and type(value) is not type(typing.Iterable):
                for member, method in sorted(vars(value).items()):
                    if type(method) is staticmethod or type(method) is classmethod: method = method.__func__
                    elif type(method) is property: method = method.fget
                    if type(method) is FunctionType: members.append((member, _default(method, _pairs=dispatch)))
            result.append((name, _Identity(value), tuple(members)))
        elif type(value) is ModuleType:
            result.append((name, value))
        elif callable(value):
            raise TypeError('Custom callable cannot confer proof authority')
        elif name.isupper() and not name.startswith(('_LAST', '_VALIDATION_COUNTS')) and name not in ('_CACHES', '_UNAVAILABLE'):
            try: constant = _default(value)
            except (TypeError, RecursionError): continue
            result.append((name, constant))
    return tuple(result)


def _files():
    paths = sorted(Path(__file__).resolve().parent.glob('*.py'))
    paths += [Path(importlib.import_module(name).__file__) for name in _BINARIES]
    result = []
    for path in paths:
        info = path.stat()
        result.append((str(path), info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns))
    return tuple(result)


def implementation_epoch():
    """A process epoch retaining code/default references against identity reuse."""
    global _LAST_SIGNATURE, _LAST_EPOCH, _LAST_FAST
    # Run before imports, any/all/type, or the seal's mutable helper dispatch.
    if not _proof_dispatch(): return None
    from .surgical_verified_cache import _MODULES
    from .surgical_observed_cache import light_implementation_seal
    try:
        from .surgical_idle_native import strict_live_identity
        modules = [importlib.import_module('resectionlab.'+name) for name in dict.fromkeys((*_MODULES, *_DOMAIN, *_LIGHT_MODULES))]
        if any(type(module) is not ModuleType for module in modules): return None
        if any(getattr(module.__spec__, '_initializing', False) for module in modules): return None
        # Qualify defaults before the existing physics seal can format them.
        dispatch = []
        namespaces = tuple((m, m.__name__, vars(m).get('__file__'), _namespace(m, dispatch)) for m in modules)
        native = _default(strict_live_identity())
        files = _files()
        fast = (namespaces, native, files)
        with _LOCK:
            if fast == _LAST_FAST: return _LAST_EPOCH
        signature = (namespaces, light_implementation_seal())
        # File metadata is only a shortcut to an already hashed seal. A write,
        # replacement, timestamp/inode or loaded binding change recaptures it.
        if files != _files(): return None
    except (OSError, ValueError, TypeError, RecursionError, ImportError, AttributeError):
        return None
    with _LOCK:
        if signature != _LAST_SIGNATURE:
            _LAST_SIGNATURE = signature; _LAST_EPOCH += 1
        _LAST_FAST = fast
        return _LAST_EPOCH


def _json(value):
    pending = [(value, False)]; active = set(); done = set(); count = 0
    while pending:
        current, leaving = pending.pop(); count += 1
        if count > 1_000_000: return False
        kind = type(current)
        if kind is dict or kind is list:
            identity = id(current)
            if leaving:
                active.remove(identity); done.add(identity); continue
            if identity in done: continue
            if identity in active: return False
            active.add(identity); pending.append((current, True))
            if kind is dict:
                if any(type(k) is not str for k in current): return False
                pending.extend((v, False) for v in current.values())
            else: pending.extend((v, False) for v in current)
        elif kind is float:
            if not math.isfinite(current): return False
        elif current is not None and kind is not str and kind is not int and kind is not bool:
            return False
    return True


def _stock_scene(scene, engine):
    from .surgical_scene import SurgicalScene
    from .surgical_engine import SurgicalEngine
    if type(scene) is not SurgicalScene or SurgicalScene.__getattribute__ is not object.__getattribute__: return False
    owners = [(scene, SurgicalScene)]
    if engine is not None: owners.append((engine, SurgicalEngine))
    for owner, cls in owners:
        if type(owner) is not cls or cls.__getattribute__ is not object.__getattribute__: return False
        names = set(vars(owner))
        for base in cls.__mro__:
            if any(name in names and (callable(value) or type(value) is property)
                   for name, value in vars(base).items()): return False
    return True


def _json_certificate(values):
    """Reference-only proof of a previously qualified exact JSON graph."""
    rows = []; pending = list(values); seen = set()
    while pending:
        current = pending.pop(); kind = type(current)
        if kind is not dict and kind is not list or id(current) in seen: continue
        seen.add(id(current))
        keys = tuple(current) if kind is dict else None
        children = tuple(current.values()) if kind is dict else tuple(current)
        rows.append((current, keys, children))
        pending.extend(children)
    return tuple(rows)


def _same_json(certificate):
    for current, keys, children in certificate:
        if keys is None:
            if (type(current) is not list or list.__len__(current) != len(children) or
                    not all(map(_same_object, current, children))): return False
        elif (type(current) is not dict or dict.__len__(current) != len(keys) or
                not all(map(_same_object, current, keys)) or
                not all(map(_same_object, current.values(), children))): return False
    return True


def _clear():
    for reference in _CACHES:
        cache = reference()
        if cache is not None: OrderedDict.clear(cache)


class _Invocation:
    def __init__(self, scene, engine, inputs, parent):
        self.scene, self.engine, self.inputs, self.parent = scene, engine, inputs, parent
        self.closed = self.tainted = False
        self.epoch = implementation_epoch()
        self.allowed = (self.epoch is not None and _stock_scene(scene, engine)
                        and all(_json(value) for value in inputs))
        if self.allowed and engine is not None:
            self.allowed = _json(engine.state) and _json(engine.actions) and _json(engine.history)
        if parent is not None:
            if not parent.fence() or parent.epoch != self.epoch: self.allowed = False
        from .core import semantic_digest
        self.hashes = tuple(semantic_digest(value) for value in inputs) if self.allowed else None
        self.certificate = _json_certificate(inputs) if self.allowed else None
        self.engine_binding = self.bind_engine() if self.allowed and engine is not None else None

    def bind_engine(self):
        from .surgical_two_tool import _engine_owner_signature
        from .surgical_task_admission import admission
        from .core import semantic_digest
        return (_engine_owner_signature(self.engine, require_cold=False), semantic_digest(admission(self.engine)))

    def taint(self):
        current = self
        while current is not None:
            current.tainted = True; current.allowed = False; current = current.parent

    def fence(self):
        if self.closed or not self.allowed: return False
        if implementation_epoch() != self.epoch:
            self.taint(); return False
        current = self
        scenes = set(); engines = set()
        while current is not None:
            if current.closed or not current.allowed: return False
            if not _stock_scene(current.scene, current.engine):
                self.taint(); return False
            try:
                if id(current.scene) not in scenes:
                    current.scene.require_cache_source(); scenes.add(id(current.scene))
                if current.engine is not None and id(current.engine) not in engines:
                    if not (_json(current.engine.state) and _json(current.engine.actions) and _json(current.engine.history)):
                        self.taint(); return False
                    if current.bind_engine() != current.engine_binding:
                        self.taint(); return False
                    engines.add(id(current.engine))
            except ValueError:
                self.taint(); return False
            current = current.parent
        if not _same_json(self.certificate):
            from .core import semantic_digest
            if (not all(_json(value) for value in self.inputs) or
                    tuple(semantic_digest(value) for value in self.inputs) != self.hashes):
                self.taint(); return False
            self.certificate = _json_certificate(self.inputs)
        return True

    def finish(self):
        from .core import semantic_digest
        if self.allowed:
            self.fence()
            if self.allowed and tuple(semantic_digest(value) for value in self.inputs) != self.hashes:
                self.taint()
            if self.allowed and self.engine is not None and self.bind_engine() != self.engine_binding:
                self.taint()
        if self.tainted:
            _clear()
            raise ValueError('STRATEGY_PROOF: inputs, dispatch or implementation changed during canonical validation')

    def close(self):
        self.closed = True
        self.scene = self.engine = self.inputs = self.parent = self.hashes = self.engine_binding = self.certificate = None


@contextmanager
def proof_scope(scene, *inputs, engine=None):
    previous = _ACTIVE.get()
    if previous is not None and previous.closed: previous = None
    owner = _Invocation(scene, engine, inputs, previous)
    token = _ACTIVE.set(owner)
    try:
        yield
        owner.finish()
    finally:
        if owner.tainted: _clear()
        owner.close()
        _ACTIVE.reset(token)


class ProofCache(OrderedDict):
    """Existing bounded verdict maps, partitioned by the active proof epoch."""
    def __init__(self):
        super().__init__()
        _CACHES[:] = [reference for reference in _CACHES if reference() is not None]
        _CACHES.append(weakref.ref(self))

    def _key(self, key, *, check=True):
        if not _cache_dispatch(): return None
        if not _metadata(key): return None
        owner = _ACTIVE.get()
        if owner is None or owner.closed or not owner.allowed: return None
        return (owner.epoch, key) if not check or owner.fence() else None

    def __contains__(self, key):
        qualified = self._key(key)
        return qualified is not None and OrderedDict.__contains__(self, qualified)

    def __getitem__(self, key):
        qualified = self._key(key)
        if qualified is None: raise KeyError(key)
        return OrderedDict.__getitem__(self, qualified)

    def get(self, key, default=None):
        qualified = self._key(key)
        return default if qualified is None else OrderedDict.get(self, qualified, default)

    def __setitem__(self, key, value):
        if not _cache_dispatch(): return
        if not _metadata(value): return
        qualified = self._key(key)
        if qualified is not None: OrderedDict.__setitem__(self, qualified, value)

    def move_to_end(self, key, last=True):
        # LRU ordering carries no verdict; the preceding read already fenced
        # authority. Publication and every value read remain live-qualified.
        qualified = self._key(key, check=False)
        if qualified is not None: OrderedDict.move_to_end(self, qualified, last)


def _metadata(value):
    pending = [value]; nodes = size = 0
    while pending:
        current = pending.pop(); kind = type(current); nodes += 1
        if nodes > 4096: return False
        if kind is tuple: pending.extend(current)
        elif kind is str:
            size += len(current)
            if size > 65536: return False
        elif kind is float:
            if not math.isfinite(current): return False
        elif current is not None and kind is not int and kind is not bool:
            return False
    return True

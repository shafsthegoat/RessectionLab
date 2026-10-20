"""Bounded reuse of NPZ bytes for irreversibly immutable source arrays.

Only compressed array bytes are retained. Manifests, artifacts and source hashes
are still produced by each ordinary save; mutable array storage is never reused.
"""
from collections import OrderedDict
from io import BytesIO
from threading import RLock
import weakref

import numpy as np

MAX_CACHE_BYTES = 128 * 1024**2
MAX_CACHE_ENTRIES = 8
_entries = OrderedDict()
_lock = RLock()


def _signature(arrays):
    result = []
    for name, array in arrays.items():
        if (type(array) is not np.ndarray or array.dtype.kind not in 'buif'
                or array.dtype.fields is not None or array.dtype.subdtype is not None):
            return None
        base = array
        while type(base) is np.ndarray:
            if base.flags.writeable:
                return None
            base = base.base
        # Readonly owning arrays and readonly views of mutable memory can be
        # changed externally. Core immutable_array terminates in exact bytes.
        if type(base) is not bytes:
            return None
        result.append((name, id(array), array.shape, array.dtype.str, array.strides,
                       array.__array_interface__['data'][0]))
    return tuple(result)


def encode_array_payload(arrays):
    """Return ordinary lossless NPZ bytes, reusing only an identical frozen view."""
    signature = _signature(arrays)
    if signature is not None:
        with _lock:
            entry = _entries.get(signature)
            if entry is not None:
                references, payload = entry
                if all(reference() is array for reference, array in zip(references, arrays.values())):
                    _entries.move_to_end(signature)
                    return payload
                del _entries[signature]
    buffer = BytesIO()
    np.savez_compressed(buffer, **arrays)
    payload = buffer.getvalue()
    # Metadata such as ndarray.shape remains mutable even on bytes-backed
    # storage. Do not publish an encoding whose view changed during the save.
    if (signature is not None and signature == _signature(arrays)
            and len(payload) <= MAX_CACHE_BYTES):
        references = tuple(weakref.ref(array) for array in arrays.values())
        with _lock:
            _entries[signature] = (references, payload)
            _entries.move_to_end(signature)
            while (len(_entries) > MAX_CACHE_ENTRIES
                    or sum(len(entry[1]) for entry in _entries.values()) > MAX_CACHE_BYTES):
                _entries.popitem(last=False)
    return payload

"""Lossless bounded storage for long full-case surgical action archives.

This is a transport codec, not checkpoint admission. Decoded snapshots still
undergo the engine's normal independent replay or authenticated local cache
admission. Imaging arrays and case/source identity are outside this envelope.
"""
from __future__ import annotations

import base64
from hashlib import sha256
import gzip
from io import BytesIO
import json
import re
import zlib

SCHEMA = 'resectionlab.surgical-toolbox-archive.v1'
ENCODING = 'gzip-json-base64'
MAX_UNCOMPRESSED_BYTES = 128 * 1024**2
MAX_ENCODED_BYTES = 16 * 1024**2
_FIELDS = {'schema', 'encoding', 'sha256', 'uncompressedBytes', 'payload'}


def encode_artifacts(artifacts):
    """Encode only a fullCase toolbox; leave all other artifact values intact."""
    result = dict(artifacts)
    toolbox = result.get('surgical_toolbox')
    if not isinstance(toolbox, dict) or not toolbox.get('fullCase'):
        return result
    raw = json.dumps(toolbox, separators=(',', ':'), sort_keys=True, allow_nan=False).encode('utf-8')
    if len(raw) > MAX_UNCOMPRESSED_BYTES:
        raise ValueError('Full-case surgical archive exceeds the 128 MiB expanded JSON limit')
    payload = base64.b64encode(gzip.compress(raw, compresslevel=6, mtime=0)).decode('ascii')
    if len(payload) > MAX_ENCODED_BYTES:
        raise ValueError('Compressed full-case surgical archive exceeds the bounded manifest payload')
    result['surgical_toolbox'] = {'schema': SCHEMA, 'encoding': ENCODING,
                                  'sha256': sha256(raw).hexdigest(),
                                  'uncompressedBytes': len(raw), 'payload': payload}
    return result


def _invalid_constant(value):
    raise ValueError('Nonfinite JSON value in surgical archive: ' + value)


def decode_artifacts(artifacts):
    """Decode a recognized envelope, bounding inflation before JSON parsing."""
    result = dict(artifacts)
    envelope = result.get('surgical_toolbox')
    if not isinstance(envelope, dict):
        return result
    schema = envelope.get('schema')
    looks_encoded = {'encoding', 'uncompressedBytes', 'payload'} <= set(envelope)
    if not looks_encoded and (not isinstance(schema, str) or not schema.startswith('resectionlab.surgical-toolbox-archive.')):
        return result  # Original raw toolbox archives remain readable.
    if schema != SCHEMA or set(envelope) != _FIELDS or envelope['encoding'] != ENCODING:
        raise ValueError('Unsupported or malformed surgical archive envelope')
    length = envelope['uncompressedBytes']
    if type(length) is not int or not 1 <= length <= MAX_UNCOMPRESSED_BYTES:
        raise ValueError('Surgical archive expanded length exceeds the bounded JSON limit')
    payload, expected = envelope['payload'], envelope['sha256']
    if not isinstance(payload, str) or len(payload) > MAX_ENCODED_BYTES:
        raise ValueError('Invalid surgical archive payload length')
    if not isinstance(expected, str) or re.fullmatch('[0-9a-f]{64}', expected) is None:
        raise ValueError('Invalid surgical archive checksum declaration')
    try:
        compressed = base64.b64decode(payload, validate=True)
        with gzip.GzipFile(fileobj=BytesIO(compressed)) as stream:
            raw = stream.read(length + 1)
    except (ValueError, OSError, EOFError, zlib.error) as error:
        raise ValueError('Invalid surgical archive compression or base64 payload') from error
    if len(raw) != length:
        raise ValueError('Surgical archive expanded length differs from its declaration')
    if sha256(raw).hexdigest() != expected:
        raise ValueError('Surgical archive checksum differs')
    toolbox = json.loads(raw, parse_constant=_invalid_constant)
    if not isinstance(toolbox, dict) or not isinstance(toolbox.get('fullCase'), dict) or not toolbox['fullCase']:
        raise ValueError('Decoded surgical archive is not a full-case toolbox')
    result['surgical_toolbox'] = toolbox
    return result

"""Lossless bounded storage for long surgical action and planning archives.

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
STAGED_SCHEMA = 'resectionlab.staged-planning-archive.v1'
STAGED_REPORT_SCHEMA = 'resectionlab.staged-anatomy-planning.v1'
ENCODING = 'gzip-json-base64'
MAX_UNCOMPRESSED_BYTES = 128 * 1024**2
MAX_ENCODED_BYTES = 16 * 1024**2
_FIELDS = {'schema', 'encoding', 'sha256', 'uncompressedBytes', 'payload'}


def _encode(value, schema):
    raw = json.dumps(value, separators=(',', ':'), sort_keys=True, allow_nan=False).encode('utf-8')
    if len(raw) > MAX_UNCOMPRESSED_BYTES:
        raise ValueError('Full-case surgical archive exceeds the 128 MiB expanded JSON limit')
    payload = base64.b64encode(gzip.compress(raw, compresslevel=6, mtime=0)).decode('ascii')
    if len(payload) > MAX_ENCODED_BYTES:
        raise ValueError('Compressed full-case surgical archive exceeds the bounded manifest payload')
    return {'schema': schema, 'encoding': ENCODING,
            'sha256': sha256(raw).hexdigest(), 'uncompressedBytes': len(raw), 'payload': payload}


def encode_artifacts(artifacts):
    """Keep staged native branches raw; compress only their bulky evidence.

    Historical readers can still inspect the native source and execute frozen
    policies while ignoring the added staged-planning field. Full-case archives
    retain their existing whole-toolbox format and admission rules.
    """
    result = dict(artifacts)
    toolbox = result.get('surgical_toolbox')
    if not isinstance(toolbox, dict):
        return result
    if toolbox.get('fullCase'):
        result['surgical_toolbox'] = _encode(toolbox, SCHEMA)
    elif isinstance(toolbox.get('stagedPlanning'), dict) and \
            toolbox['stagedPlanning'].get('schema') == STAGED_REPORT_SCHEMA:
        result['surgical_toolbox'] = {**toolbox,
            'stagedPlanning': _encode(toolbox['stagedPlanning'], STAGED_SCHEMA)}
    return result


def _invalid_constant(value):
    raise ValueError('Nonfinite JSON value in surgical archive: ' + value)


def _decode(envelope, schema, max_bytes=None):
    if max_bytes is None:
        max_bytes = MAX_UNCOMPRESSED_BYTES
    if envelope.get('schema') != schema or set(envelope) != _FIELDS or envelope['encoding'] != ENCODING:
        raise ValueError('Unsupported or malformed surgical archive envelope')
    length = envelope['uncompressedBytes']
    if type(length) is not int or not 1 <= length <= max_bytes:
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
    return json.loads(raw, parse_constant=_invalid_constant), length


def _looks_encoded(value, prefix):
    return isinstance(value, dict) and (
        {'encoding', 'uncompressedBytes', 'payload'} <= set(value) or
        isinstance(value.get('schema'), str) and value['schema'].startswith(prefix))


def decode_artifacts(artifacts):
    """Decode recognized envelopes; never admit native state from this codec."""
    result = dict(artifacts)
    toolbox = result.get('surgical_toolbox')
    if not isinstance(toolbox, dict):
        return result
    remaining = MAX_UNCOMPRESSED_BYTES
    if _looks_encoded(toolbox, 'resectionlab.surgical-toolbox-archive.'):
        toolbox, length = _decode(toolbox, SCHEMA)
        if not isinstance(toolbox, dict) or not isinstance(toolbox.get('fullCase'), dict) or not toolbox['fullCase']:
            raise ValueError('Decoded surgical archive is not a full-case toolbox')
        remaining -= length
    staged = toolbox.get('stagedPlanning')
    if _looks_encoded(staged, 'resectionlab.staged-planning-archive.'):
        report, _ = _decode(staged, STAGED_SCHEMA, remaining)
        if not isinstance(report, dict) or report.get('schema') != STAGED_REPORT_SCHEMA:
            raise ValueError('Decoded staged planning archive has an unsupported report')
        toolbox = {**toolbox, 'stagedPlanning': report}
    result['surgical_toolbox'] = toolbox
    return result

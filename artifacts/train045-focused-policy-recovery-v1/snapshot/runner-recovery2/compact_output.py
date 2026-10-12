"""Bounded lossless gzip JSON transport; no schema, planning or geometry change."""
import gzip
import hashlib
import io
import json

MAX_RAW = 32*1024**2
MAX_GZIP = 3*1024**2


def canonical(value):
    return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()


def encode_gzip(value):
    raw=canonical(value)+b'\n'
    if len(raw)>MAX_RAW:raise InterruptedError('Decompressed diagnostic JSON bound')
    compressed=gzip.compress(raw,compresslevel=6,mtime=0)
    if len(compressed)>MAX_GZIP:raise InterruptedError('Compressed diagnostic JSON bound')
    return compressed,{'encoding':'gzip-json-canonical-v1','bytes':len(compressed),
        'sha256':hashlib.sha256(compressed).hexdigest(),'decompressed_bytes':len(raw),
        'decompressed_sha256':hashlib.sha256(raw).hexdigest(),'decompressed_limit':MAX_RAW}


def decode_gzip(blob,record):
    """Bounded pinned trace transport; independent outcome reader remains separate."""
    if (not 0<len(blob)<=MAX_GZIP or len(blob)!=record['bytes']
        or hashlib.sha256(blob).hexdigest()!=record['sha256']
        or record['decompressed_limit']!=MAX_RAW
        or not 0<record['decompressed_bytes']<=MAX_RAW):
        raise ValueError('Compressed diagnostic binding or bound differs')
    with gzip.GzipFile(fileobj=io.BytesIO(blob),mode='rb') as stream:raw=stream.read(MAX_RAW+1)
    if (len(raw)!=record['decompressed_bytes'] or len(raw)>MAX_RAW
            or hashlib.sha256(raw).hexdigest()!=record['decompressed_sha256']):
        raise ValueError('Decompressed diagnostic binding or bound differs')
    return json.loads(raw)

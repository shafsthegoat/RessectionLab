"""Prospective, disabled-by-default T1/MRA TAR header metadata reader.

Only request 512-byte TAR header records. Skip member bodies arithmetically.
No archive extraction, NIfTI parser, numeric array or model imports.
"""
from pathlib import Path, PurePosixPath
from datetime import datetime, timezone
import argparse
import hashlib
import json
import re
import ssl
import tarfile
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[2]
PREP = Path(__file__).resolve().parent
SCHEMA = 'ixi-t1-mra-header-metadata-protocol-v1'

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def walk(fetch, total_bytes, max_headers=1024):
    """fetch(offset) must return one authenticated 512-byte header record."""
    assert total_bytes > 0 and total_bytes % 512 == 0
    offset = 0
    rows = []
    seen = set()
    for _ in range(max_headers):
        assert offset + 512 <= total_bytes, 'missing TAR end records'
        block = fetch(offset)
        assert len(block) == 512
        if block == bytes(512):
            # Require a complete second zero record and bounded all-zero padding.
            assert total_bytes - offset <= 10240, 'unexpected TAR trailing extent'
            assert offset + 1024 <= total_bytes, 'missing second zero record'
            for tail in range(offset + 512, total_bytes, 512):
                assert fetch(tail) == bytes(512), 'nonzero TAR trailer'
            return rows
        info = tarfile.TarInfo.frombuf(block, 'utf-8', 'strict')
        assert info.type in (tarfile.REGTYPE, tarfile.AREGTYPE, tarfile.DIRTYPE), 'unsupported TAR extension/link/special member'
        name = info.name
        assert name and len(name.encode('utf-8')) <= 256 and '\\' not in name
        pp = PurePosixPath(name)
        assert not pp.is_absolute() and '..' not in pp.parts and name not in seen
        seen.add(name)
        assert info.size >= 0 and (info.type != tarfile.DIRTYPE or info.size == 0)
        end = offset + 512 + ((info.size + 511) // 512) * 512
        assert end + 1024 <= total_bytes, 'member outside archive/end extent'
        rows.append({'name': name, 'size': info.size, 'type': 'directory' if info.isdir() else 'file', 'header_offset': offset, 'header_sha256': hashlib.sha256(block).hexdigest()})
        offset = end
    raise ValueError('TAR header count cap exceeded')

class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        raise RuntimeError('redirect forbidden')

def execute(protocol, output, opener=None):
    assert protocol['schema'] == SCHEMA and protocol['execution_released'] is True
    assert protocol['archives'] and len(protocol['archives']) == 2
    assert {a['filename'] for a in protocol['archives']} == {'IXI-T1.tar', 'IXI-MRA.tar'}
    assert protocol['max_headers_per_archive'] == 1024
    assert protocol['max_response_body_bytes'] == 1100000
    assert protocol['max_requests'] == 2100 and protocol['deadline_seconds'] == 1800
    assert not output.exists(), 'Preserve every prior metadata attempt'
    output.mkdir()
    ctx = ssl.create_default_context()
    assert ctx.verify_mode == ssl.CERT_REQUIRED and ctx.check_hostname
    if opener is None:
        opener = urllib.request.build_opener(NoRedirect(), urllib.request.HTTPSHandler(context=ctx))
    started = time.monotonic()
    transferred = 0
    requests = 0
    inventories = []
    result = {'schema': 'ixi-tar-header-metadata-result-v1', 'started_utc': datetime.now(timezone.utc).isoformat(), 'status': 'started', 'archive_member_body_bytes_read': 0, 'image_header_or_array_bytes_read': 0, 'payload_decoding': False}
    try:
        for archive in protocol['archives']:
            assert archive['url'] == 'https://biomedic.doc.ic.ac.uk/brain-development/downloads/IXI/' + archive['filename']
            target = output / archive['filename']
            target.mkdir()
            def fetch(offset):
                nonlocal requests, transferred
                assert requests < protocol['max_requests']
                assert transferred + 512 <= protocol['max_response_body_bytes']
                assert time.monotonic() - started < protocol['deadline_seconds']
                assert 0 <= offset and offset + 512 <= archive['bytes'] and offset % 512 == 0
                requests += 1
                req = urllib.request.Request(archive['url'], headers={'Range': f'bytes={offset}-{offset + 511}', 'If-Match': archive['etag_opaque'], 'Accept-Encoding': 'identity'})
                intent = {'url': archive['url'], 'offset': offset, 'bytes_requested': 512, 'etag': archive['etag_opaque'], 'TLS_verified': True}
                stem = f'{requests:04d}-{offset:012d}'
                (target / (stem + '-intent.json')).write_text(json.dumps(intent, indent=2) + '\n')
                with opener.open(req, timeout=20) as response:
                    # Validate all headers before any response body read.
                    assert response.status == 206 and response.geturl() == archive['url'], 'server ignored Range or changed URL'
                    h = response.headers
                    assert h.get('Content-Range') == f'bytes {offset}-{offset + 511}/{archive["bytes"]}'
                    assert h.get('Content-Length') == '512' and h.get('ETag') == archive['etag_opaque']
                    assert h.get('Content-Encoding', 'identity').lower() == 'identity'
                    block = response.read(512)
                    transferred += len(block)
                    assert len(block) == 512, 'truncated metadata record'
                    (target / (stem + '-header.bin')).write_bytes(block)
                    (target / (stem + '-response.json')).write_text(json.dumps({'status': response.status, 'headers': dict(h), 'bytes_read': len(block), 'sha256': hashlib.sha256(block).hexdigest()}, indent=2) + '\n')
                return block
            rows = walk(fetch, archive['bytes'], protocol['max_headers_per_archive'])
            inventories.append({'filename': archive['filename'], 'archive_bytes': archive['bytes'], 'etag': archive['etag_opaque'], 'members': rows})
        result.update(status='complete_metadata_only', inventories=inventories)
    except BaseException as error:
        result.update(status='failed_preserved_no_retry', error_type=type(error).__name__, error=str(error), inventories= inventories)
        raise
    finally:
        result.update(finished_utc=datetime.now(timezone.utc).isoformat(), metadata_body_bytes_read=transferred, requests=requests, elapsed_seconds=time.monotonic()-started)
        (output / 'receipt.json').write_text(json.dumps(result, indent=2) + '\n')
    return result

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--protocol-sha256', required=True)
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    assert sha(args.protocol) == args.protocol_sha256
    p = json.loads(args.protocol.read_text())
    assert p['schema'] == SCHEMA and p['reader_sha256'] == sha(Path(__file__))
    assert sha(PREP / 'source-contract.json') == p['source_contract_sha256']
    source = json.loads((PREP / 'source-contract.json').read_text())
    expected = [a for a in source['raw_archives'] if a['filename'] in ['IXI-T1.tar','IXI-MRA.tar']]
    assert p['archives'] == expected
    assert p['output_path'] == 'build/ixi-t1-mra-header-metadata-execution-v1'
    if args.check_only:
        print(json.dumps({'status':'check_only','execution_released':p['execution_released'],'network_requests':0,'payload_bytes_read':0}))
        return
    assert p['execution_released'] is True, 'Metadata execution is not released'
    result = execute(p, ROOT / p['output_path'])
    print(json.dumps({k:v for k,v in result.items() if k!='inventories'}, indent=2))

if __name__ == '__main__':
    main()

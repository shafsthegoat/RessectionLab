"""Independent read-only verifier; authentic case3 and declared numeric values only.

Packet layout independently derived from authenticated vitaldb 1.7.2 utils.py.
MIT Copyright (c) 2021 VitalDB. Full upstream notice retained at
../../artifacts/vitaldb-recorded-component-v1/upstream-MIT-LICENSE.txt.
No production reader import; no network or upstream library execution.
"""
from collections import Counter
from pathlib import Path
import hashlib
import json
import math
import struct
import zlib

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
SOURCE = ROOT / 'data/vitaldb-1.0.0/mirror-01/0003.vital'
SOURCE_SHA = '573db0941d580167833f84497f5d1c4a1391443eeec3f95852287ea09db024e4'
COMPONENT_SHA = '7170c6d5c521b845e937eb69b16d0083650823ae8da4a6da201eeff71bfe9679'
SOURCE_BYTES = 6537712
FORMATS = {1: 'f', 2: 'd', 3: 'b', 4: 'B', 5: 'h', 6: 'H', 7: 'l', 8: 'L'}


def read_actual():
    declaration_bytes = (ROOT / 'manifests/vitaldb-native-component-v1.json').read_bytes()
    assert hashlib.sha256(declaration_bytes).hexdigest() == COMPONENT_SHA
    requested = json.loads(declaration_bytes)['required_tracks']
    with SOURCE.open('rb') as f:
        compressed = f.read(SOURCE_BYTES + 1)
    assert len(compressed) == SOURCE_BYTES
    assert hashlib.sha256(compressed).hexdigest() == SOURCE_SHA
    decoder = zlib.decompressobj(31)
    raw = decoder.decompress(compressed, 134217729)
    assert len(raw) <= 134217728 and decoder.eof
    assert not decoder.unused_data and not decoder.unconsumed_tail
    assert raw[:4] == b'VITA'
    version, hlen = struct.unpack_from('<LH', raw, 4)
    assert (version, hlen) == (3, 10)
    assert len(raw) >= 10 + hlen
    header_hex = raw[:10 + hlen].hex()
    pos = 10 + hlen
    devices = {}
    all_tracks = {}
    selected_headers = {}
    selected_records = []
    per_track = Counter()
    types = Counter()
    ordinal = 0
    maximum_packet = 0
    while pos < len(raw):
        assert len(raw) - pos >= 5
        packet_type, length = struct.unpack_from('<BL', raw, pos)
        pos += 5
        ordinal += 1
        assert ordinal <= 1000000 and length <= 4194304
        assert pos + length <= len(raw)
        start, end = pos, pos + length
        pos = end
        types[packet_type] += 1
        maximum_packet = max(maximum_packet, length)
        if packet_type in (9, 0):
            assert length <= 4096
            q = start
            def num(fmt):
                nonlocal q
                n = struct.calcsize('<' + fmt)
                assert q + n <= end
                result = struct.unpack_from('<' + fmt, raw, q)[0]
                q += n
                return result
            def string():
                nonlocal q
                size = num('L')
                assert size <= 4096 and q + size <= end
                result = raw[q:q + size].decode('utf8')
                q += size
                return result
            if packet_type == 9:
                did, dtype, dname = num('L'), string(), string()
                if q < end:
                    string()
                assert q == end and did not in devices and len(devices) < 256
                devices[did] = dname or dtype
            else:
                tid, kind, fmt = num('H'), num('B'), num('B')
                tname, unit = string(), string()
                mindisp, maxdisp, color = num('f'), num('f'), num('L')
                srate, gain, offset = num('f'), num('d'), num('d')
                montype, did = num('B'), num('L')
                extension = {}
                for key, code in [('record_bytes', 'L'), ('source_dtstart', 'd'), ('source_dtend', 'd')]:
                    if q < end:
                        extension[key] = num(code)
                assert q == end and tid not in all_tracks and len(all_tracks) < 1024
                assert not did or did in devices
                full = (devices[did] + '/' if did else '') + tname
                all_tracks[tid] = full
                if full in requested:
                    spec = requested[full]
                    assert kind == spec['type'] == 2 and fmt == spec['format'] == 1
                    assert unit == spec['native_unit'] and srate == 0 and gain == 1 and offset == 0
                    assert full not in selected_headers
                    selected_headers[full] = {'track_id': tid, 'name': full, 'type': kind,
                        'format': fmt, 'unit': unit, 'sample_rate': srate, 'gain': gain,
                        'offset': offset, 'monitor_type': montype, 'device_id': did,
                        'extension': extension}
        elif packet_type == 1:
            assert length >= 12
            info_size = struct.unpack_from('<H', raw, start)[0]
            tid = struct.unpack_from('<H', raw, start + 10)[0]
            assert 10 <= info_size <= 4096 and 2 + info_size <= length
            assert tid in all_tracks
            full = all_tracks[tid]
            if full not in requested:
                continue  # No timestamp or value decoding for excluded records.
            h = selected_headers[full]
            value_start = start + 2 + info_size
            size = struct.calcsize('<' + FORMATS[h['format']])
            assert value_start + size == end
            timestamp_bytes = raw[start + 2:start + 10]
            timestamp = struct.unpack('<d', timestamp_bytes)[0]
            assert math.isfinite(timestamp) and timestamp >= 0
            value_bytes = raw[value_start:end]
            value = struct.unpack('<' + FORMATS[h['format']], value_bytes)[0]
            kind = 'finite'
            if not math.isfinite(value):
                kind = 'nan' if math.isnan(value) else ('positive_infinity' if value > 0 else 'negative_infinity')
                value = None
            per_track[full] += 1
            assert len(selected_records) < 250000
            selected_records.append({'track': full, 'unit': h['unit'],
                'source_timestamp_seconds': timestamp, 'timestamp_bytes_hex': timestamp_bytes.hex(),
                'record_header_extension_hex': raw[start + 12:value_start].hex(),
                'packet_ordinal': ordinal, 'track_ordinal': per_track[full],
                'value': value, 'value_kind': kind, 'value_bytes_hex': value_bytes.hex()})
        elif packet_type == 10:
            pass  # No raw device traffic decoding.
        elif packet_type == 6:
            assert length >= 1
            command = raw[start]
            if command == 5:
                assert length >= 3
                n = struct.unpack_from('<H', raw, start + 1)[0]
                assert n <= 1024 and 3 + 2 * n == length
            else:
                assert command == 6 and length == 1
        else:
            raise AssertionError(f'Unexpected packet type {packet_type}')
    assert pos == len(raw) and set(selected_headers) == set(requested)
    assert all(per_track[name] > 0 for name in requested)
    tracks = {}
    for name in requested:
        records = [x for x in selected_records if x['track'] == name]
        times = [x['source_timestamp_seconds'] for x in records]
        gaps = [b - a for a, b in zip(sorted(times), sorted(times)[1:])]
        tracks[name] = {'records': len(records), 'first_native_timestamp': min(times),
            'last_native_timestamp': max(times), 'duplicate_timestamps': len(times) - len(set(times)),
            'out_of_order_in_source': sum(b < a for a, b in zip(times, times[1:])),
            'nonfinite_values': sum(x['value_kind'] != 'finite' for x in records),
            'max_gap_seconds': max(gaps)}
    report = {'source_sha256': SOURCE_SHA, 'source_bytes': len(compressed),
        'component_declaration_sha256': COMPONENT_SHA,
        'independent_reader_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'header_hex': header_hex, 'header_size': hlen,
        'native_header_time_origin': None, 'time_basis': 'unmapped_native_source_seconds',
        'decompressed_bytes': len(raw), 'packet_count': ordinal,
        'packet_type_counts': dict(types), 'maximum_packet_bytes': maximum_packet,
        'device_count': len(devices), 'track_count': len(all_tracks),
        'selected_records': len(selected_records), 'selected_headers': selected_headers,
        'tracks': tracks, 'complete_single_gzip_crc_eof_validated': True,
        'unrelated_values_decoded': 0}
    return report, selected_records


if __name__ == '__main__':
    report, records = read_actual()
    (OUT / 'independent-summary.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    with (OUT / 'independent-records.jsonl').open('w') as out:
        for record in records:
            out.write(json.dumps(record, sort_keys=True, separators=(',', ':'), allow_nan=False) + '\n')
    print(json.dumps(report, indent=2, allow_nan=False))

"""Authentic-source replay checks and pure byte/clock/refusal controls.

No generated patient or observation fixture. Bytes used in integrity controls are
only an empty gzip container and copies of the real source with damaged trailers.
"""
from __future__ import annotations

from copy import deepcopy
import gzip
import hashlib
import io
import json
from pathlib import Path
import struct

import pytest

import resectionlab.vitaldb_observed as vital

SOURCE = vital.ROOT / 'data/vitaldb-1.0.0/mirror-01/0003.vital'


@pytest.fixture(scope='module')
def actual():
    if not SOURCE.exists():
        pytest.skip('Authenticated source unavailable; actual-source acceptance unproved')
    return vital._read_local(SOURCE)


def test_url_and_partial_refusal_before_decode(tmp_path):
    with pytest.raises(vital.VitalDBError, match='local file'):
        vital._read_local('https://physionet.org/unauthorized.vital')
    partial = tmp_path / 'incomplete.vital'
    partial.write_bytes(b'')  # Empty file is a pure format/refusal control.
    with pytest.raises(vital.VitalDBError, match='size mismatch'):
        vital._read_local(partial)


def test_declaration_role_bound_before_file_access(monkeypatch, tmp_path):
    changed = tmp_path / 'role.json'
    changed.write_bytes(vital.DECLARATION.read_bytes() + b'\n')
    monkeypatch.setattr(vital, 'DECLARATION', changed)
    with pytest.raises(vital.VitalDBError, match='declaration changed'):
        vital._read_local(SOURCE)


def test_packet_field_and_utf8_refusals():
    with pytest.raises(vital.VitalDBError, match='packet length'):
        vital._Cursor(b'\x00').number('L')
    with pytest.raises(vital.VitalDBError, match='UTF-8'):
        vital._Cursor(struct.pack('<L', 1) + b'\xff').string()
    with pytest.raises(vital.VitalDBError, match='string bound'):
        vital._Cursor(struct.pack('<L', 4097)).string()


def test_exact_numeric_contract_refusals():
    name = 'Orchestra/PPF20_RATE'
    for kind, fmt, unit, sample_rate, gain, offset in (
        (1, 1, 'mL/h', 0, 1, 0), (2, 99, 'mL/h', 0, 1, 0),
        (2, 1, 'mg/hr', 0, 1, 0), (2, 1, 'mL/h', 0, 2, 0),
        (2, 1, 'mL/h', 0, float('nan'), 0), (2, 1, 'mL/hr', 0, 1, 0),
    ):
        with pytest.raises(vital.VitalDBError):
            vital._validate_track(name, kind, fmt, unit, sample_rate, gain, offset)


def test_gzip_container_eof_crc_and_trailing_bytes():
    empty_container = gzip.compress(b'', mtime=0)
    stream = vital._GzipStream(io.BytesIO(empty_container))
    assert stream.read(1) == b'' and stream.done
    for data in (empty_container[:-1], empty_container + b'garbage',
                 empty_container + empty_container,
                 empty_container[:-8] + bytes([empty_container[-8] ^ 1]) + empty_container[-7:]):
        with pytest.raises(vital.VitalDBError):
            vital._GzipStream(io.BytesIO(data)).read(1)


def test_authentic_file_fixity_clock_and_selected_tracks(actual):
    assert hashlib.sha256(SOURCE.read_bytes()).hexdigest() == vital.SOURCE_SHA256
    header = json.loads(actual.header_json)
    assert header['clock_origin_seconds'] is None
    assert header['recording_start_mapping_status'] == 'unresolved'
    assert header['source_dtstart'] in (0, None)
    assert set(json.loads(actual.track_headers_json)) == set(vital.TRACKS)
    assert json.loads(actual.parse_stats_json)['complete_gzip_crc_eof_validated']
    assert actual.observations
    for event in actual.observations:
        assert struct.unpack('<d', bytes.fromhex(event.timestamp_bytes_hex))[0] == event.source_timestamp_seconds
        assert event.unit == vital.TRACKS[event.track]
    assert len({event.packet_ordinal for event in actual.observations}) == len(actual.observations)


def test_original_arterial_contract_remains_failed(actual):
    result = vital.check_original_request(SOURCE)
    assert result['status'] == 'refused'
    assert set(result['missing_required_tracks']) == set(vital.MISSING_ARTERIAL_TRACKS)
    assert result['selected_record_values_decoded'] == 0
    assert result['original_declaration_unchanged']
    assert set(vital.TRACKS).isdisjoint(vital.MISSING_ARTERIAL_TRACKS)


def test_authentic_source_damage_refused_before_decode(actual, tmp_path):
    data = bytearray(SOURCE.read_bytes())
    data[-8] ^= 1  # Actual compressed-file integrity control, no fabricated record.
    altered = tmp_path / 'altered.vital'
    altered.write_bytes(data)
    with pytest.raises(vital.VitalDBError, match='SHA256'):
        vital._read_local(altered)
    # Direct container validation additionally exercises trailer CRC, independently
    # of the earlier admission hash guard. No altered record is admitted/replayed.
    stream = vital._GzipStream(io.BytesIO(data))
    with pytest.raises(vital.VitalDBError, match='CRC'):
        while stream.read(65536):
            pass


def test_actual_decompression_resource_refusal(actual):
    with SOURCE.open('rb') as f:
        stream = vital._GzipStream(f, limit=32)
        with pytest.raises(vital.VitalDBError, match='Decompressed byte bound'):
            stream.read(64)


def test_real_clock_prefix_asynchronous_age_and_no_future_metadata(actual):
    ordered = sorted(actual.observations, key=lambda e: (e.source_timestamp_seconds, e.packet_ordinal))
    times = sorted({e.source_timestamp_seconds for e in ordered})
    target = times[len(times) // 2]
    replay = vital.ObservedReplay(SOURCE)
    frame = replay.advance(target)
    prefix = [e.to_dict() for e in ordered if e.source_timestamp_seconds <= target]
    assert replay.visible_history() == prefix
    assert all(e['source_timestamp_seconds'] <= target for e in frame['newly_released_records'])
    assert {e['packet_ordinal'] for e in frame['newly_released_records']} <= {e['packet_ordinal'] for e in prefix}
    forbidden = {'selected_records', 'source_dtend', 'record_bytes', 'packet_count', 'max_gap_seconds', 'first_timestamp_seconds', 'last_timestamp_seconds', 'value_range'}
    def keys(value):
        if isinstance(value, dict):
            return set(value) | set().union(*(keys(v) for v in value.values()))
        if isinstance(value, list):
            return set().union(*(keys(v) for v in value))
        return set()
    assert not keys(frame) & forbidden
    for name in vital.TRACKS:
        visible = [event for event in prefix if event['track'] == name]
        entry = frame['last_recorded_observations'][name]
        if visible:
            assert entry['observation'] == visible[-1]
            assert entry['source_record_age_seconds'] == target - visible[-1]['source_timestamp_seconds']
            assert entry['actual_measurement_age_unknown'] is True
        else:
            assert entry is None
    # Positive gaps originate in the actual series, not a fabricated timestamped case.
    gaps = [(a, b) for a, b in zip(times, times[1:]) if b > a]
    a, b = max(gaps, key=lambda pair: pair[1] - pair[0])
    gap_replay = vital.ObservedReplay(SOURCE)
    before = gap_replay.advance(a)
    between = gap_replay.advance(a + (b - a) / 2)
    assert between['newly_released_records'] == []
    for name, entry in before['last_recorded_observations'].items():
        if entry is not None:
            later = between['last_recorded_observations'][name]
            assert later['observation'] == entry['observation']
            assert later['source_record_age_seconds'] > entry['source_record_age_seconds']
            assert later['source_record_at_this_replay_time'] is False


def test_real_reopen_future_release_and_defensive_copies(actual):
    times = sorted({e.source_timestamp_seconds for e in actual.observations})
    t1, t2 = times[len(times)//3], times[2*len(times)//3]
    replay = vital.ObservedReplay(SOURCE)
    replay.advance(t1)
    snap = replay.snapshot()
    reopened = vital.ObservedReplay.reopen(SOURCE, json.loads(json.dumps(snap)))
    assert replay.advance(t2) == reopened.advance(t2)
    assert replay.visible_history() == reopened.visible_history()
    frame = replay.frame()
    frame['binding']['role'] = 'EVAL'
    history = replay.visible_history()
    history[0]['value'] = 'mutated'
    assert replay.frame()['binding']['role'] == 'DEVELOPMENT'
    assert replay.visible_history()[0]['value'] != 'mutated'
    for clock in (float('nan'), float('inf'), -1, True, '1'):
        with pytest.raises(vital.VitalDBError):
            replay.advance(clock)
    for field, value in (('released_records', snap['released_records']+1),
                         ('released_prefix_sha256', '0'*64), ('clock_seconds', -1)):
        damaged = deepcopy(snap)
        damaged[field] = value
        with pytest.raises(vital.VitalDBError):
            vital.ObservedReplay.reopen(SOURCE, damaged)
    for field, value in (('role', 'EVAL'), ('source_sha256', '0'*64), ('reader_code_sha256', '0'*64)):
        damaged = deepcopy(snap)
        damaged['binding'][field] = value
        with pytest.raises(vital.VitalDBError, match='binding mismatch'):
            vital.ObservedReplay.reopen(SOURCE, damaged)


def test_legal_and_parser_evidence_bound_and_changed_rights_refused(monkeypatch, tmp_path):
    evidence = vital._admission_evidence()
    assert set(evidence) == {'rights_text_sha256', 'mit_notice_sha256',
        'rights_and_parser_receipt_sha256', 'parser_archive_sha256', 'upstream_utils_sha256'}
    damaged = tmp_path / 'artifacts/vitaldb-recorded-component-v1/PhysioNet-CC-BY-4.0.txt'
    damaged.parent.mkdir(parents=True)
    original = (vital.ROOT / 'artifacts/vitaldb-recorded-component-v1/PhysioNet-CC-BY-4.0.txt').read_bytes()
    damaged.write_bytes(original + b'\n')
    monkeypatch.setattr(vital, 'ROOT', tmp_path)
    with pytest.raises(vital.VitalDBError, match='admission evidence changed: rights_text'):
        vital._admission_evidence()


def test_native_component_and_clock_declarations_are_admission_gates(monkeypatch, tmp_path):
    for attribute in ('COMPONENT_DECLARATION', 'CLOCK_REVIEW'):
        original = getattr(vital, attribute)
        changed = tmp_path / original.name
        changed.write_bytes(original.read_bytes() + b'\n')
        with monkeypatch.context() as patch:
            patch.setattr(vital, attribute, changed)
            with pytest.raises(vital.VitalDBError, match='changed'):
                vital._declaration()


def test_cli_source_hardlink_output_alias_and_existing_output_refusals(actual, tmp_path):
    import os
    import runpy
    cli = runpy.run_path(str(vital.ROOT / 'scripts/replay_vitaldb_component.py'))
    validate, write = cli['_output_paths'], cli['_write_new']
    before = hashlib.sha256(SOURCE.read_bytes()).hexdigest()
    with pytest.raises(vital.VitalDBError, match='aliases the authenticated input'):
        validate(SOURCE, [SOURCE])
    alias = tmp_path / 'source-hardlink.vital'
    os.link(SOURCE, alias)
    with pytest.raises(vital.VitalDBError, match='aliases the authenticated input'):
        validate(SOURCE, [alias])
    candidate = vital.ROOT / 'outputs' / 'vitaldb-test-output-new.json'
    assert not candidate.exists()
    with pytest.raises(vital.VitalDBError, match='alias each other'):
        validate(SOURCE, [candidate, candidate])
    with pytest.raises(vital.VitalDBError, match='already exists'):
        validate(SOURCE, [vital.DECLARATION])
    existing = tmp_path / 'existing.txt'
    existing.write_text('retain')
    with pytest.raises(FileExistsError):
        write(existing, 'overwrite')
    assert existing.read_text() == 'retain'
    assert hashlib.sha256(SOURCE.read_bytes()).hexdigest() == before


def test_actual_record_resource_bounds_refuse_without_partial_admission(actual, monkeypatch):
    with monkeypatch.context() as patch:
        patch.setitem(vital.LIMITS, 'packet_bytes', 100)
        with pytest.raises(vital.VitalDBError, match='bound exceeded'):
            vital._read_local(SOURCE)
    with monkeypatch.context() as patch:
        patch.setitem(vital.LIMITS, 'records', 1)
        with pytest.raises(vital.VitalDBError, match='record bound exceeded'):
            vital._read_local(SOURCE)

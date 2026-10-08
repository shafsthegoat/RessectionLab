"""Compare authenticated records to independent decoding; no synthetic records."""
from copy import deepcopy
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys

from independent_reader import read_actual, ROOT, OUT, SOURCE

sys.path.insert(0, str(ROOT / 'src'))
import resectionlab.vitaldb_observed as production

report, expected = read_actual()
source_before = hashlib.sha256(Path(production.__file__).read_bytes()).hexdigest()
actual = production._read_local(SOURCE)
actual_records = [o.to_dict() for o in actual.observations]
assert actual_records == expected, 'At least one metadata/value/timestamp/ordinal differs'
assert json.loads(actual.track_headers_json) == report['selected_headers']
stats = json.loads(actual.parse_stats_json)
assert stats['packet_count'] == report['packet_count']
assert stats['packet_type_counts'] == {str(k): v for k, v in report['packet_type_counts'].items()}
assert stats['decompressed_bytes'] == report['decompressed_bytes']
assert stats['compressed_bytes'] == report['source_bytes']
assert stats['max_packet_bytes'] == report['maximum_packet_bytes']
assert stats['complete_gzip_crc_eof_validated'] is True
header = json.loads(actual.header_json)
assert header['source_dtstart'] is None and header['source_dtend'] is None
assert header['clock_origin_seconds'] is None
assert header['time_basis'] == 'unmapped_native_source_seconds'
assert header['recording_start_mapping_status'] == 'unresolved'

ordered = sorted(expected, key=lambda r: (r['source_timestamp_seconds'], r['packet_ordinal']))
times = sorted({r['source_timestamp_seconds'] for r in ordered})
replay = production.ObservedReplay(SOURCE)
before = replay.advance(math.nextafter(times[0], -math.inf))
assert before['newly_released_records'] == []
assert all(value is None for value in before['last_recorded_observations'].values())
assert replay.visible_history() == []
forbidden = {'selected_records', 'source_dtend', 'record_bytes', 'packet_count',
             'max_gap_seconds', 'first_timestamp_seconds', 'last_timestamp_seconds', 'value_range'}

def keys(value):
    if isinstance(value, dict):
        return set(value) | set().union(*(keys(v) for v in value.values()))
    if isinstance(value, list):
        return set().union(*(keys(v) for v in value))
    return set()

cursor = 0
last = {}
midpoint_snapshot = None
midpoint_history = None
for i, clock in enumerate(times):
    start = cursor
    while cursor < len(ordered) and ordered[cursor]['source_timestamp_seconds'] <= clock:
        last[ordered[cursor]['track']] = ordered[cursor]
        cursor += 1
    frame = replay.advance(clock)
    assert frame['newly_released_records'] == ordered[start:cursor]
    assert frame['clock_seconds'] == clock and frame['clock_origin_seconds'] is None
    assert frame['time_basis'] == 'unmapped_native_source_seconds'
    assert frame['pressure_modality'] == 'intermittent_noninvasive_cuff_pressure'
    assert not keys(frame) & forbidden
    for track in report['selected_headers']:
        entry = frame['last_recorded_observations'][track]
        if track not in last:
            assert entry is None
        else:
            assert entry['observation'] == last[track]
            assert entry['source_record_age_seconds'] == clock - last[track]['source_timestamp_seconds']
            assert entry['source_record_at_this_replay_time'] == (clock == last[track]['source_timestamp_seconds'])
            assert entry['actual_measurement_age_unknown'] is True
    if i == len(times) // 2:
        midpoint_snapshot = replay.snapshot()
        midpoint_history = replay.visible_history()
assert cursor == len(expected) and replay.visible_history() == ordered
assert midpoint_snapshot is not None
reopened = production.ObservedReplay.reopen(SOURCE, json.loads(json.dumps(midpoint_snapshot)))
assert reopened.visible_history() == midpoint_history
reopened.advance(times[-1])
assert reopened.frame() == replay.frame()
assert reopened.visible_history() == ordered
assert reopened.snapshot() == replay.snapshot()
prefix = hashlib.sha256()
for item in ordered:
    prefix.update(json.dumps(item, sort_keys=True, separators=(',', ':'), allow_nan=False).encode() + b'\n')
assert replay.snapshot()['released_prefix_sha256'] == prefix.hexdigest()

binding_refusals = []
for key in ['source_sha256', 'declaration_sha256', 'component_declaration_sha256',
            'clock_review_sha256', 'reader_code_sha256', 'source_release', 'case_id',
            'subject_id', 'role', 'time_basis', 'admission_evidence']:
    changed = deepcopy(midpoint_snapshot)
    changed['binding'][key] = None
    try:
        production.ObservedReplay.reopen(SOURCE, changed)
    except production.VitalDBError as exc:
        assert 'binding mismatch' in str(exc)
        binding_refusals.append(key)
    else:
        raise AssertionError(f'Changed snapshot {key} admitted')

clock_refusals = 0
for value in [float('nan'), float('inf'), -1, True, '1']:
    try:
        replay.advance(value)
    except production.VitalDBError:
        clock_refusals += 1
    else:
        raise AssertionError('Invalid/backward clock accepted')

spec = importlib.util.spec_from_file_location('vitaldb_cli_review', ROOT / 'scripts/replay_vitaldb_component.py')
cli = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cli)
output_refusals = 0
for paths in [[SOURCE], [OUT / 'same.json', OUT / 'same.json'], [ROOT / 'manifests/vitaldb-native-component-v1.json']]:
    try:
        cli._output_paths(SOURCE, paths)
    except production.VitalDBError:
        output_refusals += 1
    else:
        raise AssertionError('Protected/alias output accepted')
assert hashlib.sha256(Path(production.__file__).read_bytes()).hexdigest() == source_before
result = {'status': 'passed', 'production_reader_sha256': source_before,
    'independent_reader_sha256': report['independent_reader_sha256'],
    'source_sha256': report['source_sha256'],
    'all_selected_record_fields_exactly_equal': len(expected),
    'all_selected_track_headers_equal': len(report['selected_headers']),
    'native_clock_event_boundaries_checked': len(times),
    'all_frames_have_only_observed_prefix': True,
    'all_record_ages_equal_unshifted_native_differences': True,
    'unknown_clock_origin_preserved': True,
    'midpoint_reopen_history_and_continuation_equal': True,
    'independent_full_prefix_digest_equal': True,
    'snapshot_binding_changes_refused': binding_refusals,
    'invalid_or_backward_clocks_refused': clock_refusals,
    'protected_or_alias_output_checks_refused': output_refusals,
    'no_scientific_download_or_synthetic_patient_records': True,
    'unrelated_values_decoded': 0}
(OUT / 'implementation-comparison.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result, indent=2))

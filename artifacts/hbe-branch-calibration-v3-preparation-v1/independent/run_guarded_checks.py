"""Independent preparation checks; metadata plus synthetic axial rows only."""
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
study = json.loads((ROOT/'manifests/experiments/hbe-01-03-branch-calibration-v3.json').read_text())
forbidden_paths = {ROOT/b['path'] for q in study['axial_references'].values()
                   for b in q['native_primitives'].values()}
forbidden_paths.update(ROOT/b['path'] for q in study['torsion_references'].values()
                       for b in q['primitives'].values())
audit = {'forbidden_file_attempts': [], 'forbidden_process_attempts': [],
         'forbidden_scientific_execution_attempts': []}


def guard(event, args):
    if event in {'subprocess.Popen', 'os.system', 'os.posix_spawn', 'os.fork', 'os.exec'}:
        audit['forbidden_process_attempts'].append(event)
        raise AssertionError('Subprocess and native execution forbidden during independent checks')
    if event == 'open' and isinstance(args[0], (str, bytes, os.PathLike)):
        path = Path(os.fsdecode(args[0])).resolve()
        if path.is_relative_to(ROOT/'data/mechanics') or path in forbidden_paths:
            audit['forbidden_file_attempts'].append(str(path))
            raise AssertionError('Measured archive and native primitive access forbidden')
        mode, flags = args[1:3]
        writing = (isinstance(mode, str) and any(v in mode for v in 'wax+')) or bool(
            flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND))
        if writing and path.is_relative_to(ROOT) and not path.is_relative_to(OUT):
            audit['forbidden_file_attempts'].append(str(path))
            raise AssertionError('Independent checks may write only their evidence directory')


sys.addaudithook(guard)
from scripts import mechanics_hbe_branch_calibration_v3 as core
from scripts import mechanics_hbe_branch_calibration_v3_experiment as runner
from scripts import mechanics_hbe_branch_calibration_v3_readout as readout
from scripts import mechanics_hbe_branch_calibration_v2 as previous


def forbid_scientific(*args, **kwargs):
    audit['forbidden_scientific_execution_attempts'].append('fit_native_qualification_or_prepare')
    raise AssertionError('Fit, native, qualification and deck execution forbidden')


for module, name in [(core.evaluation, 'fit_scale'), (core.runtime, 'supervise'),
                     (core.old, 'solve'), (core.common, 'pure_child'),
                     (core, 'qualify'), (previous, 'qualify'),
                     (runner, 'prepare'), (readout, 'stream_frames')]:
    setattr(module, name, forbid_scientific)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


registry = core.Registry(ROOT)
records = core.verify_csv_header_migration(study, registry)
registry.verify_all(time.monotonic()+10)
diagnosis = records['diagnosis']
header_results = []
bad_headers = [b'force,displacement\n', b'Displacement,force\n',
               b'displacement, force\n', b'displacement,displacement\n',
               b'\xef\xbb\xbfdisplacement,force\n', b'\ndisplacement,force\n',
               b'#comment\ndisplacement,force\n', b'displacement,force,extra\n',
               b'unknown,header\n', b'']
for index, (branch, sign) in enumerate((('compression', -1), ('tension', 1))):
    literal = diagnosis['members'][index]['raw_first_line'].encode()
    assert literal == b'displacement,force\n'
    # All numerical rows are independently invented, never copied from measured data.
    synthetic_rows = f'0,0\n{sign*.0004},{sign*.0031}\n{sign*.0009},{sign*.0072}\n'.encode()
    payload = literal+synthetic_rows
    fixture = OUT/f'synthetic-{branch}-real-header.csv'
    fixture.write_bytes(payload)
    curve = core.access.parse_member_csv(payload, branch, study['csv_schemas'][branch])
    assert curve.coordinate == (0., sign*.0004, sign*.0009)
    assert curve.response == (0., sign*.0031, sign*.0072)
    assert curve.source_sha256 == hashlib.sha256(payload).hexdigest()
    rejected = 0
    for wrong in bad_headers:
        try:
            core.access.parse_member_csv(wrong+synthetic_rows, branch, study['csv_schemas'][branch])
        except ValueError as exc:
            assert 'header differs; no autodetection' in str(exc)
            rejected += 1
        else:
            raise AssertionError(f'Unexpected accepted malformed header: {wrong!r}')
    for middle in (literal, b'unknown,row\n'):
        try:
            core.access.parse_member_csv(literal+b'0,0\n'+middle+synthetic_rows,
                                         branch, study['csv_schemas'][branch])
        except ValueError:
            rejected += 1
        else:
            raise AssertionError('Interior invalid row was silently skipped')
    try:
        core.access.parse_member_csv(payload, branch, records['previous_study']['csv_schemas'][branch])
    except ValueError as exc:
        assert "could not convert string to float: 'displacement'" == str(exc)
        rejected += 1
    else:
        raise AssertionError('Frozen v2 headerless declaration unexpectedly parsed real header')
    header_results.append({'branch': branch, 'fixture_sha256': sha(fixture),
                           'valid_cases_passed': 1, 'invalid_cases_rejected': rejected,
                           'only_literal_header_from_real_evidence': True,
                           'all_numeric_rows_synthetic': True})

import pytest
code = pytest.main(['-q', '-p', 'no:cacheprovider',
                    'tests/test_mechanics_hbe_branch_calibration_v2.py',
                    'tests/test_mechanics_hbe_branch_calibration_v3.py',
                    '-k', 'not test_v2_fit_is_same_objective_on_analytical_observations'])
result = {'schema': 'hbe-branch-calibration-v3-independent-guarded-checks-v1',
          'pytest_exit_code': int(code), 'header_fixtures': header_results,
          'lineage_bound_and_rehashed': registry.snapshot(), 'guard_audit': audit,
          'excluded_test_reason': 'One v2 analytical fitting test deselected to honor no fitting, including synthetic fitting.',
          'measured_archive_opened': False, 'torsion_content_opened': False,
          'fit_called': False, 'native_called': False}
(OUT/'guarded-checks.json').write_text(json.dumps(result, indent=2, sort_keys=True)+'\n')
assert not any(audit.values()), audit
raise SystemExit(int(code))

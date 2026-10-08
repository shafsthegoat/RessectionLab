"""Pure clock/file accounting controls; no patient data or solver responses."""
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import mechanics_hbe_halfheight_spatial_experiment as runner


def finish(tmp_path, monkeypatch, elapsed, output_cap=10000):
    reads = []
    def clock():
        reads.append(elapsed)
        assert len(reads) == 1
        return 100 + elapsed
    monkeypatch.setattr(runner.time, 'monotonic', clock)
    result = runner.finalize_result(tmp_path, tmp_path, tmp_path,
        {'status': 'prepared_not_solved', 'note': 'µ'}, 100, 60, output_cap)
    assert len(reads) == 1
    assert result['phase_elapsed_seconds'] == elapsed
    return result


@pytest.mark.parametrize('elapsed,passed', [(59.5, True), (60, False), (61, False), (-1, False)])
def test_single_elapsed_read_controls_receipt_and_acceptance(tmp_path, monkeypatch, elapsed, passed):
    result = finish(tmp_path, monkeypatch, elapsed)
    assert (result['status'] == 'prepared_not_solved') is passed
    assert (tmp_path/'result.json').read_bytes() == runner.access.canonical_json(result)


@pytest.mark.parametrize('excess', [0, 1])
def test_exact_byte_cap_counts_receipt_and_leftover_temp(tmp_path, monkeypatch, excess):
    # Calculate serialization size from a real first receipt, then solve a pure
    # integer byte-count identity at a fixed cap. No scientific record is made.
    first = tmp_path/'size'; first.mkdir()
    candidate = finish(first, monkeypatch, 1)
    candidate['retained_bytes_before_result'] = 9500
    for _ in range(4):
        candidate['retained_bytes_before_result'] = 10000-len(runner.access.canonical_json(candidate))
    padding = candidate['retained_bytes_before_result']
    assert padding + len(runner.access.canonical_json(candidate)) == 10000
    target = tmp_path/'run'; target.mkdir()
    (target/'state.json.tmp').write_bytes(b' '*(padding+excess))
    result = finish(target, monkeypatch, 1)
    assert (result['status'] == 'prepared_not_solved') is (excess == 0)
    assert result['retained_bytes_before_result'] == padding+excess
    if not excess:
        assert runner.old.meshing.tree_bytes(target) == 10000
    else:
        assert result['final_cap_failure'] == 'retained_output_cap_including_receipt'


def test_symlink_refused_before_publication(tmp_path, monkeypatch):
    (tmp_path/'alias').symlink_to(tmp_path/'missing')
    with pytest.raises(ValueError, match='symlink'):
        finish(tmp_path, monkeypatch, 1)
    assert not (tmp_path/'result.json').exists()


def test_existing_receipt_preserved(tmp_path, monkeypatch):
    path = tmp_path/'result.json'; path.write_bytes(b'original')
    with pytest.raises(FileExistsError):
        finish(tmp_path, monkeypatch, 1)
    assert path.read_bytes() == b'original'


def test_publication_error_never_returns_success(tmp_path, monkeypatch):
    def fail(*args):
        raise OSError('write unavailable')
    monkeypatch.setattr(runner.old, 'saved', fail)
    with pytest.raises(OSError, match='write unavailable'):
        finish(tmp_path, monkeypatch, 1)

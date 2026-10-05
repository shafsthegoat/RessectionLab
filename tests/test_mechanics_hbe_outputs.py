"""Parser contracts plus a compact saved solver-log regression; no HBE curves."""
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

SPEC=importlib.util.spec_from_file_location('hbe_outputs',Path(__file__).resolve().parents[1]/'scripts/mechanics_hbe_outputs.py')
o=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(o)


def text():
    return '\n'.join(f'*Step = {i}\n*Time = {t}\n*Data = nodes\n2,2,3,4\n1,1,2,3' for i,t in enumerate([0,.5,1]))


def parse(value):
    return list(o.iter_data_records(value.splitlines(keepends=True),expected_times=[0,.5,1],item_count=2,field_count=3,record_name='nodes'))


def test_complete_initial_and_converged_records_in_canonical_id_order():
    records=parse(text())
    assert [r['step'] for r in records]==[0,1,2]
    assert records[0]['values'].tolist()==[[1,2,3],[2,3,4]]


@pytest.mark.parametrize('corrupt',[lambda s:s[s.index('*Step = 1'):],lambda s:s.replace('2,2,3,4','1,2,3,4'),lambda s:s.replace('2,2,3,4','2,nan,3,4'),lambda s:s.replace('*Time = 0.5','*Time = 0.4'),lambda s:s.replace('*Data = nodes','*Data = unbound')])
def test_missing_zero_duplicate_nonfinite_wrongtime_wrongname_rejected(corrupt):
    with pytest.raises(ValueError):parse(corrupt(text()))


def test_solver_requires_every_residual_and_rejects_failure_or_bad_final_norm():
    lines=['Nonlinear solution status: time= .5',' residual 1e-6 1e-15 1e-14',
           'Nonlinear solution status: time= 1',' residual 1e-6 1e-15 1e-14','N O R M A L T E R M I N A T I O N']
    args=dict(expected_times=[0,.5,1],residual_floor_N2=1e-20)
    assert o.check_solver_records(lines,**args)['passed']
    with pytest.raises(ValueError):o.check_solver_records(lines[:-1],**args)
    with pytest.raises(ValueError):o.check_solver_records(lines+['negative jacobian'],**args)
    lines[-2]=' residual 1e-6 1e-5 1e-14'
    assert not o.check_solver_records(lines,**args)['passed']


def test_actual_saved_coarse_headers_use_nine_significant_digits():
    directory=Path(__file__).resolve().parents[1]/'artifacts/mechanics/hbe-primitive-time-format-v1'
    provenance=json.loads((directory/'fixture-provenance.json').read_text())
    path=Path(__file__).resolve().parents[1]/provenance['extracted_fixture']['path']
    data=path.read_bytes()
    assert hashlib.sha256(data).hexdigest()==provenance['extracted_fixture']['sha256']
    records=list(o.iter_data_records(
        data.decode().splitlines(keepends=True), expected_times=[i/60 for i in range(61)],
        item_count=1, field_count=9, record_name='mechanics_nodes_si',
    ))
    assert len(records)==61
    assert records[1]['time']==.0166666667
    assert records[1]['declared_time']==1/60


@pytest.mark.parametrize('steps',[60,120])
def test_declared_grid_matches_tagged_header_format_without_broad_tolerance(steps):
    lines='\n'.join(
        f'*Step = {i}\n*Time = {i/steps:.9g}\n*Data = nodes\n1,1,2,3'
        for i in range(steps+1)
    )
    args=dict(expected_times=[i/steps for i in range(steps+1)],item_count=1,field_count=3,record_name='nodes')
    assert len(list(o.iter_data_records(lines.splitlines(),**args)))==steps+1
    wrong=lines.replace(f'*Time = {1/steps:.9g}',f'*Time = {1/steps+1e-8:.9g}',1)
    with pytest.raises(ValueError,match='physical load time'):
        list(o.iter_data_records(wrong.splitlines(),**args))


def test_declared_grid_alias_at_header_precision_rejected_before_read():
    def unopened():
        raise AssertionError('Aliased declaration must fail before primitive IO')
        yield ''
    with pytest.raises(ValueError,match='aliases'):
        list(o.iter_data_records(unopened(),expected_times=[0,.5,.50000000001,1],
                                 item_count=1,field_count=1,record_name='nodes'))

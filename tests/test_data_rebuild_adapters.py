"""Small offline regressions for source selection and frozen-plan replacement."""
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from data_rebuild_prepare_adapted import typed_files
from data_rebuild_batch import effective_subjects
from resectionlab.data_root import sha256_file


def test_json_sidecars_cannot_replace_their_nifti_observation():
    rows=[{'path':'source_T1w.nii.gz','modality':'T1w'},
          {'path':'source_T1w.json','modality':'T1w'},
          {'path':'._source_T1w.nii.gz','modality':'T1w'}]
    images,metadata=typed_files({'acquired_files':rows})
    assert images==rows[:1] and metadata==rows[1:2]
    with pytest.raises(ValueError,match='Ambiguous'):
        typed_files({'acquired_files':rows+[{'path':'other_T1w.nii.gz','modality':'T1w'}]})


def test_resource_replacement_keeps_original_plan_and_roles(tmp_path):
    catalog=tmp_path/'catalog';(catalog/'healthy-batch').mkdir(parents=True)
    plan={'subjects':[{'subject':'IXI051','person_id':'IXI:051','role':'TRAIN','mass_effect':True}]}
    plan_path=catalog/'healthy-batch/plan-v5.json';plan_path.write_text(json.dumps(plan))
    before=plan_path.read_bytes();source=tmp_path/'source.nii.gz';source.write_bytes(b'source bytes')
    replacement={'blocked_generation_host':'IXI:051','replacement':{'subject':'IXI084','person_id':'IXI:084','role':'TRAIN'},
        'replacement_acquisition':{'status':'bytes_and_header_validated_awaiting_independent_review',
            'files':[{'path':source.name,'sha256':sha256_file(source)}]}}
    path=catalog/'healthy-host-replacement-IXI051.json';path.write_text(json.dumps(replacement))
    effective=effective_subjects(tmp_path,plan,plan_path)
    assert effective==[{'subject':'IXI084','person_id':'IXI:084','role':'TRAIN','variants':4,'mass_effect':False}]
    assert plan_path.read_bytes()==before
    assert effective_subjects(tmp_path,plan,plan_path)==effective
    source.write_bytes(b'changed source')
    with pytest.raises(ValueError,match='source changed'):
        effective_subjects(tmp_path,plan,plan_path)

"""Independent generated metadata boundary controls. No actual-native entry call."""
from pathlib import Path
import importlib.util
import json
import hashlib
import pytest
ROOT=Path(__file__).resolve().parents[2]
AUTHOR=ROOT/'build/hbe-v5-native-comparison-preparation-v1'
spec=importlib.util.spec_from_file_location('independent_author_fixture',AUTHOR/'test_generated.py')
f=importlib.util.module_from_spec(spec);spec.loader.exec_module(f)
n=f.n

@pytest.mark.parametrize('field',['check_clock','terminal_clock','native_calls','readout_calls'])
def test_original_wrapper_timing_and_call_equations_refuse_contradictions(field):
 side,envelope,native,kwargs=f.extension_fixture()
 if field=='check_clock':side['extension_resource_charge']['launcher_elapsed_seconds_at_check']+=10.
 elif field=='terminal_clock':side['launcher_elapsed_seconds_terminal']+=10.
 elif field=='native_calls':side['native_calls_attempted']=2
 else:side['hbe_readout_calls_attempted']=2
 with pytest.raises(ValueError): n.extension_charge(side,envelope,native,**kwargs)


def test_final_original_chain_change_prevents_return(tmp_path,monkeypatch):
 manifest,binding,events=f.orchestration_fixture(tmp_path,monkeypatch)
 original=n.remaining.validate_prior_chain
 calls=0
 def changed(*args,**kwargs):
  nonlocal calls
  result=original(*args,**kwargs);calls+=1
  if calls==2:result['native_seconds']+=.1
  return result
 monkeypatch.setattr(n.remaining,'validate_prior_chain',changed)
 with pytest.raises(ValueError,match='native_chain_changed_after_comparison'):n.compare_native_rows(tmp_path,binding)
 assert calls==2


def test_late_loaded_origin_swap_prevents_return(tmp_path,monkeypatch):
 manifest,binding,events=f.orchestration_fixture(tmp_path,monkeypatch)
 calls=0
 def origins(*args,**kwargs):
  nonlocal calls
  calls+=1
  return {'scripts.generated':'scripts/generated.py' if calls==1 else 'scripts/swapped.py'}
 monkeypatch.setattr(n,'_verify_comparison_sources',origins)
 with pytest.raises(ValueError,match='comparison_import_origins_changed'):n.compare_native_rows(tmp_path,binding)
 assert calls==2


def test_later_extension_cannot_be_silently_omitted(tmp_path,monkeypatch):
 manifest,binding,events=f.orchestration_fixture(tmp_path,monkeypatch)
 manifest['rows'][n.remaining.ORDER[9]]['policy_extension']=None
 binding=f.write_bound(tmp_path,binding['path'],manifest)
 with pytest.raises(ValueError,match='continuation_policy_evidence_required'):n.compare_native_rows(tmp_path,binding)

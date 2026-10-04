import importlib.util,sys,pytest
from pathlib import Path
name='resectionlab._axis_inspection_prior_review'
spec=importlib.util.spec_from_file_location(name,Path('artifacts/native-axis-inspection-independent-v1/pre-review-source.py'))
module=importlib.util.module_from_spec(spec)
sys.modules[name]=module
spec.loader.exec_module(module)
class Plugin:
 def pytest_collection_modifyitems(self,items):
  for item in items:
   item.module.facade=module
raise SystemExit(pytest.main(['tests/test_native_axis_refinement_review.py','-q','-k','known_oblique_lps'],plugins=[Plugin()]))

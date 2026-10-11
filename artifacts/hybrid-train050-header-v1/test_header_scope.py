"""Generated-only scope and exact-header terminal acceptance checks."""
import ast
import hashlib
import json
from pathlib import Path
import tempfile
import types
import unittest

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
def module(path):
 value=types.ModuleType(path.stem);value.__file__=str(path)
 exec(compile(path.read_bytes(),str(path),'exec'),value.__dict__)
 return value
qc=module(HERE/'stage/src/resectionlab/remind_planning_qc.py')
owner=module(HERE/'run_headers_owned.py')
def case(subject='ReMIND-050',role='TRAIN'):
 return {'patient_id':subject,'patient_group':'ReMIND:'+subject.split('-')[1],'role':role,
         'series':[{'kind':k} for k in ('structural_t1ce','whole_tumor','cerebrum')]}
class Controls(unittest.TestCase):
 def test_fixed050_public_allowed(self):qc.validate_public_scope(case(),public_only=True)
 def test_extra_private_refused(self):
  v=case();v['series'].append({'kind':'ventricles'})
  with self.assertRaisesRegex(ValueError,'exact_public_series_only'):qc.validate_public_scope(v,public_only=True)
 def test_other_new_person_refused(self):
  with self.assertRaisesRegex(ValueError,'frozen_public_pilot_role'):qc.validate_public_scope(case('ReMIND-051'),public_only=True)
 def test_frozen_role_required(self):
  v=case();cohort={'members':[{'subject':'ReMIND-050','patient_group':'ReMIND:050','role':'SELECT'}]}
  with self.assertRaisesRegex(ValueError,'immutable_public_pilot_role'):qc.validate_role(v,cohort,public_only=True)
 def test_only_whitelist_ast_delta(self):
  before=ast.parse((ROOT/'src/resectionlab/remind_planning_qc.py').read_bytes());after=ast.parse((HERE/'stage/src/resectionlab/remind_planning_qc.py').read_bytes())
  def strip(tree):tree.body=[n for n in tree.body if not (isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='PUBLIC_SUBJECTS_BY_ROLE' for t in n.targets))];return ast.dump(tree,include_attributes=False)
  self.assertEqual(strip(before),strip(after))
 def test_terminal_requires_full_header_only(self):
  v={'status':'header_geometry_and_ancestry_projected','case_sha256':'case','executing_script_sha256':'reader','patient_id':'ReMIND-050','role':'TRAIN','public_only':True,'private_reference_loaded':False,'training_admitted':False,'optimizer_updates_performed':0,'objects_verified':194,'source_file_returned_bytes':27142222,'pixel_decode_calls':0,'native_conversion_performed':False}
  with tempfile.TemporaryDirectory() as d:
   p=Path(d)/'result.json';p.write_text(json.dumps(v));self.assertIsNone(owner.terminal_result(p,'headers','ReMIND-050','case','reader')[2])
   for key,bad in [('objects_verified',193),('pixel_decode_calls',1),('native_conversion_performed',True),('training_admitted',True)]:
    changed=dict(v);changed[key]=bad;p.write_text(json.dumps(changed));self.assertIsNotNone(owner.terminal_result(p,'headers','ReMIND-050','case','reader')[2],key)
 def test_crop_scope_and_fixed_phase_stay_closed(self):
  tree=ast.parse((HERE/'run_headers_owned.py').read_bytes())
  loops=[n for n in ast.walk(tree) if isinstance(n,ast.For) and isinstance(n.target,ast.Name) and n.target.id=='phase']
  self.assertEqual(len(loops),1);self.assertEqual(ast.literal_eval(loops[0].iter),('headers',))
  text=(HERE/'stage/src/resectionlab/remind_planning_qc.py').read_text()
  self.assertIn('("ReMIND-002", "ReMIND-015", "ReMIND-018", "ReMIND-045")',text)
if __name__=='__main__':unittest.main()

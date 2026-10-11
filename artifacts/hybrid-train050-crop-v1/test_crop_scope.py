"""Source-only explicit050 scope checks; no source files or arrays opened."""
import ast
from pathlib import Path
import unittest

HERE=Path(__file__).resolve().parent
HEADER=HERE/'stage/src/resectionlab/remind_planning_qc.py'
CROP=HERE/'crop-stage/src/resectionlab/remind_planning_qc.py'
def domain_scope(tree):
 function=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='run_crop')
 calls=[n for n in ast.walk(function) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='need' and len(n.args)>1 and isinstance(n.args[1],ast.Constant) and n.args[1].value=='domain_union_exact_TRAIN_public_scope']
 assert len(calls)==1
 return calls[0]
class ScopeControls(unittest.TestCase):
 def test_only_declared_scope_delta(self):
  a=ast.parse(HEADER.read_bytes());b=ast.parse(CROP.read_bytes())
  before=domain_scope(a);after=domain_scope(b)
  before.args[0]=after.args[0]
  self.assertEqual(ast.dump(a,include_attributes=False),ast.dump(b,include_attributes=False))
 def test050_only_new_domain_subject_and_originals_preserved(self):
  a=domain_scope(ast.parse(HEADER.read_bytes())).args[0]
  b=domain_scope(ast.parse(CROP.read_bytes())).args[0]
  def allowed(expression,subject,role='TRAIN',public=True):return eval(compile(ast.Expression(expression),'<generated scope>','eval'),{}, {'public_only':public,'case':{'role':role,'patient_id':subject}})
  for s in ['ReMIND-002','ReMIND-015','ReMIND-018','ReMIND-045']:
   self.assertTrue(allowed(a,s));self.assertTrue(allowed(b,s))
  self.assertFalse(allowed(a,'ReMIND-050'));self.assertTrue(allowed(b,'ReMIND-050'))
  for s,role,public in [('ReMIND-051','TRAIN',True),('ReMIND-013','SELECT',True),('ReMIND-037','SELECT',True),('ReMIND-067','MEASUREMENT_EVAL',True),('ReMIND-050','SELECT',True),('ReMIND-050','TRAIN',False)]:self.assertFalse(allowed(b,s,role,public))
if __name__=='__main__':unittest.main()

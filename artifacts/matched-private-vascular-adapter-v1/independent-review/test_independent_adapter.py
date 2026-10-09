"""Generated callback-mutation regressions; no model, image, patient or network."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2];C=ROOT/'build/matched-private-vascular-adapter-v1'
sys.path.insert(0,str(C))
import test_adapter as t
fixture=t.fixture
tmp_path=t.tmp_path

def run(fixture,sealed,refs,bindings,loaders,out):
 return t.adapter.evaluate_matched_private_vessels(**sealed,spec=fixture[0],planning_identity=fixture[1],reference_bindings=bindings,load_references=loaders,output_directory=out)

def test_caller_maps_cannot_change_preselected_reference_world(fixture,tmp_path):
 sealed,manifest=t.stage(fixture,tmp_path/'sealed');refs=t.references(fixture,sealed,manifest);other=t.references(fixture,sealed,manifest,positive=False)
 bindings={m:r.binding for m,r in refs.items()};calls=[]
 def get(m):calls.append(m);return refs[m]
 loaders={m:(lambda m=m:get(m)) for m in refs}
 def first():
  bindings['IL']=other['IL'].binding;loaders['IL']=lambda:other['IL']
  return get('SEARCH')
 loaders['SEARCH']=first
 report=run(fixture,sealed,refs,bindings,loaders,tmp_path/'evaluation')
 assert calls==list(t.adapter.METHODS)
 assert report['methods']['IL']['evaluation']['whole_tool']['positive_reference_cells']==2
 assert report['status']=='completed_fixed_generated_method_denominator'

def test_inplace_binding_with_recomputed_fingerprint_refuses_before_next_loader(fixture,tmp_path):
 sealed,manifest=t.stage(fixture,tmp_path/'sealed');refs=t.references(fixture,sealed,manifest)
 bindings={m:r.binding for m,r in refs.items()};calls=[]
 def get(m):calls.append(m);return refs[m]
 loaders={m:(lambda m=m:get(m)) for m in refs}
 def first():
  b=bindings['IL'];object.__setattr__(b,'mra_source_sha256','a'*64);object.__setattr__(b,'fingerprint',t.semantic_digest(b.record()))
  b.assert_intact() # Current self-consistency is insufficient; expected identity must be retained.
  return get('SEARCH')
 loaders['SEARCH']=first
 report=run(fixture,sealed,refs,bindings,loaders,tmp_path/'evaluation')
 assert 'IL' not in calls
 assert report['methods']['IL']['evaluation_status']=='evaluation_failed'
 assert report['status']=='batch_integrity_failed'
 assert set(report['methods'])==set(t.adapter.METHODS)

def test_late_mutation_of_already_scored_binding_invalidates_batch(fixture,tmp_path):
 sealed,manifest=t.stage(fixture,tmp_path/'sealed');refs=t.references(fixture,sealed,manifest)
 bindings={m:r.binding for m,r in refs.items()}
 loaders={m:(lambda r=r:r) for m,r in refs.items()}
 def last():
  b=bindings['SEARCH'];object.__setattr__(b,'mra_source_sha256','a'*64);object.__setattr__(b,'fingerprint',t.semantic_digest(b.record()))
  return refs['HYBRID']
 loaders['HYBRID']=last
 report=run(fixture,sealed,refs,bindings,loaders,tmp_path/'evaluation')
 assert report['status']=='batch_integrity_failed'
 assert set(report['methods'])==set(t.adapter.METHODS)
 assert report['methods']['SEARCH']['evaluation']['reference_binding_hash']!=bindings['SEARCH'].fingerprint

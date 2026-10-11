"""Generated analytic evaluator controls; no producer or acquired inputs.

Root runs this prepared suite only in the coordinated bounded control slot.
"""
from pathlib import Path
import importlib.util
import math
import struct
import unittest

PATH=Path(__file__).with_name('plane_volume_evaluator.py')
SPEC=importlib.util.spec_from_file_location('independent_partial_plane_evaluator',PATH)
E=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(E)
IDENTITY=((1.,0.,0.,0.),(0.,1.,0.,0.),(0.,0.,1.,0.),(0.,0.,0.,1.))
CENTERED=((1.,0.,0.,-1.),(0.,1.,0.,-1.),(0.,0.,1.,-1.),(0.,0.,0.,1.))
KEY=(1,1,1)

def stored32(value):return struct.unpack('!f',struct.pack('!f',value))[0]
def bounds(value):return {KEY:E.float32_fraction_enclosure(stored32(value))}
def partial(mask,fraction=.5,**kwargs):
 return E.evaluate_certified_union(CENTERED,1,[],{KEY:mask},bounds(fraction),tissue_plane_z_mm=-1.,target_plane_z_mm=0.,**kwargs)

class IndependentPlaneVolumeTests(unittest.TestCase):
 def test_axis_aligned_known_volumes(self):
  for plane,expected in [(-1.,1.),(-.5,1.),(-.25,.75),(0.,.5),(.25,.25),(.5,0.),(1.,0.)]:
   with self.subTest(plane=plane):self.assertAlmostEqual(E.affine_box_halfspace_volume(IDENTITY,(0,0,0),1.,plane),expected,places=14)

 def test_rotated_oblique_triangle_and_reflection(self):
  q=math.sqrt(.5)
  for sign in (1.,-1.):
   affine=((q,0.,q,0.),(0.,sign,0.,0.),(-q,0.,q,0.),(0.,0.,0.,1.))
   self.assertAlmostEqual(E.affine_box_halfspace_volume(affine,(0,0,0),1.,q/2),.125,places=13)
   self.assertAlmostEqual(E.affine_box_halfspace_volume(affine,(0,0,0),1.,0.),.5,places=13)

 def test_shear_translation_and_scaled_leaf(self):
  affine=((1.,0.,0.,5.),(0.,1.,0.,-3.),(1.,0.,1.,2.),(0.,0.,0.,1.))
  self.assertAlmostEqual(E.affine_box_halfspace_volume(affine,(0,0,0),1.,2.5),.125,places=13)
  self.assertAlmostEqual(E.affine_box_halfspace_volume(affine,(0,0,0),.5,2.),.5*.5**3,places=13)

 def test_float32_rounding_encloses_declared_true_fraction(self):
  for exact in (0.,-0.,1.,.1,1/3,.5,1e-30):
   lo,hi=E.float32_fraction_enclosure(stored32(exact))
   self.assertLessEqual(lo,exact);self.assertGreaterEqual(hi,exact)
   self.assertGreaterEqual(lo,0.);self.assertLessEqual(hi,1.)
  with self.assertRaises(ValueError):E.float32_fraction_enclosure(.1)
  for bad in (-.1,1.1,math.nan,math.inf):
   with self.assertRaises(ValueError):E.float32_fraction_enclosure(bad)

 def test_same_parent_mass_cannot_localize_evacuated_target(self):
  bottom,top=partial(0x55),partial(0xaa)
  self.assertEqual(bottom['source_evacuated_mm3'],top['source_evacuated_mm3'])
  self.assertEqual(bottom['cumulative_target_interval_source_mm3'],top['cumulative_target_interval_source_mm3'])
  self.assertEqual(bottom['physical_target_evacuated_mm3'],0.)
  self.assertEqual(top['physical_target_evacuated_mm3'],.5)
  self.assertTrue(bottom['physical_target_within_source_interval']);self.assertTrue(top['physical_target_within_source_interval'])
  # f*R=.25 is neither actual value; both allocations satisfy the same parent mass.
  self.assertNotEqual(top['physical_target_evacuated_mm3'],.5*.5)

 def test_support_boundary_outer_voxelization_kept_separate(self):
  partial_row=E.evaluate_certified_union(CENTERED,1,[],{KEY:0x55},bounds(.25),tissue_plane_z_mm=0.,target_plane_z_mm=.25)
  full_row=E.evaluate_certified_union(CENTERED,1,[KEY],{},bounds(.25),tissue_plane_z_mm=0.,target_plane_z_mm=.25)
  self.assertEqual(partial_row['source_evacuated_mm3'],.5)
  self.assertEqual(partial_row['physical_tissue_evacuated_lower_mm3'],0.)
  self.assertEqual(partial_row['physical_target_evacuated_mm3'],0.)
  self.assertEqual(full_row['source_evacuated_mm3'],1.)
  self.assertEqual(full_row['physical_tissue_evacuated_lower_mm3'],.5)
  self.assertEqual(full_row['physical_target_evacuated_mm3'],.25)
  self.assertTrue(partial_row['physical_target_within_source_interval']);self.assertTrue(full_row['physical_target_within_source_interval'])

 def test_cumulative_target_interval_not_sum_of_independent_allocations(self):
  self.assertEqual(E.target_allocation_interval(1.,.25,(.5,.5)),(0.,.25))
  self.assertEqual(E.target_allocation_interval(1.,.5,(.5,.5)),(0.,.5))
  self.assertEqual(E.target_allocation_interval(1.,1.,(.5,.5)),(.5,.5))
  self.assertGreater(.25+.5,E.target_allocation_interval(1.,.5,(.5,.5))[1])

 def test_source_target_interval_inconsistency_remains_visible(self):
  row=partial(0xaa,fraction=0.)
  self.assertFalse(row['physical_target_within_source_interval'])
  self.assertFalse(row['parent_target_mass_enclosure_pass'])
  self.assertEqual(row['physical_target_evacuated_mm3'],.5)

 def test_duplicate_completed_or_out_of_range_leaf_state_refuses(self):
  for removed,parts in [([KEY,KEY],{}),([KEY],{KEY:1}),([],{KEY:255}),([],{KEY:0}),([],{KEY:256})]:
   with self.subTest(removed=removed,parts=parts):
    with self.assertRaises(ValueError):E.evaluate_certified_union(CENTERED,1,removed,parts,bounds(.5))

 def test_explicit_work_limits_and_cancellation_refuse(self):
  with self.assertRaises(E.EvaluationLimit):partial(0xaa,max_leaf_visits=1)
  with self.assertRaises(E.EvaluationLimit):partial(0xaa,cancelled=lambda:True)
  with self.assertRaises(E.EvaluationLimit):E.evaluate_certified_union(CENTERED,1,[KEY,(1,1,2)],{}, {KEY:(0.,1.),(1,1,2):(0.,1.)},max_parent_cells=1)

 def test_whole_cells_once_and_one_sided_partial_no_leaf_visits(self):
  keys=[(i,1,4) for i in range(100)]
  row=E.evaluate_certified_union(IDENTITY,3,keys,{(100,1,4):1},{k:(1.,1.) for k in keys+[(100,1,4)]})
  self.assertEqual(row['source_evacuated_mm3'],100+1/512)
  self.assertEqual(row['physical_target_evacuated_mm3'],100+1/512)
  self.assertEqual(row['work']['leaf_visits'],0)
  self.assertEqual(row['work']['face_clips'],0)
  self.assertEqual(row['whole_cell_physical_lower_mm3'],100.)

 def test_bad_affine_and_nonnested_reference_refuse(self):
  with self.assertRaises(ValueError):E.affine_box_halfspace_volume(((0,0,0,0),)*3+((0,0,0,1),),(0,0,0),1.,0.)
  with self.assertRaises(ValueError):E.evaluate_certified_union(IDENTITY,1,[],{}, {},tissue_plane_z_mm=1.,target_plane_z_mm=0.)

if __name__=='__main__':unittest.main()

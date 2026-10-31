import numpy as np

from resectionlab.surgical_engine import SurgicalEngine
from resectionlab.surgical_geometry_view import raster_view
from resectionlab.surgical_scene import SurgicalScene


def test_conditional_raster_matches_canonical_owner_union_and_does_not_mutate_engine():
    labels=np.zeros((4,4,4),np.uint16);labels[1,1,1]=1;labels[2,1,1]=2
    scene=SurgicalScene(labels,np.eye(4),[{'id':1,'material':'soft_tissue'},{'id':2,'material':'tumor'}],{'kind':'generated'}, {})
    engine=SurgicalEngine(scene);cell=int(np.ravel_multi_index((1,1,1),scene.shape))
    engine.state['deformation']={str(cell):[.6,0.,0.]};before=engine.snapshot()
    u=np.zeros((1,8,3));u[:,:,0]=.6
    conditional,pairs=raster_view(scene,[],[cell],u);reference=engine.occupancy();all_ids=np.arange(labels.size)
    np.testing.assert_array_equal(conditional[all_ids],reference[all_ids])
    assert conditional.material_ids(all_ids)==reference.material_ids(all_ids)
    assert conditional.material_ids([37])==[21,37] and pairs==[[21,37]]
    assert engine.snapshot()==before

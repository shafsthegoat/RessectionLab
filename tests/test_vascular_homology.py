"""Independent topology examples and GF2 certificate checks for source subgraphs."""
import itertools
import numpy as np
import pytest
from resectionlab.vascular_homology import representative_subgraph,HomologyLimit


def graph(points):
    xyz=np.array(points,np.int64)
    edges=np.array([(a,b) for a,b in itertools.combinations(range(len(xyz)),2)
        if np.max(np.abs(xyz[a]-xyz[b]))==1],np.int64).reshape(-1,2)
    return xyz,edges


def test_filled_triangle_removes_contractible_chord_without_new_edges_or_nodes():
    xyz,edges=graph([(0,0,0),(1,0,0),(1,1,0)])
    result=representative_subgraph(xyz,edges)
    assert result['boundary_rank']==1 and result['betti1']==0
    assert len(result['edges'])==2 and result['components']==1
    np.testing.assert_array_equal(result['edges'],edges[result['source_edge_indices']])


def test_true_ring_loop_is_retained_but_local_shortcuts_are_removed():
    points=[p for p in itertools.product(range(5),range(5)) if p[0] in (0,4) or p[1] in (0,4)]
    xyz,edges=graph([(x,y,0) for x,y in points]);result=representative_subgraph(xyz,edges)
    assert result['betti1']==1 and result['components']==1
    assert result['boundary_rank']>0
    assert len(result['edges'])-len(xyz)+1==1


def test_eight_voxels_with_common_corner_have_no_spurious_cycles():
    xyz,edges=graph(list(itertools.product((0,1),repeat=3)))
    assert len(edges)==28
    result=representative_subgraph(xyz,edges)
    assert result['betti1']==0 and result['boundary_rank']==21
    assert len(result['edges'])==7 and result['triangle_count']==56


def test_disconnected_components_and_isolated_source_vertices_remain():
    xyz,edges=graph([(0,0,0),(1,0,0),(1,1,0),(8,8,8),(12,12,12),(13,12,12)])
    result=representative_subgraph(xyz,edges)
    assert result['components']==3 and result['betti1']==0
    assert len(result['edges'])==len(xyz)-3


def test_hollow_cube_has_zero_H1_but_still_needs_separate_H2_check():
    xyz,edges=graph([p for p in itertools.product(range(3),repeat=3) if p!=(1,1,1)])
    result=representative_subgraph(xyz,edges)
    assert result['betti1']==0
    # A1D graph cannot represent the cube shell's one enclosed cavity.
    assert len(result['edges'])==len(xyz)-1


def test_certificate_basis_is_independent_and_has_exact_declared_pivots():
    xyz,edges=graph([(x,y,0) for x,y in itertools.product(range(4),repeat=2)])
    result=representative_subgraph(xyz,edges)
    rows=[sum(int(word)<<(64*i) for i,word in enumerate(words)) for words in result['boundary_basis_words']]
    pivots=result['boundary_pivot_chord_columns']
    assert len(rows)==result['boundary_rank']
    for row,pivot in zip(rows,pivots): assert row.bit_length()-1==pivot
    # Separate dense GF2 elimination establishes row rank and complement.
    width=len(result['chord_edge_indices'])
    array=np.array([[(row>>bit)&1 for bit in range(width)] for row in rows],np.uint8)
    rank=0
    for column in range(width):
        found=np.flatnonzero(array[rank:,column])
        if not len(found):continue
        k=rank+int(found[0]);array[[rank,k]]=array[[k,rank]]
        for other in range(len(array)):
            if other!=rank and array[other,column]:array[other]^=array[rank]
        rank+=1
    assert rank==result['boundary_rank']
    assert len(set(result['removed_edge_indices'])&set(result['forest_edge_indices']))==0
    # Membership and completeness are checked against all independently
    # enumerated raw-graph triangle boundaries, not just declared pivots.
    edge_lookup={tuple(pair):k for k,pair in enumerate(edges)}
    columns={int(edge):column for column,edge in enumerate(result['chord_edge_indices'])}
    triangles=[]
    for a,b,c in itertools.combinations(range(len(xyz)),3):
        pairs=[(a,b),(a,c),(b,c)]
        if all(pair in edge_lookup for pair in pairs):triangles.append([edge_lookup[pair] for pair in pairs])
    def projected(triples):
        out=np.zeros((len(triples),width),np.uint8)
        for i,triple in enumerate(triples):
            for edge in triple:
                if int(edge) in columns:out[i,columns[int(edge)]]^=1
        return out
    def dense_rank(values):
        values=values.copy();rank=0
        for column in range(values.shape[1]):
            found=np.flatnonzero(values[rank:,column])
            if not len(found):continue
            k=rank+int(found[0]);values[[rank,k]]=values[[k,rank]]
            for other in range(rank+1,len(values)):
                if values[other,column]:values[other]^=values[rank]
            rank+=1
        return rank
    all_rows=projected(triangles);generators=projected(result['triangle_generators_edge_indices'])
    original_basis=np.array([[(row>>bit)&1 for bit in range(width)] for row in rows],np.uint8)
    assert dense_rank(all_rows)==dense_rank(generators)==rank
    assert dense_rank(np.vstack((all_rows,original_basis)))==rank


def test_missing_or_nonlocal_edge_cannot_claim_voxel_nerve_certificate():
    xyz,edges=graph([(0,0,0),(1,0,0),(1,1,0)])
    with pytest.raises(ValueError,match='Complete26'):representative_subgraph(xyz,edges[:-1])
    with pytest.raises(ValueError,match='voxel-intersection'):
        representative_subgraph(np.array([[0,0,0],[3,0,0]]),np.array([[0,1]]))


def test_caps_abstain_without_partial_representative():
    xyz,edges=graph(list(itertools.product((0,1),repeat=3)))
    with pytest.raises(HomologyLimit,match='chord cap'):representative_subgraph(xyz,edges,max_chords=2)
    with pytest.raises(HomologyLimit,match='triangle cap'):representative_subgraph(xyz,edges,max_triangles=2)

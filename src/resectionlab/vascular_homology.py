"""H1 representative subgraph of a complete26 voxel-intersection graph.

Closed axis-aligned voxel cells form a convex good cover. Pairwise-intersecting
boxes have common intersection, so the nerve is the clique complex of complete
26 adjacency. Its H1 is graph cycle space modulo triangle boundaries (GF2).

Keep a deterministic spanning forest, project triangle boundaries onto chord
coordinates, and delete their echelon-pivot chords. Nonpivot fundamental cycles
are a complement to that boundary subspace. Inclusion therefore induces an H0
and H1 isomorphism; this does NOT establish full homotopy, geometry or physiology.
Higher homology must be checked separately. No edge or source node is invented.
"""
from __future__ import annotations

import itertools
import numpy as np

ALGORITHM = 'voxel-nerve-H1-representatives-v1'


class HomologyLimit(ValueError):
    """Resource abstention; no partial representative graph is admitted."""


def representative_subgraph(indices, edges, *, max_chords=4096, max_triangles=1_000_000):
    for value,maximum in ((max_chords,4096),(max_triangles,1_000_000)):
        if type(value) is not int or not 0 < value <= maximum: raise ValueError('Invalid homology cap')
    xyz=np.asarray(indices); edges=np.asarray(edges)
    if (xyz.ndim != 2 or xyz.shape[1] != 3 or xyz.dtype.kind not in 'iu' or len(xyz)>100_000
            or edges.ndim != 2 or edges.shape[1] != 2 or edges.dtype.kind not in 'iu' or len(edges)>1_000_000):
        raise ValueError('Bounded integer source indices and edges required')
    if xyz.size and (int(xyz.min()) < -(2**31) or int(xyz.max()) >= 2**31):
        raise ValueError('Source voxel coordinate exceeds declared integer domain')
    xyz=xyz.astype(np.int64,copy=False)
    n=len(xyz); m=len(edges)
    if edges.size and (edges.min()<0 or edges.max()>=n or np.any(edges[:,0]>=edges[:,1])):
        raise ValueError('Distinct canonical source-edge endpoints required')
    lookup={tuple(map(int,p)):i for i,p in enumerate(xyz)}
    if len(lookup)!=n: raise ValueError('Distinct source voxel indices required')
    neighbor=[set() for _ in range(n)]; edge_index={}
    for k,(a,b) in enumerate(edges):
        a,b=int(a),int(b)
        if (a,b) in edge_index or np.max(np.abs(xyz[a].astype(np.int64)-xyz[b])) != 1:
            raise ValueError('Distinct26 voxel-intersection edges required')
        neighbor[a].add(b);neighbor[b].add(a);edge_index[(a,b)]=k
    offsets=[p for p in itertools.product((-1,0,1),repeat=3) if p!=(0,0,0)]
    for i,p in enumerate(xyz):
        for delta in offsets:
            key=tuple(int(v)+d for v,d in zip(p,delta));j=lookup.get(key)
            if j is not None and j not in neighbor[i]:
                raise ValueError('Complete26 adjacency required for the voxel-nerve certificate')
    parent=list(range(n)); tree=np.zeros(m,bool)
    def find(i):
        while parent[i]!=i:
            parent[i]=parent[parent[i]];i=parent[i]
        return i
    # Sorting by source-node pair fixes all ties independently of image values,
    # physical outcome, conductance, caliber or target anatomy.
    components=n
    for (a,b),k in sorted(edge_index.items()):
        ra,rb=find(a),find(b)
        if ra!=rb:
            parent[rb]=ra;tree[k]=True;components-=1
    chords=np.flatnonzero(~tree)
    if len(chords)>max_chords: raise HomologyLimit('Vascular H1 chord cap exceeded')
    chord_column={int(k):i for i,k in enumerate(chords)}
    basis={}; triangles=0; generators=[]
    for a in range(n):
        for b in sorted(v for v in neighbor[a] if v>a):
            for c in sorted(v for v in neighbor[a]&neighbor[b] if v>b):
                triangles+=1
                if triangles>max_triangles: raise HomologyLimit('Vascular H1 triangle cap exceeded')
                row=0
                for pair in ((a,b),(a,c),(b,c)):
                    column=chord_column.get(edge_index[pair])
                    if column is not None: row^=1<<column
                # Highest-bit pivots certify that vectors with zero pivot
                # coordinates intersect the boundary space only at zero.
                while row:
                    pivot=row.bit_length()-1
                    if pivot in basis: row^=basis[pivot]
                    else:
                        basis[pivot]=row
                        generators.append([edge_index[pair] for pair in ((a,b),(a,c),(b,c))]);break
    pivots=sorted(basis,reverse=True)
    removed=chords[pivots] if pivots else np.empty(0,np.int64)
    keep=np.ones(m,bool);keep[removed]=False
    word_count=(len(chords)+63)//64
    words=np.zeros((len(pivots),word_count),np.uint64)
    for k,pivot in enumerate(pivots):
        for word in range(word_count): words[k,word]=(basis[pivot]>>(64*word))&((1<<64)-1)
    return {'edges':edges[keep].copy(),'source_edge_indices':np.flatnonzero(keep),
        'forest_edge_indices':np.flatnonzero(tree),'chord_edge_indices':chords,
        'boundary_pivot_chord_columns':np.array(pivots,np.int64),
        'removed_edge_indices':removed,'boundary_basis_words':words,
        'triangle_generators_edge_indices':np.array(generators,np.int64).reshape(-1,3),
        'boundary_rank':len(basis),'triangle_count':triangles,'components':components,
        'betti1':len(chords)-len(basis)}

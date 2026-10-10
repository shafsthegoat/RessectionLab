"""Read-only public-mask geometry measurement; does not define a legal cavity."""
import hashlib,itertools
import numpy as np
from scipy.ndimage import binary_fill_holes,label,generate_binary_structure
from resectionlab.geometry import GeometryScene,GENERIC_TOOLS,capsule_voxel_indices
EPS=1e-8
BINS=('wholly_proximal','straddling_or_touching_aperture','wholly_inward')
WITNESS_LIMIT=4

def digest(indices):
    a=np.ascontiguousarray(indices,dtype=np.int64)
    return hashlib.sha256(str(a.shape).encode()+b'|int64|'+a.tobytes()).hexdigest()

def depth_bins(indices,affine,entry,normal):
    centres=np.asarray(indices)@affine[:3,:3].T+affine[:3,3]
    depth=(centres-entry)@normal
    half=.5*sum(abs(float(normal@affine[:3,i])) for i in range(3))
    low,high=depth-half,depth+half
    proximal=high < -EPS;inward=low > EPS
    return dict(zip(BINS,(proximal,~(proximal|inward),inward))),low,high

def summarize(indices,affine,entry,normal):
    bins,low,high=depth_bins(indices,affine,entry,normal)
    return {'count':len(indices),'indices_sha256':digest(indices),'depth_range_mm':None if not len(indices) else [float(low.min()),float(high.max())],
      'by_full_cell_depth':{k:{'count':int(v.sum()),'indices_sha256':digest(indices[v]),'witness_indices':indices[v][:WITNESS_LIMIT].tolist(),'witness_limit':WITNESS_LIMIT} for k,v in bins.items()}}

def query_parts(entry,tip,tool):
    delta=tip-entry;length=float(np.linalg.norm(delta))
    if length<=1e-9:raise ValueError('Nonzero saved motion required')
    axis=delta/length
    return [('initial_shaft',entry-tool.working_length_mm*axis,entry-tool.tip_length_mm*axis,tool.shaft_radius_mm),
      ('initial_tip',entry-tool.tip_length_mm*axis,entry,tool.tip_radius_mm),
      ('swept_shaft',entry-tool.working_length_mm*axis,tip-tool.tip_length_mm*axis,tool.shaft_radius_mm),
      ('swept_tip',entry-tool.tip_length_mm*axis,tip,tool.tip_radius_mm)]

def aperture_candidates(S,T,Ds,affine,entry,normal,radius,checkpoint=lambda:None):
    """Fixed disc; a proximal cell's distal face touches the aperture plane."""
    axis=affine[:3,0]/np.linalg.norm(affine[:3,0])
    if min(np.linalg.norm(normal-axis),np.linalg.norm(normal+axis))>1e-10:
        raise ValueError('Fixed native-axis0 aperture normal required; oblique face contact is not modeled')
    O=S|T;D=Ds|T;U=~D;F=Ds&~O
    connected=~binary_fill_holes(O|U)
    labels,n=label(F,structure=generate_binary_structure(3,1));sizes=np.bincount(labels.ravel())
    # Bounded physical sphere contains every potential disc-adjacent affine cell.
    inverse=np.linalg.inv(affine);centre=entry@inverse[:3,:3].T+inverse[:3,3]
    corners=np.array(list(itertools.product((-.5,.5),repeat=3)))@affine[:3,:3].T
    cell_radius=float(np.linalg.norm(corners,axis=1).max());extent=np.linalg.norm(inverse[:3,:3],axis=1)*(radius+cell_radius)
    lo=np.maximum(0,np.floor(centre-extent).astype(int));hi=np.minimum(np.array(S.shape)-1,np.ceil(centre+extent).astype(int))
    indices=np.array(list(itertools.product(*(range(lo[i],hi[i]+1) for i in range(3)))),dtype=np.int64).reshape(-1,3)
    _,low,high=depth_bins(indices,affine,entry,normal)
    points=indices@affine[:3,:3].T+affine[:3,3]
    relative=points[:,None,:]+corners[None,:,:]-entry
    depths=relative@normal;radial=relative-depths[:,:,None]*normal
    inside=np.linalg.norm(radial,axis=2).max(1)<radius-EPS
    face=(high>=-EPS)&(high<=EPS)&(low < -EPS)
    geometric=indices[face&inside];candidates=geometric[F[tuple(geometric.T)]]
    components=sorted(set(int(v) for v in labels[tuple(candidates.T)]));components=[v for v in components if v!=0]
    touching=set();touchingS=set();touchingT=set()
    for cell in candidates:
      for axis in range(3):
       for sign in (-1,1):
        q=cell.copy();q[axis]+=sign
        if np.all(q>=0) and np.all(q<np.array(S.shape)):
         key=tuple(int(x) for x in q)
         if O[key]:touching.add(key)
         if S[key]:touchingS.add(key)
         if T[key]:touchingT.add(key)
    boundary=np.zeros(S.shape,bool)
    for axis in range(3):
      for endpoint in (0,-1):
       sl=[slice(None)]*3;sl[axis]=endpoint;boundary[tuple(sl)]=True
    checkpoint()
    return {'definition':'diagnostic K: cell lies proximal and its distal face touches fixed axis0 aperture plane within1e-8mm; entire projected cell footprint strictly inside unchanged radius; Ds true and S,T false',
      'not_an_admitted_cavity':True,'source_zero_is_physical_air':False,
      'boundary_cell_count':int(boundary.sum()),'boundary_U_count':int((boundary&U).sum()),'boundary_known_zero_seed_count':int((boundary&F).sum()),
      'actual_initial_connected_free_count':int(connected.sum()),'known_source_zero_component_count':int(n),
      'geometric_disc_face_candidates':len(geometric),'source_known_zero_candidates':summarize(candidates,affine,entry,normal),
      'candidate_connected_components':[{'component_id':i,'source_known_zero_cells':int(sizes[i])} for i in components],
      'candidate_face_adjacent_material_cells':len(touching),'candidate_face_adjacent_raw_S_cells':len(touchingS),'candidate_face_adjacent_full_T_cells':len(touchingT),
      'no_cells_cleared_seeded_or_marked_known':True}

def diagnose_case(S,T,Ds,affine,aperture,inventory,checkpoint=lambda:None,on_query=lambda:None):
    if S.dtype!=bool or T.dtype!=bool or Ds.dtype!=bool or S.shape!=T.shape or S.shape!=Ds.shape:raise ValueError('Exact shared binary grids')
    if np.any(S&~Ds):raise ValueError('Raw support outside source domain')
    O=S|T;D=Ds|T;U=~D
    entry=np.asarray(aperture['centre_mm'],float);normal=np.asarray(aperture['normal_inward'],float)
    if abs(float(normal@normal)-1)>1e-9:raise ValueError('Unit fixed aperture normal required')
    scene=GeometryScene(np.zeros(S.shape,bool),affine);tools={t.tool_id:t for t in GENERIC_TOOLS}
    if scene._orthogonal_spacing is None:raise ValueError('Pinned orthogonal planning grid required')
    result={'scope':'unchanged public masks and saved motions; geometric measurements only','queries':[],
      'count_interpretation':'Raw S and full T may overlap. Initial/swept shaft/tip measurements also overlap; their counts are not an additive motion union, removed volume or strategy outcome.',
      'classification_and_strict_disc_tolerance_mm':EPS,'capsule_contact_tolerance':'unchanged1e-9mm in pinned geometry.py; orthogonal exact-cell capsule supercover',
      'raw_S_full_T_U_global_depth_counts':{k:summarize(np.argwhere(a),affine,entry,normal) for k,a in [('raw_S',S),('full_T',T),('derived_U',U)]},
      'initial_exposure':aperture_candidates(S,T,Ds,affine,entry,normal,aperture['radius_mm'],checkpoint),
      'geometric_queries':0,'native_previews':0,'transitions':0,'models':0,'source_or_state_modified':False}
    for saved in inventory['emitted']:
      checkpoint();start=np.asarray(saved['entry_mm'],float);end=np.asarray(saved['tip_mm'],float);tool=tools[saved['tool_id']]
      for part,a,b,radius in query_parts(start,end,tool):
        checkpoint();on_query();result['geometric_queries']+=1;indices=capsule_voxel_indices(scene,a,b,radius)
        selected={k:indices[mask[tuple(indices.T)]] for k,mask in [('raw_S',S),('full_T',T),('derived_U',U)]}
        result['queries'].append({'proposal_id':saved['proposal_id'],'tool_id':tool.tool_id,'part':part,'start_mm':a.tolist(),'end_mm':b.tolist(),'radius_mm':radius,
          'saved_reason':saved['reason'],'full_in_grid_supercover_cells':len(indices),'full_in_grid_supercover_sha256':digest(indices),
          'intersections':{k:summarize(v,affine,entry,normal) for k,v in selected.items()},
          'outside_image_geometry':'unassessed; this query enumerates only within-grid cells',
          'aperture_depth_is_anatomical_intracranial_classification':False})
    result['unchanged_world_legal_motion_claim']=False
    return result

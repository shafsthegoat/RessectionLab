"""Seeded lobulated implicit surfaces constrained by an estimated tissue domain.

This is an engineering observation model, not a clinical tumor-growth model.
The domain closes surfaces smoothly in physical space; no bounding-box clipping
or axis-aligned solid is used to construct a tumor.
"""
from __future__ import annotations
import numpy as np
from scipy.ndimage import distance_transform_edt, label, gaussian_filter
from scipy.spatial import ConvexHull


def signed_domain(mask, affine):
    mask=np.asarray(mask,bool);a=np.asarray(affine,float)
    spacing=np.linalg.norm(a[:3,:3],axis=0)
    if mask.ndim!=3 or not mask.any() or np.any(spacing<=0):raise ValueError('Nonempty physical tissue domain required')
    directions=a[:3,:3]/spacing
    if not np.allclose(directions.T@directions,np.eye(3),atol=1e-4):raise ValueError('Orthogonal physical grid required')
    return (distance_transform_edt(mask,sampling=spacing)-distance_transform_edt(~mask,sampling=spacing)).astype(np.float32)


def shape_metrics(mask, affine):
    """Independent voxel and closed-surface checks that reject cuboids."""
    from skimage.measure import marching_cubes,mesh_surface_area
    mask=np.asarray(mask,bool);occupied=np.argwhere(mask)
    if not len(occupied):raise ValueError('Empty shape')
    low=occupied.min(0);high=occupied.max(0)+1;spacing=np.linalg.norm(np.asarray(affine)[:3,:3],axis=0)
    cropped=np.pad(mask[tuple(slice(int(a),int(b))for a,b in zip(low,high))],2)
    scalar=gaussian_filter(cropped.astype(np.float32),.45)
    vertices,faces,_,_=marching_cubes(scalar,.5,spacing=spacing)
    triangles=vertices[faces].astype(float)
    volume=abs(float(np.einsum('ij,ij->i',triangles[:,0],np.cross(triangles[:,1],triangles[:,2])).sum()/6))
    hull=ConvexHull(vertices);surface=float(mesh_surface_area(vertices,faces))
    components,count=label(mask)
    return {'voxel_count':int(mask.sum()),'volume_mm3':float(mask.sum()*abs(np.linalg.det(np.asarray(affine)[:3,:3]))),
        'bounding_box_fill':float(mask.sum()/np.prod(high-low)),'surface_solidity':volume/float(hull.volume),
        'sphericity':float(np.pi**(1/3)*(6*volume)**(2/3)/surface),'components':int(count),
        'is_cuboid':bool(mask.sum()==np.prod(high-low)),'bounds_native_exclusive':[low.tolist(),high.tolist()]}


def organic_geometry(domain, affine, *, center_voxel, seed, radii_mm=(9.,12.,13.), signed=None):
    domain=np.asarray(domain,bool);a=np.asarray(affine,float);spacing=np.linalg.norm(a[:3,:3],axis=0)
    center=np.asarray(center_voxel,int);radii=np.asarray(radii_mm,float)
    if center.shape!=(3,) or np.any(center<0) or np.any(center>=domain.shape) or not domain[tuple(center)]:
        raise ValueError('Center must lie inside the declared tissue domain')
    if radii.shape!=(3,) or not np.isfinite(radii).all() or np.any(radii<3) or np.any(radii>30):raise ValueError('Radii must be finite physical sizes in [3,30] mm')
    distance=signed_domain(domain,a)if signed is None else np.asarray(signed)
    if distance.shape!=domain.shape or not np.isfinite(distance).all():raise ValueError('Invalid signed tissue domain')
    rng=np.random.default_rng(seed);rotation,_=np.linalg.qr(rng.normal(size=(3,3)))
    if np.linalg.det(rotation)<0:rotation[:,0]*=-1
    count=int(rng.integers(3,7));directions=rng.normal(size=(count,3));directions/=np.linalg.norm(directions,axis=1)[:,None]
    amplitudes=rng.uniform(.25,.65,count);sharpness=rng.uniform(6.,13.,count);phase=rng.uniform(-np.pi,np.pi,3)
    # An analytic bound covers even several lobes pointing in one direction.
    # The allocation crop must never become a flat face of the generated shape.
    maximum_radius=1.+float(amplitudes.sum())+.13
    half=np.ceil((radii.max()*1.35*maximum_radius+2.)/spacing).astype(int)
    low=np.maximum(0,center-half);high=np.minimum(domain.shape,center+half+1)
    roi=tuple(slice(int(x),int(y))for x,y in zip(low,high))
    grid=np.stack(np.meshgrid(*(np.arange(x,y,dtype=np.float32)-c for x,y,c in zip(low,high,center)),indexing='ij'),-1)
    local=grid@((a[:3,:3].T@rotation)/radii).astype(np.float32)
    radial=np.linalg.norm(local,axis=-1);unit=local/np.maximum(radial[...,None],1e-6)
    radius=np.ones(radial.shape,np.float32)
    for direction,amplitude,kappa in zip(directions,amplitudes,sharpness):
        radius+=amplitude*np.exp(kappa*(unit@direction-1))
    # Low-frequency asymmetry between lobes, rather than voxel noise.
    radius+=.13*np.sin(3*unit[...,0]+phase[0])*np.cos(4*unit[...,1]+phase[1])*np.sin(3*unit[...,2]+phase[2])
    q=radial/radius
    allowed=distance[roi]-.35
    def level(threshold):
        raw=(threshold-q)*radii.min()
        # Smooth minimum preserves both inequalities, hence containment.
        field=-.7*np.logaddexp(-raw/.7,-allowed/.7)
        return field>=0
    whole=level(1.);core=level(.57);outer=level(1.35)
    local_center=tuple(center-low)
    def seed_component(mask):
        numbered,_=label(mask);component=numbered[local_center]
        if component==0:raise ValueError('Implicit surface does not contain its seed')
        return numbered==component
    before=int(whole.sum());whole=seed_component(whole);core=seed_component(core)&whole;outer=seed_component(outer)
    masks={}
    for name,small in [('synthetic_core',core),('synthetic_rim',whole&~core),('synthetic_edema',outer&~whole)]:
        full=np.zeros(domain.shape,bool);full[roi]=small
        if not full.any() or np.any(full&~domain):raise ValueError('Empty or out-of-domain organic compartment')
        if any(np.take(full,[0,n-1],axis=ax).any()for ax,n in enumerate(domain.shape)):
            raise ValueError('Organic shape reaches image boundary; preserve larger source FOV')
        masks[name]=full
    target=masks['synthetic_core']|masks['synthetic_rim'];metrics=shape_metrics(target,a)
    if metrics['is_cuboid'] or metrics['bounding_box_fill']>.75 or metrics['surface_solidity']>.98 or metrics['components']!=1:
        raise ValueError('Organic shape gate refused an overly solid, regular or disconnected target')
    return masks,{'recipe':'lobulated-smooth-tissue-domain-v1','seed':int(seed),'center_voxel':center.tolist(),
        'center_ras_mm':(a[:3,:3]@center+a[:3,3]).tolist(),'radii_mm':radii.tolist(),'rotation_ras':rotation.tolist(),
        'lobe_directions_local':directions.tolist(),'lobe_amplitudes':amplitudes.tolist(),'lobe_sharpness':sharpness.tolist(),
        'number_of_shape_lobes':count,'phase':phase.tolist(),'domain_surface_definition':'smooth minimum of lobulated radial field and estimated tissue signed distance minus0.35mm',
        'removed_non_seed_component_voxels':before-int(whole.sum()),'metrics':metrics,'clinical_validation':False,
        'axis_aligned_box_used_as_shape':False,'outside_tissue_voxels':0}

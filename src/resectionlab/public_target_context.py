"""Compact context from permitted target/support/frame and committed cavity only.

This is a lossy observation, not an alternate goal, score, simulator or admission
gate. Uncovered membership is unknown; outside-support membership is retained.
"""
from dataclasses import dataclass, field
from typing import Mapping
import numpy as np

from .core import array_digest, freeze_json, semantic_digest, thaw_json

VERSION='full-supplied-public-target-context-v1'
FEATURE_WIDTH=16
RELATION_WIDTH=4
MAX_SOURCE_VOXELS=32_000_000
_PREPARED=object()


def _immutable(value,dtype):
    value=np.ascontiguousarray(value,dtype=dtype)
    return np.frombuffer(value.tobytes(),dtype=value.dtype).reshape(value.shape)


def _layout(value):
    root=value
    while isinstance(root,np.ndarray) and root.base is not None:root=root.base
    if value.flags.writeable or not isinstance(root,bytes):
        raise ValueError('Public target array lost immutable backing')
    return (id(value),id(root),value.shape,value.dtype.str,value.strides)


def _hash(value):
    return isinstance(value,str) and len(value)==71 and value.startswith('sha256:') and all(c in '0123456789abcdef' for c in value[7:])


@dataclass(frozen=True)
class PublicTargetContext:
    static: Mapping
    committed_overlap_mm3: float
    committed_cavity_hash: str
    local_cavity_hash: str
    _identity: str=field(init=False,repr=False)

    def __post_init__(self):
        record=freeze_json(self.static)
        keys={'version','source_hash','track','source_kind','derivation','target_hash','domain_hash','support_hash',
            'source_affine_ras_mm','source_shape','crop_origin','crop_shape','crop_affine_hash','local_target_hash','local_support_hash',
            'local_domain_hash','available','domain_coverage_fraction','mass_mm3','centroid_ras_mm',
            'bbox_axis_min_mm','bbox_axis_max_mm','crop_mass_fraction','unsupported_mass_fraction','mass_scope'}
        if (set(record)!=keys or record['version']!=VERSION or not _hash(record['source_hash'])
                or record['track']!='annotation_assisted' or type(record['available']) is not bool
                or record['source_kind']!=('supplied_annotation' if record['available'] else 'unavailable')
                or record['mass_scope']!='covered_supplied_target_membership_no_support_clipping'
                or any(not _hash(record[k]) for k in ('target_hash','domain_hash','support_hash','crop_affine_hash',
                    'local_target_hash','local_domain_hash','local_support_hash'))):
            raise ValueError('Exact permitted public target context required')
        numbers=[record[k] for k in ('domain_coverage_fraction','mass_mm3','crop_mass_fraction','unsupported_mass_fraction')]
        if (not np.isfinite(numbers).all() or record['mass_mm3']<0
                or any(not 0<=record[k]<=1 for k in ('domain_coverage_fraction','crop_mass_fraction','unsupported_mass_fraction'))
                or not np.isfinite(self.committed_overlap_mm3) or not 0<=self.committed_overlap_mm3<=record['mass_mm3']+1e-9
                or not _hash(self.committed_cavity_hash) or not _hash(self.local_cavity_hash)):
            raise ValueError('Public target membership/coverage is invalid')
        affine=np.asarray(record['source_affine_ras_mm'],np.float64)
        origin,shape,source_shape=record['crop_origin'],record['crop_shape'],record['source_shape']
        if (affine.shape!=(4,4) or not np.isfinite(affine).all() or not np.array_equal(affine[3],[0,0,0,1])
                or any(len(v)!=3 for v in (origin,shape,source_shape))
                or any(type(a) is not int or type(n) is not int or type(m) is not int
                    or a<0 or not 1<=n<=64 or a+n>m for a,n,m in zip(origin,shape,source_shape))):
            raise ValueError('Public target source/crop frame is malformed')
        spacing=np.linalg.norm(affine[:3,:3],axis=0)
        if np.any(spacing<=0) or not np.allclose((affine[:3,:3]/spacing).T@(affine[:3,:3]/spacing),np.eye(3),rtol=0,atol=1e-7):
            raise ValueError('Public target frame requires orthogonal source axes')
        crop_affine=affine.copy();crop_affine[:3,3]+=affine[:3,:3]@origin
        if array_digest(crop_affine)!=record['crop_affine_hash']:
            raise ValueError('Public target crop affine differs from its source frame')
        if (not record['available'] and any(record[k]!=0 for k in ('domain_coverage_fraction','mass_mm3','crop_mass_fraction','unsupported_mass_fraction'))
                or record['mass_mm3']>0 and record['domain_coverage_fraction']==0):
            raise ValueError('Unknown target evidence cannot become positive or known-zero coverage')
        if record['mass_mm3']==0:
            if any(record[k] is not None for k in ('centroid_ras_mm','bbox_axis_min_mm','bbox_axis_max_mm')):
                raise ValueError('Zero known target mass cannot supply a fictitious location')
        elif any(np.asarray(record[k]).shape!=(3,) or not np.isfinite(record[k]).all()
                 for k in ('centroid_ras_mm','bbox_axis_min_mm','bbox_axis_max_mm')):
            raise ValueError('Finite physical target location and extent required')
        object.__setattr__(self,'static',record)
        object.__setattr__(self,'_identity',semantic_digest(self.record()))

    def record(self):
        return {'static':self.static,'committed_overlap_mm3':self.committed_overlap_mm3,
            'committed_cavity_hash':self.committed_cavity_hash,'local_cavity_hash':self.local_cavity_hash}

    def assert_intact(self):
        if semantic_digest(self.record())!=self._identity:raise ValueError('Public target context changed')

    @property
    def fingerprint(self):self.assert_intact();return self._identity

    @property
    def static_fingerprint(self):self.assert_intact();return semantic_digest(self.static)

    def require_observation(self,observation):
        self.assert_intact();s=self.static
        if (observation.source_id!=s['source_hash'] or observation.track!=s['track']
                or tuple(observation.image_channels.shape[1:])!=tuple(s['crop_shape'])
                or array_digest(observation.affine_ras_mm)!=s['crop_affine_hash']
                or bool(observation.channel_available[2])!=s['available']
                or array_digest(observation.image_channels[2])!=s['local_target_hash']
                or array_digest(observation.coverage[2])!=s['local_domain_hash']
                or not observation.channel_available[1] or not observation.coverage[1].all()
                or array_digest(observation.image_channels[1])!=s['local_support_hash']
                or array_digest(observation.image_channels[3])!=self.local_cavity_hash):
            raise ValueError('Public target context differs from observation source/crop/cavity')


@dataclass(frozen=True)
class PreparedPublicTarget:
    static: Mapping
    _target: np.ndarray=field(repr=False)
    _capability: object=field(repr=False)
    _identity: tuple=field(init=False,repr=False)

    def __post_init__(self):
        if self._capability is not _PREPARED:raise ValueError('Prepare target from explicit permitted arrays')
        object.__setattr__(self,'static',freeze_json(self.static))
        object.__setattr__(self,'_identity',(semantic_digest(self.static),_layout(self._target)))

    def assert_intact(self):
        if self._identity!=(semantic_digest(self.static),_layout(self._target)):
            raise ValueError('Prepared permitted target changed')

    @property
    def fingerprint(self):self.assert_intact();return self._identity[0]

    def observe(self,committed_cavity):
        self.assert_intact();cavity=np.asarray(committed_cavity)
        if cavity.dtype!=bool or cavity.shape!=self._target.shape:
            raise ValueError('Committed observed cavity must match the full source grid')
        s=self.static;region=tuple(slice(a,a+n) for a,n in zip(s['crop_origin'],s['crop_shape']))
        volume=abs(float(np.linalg.det(np.asarray(s['source_affine_ras_mm'])[:3,:3])))
        overlap=float(self._target[cavity].sum(dtype=np.float64))*volume
        return PublicTargetContext(s,overlap,array_digest(cavity),array_digest(cavity[region].astype(np.float32)))


def prepare_public_target(*,nominal_target,target_domain,observed_support,affine_ras_mm,
        crop_origin,crop_shape,source_hash,track='annotation_assisted',
        source_kind='supplied_annotation',derivation='explicit supplied task region'):
    """No task/reference/reward API: only explicitly permitted source arrays.

None target/domain denotes unavailable evidence. An available all-zero covered
field denotes known-zero membership. Unknown payload is zero before statistics
and identity. Native construction separately refuses positive labels outside its
declared supplied domain; this pure observation helper cannot certify acquisition.
"""
    support=np.asarray(observed_support)
    if (support.dtype!=bool or support.ndim!=3 or not 0<support.size<=MAX_SOURCE_VOXELS
            or any(n<1 for n in support.shape) or not _hash(source_hash)
            or track!='annotation_assisted' or not isinstance(derivation,str) or not 0<len(derivation)<=2048):
        raise ValueError('Bounded supplied public source and derivation required')
    affine=np.asarray(affine_ras_mm,dtype=np.float64)
    if affine.shape!=(4,4) or not np.isfinite(affine).all() or not np.array_equal(affine[3],[0,0,0,1]):
        raise ValueError('Public target requires a finite physical source frame')
    spacing=np.linalg.norm(affine[:3,:3],axis=0)
    if np.any(spacing<=0) or not np.allclose((affine[:3,:3]/spacing).T@(affine[:3,:3]/spacing),np.eye(3),rtol=0,atol=1e-7):
        raise ValueError('Public target source axes must be orthogonal')
    origin,shape=tuple(crop_origin),tuple(crop_shape)
    if (len(origin)!=3 or len(shape)!=3 or any(type(a) is not int or type(n) is not int
            or a<0 or not 1<=n<=64 or a+n>support.shape[i] for i,(a,n) in enumerate(zip(origin,shape)))):
        raise ValueError('Explicit in-source actor crop required')
    available=nominal_target is not None
    if not available:
        if target_domain is not None or source_kind!='unavailable':raise ValueError('Unavailable target cannot have a domain')
        values=np.zeros(support.shape,np.float32);domain=np.zeros(support.shape,bool)
    else:
        raw=np.asarray(nominal_target);domain=np.asarray(target_domain)
        if (source_kind!='supplied_annotation' or raw.shape!=support.shape or raw.dtype.kind not in 'biuf'
                or domain.dtype!=bool or domain.shape!=support.shape):
            raise ValueError('Supplied membership and explicit source-aligned domain required')
        known=raw[domain]
        if not np.isfinite(known).all() or np.any(known<0) or np.any(known>1):
            raise ValueError('Known supplied membership must be finite in [0,1]')
        values=np.where(domain,raw,0).astype(np.float32)
    values=_immutable(values,np.float32);domain=_immutable(domain,bool)
    region=tuple(slice(a,a+n) for a,n in zip(origin,shape))
    voxel=abs(float(np.linalg.det(affine[:3,:3])));total=float(values.sum(dtype=np.float64))
    centroid=lo=hi=None
    if total>0:
        projections=[values.sum(axis=tuple(j for j in range(3) if j!=i),dtype=np.float64) for i in range(3)]
        center=np.array([float(np.dot(np.arange(len(p)),p))/total for p in projections])
        centroid=(affine[:3,:3]@center+affine[:3,3]).tolist()
        bounds=[np.flatnonzero(p>0) for p in projections]
        lo=((np.array([b[0] for b in bounds])-.5)*spacing).tolist()
        hi=((np.array([b[-1] for b in bounds])+.5)*spacing).tolist()
    crop_affine=affine.copy();crop_affine[:3,3]+=affine[:3,:3]@origin
    record={'version':VERSION,'source_hash':source_hash,'track':track,'source_kind':source_kind,'derivation':derivation,
        'target_hash':array_digest(values),'domain_hash':array_digest(domain),'support_hash':array_digest(support),
        'source_affine_ras_mm':affine.tolist(),'source_shape':list(support.shape),'crop_origin':list(origin),'crop_shape':list(shape),
        'crop_affine_hash':array_digest(crop_affine),'local_target_hash':array_digest(values[region]),
        'local_support_hash':array_digest(support[region].astype(np.float32)),
        'local_domain_hash':array_digest(domain[region]),'available':available,
        'domain_coverage_fraction':float(np.count_nonzero(domain))/domain.size,'mass_mm3':total*voxel,
        'centroid_ras_mm':centroid,'bbox_axis_min_mm':lo,'bbox_axis_max_mm':hi,
        'crop_mass_fraction':float(values[region].sum(dtype=np.float64))/total if total else 0.,
        'unsupported_mass_fraction':float(values[~support].sum(dtype=np.float64))/total if total else 0.,
        'mass_scope':'covered_supplied_target_membership_no_support_clipping'}
    return PreparedPublicTarget(record,values,_PREPARED)


def public_target_features(observation,*,reference_mm):
    """Fixed-width global context and public centroid-to-tip relation only."""
    context=observation.public_target_context
    if type(context) is not PublicTargetContext:raise TypeError('Opt-in public target context required')
    context.require_observation(observation);s=context.static
    if isinstance(reference_mm,bool) or not np.isfinite(reference_mm) or reference_mm<=0:
        raise ValueError('Positive physical feature reference required')
    affine=np.asarray(s['source_affine_ras_mm'],np.float64)
    axes=affine[:3,:3]/np.linalg.norm(affine[:3,:3],axis=0)
    access=np.asarray(observation.state_features[3:6],np.float64)
    position=np.zeros(9);relations=np.zeros((len(observation.action_ids),RELATION_WIDTH))
    mass=s['mass_mm3']
    if mass>0:
        center=np.asarray(s['centroid_ras_mm']);access_axis=(access-affine[:3,3])@axes
        position=np.concatenate(((center-access)@axes,
            np.asarray(s['bbox_axis_min_mm'])-access_axis,np.asarray(s['bbox_axis_max_mm'])-access_axis))/reference_mm
        from .goal_relation_spatial_policy import physical_goal_tip_relations
        relations=physical_goal_tip_relations(center,observation.action_geometry[:,4:7],affine,reference_mm=reference_mm)
    relations[0]=0.;relations[~np.asarray(observation.action_mask,dtype=bool)]=0.
    global_values=np.r_[float(s['available']),s['domain_coverage_fraction'],float(mass>0),position,
        np.log1p(mass/reference_mm**3),s['crop_mass_fraction'],s['unsupported_mass_fraction'],
        context.committed_overlap_mm3/mass if mass else 0.]
    if global_values.shape!=(FEATURE_WIDTH,) or not np.isfinite(global_values).all() or not np.isfinite(relations).all():
        raise ValueError('Nonfinite public target features')
    return global_values,relations

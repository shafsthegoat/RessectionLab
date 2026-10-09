"""Generated-only private vessel evaluator; no image loader or patient admission.

The caller owns the private loader. This is accidental-misuse protection, not a
Python sandbox, source-authenticity proof, or a clinical clearance mechanism.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from types import SimpleNamespace
from collections.abc import Mapping
import hashlib
import json
import os
import re
import stat
import time

import numpy as np

from resectionlab.core import array_digest, freeze_json, immutable_array, semantic_digest, thaw_json
from resectionlab.research_estimate_planning import (
    FrozenResearchPlan, ResearchEstimatePlanningSpec, research_strategy_to_record,
    research_strategy_from_record, _replay_strategy_nominal,
)
from resectionlab.evaluation import independent_check_native_history
from resectionlab.functional_events import AxialToolSweep, sweeps_from_native_history
from resectionlab.independent_geometry_batch import segment_box_contact_indices

VERSION = 'generated-private-vascular-evaluator-v1'
ROOT = Path(__file__).resolve().parents[2]
MAX_DOCUMENT_BYTES = 2*1024**2
MAX_REFERENCE_VOXELS = 32**3
MAX_PLANNING_VOXELS = 16**3
MAX_MICROSTEPS = 256


def need(condition, reason):
    if not condition:
        raise ValueError(reason)


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def _hash(value):
    need(isinstance(value,str) and re.fullmatch(r'(sha256:)?[a-f0-9]{64}',value), 'invalid_hash')
    return value.removeprefix('sha256:')


def _json(value):
    return (json.dumps(thaw_json(freeze_json(value)),sort_keys=True,indent=2,allow_nan=False)+'\n').encode()


def _safe_path(path):
    path=Path(path)
    need('..' not in path.parts,'path_traversal_forbidden')
    path=path.absolute()
    need(path.is_relative_to(ROOT/'build'),'ignored_build_output_required')
    need(not any(p.is_symlink() for p in (path,*path.parents)), 'symlink_path_forbidden')
    return path


def _write_new(path, value):
    path=_safe_path(path);raw=_json(value)
    need(len(raw)<=MAX_DOCUMENT_BYTES,'document_budget')
    flags=os.O_WRONLY|os.O_CREAT|os.O_EXCL|getattr(os,'O_NOFOLLOW',0)
    fd=os.open(path,flags,0o600)
    with os.fdopen(fd,'wb') as stream:
        stream.write(raw);stream.flush();os.fsync(stream.fileno())
    _sync_directory(path.parent)
    return digest(raw)


def _sync_directory(path):
    directory=os.open(path,os.O_RDONLY)
    try: os.fsync(directory)
    finally: os.close(directory)


def _read_bound(path, expected):
    path=_safe_path(path)
    fd=os.open(path,os.O_RDONLY|getattr(os,'O_NOFOLLOW',0)|getattr(os,'O_NONBLOCK',0))
    with os.fdopen(fd,'rb') as stream:
        details=os.fstat(stream.fileno())
        need(stat.S_ISREG(details.st_mode) and details.st_size<=MAX_DOCUMENT_BYTES,'seal_missing_or_oversized')
        raw=stream.read(MAX_DOCUMENT_BYTES+1)
    need(len(raw)<=MAX_DOCUMENT_BYTES,'seal_read_budget')
    need(digest(raw)==_hash(expected),'seal_file_changed')
    return json.loads(raw)


def _grid(shape, affine):
    shape=tuple(shape);a=np.asarray(affine,dtype=float)
    need(len(shape)==3 and all(type(n) is int and 1<=n<=64 for n in shape)
         and np.prod(shape)<=MAX_REFERENCE_VOXELS,'reference_grid_budget')
    need(a.shape==(4,4) and np.isfinite(a).all() and np.array_equal(a[3],[0,0,0,1]),'invalid_frame')
    spacing=np.linalg.norm(a[:3,:3],axis=0)
    need(np.all(spacing>=.01) and np.all(spacing<=100)
         and np.allclose((a[:3,:3]/spacing).T@(a[:3,:3]/spacing),np.eye(3),atol=1e-9,rtol=0),'unsupported_shear_or_spacing')
    return semantic_digest({'shape':list(shape),'affine_ras_mm':a.tolist()})


def _rigid(matrix):
    a=np.asarray(matrix,dtype=float)
    need(a.shape==(4,4) and np.isfinite(a).all() and np.array_equal(a[3],[0,0,0,1])
         and np.allclose(a[:3,:3].T@a[:3,:3],np.eye(3),atol=1e-9,rtol=0)
         and np.isclose(np.linalg.det(a[:3,:3]),1.,atol=1e-9,rtol=0)
         and np.max(np.abs(a[:3,3]))<=10000,'unsupported_transform')
    return a


def _public_identity(value, spec):
    need(isinstance(value,Mapping) and set(value)=={'source_domain','dataset','person_id','role',
         'structural_source_sha256','structural_frame_sha256'},'identity_fields')
    value=dict(value)
    need(value['source_domain']=='generated_fixture' and value['dataset']=='generated-vascular-control'
         and isinstance(value['person_id'],str) and re.fullmatch(r'generated:[a-z0-9_-]{1,64}',value['person_id'])
         and value['role']=='GENERATED_DEVELOPMENT','real_person_admission_not_implemented')
    need(type(spec) is ResearchEstimatePlanningSpec and len(spec.sources)==1
         and spec.sources[0].modality==spec.actor_modality=='T1'
         and spec.sources[0].image.size<=MAX_PLANNING_VOXELS and spec.horizon<=4,'bounded_T1_spec_required')
    need(spec.declaration.patient_group.startswith('GENERATED:')
         and value['person_id']=='generated:'+spec.declaration.patient_group.removeprefix('GENERATED:'),'person_declaration_mismatch')
    need(_hash(value['structural_source_sha256'])==_hash(spec.sources[0].source_file_sha256)
         and _hash(value['structural_frame_sha256'])==_hash(spec.sources[0].frame_sha256),'structural_identity_mismatch')
    spec.assert_intact()
    return value


def write_strategy_seal(path, *, plan, spec, planning_identity):
    """Persist one complete nominal strategy; no reference binding enters here."""
    identity=_public_identity(planning_identity,spec)
    need(type(plan) is FrozenResearchPlan,'typed_plan_required')
    strategy=research_strategy_to_record(plan,spec)
    _history_budget(strategy)
    payload={'schema':VERSION,'planning_identity':identity,'strategy':strategy}
    return _write_new(path,{**payload,'content_hash':semantic_digest(payload)})


def _history_budget(strategy):
    history=strategy['physical_history']
    need(1<=len(history)<=4 and sum(len(r.get('microsteps',())) for r in history)<=MAX_MICROSTEPS,'history_budget')


@dataclass(frozen=True,slots=True)
class VascularReferenceBinding:
    """Evaluator-owned expected manifest; never passed to planner or strategy writer."""
    strategy_file_sha256: str
    planning_identity_hash: str
    person_id: str
    role: str
    structural_source_sha256: str
    structural_frame_sha256: str
    mra_source_sha256: str
    annotation_source_sha256: str
    reference_frame_hash: str
    mask_hash: str
    coverage_hash: str
    planning_to_reference_ras_mm: tuple
    transform_provenance_sha256: str
    source_lineage: Mapping
    fingerprint: str=field(init=False)

    def __post_init__(self):
        for name in ('strategy_file_sha256','planning_identity_hash','structural_source_sha256','structural_frame_sha256',
                     'mra_source_sha256','annotation_source_sha256','reference_frame_hash','mask_hash','coverage_hash','transform_provenance_sha256'):
            _hash(getattr(self,name))
        need(isinstance(self.person_id,str) and re.fullmatch(r'generated:[a-z0-9_-]{1,64}',self.person_id)
             and self.role=='GENERATED_DEVELOPMENT','reference_person_scope')
        transform=_rigid(self.planning_to_reference_ras_mm)
        object.__setattr__(self,'planning_to_reference_ras_mm',tuple(tuple(float(v) for v in row) for row in transform))
        lineage=dict(self.source_lineage)
        need(set(lineage)=={'source_domain','kind','method_record_sha256','initializer_model_sha256','review_status',
             'coverage_meaning','transform_direction'},'lineage_fields')
        need(lineage['source_domain']=='generated_fixture' and lineage['kind']=='generated_control'
             and lineage['initializer_model_sha256'] in ([],()) and lineage['review_status']=='generated_not_human_reviewed'
             and lineage['coverage_meaning']=='annotation_domain_not_vessel_completeness'
             and lineage['transform_direction']=='planning_RAS_mm_to_reference_RAS_mm','lineage_scope')
        _hash(lineage['method_record_sha256'])
        object.__setattr__(self,'source_lineage',freeze_json(lineage))
        object.__setattr__(self,'fingerprint',semantic_digest(self.record()))

    def record(self):
        return {name:thaw_json(getattr(self,name)) for name in self.__dataclass_fields__ if name!='fingerprint'}

    def assert_intact(self):
        need(self.fingerprint==semantic_digest(self.record()),'reference_binding_changed')


@dataclass(frozen=True,slots=True)
class VascularReference:
    binding: VascularReferenceBinding
    mask: np.ndarray
    coverage: np.ndarray
    affine_ras_mm: np.ndarray

    def __post_init__(self):
        need(type(self.binding) is VascularReferenceBinding,'reference_binding_type')
        mask,coverage=np.asarray(self.mask),np.asarray(self.coverage)
        need(mask.ndim==3 and mask.dtype==coverage.dtype==np.bool_ and mask.shape==coverage.shape,'reference_arrays')
        need(not np.any(mask & ~coverage),'positive_outside_declared_annotation_domain')
        _grid(mask.shape,self.affine_ras_mm)
        object.__setattr__(self,'mask',immutable_array(mask,bool))
        object.__setattr__(self,'coverage',immutable_array(coverage,bool))
        object.__setattr__(self,'affine_ras_mm',immutable_array(self.affine_ras_mm,np.float64))
        self.assert_intact()

    def assert_intact(self):
        self.binding.assert_intact()
        need(array_digest(self.mask)==self.binding.mask_hash and array_digest(self.coverage)==self.binding.coverage_hash
             and _grid(self.mask.shape,self.affine_ras_mm)==self.binding.reference_frame_hash,'reference_content_binding')


def _preflight(path, seal_sha256, spec, identity, binding, check):
    identity=_public_identity(identity,spec)
    value=_read_bound(path,seal_sha256)
    need(set(value)=={'schema','planning_identity','strategy','content_hash'} and value['schema']==VERSION,'seal_schema')
    need(value['planning_identity']==identity,'seal_identity')
    need(value['content_hash']==semantic_digest({k:v for k,v in value.items() if k!='content_hash'}),'seal_content_hash')
    _history_budget(value['strategy'])
    plan=research_strategy_from_record(value['strategy'],spec)
    check()
    nominal=_replay_strategy_nominal(plan,spec)
    check()
    need(type(binding) is VascularReferenceBinding,'expected_reference_binding_type')
    binding.assert_intact()
    need(_hash(binding.strategy_file_sha256)==_hash(seal_sha256)
         and binding.planning_identity_hash==semantic_digest(identity)
         and binding.person_id==identity['person_id'] and binding.role==identity['role']
         and _hash(binding.structural_source_sha256)==_hash(identity['structural_source_sha256'])
         and _hash(binding.structural_frame_sha256)==_hash(identity['structural_frame_sha256']),'reference_person_or_planning_source_mismatch')
    history=value['strategy']['physical_history']
    case=nominal.case
    source=SimpleNamespace(mri=case.structural_intensity,affine=case._native_affine_ras_mm,frame='RAS+',semantic_hash=case.source_hash)
    certificate=independent_check_native_history(source,case.tools,history,tissue_mask=case.observed_support,
        access=case.access,hard_exclusion=nominal._config.hard_exclusion,geometry_frame='RAS+',distance_backend='batch',cancelled=check)
    need(certificate.feasible,'nominal_geometry_audit_failed')
    # All planning/replay/geometry work completes before private loading. These
    # detached facts are the entire evaluator input; no task crosses the boundary.
    return {'seal':value,'plan_seal_hash':plan.seal_hash,'history':history,'tools':case.tools,
        'planning_affine':np.array(case._native_affine_ras_mm,copy=True),'planning_shape':case.observed_support.shape,
        'geometry_certificate':asdict(certificate),'recording_replay_calls':len(plan.action_ids)*2}


def _capsule_cells(start,end,radius,shape,affine,check):
    spacing=np.linalg.norm(affine[:3,:3],axis=0);rotation=affine[:3,:3]/spacing
    a=rotation.T@(start-affine[:3,3]);b=rotation.T@(end-affine[:3,3])
    outside=bool(np.any(np.minimum(a,b)-radius<-.5*spacing) or np.any(np.maximum(a,b)+radius>(np.array(shape)-.5)*spacing))
    lo=np.maximum(np.floor((np.minimum(a,b)-radius)/spacing-.5).astype(int),0)
    hi=np.minimum(np.ceil((np.maximum(a,b)+radius)/spacing+.5).astype(int),np.array(shape)-1)
    if np.any(hi<lo):return set(),outside
    indices=np.indices(tuple(hi-lo+1)).reshape(3,-1).T+lo
    centres=indices*spacing
    rows=segment_box_contact_indices(a,b,centres-spacing/2,centres+spacing/2,radius,batch_size=256,cancelled=check)
    return {tuple(int(v) for v in row) for row in indices[rows]},outside


def _counts(cells, reference, outside=False):
    keys=np.asarray(sorted(cells),dtype=int).reshape(-1,3)
    index=tuple(keys.T)
    positive=int(reference.mask[index].sum()) if len(keys) else 0
    known=int(reference.coverage[index].sum()) if len(keys) else 0
    unknown=len(keys)-known
    volume=float(abs(np.linalg.det(reference.affine_ras_mm[:3,:3])))
    return {'touched_reference_cells':len(keys),'positive_reference_cells':positive,
        'positive_cell_volume_upper_bound_mm3':positive*volume,'unknown_reference_cells':unknown,
        'unknown_in_grid_cell_volume_mm3':unknown*volume,'outside_reference_fov':outside,
        'annotated_positive_encounter':True if positive else None if unknown or outside else False,
        'annotation_coverage_complete_for_sweep':not unknown and not outside,
        'biological_vessel_free':None,'clinical_injury_probability':None}


def _removed_overlap(context,reference,transform):
    removed={tuple(v) for row in context['history'] for v in row.get('removed_indices_native',())}
    mapping=np.linalg.inv(reference.affine_ras_mm)@transform@context['planning_affine']
    nearest=np.rint(mapping[:3,:3]);offset=np.rint(mapping[:3,3])
    congruent=(np.allclose(mapping[:3,:3],nearest,atol=1e-8,rtol=0)
        and np.allclose(mapping[:3,3],offset,atol=1e-8,rtol=0)
        and np.array_equal(np.abs(nearest).sum(axis=0),[1,1,1])
        and np.array_equal(np.abs(nearest).sum(axis=1),[1,1,1]))
    base={'removed_planning_cells':len(removed),'method':'congruent_discrete_cell_overlap_with_index_roundoff_normalization',
          'index_mapping_normalization_tolerance':1e-8,
          'continuous_tissue_or_vessel_damage_assessed':False}
    if not congruent:
        return {**base,'status':'unsupported_noncongruent_grids','positive_overlap_cells':None,
                'positive_overlap_mm3':None,'unknown_removed_cells':len(removed)}
    mapped={tuple(int(v) for v in nearest@np.array(cell)+offset) for cell in removed}
    inside={cell for cell in mapped if all(0<=v<n for v,n in zip(cell,reference.mask.shape))}
    counted=_counts(inside,reference)
    return {**base,'status':'computed','positive_overlap_cells':counted['positive_reference_cells'],
        'positive_overlap_mm3':counted['positive_cell_volume_upper_bound_mm3'],
        'unknown_removed_cells':counted['unknown_reference_cells']+len(mapped-inside),
        'outside_reference_fov_removed_cells':len(mapped-inside)}


def evaluate_private_vessels(*, seal_path, seal_sha256, spec, planning_identity,
        reference_binding, load_reference, output_directory, wall_seconds=10.):
    """One callback after durable seal validation; complete fixed history only.

    Bounds are in-process array/history/output/checkpoint limits. An arbitrary
    blocking loader requires an external supervisor before real use; this
    generated-only API neither grants payload access nor contains hostile code.
    """
    need(isinstance(wall_seconds,(float,int)) and not isinstance(wall_seconds,bool)
         and 0<wall_seconds<=30 and callable(load_reference),'bounded_evaluation_required')
    started=time.monotonic()
    def check():
        if time.monotonic()-started>=wall_seconds:raise TimeoutError('evaluation_wall_budget')
        return False
    context=_preflight(seal_path,seal_sha256,spec,planning_identity,reference_binding,check)
    check()
    return _evaluate_preflighted_private_vessels(context=context,seal_path=seal_path,
        seal_sha256=seal_sha256,spec=spec,reference_binding=reference_binding,
        expected_binding_hash=reference_binding.fingerprint,load_reference=load_reference,
        output_directory=output_directory,started=started,check=check)


def _evaluate_preflighted_private_vessels(*,context,seal_path,seal_sha256,spec,
        reference_binding,expected_binding_hash,load_reference,output_directory,started,check):
    """Internal scoring half; batch caller must preflight every method first."""
    check();reference_binding.assert_intact();spec.assert_intact()
    need(reference_binding.fingerprint==expected_binding_hash,'reference_binding_changed_after_preflight')
    need(_read_bound(seal_path,seal_sha256)==context['seal'],'seal_changed_after_preflight')
    directory=_safe_path(output_directory)
    directory.mkdir(parents=False,exist_ok=False)
    _sync_directory(directory.parent)
    base={'schema':VERSION,'strategy_file_sha256':_hash(seal_sha256),'plan_seal_hash':context['plan_seal_hash'],
          'reference_binding_hash':reference_binding.fingerprint,'scope':'generated_geometry_and_reference_mask_only',
          'patient_admission':False,'file_access_isolation':False,'private_loader_calls':0,
          'planning_after_private_load':False,'clinical_injury_probability':None,'glioma_target_truth':None}
    _write_new(directory/'attempt.json',{**base,'status':'reserved_before_private_load'})
    expected_binding=expected_binding_hash
    try:
        check();base['private_loader_calls']=1
        reference=load_reference()
        check()
        need(type(reference) is VascularReference,'private_reference_type')
        reference.assert_intact();reference_binding.assert_intact()
        need(reference.binding.fingerprint==expected_binding==reference_binding.fingerprint,'loaded_reference_manifest_mismatch')
        need(_read_bound(seal_path,seal_sha256)==context['seal'],'seal_changed_during_private_load')
        spec.assert_intact()
        transform=_rigid(reference.binding.planning_to_reference_ras_mm)
        parts={'shaft':set(),'tip':set()};outside={'shaft':False,'tip':False};per_action=[]
        for sweep in sweeps_from_native_history(context['history'],context['tools']):
            check()
            mapped=AxialToolSweep(sweep.tool,tuple(transform[:3,:3]@sweep.tip_start_mm+transform[:3,3]),
                tuple(transform[:3,:3]@sweep.tip_end_mm+transform[:3,3]),tuple(transform[:3,:3]@sweep.axis_unit))
            action_cells=set();action_outside=False
            for name,(start,end,radius) in zip(('shaft','tip'),mapped.capsules()):
                cells,out=_capsule_cells(start,end,radius,reference.mask.shape,reference.affine_ras_mm,check)
                parts[name].update(cells);outside[name]|=out;action_cells.update(cells);action_outside|=out
            per_action.append(_counts(action_cells,reference,action_outside))
        union=parts['shaft']|parts['tip']
        report={**base,'status':'evaluated_generated_vascular_reference','source_lineage':thaw_json(reference.binding.source_lineage),
            'full_history_hash':semantic_digest(context['history']),'nominal_geometry':context['geometry_certificate'],
            'action_count':len(context['history']),'nonstop_sweeps':len(per_action),
            'microstep_count':sum(len(r.get('microsteps',())) for r in context['history']),
            'shaft':_counts(parts['shaft'],reference,outside['shaft']),'tip':_counts(parts['tip'],reference,outside['tip']),
            'whole_tool':_counts(union,reference,any(outside.values())),'per_nonstop_action':per_action,
            'removed_overlap':_removed_overlap(context,reference,transform),
            'nominal_replay_transition_calls':context['recording_replay_calls'],
            'contact_tolerance_squared_mm2':1e-10,
            'budget_scope':'Cooperative checkpoints only; existing nominal replay and caller loader are not interruptible. External supervision required for real use.',
            'geometry_scope':'Complete axial insertion plus reverse withdrawal; tool changes outside anatomy. Touched cell volume is an upper-bound surrogate, not continuous intersection volume.',
            'reference_scope':'Annotation domain is not vessel completeness. Unknown coverage and outside-FOV anatomy are not vessel-free.',
            'elapsed_seconds':time.monotonic()-started}
        check();reference.assert_intact();reference_binding.assert_intact()
        need(_read_bound(seal_path,seal_sha256)==context['seal'],'seal_changed_during_evaluation')
    except Exception as error:
        report={**base,'status':'evaluation_failed','error_type':type(error).__name__,
                'reason':'private_load_or_reference_evaluation_failed','outcomes':None,'elapsed_seconds':time.monotonic()-started}
    report=json.loads(_json(report))
    _write_new(directory/'report.json',report)
    return report

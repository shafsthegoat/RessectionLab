"""Bounded read-only first-hit diagnosis; no independent geometry certification."""
import dataclasses
from itertools import product
import numpy as np
from resectionlab.core import semantic_digest,thaw_json
from resectionlab.native_resection import contained_capsule_cells
from resectionlab.post_exposure import preentry_tip

ROUND_OFF_CUSHION_MM=1e-6  # Reporting-only enlarged envelope; never native clearance.


def inspect_rejected(task,ray,*,cell_limit=32):
    """Copy the engine's already-computed rejected interval, never reconstruct it."""
    task._assert_frozen();engine=task._engine;before=engine.state_hash
    if not task._planning:raise ValueError('Public nominal planning task required')
    result=engine.preview_stroke(ray.tool_id,ray.tip_mm,entry_mm=ray.entry_mm,
        obstruction_diagnostics=True,obstruction_cell_limit=cell_limit)
    sidecar=result.obstruction_diagnostic
    record={'action_id':ray.proposal_id,'family':ray.family,'column_index':ray.column_index,
        'tool_id':ray.tool_id,'entry_mm':list(ray.entry_mm),'tip_mm':list(ray.tip_mm),
        'feasible':result.feasible,'reason':result.reason,'diagnostic':None,
        'first_returned_microstep_start_mm':None if not result.microsteps else list(result.microsteps[0].tip_start_mm)}
    if result.reason=='SHAFT_BLOCKED_BY_REMAINING_NATIVE_TISSUE':
        if sidecar is None:raise ValueError('Native first-hit evidence missing')
        data=thaw_json(sidecar);unsigned={k:v for k,v in data.items() if k!='fingerprint'}
        tool=next(t for t in engine.config.tools if t.tool_id==ray.tool_id)
        if (semantic_digest(unsigned)!=data['fingerprint'] or data['source_state_hash']!=before
            or data['source_hash']!=engine.config.source_hash or data['decision_model_hash']!=engine.config.fingerprint
            or data['tool']!=dataclasses.asdict(tool) or data['entry_mm']!=list(ray.entry_mm)
            or data['requested_tip_mm']!=list(ray.tip_mm) or data['failure_tip_mm']!=list(result.failure_tip_mm)
            or len(result.removed_indices_native) or data['prior_temporary_removals_committed'] is not False):
            raise ValueError('Native first-hit binding changed')
        record['diagnostic']=data
    elif sidecar is not None:raise ValueError('First-hit evidence on a different failure')
    task._assert_frozen()
    if engine.state_hash!=before:raise RuntimeError('Diagnostic changed committed state')
    return record


def classify_cells(task,cells,*,guard=lambda:None):
    """Whole-cell capacity over every source-center endpoint on all fixed axes.

    Uses the native containment kernel on the FULL active sweep, including
    pre-entry and spherical caps. This envelope contains every microstep sweep,
    but ignores exposure, shaft clearance, microstep timing, aperture and whether
    a particular endpoint is actually emitted. Passing is optimistic capacity,
    never permission to cut. Only failing every1e-6mm enlarged envelope excludes whole-cell cutting
    only within this frozen source-normal axis/endpoint family.
    """
    task._assert_frozen();case=task.case;proposer=case._nominal_proposer;cfg=task._config
    if (not task._planning or case.proposal_mode!='nominal_cavity_v1'
        or case.proposal_config.obstruction_opening or len(proposer._columns)!=13):
        raise ValueError('Fixed nonrecursive13-axis public family required')
    cells=np.asarray(cells,dtype=np.int64).reshape(-1,3)
    if len(cells)>128 or len(np.unique(cells,axis=0))!=len(cells):raise ValueError('Distinct bounded actual cells required')
    if len(cells) and (np.any(cells<0) or np.any(cells>=cfg.tissue_mask.shape)
        or not np.all(task._engine.remaining_mask[tuple(cells.T)])):raise ValueError('Only actual committed remaining cells may be classified')
    before=task._engine.state_hash;rows=[];calls=0;any_covered=set();any_enlarged=set()
    # Reporting exclusion applies only at bounded millimetre scales. The
    # production admitted orthogonal frame is retained; this is not a new
    # collision tolerance or an arbitrary-affine numerical guarantee.
    grid_corners=np.asarray(tuple(product(*[(-.5,n-.5) for n in cfg.tissue_mask.shape])))
    world_corners=grid_corners@cfg.affine[:3,:3].T+cfg.affine[:3,3]
    dimensions=np.asarray([[t.tip_radius_mm,t.shaft_radius_mm,t.working_length_mm,t.tip_length_mm] for t in cfg.tools])
    scale=float(np.max(np.abs(world_corners))) if np.isfinite(world_corners).all() else float('inf')
    span=float(np.max(np.ptp(world_corners,axis=0))) if np.isfinite(world_corners).all() else float('inf')
    bound=max(scale,span,float(np.max(np.abs(cfg.access.center_mm))))+float(np.max(dimensions))
    applicable=bool(np.isfinite(dimensions).all() and np.isfinite(bound) and bound<=1e4
        and all(t.tip_length_mm>=1. for t in cfg.tools))
    numeric={'applicable':applicable,'coordinate_span_plus_max_tool_dimension_bound_mm':bound if np.isfinite(bound) else None,
        'maximum_allowed_mm':1e4,'minimum_active_tip_length_mm':1.,'enlargement_mm':ROUND_OFF_CUSHION_MM}
    if not applicable:
        return {'scope':'numerical applicability refused; no geometric exclusion',
            'numerical_applicability':numeric,'native_containment_kernel_calls':0,'additional_native_previews':0,'pairs':[],
            'outside_every_fixed_axis_tip_envelope':[],'inside_at_least_one_optimistic_envelope':[],
            'roundoff_ambiguous_cells':cells.tolist(),'tools':[dataclasses.asdict(t) for t in cfg.tools],
            'offsets_source_voxels':[list(v) for v in case.proposal_config.offsets_source_voxels]}

    for col,(offset,column) in enumerate(zip(case.proposal_config.offsets_source_voxels,proposer._columns)):
        for tool in cfg.tools:
            found={};enlarged=set();count=0;zero_length=0
            if all(0<=v<cfg.tissue_mask.shape[d] for d,v in zip(proposer._transverse,column)):
                for axial in range(cfg.tissue_mask.shape[proposer._axis]):
                    guard();voxel=np.zeros(3,np.int64);voxel[proposer._axis]=axial;voxel[list(proposer._transverse)]=column
                    # Identical expression and native affine to provider.propose.
                    tip=cfg.affine[:3,:3]@voxel+cfg.affine[:3,3]
                    depth=float((tip-cfg.access.center_mm)@cfg.access.normal_inward)
                    if depth<=0:continue
                    entry=tip-depth*cfg.access.normal_inward;length=float(np.linalg.norm(tip-entry))
                    if length<1e-9:
                        zero_length+=1;continue  # Native preview refuses this degenerate stroke.
                    axis=(tip-entry)/length
                    start=entry if cfg.post_exposure is None else preentry_tip(tool,entry,axis,cfg.access.normal_inward)
                    contained=contained_capsule_cells(task._engine._cell_scene,cells,
                        start-tool.tip_length_mm*axis,tip,tool.tip_radius_mm)
                    generous=contained_capsule_cells(task._engine._cell_scene,cells,
                        start-tool.tip_length_mm*axis,tip,tool.tip_radius_mm+ROUND_OFF_CUSHION_MM+1e-10)
                    enlarged.update(tuple(int(x) for x in c) for c in generous)
                    calls+=2;count+=1
                    for c in contained:
                        key=tuple(int(x) for x in c)
                        found.setdefault(key,{'endpoint_native':voxel.tolist(),'entry_mm':entry.tolist(),
                            'tip_mm':tip.tolist(),'full_active_start_mm':(start-tool.tip_length_mm*axis).tolist()})
            any_covered.update(found);any_enlarged.update(enlarged)
            rows.append({'column_index':col,'offset_source_voxels':list(offset),'tool_id':tool.tool_id,
                'positive_source_endpoints_checked':count,'native_zero_length_endpoints_refused':zero_length,'cell_capacity':[{'cell':c.tolist(),
                'inside_some_full_active_sweep':tuple(c) in found,
                'inside_enlarged_roundoff_envelope':tuple(c) in enlarged,'first_envelope':found.get(tuple(c))} for c in cells]})
    task._assert_frozen()
    if task._engine.state_hash!=before:raise RuntimeError('Coverage check changed state')
    return {'scope':'optimistic full-sweep containment; no native feasibility or future sequence certificate',
        'numerical_applicability':numeric,'tools':[dataclasses.asdict(t) for t in cfg.tools],
        'offsets_source_voxels':[list(v) for v in case.proposal_config.offsets_source_voxels],
        'axis_set_state_independent':True,'axis_or_tool_changes_allowed':False,
        'whole_cell_inward_tolerance_mm':1e-10,'reporting_roundoff_cushion_mm':ROUND_OFF_CUSHION_MM,'native_containment_kernel_calls':calls,
        'additional_native_previews':0,'pairs':rows,
        'outside_every_fixed_axis_tip_envelope':[c.tolist() for c in cells if tuple(c) not in any_enlarged],
        'roundoff_ambiguous_cells':[c.tolist() for c in cells if tuple(c) in any_enlarged and tuple(c) not in any_covered],
        'inside_at_least_one_optimistic_envelope':[c.tolist() for c in cells if tuple(c) in any_covered]}

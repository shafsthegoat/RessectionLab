"""Versioned public goal-to-candidate-tip features; no task/outcome access."""
from dataclasses import asdict
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from .goal_mode_spatial_policy import GoalModeSpatialPolicy, MODE_ORDER, _mode_candidate_critic_context
from .spatial_policy import sample_ray_features, parameter_hash

POLICY_VERSION='public-goal-relation-spatial-ray-conv-policy-v1'
CHECKPOINT_VERSION='public-goal-relation-spatial-checkpoint-v1'
RELATION_VERSION='public-goal-minus-tip-crop-axis-mm-v1'
RELATION_WIDTH=4
SUMMARY_WIDTH=18


def physical_goal_tip_relations(goal_ras_mm, tips_ras_mm, affine_ras_mm, *, reference_mm):
    """Public point-minus-tip vector in orthogonal crop-axis mm, and distance.

    This pure array API cannot inspect tasks, previews, rewards, private labels or
    candidate identifiers. Joint source/point/pose rotations and translations
    preserve output; changing physical scale without changing reference does not.
    """
    raw=(np.asarray(goal_ras_mm),np.asarray(tips_ras_mm),np.asarray(affine_ras_mm))
    if any(np.iscomplexobj(value) or value.dtype.kind not in 'fiu' for value in raw):
        raise ValueError('Finite real physical relation inputs required')
    goal,tips,affine=(np.asarray(value,dtype=np.float64) for value in raw)
    if (goal.shape!=(3,) or tips.ndim!=2 or tips.shape[1]!=3 or affine.shape!=(4,4)
            or any(not np.isfinite(value).all() for value in (goal,tips,affine))
            or isinstance(reference_mm,bool) or not np.isfinite(reference_mm) or reference_mm<=0
            or not np.allclose(affine[3],(0,0,0,1),rtol=0,atol=1e-12)):
        raise ValueError('Malformed public goal/tip physical relation')
    linear=affine[:3,:3];spacing=np.linalg.norm(linear,axis=0)
    if np.any(spacing<=0):raise ValueError('Nonzero source spacing required')
    axes=linear/spacing
    if not np.allclose(axes.T@axes,np.eye(3),rtol=0,atol=1e-7):
        raise ValueError('Goal-relation v1 requires orthogonal source axes')
    delta=goal[None]-tips
    return np.column_stack((delta@axes,np.linalg.norm(delta,axis=1)))/float(reference_mm)


def public_goal_relations(observation, *, context, reference_mm):
    # The exact source/goal/grid/crop/frame binding is checked before extraction.
    from .public_surface_contact import SurfaceContactObservation,SurfaceContactDevelopmentContext
    if type(observation) is not SurfaceContactObservation or type(context) is not SurfaceContactDevelopmentContext:
        raise TypeError('Exact detached public observation and independently bound context required')
    context.require_observation(observation)
    base=observation.base.base;indices=np.argwhere(observation.public_goal_grid)
    if indices.shape!=(1,3):raise ValueError('Exactly one already-public goal cell required')
    goal=(base.affine_ras_mm@np.r_[indices[0],1.])[:3]
    relation=physical_goal_tip_relations(goal,base.action_geometry[:,4:7],base.affine_ras_mm,reference_mm=reference_mm)
    relation[0]=0.
    # Masked geometry cannot influence STOP or any permitted candidate score.
    relation[~np.asarray(observation.action_mask,dtype=bool)]=0.
    return relation


def legal_relation_summary(relations, mask, modes):
    """Per-mode mean4/max4/log1p(count), independent of candidate ordering."""
    if relations.ndim!=2 or relations.shape[1]!=RELATION_WIDTH or len(mask)!=len(relations) or len(modes)!=len(relations):
        raise ValueError('Relation summary inventory differs')
    pieces=[]
    for mode in ('aspirate','probe'):
        chosen=mask & torch.tensor([m==mode for m in modes],dtype=torch.bool,device=relations.device)
        values=relations[chosen]
        pieces.append(torch.cat((values.mean(0),values.max(0).values,
            relations.new_tensor([np.log1p(len(values))]))) if len(values) else relations.new_zeros(9))
    return torch.cat(pieces)


def _expanded_linear(original, extra):
    # Preserve every old tensor column/bias exactly; every new column is zero.
    replacement=nn.Linear(original.in_features+extra,original.out_features,bias=original.bias is not None,
                          device=original.weight.device,dtype=original.weight.dtype)
    with torch.no_grad():
        replacement.weight.zero_();replacement.weight[:,:original.in_features].copy_(original.weight)
        if original.bias is not None:replacement.bias.copy_(original.bias)
    return replacement


class GoalRelationSpatialPolicy(GoalModeSpatialPolicy):
    def __init__(self,config=None):
        super().__init__(config)
        self.actor[0]=_expanded_linear(self.actor[0],RELATION_WIDTH)
        self.stop[0]=_expanded_linear(self.stop[0],SUMMARY_WIDTH)

    def architecture_record(self):
        record=super().architecture_record()
        record.update(version=POLICY_VERSION,checkpoint_version=CHECKPOINT_VERSION,
            relation_version=RELATION_VERSION,
            action_relation='goal_minus_candidate_tip_crop_axis_mm_and_euclidean_distance_divided_by_physical_reference',
            relation_source='context_bound_public_goal_one_hot_affine_and_current_candidate_tip_poses_only',
            stop_relation_summary='legal_aspirate_then_probe_mean4_max4_log1p_count;empty_group_zero',
            relation_frame='orthogonal_crop_axes;joint_RAS_rigid_transform_invariant;shear_refused',
            initialization='original_shared_tensors_exact;four_actor_and_eighteen_STOP_new_input_columns_zero',
            critic_relation_change='none_IL_only_control',legacy_goal_mode_checkpoint_compatible=False)
        return record

    def checkpoint_identity(self):
        identity=super().checkpoint_identity()
        identity.update(version=CHECKPOINT_VERSION,policy_version=POLICY_VERSION)
        return identity

    def forward(self,observation,*,context):
        volume,grid,inside,geometry,state,spacing,mask=self._inputs(observation,context=context)
        relation=torch.tensor(public_goal_relations(observation,context=context,
            reference_mm=self.config.physical_reference_mm),dtype=volume.dtype,device=volume.device)
        features=self.encoder(volume[None])[0]
        spatial=F.adaptive_avg_pool3d(features[None],(2,2,2)).flatten()
        global_context=torch.cat((spatial,state,spacing))
        rays=sample_ray_features(features,grid)*inside[:,:,None]
        rays=torch.cat((rays,inside[:,:,None]),dim=-1).flatten(1)
        rows=torch.cat((global_context.expand(len(geometry),-1),geometry,rays,relation),dim=-1)
        scores=self.actor(rows).squeeze(-1)
        stop_context=torch.cat((global_context,legal_relation_summary(relation,mask,observation.action_modes)))
        logits=torch.cat((self.stop(stop_context).reshape(1),scores[1:])).masked_fill(~mask,-torch.inf)
        critic_context=(torch.cat((global_context,_mode_candidate_critic_context(geometry,rays,mask)))
                        if self.config.critic_candidate_context else global_context)
        value=self.critic(critic_context).squeeze(-1)
        if not torch.isfinite(logits[mask]).all() or not torch.isfinite(value):
            raise FloatingPointError('Nonfinite public goal-relation policy output')
        return logits,value


def shared_initialization_receipt(baseline, expanded):
    """Exact tensor control, with no forward, checkpoint load or optimizer."""
    if type(baseline) is not GoalModeSpatialPolicy or type(expanded) is not GoalRelationSpatialPolicy:
        raise TypeError('Exact old and expanded initial models required')
    if baseline.config!=expanded.config:raise ValueError('Shared initialization config changed')
    for name,value in baseline.state_dict().items():
        actual=expanded.state_dict()[name]
        if name in ('actor.0.weight','stop.0.weight'):
            if not torch.equal(actual[:,:value.shape[1]],value) or torch.count_nonzero(actual[:,value.shape[1]:]):
                raise ValueError('Shared initial columns or zero added columns changed')
        elif not torch.equal(actual,value):raise ValueError('Shared initial tensor changed: '+name)
    return {'version':'public-goal-relation-shared-initialization-v1','shared_tensors_exact':True,
        'added_input_columns_zero':True,'baseline_parameter_hash':parameter_hash(baseline),
        'expanded_parameter_hash':parameter_hash(expanded),
        'baseline_parameters':sum(p.numel() for p in baseline.parameters()),
        'expanded_parameters':sum(p.numel() for p in expanded.parameters()),
        'added_parameters':sum(p.numel() for p in expanded.parameters())-sum(p.numel() for p in baseline.parameters())}

#!/usr/bin/env python3
"""Bounded local access diagnostics and explicitly approved finite execution.

Diagnostic modes are read-only: they inspect hash-bound native checkpoints and
source geometry without engine actions, tissue removal, or mask reclassification.
Geometric findings do not authorize execution. Only --execute-protocol invokes
public BridgeSession preview/step, and it requires the exact independently
approved protocol, decision, source, and preserved physical checkpoint. Its fixed
action budget and native tool, shaft, material, and task-domain guards still apply.
Earlier execution snapshots and reports retain their original code hashes.
"""
from __future__ import annotations

import argparse
from collections import deque
import heapq
import math
import json
from pathlib import Path
import re
import sys
import time

import nibabel as nib
import numpy as np
from scipy.spatial.transform import Rotation

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO/'src')]
from resectionlab.core import array_digest, semantic_digest
from resectionlab.data_root import checked_root, safe_path, atomic_json, sha256_file
from resectionlab.geometry import GeometryScene, capsule_voxel_indices
from resectionlab.surgical_motion import envelopes, swept_envelopes
from resectionlab.surgical_scene import from_case
from resectionlab.surgical_task_domain import reviewed_domain
from resectionlab.surgical_tools import TOOLS
from resectionlab.workspace_bundle import load_workspace


def distance_to_boxes(point, centers, halfwidth):
    """Exact sphere-center distance to orthogonal source voxel boxes."""
    if not len(centers): return float('inf')
    delta = np.maximum(np.abs(np.asarray(centers)-point)-halfwidth, 0.)
    return float(np.sqrt(np.min(np.sum(delta*delta, axis=1))))


def lattice_neighbors(node, valid):
    for axis in range(3):
        for sign in (-1, 1):
            other = list(node); other[axis] += sign; other = tuple(other)
            if other in valid: yield other


def widest_path(clearances, start, goals):
    """Maximize the minimum node clearance in a finite, declared lattice."""
    if start not in clearances: return [], None
    score = {start: clearances[start]}; previous = {}; queue = [(-score[start], start)]
    while queue:
        negative, node = heapq.heappop(queue)
        if -negative < score[node]: continue
        if node in goals:
            path = [node]
            while node in previous: node = previous[node]; path.append(node)
            return path[::-1], float(-negative)
        for other in lattice_neighbors(node, clearances):
            candidate = min(score[node], clearances[other])
            if candidate > score.get(other, -1.):
                score[other] = candidate; previous[other] = node
                heapq.heappush(queue, (-candidate, other))
    return [], None


def shortest_free_path(clearances, start, goals, radius):
    valid = {node for node, clearance in clearances.items() if clearance > radius+1e-9}
    if start not in valid: return []
    queue = deque([start]); previous = {start: None}
    while queue:
        node = queue.popleft()
        if node in goals:
            path = []
            while node is not None: path.append(node); node = previous[node]
            return path[::-1]
        for other in lattice_neighbors(node, valid):
            if other not in previous: previous[other] = node; queue.append(other)
    return []


def finite_sweeps(tool, params, first, second):
    from scipy.spatial.transform import Slerp
    a,b=np.asarray(first['position']),np.asarray(second['position'])
    rotations=Rotation.from_quat([first['quaternion'],second['quaternion']])
    angle=(rotations[1]*rotations[0].inv()).magnitude()
    count=max(1,int(np.ceil(np.linalg.norm(b-a)/.25)),int(np.ceil(angle/np.deg2rad(1))))
    interpolation=Slerp([0.,1.],rotations);previous=first;result=[]
    for fraction in np.linspace(0,1,count+1)[1:]:
        current={'position':((1-fraction)*a+fraction*b).tolist(),'quaternion':interpolation(fraction).as_quat().tolist()}
        result.extend(swept_envelopes(tool,params,previous,current));previous=current
    return result


def diagnose(replay, actions, *, root=None, name='012-channel-diagnosis-v1'):
    root = checked_root(root, required_bytes=512*1024**2)
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,127}', name): raise ValueError('Plain immutable output token required')
    def contained(value):
        path = Path(value).resolve()
        if not path.is_relative_to(root): raise ValueError('Input outside guarded root')
        return safe_path(root, path.relative_to(root))
    replay, actions = contained(replay), contained(actions)
    case, artifacts, _ = load_workspace(replay)
    if case.frame!='RAS+': raise ValueError('Exact RAS physical source frame required')
    if case.metadata['data_case']['family_id'] != 'TopCoW:012': raise ValueError('This bounded study is CTA012 only')
    record = artifacts['surgical_toolbox']; snapshot = record['branches'][record['branchId']]
    state = snapshot['checkpoint']['state']
    if snapshot['checkpoint']['stateId'] != snapshot['stateId']: raise ValueError('Checkpoint identity differs')
    if state.get('holds') or any(np.linalg.norm(v)>0 for v in state.get('deformation', {}).values()):
        raise ValueError('This static contact study requires undeformed, released source geometry')
    if state.get('toolMechanicsRevision') != 2: raise ValueError('Unchanged revised CUSA geometry required')
    if TOOLS['cusa']['radiusMm']!=1.5 or TOOLS['cusa']['shaftRadiusMm']!=.5 or state['slots']['right']['parameters'].get('radiusMm',1.5)!=1.5:
        raise ValueError('Stock CUSA dimensions must remain unchanged')
    scene = from_case(case); domain = reviewed_domain(scene)
    if domain is None: raise ValueError('Persisted reviewed task domain required')
    if not np.allclose(case.affine[:3, :3], np.diag(np.diag(case.affine[:3, :3]))):
        raise ValueError('Exact local voxel-box distances require this native orthogonal axial grid')
    rows = json.loads(actions.read_text())
    partial = [row for row in rows if row['label']=='genuine-intracranial-target-approach'][-1]
    slot = partial['slotsAfter']['right']; origin = np.asarray(slot['position']); quaternion = slot['quaternion']
    rotation = Rotation.from_quat(quaternion); inward = rotation.apply([0., 0., 1.]); u, v = rotation.apply([1.,0.,0.]), rotation.apply([0.,1.,0.])
    directory = safe_path(root, 'integrated-v1/access/'+name)
    if directory.exists(): raise ValueError('Preserve existing analysis; choose a new output name')
    directory.mkdir(parents=True)
    specification = {'schema':'finite-local-channel-study-v1', 'replay_path':str(replay.relative_to(root)),
        'replay_sha256':sha256_file(replay), 'recorded_actions_sha256':sha256_file(actions),
        'checkpoint_state_id':snapshot['stateId'], 'physical_state_digest':semantic_digest(state),
        'tip_radius_mm':1.5, 'shaft_radius_mm':.5, 'shaft_length_mm':40.,
        'fixed_quaternion':quaternion, 'last_reached_tip_ras_mm':origin.tolist(),
        'inward_unit':inward.tolist(), 'tangent_u':u.tolist(), 'tangent_v':v.tolist(),
        'grid_step_mm':.25, 'inward_depth_range_mm':[-1.,3.], 'tangential_radius_mm':1.,
        'rotation_search':False, 'source_or_state_mutations':False,
        'barrier_refinement':{'plane_selection':'minimum maximum sampled plane clearance','tangent_step_mm':.0625,
            'continuous_cover_bound':'sqrt(2)*step, by rounding each tangent coordinate toward zero inside the unit disk'},
        'stock_kerf_inspection':{'instrument':'craniotome','catalog_radius_mm':1.,'shaft_radius_mm':.5,
            'working_geometry':'both stock 0.4mm crossbar capsules, default depth2mm; never a radius1 sphere',
            'inward_depths_mm':[0.,.25,.5,.75,1.,1.25,1.5],
            'tangent_offsets_mm':[[0.,0.],[.5,0.],[-.5,0.],[0.,.5],[0.,-.5]],
            'roll_angles_deg':[0.,45.,90.,135.],'effects_or_actions_executed':False},
        'scope':'One fixed-axis local steering family through the already drilled channel. No extra drilling, target relocation, smaller tool, reset, or native guard change. No global approach infeasibility or clinical safety claim.'}
    atomic_json(directory/'prospective-geometry-study.json', specification)
    labels = scene.labels.copy().ravel(); removed = np.asarray(state['removed'], int); labels[removed] = 0
    bone_ids = [row['id'] for row in scene.regions if row['material']=='bone']
    vessel_ids = [row['id'] for row in scene.regions if row['material']=='vessel']
    hard = np.isin(labels, bone_ids+vessel_ids).reshape(scene.shape)
    depths = np.arange(-1.,3.001,.25); lateral = np.arange(-1.,1.001,.25)
    points = {(k,i,j):origin+depth*inward+du*u+dv*v for k,depth in enumerate(depths)
        for i,du in enumerate(lateral) for j,dv in enumerate(lateral) if du*du+dv*dv<=1.+1e-12}
    world = np.asarray(list(points.values())); inverse = np.linalg.inv(case.affine)
    corners = np.array([world.min(0)-3.,world.max(0)+3.])@inverse[:3,:3].T+inverse[:3,3]
    low = np.maximum(0,np.floor(corners.min(0)-1).astype(int)); high = np.minimum(np.asarray(scene.shape)-1,np.ceil(corners.max(0)+1).astype(int))
    roi = tuple(slice(a,b+1) for a,b in zip(low,high)); local = np.argwhere(hard[roi])+low
    centers = case.voxel_to_world(local); halfwidth = np.asarray(case.spacing_mm)/2
    clearance = {node: min(3.,distance_to_boxes(point,centers,halfwidth)) for node,point in points.items()}
    start = (4,4,4); goals = {node for node in points if node[0]==len(depths)-1}
    path = shortest_free_path(clearance,start,goals,1.5); widest, bottleneck = widest_path(clearance,start,goals)
    plane_maxima=[max(value for node,value in clearance.items() if node[0]==k) for k in range(len(depths))]
    plane_index=int(np.argmin(plane_maxima));fine_step=.0625;fine_values=[]
    for du in np.arange(-1,1.00001,fine_step):
        for dv in np.arange(-1,1.00001,fine_step):
            if du*du+dv*dv<=1.+1e-12:
                point=origin+depths[plane_index]*inward+du*u+dv*v
                fine_values.append(min(3.,distance_to_boxes(point,centers,halfwidth)))
    cover_bound=float(np.sqrt(2)*fine_step);upper=float(max(fine_values)+cover_bound)
    barrier={'inward_depth_mm':float(depths[plane_index]),'fine_disk_sample_count':len(fine_values),
        'maximum_sampled_distance_mm':float(max(fine_values)),'disk_cover_radius_bound_mm':cover_bound,
        'continuous_disk_clearance_upper_bound_mm':upper,'fixed_CUSA_radius_mm':1.5,
        'blocks_entire_continuous_radius1mm_disk':upper<1.5-1e-9,
        'proof':'Distance to retained closed voxel boxes is 1-Lipschitz. Rounding both tangent coordinates toward zero preserves the radius1mm disk and moves at most sqrt(2)*step. A bound below the unchanged tip radius blocks this cross-section regardless of shaft orientation; no global approach claim.'}
    grid = GeometryScene(np.zeros(scene.shape,bool),case.affine)
    def contacts(cover):
        hits = {'shaft':set(),'working':set()}
        for a,b,radius,part in cover:
            ijk = capsule_voxel_indices(grid,a,b,radius)
            if len(ijk): hits[part].update(np.ravel_multi_index(ijk.T,scene.shape).tolist())
        return {part:{row['name']:int(np.count_nonzero(labels[list(cells)]==row['id'])) for row in scene.regions}
            for part,cells in hits.items()}
    def contact_cells(cover):
        result={'shaft':set(),'working':set()}
        for aa,bb,radius,part in cover:
            ijk=capsule_voxel_indices(grid,aa,bb,radius)
            if len(ijk):result[part].update(np.ravel_multi_index(ijk.T,scene.shape).tolist())
        return result
    def voxel_rows(flat):
        ids=np.asarray(sorted(set(flat)),int)
        if not len(ids):return []
        ijk=np.array(np.unravel_index(ids,scene.shape)).T;xyz=case.voxel_to_world(ijk)
        return [{'cell_id':int(cell),'ijk':index.tolist(),'ras_mm':point.tolist(),
                 'owned_label':int(labels[cell]),'source_CT_scaled_intensity':float(case.mri[tuple(index)]),
                 'source_brain_mask':bool(case.compartments['brain'][tuple(index)]),
                 'source_skull_mask':bool(case.compartments['estimated_skull'][tuple(index)])}
            for cell,index,point in zip(ids,ijk,xyz)]
    def pose(point): return {'position':np.asarray(point).tolist(),'quaternion':quaternion}
    params = {'radiusMm':1.5,'depthMm':2.,'contactGeometryRevision':2}
    endpoint = envelopes(TOOLS['cusa'],params,origin,quaternion)
    contact_rows = [{'kind':'last_reached_actual_pose','position':origin.tolist(),'contacts':contacts(endpoint)}]
    raw_sweep=None
    for event in partial['outcome'].get('events',[]):
        if event.get('kind')=='contact_stop':
            point = np.asarray(event['position']); contact_rows.append({'kind':'recorded_requested_stop_pose_not_assumed_achieved',
                'position':point.tolist(),'contacts':contacts(envelopes(TOOLS['cusa'],params,point,quaternion))})
            attempted=swept_envelopes(TOOLS['cusa'],params,pose(origin),pose(point));attempted_cells=contact_cells(attempted)
            raw_sweep={'kind':'unclamped_requested_sweep_bound_not_reconstructed_physical_microstep',
                'requested_delta_mm':float(np.linalg.norm(point-origin)), 'contacts':contacts(attempted),
                'retained_skull_voxels':{part:voxel_rows([cell for cell in ids if labels[cell] in bone_ids]) for part,ids in attempted_cells.items()},
                'caution':'The recorded requested point can run ahead of the CUSA contact-limited pose. This is a requested whole-motion bound, not a claim that the physical tool achieved that sweep.'}
    forward=next((k for k in range(5,len(depths)) if clearance[(k,4,4)]<=1.5+1e-9),None)
    first_forward=None
    if forward is not None:
        point=points[(forward,4,4)];ids=contact_cells(envelopes(TOOLS['cusa'],params,point,quaternion))['working']
        first_forward={'inward_depth_mm':float(depths[forward]),'position_ras_mm':point.tolist(),
            'retained_skull_voxels':voxel_rows([cell for cell in ids if labels[cell] in bone_ids]),
            'interpretation':'First sampled nominal-forward static obstruction, not the reconstructed failed native microstep.'}
    atomic_json(directory/'source-contact-voxel-inventory.json',{'first_nominal_forward':first_forward,'requested_sweep_bound':raw_sweep})
    kerf_candidates=[];kerf_params={'radiusMm':TOOLS['craniotome']['radiusMm'],'depthMm':2.}
    if kerf_params['radiusMm']!=1. or TOOLS['craniotome']['shaftRadiusMm']!=.5:raise ValueError('Stock kerf dimensions differ')
    blockers=set(row['cell_id'] for row in (first_forward or {}).get('retained_skull_voxels',[]))
    shaft_cache={}
    for depth in specification['stock_kerf_inspection']['inward_depths_mm']:
        for du,dv in specification['stock_kerf_inspection']['tangent_offsets_mm']:
            point=origin+depth*inward+du*u+dv*v;key=(depth,du,dv)
            for roll in specification['stock_kerf_inspection']['roll_angles_deg']:
                orientation=rotation*Rotation.from_euler('z',roll,degrees=True);q=orientation.as_quat().tolist()
                cover=envelopes(TOOLS['craniotome'],kerf_params,point,q)
                work=contact_cells([segment for segment in cover if segment[3]=='working'])['working']
                bone=[cell for cell in work if labels[cell] in bone_ids]
                unsupported=[cell for cell in work if labels[cell]!=0 and labels[cell] not in bone_ids]
                if not bone or unsupported:continue
                if key not in shaft_cache:shaft_cache[key]=contact_cells([segment for segment in cover if segment[3]=='shaft'])['shaft']
                if any(labels[cell]!=0 for cell in shaft_cache[key]):continue
                xyz=case.voxel_to_world(np.array(np.unravel_index(np.asarray(bone),scene.shape)).T)
                local=orientation.inv().apply(xyz-point)
                eligible=[cell for cell,xyz in zip(bone,local) if abs(xyz[2])<=2. and abs(xyz[1])<=min(case.spacing_mm)*.5]
                if not eligible:continue
                kerf_candidates.append({'inward_depth_mm':depth,'offset_u_mm':du,'offset_v_mm':dv,'roll_deg':roll,
                    'pose':{'position':point.tolist(),'quaternion':q},'working_retained_bone_cells':len(bone),
                    'working_unsupported_material_cells':0,'shaft_retained_material_cells':0,
                    'potential_end_pose_kerf_cells':voxel_rows(eligible),'first_nominal_blocker_overlap':len(set(eligible)&blockers),
                    'potential_end_pose_kerf_mm3':len(eligible)*case.voxel_volume_mm3,
                    'effect_prediction_scope':'Static end-pose eligibility using existing kerf local-z and local-y rules; no cells are removed or reclassified here.'})
    kerf_candidates.sort(key=lambda row:(-row['first_nominal_blocker_overlap'],row['inward_depth_mm'],
        row['offset_u_mm']**2+row['offset_v_mm']**2,-row['potential_end_pose_kerf_mm3']))
    external_pose=state['slots']['right'];qualified=[]
    for candidate in kerf_candidates[:8]:
        q=candidate['pose']['quaternion'];first={'position':external_pose['position'],'quaternion':q}
        middle={'position':origin.tolist(),'quaternion':q}
        cover=finite_sweeps(TOOLS['craniotome'],kerf_params,first,middle)+finite_sweeps(TOOLS['craniotome'],kerf_params,middle,candidate['pose'])
        counts=contacts(cover);inside=True
        for aa,bb,radius,part in cover:
            ends=case.world_to_voxel(np.asarray([aa,bb]));extent=np.linalg.norm(inverse[:3,:3],axis=1)*radius
            inside &= bool(np.all(ends.min(0)-extent>=domain.minimum) and np.all(ends.max(0)+extent<=domain.maximum))
        names={row['name'] for row in scene.regions if row['material']!='bone'}
        candidate['approach_swept_contacts']=counts;candidate['approach_inside_reviewed_box']=inside
        candidate['static_approach_qualified']=inside and all(counts['working'][name]==0 for name in names) and all(value==0 for value in counts['shaft'].values())
        if candidate['static_approach_qualified']:qualified.append(candidate)
    edge_rows=[]
    for first,second in zip(path,path[1:]):
        cover=swept_envelopes(TOOLS['cusa'],params,pose(points[first]),pose(points[second]))
        counts=contacts(cover)
        inside=True
        for a,b,radius,part in cover:
            ends=case.world_to_voxel(np.asarray([a,b])); extent=np.linalg.norm(inverse[:3,:3],axis=1)*radius
            inside &= bool(np.all(ends.min(0)-extent>=domain.minimum) and np.all(ends.max(0)+extent<=domain.maximum))
        edge_rows.append({'from_node':list(first),'to_node':list(second),'contacts':counts,'inside_reviewed_box':inside})
    hard_names=[row['name'] for row in scene.regions if row['material'] in ('bone','vessel')]
    full_hard_clear=bool(path) and all(row['inside_reviewed_box'] and
        all(row['contacts'][part][name]==0 for part in ('shaft','working') for name in hard_names) for row in edge_rows)
    result={'schema':'read-only-local-channel-diagnosis-v1','specification':specification,
        'source_labels_hash':array_digest(scene.labels),'executed_removed_cells':len(removed),
        'material_contact_diagnostics':contact_rows,'candidate_center_count':len(points),
        'free_tip_centers':sum(value>1.5+1e-9 for value in clearance.values()),
        'greatest_minimum_sampled_center_clearance_mm':bottleneck,
        'continuous_cross_section_barrier':barrier,
        'source_contact_inventory':'source-contact-voxel-inventory.json',
        'static_stock_kerf_candidate_count':len(kerf_candidates),'inspected_top_stock_kerf_candidates':kerf_candidates[:8],
        'qualified_stock_kerf_approach_count':len(qualified),
        'shortest_fixed_axis_tip_path_nodes':[list(node) for node in path],
        'widest_center_path_nodes':[list(node) for node in widest],
        'selected_path_edge_full_tool_checks':edge_rows,'selected_path_full_tool_hard_geometry_clear':full_hard_clear,
        'sampled_original_centerline_clearance_mm':[{'depth_mm':float(depths[k]),'clearance_mm':clearance[(k,4,4)]} for k in range(len(depths))],
        'working_tip_path_ras_mm':[points[node].tolist() for node in path],
        'interpretation':'Lattice-node clearance is a local geometric diagnostic, not proof over all continuous configurations. Full swept checks may reject the selected path. Retained shaft brain contacts are reported and require actual ordinary CUSA dynamics; none are erased here.',
        'new_actions_executed':0,'source_state_changed':False}
    atomic_json(directory/'report.json',result)
    protocol={'schema':'prospective-stock-kerf-local-CUSA-protocol-v1','status':'pending_independent_review_no_execution' if qualified else 'withheld_no_qualified_stock_geometry',
        'source_replay_sha256':sha256_file(replay),'initial_state_id':snapshot['stateId'],'physical_prefix_digest':semantic_digest(state),
        'source_arrays_or_obstacles_changed':False,'extra_burr_drilling':False,'target_relocated':False,
        'candidate':qualified[0] if qualified else None,
        'scope':'One stock kerf approach followed by the original stock CUSA local passage attempt only; source anatomy/anchors are not invented. No flap manipulation or extraction. Existing physical prefix, thermal burden, and all guards remain authoritative.',
        'steps':(['Resume exact guarded checkpoint without reset','Exchange withdrawn right-hand CUSA to stock craniotome',
            'Place at recorded external pose with the candidate roll; no material contact during placement',
            'Public preview one active stroke through the recorded last-reached pose to the candidate pose; commit only after independent approval and positive bone-only removal with zero brain/vessel/target injury/removal',
            'Checked withdrawal along the same approach, then withdrawn catalog exchange back to CUSA',
            'Public preview the original CUSA centerline to 3mm beyond the old last-reached point, at most eight 2-second actions with all ordinary guards; retain refusals',
            'Checked withdrawal to the original external pose']) if qualified else []}
    atomic_json(directory/'prospective-stock-kerf-protocol.json',protocol)
    _render(case,scene.labels,labels.reshape(scene.shape),origin,inward,u,path,points,directory/'source-channel-contact.png')
    return {'report_path':str(directory/'report.json'),'selected_path_found':bool(path),
        'full_tool_hard_geometry_clear':full_hard_clear,'bottleneck_mm':bottleneck,
        'continuous_barrier':barrier,'kerf_candidates':len(kerf_candidates),'qualified_kerf_approaches':len(qualified),
        'actual_contacts':contact_rows,'path_pose_count':len(path)}


def _render(case,source_labels,retained,origin,inward,side,path,points,destination):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from scipy.ndimage import map_coordinates
    horizontal,vertical=np.linspace(-4,4,161),np.linspace(-5,6,221);aa,bb=np.meshgrid(horizontal,vertical)
    world=origin+aa[:,:,None]*side+bb[:,:,None]*inward; coords=np.moveaxis(case.world_to_voxel(world),-1,0)
    image=map_coordinates(case.mri,coords,order=1,mode='constant',cval=np.nan)
    fig,axes=plt.subplots(1,3,figsize=(13,7));extent=[-4,4,-5,6]
    for ax in axes:ax.imshow(image,cmap='gray',origin='lower',extent=extent,vmin=-10,vmax=100)
    for ax,array in zip(axes[1:],[source_labels,retained]):
        for label,color in [(1,'lime'),(3,'cyan'),(4,'red'),(5,'red')]:
            plane=map_coordinates((array==label).astype(np.uint8),coords,order=0,mode='constant',cval=0)
            if plane.any() and not plane.all():ax.contour(plane,levels=[.5],colors=[color],linewidths=.8,extent=extent)
    for ax in axes:
        ax.add_patch(plt.Circle((0,0),1.5,fill=False,edgecolor='yellow'));ax.axhline(0,color='yellow',linewidth=.5)
        ax.set_xlabel('mm along one tool-tangent axis');ax.set_ylabel('mm inward from actual last reached tip')
    if path:
        delta=np.asarray([points[node]-origin for node in path]);axes[2].plot(delta@side,delta@inward,color='magenta',linewidth=1)
    for ax,title in zip(axes,['Unchanged baseline CTA','Original source mask ownership','Actual retained ownership after recorded actions']):ax.set_title(title,fontsize=9)
    fig.suptitle('CTA012 local residual port: yellow = unchanged 1.5 mm CUSA tip at actual last reached pose\nCyan estimated skull; green estimated brain; red supplied vessels. Magenta projection is a read-only candidate, not an executed path.\nDisplay reslice only; original source arrays, obstacles and saved physical state remain unchanged.',fontsize=9)
    fig.tight_layout(rect=(0,0,1,.91));fig.savefig(destination,dpi=140);plt.close(fig)


def make_protocol(diagnosis, *, root=None, name='012-stock-kerf-protocol-v1'):
    root=checked_root(root,required_bytes=256*1024**2)
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,127}',name):raise ValueError('Plain immutable token required')
    diagnosis=Path(diagnosis).resolve()
    if not diagnosis.is_relative_to(root):raise ValueError('Diagnosis outside guarded root')
    report=json.loads((diagnosis/'report.json').read_text());draft=json.loads((diagnosis/'prospective-stock-kerf-protocol.json').read_text())
    candidate=draft['candidate']
    if candidate is None or not candidate['static_approach_qualified']:raise ValueError('No qualified stock geometry candidate')
    specification=report['specification'];replay=safe_path(root,specification['replay_path'])
    if sha256_file(replay)!=specification['replay_sha256']:raise ValueError('Exact guarded source changed')
    case,artifacts,_=load_workspace(replay);record=artifacts['surgical_toolbox'];snapshot=record['branches'][record['branchId']]
    state=snapshot['checkpoint']['state']
    if snapshot['stateId']!=specification['checkpoint_state_id'] or semantic_digest(state)!=specification['physical_state_digest']:
        raise ValueError('Source physical prefix changed')
    scene=from_case(case);domain=reviewed_domain(scene);grid=GeometryScene(np.zeros(scene.shape,bool),case.affine)
    labels=scene.labels.copy().ravel();labels[np.asarray(state['removed'],int)]=0
    external=state['slots']['right'];q=candidate['pose']['quaternion'];original_q=specification['fixed_quaternion']
    origin=np.asarray(specification['last_reached_tip_ras_mm']);goal=origin+3*np.asarray(specification['inward_unit'])
    def pose(point,quaternion):return {'position':list(point),'quaternion':quaternion}
    kerf_external=pose(external['position'],q);kerf_middle=pose(origin,q);kerf_end=candidate['pose']
    cusa_external=pose(external['position'],original_q);cusa_middle=pose(origin,original_q);cusa_goal=pose(goal,original_q)
    def action(label,operation,**kw):return {'label':label,'action':{'slot':'right','operation':operation,**kw}}
    actions=[action('withdrawn-stock-exchange-to-craniotome','exchange',instrument='craniotome',duration=.1),
        action('external-rolled-stock-kerf-placement','place',pose=kerf_external,duration=.1),
        {**action('single-reviewed-stock-kerf','apply',path=[kerf_middle,kerf_end],duration=1.),'commit_rule':'positive_authorized_bone_only'},
        action('checked-kerf-withdrawal','withdraw',path=[kerf_middle,kerf_external],duration=1.),
        action('withdrawn-stock-exchange-to-CUSA','exchange',instrument='cusa',duration=.1),
        action('external-original-CUSA-placement','place',pose=cusa_external,duration=.1),
        {**action('original-CUSA-cleared-channel-reentry','apply',path=[cusa_middle],duration=1.),'requires_positive_kerf':True},
        {**action('original-CUSA-local-passage','apply',path=[cusa_goal],duration=2.),'repeat':8,
            'requires_positive_kerf':True,'stop_when':'local_goal_or_guard_or_budget'},
        action('checked-final-CUSA-withdrawal','withdraw',path=[cusa_external],duration=1.)]
    directory=safe_path(root,'integrated-v1/access/'+name)
    if directory.exists():raise ValueError('Preserve existing prospective protocol')
    directory.mkdir(parents=True)
    params={'radiusMm':1.,'depthMm':2.,'contactGeometryRevision':2}
    covers={'kerf_approach':finite_sweeps(TOOLS['craniotome'],params,kerf_external,kerf_middle)+finite_sweeps(TOOLS['craniotome'],params,kerf_middle,kerf_end),
        'kerf_withdrawal':finite_sweeps(TOOLS['craniotome'],params,kerf_end,kerf_middle)+finite_sweeps(TOOLS['craniotome'],params,kerf_middle,kerf_external),
        'external_roll_bound_not_an_in_tissue_motion':finite_sweeps(TOOLS['craniotome'],params,pose(external['position'],original_q),kerf_external)}
    geometry={};inverse=np.linalg.inv(case.affine)
    for key,cover in covers.items():
        inside=True;hit={'shaft':set(),'working':set()}
        for aa,bb,radius,part in cover:
            ijk=capsule_voxel_indices(grid,aa,bb,radius)
            if len(ijk):hit[part].update(np.ravel_multi_index(ijk.T,scene.shape).tolist())
            points=case.world_to_voxel(np.array([aa,bb]));extent=np.linalg.norm(inverse[:3,:3],axis=1)*radius
            inside &=bool(np.all(points.min(0)-extent>=domain.minimum) and np.all(points.max(0)+extent<=domain.maximum))
        geometry[key]={'inside_reviewed_box':inside,'capsules':[{'a':aa.tolist(),'b':bb.tolist(),'radiusMm':float(radius),'part':part}for aa,bb,radius,part in cover],
            'retained_material_cells':{part:{row['name']:int(np.count_nonzero(labels[list(ids)]==row['id']))for row in scene.regions}for part,ids in hit.items()}}
    atomic_json(directory/'full-tool-swept-geometry.json',geometry)
    thermal=safe_path(root,'integrated-v1/probes/012-conditional-centerline-execution-v1/report.json')
    old=json.loads(thermal.read_text())
    expected_bone=[row['cell_id']for row in candidate['potential_end_pose_kerf_cells']]
    protocol={'schema':'prospective-public-stock-kerf-local-CUSA-v1','status':'pending_independent_review_do_not_execute',
        'source_path':str(replay.relative_to(root)),'source_sha256':sha256_file(replay),'initial_state_id':snapshot['stateId'],
        'full_physical_prefix_digest':semantic_digest(state),'original_action_count':len(snapshot['actions']),
        'original_actions_digest':semantic_digest(snapshot['actions']),'resume_existing_checkpoint':True,'reset_forbidden':True,
        'thermal_prefix':{'path':str(thermal.relative_to(root)),'sha256':sha256_file(thermal),
            'historical_max_recorded_spatial_temperature_C':old['trajectory_max_recorded_spatial_temperature_C']},
        'diagnosis_sha256':sha256_file(diagnosis/'report.json'),'selected_geometry_candidate':candidate,
        'resolved_stock_parameters':{'craniotome':{'rpm':40000.,'forceN':1.,'duty':1.,'radiusMm':1.,'depthMm':2.,'flowMm3PerSecond':20.,'power':1.,'jawOpeningMm':2.,'contactGeometryRevision':2},
            'cusa':{'rpm':40000.,'forceN':1.,'duty':1.,'radiusMm':1.5,'depthMm':2.,'flowMm3PerSecond':20.,'power':1.,'jawOpeningMm':2.,'contactGeometryRevision':2}},
        'tool_dimensions_changed':False,'geometry_file':'full-tool-swept-geometry.json','geometry_sha256':sha256_file(directory/'full-tool-swept-geometry.json'),
        'actions':actions,'budgets':{'maximum_new_step_calls':16,'maximum_new_preview_calls':16,'maximum_new_native_calls':32,
            'maximum_kerf_apply_commits':1,'maximum_CUSA_local_apply_commits':8,'maximum_declared_action_duration_seconds':20.4,
            'maximum_new_bone_mm3':candidate['potential_end_pose_kerf_mm3'],'maximum_new_brain_mm3':34.,'maximum_new_target_mm3':0.,'maximum_new_vessel_mm3':0.},
        'authorized_kerf_source_cells':expected_bone,
        'runtime_predicates':{'every_step_requires_public_preview':True,'kerf_commit_requires':[
            'accepted public preview; no material/shaft/domain/tool refusal','positive bone removal, subset of authorized source-cell IDs',
            'zero new brain/vessel/target removal or vessel encounter; bone delta within frozen limit',
            'unchanged source labels and original prefix; native expectedStateId must match'],
            'CUSA_commit_requires':['accepted or TISSUE_ASPIRATION_IN_PROGRESS public preview only','zero new bone/vessel/target removal and no new vessel encounter',
                'new brain removal <=34mm3 across this finite local test','all native/domain/full-tool guards unchanged'],
            'stop':['withhold any failing physical action and retain its preview','stop local attempts at <=.05mm goal distance or first physical refusal or budget exhaustion',
                'always attempt approved checked withdrawal/withdrawn instrument cleanup; do not replace failed paths'],
            'failed_kerf_skips_CUSA_material_actions':True},
        'clinical_interpretation':'None. A finite conditional geometry test with uncalibrated mechanics; all original modeled thermal/removal/time costs remain. No source anatomy edit, extra drilling, new anchors, flap manipulation, target relocation or successful passage claim.'}
    atomic_json(directory/'prospective-public-protocol.json',protocol)
    _render_kerf(case,scene.labels,labels.reshape(scene.shape),candidate,directory/'stock-kerf-full-tool-source.png')
    return {'protocol_path':str(directory/'prospective-public-protocol.json'),'protocol_sha256':sha256_file(directory/'prospective-public-protocol.json'),
        'geometry_sha256':sha256_file(directory/'full-tool-swept-geometry.json'),'source_overlay_sha256':sha256_file(directory/'stock-kerf-full-tool-source.png')}


def _render_kerf(case,original,retained,candidate,destination):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon,Circle
    from scipy.ndimage import map_coordinates
    point=np.asarray(candidate['pose']['position']);rotation=Rotation.from_quat(candidate['pose']['quaternion'])
    xaxis=rotation.apply([1.,0.,0.]);zaxis=rotation.apply([0.,0.,1.])
    covers=envelopes(TOOLS['craniotome'],{'radiusMm':1.,'depthMm':2.},point,candidate['pose']['quaternion'])
    fig,axes=plt.subplots(1,3,figsize=(12,10))
    for index,ax in enumerate(axes):
        xs=np.linspace(-8,8,257)if index==0 else np.linspace(-3,3,241)
        zs=np.linspace(-42,4,737)if index==0 else np.linspace(-4,3,281)
        xx,zz=np.meshgrid(xs,zs);world=point+xx[:,:,None]*xaxis+zz[:,:,None]*zaxis;coords=np.moveaxis(case.world_to_voxel(world),-1,0)
        image=map_coordinates(case.mri,coords,order=1,mode='constant',cval=np.nan)
        extent=[xs[0],xs[-1],zs[0],zs[-1]];ax.imshow(image,origin='lower',cmap='gray',vmin=-10 if index!=1 else 100,vmax=100 if index!=1 else 1600,extent=extent)
        ownership=original if index==1 else retained
        for label,color in[(1,'lime'),(3,'cyan'),(4,'red'),(5,'red')]:
            plane=map_coordinates((ownership==label).astype(np.uint8),coords,order=0,mode='constant',cval=0)
            if plane.any()and not plane.all():ax.contour(plane,levels=[.5],colors=[color],linewidths=.65,extent=extent)
        for aa,bb,radius,part in covers:
            a=np.array([(aa-point)@xaxis,(aa-point)@zaxis]);b=np.array([(bb-point)@xaxis,(bb-point)@zaxis]);vec=b-a
            normal=np.array([-vec[1],vec[0]])/np.linalg.norm(vec)
            ax.add_patch(Polygon([a+radius*normal,b+radius*normal,b-radius*normal,a-radius*normal],fill=False,edgecolor='blue',linewidth=.7))
            ax.add_patch(Circle(a,radius,fill=False,edgecolor='blue',linewidth=.7));ax.add_patch(Circle(b,radius,fill=False,edgecolor='blue',linewidth=.7))
        ax.set_xlim(xs[0],xs[-1]);ax.set_ylim(zs[0],zs[-1])
        ax.set_xlabel('mm in rolled cutting plane');ax.set_ylabel('mm along stock tool axis')
    for ax,title in zip(axes,['Full stock 40mm shaft / retained anatomy','Original masks / CT bone window','Retained masks / CT tissue window']):ax.set_title(title,fontsize=8)
    fig.suptitle('CTA012 prospective stock craniotome: roll135°, catalog width radius1mm, BOTH 0.4mm crossbar capsules\nNo 1mm sphere approximation; source CT unchanged; cyan skull, green brain, blue complete tool. No action executed.',fontsize=9)
    fig.tight_layout(rect=(0,0,1,.93));fig.savefig(destination,dpi=140);plt.close(fig)


def preview_permitted(kind, before, preview, protocol, baseline):
    """Additional task predicate only; never substitute for native guards."""
    metrics=preview.get('view',{}).get('metrics',preview.get('metrics',{}))
    removed=set(preview.get('deltas',{}).get('removedCells',[]))
    allowed_reason=(None,'TISSUE_ASPIRATION_IN_PROGRESS')if kind=='cusa' else(None,)
    if preview.get('status')not in('accepted','partial')or preview.get('reason')not in allowed_reason:return False
    if kind!='cusa'and preview.get('status')!='accepted':return False
    if metrics.get('targetRemovedMm3',0)!=before.get('targetRemovedMm3',0):return False
    if metrics.get('vesselEncounters',0)!=before.get('vesselEncounters',0):return False
    for key in('activeVesselWounds','severedVesselConnections','modeledVesselFluidGeneratedMm3'):
        if metrics.get(key,0)!=before.get(key,0):return False
    for key in('damagedFaceCount','woundCount','woundAreaMm2'):
        if metrics.get('vesselPhysicsV2',{}).get(key,0)!=before.get('vesselPhysicsV2',{}).get(key,0):return False
    old=before.get('removedByLabelMm3',{});new=metrics.get('removedByLabelMm3',{})
    for key in set(old)|set(new):
        if key.startswith('supplied_')and new.get(key,0)!=old.get(key,0):return False
    bone=new.get('estimated_skull',0)-old.get('estimated_skull',0)
    brain=new.get('brain',0)-old.get('brain',0)
    if kind=='kerf':
        return(bool(removed)and removed<=set(protocol['authorized_kerf_source_cells'])and
            0<bone<=protocol['budgets']['maximum_new_bone_mm3']+1e-8 and abs(brain)<1e-10 and
            metrics.get('brainMaxDisplacementMm',0)==before.get('brainMaxDisplacementMm',0)and
            metrics.get('brainElasticEnergyNmm',0)==before.get('brainElasticEnergyNmm',0))
    if kind=='cusa':
        return(abs(bone)<1e-10 and new.get('brain',0)-baseline.get('removedByLabelMm3',{}).get('brain',0)
            <=protocol['budgets']['maximum_new_brain_mm3']+1e-8)
    return not removed and abs(bone)<1e-10 and abs(brain)<1e-10


def execute_protocol(protocol_path, decision_path, *, root=None, name='012-stock-kerf-execution-v1'):
    from resectionlab.desktop_bridge import BridgeSession,_Request
    from resectionlab.surgical_environment import standardize
    from resectionlab.surgical_engine import SurgicalEngine
    root=checked_root(root,required_bytes=1024**3)
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,127}',name):raise ValueError('Plain immutable token required')
    def contained(value):
        path=Path(value).resolve()
        if not path.is_relative_to(root):raise ValueError('Input outside guarded root')
        return safe_path(root,path.relative_to(root))
    protocol_path,decision_path=contained(protocol_path),contained(decision_path)
    protocol=json.loads(protocol_path.read_text());decision=json.loads(decision_path.read_text())
    if decision.get('decision')!='accepted_for_bounded_stock_kerf_execution' or decision.get('protocol_sha256')!=sha256_file(protocol_path):
        raise ValueError('Exact independently approved prospective protocol required')
    source=safe_path(root,protocol['source_path'])
    if sha256_file(source)!=protocol['source_sha256']or decision.get('source_sha256')!=protocol['source_sha256']:
        raise ValueError('Exact reviewed guarded source differs')
    thermal=safe_path(root,protocol['thermal_prefix']['path'])
    if sha256_file(thermal)!=protocol['thermal_prefix']['sha256']:raise ValueError('Historical thermal evidence differs')
    case,artifacts,_=load_workspace(source);old_record=artifacts['surgical_toolbox'];old_snapshot=old_record['branches'][old_record['branchId']]
    for instrument,radius in [('craniotome',1.),('cusa',1.5)]:
        if TOOLS[instrument]['radiusMm']!=radius or TOOLS[instrument]['shaftRadiusMm']!=.5:
            raise ValueError('Reviewed stock geometry changed')
    if old_snapshot['checkpoint']['state'].get('toolMechanicsRevision')!=2 or old_snapshot['checkpoint']['state'].get('physicalClockRevision')!=1:
        raise ValueError('Reviewed mechanics/clock revisions differ')
    if old_snapshot['stateId']!=protocol['initial_state_id']or semantic_digest(old_snapshot['checkpoint']['state'])!=protocol['full_physical_prefix_digest']:
        raise ValueError('Physical checkpoint differs; reset/transplant prohibited')
    directory=safe_path(root,'integrated-v1/access/'+name)
    if directory.exists():raise ValueError('Preserve prior execution')
    directory.mkdir(parents=True);code_sha=sha256_file(__file__)
    code_path=safe_path(root,'integrated-v1/recipes/'+code_sha+'-'+Path(__file__).name)
    code_path.parent.mkdir(parents=True,exist_ok=True)
    if not code_path.exists():code_path.write_bytes(Path(__file__).read_bytes())
    if sha256_file(code_path)!=code_sha:raise ValueError('Immutable execution source snapshot differs')
    records=[];started=time.perf_counter();session=BridgeSession(directory/'transfers',run_dir=directory/'runs')
    def call(operation,args):return session.execute(operation,args,_Request(operation),lambda *_:None)
    current=None
    try:
        identity=call('loadCase',{'path':str(source)})['caseHash']
        current=call('surgicalRehearsal',{'caseHash':identity,'command':'inspect'})
        if current['view']['stateId']!=protocol['initial_state_id']:raise ValueError('Public inspection differs from frozen checkpoint')
        baseline=current['view']['metrics'];positive_kerf=False;material_stopped=False;local_goal_reached=False;preview_calls=step_calls=0
        goal=np.array(protocol['actions'][7]['action']['path'][-1]['position'])
        for group in protocol['actions']:
            label=group['label'];action=group['action'];kind='kerf'if label=='single-reviewed-stock-kerf' else 'cusa'if action['operation']=='apply' else 'setup'
            if group.get('requires_positive_kerf')and(not positive_kerf or material_stopped):continue
            for iteration in range(group.get('repeat',1)):
                if preview_calls>=protocol['budgets']['maximum_new_preview_calls']:raise ValueError('Frozen preview budget exhausted')
                before=current['view'];preview=call('surgicalRehearsal',{'caseHash':identity,'command':'preview','expectedStateId':before['stateId'],'action':action});preview_calls+=1
                allowed=preview_permitted(kind,before['metrics'],preview,protocol,baseline)
                row={'label':label,'iteration':iteration,'action':action,'state_before':before['stateId'],'preview':preview,
                    'task_predicate_passed':allowed,'committed':False,'actual_metrics_after':before['metrics'],
                    'actual_slots_before':before['slots'],'actual_slots_after':before['slots']}
                records.append(row);atomic_json(directory/'actions.json',records)
                if not allowed:
                    if kind in('kerf','cusa'):material_stopped=True;break
                    # No unreviewed cleanup replacement or exchange while inserted.
                    raise ValueError('Approved setup/withdrawal preview refused: '+str(preview.get('reason')))
                if step_calls>=protocol['budgets']['maximum_new_step_calls']:raise ValueError('Frozen step budget exhausted')
                committed=call('surgicalRehearsal',{'caseHash':identity,'command':'step','expectedStateId':before['stateId'],'action':action});step_calls+=1
                current=committed;row.update(committed=True,outcome=committed.get('outcome'),state_after=current['view']['stateId'],
                    preview_step_state_equal=current['view']['stateId']==preview.get('stateId'),actual_metrics_after=current['view']['metrics'],
                    actual_slots_after=current['view']['slots'])
                atomic_json(directory/'actions.json',records)
                if not row['preview_step_state_equal']:raise ValueError('Actual public step differs from preview; no further material work')
                if kind=='kerf':positive_kerf=True
                if kind=='cusa'and label=='original-CUSA-local-passage'and np.linalg.norm(np.array(current['view']['slots']['right']['position'])-goal)<=.05:
                    local_goal_reached=True;break
        saved=directory/'replay.ressectionlab';call('saveCase',{'caseHash':identity,'path':str(saved)})
        reopened,new_artifacts,_=load_workspace(saved);record=new_artifacts['surgical_toolbox'];snapshot=record['branches'][record['branchId']]
        replayed=SurgicalEngine.restore(standardize(from_case(reopened,record['mapping']),physics_profile=record['physicsProfile']),snapshot,use_cache=False)
        result={'schema':'bounded-stock-kerf-local-CUSA-execution-v1','source_sha256':sha256_file(source),
            'protocol_sha256':sha256_file(protocol_path),'decision_sha256':sha256_file(decision_path),
            'initial_state_id':protocol['initial_state_id'],'final_state_id':replayed.state_id,
            'source_case_preserved':reopened.semantic_hash==case.semantic_hash,
            'all_37_original_actions_preserved':snapshot['actions'][:len(old_snapshot['actions'])]==old_snapshot['actions'],
            'original_action_count':len(old_snapshot['actions']),'new_action_count':len(snapshot['actions'])-len(old_snapshot['actions']),
            'native_action_count':len(snapshot['actions']),'local_goal_reached':local_goal_reached,
            'initial_metrics':baseline,'final_metrics':replayed.metrics(),'positive_bone_only_kerf_committed':positive_kerf,
            'local_material_work_stopped':material_stopped,'preview_calls':preview_calls,'step_calls':step_calls,
            'exact_cold_replay':replayed.state_id==current['view']['stateId'],
            'all_tools_withdrawn':all(not slot['inserted']and not slot['active']and not slot['grip']for slot in replayed.state['slots'].values()),
            'historical_max_recorded_spatial_temperature_C':max([protocol['thermal_prefix']['historical_max_recorded_spatial_temperature_C']]+
                [row['actual_metrics_after'].get('drillMaxTemperatureC',37.)for row in records]),
            'saved_path':str(saved.relative_to(root)),'saved_sha256':sha256_file(saved),'code_sha256':code_sha,
            'code_snapshot':str(code_path.relative_to(root)),'elapsed_seconds':time.perf_counter()-started,
            'scope':'Exact reviewed finite continuation only. No source edits/reset/extra drilling or clinical interpretation; preserve every negative preview and physical burden.'}
        atomic_json(directory/'report.json',result);return result
    except Exception as error:
        failure={'error':str(error),'records':len(records),'source_sha256':sha256_file(source),'code_sha256':code_sha}
        if current is not None:
            try:
                failed=directory/'failure-retained-state.ressectionlab';call('saveCase',{'caseHash':identity,'path':str(failed)})
                failure['saved_path']=str(failed.relative_to(root));failure['saved_sha256']=sha256_file(failed)
            except Exception as save_error:failure['save_error']=str(save_error)
        atomic_json(directory/'failure.json',failure);raise
    finally:session.close()


def scheduled_segment(start, end, duration, spacing):
    """Exact nonarticulating production schedule, not a velocity choice."""
    from scipy.spatial.transform import Slerp
    a,b=np.asarray(start['position']),np.asarray(end['position'])
    rotations=Rotation.from_quat([start['quaternion'],end['quaternion']])
    angle=(rotations[1]*rotations[0].inv()).magnitude()
    n=max(1,math.ceil((np.linalg.norm(b-a)+40*angle)/min(.4,float(np.min(spacing))/3)),math.ceil(duration/.1))
    slerp=Slerp([0,1],rotations)
    return [{'position':((1-t)*a+t*b).tolist(),'quaternion':slerp(t).as_quat().tolist()}
        for t in np.arange(1,n+1)/n]


def diagnose_reentry(*, root=None, name='012-reentry-cadence-diagnosis-v1'):
    """Reconstruct read-only geometry from receipts; never preview/step an engine."""
    from resectionlab.surgical_cusa_v3 import contact_limited_pose
    root=checked_root(root,required_bytes=512*1024**2)
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,127}',name):raise ValueError('Plain immutable token required')
    execution=safe_path(root,'integrated-v1/access/012-stock-kerf-execution-v1')
    backend=execution/'backend-snapshot'
    bound={p.name:sha256_file(p) for p in backend.glob('*.py')}
    for module in ('surgical_engine.py','surgical_cusa_v3.py','surgical_motion.py','geometry.py','surgical_tools.py','surgical_task_domain.py'):
        if sha256_file(REPO/'src/resectionlab'/module)!=bound[module]:raise ValueError('Snapshotted production module changed: '+module)
    directory=safe_path(root,'integrated-v1/access/'+name)
    if directory.exists():raise ValueError('Preserve diagnosis')
    directory.mkdir()
    saved=execution/'replay.ressectionlab';case,artifacts,_=load_workspace(saved)
    rec=artifacts['surgical_toolbox'];snap=rec['branches'][rec['branchId']];state=snap['checkpoint']['state']
    if len(snap['actions'])!=44 or snap['stateId']!='sha256:168871aa550dffa33690d8ef8cd85808d145e3b82e488ef2f70ad83407923a55':raise ValueError('Exact retained44-action state required')
    scene=from_case(case);grid=GeometryScene(np.zeros(scene.shape,bool),case.affine)
    rows=json.loads(safe_path(root,'integrated-v1/probes/012-conditional-centerline-execution-v1/actions.json').read_text())
    old=rows[36];new=json.loads((execution/'actions.json').read_text())[6]
    labels=scene.labels.ravel();by_label={row['id']:row for row in scene.regions};params={'radiusMm':1.5,'depthMm':2.,'contactGeometryRevision':2}
    final_removed=set(state['removed']);old_events=[e for e in old['outcome']['events']if e['kind']=='cusa_contact']
    old_removed={c for e in old_events for c in e['removedCells']};kerf={22176725,22176726}
    if state.get('deformation')or state.get('holds'):raise ValueError('Undeformed released source required')
    class ReceiptGeometry:
        def __init__(self,removed):
            self.scene=scene;self.by_label=by_label
            self.state={'removed':set(removed),'toolMechanicsRevision':2,'holds':{},'deformation':{}}
        def touched(self,a,b,radius):
            ijk=capsule_voxel_indices(grid,a,b,radius)
            ids=np.ravel_multi_index(ijk.T,scene.shape)if len(ijk)else[]
            return [int(i)for i in ids if labels[i] and int(i)not in self.state['removed']]
    def pose(p,q):return {'position':np.asarray(p).tolist(),'quaternion':np.asarray(q).tolist()}
    def contacts(proxy,start,end):
        result={'shaft':set(),'working':set()};caps=swept_envelopes(TOOLS['cusa'],params,start,end)
        for a,b,r,part in caps:result[part].update(proxy.touched(a,b,r))
        return {part:{'bone':[c for c in sorted(ids)if by_label[int(labels[c])]['material']=='bone'],
            'vessel':[c for c in sorted(ids)if by_label[int(labels[c])]['material']=='vessel'],
            'soft_or_target':[c for c in sorted(ids)if by_label[int(labels[c])]['material']in('soft_tissue','tumor')]}
            for part,ids in result.items()},[{'a':a.tolist(),'b':b.tolist(),'radiusMm':float(r),'part':part}for a,b,r,part in caps]
    reconstructed={}
    for key,row,start,receipt,removed,events in [
        ('old_committed',old,old['slotsBefore']['right'],old['outcome'],final_removed-kerf-old_removed,old_events),
        ('new_uncommitted_preview',new,new['actual_slots_before']['right'],new['preview'],final_removed,
         [e for e in new['preview']['events']if e['kind']=='cusa_contact'])]:
        proxy=ReceiptGeometry(removed);schedule=scheduled_segment(start,row['action']['path'][0],row['action']['duration'],scene.spacing)
        current=pose(start['position'],start['quaternion']);records=[]
        for i,requested in enumerate(schedule[:len(events)+1]):
            limited,_=contact_limited_pose(proxy,current['position'],requested['position'])
            endpoint=pose(limited,requested['quaternion']);hits,caps=contacts(proxy,current,endpoint)
            record={'sample_index_zero_based':i,'requested_pose':requested,'previous_achieved_pose':current,
                'contact_limited_candidate_pose':endpoint,'full_capsules':caps,'contacts':hits,
                'receipt_success':i<len(events),'dt_seconds':row['action']['duration']/len(schedule)}
            record['exact_translation_sphere_sweep_bone_cells']=[c for c in proxy.touched(current['position'],endpoint['position'],1.5)
                if by_label[int(labels[c])]['material']=='bone']
            records.append(record)
            if i<len(events):
                if hits['shaft']['bone']or hits['shaft']['vessel']or hits['working']['bone']or hits['working']['vessel']:
                    raise ValueError('Reconstruction disagrees with successful native sample')
                current=endpoint;proxy.state['removed'].update(events[i]['removedCells'])
        expected=old['slotsAfter']['right']if key=='old_committed'else new['preview']['view']['slots']['right']
        if not np.allclose(current['position'],expected['position'],atol=1e-9,rtol=0):raise ValueError('Actual achieved reconstruction mismatch')
        if not np.allclose(records[-1]['requested_pose']['position'],receipt['contactPosition'],atol=1e-9,rtol=0):raise ValueError('Rejected requested sample mismatch')
        reconstructed[key]={'action':row['action'],'initial_slot':start,'schedule_count':len(schedule),
            'nominal_speed_mm_per_second':float(np.linalg.norm(np.subtract(schedule[-1]['position'],start['position']))/row['action']['duration']),
            'success_sample_count':len(events),'achieved_pose_before_rejected_sample':current,
            'rejected_sample':records[-1],'samples':records,'resolved_parameters':{
                'rpm':40000.,'forceN':1.,'duty':1.,'radiusMm':1.5,'depthMm':2.,'flowMm3PerSecond':20.,'power':1.,'jawOpeningMm':2.,'contactGeometryRevision':2}}
    old_study=reconstructed['old_committed'];new_study=reconstructed['new_uncommitted_preview']
    # A single exact achieved-prefix proposal, not a rate/pose search. Reuse each
    # already executed contact-limited pose at its original physical microstep dt.
    path=[x['contact_limited_candidate_pose']for x in old_study['samples']if x['receipt_success']]
    proxy=ReceiptGeometry(final_removed);previous=pose(new['actual_slots_before']['right']['position'],new['actual_slots_before']['right']['quaternion']);prefix_hits=[]
    for endpoint in path:
        hits,caps=contacts(proxy,previous,endpoint);prefix_hits.append({'from':previous,'to':endpoint,'contacts':hits,'capsules':caps});previous=endpoint
    clean=all(not x['contacts'][part][material]for x in prefix_hits for part in ('working','shaft')for material in ('bone','vessel'))
    from resectionlab.surgical_task_domain import _outside
    domain=reviewed_domain(scene)
    domain_pass=all(_outside(scene,domain,[(np.asarray(c['a']),np.asarray(c['b']),c['radiusMm'],c['part'])for c in x['capsules']])is None for x in prefix_hits)
    reconstructed_counts=[len(scheduled_segment(x['from'],x['to'],old['action']['duration']/old_study['schedule_count'],scene.spacing))for x in prefix_hits]
    if any(n!=1 for n in reconstructed_counts):raise ValueError('Original achieved prefix would be resubdivided')
    proposal={'schema':'read-only-original-achieved-prefix-proposal-v1','execution_approved':False,
        'source_path':str(saved.relative_to(root)),'source_sha256':sha256_file(saved),'source_state_id':snap['stateId'],
        'action':{'slot':'right','operation':'apply','path':path,'duration':len(path)*old['action']['duration']/old_study['schedule_count']},
        'basis':'Exactly88 already-executed achieved poses at original2/169-second cadence; no tuned velocity/pose/geometry search. Requires new independent prospective review before any public call.',
        'read_only_full_sweeps_have_zero_bone_or_vessel_contacts':clean,'full_tool_FOV_reviewed_domain_pass':domain_pass,
        'one_native_sample_per_original_achieved_segment':True,'source_arrays_mutated':False}
    atomic_json(directory/'original-achieved-prefix-proposal.json',proposal)
    atomic_json(directory/'full-original-achieved-prefix-sweeps.json',prefix_hits)
    atomic_json(directory/'reconstructed-microsteps.json',reconstructed)
    last_moving=next(x for x in reversed(old_study['samples'][:-1])if np.linalg.norm(np.subtract(x['contact_limited_candidate_pose']['position'],x['previous_achieved_pose']['position']))>1e-9)
    _render_reentry_witnesses(case,scene,final_removed,[last_moving,old_study['rejected_sample'],new_study['rejected_sample']],directory/'source-sweep-witnesses.png')
    report={'schema':'CTA012-read-only-reentry-diagnosis-v1','source_state_id':snap['stateId'],'source_sha256':sha256_file(saved),
        'all44_actions_unchanged':True,'production_modules_match_execution_snapshot':True,'backend_hashes':bound,
        'new_public_actions_or_previews':0,'diagnosis_driver_sha256':sha256_file(__file__),
        'old_requested_speed_mm_per_second':old_study['nominal_speed_mm_per_second'],
        'new_requested_speed_mm_per_second':new_study['nominal_speed_mm_per_second'],
        'old_schedule_count':old_study['schedule_count'],'new_schedule_count':new_study['schedule_count'],
        'old_final_accepted_sample':old_study['samples'][-2],'old_rejected_sample':old_study['rejected_sample'],
        'old_last_moving_accepted_sample':last_moving,
        'new_final_accepted_sample':new_study['samples'][-2],'new_rejected_sample':new_study['rejected_sample'],
        'new_uncommitted_preview_achieved_pose':new_study['achieved_pose_before_rejected_sample'],
        'actual_committed_tool_during_refusal':'External; preview is detached and none of its71 successful samples were committed.',
        'proposal_sha256':sha256_file(directory/'original-achieved-prefix-proposal.json'),
        'proposed_original_prefix_sweep_bone_vessel_clear':clean,
        'proposed_original_prefix_full_tool_FOV_reviewed_domain_pass':domain_pass,
        'interpretation':'The new reentry is nominally slower, not faster. Both schedules are spatially governed. The original CUSA was contact-limited and arrived via a0.080067mm movement followed by dwells; the new direct interpolation arrives via0.165455mm. Its conservative midpoint sphere radius1.582727mm intersects retained skull cell22060722 although the exact translation sphere sweep and static endpoint do not. This is a schedule/segmentation-dependent conservative geometric refusal, not evidence that removed bone regrew or the endpoint is occupied. Old rejected requested vs contact-limited vs achieved positions remain distinct.',
        'scope':'Geometric reconstruction from existing receipts only. No effects replayed, source/state edits, rate search, public preview or physical extension.'}
    atomic_json(directory/'report.json',report)
    return {'report':str(directory/'report.json'),'report_sha256':sha256_file(directory/'report.json'),'proposal_sha256':report['proposal_sha256'],'prefix_bone_vessel_clear':clean}


def _render_reentry_witnesses(case,scene,removed,witnesses,destination):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle,Polygon
    from scipy.ndimage import map_coordinates
    anchor=np.asarray(witnesses[-1]['contact_limited_candidate_pose']['position'])
    q=witnesses[-1]['contact_limited_candidate_pose']['quaternion'];zaxis=Rotation.from_quat(q).apply([0.,0.,1.])
    cell=22060722;ijk=np.array(np.unravel_index(cell,scene.shape));bone=case.voxel_to_world(ijk)
    xaxis=bone-anchor-zaxis*((bone-anchor)@zaxis);xaxis/=np.linalg.norm(xaxis)
    fig,axes=plt.subplots(1,4,figsize=(16,6))
    panels=[*witnesses,witnesses[-1]]
    for index,(ax,w)in enumerate(zip(axes,panels)):
        xs=np.linspace(-3,3,241)if index<3 else np.linspace(-4,4,161)
        zs=np.linspace(-3,3,241)if index<3 else np.linspace(-42,3,901)
        xx,zz=np.meshgrid(xs,zs);world=anchor+xx[:,:,None]*xaxis+zz[:,:,None]*zaxis;coords=np.moveaxis(case.world_to_voxel(world),-1,0)
        extent=[xs[0],xs[-1],zs[0],zs[-1]]
        ax.imshow(map_coordinates(case.mri,coords,order=1,mode='constant',cval=np.nan),origin='lower',cmap='gray',vmin=0,vmax=1200,extent=extent)
        panel_removed=set(removed)-{22176725,22176726}if index<2 else set(removed)
        retained=scene.labels.copy().ravel();retained[list(panel_removed)]=0;retained=retained.reshape(scene.shape)
        plane=map_coordinates((retained==3).astype(float),coords,order=0,mode='constant',cval=0)
        if plane.any()and not plane.all():ax.contour(plane,[.5],colors=['cyan'],extent=extent,linewidths=.6)
        project=lambda p:np.array([(np.asarray(p)-anchor)@xaxis,(np.asarray(p)-anchor)@zaxis])
        a=project(w['previous_achieved_pose']['position']);b=project(w['contact_limited_candidate_pose']['position']);vec=b-a
        if np.linalg.norm(vec)>1e-10:
            normal=np.array([-vec[1],vec[0]])/np.linalg.norm(vec)
            ax.add_patch(Polygon([a+1.5*normal,b+1.5*normal,b-1.5*normal,a-1.5*normal],fill=False,edgecolor='lime',linewidth=.7))
        for p in(a,b):ax.add_patch(Circle(p,1.5,fill=False,edgecolor='lime',linewidth=.7))
        for c in w['full_capsules']:
            if index<3 and c['part']=='shaft':continue
            ca,cb=project(c['a']),project(c['b']);r=c['radiusMm'];v=cb-ca
            if np.linalg.norm(v)>1e-10:
                normal=np.array([-v[1],v[0]])/np.linalg.norm(v)
                ax.add_patch(Polygon([ca+r*normal,cb+r*normal,cb-r*normal,ca-r*normal],fill=False,edgecolor='magenta',linewidth=.6))
            for p in(ca,cb):ax.add_patch(Circle(p,r,fill=False,edgecolor='magenta',linewidth=.7))
        ax.scatter(*project(bone),c='red',s=10);ax.scatter(*a,c='yellow',s=10);ax.scatter(*b,c='lime',s=10)
        ax.set_xlim(xs[0],xs[-1]);ax.set_ylim(zs[0],zs[-1]);ax.set_xlabel('tangent mm');ax.set_ylabel('axial mm')
    for ax,title in zip(axes,['Old accepted sample82 / before kerf','Old rejected sample88 / before kerf','New rejected sample71 / after kerf','New full40mm tool conservative cover']):ax.set_title(title,fontsize=8)
    fig.suptitle('CTA012 unchanged acquired baseline / retained skull. Green exact sphere translation; magenta production conservative cover; red cell22060722.\nRequested, contact-limited candidate and actual achieved poses are separate; no new public action or preview.',fontsize=9)
    fig.tight_layout(rect=(0,0,1,.91));fig.savefig(destination,dpi=150);plt.close(fig)


def diagnose_remaining_port(*,root=None,name='012-post-kerf-nominal-port-v1'):
    """Exact nominal passage and the previously declared140 kerf poses only."""
    from scipy.optimize import minimize_scalar,brentq
    from resectionlab.geometry import _segment_cell_distances
    from resectionlab.surgical_task_domain import _outside
    root=checked_root(root,required_bytes=512*1024**2)
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,127}',name):raise ValueError('Plain immutable token required')
    source=safe_path(root,'integrated-v1/access/012-stock-kerf-execution-v1/replay.ressectionlab')
    if sha256_file(source)!='a9a31af4952b4d12a7230e3989b479d9d3868737450cadd0b3a8cb7cc60ba4f4':raise ValueError('Exact44-action source required')
    case,artifacts,_=load_workspace(source);record=artifacts['surgical_toolbox'];snapshot=record['branches'][record['branchId']];state=snapshot['checkpoint']['state']
    if len(snapshot['actions'])!=44 or state.get('deformation')or state.get('holds'):raise ValueError('Exact released undeformed44-action source required')
    prior=safe_path(root,'integrated-v1/access/012-channel-diagnosis-v2/report.json')
    specification=json.loads(prior.read_text())['specification'];origin=np.array(specification['last_reached_tip_ras_mm']);q=specification['fixed_quaternion']
    rotation=Rotation.from_quat(q);axis,u,v=rotation.apply([0.,0.,1.]),rotation.apply([1.,0.,0.]),rotation.apply([0.,1.,0.])
    scene=from_case(case);domain=reviewed_domain(scene);labels=scene.labels.copy().ravel();labels[state['removed']]=0
    grid=GeometryScene(np.zeros(scene.shape,bool),case.affine);regions={r['id']:r for r in scene.regions}
    directory=safe_path(root,'integrated-v1/access/'+name)
    if directory.exists():raise ValueError('Preserve diagnosis')
    directory.mkdir()
    def cells(cover):
        result={'working':set(),'shaft':set()}
        for a,b,r,part in cover:
            indices=capsule_voxel_indices(grid,a,b,r)
            if len(indices):result[part].update(int(c)for c in np.ravel_multi_index(indices.T,scene.shape)if labels[c])
        return result
    def classified(ids):return {r['name']:sorted(c for c in ids if labels[c]==r['id'])for r in scene.regions}
    def voxel_rows(ids):
        output=[]
        for c in sorted(ids):
            index=np.array(np.unravel_index(c,scene.shape));output.append({'cell_id':c,'ijk':index.tolist(),
                'ras_mm':case.voxel_to_world(index).tolist(),'source_CT_scaled_intensity':float(case.mri[tuple(index)]),
                'retained_region':regions[int(labels[c])]['name']})
        return output
    endpoint=origin+3*axis
    # Exact unions for this fixed-axis translation: sphere sweeps a capsule;
    # the collinear40mm shaft union is one capsule as well. No midpoint inflation.
    nominal=[(origin,endpoint,1.5,'working'),(origin-40*axis,endpoint,.5,'shaft')]
    hits=cells(nominal);bone={c for c in hits['working']if regions[int(labels[c])]['material']=='bone'}
    starts=[]
    for c in bone:
        center=case.voxel_to_world(np.array(np.unravel_index(c,scene.shape)))
        f=lambda depth:distance_to_boxes(origin+depth*axis,np.asarray([center]),np.array(case.spacing_mm)/2)-1.5
        optimum=minimize_scalar(f,bounds=(0.,3.),method='bounded',options={'xatol':1e-13})
        if f(0)<=0:entry=0.
        elif optimum.fun<=0:entry=float(brentq(f,0.,optimum.x,xtol=1e-13))
        elif f(3.)<=0:entry=float(brentq(f,0.,3.,xtol=1e-13))
        else:raise ValueError('Capsule witness and exact boxdistance disagree')
        starts.append({'cell_id':c,'first_closed_cell_tip_contact_depth_mm':entry})
    starts.sort(key=lambda x:x['first_closed_cell_tip_contact_depth_mm'])
    # Re-evaluate exactly the frozen candidate family. No new centers/angles.
    family=specification['stock_kerf_inspection'];kerf_params={'radiusMm':1.,'depthMm':2.};rows=[];qualified_union=set();all_static_union=set()
    external=state['slots']['right'];pose=lambda p,quat:{'position':np.asarray(p).tolist(),'quaternion':np.asarray(quat).tolist()}
    def eligible(ids,point,orientation):
        output=set()
        for c in ids:
            if regions[int(labels[c])]['material']!='bone':continue
            xyz=case.voxel_to_world(np.array(np.unravel_index(c,scene.shape)));local=orientation.inv().apply(xyz-point)
            if abs(local[2])<=2. and abs(local[1])<=min(case.spacing_mm)*.5:output.add(c)
        return output
    for depth in family['inward_depths_mm']:
        for du,dv in family['tangent_offsets_mm']:
            point=origin+depth*axis+du*u+dv*v
            for roll in family['roll_angles_deg']:
                orientation=rotation*Rotation.from_euler('z',roll,degrees=True);quat=orientation.as_quat();end=pose(point,quat)
                static=cells(envelopes(TOOLS['craniotome'],kerf_params,point,quat));working=static['working']
                cut=eligible(working,point,orientation);static_bone_only=bool(cut)and not static['shaft']and all(regions[int(labels[c])]['material']=='bone'for c in working)
                row={'depth_mm':depth,'offset_u_mm':du,'offset_v_mm':dv,'roll_deg':roll,'pose':end,
                    'static_contacts':{part:classified(ids)for part,ids in static.items()},'static_effect_eligible_bone_cells':sorted(cut),
                    'static_bone_only':static_bone_only,'full_approach_qualified':False}
                if static_bone_only:
                    all_static_union.update(cut)
                    first,middle=pose(external['position'],quat),pose(origin,quat)
                    cover=finite_sweeps(TOOLS['craniotome'],kerf_params,first,middle)+finite_sweeps(TOOLS['craniotome'],kerf_params,middle,end)
                    swept=cells(cover);inside=_outside(scene,domain,cover)is None
                    qualified=inside and not swept['shaft']and all(regions[int(labels[c])]['material']=='bone'for c in swept['working'])
                    row.update(full_approach_qualified=qualified,full_approach_FOV_domain_pass=inside,
                        full_approach_contacts={part:classified(ids)for part,ids in swept.items()})
                    if qualified:
                        # Only an optimistic effect upper bound along this same
                        # declared approach. No dose/effects or removals replayed.
                        potential=set();previous=first
                        for waypoint in(middle,end):
                            for sample in scheduled_segment(previous,waypoint,.5,scene.spacing):
                                cc=cells(envelopes(TOOLS['craniotome'],kerf_params,sample['position'],sample['quaternion']))
                                potential.update(eligible(cc['working'],np.array(sample['position']),orientation))
                            previous=waypoint
                        row['approach_potential_effect_upper_bound_cells']=sorted(potential);qualified_union.update(potential)
                rows.append(row)
    barrier_report=json.loads(safe_path(root,'integrated-v1/access/012-post-kerf-channel-diagnosis-v1/report.json').read_text())
    uncovered=bone-qualified_union
    _render_remaining_port(case,scene,state['removed'],origin,axis,u,v,bone,uncovered,directory/'source-nominal-port.png')
    atomic_json(directory/'declared-140-stock-kerf-candidates.json',rows)
    atomic_json(directory/'nominal-source-bone-cells.json',voxel_rows(bone))
    report={'schema':'read-only-current44-nominal-port-v1','source_path':str(source.relative_to(root)),
        'source_sha256':sha256_file(source),'source_state_id':snapshot['stateId'],'action_count_preserved':44,
        'no_new_public_preview_or_step':True,'no_obstacle_source_or_state_edits':True,'diagnosis_driver_sha256':sha256_file(__file__),
        'origin_ras_mm':origin.tolist(),'goal_ras_mm':endpoint.tolist(),'quaternion':q,
        'tool_geometry':{'radius_mm':1.5,'shaft_radius_mm':.5,'shaft_length_mm':40.,'revision':2},
        'exact_nominal_3mm_translation_contacts':{part:classified(ids)for part,ids in hits.items()},
        'exact_working_retained_bone_count':len(bone),'exact_working_retained_bone_mm3':len(bone)*case.voxel_volume_mm3,
        'first_bone_entry_by_cell':starts,'nominal_centerline_clearance_samples':barrier_report['sampled_original_centerline_clearance_mm'],
        'continuous_radius1mm_local_steering_barrier':barrier_report['continuous_cross_section_barrier'],
        'remaining_bone_is_exact_nominal_obstruction_not_midpoint_cover_only':bool(bone),
        'declared_kerf_family_count':len(rows),'static_bone_only_candidates':sum(r['static_bone_only']for r in rows),
        'full_approach_qualified_candidates':sum(r['full_approach_qualified']for r in rows),
        'all_static_eligible_cells':sorted(all_static_union),'qualified_approach_optimistic_kerf_union':sorted(qualified_union),
        'nominal_bone_cells_not_in_any_qualified_approach_effect_upper_bound':voxel_rows(uncovered),
        'uncovered_cell_static_candidate_references':{str(c):[i for i,r in enumerate(rows)if c in r['static_effect_eligible_bone_cells']]for c in sorted(uncovered)},
        'all_nominal_blockers_addressable_by_currently_qualified_declared_approach_upper_bound':not uncovered,
        'conclusion':'Measured bounded barrier: the3mm nominal CUSA passage remains obstructed by retained bone after the two-cell kerf. A smaller conservative cover can resolve external reentry refusal but cannot erase these exact nominal intersections. Stock-kerf rows are static geometric/effect upper bounds, not prospective authorization or a dose/sequence feasibility proof. The25qualified approaches leave6nominal cells; the3other bone-only endpoints fail their declared conservative approach cover on retained brain. Other listed candidate endpoints have exact nominal brain contacts. This does not prove that different segmentation, a newly reviewed trajectory, partial endpoint, or a combined future material sequence is impossible. No such extension was searched or authorized. No reentry-only target progress claim.',
        'scope':'Only previously declared radius1mm local steering and frozen140 stock-kerf poses/approaches; no global all-approach impossibility, source-label diagnosis, tuned speed/pose search or clinical safety claim.'}
    atomic_json(directory/'report.json',report)
    return {'report':str(directory/'report.json'),'sha256':sha256_file(directory/'report.json'),
        'bone_cells':len(bone),'earliest_bone_depth_mm':starts[0]['first_closed_cell_tip_contact_depth_mm']if starts else None,
        'qualified_kerf_candidates':report['full_approach_qualified_candidates'],'uncovered_nominal_bone_cells':len(uncovered)}


def _render_remaining_port(case,scene,removed,origin,axis,u,v,bone,uncovered,destination):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle,Rectangle
    from scipy.ndimage import map_coordinates
    labels=scene.labels.copy().ravel();labels[list(removed)]=0;labels=labels.reshape(scene.shape)
    fig,axes=plt.subplots(1,3,figsize=(12,7))
    for index,ax in enumerate(axes):
        tangent=u if index!=2 else v;xs=np.linspace(-4,4,241);zs=np.linspace(-42,5,941)if index==0 else np.linspace(-3,5,321)
        xx,zz=np.meshgrid(xs,zs);world=origin+xx[:,:,None]*tangent+zz[:,:,None]*axis;coords=np.moveaxis(case.world_to_voxel(world),-1,0)
        extent=[xs[0],xs[-1],zs[0],zs[-1]]
        ax.imshow(map_coordinates(case.mri,coords,order=1,mode='constant',cval=np.nan),origin='lower',cmap='gray',vmin=0,vmax=1200,extent=extent)
        plane=map_coordinates((labels==3).astype(float),coords,order=0,mode='constant',cval=0)
        if plane.any()and not plane.all():ax.contour(plane,[.5],colors=['cyan'],extent=extent,linewidths=.6)
        for z,r in[(0,1.5),(3,1.5)]:ax.add_patch(Circle((0,z),r,fill=False,edgecolor='lime',linewidth=.7))
        ax.add_patch(Rectangle((-1.5,0),3.,3.,fill=False,edgecolor='lime',linewidth=.7))
        if index==0:
            ax.add_patch(Rectangle((-.5,-40),1.,43.,fill=False,edgecolor='blue',linewidth=.7))
            for z in(-40,3):ax.add_patch(Circle((0,z),.5,fill=False,edgecolor='blue',linewidth=.7))
        for c in bone:
            xyz=case.voxel_to_world(np.array(np.unravel_index(c,scene.shape)))-origin
            ax.scatter(xyz@tangent,xyz@axis,c='red'if c in uncovered else 'gold',s=10)
        ax.set_xlim(xs[0],xs[-1]);ax.set_ylim(zs[0],zs[-1]);ax.set_xlabel('tangent mm');ax.set_ylabel('inward mm')
    for ax,title in zip(axes,['Exact40mm shaft/3mm passage union','Local source/calvarial rim: tangent U','Local source/calvarial rim: tangent V']):ax.set_title(title,fontsize=8)
    fig.suptitle('CTA012 retained 44-action state: exact nominal tool union, no midpoint inflation. Cyan retained skull; green 1.5mm tip union; blue 0.5mm shaft.\nProjected source-cell centers: red 6 outside optimistic eligible union of 25 qualified declared kerf approaches; gold 10 inside. No effects/actions executed.',fontsize=8)
    fig.tight_layout(rect=(0,0,1,.92));fig.savefig(destination,dpi=150);plt.close(fig)


def linear_filter_interval(lo,hi,offset,slope,half):
    """Intersect a parameter interval with |offset - t*slope| <= half."""
    if slope==0:return(lo,hi)if abs(offset)<=half else None
    aa,bb=sorted(((offset-half)/slope,(offset+half)/slope));lo=max(lo,aa);hi=min(hi,bb)
    return(lo,hi)if lo<=hi else None


def certify_fixed_kerf(*,root=None,name='012-fixed-kerf-cover-certification-v1'):
    """Bound the SAME frozen curves; never change geometry or execute effects."""
    from resectionlab.geometry import _segment_cell_distances
    from resectionlab.surgical_task_domain import _outside
    root=checked_root(root,required_bytes=512*1024**2)
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,127}',name):raise ValueError('Plain immutable token required')
    base=safe_path(root,'integrated-v1/access/012-post-kerf-nominal-port-v3')
    report=json.loads((base/'report.json').read_text());rows=json.loads((base/'declared-140-stock-kerf-candidates.json').read_text())
    source=safe_path(root,report['source_path'])
    if sha256_file(source)!=report['source_sha256']:raise ValueError('Exact44-action source changed')
    case,artifacts,_=load_workspace(source);record=artifacts['surgical_toolbox'];snapshot=record['branches'][record['branchId']];state=snapshot['checkpoint']['state']
    scene=from_case(case);grid=GeometryScene(np.zeros(scene.shape,bool),case.affine);domain=reviewed_domain(scene)
    labels=scene.labels.copy().ravel();labels[state['removed']]=0;regions={r['id']:r for r in scene.regions}
    directory=safe_path(root,'integrated-v1/access/'+name)
    if directory.exists():raise ValueError('Preserve certification')
    directory.mkdir()
    limits={'initial_partition_maximum_mm':.25,'maximum_bisection_depth':16,
        'minimum_subinterval_length_mm':1e-5,'maximum_cover_queries_per_curve':200000,
        'eligibility_maximum_bisection_depth':20,'eligibility_minimum_interval_length_mm':1e-6,
        'eligibility_maximum_distance_queries_per_cell_leg':10000,'eligibility_exclusion_tolerance_mm':1e-9,
        'unresolved_policy':'withhold, never treat as clear or eligible',
        'scope':'Exactly3previous coarse-cover-only rejections within frozen140 endpoints/28fixed-roll approach curves. Existing25coarse certificates retained. No physical calls or tuning.'}
    atomic_json(directory/'prospective-mathematical-limits.json',{'limits':limits,'source_sha256':report['source_sha256'],
        'source_state_id':snapshot['stateId'],'fixed_curve_inventory_sha256':sha256_file(base/'declared-140-stock-kerf-candidates.json')})
    params={'radiusMm':1.,'depthMm':2.};origin=np.array(report['origin_ras_mm']);external=np.array(state['slots']['right']['position'])
    def pose(point,q):return {'position':np.asarray(point).tolist(),'quaternion':np.asarray(q).tolist()}
    def hits(cover):
        result={'working':set(),'shaft':set()}
        for a,b,r,part in cover:
            indices=capsule_voxel_indices(grid,a,b,r)
            if len(indices):result[part].update(int(c)for c in np.ravel_multi_index(indices.T,scene.shape)if labels[c])
        return result
    def forbidden(cells):return {'shaft':sorted(cells['shaft']),
        'working':sorted(c for c in cells['working']if regions[int(labels[c])]['material']!='bone')}
    def bad(cells):return bool(cells['shaft']or cells['working'])
    certificates=[];certified=[]
    for index,row in enumerate(rows):
        if not row['static_bone_only']:continue
        q=row['pose']['quaternion'];waypoints=[pose(external,q),pose(origin,q),row['pose']]
        if row['full_approach_qualified']:
            certified.append(index);certificates.append({'candidate_index':index,'status':'certified_by_existing_coarse_bound','new_adaptive_queries':0});continue
        counter=[0];leaves=[];witnesses=[];maxdepth=[0]
        def certify(a,b,depth):
            maxdepth[0]=max(maxdepth[0],depth)
            if counter[0]>=limits['maximum_cover_queries_per_curve']:return 'unresolved_query_budget'
            counter[0]+=1
            cover=swept_envelopes(TOOLS['craniotome'],params,a,b);allhits=hits(cover);contact=forbidden(allhits);outside=_outside(scene,domain,cover)
            if not bad(contact)and outside is None:
                leaves.append({'from':a,'to':b,'bisection_depth':depth,'full_cover_material_clear':True,'FOV_domain_pass':True,
                    'possible_bone_cells':sorted(allhits['working'])});return 'certified'
            midpoint=pose((np.array(a['position'])+np.array(b['position']))/2,q)
            for sample in(a,midpoint,b):
                static=envelopes(TOOLS['craniotome'],params,sample['position'],q);actual=forbidden(hits(static));out=_outside(scene,domain,static)
                if bad(actual)or out is not None:
                    witnesses.append({'pose':sample,'exact_nominal_forbidden_cells':actual,'domain_issue':out,
                        'interpretation':'Actual unchanged static whole-tool contact on the fixed curve, not inflated-cover evidence.'})
                    return 'unsafe_nominal_contact'
            distance=float(np.linalg.norm(np.subtract(b['position'],a['position'])))
            if depth>=limits['maximum_bisection_depth']or distance<=limits['minimum_subinterval_length_mm']:
                witnesses.append({'from':a,'to':b,'remaining_conservative_forbidden_cells':contact,'domain_issue':outside,'bisection_depth':depth})
                return 'unresolved_resolution_limit'
            left=certify(a,midpoint,depth+1)
            if left!='certified':return left
            return certify(midpoint,b,depth+1)
        status='certified'
        for first,last in zip(waypoints,waypoints[1:]):
            count=max(1,int(np.ceil(np.linalg.norm(np.subtract(last['position'],first['position']))/.25)))
            previous=first
            for fraction in np.arange(1,count+1)/count:
                endpoint=pose((1-fraction)*np.array(first['position'])+fraction*np.array(last['position']),q)
                status=certify(previous,endpoint,0);previous=endpoint
                if status!='certified':break
            if status!='certified':break
        certificate={'candidate_index':index,'status':status,'new_adaptive_queries':counter[0],
            'maximum_bisection_depth_used':maxdepth[0],'certified_leaf_intervals':leaves,'witnesses':witnesses}
        certificates.append(certificate)
        if status=='certified':certified.append(index)
    atomic_json(directory/'fixed-curve-certificates.json',certificates)
    # Continuous eligibility needs both real capsule/cell intersection and the
    # production center-plane filters at the SAME pose. Certify an existence
    # witness or exclude by a Lipschitz bound; unresolved cells remain separate.
    confirmed_union=set();possible_union=set();eligibility_rows=[]
    prior_unmet={r['cell_id']for r in report['nominal_bone_cells_not_in_any_qualified_approach_effect_upper_bound']}
    for index in certified:
        row=rows[index];q=row['pose']['quaternion'];orientation=Rotation.from_quat(q);xyzaxes=orientation.as_matrix()
        waypoints=[external,origin,np.array(row['pose']['position'])];confirmed=set();possible=set();witness=[]
        for leg,(start,end)in enumerate(zip(waypoints,waypoints[1:])):
            delta=end-start;length=float(np.linalg.norm(delta))
            if length<1e-14:continue
            # Reuse the certified full-curve covering inventory. It only
            # enumerates candidates; it never certifies effect eligibility.
            certificate=next(c for c in certificates if c['candidate_index']==index)
            if row['full_approach_qualified']:
                candidates={cell for region in scene.regions if region['material']=='bone'
                    for cell in row['full_approach_contacts']['working'][region['name']]}
            else:candidates={cell for leaf in certificate['certified_leaf_intervals']for cell in leaf['possible_bone_cells']}
            for cell in sorted(candidates):
                if cell in confirmed:continue
                ijk=np.array(np.unravel_index(cell,scene.shape));center=case.voxel_to_world(ijk);local=orientation.inv().apply(center-start);slope=orientation.inv().apply(delta)
                interval=(0.,1.)
                for component,half in((1,min(case.spacing_mm)/2),(2,2.)):
                    interval=linear_filter_interval(*interval,local[component],slope[component],half)
                    if interval is None:break
                if interval is None:continue
                def distance(t):
                    point=start+t*delta;ds=[]
                    for z in(0.,2.):
                        aa=point+orientation.apply([-1.,0.,-z]);bb=point+orientation.apply([1.,0.,-z])
                        ds.append(float(_segment_cell_distances(grid,ijk[None,:],aa,bb)[0]))
                    return min(ds)
                queries=[0]
                def exists(lo,hi,depth):
                    if queries[0]>=limits['eligibility_maximum_distance_queries_per_cell_leg']:return 'unresolved'
                    queries[0]+=1
                    mid=(lo+hi)/2;d=distance(mid)
                    if d<=.4:
                        witness.append({'cell_id':cell,'leg':leg,'fraction':mid,'pose':pose(start+mid*delta,q),
                            'exact_crossbar_to_closed_cell_distance_mm':d,'center_plane_filters_pass':True});return 'confirmed'
                    if d-length*(hi-lo)/2>.4+limits['eligibility_exclusion_tolerance_mm']:return 'excluded'
                    if depth>=limits['eligibility_maximum_bisection_depth']or length*(hi-lo)<=limits['eligibility_minimum_interval_length_mm']:return 'unresolved'
                    left=exists(lo,mid,depth+1)
                    if left=='confirmed':return left
                    right=exists(mid,hi,depth+1)
                    if right=='confirmed':return right
                    return 'unresolved'if'unresolved'in(left,right)else'excluded'
                status=exists(*interval,0)
                if status=='confirmed':confirmed.add(cell);possible.discard(cell)
                elif status=='unresolved':possible.add(cell)
        confirmed_union.update(confirmed);possible_union.update(possible)
        eligibility_rows.append({'candidate_index':index,'confirmed_geometric_kerf_eligible_cells':sorted(confirmed),
            'unresolved_possible_cells':sorted(possible-confirmed),'exact_witnesses':witness})
    atomic_json(directory/'continuous-kerf-eligibility.json',eligibility_rows)
    output={'schema':'read-only-fixed-kerf-continuous-certification-v1','source_sha256':sha256_file(source),'source_state_id':snapshot['stateId'],
        'source_action_count':len(snapshot['actions']),'no_public_calls_or_effects':True,'limits':limits,
        'frozen_candidate_count':len(rows),'frozen_bone_only_approach_count':len(certificates),
        'newly_certified_curve_indices':[c['candidate_index']for c in certificates if c['status']=='certified'],
        'nominal_unsafe_curve_indices':[c['candidate_index']for c in certificates if c['status']=='unsafe_nominal_contact'],
        'unresolved_curve_indices':[c['candidate_index']for c in certificates if c['status'].startswith('unresolved')],
        'all_certified_curve_indices':certified,'confirmed_continuous_geometric_eligible_union':sorted(confirmed_union),
        'unresolved_eligibility_cells':sorted(possible_union-confirmed_union),
        'six_prior_unmet_cells':sorted(prior_unmet),'prior_unmet_confirmed_eligible':sorted(prior_unmet&confirmed_union),
        'prior_unmet_still_not_confirmed':sorted(prior_unmet-confirmed_union),
        'all_six_have_confirmed_geometric_eligibility':prior_unmet<=confirmed_union,
        'other_frozen_endpoint_exact_working_brain_contact_indices':[i for i,r in enumerate(rows)if r['static_contacts']['working']['brain']],
        'remaining_four_endpoint_eligibility_references':{str(c):[i for i,r in enumerate(rows)if c in r['static_effect_eligible_bone_cells']]for c in sorted(prior_unmet-confirmed_union)},
        'interpretation':'Continuous geometric existence of a stock kerf-eligible pose predicts an effect only if an explicit ordinary public stroke is prospectively approved and accepted. No bone is subtracted here; no cumulative physical/dose/thermal sequence or CUSA passage is certified. Existing native material/domain/sweep guards remain unchanged and can refuse a physical discretization. Unresolved bounds are never clear. Distance witnesses require <=0.4mm exact crossbar-to-cell distance; exclusions use a1e-9mm conservative tolerance plus the fixed-translation1-Lipschitz bound. This is finite floating-point certification against unchanged voxel geometry, not subvoxel anatomical certainty. The4remaining nominal blockers have no confirmed eligible pose on these28current-state curves; alternative future combined tissue effects or new curve families are outside scope.',
        'clinical_validation':False,'execution_approved':False,'diagnosis_driver_sha256':sha256_file(__file__)}
    atomic_json(directory/'report.json',output)
    return {'report':str(directory/'report.json'),'report_sha256':sha256_file(directory/'report.json'),
        'newly_certified':output['newly_certified_curve_indices'],'nominal_unsafe':output['nominal_unsafe_curve_indices'],
        'unresolved':output['unresolved_curve_indices'],'six_unmet_now_confirmed':output['prior_unmet_confirmed_eligible'],
        'still_unmet':output['prior_unmet_still_not_confirmed']}


def export_endpoint_envelope_facts(*,root=None,name='012-remaining-blocker-envelope-facts-v1'):
    """Export source facts only; do not infer CSF/parenchyma or change masks."""
    import csv
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from resectionlab.geometry import point_segment_distances
    root=checked_root(root,required_bytes=512*1024**2)
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,127}',name):raise ValueError('Plain immutable token required')
    prior=safe_path(root,'integrated-v1/access/012-post-kerf-nominal-port-v3')
    continuous=safe_path(root,'integrated-v1/access/012-fixed-kerf-cover-certification-v2/report.json')
    remaining=json.loads(continuous.read_text())['prior_unmet_still_not_confirmed']
    if remaining!=[22292417,22292418,22292728,22523179]:raise ValueError('Exact four reviewed remaining blockers required')
    rows=json.loads((prior/'declared-140-stock-kerf-candidates.json').read_text());source=safe_path(root,'integrated-v1/access/012-stock-kerf-execution-v1/replay.ressectionlab')
    if sha256_file(source)!='a9a31af4952b4d12a7230e3989b479d9d3868737450cadd0b3a8cb7cc60ba4f4':raise ValueError('Exact retained44-action source required')
    case,artifacts,_=load_workspace(source);record=artifacts['surgical_toolbox'];snapshot=record['branches'][record['branchId']];state=snapshot['checkpoint']['state']
    if len(snapshot['actions'])!=44:raise ValueError('Retain44-action prefix')
    scene=from_case(case);labels=scene.labels.copy().ravel();labels[state['removed']]=0
    grid=GeometryScene(np.zeros(scene.shape,bool),case.affine)
    directory=safe_path(root,'integrated-v1/access/'+name)
    if directory.exists():raise ValueError('Preserve source-fact export')
    directory.mkdir()
    endpoint_map={str(c):[i for i,row in enumerate(rows)if c in row['static_effect_eligible_bone_cells']]for c in remaining}
    selected=sorted({i for indices in endpoint_map.values()for i in indices});facts=[];brain_union=set();voxel_associations={}
    def voxel_fact(cell):
        ijk=np.array(np.unravel_index(cell,scene.shape));return{'source_cell_id':cell,'native_ijk':ijk.tolist(),
            'ras_mm':case.voxel_to_world(ijk).tolist(),'stored_baseline_CTA_scaled_intensity':float(case.mri[tuple(ijk)]),
            'retained_owned_label':int(labels[cell]),'original_owned_label':int(scene.labels.ravel()[cell]),
            'source_brain_envelope_membership':bool(case.compartments['brain'][tuple(ijk)]),
            'source_estimated_skull_membership':bool(case.compartments['estimated_skull'][tuple(ijk)]),
            'already_removed':cell in state['removed']}
    for i in selected:
        row=rows[i];cover=envelopes(TOOLS['craniotome'],{'radiusMm':1.,'depthMm':2.},row['pose']['position'],row['pose']['quaternion'])
        actual={'working':set(),'shaft':set()}
        for a,b,r,part in cover:
            indices=capsule_voxel_indices(grid,a,b,r)
            if len(indices):actual[part].update(int(c)for c in np.ravel_multi_index(indices.T,scene.shape)if labels[c]==1)
        for part in('working','shaft'):
            if actual[part]!=set(row['static_contacts'][part]['brain']):raise ValueError('Frozen endpoint envelope contacts changed')
        associated=[c for c in remaining if i in endpoint_map[str(c)]]
        contacts=sorted(actual['working']|actual['shaft']);brain_union.update(contacts)
        for cell in contacts:voxel_associations.setdefault(cell,[]).append(i)
        values=[voxel_fact(c)['stored_baseline_CTA_scaled_intensity']for c in contacts]
        facts.append({'candidate_index':i,'pose':row['pose'],'depth_mm':row['depth_mm'],'offset_u_mm':row['offset_u_mm'],
            'offset_v_mm':row['offset_v_mm'],'roll_deg':row['roll_deg'],'associated_remaining_bone_blockers':associated,
            'working_brain_envelope_cells':sorted(actual['working']),'shaft_brain_envelope_cells':sorted(actual['shaft']),
            'source_contact_voxel_facts':[voxel_fact(c)for c in contacts],
            'stored_intensity_summary':{'minimum':min(values),'median':float(np.median(values)),'maximum':max(values)},
            'material_guard_fact':'Actual uninflated stock crossbar/shaft envelope intersects retained region brain; no CSF or parenchyma identity is established.'})
    atomic_json(directory/'candidate-envelope-contacts.json',facts)
    union=[{**voxel_fact(c),'candidate_indices':voxel_associations[c]}for c in sorted(brain_union)]
    atomic_json(directory/'unique-envelope-source-voxels.json',union)
    with(directory/'unique-envelope-source-voxels.csv').open('w',newline='')as handle:
        writer=csv.writer(handle);writer.writerow(['source_cell_id','i','j','k','R_mm','A_mm','S_mm','stored_baseline_CTA_scaled_intensity','source_brain_envelope','source_estimated_skull','already_removed','candidate_indices'])
        for item in union:writer.writerow([item['source_cell_id'],*item['native_ijk'],*item['ras_mm'],item['stored_baseline_CTA_scaled_intensity'],item['source_brain_envelope_membership'],item['source_estimated_skull_membership'],item['already_removed'],';'.join(map(str,item['candidate_indices']))])
    fig,axes=plt.subplots(len(facts),3,figsize=(12,3*len(facts)),squeeze=False)
    flatmask=np.zeros(scene.shape,bool).ravel();flatmask[list(brain_union)]=True;contactmask=flatmask.reshape(scene.shape)
    brain=case.compartments['brain'];skull=case.compartments['estimated_skull']
    for row_number,fact in enumerate(facts):
        ids=np.array(fact['working_brain_envelope_cells']);ijk=np.array(np.unravel_index(ids,scene.shape)).T
        middle=ijk[np.argmin(np.sum((ijk-ijk.mean(0))**2,axis=1))]
        cover=envelopes(TOOLS['craniotome'],{'radiusMm':1.,'depthMm':2.},fact['pose']['position'],fact['pose']['quaternion'])
        for ax,fixed in zip(axes[row_number],[2,1,0]):
            free=[a for a in range(3)if a!=fixed];lo=np.maximum(0,middle-np.ceil(4./np.array(case.spacing_mm)).astype(int));hi=np.minimum(np.array(scene.shape)-1,middle+np.ceil(4./np.array(case.spacing_mm)).astype(int))
            aa,bb=np.meshgrid(np.arange(lo[free[0]],hi[free[0]]+1),np.arange(lo[free[1]],hi[free[1]]+1));indices=np.zeros((*aa.shape,3),int);indices[:,:,fixed]=middle[fixed];indices[:,:,free[0]]=aa;indices[:,:,free[1]]=bb
            index_tuple=tuple(np.moveaxis(indices,-1,0));world=case.voxel_to_world(indices.reshape(-1,3));extent=[lo[free[0]]-.5,hi[free[0]]+.5,lo[free[1]]-.5,hi[free[1]]+.5]
            ax.imshow(case.mri[index_tuple],origin='lower',cmap='gray',vmin=-20,vmax=150,extent=extent,interpolation='nearest')
            for mask,color in((brain,'lime'),(skull,'cyan')):
                plane=mask[index_tuple]
                if plane.any()and not plane.all():ax.contour(aa,bb,plane,[.5],colors=[color],linewidths=.6)
            own=np.zeros(scene.shape,bool).ravel();own[fact['working_brain_envelope_cells']]=True;plane=own.reshape(scene.shape)[index_tuple]
            ax.imshow(np.ma.masked_where(~plane,plane),origin='lower',cmap='autumn',vmin=0,vmax=1,alpha=.55,extent=extent,interpolation='nearest')
            for a,b,r,part in cover:
                signed=(point_segment_distances(world,np.array(a),np.array(b))-r).reshape(aa.shape)
                if signed.min()<=0<=signed.max():ax.contour(aa,bb,signed,[0.],colors=['blue'],linewidths=.65)
            ax.set_title(f"Candidate {fact['candidate_index']}, {len(fact['working_brain_envelope_cells'])} exact envelope contacts; IJK axis {fixed}={middle[fixed]}",fontsize=7)
            ax.set_xlabel(f'native index {free[0]}',fontsize=7);ax.set_ylabel(f'native index {free[1]}',fontsize=7)
    fig.suptitle('CTA012 unchanged baseline source / exact unsafe stock-kerf endpoint contacts. Orange: recorded brain-envelope voxels; green envelope boundary; cyan skull; blue exact tool section.\nDisplay window -20..150 stored scaled units only; no source threshold or tissue classification. Intracranial envelope includes CSF; these views do not establish parenchyma/CSF identity.',fontsize=8)
    fig.tight_layout(rect=(0,0,1,.96));fig.savefig(directory/'source-orthogonal-envelope-contacts.png',dpi=150);plt.close(fig)
    values=[item['stored_baseline_CTA_scaled_intensity']for item in union]
    output={'schema':'read-only-remaining-four-blocker-source-facts-v1','source_path':str(source.relative_to(root)),
        'source_sha256':sha256_file(source),'source_state_id':snapshot['stateId'],'native_action_count':44,
        'source_image_array_digest':array_digest(case.mri),'source_owned_label_array_digest':array_digest(scene.labels),
        'source_brain_envelope_array_digest':array_digest(brain),'source_affine':case.affine.tolist(),'source_shape':list(scene.shape),
        'remaining_bone_blocker_source_facts':[voxel_fact(c)for c in remaining],'blocker_to_candidate_indices':endpoint_map,
        'unsafe_endpoint_count':len(selected),'unique_brain_envelope_contact_voxel_count':len(union),
        'stored_intensity_summary':{'minimum':min(values),'median':float(np.median(values)),'maximum':max(values)},
        'intensity_role':'Exact float values from unchanged stored acquired baseline CTA array; scaled source intensities, no new normalization, threshold or clinical tissue interpretation.',
        'facts':'These are retained estimated intracranial-envelope voxels, including possible CSF, owned as brain by the existing source mapping. Exact whole stock craniotome endpoints contact them and the existing material guard treats them as unsupported.',
        'unknowns':['No reviewed source-derived CSF versus parenchyma classification for these cells.','CT scaled intensity and these local views alone do not establish clinical tissue identity.','A future separately reviewed source-derived parenchyma/CSF preparation could test whether a more specific estimate is supported; these facts do not authorize any relabeling or promise an open surgical route.'],
        'all_source_masks_and_material_mapping_preserved':True,'source_state_preserved':True,'new_public_calls':0,
        'no_reclassification_thresholding_model_reopen_or_physical_search':True,'diagnosis_driver_sha256':sha256_file(__file__)}
    atomic_json(directory/'report.json',output)
    return{'report':str(directory/'report.json'),'sha256':sha256_file(directory/'report.json'),
        'unsafe_endpoint_count':len(selected),'unique_envelope_voxel_count':len(union),'intensity_summary':output['stored_intensity_summary']}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--replay');parser.add_argument('--actions');parser.add_argument('--diagnosis')
    parser.add_argument('--execute-protocol');parser.add_argument('--decision')
    parser.add_argument('--diagnose-reentry',action='store_true')
    parser.add_argument('--diagnose-remaining-port',action='store_true')
    parser.add_argument('--certify-fixed-kerf',action='store_true')
    parser.add_argument('--export-endpoint-envelope-facts',action='store_true')
    parser.add_argument('--root');parser.add_argument('--name',default='012-channel-diagnosis-v1');args=parser.parse_args()
    if args.export_endpoint_envelope_facts:result=export_endpoint_envelope_facts(root=args.root,name=args.name)
    elif args.certify_fixed_kerf:result=certify_fixed_kerf(root=args.root,name=args.name)
    elif args.diagnose_remaining_port:result=diagnose_remaining_port(root=args.root,name=args.name)
    elif args.diagnose_reentry:result=diagnose_reentry(root=args.root,name=args.name)
    elif args.execute_protocol:
        if not args.decision:parser.error('--decision required for execution')
        result=execute_protocol(args.execute_protocol,args.decision,root=args.root,name=args.name)
    elif args.diagnosis:result=make_protocol(args.diagnosis,root=args.root,name=args.name)
    else:
        if not args.replay or not args.actions:parser.error('--replay and --actions required for diagnosis')
        result=diagnose(args.replay,args.actions,root=args.root,name=args.name)
    print(json.dumps(result,indent=2))

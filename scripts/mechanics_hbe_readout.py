"""Read one completed specimen run from hash-bound primitives, without solving."""
from __future__ import annotations

from contextlib import ExitStack
import itertools
import math

import numpy as np

from scripts.mechanics_hbe_access import local_path, verify_binding, expected_runs
from scripts.mechanics_hbe_outputs import iter_data_records, check_solver_records
from scripts.mechanics_hbe_physics import HexMesh, fixed_probes, energy_work_check, compare_refinement


def read_run(root, bindings, *, protocol_sha256, expected_branch, expected_mesh_N,
             expected_steps, expected_mu_Pa, retain_scale_primitives=False):
    """Return (JSON-compatible numerical receipt, optional primitive arrays).

    The caller fixes identity before a run. This function rejects mismatched
    loading/deck/source identities and exhausts both strict record generators.
    """
    if set(bindings)!={'mesh','deck','loading','nodes','elements','solver'}:
        raise ValueError('Exactly six bound run inputs required')
    for binding in bindings.values():verify_binding(root,binding)
    metadata=verify_binding(root,bindings['mesh'],maximum_bytes=16*1024**2,read_json=True)
    loading=verify_binding(root,bindings['loading'],maximum_bytes=1024**2,read_json=True)
    identity=(loading.get('branch'),metadata.get('mesh_N'),loading.get('steps'),loading.get('mu_Pa'))
    if identity!=(expected_branch,expected_mesh_N,expected_steps,expected_mu_Pa):
        raise ValueError('Run differs from prospectively requested identity')
    if expected_branch not in ('compression','tension','torsion_neg','torsion_pos') or expected_mesh_N not in (4,8,12) or expected_steps not in (60,120):
        raise ValueError('Run outside frozen specimen task')
    if not math.isfinite(expected_mu_Pa) or expected_mu_Pa<=0:
        raise ValueError('Positive finite modulus required')
    if loading.get('protocol_sha256')!=protocol_sha256 or loading.get('mesh_sha256')!=bindings['mesh']['sha256'] or loading.get('deck_sha256')!=bindings['deck']['sha256']:
        raise ValueError('Loading metadata is not bound to protocol/mesh/deck')
    if loading.get('node_fields')!='x;y;z;ux;uy;uz;Rx;Ry;Rz' or loading.get('element_fields')!='sx;sy;sz;sxy;syz;sxz;J;sed' or loading.get('raw_reaction_convention')!='body_on_constraint':
        raise ValueError('Primitive units/order/reaction convention differs')
    mesh=HexMesh.from_manifest(metadata);X=mesh.rest_nodes_m
    if mesh.radius_m!=.004 or mesh.height_m!=.00489159:
        raise ValueError('Specimen geometry differs from fixed declaration')
    if loading.get('prescribed_dofs')!={'top':'xyz','bottom':'xyz'} or not math.isclose(loading.get('K_Pa',0),149*expected_mu_Pa/3,rel_tol=1e-14):
        raise ValueError('Material or prescribed fixture differs')
    times=np.linspace(0,1,expected_steps+1)
    saved_times=np.asarray(loading.get('times',[]),float)
    if saved_times.shape!=times.shape or not np.isfinite(saved_times).all() or not np.allclose(saved_times,times,rtol=0,atol=2e-15):
        raise ValueError('Loading time grid differs')
    sign=-1 if expected_branch in ('compression','torsion_neg') else 1
    endpoint=sign*.15*mesh.height_m/(mesh.radius_m if expected_branch.startswith('torsion') else 1)
    coordinate=times*endpoint;saved_coordinate=np.asarray(loading.get('load_coordinate',[]),float)
    if saved_coordinate.shape!=coordinate.shape or not np.isfinite(saved_coordinate).all() or not np.allclose(saved_coordinate,coordinate,rtol=2e-14,atol=1e-18):
        raise ValueError('Physical loading coordinate differs')
    if loading.get('load_coordinate_units')!=('rad' if expected_branch.startswith('torsion') else 'm'):
        raise ValueError('Physical loading units differ')
    mu=expected_mu_Pa;R=mesh.radius_m;F0=mu*R*R;T0=F0*R
    # Preserve the declared deck arithmetic for arbitrary fitted scales too.
    residual_floor=(1e-10*mu*R**2)**2
    if loading.get('min_residual_N2')!=residual_floor or not math.isfinite(residual_floor) or residual_floor<=0:
        raise ValueError('Declared residual floor differs')
    with local_path(root,bindings['solver']['path']).open(encoding='utf-8',errors='strict') as stream:
        solver=check_solver_records(stream,expected_times=times,residual_floor_N2=residual_floor)
    probe_map=mesh.probe_map(fixed_probes(R,mesh.height_m))
    forces=[];torques=[];energies=[];probes=[];minimum_J=math.inf
    ratios={name:0. for name in ('force_balance','moment_balance','prescribed_motion','primitive_consistency')}
    raw_current=[];raw_reactions=[];diagnostics=[]
    with ExitStack() as stack:
        node_stream=stack.enter_context(local_path(root,bindings['nodes']['path']).open(encoding='utf-8',errors='strict'))
        element_stream=stack.enter_context(local_path(root,bindings['elements']['path']).open(encoding='utf-8',errors='strict'))
        node_records=iter_data_records(node_stream,expected_times=times,item_count=len(X),field_count=9,record_name='mechanics_nodes_si')
        element_records=iter_data_records(element_stream,expected_times=times,item_count=mesh.element_count,field_count=8,record_name='mechanics_elements_si')
        for nodes,elements in itertools.zip_longest(node_records,element_records):
            if nodes is None or elements is None or nodes['step']!=elements['step']:
                raise ValueError('Unsynchronized complete primitive states required')
            if np.any(elements['values'][:, 6] <= 0):
                raise ValueError('Nonpositive logged element Jacobian conflicts with valid deformation')
            n=nodes['values'];current=n[:,:3];raw=n[:,6:9];fraction=nodes['declared_time']
            state=mesh.read_frame(current,raw,mu_Pa=mu,branch=expected_branch,fraction=fraction)
            motion_error=float(np.max(np.linalg.norm(n[:,3:6]-(current-X),axis=1)))
            consistency_ratio=motion_error/(1e-8*R)
            if fraction==0:
                consistency_ratio=max(consistency_ratio,float(np.max(np.linalg.norm(current-X,axis=1)))/(1e-8*R),float(np.max(np.linalg.norm(raw,axis=1)))/(1e-8*F0))
            consistency_ratio=max(consistency_ratio,state['free_node_reaction_max_N']/(1e-8*F0))
            moments=np.cross(current,raw)
            ratios['force_balance']=max(ratios['force_balance'],np.linalg.norm(state['net_force_N'])/(1e-8*F0+1e-5*np.linalg.norm(raw,axis=1).sum()))
            ratios['moment_balance']=max(ratios['moment_balance'],np.linalg.norm(state['net_moment_Nm'])/(1e-8*T0+1e-5*np.linalg.norm(moments,axis=1).sum()))
            ratios['prescribed_motion']=max(ratios['prescribed_motion'],state['prescribed_error_m']/(1e-8*R))
            ratios['primitive_consistency']=max(ratios['primitive_consistency'],consistency_ratio)
            minimum_J=min(minimum_J,state['minimum_sampled_J'])
            forces.append(state['applied_force_N']);torques.append(state['applied_torque_Nm']);energies.append(state['energy_J'])
            probes.append(mesh.interpolate_displacement(current,probe_map).tolist())
            diagnostics.append({'step':nodes['step'],'logged_J_min':float(elements['values'][:,6].min()),
                                'logged_sed_min_Pa':float(elements['values'][:,7].min()),'logged_sed_max_Pa':float(elements['values'][:,7].max())})
            if retain_scale_primitives:
                raw_current.append(current.copy());raw_reactions.append(raw.copy())
    response=torques if expected_branch.startswith('torsion') else forces
    work=energy_work_check(coordinate,response,energies,mu_Pa=mu,radius_m=R,height_m=mesh.height_m)
    ratios['work_energy']=work['maximum_error_J']/work['limit_J']
    ratios['solver_residual']=max(row['actual_N2']/row['limit_N2'] for row in solver['states'])
    if not all(math.isfinite(float(v)) for v in ratios.values()):raise ValueError('Nonfinite numerical criterion')
    criteria={name:{'actual':float(value),'limit':1.,'units':'ratio_to_declared_numerical_tolerance'} for name,value in ratios.items()}
    criteria['minimum_sampled_J']={'actual':minimum_J,'limit':0.,'units':'dimensionless_strictly_positive'}
    receipt={'schema':'hbe-run-readout-v1','branch':expected_branch,'mesh_N':expected_mesh_N,'steps':expected_steps,
             'mu_Pa':mu,'frame_count':len(forces),'protocol_sha256':protocol_sha256,'primitive_bindings':bindings,
             'load_coordinate':coordinate.tolist(),'applied_force_N':forces,'applied_torque_Nm':torques,
             'energy_J':energies,'probe_displacements_m':probes,'criteria':criteria,
             'passed':all(value<=1 for value in ratios.values()) and minimum_J>0,
             'solver':solver,'energy_work':work,'diagnostics':diagnostics,
             'logged_sed_used_for_energy_gate':False,'sampled_J_positivity_is_not_everywhere_proof':True}
    for binding in bindings.values():verify_binding(root,binding)
    cache={'mesh':mesh,'current_nodes_m':np.array(raw_current),'raw_reactions_N':np.array(raw_reactions)} if retain_scale_primitives else None
    return receipt,cache


def _refinement_group(coarse,medium,fine,*,kind):
    branch=fine['branch'];field='applied_torque_Nm' if branch.startswith('torsion') else 'applied_force_N'
    unit='Nm' if branch.startswith('torsion') else 'N';R=.004;mu=1000.
    scale=mu*R**(3 if unit=='Nm' else 2)
    def sample(receipt):
        stride=receipt['steps']//60
        return {'response':np.asarray(receipt[field])[::stride],
                'probe_displacements_m':np.asarray(receipt['probe_displacements_m'])[::stride]}
    result=compare_refinement(sample(coarse),sample(medium),sample(fine),response_scale=scale,radius_m=R,kind=kind)
    metrics={'reaction':{'actual':result['response_change'],'limit':result['response_limit'],'units':unit},
             'motion':{'actual':result['motion_change_m'],'limit':result['motion_limit_m'],'units':'m'}}
    if kind=='mesh':
        for name,change,previous,floor,units in (
            ('reaction_trend',result['response_change'],result['coarse_response_change'],1e-5*scale,unit),
            ('motion_trend',result['motion_change_m'],result['coarse_motion_change_m'],result['motion_limit_m'],'m')):
            if max(change,previous)<=floor:
                metrics[name]={'actual':max(change,previous),'limit':floor,'units':units}
            else:
                metrics[name]={'actual':change,'limit':previous,'units':units,'comparison':'lt'}
    return metrics


def _scale_group(reference,scaled,reference_cache,scaled_cache):
    if reference_cache is None or scaled_cache is None or reference_cache['mesh'].fingerprint!=scaled_cache['mesh'].fingerprint:
        raise ValueError('Matching immutable scale-pair mesh primitives required')
    mesh=reference_cache['mesh'];X=mesh.rest_nodes_m
    x1,x2,r1,r2=(np.asarray(value,dtype=float) for value in (reference_cache['current_nodes_m'],scaled_cache['current_nodes_m'],reference_cache['raw_reactions_N'],scaled_cache['raw_reactions_N']))
    if any(v.shape!=(121,len(X),3) or not np.isfinite(v).all() for v in (x1,x2,r1,r2)):
        raise ValueError('Complete fine120 scale-pair primitives required')
    factor=scaled['mu_Pa']/reference['mu_Pa'];R=mesh.radius_m
    if not math.isfinite(factor) or factor<=0:raise ValueError('Positive finite modulus ratio required')
    motion_limit=1e-6*R+1e-5*float(np.max(np.abs(x1-X)))
    reaction_limit=1e-7*scaled['mu_Pa']*R**2+1e-4*np.abs(factor*r1)
    t1=np.asarray(reference['applied_torque_Nm']);t2=np.asarray(scaled['applied_torque_Nm'])
    torque_limit=1e-7*scaled['mu_Pa']*R**3+1e-4*np.abs(factor*t1)
    metrics={'motion':{'actual':float(np.max(np.abs(x2-x1))),'limit':motion_limit,'units':'m'},
             'reaction':{'actual':float(np.max(np.abs(r2-factor*r1)/reaction_limit)),'limit':1.,'units':'ratio_to_declared_scale_tolerance'},
             'torque':{'actual':float(np.max(np.abs(t2-factor*t1)/torque_limit)),'limit':1.,'units':'ratio_to_declared_scale_tolerance'}}
    if not all(math.isfinite(row['actual']) and math.isfinite(row['limit']) for row in metrics.values()):raise ValueError('Nonfinite scale criterion')
    return metrics


def build_numerical_evidence(runs,caches,*,source_bindings,protocol_sha256,fitted_mu_Pa=None):
    """Build measured numerical comparisons for 18 pre-fit or all 20 run receipts.

    Write this result, then call the appropriate access-layer validator. Failed
    numeric comparisons remain in the result for durable negative reporting.
    """
    required=expected_runs()
    if fitted_mu_Pa is None:required={k:v for k,v in required.items() if v[-1]!='fitted'}
    if set(runs)!=set(required):raise ValueError('Wrong declared run inventory')
    for key,(branch,N,steps,role) in required.items():
        row=runs[key];mu={'reference':1000.,'double_mu':2000.,'fitted':fitted_mu_Pa}[role]
        if (row['branch'],row['mesh_N'],row['steps'],row['mu_Pa'],row['frame_count'])!=(branch,N,steps,mu,steps+1) or row['protocol_sha256']!=protocol_sha256:
            raise ValueError('Run receipt identity differs')
    report={'schema':'hbe-numerical-evidence-v1','protocol_sha256':protocol_sha256,'source_bindings':source_bindings,
            'runs':runs,'mesh':{},'step':{},'scale':{}}
    for branch in ('compression','tension','torsion_neg','torsion_pos'):
        coarse,medium,fine=(runs[f'{branch}:N{N}:S60:reference'] for N in (4,8,12))
        report['mesh'][branch]=_refinement_group(coarse,medium,fine,kind='mesh')
        refined=runs[f'{branch}:N12:S120:reference']
        report['step'][branch]=_refinement_group(fine,fine,refined,kind='step')
    for branch in ('compression','torsion_pos'):
        ref=f'{branch}:N12:S120:reference';double=f'{branch}:N12:S120:double_mu'
        report['scale'][branch]=_scale_group(runs[ref],runs[double],caches[ref],caches[double])
    if fitted_mu_Pa is not None:
        report['fitted_confirmation']={}
        for branch in ('compression','tension'):
            ref=f'{branch}:N12:S120:reference';fitted=f'{branch}:N12:S120:fitted'
            report['fitted_confirmation'][branch]=_scale_group(runs[ref],runs[fitted],caches[ref],caches[fitted])
    return report

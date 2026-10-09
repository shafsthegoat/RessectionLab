"""Actual-modulus, one-frame-at-a-time fitted confirmations on fixed saved meshes."""
from contextlib import ExitStack
import itertools
import math
from pathlib import Path
import time
import numpy as np
from scripts import mechanics_hbe_branch_calibration as core
from scripts import mechanics_hbe_outputs as outputs

common, inherited, access = core.common, core.inherited, core.access
R, H, TIMES = common.R, common.H, common.TIMES


def half_frame(model, current, raw, *, full_displacement_m, mu):
    """Original half-frame equations with explicit actual modulus, no global mutation."""
    mu = core.positive(mu)
    current = inherited.finite_array(current, model.Xh.shape)
    raw = inherited.finite_array(raw, model.Xh.shape)
    d = float(full_displacement_m)
    if not math.isfinite(d):raise ValueError('Finite displacement required')
    bottom_error = np.linalg.norm(current[model.bottom]-model.Xh[model.bottom], axis=1).max()
    mid_error = np.abs(current[model.midplane,2]-model.Xh[model.midplane,2]-d/2).max()
    free_raw = raw.copy();free_raw[model.bottom]=0;free_raw[model.midplane,2]=0
    F0, T0 = mu*model.radius**2, mu*model.radius**3
    moments = np.cross(current,raw)
    ratios = {
        'force_balance':float(np.linalg.norm(raw.sum(axis=0))/(1e-8*F0+1e-5*np.linalg.norm(raw,axis=1).sum())),
        'moment_balance':float(np.linalg.norm(moments.sum(axis=0))/(1e-8*T0+1e-5*np.linalg.norm(moments,axis=1).sum())),
        'prescribed_motion':float(max(bottom_error,mid_error)/(1e-8*model.radius)),
        'free_dof_reaction':float(np.linalg.norm(free_raw,axis=1).max()/(1e-8*F0))}
    return dict(model.half.deformation(current,mu), force_from_bottom_N=float(raw[model.bottom,2].sum()),
                force_from_midplane_N=float(-raw[model.midplane,2].sum()), ratios=ratios)


def frame_ratios(mesh,state,raw,*,mu):
    F0 = core.positive(mu)*mesh.radius_m**2
    T0 = F0*mesh.radius_m
    moments = np.cross(state['current_nodes_m'],raw)
    return {'force_balance':np.linalg.norm(state['net_force_N'])/(1e-8*F0+1e-5*np.linalg.norm(raw,axis=1).sum()),
            'moment_balance':np.linalg.norm(state['net_moment_Nm'])/(1e-8*T0+1e-5*np.linalg.norm(moments,axis=1).sum()),
            'prescribed_motion':state['prescribed_error_m']/(1e-8*mesh.radius_m),
            'free_node_reaction':state['free_node_reaction_max_N']/(1e-8*F0)}


def records(stack,root,binding,count,fields,name):
    allowed={(39610,9,'mechanics_nodes_si'):768*1024**2,(34992,8,'mechanics_elements_si'):512*1024**2,
             (12439,9,'mechanics_nodes_si'):256*1024**2,(10368,8,'mechanics_elements_si'):256*1024**2}
    key=(count,fields,name)
    if key not in allowed:raise ValueError('Exact declared half primitive dimensions required')
    stream=stack.enter_context(access.local_path(root,binding['path']).open(encoding='utf-8',errors='strict'))
    return outputs._iter_data_records(stream,expected_times=TIMES,item_count=count,field_count=fields,
                                     record_name=name,maximum_bytes=allowed[key],maximum_items=count)


def stream_frames(root, half_bindings, model, *, branch, mu, deadline):
    """Accepted 121-state half/full equations, one native frame at a time."""
    mu = core.positive(mu)
    if branch not in ('compression','tension'):
        raise ValueError('Axial half-height branch required')
    coordinate = TIMES * (-1 if branch == 'compression' else 1) * .15 * H
    with access.local_path(root, half_bindings['solver']['path']).open(encoding='utf-8', errors='strict') as stream:
        solver = inherited.check_solver_records(stream, expected_times=TIMES,
                                                residual_floor_N2=(1e-10*mu*R**2)**2)
    probe_map = model.full.probe_map(inherited.fixed_probes(R, H))
    half_ratios, full_ratios, reconstruction_ratios = {}, {}, {}
    half_energy, bottom_force, midplane_force = [], [], []
    full_energy, full_force, full_torque, full_probes = [], [], [], []
    minimum_half_J = minimum_full_J = math.inf
    shared_error = 0.
    F0, E0 = mu*R**2, mu*R**2*H
    with ExitStack() as stack:
        streams = (
            records(stack, root, half_bindings['nodes'], len(model.Xh), 9, 'mechanics_nodes_si'),
            records(stack, root, half_bindings['elements'], model.half.element_count, 8, 'mechanics_elements_si'))
        for hn, he in itertools.zip_longest(*streams):
            if hn is None or he is None or hn['step'] != he['step']:
                raise ValueError('Two complete synchronized native primitive streams required')
            common.remaining_seconds(deadline, 600)
            index = hn['step']
            if index != len(full_force) or np.any(he['values'][:, 6] <= 0):
                raise ValueError('Out-of-order state or nonpositive logged element Jacobian')
            n = hn['values']
            current, raw = n[:, :3], n[:, 6:9]
            half = half_frame(model, current, raw, full_displacement_m=coordinate[index], mu=mu)
            lifted = model.lift(current, raw, he['values'], coordinate[index])
            full = model.full.read_frame(lifted['current_nodes_m'], lifted['raw_reactions_N'],
                                         mu_Pa=mu, branch=branch, fraction=TIMES[index])
            full['current_nodes_m'] = lifted['current_nodes_m']
            inherited._maxima(half_ratios, half['ratios'])
            inherited._maxima(full_ratios, frame_ratios(model.full, full, lifted['raw_reactions_N'], mu=mu))
            consistency = float(np.linalg.norm(n[:, 3:6]-(current-model.Xh), axis=1).max())/(1e-8*R)
            if index == 0:
                consistency = max(consistency,
                    float(np.linalg.norm(current-model.Xh, axis=1).max())/(1e-8*R),
                    float(np.linalg.norm(raw, axis=1).max())/(1e-8*F0))
            inherited._maxima(half_ratios, {'primitive_consistency': consistency})
            # These are internal half/full reconstruction identities, with the
            # accepted equivalence tolerances; no external native-full run exists.
            force_limit = 1e-7*F0 + 1e-4*abs(full['applied_force_N'])
            energy_limit = 1e-7*E0 + 1e-4*abs(full['energy_J'])
            inherited._maxima(reconstruction_ratios, {
                'bottom_force_scale_one': abs(half['force_from_bottom_N']-full['applied_force_N'])/force_limit,
                'midplane_force_scale_one': abs(half['force_from_midplane_N']-full['applied_force_N'])/force_limit,
                'half_energy_scale_two': abs(2*half['energy_J']-full['energy_J'])/energy_limit})
            shared_error = max(shared_error, lifted['maximum_shared_displacement_mismatch_m'])
            minimum_half_J = min(minimum_half_J, half['minimum_sampled_J'])
            minimum_full_J = min(minimum_full_J, full['minimum_sampled_J'])
            half_energy.append(half['energy_J'])
            bottom_force.append(half['force_from_bottom_N'])
            midplane_force.append(half['force_from_midplane_N'])
            full_energy.append(full['energy_J'])
            full_force.append(full['applied_force_N'])
            full_torque.append(full['applied_torque_Nm'])
            full_probes.append(model.full.interpolate_displacement(lifted['current_nodes_m'], probe_map).tolist())
            yield {'native_current':current, 'native_raw':raw,
                   'current':lifted['current_nodes_m'], 'raw':lifted['raw_reactions_N'],
                   'torque':full['applied_torque_Nm'], 'index':index}
    if len(full_force) != 121:
        raise ValueError('All initial and converged native states are required')
    half_work = inherited.energy_work_check(coordinate/2, midplane_force, half_energy,
                                            mu_Pa=mu, radius_m=R, height_m=H/2)
    full_work = inherited.energy_work_check(coordinate, full_force, full_energy,
                                            mu_Pa=mu, radius_m=R, height_m=H)
    inherited._maxima(half_ratios, {
        'work_energy': half_work['maximum_error_J']/half_work['limit_J'],
        'solver_residual': max(row['actual_N2']/row['limit_N2'] for row in solver['states'])})
    inherited._maxima(full_ratios, {'work_energy': full_work['maximum_error_J']/full_work['limit_J']})
    half_pass = all(v <= 1 for v in half_ratios.values()) and minimum_half_J > 0
    full_pass = all(v <= 1 for v in full_ratios.values()) and minimum_full_J > 0
    reconstruction_pass = all(v <= 1 for v in reconstruction_ratios.values())
    return {
        'branch': branch, 'steps': 120, 'mu_Pa': mu, 'frame_count': 121,
        'passed': half_pass and full_pass and reconstruction_pass,
        'half_native': {'passed': half_pass, 'criteria': inherited._criteria(half_ratios, minimum_half_J),
                        'solver': solver, 'energy_work': half_work, 'energy_J': half_energy,
                        'force_from_bottom_N': bottom_force, 'force_from_midplane_N': midplane_force,
                        'load_coordinate_m': (coordinate/2).tolist()},
        'reconstructed_full': {'passed': full_pass, 'criteria': inherited._criteria(full_ratios, minimum_full_J),
                               'provenance': 'reflection_of_native_half_fields_not_a_full_native_solve',
                               'native_full_solver_claim': False},
        'reconstruction_consistency': {'passed': reconstruction_pass,
            'criteria': {key: {'actual': value, 'limit': 1., 'units': 'ratio_to_accepted_reconstruction_tolerance'}
                         for key, value in reconstruction_ratios.items()},
            'maximum_shared_displacement_mismatch_m': shared_error},
        'full_native_equivalence_evaluated': False,
        # Flat full-domain quantities use the original refinement function;
        # their provenance remains explicitly reconstructed above.
        'applied_force_N': full_force, 'applied_torque_Nm': full_torque,
        'probe_displacements_m': full_probes, 'energy_J': full_energy,
        'energy_work': full_work, 'load_coordinate_m': coordinate.tolist(),
        'measured_data_accessed': False, 'physical_validation_pass': None,
        'logged_sed_used_for_energy_gate': False, 'sampled_J_positivity_is_not_everywhere_proof': True}



def model_for(context,branch):
    reg=context['registry'];q=context['study']['axial_references'][branch]
    full=reg.read(q['full_mesh']);half=reg.read(q['native_primitives']['mesh']);wrapper=reg.read(q['reconstruction'])
    model=inherited.HalfHeightReconstruction(full,half,wrapper['mapping'])
    if (len(model.Xh),model.half.element_count)!=(q['counts']['nodes'],q['counts']['elements']):
        raise ValueError('Bound native model count differs')
    return model


class ScaleReduction:
    """Original component-wise scale maxima, streamed without full-state caches."""
    def __init__(self,mu,radius=R):
        self.mu=core.positive(mu);self.factor=self.mu/1000.;self.radius=radius
        self.motion=0.;self.reference_motion=0.;self.reaction=0.;self.torque=0.;self.count=0

    def update(self,reference,fitted,rest):
        if reference['index']!=self.count or fitted['index']!=self.count:
            raise ValueError('Missing or out-of-order scale-pair state')
        for field in ('current','raw'):
            if reference[field].shape != rest.shape or fitted[field].shape != rest.shape:
                raise ValueError('Scale-pair node identity/count differs')
            if not np.isfinite(reference[field]).all() or not np.isfinite(fitted[field]).all():
                raise ValueError('Nonfinite scale-pair primitive')
        x1,x2,r1,r2=(reference['current'],fitted['current'],reference['raw'],fitted['raw'])
        self.motion=max(self.motion,float(np.max(np.abs(x2-x1))))
        self.reference_motion=max(self.reference_motion,float(np.max(np.abs(x1-rest))))
        limit=1e-7*self.mu*self.radius**2+1e-4*np.abs(self.factor*r1)
        self.reaction=max(self.reaction,float(np.max(np.abs(r2-self.factor*r1)/limit)))
        t1,t2=reference['torque'],fitted['torque']
        self.torque=max(self.torque,abs(t2-self.factor*t1)/(1e-7*self.mu*self.radius**3+1e-4*abs(self.factor*t1)))
        self.count+=1

    def finish(self):
        if self.count!=121:raise ValueError('Complete121-state scale pair required')
        metrics={'motion':{'actual':self.motion,'limit':1e-6*self.radius+1e-5*self.reference_motion,'units':'m'},
                 'reaction':{'actual':self.reaction,'limit':1.,'units':'ratio_to_declared_scale_tolerance'},
                 'torque':{'actual':self.torque,'limit':1.,'units':'ratio_to_declared_scale_tolerance'}}
        if not all(math.isfinite(x) for row in metrics.values() for x in (row['actual'],row['limit'])):
            raise ValueError('Nonfinite scale-pair criterion')
        return metrics


def verify_fitted(context,branch,binding,mu):
    reg=context['registry'];study=context['study'];record=reg.bound(binding)
    expected_directory=context['directory']/'runs'/branch
    q=study['axial_references'][branch];primitives=record['primitive_bindings']
    if (record.get('schema')!='hbe-branch-fitted-execution-v1' or record.get('branch')!=branch
            or record.get('mu_Pa')!=mu or record.get('run_id')!=f'{branch}:N{q["counts"]["N"]}:S120:fitted'
            or record.get('declaration_sha256')!=core.DECLARATION_SHA256
            or record.get('release')!=context['release_binding']
            or record.get('runtime_identity')!=study['runtime_identity']
            or record.get('backend_profile')!=study['backend_profile']
            or record.get('reconstruction')!=q['reconstruction']
            or primitives.get('mesh')!=q['native_primitives']['mesh']):
        raise ValueError('Actual fitted execution identity differs')
    command=[context['context']['runtime']['executable'],'-noconfig','-no_title','-i','specimen.feb','-o','solver.log']
    if record.get('command')!=command or record['execution'].get('command')!=command or Path(record['execution']['cwd'])!=expected_directory:
        raise ValueError('Actual fitted invocation differs')
    core.receipts.require_completed_native(record['execution'],study['budgets']['native_seconds_by_branch'][branch])
    reg.primitives(branch,primitives,fitted_directory=expected_directory)
    for b in primitives.values():reg.read(b,json_value=False)
    skyline,deck,loading=core.fitted_contents(reg,study,branch,mu)
    source=record['backend_source_deck'];reg.bound(source,json_value=False)
    if (access.local_path(context['root'],source['path'])!=expected_directory/'skyline.feb'
            or access.local_path(context['root'],source['path']).read_text()!=skyline
            or access.local_path(context['root'],primitives['deck']['path']).read_text()!=deck
            or reg.read(primitives['loading'])!=loading):
        raise ValueError('Fitted deck/loading differs from exact three-scalar change')
    return record


def paired_evidence(context,fitted_runs,mu,deadline):
    if set(fitted_runs)!=set(core.AXIAL):raise ValueError('Both actual fitted native confirmations required')
    mu=core.positive(mu);reg=context['registry'];rows={};started=time.monotonic()
    for branch in core.AXIAL:
        common.remaining_seconds(deadline,600)
        record=verify_fitted(context,branch,fitted_runs[branch],mu)
        q=context['study']['axial_references'][branch];model=model_for(context,branch)
        streams=[stream_frames(context['root'],primitives,model,branch=branch,mu=modulus,deadline=deadline)
                 for primitives,modulus in [(q['native_primitives'],1000.),(record['primitive_bindings'],mu)]]
        reductions={'native':ScaleReduction(mu),'reconstructed_full':ScaleReduction(mu)};reports=[]
        try:
            for index in range(121):
                pair=[]
                for stream in streams:
                    try:pair.append(next(stream))
                    except StopIteration as error:raise ValueError('Incomplete fitted/reference native stream') from error
                reductions['reconstructed_full'].update(pair[0],pair[1],model.X)
                native=[dict(frame,current=frame['native_current'],raw=frame['native_raw']) for frame in pair]
                reductions['native'].update(native[0],native[1],model.Xh)
            for stream in streams:
                try:next(stream)
                except StopIteration as complete:reports.append(complete.value)
                else:raise ValueError('Extra native state after121')
        finally:
            for stream in streams:stream.close()
        reference=context['references'][branch]
        if any(access.canonical_json(value)!=access.canonical_json(reference[key]) for key,value in reports[0].items()):
            raise ValueError('Independent streamed reference reproduction differs')
        metrics={name:reduction.finish() for name,reduction in reductions.items()}
        scale_pass=all(row['actual']<=row['limit'] for group in metrics.values() for row in group.values())
        rows[branch]={'execution_binding':fitted_runs[branch],'reference':q['run_readout'],
                      'readout':reports[1],'scale':metrics,'passed':reports[0]['passed'] and reports[1]['passed'] and scale_pass}
        for primitives in (q['native_primitives'],record['primitive_bindings']):
            for binding in primitives.values():reg.read(binding,json_value=False)
        del streams,model,reports,pair,native
    reg.verify_all(deadline);common.remaining_seconds(deadline,600)
    return {'schema':'hbe-branch-fitted-paired-evidence-v1','declaration_sha256':core.DECLARATION_SHA256,
            'mu_Pa':mu,'branches':rows,'passed':all(row['passed'] for row in rows.values()),
            'states_per_native_run':121,'paired_stream_seconds':time.monotonic()-started,
            'historical_failures_preserved':True,'physical_validation_pass':None,
            'no_generated_response_fixture':True,'full_native_fine_equivalence_claim':False}

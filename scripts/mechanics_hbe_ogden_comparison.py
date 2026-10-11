"""One predeclared Ogden shape probe on the already exposed DEVELOPMENT specimen.

Six native calls maximum. No archive/member reader, new donor, offset, sign fit,
coordinate editing or torsion. A two-grid diagnostic is not v5 qualification.
"""
from __future__ import annotations
from contextlib import ExitStack
from copy import deepcopy
import json
import math
import os
from pathlib import Path
import re
import time
import xml.etree.ElementTree as ET
from scripts import mechanics_hbe_v5_axial_calibration as authority

need, canonical, sha = authority.need, authority.canonical, authority.sha
local, read_binding, exclusive_json = authority.local, authority.read_binding, authority.exclusive_json
CORE = 'scripts/mechanics_hbe_ogden_comparison.py'
RUNNER = 'scripts/mechanics_hbe_ogden_comparison_experiment.py'
DECLARATION = 'manifests/experiments/hbe-01-03-ogden-shape-comparison-v1.json'
# Filled by source preparation after the declaration is frozen; not an execution grant.
DECLARATION_SHA256 = 'ae6a4d4217d7531c2c112eeed5e565548050bf69c757998f74b3ed4b720ebc2f'
OUTPUT = 'outputs/mechanics/hbe-ogden-shape-comparison-v1/attempt-01'
DONE = 'completed_development_shape_comparison'
FIT_SHA = 'b6b23a297f81b9481a6a0c7b584c8d1d84907baf86dd4ecb62b78474caf9666a'
AXIAL = ('compression', 'tension')
ALPHAS = (2., -8.)
# Coarse/fine both modes precede any S120 call; no adaptive candidate/refinement.
RUNS = (('compression',16,60),('compression',24,60),('tension',16,60),
        ('tension',24,60),('compression',24,120),('tension',24,120))
CAPS = {'total_seconds':1200,'cleanup_reserve_seconds':10,'preparation_seconds':45,
        'readout_seconds':90,'profile_seconds':45,'sampled_group_rss_bytes':1024**3,
        'new_output_bytes':2*1024**3,'run_output_bytes':576*1024**2,
        'native_seconds':[120,180,60,180,300,300], 'threads':1,'attempts':1,
        'native_calls':6,'positive_scale_profiles':2,'baseline_reproduction_calls':1,'mesher_calls':0,
        'new_measured_member_reads':0,'held_out_member_reads':0,'automatic_retry':False}


def run_name(row):
    branch,n,steps=row
    need(tuple(row) in RUNS,'predeclared branch/grid/steps only')
    return f'{branch}-N{n}-S{steps}-alpha-minus8'


def runtime(root,release):
    record=read_binding(root,release['runtime_source'],maximum=1024**2)
    return record['runtime']  # Only reuse its byte-locked Python dependency closure.


def verify_sources(root,release,declaration,*,committed):
    need(set(release['source_bindings'])==set(declaration['source_paths']), 'exact source closure')
    expanded={**release,'runtime':runtime(root,release)}
    authority.verify_sources(root,expanded,{'inherited_source_bindings':release['source_bindings']},committed=committed)


def audit_loaded(root,release):
    authority.audit_loaded(root,{**release,'runtime':runtime(root,release)})


def check_release(root,binding,*,executing):
    release=read_binding(root,binding,maximum=128*1024)
    need(set(release)=={'schema','execution_released','source_commit','source_bindings','runtime_source',
                       'declaration','output_directory','caps','execution_storage_receipt'},'release schema')
    need(release['schema']=='hbe-ogden-shape-release-v1' and type(release['execution_released']) is bool,
         'explicit shape-comparison release')
    need(not executing or release['execution_released'],'separate root execution release required')
    need(release['declaration']=={'path':DECLARATION,'sha256':DECLARATION_SHA256}
         and release['output_directory']==OUTPUT and release['caps']==CAPS,'exact declaration/output/caps')
    declaration=read_binding(root,release['declaration'],maximum=128*1024)
    need(declaration['schema']=='hbe-ogden-shape-declaration-v1' and declaration['caps']==CAPS
         and declaration['alpha_candidates']==list(ALPHAS) and declaration['ordered_new_runs']==[list(x) for x in RUNS]
         and declaration['authorized_held_out_members']==[] and declaration['specimen_role']=='DEVELOPMENT'
         and declaration['fit']['sha256']==FIT_SHA,'fixed scientific scope')
    if executing:
        need(isinstance(release['source_commit'],str) and re.fullmatch('[0-9a-f]{40}',release['source_commit']),
             'committed source required')
        # Source storage reservation is not a reservation for native outputs.
        read_binding(root,release['execution_storage_receipt'],maximum=1024**2)
    return release,declaration


def evidence(root,declaration):
    """Read only authenticated source/accepted numerical metadata here, never curves."""
    e=declaration['evidence']
    comparison=read_binding(root,e['comparison'],maximum=16*1024**2)
    need(comparison['numerical_qualification_passed'] is True
         and comparison['physical_validation_pass'] is None
         and comparison['comparison_manifest']==e['manifest'],'accepted original alpha-two evidence')
    for key in ('independent_audit','independent_report','roles','protocol'):
        read_binding(root,e[key],decode=False)
    manifest=read_binding(root,e['manifest'])
    study_raw=read_binding(root,e['v5'],decode=False)
    prior_raw=read_binding(root,e['v4'],decode=False)
    study,prior=json.loads(study_raw),json.loads(prior_raw)
    need(study['previous_v4']==e['v4'],'v5 ancestry')
    return manifest,study_raw,prior_raw,study,prior


def shape_deck(reference,contract,steps):
    """Change only m1 and, for S120, the existing time/linear-load schedule."""
    from scripts import mechanics_hbe_branch_calibration_v5 as v5
    need(contract['steps']==60 and contract['mu_Pa']==1000. and steps in (60,120),'reference S60 deck required')
    before=v5._xml(reference);after=deepcopy(before)
    # Inherited decks write only these three logs; no plot/model dump is enabled.
    outputs=before.findall('Output')
    need(len(outputs)==1 and not before.findall('.//plotfile') and not before.findall('.//dumpfile'),
         'no undeclared plot/model dump')
    m=v5._unique(before,'Material/material/m1')
    need(float(m.text)==2. and float(v5._unique(before,'Material/material/c1').text)==2000.
         and float(v5._unique(before,'Material/material/k').text)==149000/3,'unchanged reference modulus/bulk law')
    v5._unique(after,'Material/material/m1').text='-8'
    endpoint=contract['full_coordinates_m'][-1]
    factor=.5 if contract['native_domain']=='lower_half_reconstructed' else 1.
    times=[i/steps for i in range(steps+1)];full=[endpoint*t for t in times];full[-1]=endpoint
    for path in v5.CONTROLS:
        a=v5._unique(before,path);b=v5._unique(after,path)
        need(a.text==('60' if path.endswith('time_steps') else format(1/60,'.17g')),'old exact time control')
        b.text=str(steps) if path.endswith('time_steps') else format(1/steps,'.17g')
    old_curves=v5._curves(before,steps=60,endpoint=endpoint,factor=factor)
    new_curves=v5._curves(after,steps=60,endpoint=endpoint,factor=factor)
    for kind in ('zero','axial'):
        points=new_curves[kind][1]
        for child in list(points):points.remove(child)
        for t,value in zip(times,([0.]*(steps+1) if kind=='zero' else full)):
            ET.SubElement(points,'pt').text=f'{t:.17g},{value:.17g}'
    # Independent complete-tree delta check: restore only the declared changes.
    check=deepcopy(after)
    v5._curves(check,steps=steps,endpoint=endpoint,factor=factor)
    v5._unique(check,'Material/material/m1').text=m.text
    for path in v5.CONTROLS:v5._unique(check,path).text=v5._unique(before,path).text
    check_curves=v5._curves(check,steps=steps,endpoint=endpoint,factor=factor)
    for kind in ('zero','axial'):
        controller,points=check_curves[kind];controller.remove(points)
        controller.append(deepcopy(old_curves[kind][1]))
    need(v5._canonical(before)==v5._canonical(check),'unexpected geometry/BC/solver/material/output change')
    raw=(ET.tostring(after,encoding='unicode',xml_declaration=True)+'\n').encode()
    updated=dict(contract,run_id=contract['run_id'].replace(':S60:reference',f':S{steps}:alpha-minus8'),steps=steps,
        times=times,full_coordinates_m=full,native_coordinates_m=[factor*x for x in full],
        ogden_alpha=-8.,shape_declaration_sha256=DECLARATION_SHA256,adapted_deck_sha256=sha(raw))
    return raw,updated


def reference(root,declaration,branch,n):
    """Authenticate inherited source deck and mesh, without replaying old primitives."""
    from scripts import mechanics_hbe_v5_frame as frame
    from scripts import mechanics_hbe_v5_stream as stream
    from scripts import mechanics_hbe_v5_source_bindings as sources
    from scripts import mechanics_hbe_branch_calibration_v5 as v5
    e=declaration['evidence'];manifest=read_binding(root,e['manifest'])
    study_raw=read_binding(root,e['v5'],decode=False);study=json.loads(study_raw)
    prior_raw=read_binding(root,e['v4'],decode=False);prior=json.loads(prior_raw)
    run_id=f'{branch}:N{n}:S60:reference';q=manifest['rows'][run_id]
    order=read_binding(root,q['work_order']);bindings=order['bindings']
    source_manifest=read_binding(root,declaration['source_deck_bindings'])
    key=source_manifest['run_source_keys'][run_id];src=source_manifest['source_decks'][key]
    need(bindings['source_deck']=={k:src[k] for k in ('path','sha256')}
         and bindings['mesh']==source_manifest['meshes'][src['mesh']],'source and mesh ancestry')
    old=read_binding(root,bindings['source_deck'],maximum=32*1024**2,decode=False)
    adapted=read_binding(root,bindings['adapted_deck'],maximum=32*1024**2,decode=False)
    contract=frame.verified_schedule(study_raw,prior_raw,run_id,old,adapted,order['adapter_receipt'])
    mesh=read_binding(root,bindings['mesh'],maximum=32*1024**2)
    sources.validate_topology_bc(mesh,old,native_domain=contract['native_domain'])
    reconstruction=None
    if contract['native_domain']=='lower_half_reconstructed':
        reconstruction=stream._bound_reconstruction(root,prior,{'N':n},bindings['mesh'],mesh)
    need(contract['mu_Pa']==1000. and tuple(contract['times'])==tuple(i/60 for i in range(61)), 'reference identity')
    need(contract['full_coordinates_m'][-1]==declaration['endpoints_m'][branch], 'fixed signed endpoint')
    return contract,mesh,reconstruction,adapted,q


def prepare(root,declaration,out):
    evidence(root,declaration)
    prepared={}
    for branch,n,steps in RUNS:
        c,_,_,old,q=reference(root,declaration,branch,n)
        raw,contract=shape_deck(old,c,steps)
        name=run_name((branch,n,steps));directory=out/name;directory.mkdir(exist_ok=False)
        fd=os.open(directory/'specimen.feb',os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
        with os.fdopen(fd,'wb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
        prepared[name]={'contract':contract,'reference_readout':q['readout'],
            'deck':{'path':str((directory/'specimen.feb').relative_to(root)),'sha256':sha(raw)}}
    return prepared


def primitive_caps(row):
    need(tuple(row) in RUNS,'declared run only')
    return {'nodes':256*1024**2,'elements':256*1024**2}


def view(readout,branch,n):
    """Pure refinement view; individual acceptance is separately computed by stream."""
    return {'run_id':readout['run_id'],'branch':branch,'mesh_N':n,'steps':readout['steps'],
        'load_coordinate_m':readout['load_coordinate_full_m'],'applied_force_N':readout['applied_force_N'],
        'probe_displacements_m':readout['probe_displacements_m']}


def check_execution(declaration,row,prepared,execution):
    stage=execution['supervision']['receipt']['native_stage'];cleanup=execution['supervision']['cleanup']
    need(execution['run_id']==prepared['contract']['run_id'] and execution['deck']==prepared['deck']
         and execution['runtime_identity']==declaration['runtime_identity']
         and execution['command']==stage['command'],'native execution identity')
    need(stage['status']=='completed_within_caps' and stage['exit_code']==0 and stage['kill_reason'] is None
         and 0<=stage['elapsed_seconds']<stage['wall_cap_seconds']<=CAPS['native_seconds'][RUNS.index(tuple(row))]
         and stage['sampled_process_group_rss_cap_bytes']==CAPS['sampled_group_rss_bytes']
         and stage['active_output_cap_bytes']==CAPS['run_output_bytes']
         and cleanup['contained'] and cleanup['direct_child_reaped'] and cleanup['exit_code']==0
         and cleanup['fallback_used'] is False and cleanup['remaining_members']==[] and cleanup['errors']==[],
         'successful owned native execution required')
    need(set(execution['primitive_bindings'])=={'nodes','elements','solver'},'exact native primitives')
    name=run_name(row)
    for key,binding in execution['primitive_bindings'].items():
        need(binding['path']==OUTPUT+'/'+name+'/'+key+'.log','declared native output path')


def read_run(root,declaration,row,prepared,execution,*,deadline):
    from scripts import mechanics_hbe_v5_stream as stream
    from scripts.mechanics_hbe_v5_n8_one_shot import file_hash
    check_execution(declaration,row,prepared,execution)
    branch,n,steps=row;c,mesh,reconstruction,old,_=reference(root,declaration,branch,n)
    raw,expected=shape_deck(old,c,steps)
    need(canonical(expected)==canonical(prepared['contract']) and sha(raw)==prepared['deck']['sha256'],
         'exact prepared shape contract')
    read_binding(root,prepared['deck'],maximum=32*1024**2,decode=False)
    bindings=execution['primitive_bindings'];caps={**primitive_caps(row),'solver':16*1024**2}
    def recheck():
        for key,cap in caps.items():
            need(file_hash(local(root,bindings[key]['path']),maximum=cap)==bindings[key]['sha256'],'native bytes changed')
    recheck()
    with ExitStack() as stack:
        opened={key:stack.enter_context(local(root,bindings[key]['path']).open(encoding='utf-8',errors='strict')) for key in caps}
        iterator=stream.evaluated_frames(expected,mesh,opened['nodes'],opened['elements'],opened['solver'],
            reconstruction=reconstruction,ogden_shape=True,primitive_caps=primitive_caps(row))
        while True:
            need(time.monotonic()<deadline,'readout deadline')
            try:next(iterator)
            except StopIteration as done:report=done.value;break
    recheck()
    report['provenance'].update(native_output_observed=True,generated_fixture_only=False,
        source_binding_checked=True,output_origin='owned_ogden_shape_probe',ogden_alpha=-8.,
        shape_declaration_sha256=DECLARATION_SHA256,primitive_bindings=bindings)
    return report


def screen(reports,*,temporal):
    from scripts.mechanics_hbe_v5_comparison import _signed_pair
    expected=RUNS if temporal else RUNS[:4]
    need(set(reports)=={run_name(x) for x in expected},'complete predeclared phase readouts')
    need(all(r['numerical_passed'] is True for r in reports.values()),'individual numerical gate failed')
    pairs={}
    for branch in AXIAL:
        coarse=view(reports[run_name((branch,16,60))],branch,16)
        fine=view(reports[run_name((branch,24,60))],branch,24)
        pairs[branch+'/mesh']=_signed_pair(coarse,fine)
        if temporal:
            finer=view(reports[run_name((branch,24,120))],branch,24)
            pairs[branch+'/time']=_signed_pair(fine,finer)
    return {'schema':'hbe-ogden-two-grid-screen-v1','pairs':pairs,
        'passed':all(p['pairwise_limit_passed'] for p in pairs.values()),
        'old_v5_numerical_qualification_passed':False,'continuum_error_bound':False,
        'physical_validation_pass':None,'diagnostic_only':True}


def baseline(root,declaration):
    from scripts.mechanics_hbe_v5_comparison import _numerical_view, _signed_pair
    manifest,_,_,study,prior=evidence(root,declaration)
    references={};pairs={}
    for branch in AXIAL:
        views={}
        for n in (16,24):
            run_id=f'{branch}:N{n}:S60:reference';q=manifest['rows'][run_id]
            r=read_binding(root,q['readout'],maximum=8*1024**2)
            views[n],passed=_numerical_view(study,prior,run_id,r)
            need(passed,'accepted baseline per-stream gate')
            if n==24:references[branch]=r
        pairs[branch]=_signed_pair(views[16],views[24])
    need(all(p['pairwise_limit_passed'] for p in pairs.values()),'matched alpha-two pair diagnostic')
    return references,pairs


def profile(saved_fit,references):
    """One positive common mu per fixed alpha; no offset/row selection/sign edit."""
    from scripts.mechanics_hbe_evaluation import curve,fit_scale,scale_prediction,paired_metrics
    need(saved_fit['schema']=='hbe-positive-scale-fit-v1' and saved_fit['reference_mu_Pa']==1000.
         and saved_fit['held_out_values_used'] is False and saved_fit['offset_or_sign_fitted'] is False
         and set(saved_fit['modes'])==set(AXIAL),'exact exposed saved fit contract')
    observed={};ref={}
    for branch in AXIAL:
        row=saved_fit['modes'][branch]
        need(len(row['coordinate'])==len(row['observed'])==len(row['weights'])==30,'all sixty signed values')
        observed[branch]=curve(branch,row['coordinate'],row['observed'],source_sha256=row['source_sha256'])
        r=references[branch];need(r['steps']==60 and len(r['load_coordinate_full_m'])==61,'matched N24/S60 references')
        ref[branch]=curve(branch,r['load_coordinate_full_m'],r['applied_force_N'])
    fit=fit_scale(observed,ref)
    for branch in AXIAL:
        need(fit['modes'][branch]['weights']==saved_fit['modes'][branch]['weights'],'original equal-mode weights')
    predictions={b:scale_prediction(ref[b],fit['scale']) for b in AXIAL}
    return {'fit':fit,'metrics':paired_metrics(observed,predictions,characteristic_response_scale=.016),
        'scope':'same_exposed_DEVELOPMENT_model_adequacy','alpha_fit_optimizer_calls':0,
        'physical_validation_pass':None,'new_independent_validation':False}


def finish(root,declaration,report_bindings,out):
    numerical={name:read_binding(root,binding,maximum=8*1024**2) for name,binding in report_bindings.items()}
    checks=screen(numerical,temporal=True)
    exclusive_json(out/'numerical-screen.json',checks,maximum=4*1024**2)
    need(checks['passed'],'new shape failed predefined two-grid/time screen')
    old,pairs=baseline(root,declaration)
    # The only measured values opened by this adapter are this already exposed JSON.
    fit=read_binding(root,declaration['fit'],maximum=128*1024)
    results={'2':profile(fit,old),'-8':profile(fit,{b:numerical[run_name((b,24,60))] for b in AXIAL})}
    # Reproduce old common-scale arithmetic, using saved predictions without an FE call.
    from scripts.mechanics_hbe_evaluation import curve,fit_scale
    observed={b:curve(b,fit['modes'][b]['coordinate'],fit['modes'][b]['observed']) for b in AXIAL}
    oldref={b:curve(b,fit['modes'][b]['coordinate'],fit['modes'][b]['reference_prediction']) for b in AXIAL}
    reproduced=fit_scale(observed,oldref)
    need(reproduced['mu_Pa']==fit['mu_Pa'] and reproduced['scale']==fit['scale'],'original baseline reproduction')
    # All candidate outcomes, including worse outcomes, are retained; no new candidate is chosen.
    output={'schema':'hbe-ogden-shape-comparison-v1','alpha_candidates':list(ALPHAS),
        'specimen':'HBE_01_03','specimen_role':'DEVELOPMENT','observed_samples':60,
        'source_fit':declaration['fit'],'units':{'coordinate':'m','force':'N','mu':'Pa','alpha':'dimensionless'},
        'fixed_K_over_mu':149/3,'grid_for_comparison':'N24/S60','candidates':results,
        'original_alpha2_baseline':{'mu_Pa':fit['mu_Pa'],'scale':fit['scale'],
            'grid':'compression N36/S120; tension N24/S120','reproduced':True},
        'baseline_pair_screen':pairs,'new_shape_screen_passed':True,
        'numerical_qualification_passed':False,'continuum_error_bound':False,
        'physical_validation_pass':None,'empirical_tolerance':None,'held_out_values_used':False,
        'fitted_mu_native_confirmed':False,'patient_properties_assigned':False,
        'limitations':['two-point alpha comparison, not continuous exponent estimation',
            'two-grid screen does not establish a continuum error bound',
            'same exposed processed cycle-average specimen; no independent measured validation',
            'zero reference and noise remain unquantified; all signed values retained',
            'no living-brain retraction, cutting, fracture or tool-contact property qualification']}
    exclusive_json(out/'comparison.json',output,maximum=512*1024)
    read_binding(root,declaration['fit'],maximum=128*1024)
    return output

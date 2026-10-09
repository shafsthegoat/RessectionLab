"""Frozen branch-qualified specimen calibration boundary; no implicit curve IO."""
from copy import deepcopy
import hashlib
import importlib
import math
from pathlib import Path
import sys
import tarfile
import time
import xml.etree.ElementTree as ET

from scripts import mechanics_hbe_access as access
from scripts import mechanics_hbe_evaluation as evaluation
from scripts import mechanics_hbe_halfheight_temporal_common as common

receipts, inherited = common.receipts, common.inherited
old, runtime, backend = receipts.old, receipts.runtime, receipts.backend
DECLARATION_PATH = 'manifests/experiments/hbe-01-03-branch-calibration-v2.json'
DECLARATION_SHA256 = '49afffc1a1efc30ca4e44fd50ebc754d9ed65d699e61f7668e2ea62c0406f102'
AXIAL = ('compression', 'tension')
TORSION = ('torsion_neg', 'torsion_pos')
NEW_SOURCES = {'branch_calibration': 'mechanics_hbe_branch_calibration_v2.py',
               'branch_calibration_readout': 'mechanics_hbe_branch_calibration_v2_readout.py',
               'branch_calibration_runner': 'mechanics_hbe_branch_calibration_v2_experiment.py'}
META_BYTES = 64*1024**2


def positive(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise ValueError('Finite positive actual modulus required')
    return float(value)


def declaration(root):
    return access.verify_binding(root, {'path': DECLARATION_PATH, 'sha256': DECLARATION_SHA256},
                                 maximum_bytes=1024**2, read_json=True)


class Registry:
    """Exact path/hash/kind caps persist through parse, freeze and reveal."""
    def __init__(self, root):
        self.root, self.entries = Path(root).resolve(), {}

    def add(self, binding, limit=META_BYTES, *, kind='metadata'):
        access.local_path(self.root, binding['path'])
        if set(binding) != {'path', 'sha256'} or not access.HEX64.fullmatch(binding['sha256']):
            raise ValueError('Exact hash-bound input required')
        record = dict(binding, maximum_bytes=limit, kind=kind)
        previous = self.entries.get(binding['path'])
        if previous is not None:
            if previous['sha256'] != binding['sha256']:
                raise ValueError('Conflicting frozen input hash')
            if kind != 'metadata' and previous != record:
                if previous['kind'] != 'metadata':
                    raise ValueError('Conflicting exact file-kind cap')
                self.entries[binding['path']] = record
        else:
            self.entries[binding['path']] = record
        return binding

    def read(self, binding, *, json_value=True):
        item = self.entries.get(binding['path'])
        if item is None or item['sha256'] != binding['sha256']:
            raise ValueError('Unregistered or changed input')
        return access.verify_binding(self.root, binding, maximum_bytes=item['maximum_bytes'], read_json=json_value)

    def bound(self, binding, *, json_value=True):
        self.add(binding)
        return self.read(binding, json_value=json_value)

    def primitives(self, branch, bindings, *, fitted_directory=None):
        if set(bindings) != inherited.PRIMITIVE_KEYS:
            raise ValueError('Exactly six primitive bindings required')
        n = 36 if branch == 'compression' else 24 if branch == 'tension' else 12
        caps = {'nodes': 768*1024**2 if n == 36 else 256*1024**2,
                'elements': 512*1024**2 if n == 36 else 256*1024**2,
                'mesh': 32*1024**2 if n == 36 else 16*1024**2,
                'deck': 32*1024**2 if n == 36 else 16*1024**2,
                'loading': 1024**2, 'solver': 16*1024**2}
        names = {'deck':'specimen.feb', 'nodes':'nodes.log', 'elements':'elements.log',
                 'loading':'loading.json', 'solver':'solver.log'}
        for kind, binding in bindings.items():
            if fitted_directory is not None and kind != 'mesh':
                if access.local_path(self.root, binding['path']) != Path(fitted_directory)/names[kind]:
                    raise ValueError('Fitted primitive outside exact run directory')
            self.add(binding, caps[kind], kind=f'N{n}/{kind}')

    def verify_all(self, deadline=None):
        for item in self.entries.values():
            if deadline is not None:
                common.remaining_seconds(deadline, 3600)
            self.read({k:item[k] for k in ('path','sha256')}, json_value=False)

    def snapshot(self):
        return [deepcopy(self.entries[k]) for k in sorted(self.entries)]


def curve_from_row(branch, row):
    coordinate = row['load_coordinate_m' if branch in AXIAL else 'load_coordinate']
    field = 'applied_force_N' if branch in AXIAL else 'applied_torque_Nm'
    return evaluation.curve(branch, coordinate, row[field])


def checked_reference(branch, row):
    N = 36 if branch == 'compression' else 24 if branch == 'tension' else 12
    if (row.get('branch'), row.get('mesh_N'), row.get('steps'), row.get('mu_Pa'), row.get('frame_count')) != (branch,N,120,1000.,121):
        raise ValueError('Reference branch, mesh, modulus or schedule differs')
    c = curve_from_row(branch, row)
    endpoint = (-1 if branch in ('compression','torsion_neg') else 1)*.15*common.H
    if branch in TORSION:
        endpoint /= common.R
    if len(c.coordinate) != 121 or any(not math.isclose(x, endpoint*i/120, rel_tol=2e-14, abs_tol=1e-18) for i,x in enumerate(c.coordinate)):
        raise ValueError('Reference physical loading grid differs')
    if branch in AXIAL:
        if row.get('passed') is not True or row.get('full_native_equivalence_evaluated') is not False:
            raise ValueError('Accepted reconstructed axial reference required')
        for field in ('half_native','reconstructed_full','reconstruction_consistency'):
            if row[field]['passed'] is not True:
                raise ValueError('Incomplete axial reference checks')
            for name, value in row[field]['criteria'].items():
                access._metric_record(value, positive=name == 'minimum_sampled_J')
    else:
        for name, value in row['criteria'].items():
            access._metric_record(value, positive=name == 'minimum_sampled_J')
    return c


def qualify(root, study, registry):
    """Authenticate prior successful branches while retaining failed aggregates."""
    references = {}
    for branch in AXIAL:
        q = study['axial_references'][branch]
        registry.primitives(branch, q['native_primitives'])
        records = {name:registry.bound(binding) for name,binding in q['artifact_bindings'].items()}
        review, result, comparison = (records[k] for k in ('independent-review/verification.json','result.json','comparison.json'))
        if (review.get('blocking_record_mismatches') != [] or review.get('status') != 'saved_execution_and_temporal_diagnostic_verified'
                or review.get('complete_comparison_canonical_equal') is not True
                or review['conclusion'].get(branch+'_temporal_candidate') is not True
                or review['conclusion'].get('calibration_released') is not False
                or review['conclusion'].get('spatial_convergence_accepted') is not False
                or result.get('status') != 'completed_numerical_diagnostic_only'
                or comparison.get(branch+'_temporal_candidate') is not True):
            raise ValueError('Actual independent branch qualification required')
        if registry.bound(review['comparison']) != comparison or registry.bound(review['result']) != result:
            raise ValueError('Independent result origin differs')
        execution = registry.bound(q['run_execution'])
        if (execution != records['execution.json'] or execution['run_id'] != q['run_id']
                or execution['primitive_bindings'] != q['native_primitives']
                or execution['reconstruction'] != q['reconstruction']
                or execution['runtime_identity'] != study['runtime_identity']
                or execution['backend_profile'] != study['backend_profile']):
            raise ValueError('Reference native execution origin differs')
        receipts.require_completed_native(execution['execution'], 2100 if branch == 'compression' else 420)
        row = registry.bound(q['run_readout']); checked_reference(branch, row)
        if row['primitive_bindings'] != q['native_primitives'] or row['execution_binding'] != q['run_execution']:
            raise ValueError('Reference readout primitive origin differs')
        references[branch] = row
        for key in ('reconstruction','full_mesh','source_deck','study'):
            registry.bound(q[key], json_value=key != 'source_deck')
        prior = study['prior_executions'][branch]
        plan, release = registry.bound(prior['baseline']), registry.bound(prior['release'])
        if (plan['source_bindings'] != prior['source_bindings'] or plan['release_binding'] != prior['release']
                or release.get('authorized') is not True or release['source_archive'] != prior['archive']
                or release['source_commit'] != prior['commit'] or release['source_bindings'] != prior['source_bindings']):
            raise ValueError('Reference committed source release differs')
        prior_study = registry.bound(q['study'])
        wanted = {}
        for key,binding in prior['source_bindings'].items():
            if binding['sha256'] != study['inherited_sources'][key]['sha256']:
                raise ValueError('Historical source identity differs')
            registry.bound(binding, json_value=False)
            wanted['scripts/'+Path(binding['path']).name] = binding['sha256']
        declarations = (list(prior_study['original_declarations'].values())+[prior_study['baseline']['declaration']]
                        if branch == 'compression' else prior_study['inherited_declarations'])
        for binding in declarations+[q['study']]:
            registry.bound(binding)
            wanted[binding['path']] = binding['sha256']
        receipts.verify_archive(root, prior['archive'], prior['commit'], wanted, len(wanted))
        registry.bound(prior['archive'], json_value=False)
    legacy = registry.bound(study['legacy']['aggregate'])
    outcome = registry.bound(study['legacy']['outcome'])
    if not outcome['failed_aggregate_criteria'] or outcome['actual_solver_calls'] != 18:
        raise ValueError('Original failed aggregate must remain preserved')
    legacy_release = registry.bound(study['legacy']['release'])
    if legacy_release['execution']['backend_profile'] != study['backend_profile']:
        raise ValueError('Original torsion backend differs')
    original_sources = legacy_release['execution']['source_bindings']
    original_archive = legacy_release['execution']['source_archive']
    registry.bound(original_archive,json_value=False)
    with tarfile.open(access.local_path(root,original_archive['path'])) as bundle:
        if bundle.pax_headers.get('comment') != legacy_release['source_commit']:
            raise ValueError('Original torsion archive commit differs')
        for key,binding in original_sources.items():
            report_key = 'run_readout' if key == 'readout' else key
            if report_key in legacy['source_bindings'] and legacy['source_bindings'][report_key] != binding:
                raise ValueError('Original torsion report/source origin differs')
            registry.bound(binding,json_value=False)
            members=[m for m in bundle.getmembers() if m.name=='scripts/'+Path(binding['path']).name]
            if len(members)!=1 or not members[0].isfile() or members[0].size>1024**2:
                raise ValueError('Original torsion source member differs')
            if hashlib.sha256(bundle.extractfile(members[0]).read()).hexdigest()!=binding['sha256']:
                raise ValueError('Original torsion archived source differs')
    for branch in TORSION:
        for group in ('mesh','step'):
            for metric in legacy[group][branch].values():
                access._metric_record(metric)
        q = study['torsion_references'][branch]
        registry.primitives(branch,q['primitives'])
        row = registry.bound(q['readout']); checked_reference(branch,row)
        if legacy['runs'][f'{branch}:N12:S120:reference'] != row:
            raise ValueError('Original torsion aggregate/readout identity differs')
        access.verify_run_execution(root, f'{branch}:N12:S120:reference', row,
                                   study['protocol']['sha256'], expected_backend_profile=study['backend_profile'])
        registry.bound(q['execution']); references[branch] = row
    for branch in ('compression','torsion_pos'):
        for metric in legacy['scale'][branch].values():
            access._metric_record(metric)
    return references


def verify_runtime_migration(study, registry):
    """Bind the completed pre-data failure; no original response member is read."""
    migration = study.get('runtime_migration', {})
    bindings = ('previous_study', 'previous_release', 'previous_state', 'previous_result',
                'previous_supervision', 'previous_publication', 'previous_started', 'diagnosis')
    if (set(migration) != set(bindings) | {'schema', 'reason', 'automatic_retry',
                                         'scientific_fields_unchanged'}
            or migration.get('schema') != 'hbe-branch-runtime-migration-v1'
            or migration.get('reason') != 'pinned_python_replaced_before_measured_data_access'
            or migration.get('automatic_retry') is not False
            or migration.get('scientific_fields_unchanged') is not True):
        raise ValueError('Exact runtime-only migration declaration required')
    records = {key: registry.bound(migration[key]) for key in bindings}
    previous = records['previous_study']
    changed = {'schema', 'study_id', 'output_root', 'interpreter'}
    old_fixed = {key: value for key, value in previous.items() if key not in changed}
    new_fixed = {key: value for key, value in study.items()
                 if key not in changed | {'runtime_migration'}}
    if (access.canonical_json(old_fixed) != access.canonical_json(new_fixed)
            or (previous.get('schema'), previous.get('study_id'), previous.get('output_root')) !=
               ('hbe-branch-calibration-v1', 'hbe-01-03-branch-calibration-v1',
                'outputs/mechanics/hbe-01-03-branch-calibration-v1')
            or (study.get('schema'), study.get('study_id'), study.get('output_root')) !=
               ('hbe-branch-calibration-v2', 'hbe-01-03-branch-calibration-v2',
                'outputs/mechanics/hbe-01-03-branch-calibration-v2')
            or set(study['interpreter']) != {'path', 'sha256'}
            or study['interpreter']['path'] != previous['interpreter']['path']
            or study['interpreter']['sha256'] == previous['interpreter']['sha256']):
        raise ValueError('Migration must preserve every scientific and historical study field')
    prior_release = records['previous_release']
    if (prior_release.get('schema') != 'hbe-branch-calibration-release-v1'
            or prior_release.get('authorized') is not True
            or prior_release.get('phase') != 'calibrate_and_validate'
            or type(prior_release.get('authorized_native_calls')) is not int
            or prior_release['authorized_native_calls'] != 2
            or prior_release.get('study') != migration['previous_study']
            or prior_release.get('interpreter') != previous['interpreter']
            or prior_release.get('backend_profile') != previous['backend_profile']):
        raise ValueError('Predecessor release identity differs')
    state = records['previous_state']
    if (state.get('schema') != 'hbe-branch-calibration-state-v1'
            or state.get('status') != 'failed_or_incomplete'
            or state.get('error') != {'type': 'ValueError', 'message': 'Interpreter identity differs'}
            or any(state.get(key) is not False for key in
                   ('calibration_access_attempted', 'calibration_responses_accessed',
                    'held_out_access_attempted', 'held_out_responses_accessed', 'automatic_retry'))
            or any(type(state.get(key)) is not int or state[key] != 0
                   for key in ('native_calls', 'mesher_calls'))
            or state.get('runs') != {} or state.get('physical_validation_pass', False) is not None
            or any(key in state for key in ('fit', 'predictions', 'freeze', 'held_out_metrics',
                                            'calibration_metrics', 'preparation_seconds'))):
        raise ValueError('Predecessor must be the preserved no-access, zero-call interpreter failure')
    supervision = records['previous_supervision']
    result = records['previous_result']
    publication = records['previous_publication']
    if (supervision.get('status') != 'failed_or_incomplete'
            or type(supervision.get('exit_code')) is not int or supervision['exit_code'] != 1
            or supervision.get('no_retry') is not True
            or supervision.get('kill_reason', False) is not None
            or supervision.get('cleanup_error', False) is not None
            or result.get('schema') != 'hbe-branch-calibration-result-v1'
            or result.get('status') != 'failed_or_incomplete'
            or result.get('release') != migration['previous_release']
            or result.get('state') != migration['previous_state']
            or access.canonical_json(result.get('supervision')) != access.canonical_json(supervision)
            or result.get('physical_validation_pass', False) is not None
            or publication.get('schema') != 'hbe-branch-calibration-publication-v1'
            or publication.get('accepted') is not False
            or publication.get('result') != migration['previous_result']
            or set(records['previous_started']) != {'release', 'no_retry'}
            or records['previous_started'].get('release') != migration['previous_release']
            or records['previous_started'].get('no_retry') is not True):
        raise ValueError('Predecessor terminal, publication or no-retry binding differs')
    diagnosis = records['diagnosis']
    actual = diagnosis.get('actual', {})
    child = diagnosis.get('sanitized_child', {})
    if (diagnosis.get('schema') != 'hbe-branch-interpreter-drift-diagnostic-v1'
            or diagnosis.get('pinned') != previous['interpreter']
            or {key: actual.get(key) for key in ('path', 'sha256')} != study['interpreter']
            or child.get('resolved') != study['interpreter']['path']
            or child.get('sha256') != study['interpreter']['sha256']
            or access.canonical_json(diagnosis.get('first_execution')) != access.canonical_json(
               {'calibration_responses_accessed': False, 'native_calls': 0,
                'state': migration['previous_state']['path'],
                'supervision': migration['previous_supervision']['path']})):
        raise ValueError('Replacement interpreter diagnosis differs')
    return records


def preflight(root, release_binding, *, deadline=None):
    root=Path(root).resolve(); study=declaration(root); reg=Registry(root)
    reg.add({'path':DECLARATION_PATH,'sha256':DECLARATION_SHA256}); release=reg.bound(release_binding)
    if (release.get('schema') != 'hbe-branch-calibration-release-v2' or release.get('authorized') is not True
            or release.get('phase') != 'calibrate_and_validate'
            or release.get('study') != {'path':DECLARATION_PATH,'sha256':DECLARATION_SHA256}
            or release.get('authorized_native_calls') != 2 or release.get('backend_profile') != study['backend_profile']
            or release.get('interpreter') != study['interpreter']
            or release.get('predecessor_failure') != study['runtime_migration']['previous_state']):
        raise ValueError('Exact committed branch-calibration release required')
    verify_runtime_migration(study, reg)
    sources=release['source_bindings']; expected={k:Path(v['path']).name for k,v in study['inherited_sources'].items()}; expected.update(NEW_SOURCES)
    if set(sources) != set(expected): raise ValueError('Exact branch source closure required')
    wanted={}
    for key,name in expected.items():
        module=importlib.import_module('scripts.'+Path(name).stem)
        binding=sources[key]
        if access.local_path(root,binding['path']) != Path(module.__file__).resolve():
            raise ValueError('Executing source origin differs')
        if key in study['inherited_sources'] and binding['sha256'] != study['inherited_sources'][key]['sha256']:
            raise ValueError('Inherited source changed')
        reg.bound(binding,json_value=False);wanted['scripts/'+name]=binding['sha256']
    for binding in study['inherited_declarations']+[release['study']]:
        reg.bound(binding);wanted[binding['path']]=binding['sha256']
    receipts.verify_archive(root,release['source_archive'],release['source_commit'],wanted,len(wanted))
    reg.bound(release['source_archive'],json_value=False)
    python=Path(sys.executable).resolve()
    if python != Path(study['interpreter']['path']).resolve() or runtime.sha(python) != study['interpreter']['sha256']:
        raise ValueError('Interpreter identity differs')
    context=backend.verify_profile(root,study['backend_profile'])
    if context['runtime_identity'] != study['runtime_identity']:raise ValueError('Backend runtime differs')
    for path,sha in context['inputs'].items():
        absolute=Path(path).resolve()
        if not absolute.is_relative_to(root):raise ValueError('Backend input outside study root')
        reg.add({'path':str(absolute.relative_to(root)),'sha256':sha})
    for key in ('protocol','roles','design','readiness','scale_proof','scale_proof_review'):
        reg.bound(study[key],json_value=key!='scale_proof')
    roles=reg.read(study['roles']); directory=root/study['output_root']/'experiment'
    if (release.get('permitted_members') != [m['path'] for m in roles['calibration']['members']]
            or release.get('conditional_held_out_members') != [m['path'] for m in roles['held_out_validation']['members']]
            or release.get('execution') != {'access_ledger_path':str((directory/'access.jsonl').relative_to(root)),
                                          'csv_schemas':study['csv_schemas']}):
        raise ValueError('Exact role members/schema/access ledger required')
    references=qualify(root,study,reg);reg.verify_all(deadline)
    return {'study':study,'registry':reg,'references':references,'release':release,
            'release_binding':release_binding,'directory':directory,'context':context,'root':root}


def fitted_contents(registry, study, branch, mu):
    mu=positive(mu);q=study['axial_references'][branch]
    # Numerical representability, not a fitted-material range or clinical cutoff.
    for value in (2*mu,149*mu/3,mu/1000.,mu*common.R**2,mu*common.R**3,
                  mu*common.R**2*common.H,(1e-10*mu*common.R**2)**2):
        positive(value)
    registry.read(q['source_deck'],json_value=False)
    registry.read(q['native_primitives']['deck'],json_value=False)
    source=access.local_path(registry.root,q['source_deck']['path']).read_text()
    tree=ET.fromstring(source)
    for path,value in [('Material/material/c1',2*mu),('Material/material/k',149*mu/3),
                       ('Control/solver/min_residual',(1e-10*mu*common.R**2)**2)]:
        tree.find(path).text=repr(value)
    skyline=ET.tostring(tree,encoding='unicode',xml_declaration=True)+'\n'
    old.check_fitted_deck(source,skyline,mu)
    deck=backend.transform_deck(skyline)
    reference=access.local_path(registry.root,q['native_primitives']['deck']['path']).read_text()
    old.check_fitted_deck(reference,deck,mu);backend.verify_deck(skyline,deck)
    loading=deepcopy(registry.read(q['native_primitives']['loading']))
    loading.update(mu_Pa=mu,K_Pa=149*mu/3,min_residual_N2=(1e-10*mu*common.R**2)**2,
                   deck_sha256=hashlib.sha256(deck.encode()).hexdigest(),role='fitted',
                   branch_calibration_declaration_sha256=DECLARATION_SHA256)
    return skyline,deck,loading


class BranchReleasedStudy(access.ReleasedStudy):
    """Retain original audited member-reader and chronology, with exact new evidence."""
    def __init__(self, context, deadline):
        self.context,self.deadline=context,deadline
        super().__init__(context['root'],context['study']['protocol'],context['release_binding'],
                         ledger_path=context['release']['execution']['access_ledger_path'])

    def _release(self):
        common.remaining_seconds(self.deadline,3600)
        c=self.context;c['registry'].verify_all(self.deadline)
        if c['registry'].read(c['release_binding']) != c['release']:
            raise ValueError('Released branch authority changed')
        return self.protocol,self.roles,c['release']

    def _read_selected(self,*args,**kwargs):
        common.remaining_seconds(self.deadline,3600)
        result=super()._read_selected(*args,**kwargs)
        common.remaining_seconds(self.deadline,3600)
        return result

    def freeze_predictions(self, *, fit_binding, prediction_binding, fitted_runs, output_path):
        from scripts.mechanics_hbe_branch_calibration_v2_readout import paired_evidence
        self._before_reveal()
        if self._calibration is None or any(e['phase']=='freeze_saved' for e in self._events()):
            raise ValueError('One prior calibration and no existing freeze required')
        self._release();c=self.context;reg=c['registry']
        fit=reg.bound(fit_binding);prediction=reg.bound(prediction_binding)
        refs={b:curve_from_row(b,c['references'][b]) for b in AXIAL}
        if access.canonical_json(fit) != access.canonical_json(evaluation.fit_scale(self._calibration,refs)):
            raise ValueError('Saved fit differs from exact original objective')
        expected=predictions(c['references'],fit['scale'])
        if prediction != expected:raise ValueError('Frozen torque prediction differs')
        numeric=paired_evidence(c,fitted_runs,fit['mu_Pa'],min(self.deadline,time.monotonic()+600))
        numeric_binding=receipts.durable_json(self.root,c['directory']/'numerical.json',numeric)
        reg.add(numeric_binding)
        if numeric['passed'] is not True:raise ValueError('Actual fitted confirmation failed; held-out remains sealed')
        reg.verify_all(self.deadline)
        freeze={'schema':'hbe-branch-parameter-prediction-freeze-v2','protocol':self.protocol_binding,
                'roles':self.protocol['roles'],'release':self.release_binding,'declaration_sha256':DECLARATION_SHA256,
                'fit':fit_binding,'predictions':prediction_binding,'numerical':numeric_binding,
                'mu_Pa':fit['mu_Pa'],'verified_file_bindings':reg.snapshot(),'held_out_responses_accessed':False,
                'calibration_member_sha256':{k:v.source_sha256 for k,v in self._calibration.items()}}
        common.remaining_seconds(self.deadline,3600)
        digest=access.exclusive_json(access.local_path(self.root,output_path),freeze)
        binding={'path':output_path,'sha256':digest};self._audit('freeze_saved',[],freeze=binding)
        common.remaining_seconds(self.deadline,3600)
        return binding

    def evaluate_held_out(self, *, freeze_binding, schemas):
        self._release();self._check_schema_release(schemas,TORSION,self.context['release'])
        events=[e for e in self._events() if e['phase']=='freeze_saved']
        if len(events)!=1 or events[0].get('freeze')!=freeze_binding:
            raise ValueError('Audited durable freeze required before held-out')
        freeze=access.verify_binding(self.root,freeze_binding,maximum_bytes=META_BYTES,read_json=True)
        if (freeze.get('schema')!='hbe-branch-parameter-prediction-freeze-v2'
                or freeze.get('release')!=self.release_binding or freeze.get('protocol')!=self.protocol_binding
                or freeze.get('declaration_sha256')!=DECLARATION_SHA256
                or freeze.get('held_out_responses_accessed') is not False
                or freeze.get('verified_file_bindings')!=self.context['registry'].snapshot()):
            raise ValueError('Frozen exact branch input inventory differs')
        reg=self.context['registry'];reg.verify_all(self.deadline)
        fit,pred=reg.read(freeze['fit']),reg.read(freeze['predictions'])
        if fit['mu_Pa']!=freeze['mu_Pa']:raise ValueError('Frozen modulus changed')
        observed=self._read_selected(self.roles,self.roles['held_out_validation']['members'],TORSION,schemas,'held_out')
        predicted={b:evaluation.curve(b,row['coordinate'],row['response']) for b,row in pred['fitted'].items()}
        return evaluation.paired_metrics(observed,predicted,characteristic_response_scale=fit['mu_Pa']*common.R**3)


def predictions(references,scale):
    result={'schema':'hbe-torsion-prediction-v1','reference':{},'fitted':{}}
    for branch in TORSION:
        reference=curve_from_row(branch,references[branch]);scaled=evaluation.scale_prediction(reference,scale)
        result['reference'][branch]={'coordinate':list(reference.coordinate),'response':list(reference.response)}
        result['fitted'][branch]={'coordinate':list(scaled.coordinate),'response':list(scaled.response),'response_unit':'Nm'}
    return result

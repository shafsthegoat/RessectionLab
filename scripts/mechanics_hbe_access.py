"""Role-bound HBE member access and durable prediction freeze.

No archive is opened on import or construction. A root-issued, hash-bound
calibration release is required before the two calibration members are read.
Held-out access additionally verifies numerical evidence and a saved freeze.
This is a reproducibility boundary, not protection against a malicious host.
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import tarfile
from datetime import datetime, timezone
import zipfile

from scripts.mechanics_hbe_evaluation import curve, fit_scale, paired_metrics, scale_prediction

HEX64=re.compile(r'[0-9a-f]{64}')
BRANCHES=('compression','tension','torsion_neg','torsion_pos')
RUNTIME_IDENTITY_SHA256='f560f386726d13b399bcfb0c781a4b78104deecf41aa977e0efe205f18fabfc4'
SOURCE_FILES={'runtime_driver':'febio_runtime.py','access':'mechanics_hbe_access.py',
              'curve_evaluator':'mechanics_hbe_evaluation.py','mesh_deck':'mechanics_hbe_mesh.py',
              'primitive_parser':'mechanics_hbe_outputs.py','physics':'mechanics_hbe_physics.py',
              'readout':'mechanics_hbe_readout.py','orchestrator':'mechanics_hbe_experiment.py'}


def digest(data):
    return hashlib.sha256(data).hexdigest()


def canonical_json(value):
    return (json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+'\n').encode()


def local_path(root,relative):
    if not isinstance(relative,str) or Path(relative).is_absolute():
        raise ValueError('Repository-relative evidence path required')
    base=Path(root).resolve()
    path=(base/relative).resolve()
    if not path.is_relative_to(base):
        raise ValueError('Evidence path escapes root')
    return path


def verify_binding(root,binding,*,maximum_bytes=256*1024**2,read_json=False):
    if set(binding)!={'path','sha256'} or not HEX64.fullmatch(binding['sha256']):
        raise ValueError('Exact path/SHA256 binding required')
    path=local_path(root,binding['path'])
    h=hashlib.sha256()
    size=0
    chunks=[]
    with path.open('rb') as stream:
        while block:=stream.read(1024**2):
            size+=len(block)
            if size>maximum_bytes:
                raise ValueError('Evidence file exceeds bound')
            h.update(block)
            if read_json:
                chunks.append(block)
    if h.hexdigest()!=binding['sha256']:
        raise ValueError('Evidence hash changed')
    return json.loads(b''.join(chunks)) if read_json else size


def exclusive_json(path,value):
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    data=canonical_json(value)
    with path.open('xb') as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    descriptor=os.open(path.parent,os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return digest(data)


def verify_release_provenance(root,release,protocol_sha256):
    """Bind actual committed source archive and established execution prerequisites."""
    execution=release.get('execution',{})
    sources=execution.get('source_bindings',{})
    if set(sources)!=set(SOURCE_FILES):
        raise ValueError('Complete executing source inventory required')
    wanted={}
    for key,filename in SOURCE_FILES.items():
        binding=sources[key]
        if local_path(root,binding['path'])!=Path(__file__).resolve().with_name(filename):
            raise ValueError('Released source differs from executing module origin')
        verify_binding(root,binding,maximum_bytes=1024**2)
        wanted[f'scripts/{filename}']=binding['sha256']
    archive=execution.get('source_archive',{})
    if archive.get('sha256')!=release['source_archive_sha256']:
        raise ValueError('Actual source archive binding required')
    verify_binding(root,archive)
    seen=set()
    with tarfile.open(local_path(root,archive['path']),'r:*') as bundle:
        if bundle.pax_headers.get('comment')!=release['source_commit']:
            raise ValueError('Git archive commit identity differs')
        for member in bundle:
            if member.name not in wanted:
                continue
            if member.name in seen or not member.isfile() or member.size>1024**2:
                raise ValueError('Duplicate/unsafe source archive member')
            stream=bundle.extractfile(member)
            if stream is None or digest(stream.read())!=wanted[member.name]:
                raise ValueError('Archived source bytes differ')
            seen.add(member.name)
    if seen!=set(wanted):
        raise ValueError('Source archive lacks required executing helpers')
    prerequisites=release['prerequisite_evidence']
    if prerequisites['runtime']['sha256']!=RUNTIME_IDENTITY_SHA256:
        raise ValueError('Exact verified local FEBio runtime identity required')
    runtime=verify_binding(root,prerequisites['runtime'],read_json=True)
    if runtime.get('runtime_version')!='4.13.0' or runtime.get('architecture')!='arm64':
        raise ValueError('Verified runtime version/architecture differs')
    patch=verify_binding(root,prerequisites['analytic_verification'],read_json=True)
    if (
        patch.get('status') != 'passed_all_five_fixed_patch_controls'
        or patch.get('runtime_identity_sha256') != RUNTIME_IDENTITY_SHA256
        or patch.get('solver_invocations') != 5
        or len(patch.get('rows', [])) != 5
        or not all(row.get('passed') is True for row in patch['rows'])
        or patch.get('stiffness_scaling', {}).get('passed') is not True
    ):
        raise ValueError('Actual accepted analytical and scale prerequisites required')
    mesh=verify_binding(root,prerequisites['mesh_validation'],read_json=True)
    # Earlier prepared decks retain their historical builder identity. The
    # current helper remains separately bound above for fitted-scale decks.
    preparation_source=execution.get('mesh_preparation_source', sources['mesh_deck'])
    verify_binding(root, preparation_source, maximum_bytes=1024**2)
    if (
        mesh.get('status') != 'prepared_not_solved'
        or mesh.get('source_sha256') != preparation_source['sha256']
        or mesh.get('protocol_sha256') != protocol_sha256
    ):
        raise ValueError('Actual bound mesh preparation required')


def _metric_record(value,*,positive=False):
    """Validate numerical observations, never trust an aggregate passed boolean."""
    if not isinstance(value,dict) or set(value) not in ({'actual','limit','units'},{'actual','limit','units','comparison'}):
        raise ValueError('Actual numerical criterion and units required, not a boolean')
    actual,limit=value['actual'],value['limit']
    if isinstance(actual,bool) or isinstance(limit,bool) or not all(isinstance(x,(float,int)) and math.isfinite(x) for x in (actual,limit)):
        raise ValueError('Finite numerical evidence required')
    if not isinstance(value['units'],str) or not value['units']:
        raise ValueError('Criterion units required')
    comparison=value.get('comparison','le')
    if comparison not in ('le','lt'):raise ValueError('Unsupported error comparison')
    if positive:
        if limit!=0 or actual<=0:raise ValueError('Positive sampled Jacobian gate failed')
    elif actual<0 or limit<0 or (actual>limit if comparison=='le' else actual>=limit):
        raise ValueError('Numerical error gate failed')


def expected_runs():
    runs={f'{b}:N{n}:S60:reference':(b,n,60,'reference') for b in BRANCHES for n in (4,8,12)}
    runs.update({f'{b}:N12:S120:reference':(b,12,120,'reference') for b in BRANCHES})
    runs.update({f'{b}:N12:S120:double_mu':(b,12,120,'double_mu') for b in ('compression','torsion_pos')})
    runs.update({f'{b}:N12:S120:fitted':(b,12,120,'fitted') for b in ('compression','tension')})
    return runs


def _check_report(report, *, protocol_sha256, fitted_mu_Pa, require_fitted, root=None):
    if report.get('schema')!='hbe-numerical-evidence-v1' or report.get('protocol_sha256')!=protocol_sha256:
        raise ValueError('Unbound numerical evidence')
    sources=report.get('source_bindings',{})
    if set(sources)!={'physics','primitive_parser','mesh_deck','curve_evaluator','run_readout'}:
        raise ValueError('Complete independent source bindings required')
    if root is not None:
        for name, value in sources.items():
            source_name = 'readout' if name == 'run_readout' else name
            if local_path(root, value['path']) != Path(__file__).resolve().with_name(SOURCE_FILES[source_name]):
                raise ValueError('Numerical source binding differs from executing helper')
            verify_binding(root, value, maximum_bytes=1024**2)
    runs=report.get('runs',{})
    required_runs=expected_runs()
    if not require_fitted:required_runs={k:v for k,v in required_runs.items() if v[-1]!='fitted'}
    if set(runs)!=set(required_runs):raise ValueError('All twenty declared run receipts required' if require_fitted else 'All eighteen reference run receipts required')
    criteria={'solver_residual','force_balance','moment_balance','prescribed_motion','work_energy','primitive_consistency','minimum_sampled_J'}
    for key,(branch,N,steps,role) in required_runs.items():
        row=runs[key]
        expected_mu={'reference':1000.,'double_mu':2000.,'fitted':fitted_mu_Pa}[role]
        if (row.get('branch'),row.get('mesh_N'),row.get('steps'),row.get('mu_Pa'),row.get('frame_count'))!=(branch,N,steps,expected_mu,steps+1):
            raise ValueError('Run identity or actual frame count differs')
        endpoint=.15*.00489159/(.004 if branch.startswith('torsion') else 1)
        if branch in ('compression','torsion_neg'):endpoint=-endpoint
        expected_coordinates=[endpoint*i/steps for i in range(steps+1)]
        coordinates=row.get('load_coordinate',[])
        if len(coordinates)!=steps+1 or any(not isinstance(x,(int,float)) or isinstance(x,bool) or not math.isfinite(x) or not math.isclose(x,y,rel_tol=2e-14,abs_tol=1e-18) for x,y in zip(coordinates,expected_coordinates)):
            raise ValueError('Run physical loading grid differs')
        for field in ('applied_force_N','applied_torque_Nm'):
            values=row.get(field,[])
            if len(values)!=steps+1 or any(not isinstance(x,(int,float)) or isinstance(x,bool) or not math.isfinite(x) for x in values):
                raise ValueError('Complete finite independently read reactions required')
        primitives=row.get('primitive_bindings',{})
        if set(primitives)!={'nodes','elements','solver','mesh','deck','loading'}:
            raise ValueError('Missing primitive provenance')
        if root is not None:
            for value in primitives.values():
                verify_binding(root, value)
        if set(row.get('criteria',{}))!=criteria:raise ValueError('Incomplete numerical criteria')
        for name,value in row['criteria'].items():_metric_record(value,positive=name=='minimum_sampled_J')
    required={'mesh':set(BRANCHES),'step':set(BRANCHES),'scale':{'compression','torsion_pos'}}
    if require_fitted:required['fitted_confirmation']={'compression','tension'}
    for kind,branches in required.items():
        group=report.get(kind,{})
        if set(group)!=branches:raise ValueError('Missing declared convergence/scale group')
        names={'reaction','motion','torque'} if kind in ('scale','fitted_confirmation') else {'reaction','motion'}
        if kind=='mesh':names|={'reaction_trend','motion_trend'}
        for row in group.values():
            if set(row)!=names:raise ValueError('Incomplete convergence/scale criteria')
            for value in row.values():_metric_record(value)
    return report


def check_numeric_report(report, *, fitted_mu_Pa=None):
    """Check freshly computed criteria, without IO or authorization to reveal data.

    This cheap immediate gate is for the orchestrator after each readout. It
    cannot replace the independent primitive replay performed at freeze time.
    """
    return _check_report(
        report,
        protocol_sha256=report.get('protocol_sha256'),
        fitted_mu_Pa=fitted_mu_Pa,
        require_fitted=fitted_mu_Pa is not None,
    )


def verify_run_execution(root, run_id, row, protocol_sha256):
    binding = row.get('execution_binding')
    if binding is None:
        raise ValueError('Actual bounded solver execution receipt required')
    record = verify_binding(root, binding, maximum_bytes=1024**2, read_json=True)
    if record.get('schema') != 'hbe-run-execution-v1':
        raise ValueError('Unrecognized solver execution evidence')
    if record.get('run_id') != run_id or record.get('protocol_sha256') != protocol_sha256:
        raise ValueError('Execution belongs to another declared run')
    if record.get('primitive_bindings') != row['primitive_bindings']:
        raise ValueError('Execution output bindings differ from readout')
    execution = record.get('execution', {})
    elapsed = execution.get('elapsed_seconds')
    if (
        type(execution.get('exit_code')) is not int
        or execution['exit_code'] != 0
        or execution.get('timed_out') is not False
        or isinstance(elapsed, bool)
        or not isinstance(elapsed, (float, int))
        or not math.isfinite(elapsed)
        or not 0 <= elapsed <= 90
    ):
        raise ValueError('Bounded successful execution was not established')
    runtime_binding = record.get('runtime_identity', {})
    if runtime_binding.get('sha256') != RUNTIME_IDENTITY_SHA256:
        raise ValueError('Run used an unverified FEBio runtime')
    runtime = verify_binding(root, runtime_binding, read_json=True)
    executable = record.get('executable', {})
    verify_binding(root, executable)
    if executable['sha256'] != runtime['executable_sha256']:
        raise ValueError('Executed binary differs from verified runtime')
    expected_command = [
        str(local_path(root, executable['path'])),
        '-noconfig', '-no_title', '-i', 'specimen.feb', '-o', 'solver.log',
    ]
    if record.get('command') != expected_command:
        raise ValueError('Actual solver command differs from frozen invocation')
    directory = local_path(root, record.get('cwd'))
    for key, filename in (
        ('deck', 'specimen.feb'), ('solver', 'solver.log'),
        ('nodes', 'nodes.log'), ('elements', 'elements.log'),
    ):
        if local_path(root, row['primitive_bindings'][key]['path']) != directory / filename:
            raise ValueError('Execution working directory differs from bound deck or outputs')
    return record


def _replay_evidence(root, report, *, protocol_sha256, fitted_mu_Pa):
    # Import late: readout shares small binding helpers from this module.
    from scripts.mechanics_hbe_readout import read_run, build_numerical_evidence

    runs = {}
    caches = {}
    for run_id, row in report['runs'].items():
        verify_run_execution(root, run_id, row, protocol_sha256)
        retain = row['mesh_N'] == 12 and row['steps'] == 120 and row['branch'] in (
            'compression', 'tension', 'torsion_pos',
        )
        reconstructed, cache = read_run(
            root, row['primitive_bindings'],
            protocol_sha256=protocol_sha256,
            expected_branch=row['branch'], expected_mesh_N=row['mesh_N'],
            expected_steps=row['steps'], expected_mu_Pa=row['mu_Pa'],
            retain_scale_primitives=retain,
        )
        reconstructed['execution_binding'] = row['execution_binding']
        runs[run_id] = reconstructed
        if cache is not None:
            caches[run_id] = cache
    return build_numerical_evidence(
        runs, caches, source_bindings=report['source_bindings'],
        protocol_sha256=protocol_sha256, fitted_mu_Pa=fitted_mu_Pa,
    )


def validate_numerical_evidence(root, binding, *, protocol_sha256, fitted_mu_Pa):
    """One independent replay before a durable parameter/prediction freeze."""
    report = verify_binding(root, binding, maximum_bytes=64*1024**2, read_json=True)
    _check_report(
        report, protocol_sha256=protocol_sha256, fitted_mu_Pa=fitted_mu_Pa,
        require_fitted=True, root=root,
    )
    rebuilt = _replay_evidence(
        root, report, protocol_sha256=protocol_sha256, fitted_mu_Pa=fitted_mu_Pa,
    )
    if canonical_json(rebuilt) != canonical_json(report):
        raise ValueError('Numerical evidence differs from independent primitive replay')
    check_numeric_report(rebuilt, fitted_mu_Pa=fitted_mu_Pa)
    return rebuilt


def validate_reference_evidence(root,binding,*,protocol_sha256):
    report = verify_binding(root, binding, maximum_bytes=64*1024**2, read_json=True)
    return _check_report(
        report, protocol_sha256=protocol_sha256, fitted_mu_Pa=None,
        require_fitted=False, root=root,
    )


def parse_member_csv(data,branch,schema):
    """Explicit source schema only; never infer column order, units or sign."""
    expected_keys={'delimiter','header','coordinate_column','response_column','coordinate_unit','response_unit'}
    if set(schema)!=expected_keys or schema['delimiter'] not in (',',';','\t'):
        raise ValueError('Explicit supported CSV schema required')
    expected_units=('rad','Nm') if branch.startswith('torsion') else ('m','N')
    if (schema['coordinate_unit'],schema['response_unit'])!=expected_units:
        raise ValueError('Source units differ; no implicit conversion')
    columns=(schema['coordinate_column'],schema['response_column'])
    if any(not isinstance(c,int) or isinstance(c,bool) or c<0 for c in columns) or columns[0]==columns[1]:
        raise ValueError('Distinct nonnegative column indices required')
    rows=list(csv.reader(io.StringIO(data.decode('utf-8')),delimiter=schema['delimiter']))
    header=schema['header']
    if header is not None:
        if not isinstance(header,list) or not rows or rows.pop(0)!=header:raise ValueError('CSV header differs; no autodetection')
    if not rows or any(not row or len(row)<=max(columns) for row in rows):
        raise ValueError('Incomplete source rows')
    if len({len(row) for row in rows})!=1:raise ValueError('Inconsistent source row width')
    x=[float(row[columns[0]]) for row in rows]
    y=[float(row[columns[1]]) for row in rows]
    return curve(branch,x,y,source_sha256=digest(data))


class ReleasedStudy:
    """Construction reads declarations only; explicit methods control curve access."""
    def __init__(self,root,protocol_binding,release_binding,*,ledger_path):
        self.root=Path(root).resolve()
        self.protocol_binding=dict(protocol_binding)
        self.release_binding=dict(release_binding)
        self.ledger_path=local_path(self.root,ledger_path)
        self.protocol=verify_binding(self.root,self.protocol_binding,maximum_bytes=1024**2,read_json=True)
        self.roles=verify_binding(self.root,self.protocol['roles'],maximum_bytes=1024**2,read_json=True)
        self._calibration=None

    def _release(self):
        protocol=verify_binding(self.root,self.protocol_binding,maximum_bytes=1024**2,read_json=True)
        roles=verify_binding(self.root,protocol['roles'],maximum_bytes=1024**2,read_json=True)
        release=verify_binding(self.root,self.release_binding,maximum_bytes=1024**2,read_json=True)
        if (
            release.get('schema') != 'hbe-calibration-release-v1'
            or not re.fullmatch(r'[0-9a-f]{40}', release.get('source_commit', ''))
            or not HEX64.fullmatch(release.get('source_archive_sha256', ''))
        ):
            raise ValueError('Explicit committed calibration release required')
        identities = {
            'protocol_sha256': self.protocol_binding['sha256'],
            'roles_sha256': protocol['roles']['sha256'],
            'archive_sha256': roles['source']['archive_sha256'],
        }
        for key, value in identities.items():
            if release.get(key) != value:
                raise ValueError('Release evidence differs')
        allowed=[m['path'] for m in roles['calibration']['members']]
        if release.get('permitted_members')!=allowed:
            raise ValueError('Release must name exactly two calibration members')
        prerequisites=release.get('prerequisite_evidence',{})
        if set(prerequisites)!={'runtime','analytic_verification','mesh_validation'}:
            raise ValueError('Missing prerequisite receipts')
        for binding in prerequisites.values():
            verify_binding(self.root,binding)
        ledger = release.get('execution', {}).get('access_ledger_path')
        if ledger is None or local_path(self.root, ledger) != self.ledger_path:
            raise ValueError('Access ledger location must be bound by the release')
        verify_release_provenance(self.root,release,self.protocol_binding['sha256'])
        return protocol,roles,release

    def _events(self):
        if not self.ledger_path.exists():
            return []
        if self.ledger_path.stat().st_size > 1024**2:
            raise ValueError('Access ledger exceeds bound')
        events = [json.loads(line) for line in self.ledger_path.read_text().splitlines()]
        calibrated = False
        frozen = False
        revealed = False
        pending = None
        for index, event in enumerate(events):
            if (
                event.get('protocol_sha256') != self.protocol_binding['sha256']
                or event.get('release_sha256') != self.release_binding['sha256']
                or event.get('sequence') != index
            ):
                raise ValueError('Access ledger identity or sequence differs')
            phase = event.get('phase')
            if phase == 'calibration_attempt':
                if frozen or revealed:
                    raise ValueError('Calibration occurred after sealing')
                pending = ('calibration', event.get('members'))
            elif phase == 'calibration_completed':
                if pending != ('calibration', event.get('members')):
                    raise ValueError('Calibration completion lacks matching attempt')
                calibrated = True
                pending = None
            elif phase == 'freeze_saved':
                if not calibrated or frozen or revealed:
                    raise ValueError('Invalid durable freeze chronology')
                verify_binding(self.root, event['freeze'], maximum_bytes=16*1024**2)
                frozen = True
            elif phase == 'held_out_attempt':
                if not frozen:
                    raise ValueError('Held-out attempt precedes saved freeze')
                revealed = True
                pending = ('held_out', event.get('members'))
            elif phase == 'held_out_completed':
                if pending != ('held_out', event.get('members')):
                    raise ValueError('Held-out completion lacks matching attempt')
                pending = None
            else:
                raise ValueError('Unknown access-ledger event')
        return events

    def _before_reveal(self):
        if any(event.get('phase') in ('held_out_attempt','held_out_completed') for event in self._events()):
            raise ValueError('Held-out exposure seals subsequent calibration and refitting')

    def _audit(self,phase,members,**extra):
        events = self._events()
        event = {
            'timestamp_utc': datetime.now(timezone.utc).isoformat(),
            'sequence': len(events),
            'phase': phase,
            'members': members,
            'protocol_sha256': self.protocol_binding['sha256'],
            'release_sha256': self.release_binding['sha256'],
            **extra,
        }
        self.ledger_path.parent.mkdir(parents=True, exist_ok=True)
        with self.ledger_path.open('ab') as stream:
            stream.write(json.dumps(event, sort_keys=True, allow_nan=False).encode()+b'\n')
            stream.flush()
            os.fsync(stream.fileno())
        descriptor = os.open(self.ledger_path.parent, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)

    def _read_selected(self,roles,members,branches,schemas,phase):
        if set(schemas)!=set(branches):
            raise ValueError('Explicit schema for each allowed member required')
        path=local_path(self.root,roles['source']['archive_path'])
        expected_hash=roles['source']['archive_sha256']
        expected_size=roles['source']['archive_bytes']
        names=[m['path'] for m in members]
        self._audit(phase+'_attempt',names)
        result={}
        with path.open('rb') as stream:
            def check_archive():
                stream.seek(0)
                h=hashlib.sha256()
                size=0
                while block:=stream.read(1024**2):
                    size+=len(block)
                    if size>expected_size:
                        raise ValueError('Archive size differs')
                    h.update(block)
                if size!=expected_size or h.hexdigest()!=expected_hash:
                    raise ValueError('Archive hash differs')
                stream.seek(0)
            check_archive()
            with zipfile.ZipFile(stream) as archive:
                if len(set(archive.namelist()))!=len(archive.infolist()):
                    raise ValueError('Duplicate ZIP member names')
                for item,branch in zip(members,branches,strict=True):
                    info=archive.getinfo(item['path'])
                    if info.file_size!=item['bytes'] or f'{info.CRC:08x}'!=item['crc32'] or info.file_size>1024**2:
                        raise ValueError('Selected member identity/size differs')
                    data=archive.read(info)
                    result[branch]=parse_member_csv(data,branch,schemas[branch])
            check_archive()
        self._audit(phase+'_completed',names,member_sha256={k:v.source_sha256 for k,v in result.items()})
        return result

    def read_calibration(self,schemas):
        self._before_reveal()
        if any(event.get('phase') == 'freeze_saved' for event in self._events()):
            raise ValueError('Calibration is sealed by the saved parameter freeze')
        _,roles,release=self._release()
        self._check_schema_release(schemas, ('compression', 'tension'), release)
        self._calibration=self._read_selected(
            roles, roles['calibration']['members'], ('compression', 'tension'),
            schemas, 'calibration',
        )
        return dict(self._calibration)

    @staticmethod
    def _check_schema_release(schemas, branches, release):
        declared = release.get('execution', {}).get('csv_schemas', {})
        expected = {branch: declared.get(branch) for branch in branches}
        if canonical_json(schemas) != canonical_json(expected):
            raise ValueError('CSV interpretation differs from the released schema')

    def freeze_predictions(self,*,fit_binding,prediction_binding,numerical_binding,output_path):
        self._before_reveal()
        if any(event.get('phase')=='freeze_saved' for event in self._events()):
            raise ValueError('This study already has a durable parameter freeze')
        protocol,roles,release=self._release()
        if self._calibration is None:
            raise ValueError('Calibration access must precede fit freeze')
        fit=verify_binding(self.root,fit_binding,maximum_bytes=1024**2,read_json=True)
        prediction=verify_binding(self.root,prediction_binding,maximum_bytes=4*1024**2,read_json=True)
        if (
            prediction.get('schema') != 'hbe-torsion-prediction-v1'
            or set(prediction.get('reference', {})) != {'torsion_neg', 'torsion_pos'}
        ):
            raise ValueError('Both pre-reveal reference torsion curves required')
        evidence=validate_numerical_evidence(
            self.root, numerical_binding,
            protocol_sha256=self.protocol_binding['sha256'], fitted_mu_Pa=fit['mu_Pa'],
        )
        refs={}
        for mode in ('compression','tension'):
            row=evidence['runs'][f'{mode}:N12:S120:reference']
            refs[mode]=curve(mode,row['load_coordinate'],row['applied_force_N'])
        recomputed=fit_scale(self._calibration,refs)
        if canonical_json(fit)!=canonical_json(recomputed):
            raise ValueError('Saved fit differs from exact read calibration/declared scale calculation')
        for mode,row in prediction['reference'].items():
            run=evidence['runs'][f'{mode}:N12:S120:reference']
            if row!={'coordinate':run['load_coordinate'],'response':run['applied_torque_Nm']}:
                raise ValueError('Prediction reference differs from bound independent run readout')
            reference=curve(mode,row['coordinate'],row['response'])
            expected=scale_prediction(reference,fit['scale'])
            saved=prediction.get('fitted',{}).get(mode)
            if saved!={'coordinate':list(expected.coordinate),'response':list(expected.response),'response_unit':'Nm'}:
                raise ValueError('Saved torsion prediction does not follow frozen scale')
        freeze = {
            'schema': 'hbe-parameter-prediction-freeze-v1',
            'created_at_utc': datetime.now(timezone.utc).isoformat(),
            'protocol': self.protocol_binding,
            'release': self.release_binding,
            'roles': protocol['roles'],
            'archive_sha256': roles['source']['archive_sha256'],
            'fit': fit_binding,
            'predictions': prediction_binding,
            'numerical_evidence': numerical_binding,
            'calibration_member_sha256': {
                k: v.source_sha256 for k, v in self._calibration.items()
            },
            'held_out_responses_accessed': False,
            'mu_Pa': fit['mu_Pa'],
        }
        # Replaying once is sufficient when every subsequently consumed byte is
        # captured by this durable freeze and checked again before reveal.
        frozen_bindings = {}
        def collect(value):
            if isinstance(value, dict):
                if set(value) == {'path', 'sha256'}:
                    old = frozen_bindings.setdefault(value['path'], value['sha256'])
                    if old != value['sha256']:
                        raise ValueError('Conflicting hashes for one frozen input')
                else:
                    for child in value.values():
                        collect(child)
            elif isinstance(value, list):
                for child in value:
                    collect(child)
        collect(evidence)
        collect(release)
        collect([fit_binding, prediction_binding, numerical_binding])
        for row in evidence['runs'].values():
            collect(verify_binding(self.root, row['execution_binding'], read_json=True))
        freeze['verified_file_bindings'] = [
            {'path': path, 'sha256': sha} for path, sha in sorted(frozen_bindings.items())
        ]
        for binding in freeze['verified_file_bindings']:
            verify_binding(self.root, binding)
        target=local_path(self.root,output_path)
        sha=exclusive_json(target,freeze)
        binding={'path':output_path,'sha256':sha}
        self._audit('freeze_saved',[],freeze=binding)
        return binding

    def evaluate_held_out(self,*,freeze_binding,schemas):
        protocol,roles,release=self._release()
        self._check_schema_release(schemas, ('torsion_neg', 'torsion_pos'), release)
        freezes=[event for event in self._events() if event.get('phase')=='freeze_saved']
        if len(freezes)!=1 or freezes[0].get('freeze')!=freeze_binding:
            raise ValueError('Freeze was not durably created by this study before reveal')
        freeze=verify_binding(self.root,freeze_binding,maximum_bytes=1024**2,read_json=True)
        if (
            freeze.get('schema') != 'hbe-parameter-prediction-freeze-v1'
            or freeze.get('protocol') != self.protocol_binding
            or freeze.get('release') != self.release_binding
            or freeze.get('roles') != protocol['roles']
            or freeze.get('archive_sha256') != roles['source']['archive_sha256']
            or freeze.get('held_out_responses_accessed') is not False
        ):
            raise ValueError('Wrong study/freeze identity')
        fit=verify_binding(self.root,freeze['fit'],maximum_bytes=1024**2,read_json=True)
        predictions=verify_binding(self.root,freeze['predictions'],maximum_bytes=4*1024**2,read_json=True)
        if fit['mu_Pa']!=freeze['mu_Pa']:
            raise ValueError('Fitted scale changed')
        bindings = freeze.get('verified_file_bindings', [])
        if not bindings:
            raise ValueError('Freeze lacks independently verified input inventory')
        for binding in bindings:
            verify_binding(self.root, binding)
        observed=self._read_selected(
            roles, roles['held_out_validation']['members'],
            ('torsion_neg', 'torsion_pos'), schemas, 'held_out',
        )
        predicted={mode:curve(mode,row['coordinate'],row['response']) for mode,row in predictions['fitted'].items()}
        return paired_metrics(observed,predicted,characteristic_response_scale=fit['mu_Pa']*.004**3)

"""Failure audit: source and small receipts only, no measured archive access."""
import datetime
import hashlib
import json
from pathlib import Path
import subprocess
import tarfile

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
RUN = ROOT/'outputs/mechanics/hbe-01-03-branch-calibration-v2'
BUILD = ROOT/'build/hbe-branch-calibration-v2'
checks = 0


def require(value, label):
    global checks
    checks += 1
    if not value:
        raise AssertionError(label)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def binding(path):
    return {'path':str(path.relative_to(ROOT)),'sha256':sha(path),'bytes':path.stat().st_size}


def verify(bound):
    path = ROOT/bound['path']
    require(sha(path) == bound['sha256'], 'binding: '+bound['path'])
    return path


def main():
    require(not (OUT/'verification.json').exists(), 'fresh independent review')
    # Whitelist small observed outputs: refuse before opening any unexpected file.
    expected = {'.started.json','experiment/state.json','experiment/result.json',
        'experiment/publication-check.json','experiment/access.jsonl','experiment/output-watch.json',
        'experiment/supervision/supervision.json','experiment/supervision/combined.log'}
    observed = {str(p.relative_to(RUN)) for p in RUN.rglob('*') if p.is_file()}
    require(observed == expected, 'no fit, prediction, deck, preparation or solver output exists')
    output_before = {p:binding(RUN/p) for p in sorted(observed)}
    state,result,pub,supervision = [read(RUN/'experiment'/p) for p in
        ('state.json','result.json','publication-check.json','supervision/supervision.json')]
    started = read(RUN/'.started.json')
    release,launch,terminal,preflight,candidate = [read(BUILD/p) for p in
        ('release.json','launch.json','terminal.json','preflight.json','release.candidate.json')]
    require(sha(BUILD/'release.json') == 'd286f46e616036be399c2acd69741ce2fcb6a629ab13d62ebbe4fc74e9ac94f7', 'exact released authority')
    require(release == dict(candidate,authorized=True), 'only authorization changed from reviewed candidate')
    for bound in (result['release'],result['state'],pub['result'],release['study'],release['source_archive']):
        verify(bound)
    require(started == {'release':result['release'],'no_retry':True}, 'original no-retry marker retained')
    require(launch['release'] == result['release'] and launch['one_attempt'] is True, 'one launch exact release')
    require(launch['preflight_sha256'] == sha(BUILD/'preflight.json'), 'same-environment preflight receipt binding')
    require(preflight['exit_code'] == 0 and preflight['sanitized_as_worker'] is True, 'separate source preflight passed')
    require(preflight['native_calls'] == preflight['measured_members_read'] == 0, 'source preflight scope recorded')
    require(preflight['log_sha256'] == sha(BUILD/'preflight.log'), 'preflight log binding')
    require(terminal['exit_code'] == supervision['exit_code'] == 1, 'outer and worker exit failure')
    require(terminal['log_sha256'] == sha(BUILD/'execution.log'), 'outer execution log binding')
    require(result['supervision'] == supervision, 'saved result supervision exact')
    require(state['status'] == result['status'] == supervision['status'] == 'failed_or_incomplete', 'all terminal statuses preserve failure')
    require(pub['accepted'] is False, 'publication does not claim success')
    require(state['error'] == {'type':'ValueError','message':"could not convert string to float: 'displacement'"}, 'exact parser failure')
    require(state['calibration_access_attempted'] is True and state['calibration_responses_accessed'] is None, 'partial calibration exposure not reported absent or complete')
    require(state['held_out_access_attempted'] is False and state['held_out_responses_accessed'] is False, 'held-out flags remain sealed')
    require(state['native_calls'] == state['mesher_calls'] == 0 and state['runs'] == {}, 'zero new native/mesher runs')
    require(state['physical_validation_pass'] is result['physical_validation_pass'] is None, 'no physical result')
    require(state['automatic_retry'] is False and supervision['no_retry'] is True, 'no automatic retry')
    for key in ('fit','predictions','preparation','preparation_seconds','freeze','held_out_metrics','calibration_metrics'):
        require(key not in state, 'stage not reached: '+key)
    require(supervision['kill_reason'] is None and supervision['error'] is None and supervision['cleanup_error'] is None,
            'failure is application exception, not resource kill or cleanup error')
    study = read(verify(release['study']))
    roles = read(verify(study['roles']))
    require(release['execution']['csv_schemas'] == study['csv_schemas'], 'declared schema preserved')
    require(all(s['header'] is None for s in study['csv_schemas'].values()), 'executed schema explicitly assumes headerless members')
    require(release['permitted_members'] == [m['path'] for m in roles['calibration']['members']], 'exact released calibration order')
    require(release['conditional_held_out_members'] == [m['path'] for m in roles['held_out_validation']['members']], 'exact conditional torsion holdout')
    events = [json.loads(line) for line in (RUN/'experiment/access.jsonl').read_text().splitlines()]
    require(len(events) == 1, 'single durable access event')
    event = events[0]
    require(event['phase'] == 'calibration_attempt' and event['sequence'] == 0, 'calibration attempt only, no completion/freeze/reveal')
    require(event['members'] == release['permitted_members'], 'ledger lists intended batch, not per-member completion')
    require(event['release_sha256'] == result['release']['sha256'] and event['protocol_sha256'] == study['protocol']['sha256'], 'ledger authority binding')
    require(roles['calibration']['members'][0] == {'path':'HBE_01/HBE_01_03/compression_c3.csv','bytes':1222,'crc32':'c7b7a162'}, 'first member identity')
    require(roles['calibration']['members'][1]['path'] == 'HBE_01/HBE_01_03/tension_c3.csv', 'second member order')
    source_root = BUILD/'source'
    source_hashes = {}
    for name,bound in release['source_bindings'].items():
        source_hashes[name] = binding(verify(bound))
    prior_review = ROOT/'build/hbe-branch-calibration-runtime-v2-independent-review-v1/final-release-readiness.json'
    require(sha(prior_review) == 'a7576f8afbbba05a5a3bc8a0372ece02b50842ee256c913ca42db6ac3050ab3a', 'prior independent archive review unchanged')
    readiness = read(prior_review)
    require(release['source_commit'] == launch['source_commit'] == readiness['source_commit'], 'exact frozen source commit')
    require(release['source_archive']['sha256'] == readiness['source_archive']['sha256'], 'exact reviewed source archive')
    archived = {}
    with tarfile.open(verify(release['source_archive']),'r:') as archive:
        require(archive.pax_headers.get('comment') == release['source_commit'], 'source archive commit comment')
        for member in archive.getmembers():
            if member.isdir():continue
            require(member.isfile() and member.size < 2*1024**2, 'regular bounded source archive member')
            archived[member.name] = hashlib.sha256(archive.extractfile(member).read()).hexdigest()
    require(len(archived) == 38, 'unchanged complete 38-file source archive')
    for name,digest in archived.items():require(sha(source_root/name) == digest,'extracted source identity: '+name)
    access_text = (source_root/'scripts/mechanics_hbe_access.py').read_text()
    runner_text = (source_root/'scripts/mechanics_hbe_branch_calibration_v2_experiment.py').read_text()
    require(access_text.index('data=archive.read(info)') < access_text.index('result[branch]=parse_member_csv(data,branch,schemas[branch])'), 'full member read precedes parse')
    require("for item,branch in zip(members,branches,strict=True):" in access_text, 'selected members read and parsed serially')
    require('rows=list(csv.reader(io.StringIO(data.decode(\'utf-8\'))' in access_text, 'entire selected payload decoded into string rows before numeric conversion')
    require('x=[float(row[columns[0]]) for row in rows]' in access_text, 'failure is coordinate conversion before response conversion')
    require(runner_text.index('calibration=released.read_calibration') < runner_text.index('fit=core.evaluation.fit_scale'), 'fit follows successful whole calibration read')
    traceback = (RUN/'experiment/supervision/combined.log').read_text()
    for token in ("line 53, in worker","line 594, in read_calibration","line 583, in _read_selected",
                  "line 433, in parse_member_csv","could not convert string to float: 'displacement'"):
        require(token in traceback,'traceback confirms exact control-flow stage: '+token)
    require('freeze_predictions' not in traceback and 'old.solve' not in traceback, 'traceback contains no later stage')
    predecessor = {}
    for key,bound in study['runtime_migration'].items():
        if isinstance(bound,dict) and set(bound) == {'path','sha256'}:
            predecessor[key]=binding(verify(bound))
    require(len(predecessor) == 8, 'v1 failure and diagnosis remain byte-identical')
    process = subprocess.run(['ps','-axo','pid=,ppid=,pgid=,stat='],capture_output=True,text=True,check=True)
    target_ids = {launch['harness_pid'],supervision['pid']}
    remaining = []
    for line in process.stdout.splitlines():
        columns = line.split()
        if len(columns)>=4 and any(int(x) in target_ids for x in columns[:3]):remaining.append(columns)
    require(remaining == [], 'no recorded harness/worker/parent/group matches remain')
    raw_bytes = sum((RUN/p).stat().st_size for p in observed)
    archive_bytes = (ROOT/release['source_archive']['path']).stat().st_size
    publication_bytes = (RUN/'experiment/publication-check.json').stat().st_size
    require(archive_bytes == result['charged_source_archive_bytes'] == 890880, 'source archive charged')
    require(raw_bytes+archive_bytes == pub['retained_including_closeout'], 'actual retained bytes match closure')
    require(raw_bytes+archive_bytes-publication_bytes == pub['retained_including_source_archive_before_closeout'], 'publication-size arithmetic')
    require(pub['retained_including_closeout'] <= pub['output_cap_bytes'] == 2*1024**3, 'output budget respected')
    require(0 < supervision['elapsed_seconds'] <= result['seconds_before_publication'] <= pub['elapsed_through_result_fsync_and_scan_seconds'] < terminal['elapsed_seconds'] < 3600, 'nested actual runtime bounds')
    require(0 < supervision['sampled_peak_process_group_rss_bytes'] < supervision['rss_cap_bytes'] == 3*1024**3, 'sampled family RSS below bound')
    watcher = read(RUN/'experiment/output-watch.json')
    require(watcher['status'] == 'watching' and watcher['maximum_active_bytes'] == 0, 'watcher snapshot predates any active native stage')
    require({p:binding(RUN/p) for p in sorted(observed)} == output_before, 'all original outputs unchanged during audit')
    execution_records = {p:binding(BUILD/p) for p in ('launch.json','terminal.json','release.json','preflight.json','preflight.log','execution.log','run-detached-once.py')}
    report = {
        'schema':'hbe-branch-calibration-v2-failure-independent-review-v1',
        'status':'failed_attempt_verified_no_retry_no_fit_no_native_calls_holdout_sealed',
        'created_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'checks_passed':checks,
        'failure':state['error'],'output_bindings':output_before,'execution_bindings':execution_records,
        'source_commit':release['source_commit'],'source_archive':binding(ROOT/release['source_archive']['path']),
        'source_bindings_verified':source_hashes,'source_archive_members_verified':len(archived),
        'predecessor_v1_and_diagnosis_unchanged':predecessor,'prior_independent_source_review_sha256':sha(prior_review),
        'exposure':{
            'evidence_kind':'Deterministic inference from authenticated source order, exact role/release order and saved traceback. Existing ledger is batch-level, not a per-member syscall/read receipt.',
            'compression':{'path':roles['calibration']['members'][0]['path'],'uncompressed_bytes_read':1222,
                'crc32':'c7b7a162','archive_sha256':roles['source']['archive_sha256'],
                'full_payload_read_and_csv_rows_decoded':True,'numeric_curve_parse_completed':False,
                'first_non_numeric_coordinate_token_from_exception':'displacement',
                'individual_member_sha256':'not persisted; this audit does not reopen the member to compute one'},
            'tension':{'path':roles['calibration']['members'][1]['path'],'uncompressed_member_payload_read':False,'reason':'Exception while parsing first member occurs before next loop iteration'},
            'held_out_torsion':{'members':release['conditional_held_out_members'],'uncompressed_response_member_reads':0,'held_out_flags':False,'freeze_or_reveal_events':0,'remained_sealed':True},
            'raw_archive_authentication':{'entire_zip_bytes_hashed_before_selected_read':True,'archive_bytes':roles['source']['archive_bytes'],'archive_sha256':roles['source']['archive_sha256'],'meaning':'Compressed archive bytes, including held-out compressed bytes, are read only for authentication; this does not decode or expose their response contents. Post-read archive recheck is not reached after the exception.'},
            'state_interpretation':'calibration_responses_accessed=null correctly preserves incomplete/partial access; do not rewrite it false or label both members successfully read.',
        },
        'stages':{'runtime_source_reference_preflight_passed':True,'calibration_read_attempts':1,'complete_calibration_sets':0,'fits':0,'prediction_artifacts':0,'deck_preparation_stages':0,'new_native_solver_calls':0,'mesher_calls':0,'parameter_freezes':0,'held_out_evaluations':0,'physical_validation_pass':None},
        'lifecycle':{'harness_pid':launch['harness_pid'],'worker_session_leader_pid':supervision['pid'],'worker_exit_code':supervision['exit_code'],'outer_exit_code':terminal['exit_code'],'kill_reason':supervision['kill_reason'],'cleanup_error':supervision['cleanup_error'],'no_retry':True,'current_numeric_process_snapshot_matches':remaining,'snapshot_time_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'watcher_note':'output-watch.json retains status=watching, but authenticated __exit__ stops/joins its thread; terminal supervisor and absent process group establish closure. Treat watcher status as last observation, not a live process or successful result.'},
        'resources':{'separate_prerequisite_preflight_seconds':preflight['elapsed_seconds'],'worker_supervision_seconds':supervision['elapsed_seconds'],'through_result_publication_seconds':pub['elapsed_through_result_fsync_and_scan_seconds'],'outer_harness_seconds':terminal['elapsed_seconds'],'sampled_peak_family_rss_bytes':supervision['sampled_peak_process_group_rss_bytes'],'raw_output_bytes':raw_bytes,'source_archive_bytes':archive_bytes,'retained_total_bytes':raw_bytes+archive_bytes,'resource_failure':False,'interpretation':'Supervision/publication/outer times overlap; do not sum. Separate pre-launch preflight is an additional preparation cost outside actual attempt clock. Memory sampling may miss short peaks.'},
        'next_version_controls':[
            'Preserve v1 runtime failure and v2 parser failure, exact original releases/ledgers/started markers; new version/output root and fresh explicit release only, never implicit retry.',
            'Replace unsupported header:null assumptions with prospectively pinned exact header schemas supported by bounded header-only evidence or authoritative source documentation. The present exception proves only the token displacement, not the full header or other member schemas. Keep units, column order and signs explicit; reject unrecognized/duplicate headers, never auto-skip arbitrary nonnumeric rows.',
            'Do not inspect torsion response values before the existing durable fit/prediction freeze and numerical confirmation. Any separately authorized header-only inspection must stop at the header and record that narrow access; otherwise validate torsion schema only at conditional reveal.',
            'Add per-member durable read-attempt and payload-read/parse-failed completion records including member identity/byte length/hash before numerical conversion, so a first-member parser failure never depends on a batch-attempt inference.',
            'Pure controls: matching literal-header and headerless CSV contracts parse; wrong declared header fails before fitting; injected first-member parse error records only that member as read and never opens second/held-out members or starts fit/deck/solver; existing output/no-retry marker refuses reuse.',
            'Keep original specimen/donor roles, fitting objective, frozen branches, tolerances, solver/profile and resource budgets; no outcome-based replacement or tuning based on this format failure.'
        ],
        'limitations':['No measured archive/member, native primitive or new scientific computation was accessed in this audit.','Exact first-member exposure is source/traceback-derived because the old ledger lists both intended calibration members and has no individual read-completion entry.','Failure is a declared CSV schema mismatch, not evidence for or against the physical model.','No automatic retry or new measured execution is authorized by this review.'],
        'audit_scope':{'measured_archive_or_curve_reads':0,'native_primitive_reads':0,'model_fits':0,'solver_calls':0,'tracked_writes':0,'source_writes':0},
        'audit_script_sha256':sha(Path(__file__)),
    }
    (OUT/'verification.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
    print(json.dumps({'status':report['status'],'checks':checks,'verification_sha256':sha(OUT/'verification.json'),'raw_output_bytes':raw_bytes,'total_output_plus_source_bytes':raw_bytes+archive_bytes,'remaining_processes':remaining}))


if __name__ == '__main__':
    main()

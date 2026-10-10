"""Native evidence admission above the frozen v5 numerical comparison.

The public entry point needs twelve complete native rows, a root-reviewed exact
input manifest and all policy sidecars. It reuses the original native-chain
verifier and frozen numerical math. No solver, replay, response loader, fitter or
release writer exists here. A partial study is a completeness refusal, not a new
scientific requirement. Archived independent prose is valid evidence when its
exact hash and the reviewed input/output identity mapping are supplied.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat
import sys

from scripts import mechanics_hbe_v5_comparison as comparison
from scripts import mechanics_hbe_branch_calibration_v5 as v5
from scripts import mechanics_hbe_v5_source_bindings as sources
from scripts import mechanics_hbe_v5_remaining_one_shot as remaining
from scripts import mechanics_hbe_v5_n12_admission as supplement

SCHEMA = 'hbe-v5-native-numerical-comparison-v1'
MAX_JSON_BYTES = 16 * 1024**2
MAX_METADATA_BYTES = 1024**2
HEX = re.compile(r'[a-f0-9]{64}\Z')
V2_SIDECAR_SHA = 'dae900fcedd2dc61f1398a00e40be40b9fa355146c20599c6f01fc5a29b45eb8'
HISTORICAL_TIMING_GAP = 'Historical N8 readout and preparation wall time were not recorded.'


def _need(condition, reason):
    if not condition:
        raise ValueError(reason)


def _digest(raw):
    return hashlib.sha256(raw).hexdigest()


def _json_no_duplicates(pairs):
    result = {}
    for key, value in pairs:
        _need(key not in result, 'duplicate_json_key')
        result[key] = value
    return result


def _bound_bytes(root, binding, *, maximum=MAX_METADATA_BYTES, suffixes=('.json',),
                 directories=('build', 'artifacts', 'outputs', 'manifests')):
    """One bounded regular-file read; metadata/readout JSON only, no raw logs."""
    _need(isinstance(binding, dict) and set(binding) == {'path', 'sha256'}, 'exact_json_binding')
    relative = Path(binding['path'])
    _need(isinstance(binding['sha256'], str) and HEX.fullmatch(binding['sha256']) is not None,
          'full_sha256_required')
    _need(not relative.is_absolute() and '..' not in relative.parts and relative.suffix in suffixes
          and relative.parts and relative.parts[0] in directories,
          'metadata_path_scope')
    if relative.parts[0] == 'outputs':
        _need(relative.parts[:2] == ('outputs', 'mechanics'), 'metadata_path_scope')
    root = Path(root).resolve()
    path = root / relative
    _need(not any(p.is_symlink() for p in (path, *path.parents) if p != root and p.is_relative_to(root)),
          'linked_metadata_path')
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    try:
        before = os.fstat(fd)
        _need(stat.S_ISREG(before.st_mode) and 0 < before.st_size <= maximum, 'bounded_regular_json_required')
        chunks, total = [], 0
        while total <= maximum:
            piece = os.read(fd, min(65536, maximum + 1 - total))
            if not piece:
                break
            chunks.append(piece); total += len(piece)
        raw = b''.join(chunks)
        after = os.fstat(fd)
        identity = lambda s: (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns)
        _need(identity(before) == identity(after) == identity(path.lstat())
              and len(raw) == before.st_size and len(raw) <= maximum
              and _digest(raw) == binding['sha256'], 'bound_metadata_bytes_changed')
    finally:
        os.close(fd)
    return raw


def _bound_json(root, binding, *, maximum=MAX_METADATA_BYTES):
    raw = _bound_bytes(root, binding, maximum=maximum)
    return _decode_json(raw)


def _decode_json(raw):
    value = json.loads(raw, object_pairs_hook=_json_no_duplicates,
                       parse_constant=lambda _: (_ for _ in ()).throw(ValueError('nonfinite_json')))
    _need(isinstance(value, dict), 'json_object_required')
    return value


class _Evidence:
    """Keep exact byte identities for the final small-evidence recheck."""
    def __init__(self, root):
        self.root, self.bindings = root, {}

    def read(self, binding, **options):
        raw = _bound_bytes(self.root, binding, **options)
        previous = self.bindings.setdefault(binding['path'], (dict(binding), dict(options)))
        _need(previous[0] == binding, 'conflicting_evidence_binding')
        return raw

    def json(self, binding, **options):
        return _decode_json(self.read(binding, **options))

    def recheck(self):
        for binding, options in self.bindings.values():
            _bound_bytes(self.root, binding, **options)


def _verify_comparison_sources(root, bindings):
    """Bind executing repository source and loaded origins, not runtime binaries.

    A separately supervised fresh interpreter/cache/runtime policy remains the
    caller's responsibility. This guard does not claim Python bytecode provenance.
    """
    root = Path(root).resolve()
    _need(isinstance(bindings, dict) and 1 <= len(bindings) <= 128, 'bounded_source_inventory')
    for path, digest in bindings.items():
        _bound_bytes(root, {'path':path,'sha256':digest}, suffixes=('.py',),
                     directories=('scripts','launchers','src'))
    origins = {}
    for name, module in tuple(sys.modules.items()):
        if not name.startswith(('scripts.', 'launchers.', 'resectionlab.')):
            continue
        filename = getattr(module, '__file__', None)
        _need(isinstance(filename, str), 'loaded_project_module_without_origin')
        path = Path(filename).resolve()
        _need(path.is_relative_to(root), 'shadow_project_module')
        relative = str(path.relative_to(root))
        expected = name.replace('.', '/') + '.py'
        if name.startswith('resectionlab.'):
            expected = 'src/' + expected
        _need(relative in (expected, expected[:-3]+'/__init__.py')
              and relative in bindings, 'unbound_loaded_project_source')
        origins[name] = relative
    required = (__name__, comparison.__name__, v5.__name__, sources.__name__,
                remaining.__name__, supplement.__name__)
    _need(all(name in origins for name in required), 'required_comparison_source_missing')
    return origins


def _review_mapping(mapping, *, run_id, receipt_sha, release_sha, readout_sha,
                    source_sha, adapted_sha, mesh_sha, exception):
    """Transparent identity mapping of existing archived independent evidence.

    This is not a new review certificate or a scientific gate. The outer trusted
    input manifest binds the mapping; the reader authenticates the original
    review bytes and optional replay-comparison JSON. An exact historical prose
    report is sufficient. No loose native supervisor 'passed' field substitutes
    for the named independent evidence.
    """
    _need(mapping.get('run_id') == run_id and mapping.get('native_receipt_sha256') == receipt_sha
          and mapping.get('release_sha256') == release_sha and mapping.get('readout_sha256') == readout_sha
          and mapping.get('source_deck_sha256') == source_sha
          and mapping.get('adapted_deck_sha256') == adapted_sha
          and mapping.get('native_mesh_sha256') == mesh_sha
          and mapping.get('exception_policy') == exception, 'independent_review_identity_mapping')
    _need(mapping.get('evidence_kind') in ('archived_prose_exact_replay','machine_replay_comparison')
          and mapping.get('comparison_scope') == ('all_non_provenance_fields' if run_id == remaining.ORDER[0]
                                                   else 'entire_readout_json')
          and isinstance(mapping.get('supporting_passage'),str) and 20 <= len(mapping['supporting_passage']) <= 2048
          and isinstance(mapping.get('review_binding'),dict), 'explicit_existing_independent_evidence_required')


def _authenticate_review(root, mapping, evidence=None):
    evidence = evidence or _Evidence(root)
    raw = evidence.read(mapping['review_binding'], maximum=256*1024, suffixes=('.md','.txt'))
    _need(mapping['supporting_passage'].encode() in raw, 'independent_review_passage_changed')
    if mapping['evidence_kind'] == 'machine_replay_comparison':
        value = evidence.json(mapping['comparison_binding'])
        _need(value.get('entire_json_equal') is True and value.get('native_calls') == 0
              and value.get('saved_stream_replay_calls') == 1
              and value.get('source_after_matches') is True
              and value.get('saved_outputs_after_match') is True, 'independent_comparison_evidence_failed')


def _finite_nonnegative(value):
    _need(type(value) in (int,float) and math.isfinite(value) and value >= 0, 'finite_resource_quantity')
    return value


def extension_charge(sidecar, envelope, native, *, descriptor, native_sha, closed_native_bytes,
                     prior_surcharge_seconds, prior_surcharge_bytes):
    """Check saved wrapper identities and add only its incremental overhead.

    Fixed original receipts stay unchanged. Every extension is explicitly mapped
    in the reviewed manifest, including its exact envelope and sidecar hashes;
    no generic native-chain acceptance silently drops wrapper costs.
    """
    _need(sidecar.get('schema') == descriptor['schema'] and sidecar.get('status') == descriptor['status']
          and sidecar.get('native_calls_attempted') == 1
          and sidecar.get('hbe_readout_calls_attempted') == 1
          and sidecar.get('native_receipt_sha256') == native_sha
          and sidecar.get('native_receipt_status') == 'passed_numerical_software_only'
          and sidecar.get('envelope_sha256') == descriptor['envelope']['sha256']
          and sidecar.get('inner_release_sha256') == native['release_sha256']
          and envelope.get('inner_release',{}).get('sha256') == native['release_sha256']
          and sidecar.get('source_commit') == envelope.get('source_commit') == native['source_commit']
          and sidecar.get('policy') == envelope.get('policy'), 'extension_native_release_identity')
    policy=sidecar['policy']; before=sidecar.get('extension_sources_before',{}); after=sidecar.get('extension_sources_after',{})
    _need(before.get('source_commit') == after.get('source_commit') == native['source_commit']
          and before.get('source_hashes') == after.get('source_hashes') == envelope.get('extension_source_bindings')
          and isinstance(before.get('source_hashes'),dict) and before['source_hashes'], 'extension_source_guards')
    for stage in ('native','readout'):
        cleanup=sidecar.get('owned_stage_cleanup',{}).get(stage,{})
        _need(cleanup.get('contained') is True and cleanup.get('direct_child_reaped') is True
              and cleanup.get('fallback_used') is False and cleanup.get('errors') == []
              and cleanup.get('remaining_members') == [] and cleanup.get('exit_code') == 0, 'extension_cleanup')
    for field,floor in (('host_before_validation','initial_available_percent_floor'),
                        ('host_before_native_reservation','pre_native_available_percent_floor'),
                        ('host_immediately_before_native_supervision','pre_native_available_percent_floor')):
        host=sidecar.get(field,{})
        _need(host.get('kernel_pressure_mask') == policy['normal_kernel_pressure_mask']
              and type(host.get('available_percent')) is int and host['available_percent'] >= policy[floor], 'extension_host_policy')
    for field,prefix in (('preflight_hint_audit','preflight'),('final_hint_audit','completed')):
        hints=sidecar.get(field,{})
        _need(hints.get('eligible_files') == hints.get('hinted_file_opens') == policy['expected_'+prefix+'_hint_opens']
              and hints.get('eligible_bytes') == policy['expected_'+prefix+'_hint_bytes']
              and len(hints.get('hinted_file_paths',[])) == hints['eligible_files'], 'extension_hint_coverage')
    _need(sidecar['final_hint_audit']['hinted_file_paths'] == sidecar['preflight_hint_audit']['hinted_file_paths']*2,
          'extension_hint_path_sequence')
    for field in ('extension_before_old_execute_wall_seconds','extension_after_old_execute_wall_seconds'):
        _need(_finite_nonnegative(sidecar.get(field)) < policy[field], 'extension_wall_cap')
    for suffix in ('before','after','terminal'):
        _need(_finite_nonnegative(sidecar.get('extension_self_peak_rss_bytes_'+suffix)) <= policy['extension_self_peak_rss_bytes'],
              'extension_rss_cap')
    charge=sidecar['extension_resource_charge']; prep=_finite_nonnegative(charge['launcher_inclusive_prep_seconds_with_finalization_reserve'])
    reserved=charge['sidecar_reserved_output_bytes']
    _need(prep < remaining.PREP_WALL and prep >= native['prep_elapsed_seconds']
          and type(reserved) is int and reserved == policy['sidecar_output_cap_bytes']
          and charge['native_closed_output_bytes_at_check'] == closed_native_bytes
          and charge['charged_current_output_bytes'] == closed_native_bytes+reserved
          and closed_native_bytes <= native['caps']['active_output_bytes']
          and charge['native_stage_seconds'] == native['native_stage']['elapsed_seconds']
          and charge['readout_stage_seconds'] == native['readout_stage']['elapsed_seconds'], 'extension_charge_identity')
    total = _finite_nonnegative(charge['launcher_elapsed_seconds_at_check'])
    terminal = _finite_nonnegative(sidecar['launcher_elapsed_seconds_terminal'])
    native_time, readout_time = charge['native_stage_seconds'], charge['readout_stage_seconds']
    _need(prep == max(0., total-native_time-readout_time)+policy['finalization_prep_reserve_seconds']
          and charge['per_row_prep_wall_seconds'] == remaining.PREP_WALL
          and terminal >= total
          and max(0.,terminal-native_time-readout_time) <= prep
          and max(0.,terminal-native_time-readout_time) < remaining.PREP_WALL,
          'extension_prep_clock_identity')
    previous=dict(sidecar['old_validation_identity']['previous'])
    _need(previous['sha256'] == native['prior_receipt_sha256'], 'extension_original_predecessors')
    for key,field in (('native_seconds','prior_native_wall_seconds'),('readout_seconds','prior_readout_wall_seconds'),
                      ('prep_seconds','prior_prep_wall_seconds'),('output_bytes','prior_active_output_bytes'),
                      ('combined_wall_seconds','prior_combined_wall_seconds'),('combined_output_bytes','prior_combined_output_bytes')):
        _need(previous[key] == native[field], 'extension_original_predecessor_accounting')
    previous['prep_seconds']+=prior_surcharge_seconds; previous['combined_wall_seconds']+=prior_surcharge_seconds
    previous['output_bytes']+=prior_surcharge_bytes; previous['combined_output_bytes']+=prior_surcharge_bytes
    if 'charged_previous' in sidecar:
        _need(sidecar['charged_previous'] == previous, 'extension_prior_surcharge_once')
    else:
        _need(prior_surcharge_seconds == prior_surcharge_bytes == 0, 'missing_continuation_charged_previous')
    expected=remaining._check_aggregate(previous,native['native_stage']['elapsed_seconds'],
        native['readout_stage']['elapsed_seconds'],prep,closed_native_bytes+reserved,1)
    _need(charge['aggregate_with_extension'] == expected, 'extension_aggregate_math')
    return {'incremental_prep_seconds':prep-native['prep_elapsed_seconds'],
            'incremental_output_bytes':reserved,'charged_aggregate':expected}


def validate_native_row(study, prior, run_id, *, readout, receipt, receipt_sha,
                        release, release_sha, work_order, independent, source_map,
                        readout_sha, exact_supplement=None):
    """Pure compact contract check after native chain authentication.

    This function returns numerical views, never an admission token or permission.
    Input dictionaries alone cannot establish native origin. The public entry point
    must authenticate all bound bytes and the original plus extension-policy chain.
    No stream provenance is changed to reuse the generated-only entry point.
    """
    _need(run_id in remaining.ORDER, 'declared_run_required')
    ordinal = remaining.ORDER.index(run_id)
    spec = v5.run_spec(study, prior, run_id)
    key = source_map['run_source_keys'][run_id]
    source = source_map['source_decks'][key]
    mesh = source_map['meshes'][source['mesh']]
    source_binding = {k: source[k] for k in ('path', 'sha256')}
    half = spec['native_domain'] == 'lower_half_reconstructed'
    _need(readout.get('schema') == 'hbe-v5-complete-stream-v1'
          and readout.get('run_id') == run_id and readout.get('steps') == spec['steps']
          and readout.get('frame_count') == spec['frame_count']
          and readout.get('representation') == ('reconstructed_full' if half else 'full_native_fixture'),
          'native_row_identity_or_representation')
    provenance = readout.get('provenance', {})
    _need(provenance.get('v5_declaration_sha256') == v5.DECLARATION_SHA256
          and provenance.get('source_deck_sha256') == source['sha256']
          and provenance.get('adapted_deck_sha256') == receipt.get('adapted_deck_sha256')
          and provenance.get('source_binding_checked') is True
          and provenance.get('measured_response_accessed') is False
          and provenance.get('patient_data_accessed') is False
          and provenance.get('physical_validation_pass') is None,
          'native_source_or_scope_binding')
    _need(receipt.get('run_id') == run_id and receipt.get('release_sha256') == release_sha
          and receipt.get('native_calls_attempted') == 1 and receipt.get('no_retry') is True
          and release.get('source_commit') == receipt.get('source_commit')
          and release.get('adapted_deck_sha256') == receipt.get('adapted_deck_sha256')
          and receipt.get('old_source_deck_sha256') == source['sha256']
          and receipt.get('native_mesh_sha256') == mesh['sha256'], 'native_receipt_release_binding')
    exception = None
    if ordinal == 0:
        _need(receipt_sha == remaining.N8_SHA and receipt.get('status') == 'passed_numerical_software_only'
              and release_sha == remaining.N8_RELEASE_SHA
              and receipt.get('source_commit') == remaining.N8_SOURCE_COMMIT
              and receipt.get('adapted_deck_sha256') == remaining.N8_DECK_SHA
              and receipt.get('saved_numerical_readout') == readout and readout_sha == receipt_sha,
              'fixed_n8_embedded_readout')
        _need(provenance.get('output_origin') == 'one_supervised_native_FEBio_attempt'
              and provenance.get('native_output_observed') is True
              and provenance.get('generated_fixture_only') is False, 'fixed_n8_provenance')
        _need(exact_supplement is None and work_order is None, 'unexpected_n8_supplement_or_work_order')
        native = receipt.get('native_output_bindings', {})
        primitives = {'source_deck': source_binding, 'mesh': mesh,
                      **{k: {field: native[name][field] for field in ('path', 'sha256')}
                         for k, name in (('adapted_deck','specimen.feb'), ('nodes','nodes.log'),
                                         ('elements','elements.log'), ('solver','solver.log'))}}
    else:
        _need(provenance.get('output_origin') == 'unverified_saved_stream'
              and provenance.get('native_output_observed') is None
              and provenance.get('generated_fixture_only') is None
              and provenance.get('reconstruction_provenance') ==
                  ('reflected_native_half_not_native_full' if half else 'full_native'), 'saved_stream_origin_flags')
        records = receipt.get('output_bindings', {})
        if ordinal == 1:
            _need(receipt_sha == remaining.N12_FAILED_RECEIPT_SHA
                  and receipt.get('status') == 'failed_or_incomplete'
                  and isinstance(exact_supplement, dict)
                  and exact_supplement.get('original_receipt_sha256') == receipt_sha
                  and exact_supplement.get('supplement_receipt_sha256') == supplement.SUPPLEMENT_SHA
                  and exact_supplement.get('historical_guard_gap') == supplement.GAP
                  and exact_supplement.get('readout') == readout,
                  'only_exact_failed_n12_supplement')
            exception = {'original_status': 'failed_or_incomplete',
                         'supplement_receipt_sha256': supplement.SUPPLEMENT_SHA,
                         'historical_guard_gap': supplement.GAP}
            # Failed packaging does not guarantee an ordinary output_bindings map.
            records = exact_supplement['original_file_bindings']
        else:
            _need(exact_supplement is None and receipt.get('status') == 'passed_numerical_software_only'
                  and receipt.get('saved_numerical_readout', {}).get('readout_sha256') == readout_sha,
                  'passed_native_receipt_required')
        _need(isinstance(work_order, dict) and work_order.get('run_id') == run_id
              and work_order.get('source_commit') == receipt.get('source_commit')
              and work_order.get('release_sha256') == release_sha
              and work_order.get('readout_token_sha256') == receipt.get('readout_token_sha256'),
              'native_readout_work_order_binding')
        primitives = work_order.get('bindings', {})
        _need(set(primitives) == {'source_deck','adapted_deck','mesh','nodes','elements','solver'}
              and primitives['source_deck'] == source_binding and primitives['mesh'] == mesh
              and provenance.get('primitive_bindings') == primitives
              and records.get('readout.json', {}).get('sha256') == readout_sha,
              'complete_native_primitive_bindings')
        for key, name in (('adapted_deck','specimen.feb'), ('nodes','nodes.log'),
                          ('elements','elements.log'), ('solver','solver.log')):
            _need(primitives[key] == {field: records[name][field] for field in ('path','sha256')},
                  'native_primitive_identity_mismatch')
    _need(primitives['adapted_deck']['sha256'] == receipt['adapted_deck_sha256'], 'adapted_deck_identity')
    _review_mapping(independent, run_id=run_id, receipt_sha=receipt_sha, release_sha=release_sha,
        readout_sha=readout_sha, source_sha=source['sha256'],
        adapted_sha=receipt['adapted_deck_sha256'], mesh_sha=mesh['sha256'], exception=exception)
    return comparison._numerical_view(study, prior, run_id, readout)


def compare_native_rows(root, reviewed_manifest_binding):
    """Compare twelve admitted saved native rows; never execute a native solve.

    The reviewed manifest SHA anchors existing review mappings and sidecars.
    Pre/post native-chain and exact-supplement verification hashes raw files;
    the caller must budget BOTH passes, including the verifier's repeated N12
    checks, and supervise a fresh runtime. This function writes no release or
    output. Generated orchestration tests replace only the explicit I/O seams.
    """
    root = Path(root).resolve()
    evidence = _Evidence(root)
    manifest = evidence.json(reviewed_manifest_binding)
    _need(manifest.get('schema') == 'hbe-v5-native-comparison-inputs-v1'
          and manifest.get('ordered_run_ids') == list(remaining.ORDER)
          and isinstance(manifest.get('rows'), dict)
          and set(manifest['rows']) == set(remaining.ORDER), 'exact_twelve_native_rows_required')
    _need(manifest.get('measured_response_access') is False
          and manifest.get('patient_data_access') is False
          and manifest.get('native_execution_release') is False, 'numerical_comparison_scope_only')
    _need(isinstance(manifest.get('native_receipts'), list)
          and len(manifest['native_receipts']) == 12, 'twelve_receipt_bindings_required')
    origins_before = _verify_comparison_sources(root, manifest['comparison_source_bindings'])
    study, prior = v5.validate_preparation(root)
    chain = remaining.validate_prior_chain(manifest['native_receipts'], 12, root=root,
        expected_profile=manifest['backend_profile'], expected_runtime=manifest['runtime_identity'])
    exact = supplement.verify_exact(root=root)
    supplement_receipt = evidence.json({'path':supplement.replay.OUTPUT+'/receipt.json',
                                       'sha256':supplement.SUPPLEMENT_SHA}, maximum=MAX_JSON_BYTES)
    source_map = evidence.json({'path':sources.BINDING_PATH,'sha256':sources.BINDING_SHA256})
    views, individual = {}, {}
    receipts, receipt_sizes = {}, {}
    policy = {'incremental_prep_seconds':0.,'incremental_output_bytes':0,'extensions':{}}
    for index, run_id in enumerate(remaining.ORDER):
        entry = manifest['rows'][run_id]
        binding = manifest['native_receipts'][index]
        _need(binding['run_id'] == run_id, 'native_receipt_order')
        raw = evidence.read({k:binding[k] for k in ('path','sha256')}, maximum=MAX_JSON_BYTES)
        receipt = _decode_json(raw)
        receipts[run_id], receipt_sizes[run_id] = receipt, len(raw)
        release = evidence.json(entry['release'])
        readout = receipt['saved_numerical_readout'] if index == 0 else evidence.json(entry['readout'], maximum=MAX_JSON_BYTES)
        work = None if index == 0 else evidence.json(entry['work_order'])
        independent = entry['independent_mapping']
        _authenticate_review(root, independent, evidence)
        narrow = None if index != 1 else {**exact,'original_file_bindings':supplement_receipt['original_file_bindings']}
        views[run_id], individual[run_id] = validate_native_row(study, prior, run_id,
            readout=readout, receipt=receipt, receipt_sha=binding['sha256'], release=release,
            release_sha=entry['release']['sha256'], work_order=work, independent=independent,
            source_map=source_map, readout_sha=binding['sha256'] if index == 0 else entry['readout']['sha256'],
            exact_supplement=narrow)
    # The known v2 event and every later continuation require explicit sidecars.
    # Their schemas/envelopes are reviewed manifest data; no wrapper is modified.
    for index, run_id in enumerate(remaining.ORDER):
        entry=manifest['rows'][run_id]
        _need('policy_extension' in entry, 'explicit_extension_disclosure_required')
        descriptor=entry['policy_extension']
        _need((index < 8 and descriptor is None) or (index >= 8 and isinstance(descriptor,dict)),
              'continuation_policy_evidence_required')
        if index == 8:
            _need(descriptor['sidecar']['sha256'] == V2_SIDECAR_SHA, 'exact_v2_policy_event_required')
        if descriptor is None:
            continue
        sidecar=evidence.json(descriptor['sidecar']); envelope=evidence.json(descriptor['envelope'])
        bindings=envelope['extension_source_bindings']
        _need(isinstance(bindings,dict) and 1 <= len(bindings) <= 32, 'bounded_extension_source_inventory')
        for path, digest in bindings.items():
            _need(isinstance(path,str) and path.startswith(('scripts/','launchers/'))
                  and '..' not in Path(path).parts and path.endswith('.py')
                  and isinstance(digest,str) and HEX.fullmatch(digest), 'extension_source_scope')
        # Reuse historical blob authentication; current working versions may have
        # advanced after the exact released event and need not impersonate it.
        supplement.replay.committed_source(envelope['source_commit'],
            {path:{'path':path,'sha256':digest} for path,digest in bindings.items()},
            paths=tuple(bindings), root=root)
        native=receipts[run_id]
        closed=receipt_sizes[run_id]+sum(record['bytes'] for record in native['output_bindings'].values())
        extra=extension_charge(sidecar,envelope,native,descriptor=descriptor,
            native_sha=manifest['native_receipts'][index]['sha256'],closed_native_bytes=closed,
            prior_surcharge_seconds=policy['incremental_prep_seconds'],
            prior_surcharge_bytes=policy['incremental_output_bytes'])
        policy['incremental_prep_seconds']+=extra['incremental_prep_seconds']
        policy['incremental_output_bytes']+=extra['incremental_output_bytes']
        policy['extensions'][run_id]={'sidecar':descriptor['sidecar'],'envelope':descriptor['envelope'],**extra}
    combined=dict(chain)
    for key in ('prep_seconds','combined_wall_seconds'):combined[key]+=policy['incremental_prep_seconds']
    for key in ('output_bytes','combined_output_bytes'):combined[key]+=policy['incremental_output_bytes']
    _need(combined['prep_seconds'] <= remaining.AGGREGATE['remaining_prep_wall_seconds']
          and combined['combined_wall_seconds'] <= remaining.AGGREGATE['known_stage_total_wall_seconds']
          and combined['combined_output_bytes'] <= remaining.AGGREGATE['active_output_bytes_all_12'], 'complete_policy_aggregate_caps')
    historical = comparison._historical_n32(root, prior)
    result = comparison._compare_views(prior, list(remaining.ORDER), views, individual, historical)
    # No partially checked result escapes if any input/source changed late.
    _need(v5.validate_preparation(root) == (study,prior)
          and comparison._historical_n32(root,prior) == historical, 'declarations_changed_after_comparison')
    _need(remaining.validate_prior_chain(manifest['native_receipts'],12,root=root,
          expected_profile=manifest['backend_profile'],expected_runtime=manifest['runtime_identity']) == chain
          and supplement.verify_exact(root=root) == exact, 'native_chain_changed_after_comparison')
    evidence.recheck()
    origins_after = _verify_comparison_sources(root, manifest['comparison_source_bindings'])
    _need(origins_before.items() <= origins_after.items(), 'comparison_import_origins_changed')
    return {**result,'schema':SCHEMA,'native_output_admitted':True,'native_execution_released':False,
        'numerical_qualification_passed':result['diagnostic_screen_passed'],
        'source_deck_bytes_authenticated':True,'adapted_deck_authenticated':True,
        'comparison_manifest':dict(reviewed_manifest_binding),'comparison_source_bindings':manifest['comparison_source_bindings'],
        'loaded_project_origins_before':origins_before,'loaded_project_origins_after':origins_after,
        'small_evidence_rechecked':len(evidence.bindings),'native_chain_verification_passes':2,
        'original_chain_accounting':chain,'extension_policy_accounting':policy,'combined_policy_accounting':combined,
        'exact_N12_exception':{k:exact[k] for k in ('original_receipt_sha256','supplement_receipt_sha256',
                                                  'original_accounting','supplement_overhead','historical_guard_gap')},
        'known_timing_gap':HISTORICAL_TIMING_GAP,'physical_validation_pass':None,
        'calibration_released':False,'measured_response_accessed':False,'patient_data_accessed':False}

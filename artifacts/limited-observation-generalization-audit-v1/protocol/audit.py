#!/usr/bin/env python3
"""Read-only metadata audit. Does not import the project or decode patient data."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    protocol_path = Path(__file__).with_name('protocol.json')
    p = read(protocol_path)
    errors, counts = [], {}
    hashes = {}
    for name, expected in p['evidence_sha256'].items():
        path = root / name
        actual = digest(path) if path.is_file() else None
        hashes[name] = {'expected': expected, 'actual': actual, 'match': actual == expected}
        if actual != expected:
            errors.append('SOURCE_CHANGED_OR_MISSING:' + name)
    cohort = read(root / 'manifests/experiments/btc-spatial-development-cohort-v1.json')
    rows = cohort['existing_development_records'] + cohort['candidates']
    role_map = {'previously_consulted_training_and_method_development': 'train',
                'population_training': 'train', 'checkpoint_selection_development': 'select',
                'unopened_frozen_development_transfer': 'unopened_development_transfer'}
    actual = {role: [] for role in role_map.values()}
    identities = set()
    for row in rows:
        person = row['patient_group']
        if person in identities:
            errors.append('DUPLICATE_BTC_PATIENT:' + person)
        identities.add(person)
        role = role_map.get(row['development_role'])
        if role is None:
            errors.append('UNKNOWN_BTC_ROLE:' + person)
            continue
        actual[role].append(person)
        if row.get('outer_role') != 'development' or row.get('eligible_for_external_final') is not False:
            errors.append('BTC_FINAL_ROLE_DRIFT:' + person)
    for role, members in actual.items():
        if sorted(members) != sorted(p['outer_roles'][role]):
            errors.append('BTC_ROLE_DRIFT:' + role)
        counts['BTC_' + role] = len(members)
    for family, path, id_key in [
        ('Lausanne','manifests/lausanne-component-cohort-v1.json','canonical_person'),
        ('RESECT','manifests/resect-component-cohort-v1.json','patient_group'),
        ('SynthRAD','manifests/synthrad-component-cohort-v1.json','patient_group')]:
        members = read(root / path)['members']
        seen = {}
        for row in members:
            person = row[id_key]
            if person in seen:
                errors.append('DUPLICATE_OR_CROSS_ROLE_PERSON:' + person)
            seen[person] = row['role']
        counts[family] = dict(sorted(Counter(row['role'] for row in members).items()))
    registry = read(root / 'manifests/cohort_registry.json')
    # Preserve the registry's explicitly incomplete historical scope.
    counts['cohort_registry_records_not_complete_inventory'] = len(registry['records'])
    final_records = [r for r in registry['records'] if r.get('outer_role', r.get('outer_split')) == 'final']
    counts['cohort_registry_final_records'] = len(final_records)
    result = read(root / 'artifacts/prepared-training-planner-comparison-v2/compact-results.json')
    historical = {k: result[k] for k in ('patients_prescribed','complete_comparisons','accepted_arms','optimizer_updates','clinical_efficacy_measured')}
    report = {
        'schema':'generalization-protocol-metadata-audit-v1',
        'protocol_sha256':digest(protocol_path),
        'metadata_audit_passed':not errors,
        'experiment_ready':False,
        'release_performed':False,
        'payloads_decoded':0, 'models_loaded':0, 'simulation_steps':0, 'optimizer_updates':0,
        'counts':counts,'historical_prepared_comparison':historical,
        'unresolved_gates':[g['id'] for g in p['current_gaps']],
        'source_hash_checks':hashes,'errors':errors,
        'interpretation':'Passing validates this metadata snapshot only. It neither admits datasets/models nor establishes generalization or scientific readiness.'}
    content = json.dumps(report,indent=2)+'\n'
    if args.output:
        target = args.output.resolve()
        output_dir = Path(__file__).resolve().parent
        if target.parent != output_dir:
            parser.error('--output must remain inside this protocol build directory')
        if target.exists():
            parser.error('--output must be new; preserve previous audits')
        target.write_text(content)
    print(json.dumps({k:report[k] for k in ('metadata_audit_passed','experiment_ready','counts','errors')},indent=2))
    return 0 if not errors else 1


if __name__ == '__main__':
    raise SystemExit(main())

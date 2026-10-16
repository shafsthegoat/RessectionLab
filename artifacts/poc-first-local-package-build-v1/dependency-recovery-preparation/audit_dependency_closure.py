"""Read failed PyInstaller TOCs and installed RECORD ownership; never execute payloads."""
from __future__ import annotations
import argparse
import ast
import gzip
import hashlib
from importlib import metadata
import json
from pathlib import Path
import re
import sys
import sysconfig

ROOT = Path(__file__).resolve().parents[3]
SNAPSHOT = ROOT / 'build/electron-sidecar/inputs/20261016T100940.347420Z'
WORK = ROOT / 'build/electron-sidecar/work/ressectionlab-engine'
ENGINE = ROOT / 'desktop/sidecar/ressectionlab-engine'
TABLES = ('PYZ-00.toc', 'COLLECT-00.toc', 'EXE-00.toc')


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def canonical(name):
    return re.sub(r'[-_.]+', '-', name).lower()


def rows(value):
    if isinstance(value, (tuple, list)):
        if len(value) == 3 and all(isinstance(item, str) for item in value):
            yield value
        else:
            for item in value:
                yield from rows(item)


def locks(text):
    result = {}
    for line in text.splitlines():
        match = re.fullmatch(r'([A-Za-z0-9_.-]+)==([^\s;]+)', line.strip())
        if match:
            name = canonical(match[1])
            if name in result:
                raise ValueError('Duplicate canonical lock entry')
            result[name] = match[2]
    return result


def ownership(included, distributions, lock):
    """Exact RECORD paths determine distribution ownership, including vendored files."""
    owners, selected = {}, []
    for dist in distributions:
        members = [row for row in dist['record_files'] if row['source'] in included]
        if not members:
            continue
        for row in members:
            owners.setdefault(row['source'], []).append(dist['name'])
        selected.append({key: value for key, value in dist.items() if key != 'record_files'} | {
            'included_members': [{'record_path': row['record_path'], 'source': row['source'],
                'frozen_entries': included[row['source']]} for row in members],
            'locked_version': lock.get(canonical(dist['name']))})
    return {'distributions': selected,
        'missing_pins': [{'name': d['name'], 'installed_version': d['version']} for d in selected if d['locked_version'] is None],
        'version_mismatches': [{'name': d['name'], 'installed_version': d['version'], 'locked_version': d['locked_version']}
            for d in selected if d['locked_version'] is not None and d['locked_version'] != d['version']],
        'unowned': sorted(set(included) - set(owners)),
        'ambiguous_record_owners': {p: names for p, names in owners.items() if len(names) != 1}}


def notice_path(path):
    if '__pycache__' in path.parts or path.suffix.lower() in {'.py', '.pyc', '.pyo'}:
        return False
    return any(p.lower() in {'license', 'licenses', 'licence', 'licences'} for p in path.parts[:-1]) or bool(
        re.match(r'^(licen[cs]e|copying|copyright|notice|authors)([._-]|$)', path.name, re.I))


def audit(lock_path):
    included, absolute_nonfiles = {}, []
    for name in TABLES:
        for entry, source, kind in rows(ast.literal_eval((WORK / name).read_text())):
            if not source or not Path(source).is_absolute():
                continue
            if Path(source).is_file():
                included.setdefault(str(Path(source).resolve()), []).append({'table': name, 'entry': entry, 'kind': kind})
            else:
                absolute_nonfiles.append({'table': name, 'entry': entry, 'source': source, 'kind': kind})
    distributions = []
    for dist in sorted(metadata.distributions(), key=lambda d: canonical(d.metadata.get('Name', ''))):
        record = [{'record_path': str(f), 'source': str(Path(dist.locate_file(f)).resolve())} for f in dist.files or ()]
        if not any(row['source'] in included for row in record):
            continue
        metadata_files, notice_files, missing_notices = [], [], []
        for row in record:
            rel, source = Path(row['record_path']), Path(row['source'])
            is_notice = notice_path(rel)
            if not is_notice and rel.name not in {'METADATA', 'RECORD', 'WHEEL', 'INSTALLER', 'direct_url.json'}:
                continue
            if not source.is_file():
                missing_notices.append(row)
                continue
            bound = row | {'bytes': source.stat().st_size, 'sha256': digest(source)}
            metadata_files.append(bound)
            if is_notice:
                notice_files.append(bound)
        license_text = dist.metadata.get('License', '')
        distributions.append({'name': dist.metadata['Name'], 'version': dist.version, 'record_files': record,
            'declared_license_expression': dist.metadata.get('License-Expression'),
            'declared_license_short': license_text if len(license_text) < 160 else None,
            'long_license_text_retained_in_hashed_METADATA': len(license_text) >= 160,
            'homepage': dist.metadata.get('Home-page'), 'project_urls': dist.metadata.get_all('Project-URL', []),
            'metadata_and_notice_files': metadata_files, 'notice_files': notice_files,
            'missing_metadata_or_notice_files': missing_notices})
    result = ownership(included, distributions, locks(lock_path.read_text()))
    runtime_root = Path(sys.base_prefix).resolve()
    classes = {'stdlib_or_runtime': [], 'captured_project': [], 'generated_build': [], 'unowned_third_party': [], 'other': []}
    for source in result.pop('unowned'):
        p = Path(source)
        group = ('unowned_third_party' if 'site-packages' in p.parts else
            'stdlib_or_runtime' if p.is_relative_to(runtime_root) else
            'captured_project' if p.is_relative_to(SNAPSHOT) else
            'generated_build' if p.is_relative_to(WORK) else 'other')
        classes[group].append(source)
    result.update(schema='failed-sidecar-actual-distribution-closure-v1',
        lock={'path': str(lock_path), 'sha256': digest(lock_path)},
        source_tables={n: digest(WORK / n) for n in TABLES},
        embedded_manifest_sha256=digest(ENGINE / '_internal/build_info/BUILD_INPUT_MANIFEST.json'),
        source_manifest_sha256=digest(SNAPSHOT / 'BUILD_INPUT_MANIFEST.json'),
        engine_sha256=digest(ENGINE / 'ressectionlab-engine'),
        total_included_source_files=len(included), classified_unowned=classes,
        absolute_nonfile_rows=absolute_nonfiles,
        declared_missing_notices=[{'name': d['name'], 'files': d['missing_metadata_or_notice_files']} for d in result['distributions'] if d['missing_metadata_or_notice_files']],
        no_notice_distributions=[d['name'] for d in result['distributions'] if not d['notice_files']],
        numerical_or_model_modules_imported=False, engine_executed=False,
        basis='Actual PYZ/COLLECT/EXE TOCs, installed distribution RECORD file membership and exact METADATA/notice hashes. No import-name guessing, installs, downloads or patient data reads.',
        retained_notice_limit='CPython static-vendor and any distribution notice gaps remain separate from lock completeness; no redistribution readiness claim.')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--lock', type=Path, default=SNAPSHOT / 'requirements-lock.txt')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.lock)
    with args.output.open('xb') as stream:
        stream.write(gzip.compress((json.dumps(result, indent=2) + '\n').encode(), mtime=0))
    print(json.dumps({'output': str(args.output), 'sha256': digest(args.output),
        'distributions': len(result['distributions']), 'included_files': result['total_included_source_files'],
        'missing_pins': result['missing_pins'], 'version_mismatches': result['version_mismatches'],
        'unowned_third_party_count': len(result['classified_unowned']['unowned_third_party']),
        'other_unclassified_count': len(result['classified_unowned']['other']),
        'ambiguous_owners_count': len(result['ambiguous_record_owners']),
        'declared_missing_notices': result['declared_missing_notices'], 'no_notice_distributions': result['no_notice_distributions']}, indent=2))

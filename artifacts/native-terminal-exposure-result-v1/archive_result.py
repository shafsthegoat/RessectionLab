"""Retain compact iteration evidence; leave sparse generated geometry local."""
import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[2]
SOURCE = Path(__file__).resolve().parent
DEST = ROOT / 'artifacts/native-terminal-exposure-result-v1'


def main():
    if DEST.exists():
        raise FileExistsError('Never overwrite an archived result')
    selected = []
    for directory in ('comparison', 'comparison-repair-v1', 'comparison-terminal-v1',
                      'independent', 'independent-validation', 'terminal-validation'):
        selected.extend(p for p in (SOURCE / directory).iterdir() if p.is_file())
    for directory in ('comparison-run-01', 'comparison-run-02', 'terminal-run-01'):
        selected.extend(SOURCE / directory / name for name in
                        ('intent.json', 'launcher.json', 'report.json', 'progress.json', 'worker.log'))
    selected.extend(SOURCE / name for name in
                    ('root-controls-01.log', 'root-controls-02.log', 'terminal-controls-01.log',
                     'monitor-controls-01.log', 'plot_result.py', 'volume-comparison.png', 'archive_result.py'))
    selected.extend(p for p in (SOURCE / 'model').iterdir() if p.is_file())
    # Keep 192 KiB for the index and human-readable result/reproduction notes.
    if sum(p.stat().st_size for p in selected) > 3 * 1024**2 - 192 * 1024:
        raise ValueError('Compact archive reservation would be exceeded')
    files = []
    for src in sorted(selected):
        rel = src.relative_to(SOURCE)
        dst = DEST / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dst)
        digest = hashlib.sha256(src.read_bytes()).hexdigest()
        if hashlib.sha256(dst.read_bytes()).hexdigest() != digest:
            raise ValueError('Copy differs')
        files.append({'archive_path': str(rel), 'source_path': str(src.relative_to(ROOT)),
                      'bytes': src.stat().st_size, 'sha256': digest})
    local_geometry = []
    for directory in ('comparison-run-02', 'terminal-run-01'):
        for p in sorted((SOURCE / directory / 'geometry').glob('*.json')):
            local_geometry.append({'path': str(p.relative_to(ROOT)), 'bytes': p.stat().st_size,
                                   'sha256': hashlib.sha256(p.read_bytes()).hexdigest()})
    index = {'version': 'native-terminal-exposure-compact-archive-v1',
             'canonical_commits': {'all_capsule': 'bc78bbb', 'terminal_only': '7ed4ef8', 'baseline': 'b16ffec'},
             'files': files, 'local_geometry_not_copied': local_geometry,
             'scope': 'Generated numerical evidence only; no patient data, weights, clinical outcomes or physical force observations.',
             'supervision': 'Terminal worker completed; original launcher failed. Independent saved science validation does not erase that failure.'}
    (DEST / 'ARCHIVE_INDEX.json').write_text(json.dumps(index, indent=2) + '\n')
    print(json.dumps({'files': len(files), 'payload_bytes': sum(x['bytes'] for x in files),
                      'geometry_files_retained_local': len(local_geometry)}))


if __name__ == '__main__':
    main()

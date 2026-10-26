"""Install hash-pinned public Raidionics tumor models without altering configs.

The archive hashes pin the retrieved official 1.2.0 release bytes; they are local
fixity pins, not independently published checksums. No inference is performed.
"""
from pathlib import Path
import argparse
import hashlib
import json
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[2]
ASSETS = {
    'glioma-gbm': ('MRI_GBM', 'Raidionics-MRI_GBM-ONNX-v12.zip', 68791937,
                   '3e3441b620ffebcb822d1026445f0a5aa46c268c4034fac37f7d26a3937b9c2c'),
    'glioma-lgg': ('MRI_LGGlioma', 'Raidionics-MRI_LGGlioma-ONNX-v12.zip', 69041134,
                   '10edeca1171b93fcea1c03374e73eb136c20f4ad9577150c2b8f78a258b58f73'),
}
RELEASE = 'https://github.com/raidionics/Raidionics-models/releases/download/1.2.0/'
LICENSE_URL = 'https://raw.githubusercontent.com/raidionics/Raidionics-models/1.2.0/LICENSE'


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def install(model, home=ROOT / 'data/models/raidionics'):
    home = Path(home)
    folder, name, size, expected = ASSETS[model]
    downloads = home / '.downloads'
    downloads.mkdir(parents=True, exist_ok=True)
    archive = downloads / name
    if not archive.exists():
        partial = archive.with_suffix('.partial')
        with urllib.request.urlopen(RELEASE + name, timeout=45) as response, partial.open('wb') as stream:
            count = 0
            while block := response.read(1024 * 1024):
                count += len(block)
                if count > size:
                    raise ValueError('Model download exceeds its pinned size')
                stream.write(block)
        if partial.stat().st_size != size or sha(partial) != expected:
            raise ValueError('Model archive size or SHA256 mismatch; partial retained')
        partial.replace(archive)
    if archive.stat().st_size != size or sha(archive) != expected:
        raise ValueError('Existing model archive does not match its pinned identity')
    expected_files = {folder + '/' + n for n in ('model.onnx', 'pipeline.json', 'pre_processing.ini', 'model_config.txt')}
    with zipfile.ZipFile(archive) as source:
        if {n for n in source.namelist() if not n.endswith('/')} != expected_files:
            raise ValueError('Unexpected model archive members')
        for name_in_zip in sorted(expected_files):
            target = home / name_in_zip
            data = source.read(name_in_zip)
            if target.exists():
                if target.read_bytes() != data:
                    raise ValueError(f'Preserve different existing model file: {target}')
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(data)
    license_path = home / folder / 'LICENSE-BSD-2-Clause.txt'
    if not license_path.exists():
        with urllib.request.urlopen(LICENSE_URL, timeout=30) as response:
            notice = response.read(65537)
        if len(notice) > 65536 or not notice.startswith(b'BSD 2-Clause License'):
            raise ValueError('Unexpected publisher license notice')
        license_path.write_bytes(notice)
    receipt = {'schema': 'resectionlab.reconstruction-assets.v1', 'adapter': model,
               'source_url': RELEASE + archive.name, 'release': '1.2.0',
               'archive_sha256': expected, 'archive_bytes': size,
               'checksum_authority': 'local pin of retrieved official release, not a publisher checksum',
               'license': 'BSD-2-Clause', 'license_url': LICENSE_URL,
               'files': {p.name: sha(p) for p in (home / folder).iterdir() if p.is_file() and p.name != 'installation.json'}}
    (home / folder / 'installation.json').write_text(json.dumps(receipt, indent=2) + '\n')
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', choices=ASSETS, action='append', required=True)
    parser.add_argument('--home', type=Path, default=ROOT / 'data/models/raidionics')
    args = parser.parse_args()
    for model in args.model:
        print(json.dumps(install(model, args.home), indent=2))


if __name__ == '__main__':
    main()

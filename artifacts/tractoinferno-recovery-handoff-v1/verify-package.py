"""Verify packaged text/source bytes only; no restore, queue, network or execution."""
from pathlib import Path
import hashlib,json,re
BASE=Path(__file__).resolve().parent
manifest=json.loads((BASE/'hash-manifest.json').read_text())
expected={r['path']:r for r in manifest['files']}
actual={str(p.relative_to(BASE)) for p in BASE.rglob('*') if p.is_file() and p!=BASE/'hash-manifest.json' and '__pycache__' not in p.parts}
assert actual==set(expected),'package file membership changed'
for name,row in expected.items():
 path=BASE/name
 assert not path.is_symlink() and path.resolve().is_relative_to(BASE),'unsafe package member'
 raw=path.read_bytes()
 assert len(raw)==row['bytes'] and hashlib.sha256(raw).hexdigest()==row['sha256'],name
 assert not re.search(rb'(?i)(?:[?&](?:x-amz-signature|x-amz-credential|x-amz-security-token|access_token|signature|token)=)',raw),'credential-bearing query in '+name
print(json.dumps({'status':'package_bytes_verified','files':len(expected),'bytes':sum(r['bytes'] for r in expected.values()),'external_runtime_dependencies_verified':False,'network_requests':0,'payload_reads':0,'restored_files':0,'recovery_launched':False}))

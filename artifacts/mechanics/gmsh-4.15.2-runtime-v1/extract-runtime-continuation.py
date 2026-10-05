"""Extract the pinned wheel privately, verify RECORD hashes, inspect linkage only."""
import base64,csv,hashlib,io,json,os,re,shutil,stat,subprocess,zipfile
from pathlib import Path,PurePosixPath
OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[2]
def digest(path):
    with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def save(path,value):
    with path.open('x') as f:json.dump(value,f,indent=2);f.write('\n')
def mapped(name):
    p=PurePosixPath(name)
    assert not p.is_absolute() and '..' not in p.parts and '\\' not in name
    if p.parts[0]=='gmsh-4.15.2.data':
        kind=p.parts[1] if len(p.parts)>1 else None
        if kind=='data':return Path(*p.parts[2:])
        if kind=='scripts':return Path('bin',*p.parts[2:])
        if kind in ('purelib','platlib'):return Path(*p.parts[2:])
        if kind is None:return None
        raise ValueError('Undeclared wheel data category: '+name)
    return Path(*p.parts)
def record_matches(value, digest_bytes):
    if not value.startswith('sha256='):return False
    encoded=value[7:]
    if re.fullmatch(r'[0-9a-f]{64}',encoded):return encoded==digest_bytes.hex()
    if re.fullmatch(r'[A-Za-z0-9_-]{43}',encoded):
        return encoded==base64.urlsafe_b64encode(digest_bytes).decode().rstrip('=')
    return False

def main():
    control=hashlib.sha256(b'wheel RECORD format control').digest()
    assert record_matches('sha256='+control.hex(),control)
    assert record_matches('sha256='+base64.urlsafe_b64encode(control).decode().rstrip('='),control)
    assert not record_matches('sha256='+'0'*64,control)
    assert not record_matches('sha256='+'A'*43,control)
    assert not record_matches('sha256='+control.hex()+'junk',control)
    assert not record_matches('md5='+control.hex(),control)
    save(OUT/'record-format-controls.json',{'passed':6,'scope':'Digest encoding controls only, no downloaded code execution','hex_supported':True,'base64_supported':True,'mismatches_and_malformed_rejected':True})
    acquisition=json.loads((OUT/'acquisition-receipt.json').read_text())
    payload=Path(acquisition['payload']);assert digest(payload)=='f6649b3e59f49272e7ee8ab282ecb4d1a6e0d627e86cf3e3b1a83fd07417e4f8'
    prefix=payload.parent/'runtime'
    existing=[p for p in prefix.rglob('*') if p.is_file()]
    assert existing==[prefix/'gmsh.py']
    assert digest(existing[0])=='f4b0935cebde899750293d87a643090e5ea33c9b18d249267f507db68a0e01c3'
    record_name='gmsh-4.15.2.dist-info/RECORD';inventory=[];total=0
    with zipfile.ZipFile(payload) as z:
        entries=z.infolist();assert sum(v.file_size for v in entries)<=256*1024*1024
        assert len(entries)==len({v.filename for v in entries})
        expected={r[0]:(r[1],r[2]) for r in csv.reader(io.StringIO(z.read(record_name).decode()))}
        mapped_files=set()
        for v in entries:
            mode=v.external_attr>>16
            assert not stat.S_ISLNK(mode) and not v.flag_bits&1
            assert v.is_dir() or stat.S_IFMT(mode) in (0,stat.S_IFREG)
            rel=mapped(v.filename)
            if rel is None:assert v.is_dir();continue
            target=prefix/rel
            assert target.resolve().is_relative_to(prefix.resolve())
            if v.is_dir():target.mkdir(parents=True,exist_ok=True);continue
            assert str(rel) not in mapped_files;mapped_files.add(str(rel))
            target.parent.mkdir(parents=True,exist_ok=True)
            h=hashlib.sha256();count=0
            if target.exists():
                assert target==prefix/'gmsh.py'
                with z.open(v) as source:
                    data=source.read();h.update(data);count=len(data);total+=count
                assert target.read_bytes()==data
            else:
                with z.open(v) as source,target.open('xb') as dest:
                    while chunk:=source.read(1024*1024):
                        count+=len(chunk);total+=len(chunk)
                        assert count<=v.file_size and total<=256*1024*1024
                        h.update(chunk);dest.write(chunk)
            assert count==v.file_size
            checksum=h.hexdigest();record_hash,record_bytes=expected[v.filename]
            if record_hash:
                assert record_matches(record_hash,h.digest())
                assert int(record_bytes)==count
            else:assert v.filename==record_name
            if target!=prefix/'gmsh.py':os.chmod(target,0o755 if mode&0o111 else 0o644)
            assert digest(target)==checksum
            inventory.append({'wheel_member':v.filename,'relative_path':str(rel),'bytes':count,'sha256':checksum,'wheel_record_verified':bool(record_hash)})
        assert {v.filename for v in entries if not v.is_dir()}==set(expected)
    save(OUT/'extracted-files.json',{'prefix':str(prefix),'file_count':len(inventory),'total_bytes':total,'files':inventory})
    module=prefix/'gmsh.py';library=prefix/'lib/libgmsh.4.15.dylib';license_path=prefix/'share/doc/gmsh/LICENSE.txt'
    commands=[['file',str(library)],['lipo','-archs',str(library)],['otool','-L',str(library)]]
    inspection=[]
    for argv in commands:
        p=subprocess.run(argv,capture_output=True,text=True,check=True,timeout=10)
        inspection.append({'argv':argv,'stdout':p.stdout,'stderr':p.stderr,'exit_code':p.returncode})
    assert inspection[1]['stdout'].strip()=='arm64'
    save(OUT/'linkage.json',{'commands':inspection})
    with (OUT/'LICENSE-upstream.txt').open('xb') as f:f.write(license_path.read_bytes())
    receipt={'schema_version':1,'status':'extracted_verified_pending_single_probe','acquisition_receipt_sha256':digest(OUT/'acquisition-receipt.json'),'extraction_driver_sha256':digest(Path(__file__)),'private_prefix':str(payload.parent),'runtime_prefix':str(prefix),'module':{'path':str(module),'sha256':digest(module),'bytes':module.stat().st_size},'library':{'path':str(library),'sha256':digest(library),'bytes':library.stat().st_size,'architecture':'arm64'},'license':{'path':str(license_path),'sha256':digest(license_path),'copied_notice':'LICENSE-upstream.txt'},'extracted_files_sha256':digest(OUT/'extracted-files.json'),'extracted_file_count':len(inventory),'extracted_bytes':total,'linkage_sha256':digest(OUT/'linkage.json'),'wheel_record_all_payload_files_verified':True,'record_encoding':'Creator uses SHA256 hex, accepted after six explicit controls; original failed base64-only verifier retained','existing_wrapper_reused_unchanged':True,'no_code_executed':True,'no_global_or_app_install':True}
    save(OUT/'extraction-receipt.json',receipt);print(json.dumps(receipt))
if __name__=='__main__':main()

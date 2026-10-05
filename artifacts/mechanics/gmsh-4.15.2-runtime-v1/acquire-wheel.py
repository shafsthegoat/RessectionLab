"""One exact public wheel transfer; inspect ZIP metadata without executing it."""
import datetime,hashlib,json,os,stat,subprocess,time,zipfile
from pathlib import Path,PurePosixPath
OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[2]
def digest(path,algorithm='sha256'):
    with path.open('rb') as f:return hashlib.file_digest(f,algorithm).hexdigest()
def save(path,value):
    with path.open('x') as f:json.dump(value,f,indent=2);f.write('\n')
def main():
    release_path=OUT/'root-release.json'
    assert digest(release_path)=='bd6135672fbe6982b7f19e814375ea5c791d98f459c930e7192e44e55c6fb95a'
    release=json.loads(release_path.read_text());spec=release['mesher']
    assert digest(OUT/'frozen-manifest.json')==release['manifest_sha256']
    prefix=ROOT/spec['private_prefix'];prefix.mkdir(parents=True,exist_ok=False)
    payload=prefix/spec['filename'];partial=payload.with_suffix(payload.suffix+'.partial')
    command=['curl','--silent','--show-error','--fail','--proto','=https','--max-time','60','--max-filesize',str(spec['bytes']),'--dump-header',str(OUT/'transfer-headers.txt'),'--output',str(partial),'--write-out','%{http_code}\n%{url_effective}\n%{size_download}\n',spec['url']]
    attempt={'schema_version':1,'started_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'release_sha256':digest(release_path),'driver_sha256':digest(Path(__file__)),'command':command,'status':'running','mesh_generation':False,'patient_or_curve_access':False}
    save(OUT/'acquisition-attempt.json',attempt)
    t=time.monotonic()
    try:
        proc=subprocess.run(command,text=True,capture_output=True,timeout=62,check=False)
        with (OUT/'transfer.log').open('x') as f:f.write(proc.stdout);f.write(proc.stderr)
        assert proc.returncode==0,('curl',proc.returncode,proc.stderr)
        lines=proc.stdout.splitlines();assert lines==['200',spec['url'],str(spec['bytes'])],lines
        assert partial.stat().st_size==spec['bytes']
        assert digest(partial)==spec['sha256']
        os.link(partial,payload);partial.unlink()
        with zipfile.ZipFile(payload) as z:
            infos=z.infolist();names=[v.filename for v in infos]
            assert len(names)==len(set(names)),'Duplicate ZIP member'
            total=sum(v.file_size for v in infos);assert total<=release['scope']['safe_extraction_uncompressed_cap_bytes']
            for v in infos:
                p=PurePosixPath(v.filename);mode=v.external_attr>>16
                assert not p.is_absolute() and '..' not in p.parts and '\\' not in v.filename
                assert not stat.S_ISLNK(mode) and not v.flag_bits&1
                assert v.is_dir() or stat.S_IFMT(mode) in (0,stat.S_IFREG),'Nonregular ZIP member'
            inventory=[{'path':v.filename,'bytes':v.file_size,'compressed_bytes':v.compress_size,'crc32':f'{v.CRC:08x}','unix_mode':oct(v.external_attr>>16),'directory':v.is_dir()} for v in infos]
        save(OUT/'wheel-inventory.json',{'member_count':len(infos),'uncompressed_bytes':total,'duplicate_paths':False,'unsafe_paths':False,'symlinks':False,'encrypted_members':False,'members':inventory})
        outcome={**attempt,'status':'verified_not_extracted','finished_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'elapsed_seconds':time.monotonic()-t,'payload':str(payload),'bytes':payload.stat().st_size,'sha256':digest(payload),'md5':digest(payload,'md5'),'wheel_inventory_sha256':digest(OUT/'wheel-inventory.json'),'zip_uncompressed_bytes':total,'zip_members':len(infos),'transfer_log_sha256':digest(OUT/'transfer.log')}
    except BaseException as error:
        outcome={**attempt,'status':'failed_no_retry','error':repr(error),'elapsed_seconds':time.monotonic()-t,'partial_retained':str(partial) if partial.exists() else None}
        save(OUT/'acquisition-receipt.json',outcome);raise
    save(OUT/'acquisition-receipt.json',outcome)
    print(json.dumps({k:outcome[k] for k in ('status','bytes','sha256','elapsed_seconds','zip_members','zip_uncompressed_bytes')}),flush=True)
if __name__=='__main__':main()

#!/usr/bin/env python3
"""Read-only, resumable data foundation inventory. Never follows directory links.

Checks mounted external identity before writes. Protected source byte access is
excluded conservatively; SHA256 is storage integrity, not image interpretation.
No copying, mutation of source data, model inference, or split reassignment.
"""
from __future__ import annotations
import argparse, collections, hashlib, json, os, pathlib, plistlib, re, sqlite3, stat, subprocess, sys, time, zipfile
ROOT=pathlib.Path(__file__).resolve().parents[1]
DATA=pathlib.Path('/Volumes/shafrir/old coding projects/RessectionLabData')
UUID='9CC3623C-E38B-3EEE-B6BB-B9A06C193700'

def mount():
    info=plistlib.loads(subprocess.check_output(['diskutil','info','-plist','/Volumes/shafrir']))
    if info.get('VolumeUUID')!=UUID or info.get('MountPoint')!='/Volumes/shafrir' or not os.path.ismount('/Volumes/shafrir') or info.get('Internal') is True:
        raise RuntimeError('Required external volume absent or identity changed')
    if info.get('ReadOnlyVolume'):raise RuntimeError('External volume read only')
    return {k:info.get(k) for k in ['VolumeUUID','MountPoint','DeviceIdentifier','FilesystemName','Internal','TotalSize','FreeSpace','Writable']}

def dump(path,value):
    mount();tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(value,indent=2,sort_keys=True)+'\n');tmp.replace(path)

def roles():
    result={};evidence=[]
    for name in ['lausanne-component-cohort-v1.json','synthrad-component-cohort-v1.json','resect-component-cohort-v1.json','experiments/remind-component-cohort-v1.json']:
        p=ROOT/'manifests'/name;b=p.read_bytes();doc=json.loads(b)
        evidence.append({'path':str(p),'sha256':hashlib.sha256(b).hexdigest(),'declared_counts':dict(collections.Counter(r['role'] for r in doc['members']))})
        for row in doc['members']:
            key=row.get('canonical_person',row.get('patient_group'));result[key]=row['role']
    p=ROOT/'manifests/experiments/btc-spatial-development-cohort-v1.json';b=p.read_bytes();d=json.loads(b)
    evidence.append({'path':str(p),'sha256':hashlib.sha256(b).hexdigest()})
    for r in d['candidates']:
        result[r['patient_group']]='TRAIN' if r['development_role']=='population_training' else 'SELECT' if 'selection' in r['development_role'] else 'UNOPENED_TRANSFER'
    p=ROOT/'testcases/downloaded-datasets/UCSF-frozen-assessment-20261027/protocol.json'
    if p.is_file():
        b=p.read_bytes();d=json.loads(b);result[d['patientId']]=d['role'];evidence.append({'path':str(p),'sha256':hashlib.sha256(b).hexdigest()})
    return result,evidence

def protection(path,assigned):
    s=str(path);low=s.lower()
    if any(x in low for x in ['upenn','utsw','frozen-assessment','external-final','measurement_eval','holdout','held-out']):return 'reserved_or_frozen_path'
    matches=re.findall(r'UCSF-PDGM-\d+',s)
    if 'lausanne' in low or 'ds003949' in low:matches += ['Lausanne:'+x for x in re.findall(r'sub-\d+',s)]
    matches += ['SynthRAD2023:'+x for x in re.findall(r'1B[A-C]\d+',s)]
    matches += ['ReMIND:'+f'{int(x):03d}' for x in re.findall(r'(?:ReMIND[-:]|remind-)(\d+)',s)]
    if 'resect' in low:matches += ['RESECT:Case'+str(int(x)) for x in re.findall(r'Case(\d+)',s)]
    matches += ['BTC:sub-PAT'+x for x in re.findall(r'(?:sub-PAT|BTC-PAT)(\d+)',s)]
    blocked=[x for x in matches if assigned.get(x) not in {None,'TRAIN','DEVELOPMENT'}]
    return ','.join(x+':'+assigned[x] for x in blocked) or None

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--snapshot',help='Safe name for a separate inventory snapshot under catalog/inventory');ap.add_argument('--skip-hash',action='store_true');ap.add_argument('--read-native-samples',action='store_true');args=ap.parse_args()
    identity=mount();out=DATA/'catalog/inventory'
    if args.snapshot:
        if not re.fullmatch(r'[a-z0-9][a-z0-9_-]*',args.snapshot):raise ValueError('Snapshot must be one safe path component')
        out=out/args.snapshot
    out.mkdir(parents=True,exist_ok=True)
    probe=out/'.write-probe';probe.write_text('verified');probe.unlink()
    assigned,evidence=roles();dump(out/'split-reservations.json',{'roles':assigned,'evidence':evidence,'UPENN-GBM':'external final reserved; no source bytes read','UTSW-Glioma':'future structural reservation; no source bytes read','policy':'No role reassignment; metadata exposure is not erased; unknown pretrained exposure remains unknown.'})
    rows=[];seen=set();totals={};errors=[]
    for base in [ROOT/'testcases',ROOT/'data']:
        for directory,dirs,files in os.walk(base,followlinks=False):
            for name in list(dirs)+files:
                p=pathlib.Path(directory)/name
                try:s=p.lstat()
                except OSError as e:errors.append({'path':str(p),'error':str(e)});continue
                if stat.S_ISDIR(s.st_mode):continue
                r={'path':str(p),'root':base.name,'top':p.relative_to(base).parts[0],'bytes':s.st_size,'allocated_bytes':s.st_blocks*512,'device':s.st_dev,'inode':s.st_ino,'mtime_ns':s.st_mtime_ns,'protected':protection(p,assigned)}
                r['kind']='symlink' if p.is_symlink() else 'file' if stat.S_ISREG(s.st_mode) else 'other'
                if r['kind']=='symlink':
                    r['target']=os.readlink(p);r['resolved']=str(p.resolve());r['broken']=not p.exists()
                    try:r['target_bytes']=p.stat().st_size
                    except OSError:r['target_bytes']=None
                rows.append(r)
                t=totals.setdefault(base.name+'/'+r['top'],collections.Counter());t['entries']+=1;t[r['kind']]+=1;t['logical_regular_bytes']+=r['bytes'] if r['kind']=='file' else 0;t['allocated_lstat_bytes']+=r['allocated_bytes'];t['dereferenced_logical_bytes']+=r.get('target_bytes') or (r['bytes'] if r['kind']=='file' else 0);t['broken_links']+=bool(r.get('broken'))
                key=(s.st_dev,s.st_ino)
                if r['kind']=='file' and key not in seen:t['unique_inode_regular_bytes']+=s.st_size;seen.add(key)
    dump(out/'files.json',rows);dump(out/'baseline.json',{'generated_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'volume':identity,'scope':['testcases','data'],'by_topdir':totals,'entries':len(rows),'native_project_paths':sum(r['path'].endswith('.ressectionlab') for r in rows),'errors':errors,'caveat':'Allocated bytes count lstat blocks; APFS clone sharing/compression cannot be inferred from st_blocks. Directory symlinks are recorded, never traversed. Protected bytes remain unhashed.'})
    print(json.dumps({'inventory_entries':len(rows),'native_paths':sum(r['path'].endswith('.ressectionlab') for r in rows),'output':str(out)}),flush=True)
    db=sqlite3.connect(out/'hash-cache.sqlite3');db.execute('CREATE TABLE IF NOT EXISTS hashes (path TEXT PRIMARY KEY, device INTEGER, inode INTEGER, bytes INTEGER, mtime INTEGER, sha256 TEXT)');hashed=[];skipped=[];last_mount=0
    if not args.skip_hash:
        # One serial reader, 4 MiB chunks, persistent checkpoints. Recheck identity
        # at every source; never hash a symlink target implicitly.
        for i,r in enumerate(rows):
            if r['kind']!='file':continue
            if r['protected']:skipped.append({'path':r['path'],'reason':r['protected']});continue
            p=pathlib.Path(r['path']);old=db.execute('SELECT device,inode,bytes,mtime,sha256 FROM hashes WHERE path=?',(str(p),)).fetchone();key=(r['device'],r['inode'],r['bytes'],r['mtime_ns'])
            try:
                if time.monotonic()-last_mount>15:mount();last_mount=time.monotonic()
                if old and tuple(old[:4])==key:digest=old[4]
                else:
                    h=hashlib.sha256()
                    with p.open('rb') as f:
                        while chunk:=f.read(4*1024*1024):h.update(chunk)
                    s=p.stat()
                    if (s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns)!=key:raise RuntimeError('Source changed while hashing')
                    digest=h.hexdigest();db.execute('INSERT OR REPLACE INTO hashes VALUES (?,?,?,?,?,?)',(str(p),*key,digest))
                hashed.append({'path':str(p),'bytes':r['bytes'],'sha256':digest})
            except Exception as e:errors.append({'path':str(p),'error':str(e)})
            if i%100==0:db.commit()
            if i%5000==0:print(json.dumps({'position':i,'hashed':len(hashed)}),flush=True)
        db.commit();groups=collections.defaultdict(list)
        for r in hashed:groups[(r['sha256'],r['bytes'])].append(r['path'])
        dump(out/'sha256.json',hashed);dump(out/'duplicates.json',[{'sha256':k[0],'bytes_each':k[1],'paths':v} for k,v in groups.items() if len(v)>1]);dump(out/'hash-exclusions.json',skipped)
    projects=[]
    for r in rows:
        if not r['path'].endswith('.ressectionlab'):continue
        p=pathlib.Path(r['path']);item={'path':str(p),'protected':r['protected'] or protection(p.resolve(),assigned)}
        if item['protected']:item['status']='protected_not_opened';projects.append(item);continue
        try:
            with zipfile.ZipFile(p) as z:
                members=z.infolist();item['members']=[{'name':x.filename,'bytes':x.file_size,'compressed_bytes':x.compress_size} for x in members]
                if z.getinfo('manifest.json').file_size>8*1024*1024:raise ValueError('Manifest exceeds inventory budget')
                m=json.loads(z.read('manifest.json'));item.update({'status':'manifest_read_only','case_id':m.get('case_id'),'format_version':m.get('format_version'),'metadata':m.get('metadata',{}),'array_sha256_recorded':m.get('array_sha256'),'self_contained_primary_arrays':'arrays.npz' in z.namelist(),'workspace_embedded':'workspace.json' in z.namelist()})
                if 'workspace.json' in z.namelist():item['workspace']=json.loads(z.read('workspace.json'))
        except Exception as e:item['status']='unreadable';item['error']=str(e)
        projects.append(item)
    dump(out/'native-projects.json',projects)
    if args.read_native_samples:
        sys.path.insert(0,str(ROOT/'src'));from resectionlab.workspace_bundle import load_workspace
        samples=[]
        # Existing development, controls and known missing-target regression.
        for name in ['meningioma-sub-06-standard-v2.ressectionlab','meningioma-sub-13-standard-v2.ressectionlab','UCSF-PDGM-0004-standard-v2.ressectionlab']:
            p=ROOT/'testcases/cases'/name
            try:
                case,artifacts,workspace=load_workspace(p);samples.append({'path':str(p),'status':'actual_app_reader_pass','case_id':case.case_id,'semantic_hash':case.semantic_hash,'workspace_extension':workspace is not None})
            except Exception as e:samples.append({'path':str(p),'status':'actual_app_reader_failure','error':str(e)})
        dump(out/'actual-reader-samples.json',samples)
    refs=subprocess.run(['rg','-n','--glob','!artifacts/**','--glob','!build/**','--glob','!testcases/**','testcases/|outputs/poc-cases|outputs/standard-environments',str(ROOT/'scripts'),str(ROOT/'src'),str(ROOT/'tests'),str(ROOT/'docs'),str(ROOT/'README.md')],capture_output=True,text=True)
    mount();(out/'path-references.txt').write_text(refs.stdout);dump(out/'errors.json',errors)
    print(json.dumps({'completed':True,'hashed':len(hashed),'protected_skipped':len(skipped),'projects':len(projects),'errors':len(errors)}),flush=True)
if __name__=='__main__':main()

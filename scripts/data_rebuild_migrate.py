#!/usr/bin/env python3
"""Reversible, hash-bound closed-evidence archival and one scoped retirement.

Retirement is limited to the explicitly authorized closed 2026-10-19 campaign.
Plans scan remaining repository/external native ZIP metadata, JSON/text indexes,
and every observed symlink without reading imaging arrays. Archives preserve
symlinks within PAX tar (never extracted onto ExFAT). Restore only into an empty
internal staging destination; verify all file bytes and link targets afterward.
"""
from __future__ import annotations
import argparse,hashlib,json,os,pathlib,shutil,stat,subprocess,sys,tarfile,time,urllib.parse,zipfile
ROOT=pathlib.Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from data_rebuild_inventory import mount,protection,roles
DATA=pathlib.Path('/Volumes/shafrir/old coding projects/RessectionLabData')
CAT=DATA/'catalog/inventory/migration';ARCHIVE=DATA/'archive/closed-evidence'
TEXT={'.json','.md','.txt','.py','.js','.cjs','.mjs','.ts','.tsx','.sh','.toml','.yml','.yaml','.csv','.html'}
SKIP={'.git','.venv','node_modules','.tools','__pycache__','.pytest_cache','.mypy_cache'}

def write(p,x):
 mount();p.parent.mkdir(parents=True,exist_ok=True);q=p.with_suffix(p.suffix+'.tmp');q.write_text(json.dumps(x,indent=2,sort_keys=True)+'\n');q.replace(p)
def digest(p):
 h=hashlib.sha256()
 with pathlib.Path(p).open('rb') as f:
  while b:=f.read(4*1024*1024):h.update(b)
 return h.hexdigest()
def checked_tree(p):
 p=pathlib.Path(p).absolute()
 if p.is_symlink() or not p.is_dir() or not p.resolve().is_relative_to(ROOT/'testcases') or p.resolve()==ROOT/'testcases':raise ValueError('Select one real, bounded directory under testcases; never collection root or symlink')
 return p.resolve()
def entries(tree):
 assigned,_=roles();result=[]
 for d,ds,fs in os.walk(tree,followlinks=False):
  for n in ds+fs:
   p=pathlib.Path(d)/n;s=p.lstat();rel=str(p.relative_to(tree));kind='symlink' if p.is_symlink() else 'directory' if p.is_dir() else 'file' if p.is_file() else 'unsupported'
   if kind=='unsupported':raise ValueError('Unsupported live special file '+str(p))
   row={'relative_path':rel,'kind':kind,'bytes':s.st_size,'allocated_bytes':s.st_blocks*512,'mode':stat.S_IMODE(s.st_mode),'uid':s.st_uid,'gid':s.st_gid,'mtime_ns':s.st_mtime_ns}
   if kind=='symlink':row['target']=os.readlink(p)
   elif kind=='file':
    if protection(p,assigned):raise ValueError('Protected bytes cannot be archived by this workflow: '+str(p))
    row['sha256']=digest(p)
   result.append(row)
 return sorted(result,key=lambda x:x['relative_path'])
def walk(base):
 for d,ds,fs in os.walk(base,followlinks=False):
  ds[:]=[n for n in ds if n not in SKIP and not n.startswith('._')]
  for n in ds+fs:
   if not n.startswith('._'):yield pathlib.Path(d)/n

def references(tree):
 refs=[];errors=[];counts={'native_metadata':0,'text_metadata':0,'symlinks':0};needle=str(tree);relative=str(tree.relative_to(ROOT));short=tree.name+'/'
 def search(p,text,kind,member=None):
  value=urllib.parse.unquote(text)
  strong=needle in value or relative in value
  # Relative references from the file's own directory, plus a bounded weak
  # name/path check for code-constructed paths or historical short Markdown.
  if strong or short in value:
   positions=[]
   for token in [needle,relative,short]:
    i=value.find(token)
    if i>=0:positions.append(i)
   i=min(positions);refs.append({'path':str(p),'kind':kind,'member':member,'strength':'explicit_path' if strong else 'relative_or_constructed_path_requires_review','context':value[max(0,i-100):i+220],'inside_inventory_evidence':str(p).startswith(str(DATA/'catalog/inventory'))})
 def stream(p,handle,kind,member=None):
  tail=''
  while block:=handle.read(1024*1024):
   value=tail+block.decode('utf8',errors='replace');before=len(refs);search(p,value,kind,member)
   if len(refs)>before:return
   tail=value[-8192:]
 configs=[pathlib.Path.home()/'Library/Application Support'/name for name in ['ressectionlab-desktop','RessectionLab','ressectionlab']]
 for base in [ROOT,DATA]+[p for p in configs if p.is_dir()]:
  for p in walk(base):
   if p==tree or p.is_relative_to(tree):continue
   try:
    if p.is_symlink():
     counts['symlinks']+=1;target=p.resolve()
     if target==tree or target.is_relative_to(tree):refs.append({'path':str(p),'kind':'symlink_dependency','target':os.readlink(p),'resolved_target':str(target),'strength':'live_dependency','inside_inventory_evidence':False})
     continue
    if not p.is_file():continue
    if p.suffix=='.ressectionlab':
     with zipfile.ZipFile(p) as z:
      for name in ['manifest.json','workspace.json']:
       if name in z.namelist():
        with z.open(name) as member:stream(p,member,'native_metadata',name)
     counts['native_metadata']+=1
    elif p.suffix in TEXT:
     with p.open('rb') as handle:stream(p,handle,'text_metadata')
     counts['text_metadata']+=1
   except Exception as e:errors.append({'path':str(p),'reason':str(e)})
 return {'scanned':counts,'references':refs,'inconclusive':errors,'source_arrays_decoded':False,'protected_zip_members_read':'manifest/workspace JSON only; never arrays or original image members','scope':'Repository excluding dependency/VCS caches, canonical data root, and existing ressectionlab-desktop/RessectionLab application-support settings; no directory-link traversal; metadata streamed in bounded1MiB chunks'}

def plan(tree):
 tree=checked_tree(tree);mount();rows=entries(tree);scan=references(tree)
 result={'schema':'ressectionlab-reversible-archive-plan-v1','original_path':str(tree),'created_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'files':rows,'logical_regular_bytes':sum(r['bytes'] for r in rows if r['kind']=='file'),'allocated_lstat_bytes':sum(r['allocated_bytes'] for r in rows),'reference_scan':scan,'decision':'archive_copy_only_pending_parent_review_and_native_replacement_proof','local_removal_authorized':False,'scope_ownership':'bounded local test-case evidence tree; source bytes protected by hash manifest','restoration':'Restore tar to original parent only when original tree absent; verify file SHA256 and exact symlink targets','archive_path':str(ARCHIVE/(tree.name+'.tar'))}
 p=CAT/(tree.name+'-plan.json');write(p,result);return p,result

def verify_archive(archive,plan):
 expected={plan['original_path'].split('/')[-1]+'/'+r['relative_path']:r for r in plan['files']};verified=0
 with tarfile.open(archive,'r:') as z:
  members={m.name.rstrip('/'):m for m in z.getmembers()}
  for key,row in expected.items():
   m=members.get(key)
   if m is None:raise ValueError('Missing archive member '+key)
   if row['kind']=='symlink':
    if not m.issym() or m.linkname!=row['target']:raise ValueError('Symlink archive mismatch '+key)
   elif row['kind']=='directory':
    if not m.isdir():raise ValueError('Directory archive mismatch '+key)
   else:
    if not (m.isfile() or m.islnk()):raise ValueError('Regular archive mismatch '+key)
    h=hashlib.sha256();f=z.extractfile(m);count=0
    while b:=f.read(4*1024*1024):h.update(b);count+=len(b)
    if count!=row['bytes'] or h.hexdigest()!=row['sha256']:raise ValueError('Archive byte/hash mismatch '+key)
   verified+=1
 return {'members_verified':verified,'archive_sha256':digest(archive),'archive_bytes':pathlib.Path(archive).stat().st_size}

def archive(plan_path):
 mount();p=pathlib.Path(plan_path);d=json.loads(p.read_text());tree=checked_tree(d['original_path']);destination=pathlib.Path(d['archive_path'])
 if entries(tree)!=d['files']:raise ValueError('Source snapshot changed; regenerate and review plan')
 usage=shutil.disk_usage(DATA)
 if usage.free<max(50*1024**3,usage.total//10)+d['logical_regular_bytes']+100*1024**2:raise RuntimeError('Archive expansion exceeds external reserve')
 mount();destination.parent.mkdir(parents=True,exist_ok=True)
 if not destination.exists():
  partial=destination.with_suffix('.tar.partial')
  if partial.exists():raise ValueError('Interrupted archive exists; preserve or inspect it before retrying')
  mount();subprocess.run(['/usr/bin/tar','--format','pax','-cf',str(partial),'-C',str(tree.parent),tree.name],check=True)
  verified=verify_archive(partial,d)
  if entries(tree)!=d['files']:raise ValueError('Source changed during archival; partial preserved, no promotion')
  mount();partial.replace(destination)
 else:verified=verify_archive(destination,d)
 receipt={'schema':'ressectionlab-verified-archive-receipt-v1','original_path':str(tree),'archive_path':str(destination),'plan_path':str(p),'plan_sha256':digest(p),**verified,'original_source_preserved':True,'local_bytes_reclaimed':0,'restoration_command':f'.venv/bin/python scripts/data_rebuild_migrate.py restore --plan {str(p)!r} --destination {str(tree.parent)!r}'}
 write(CAT/(tree.name+'-archive-receipt.json'),receipt);return receipt

def restore(plan_path,destination):
 mount();d=json.loads(pathlib.Path(plan_path).read_text());archive=pathlib.Path(d['archive_path'])
 receipt=json.loads((CAT/(pathlib.Path(d['original_path']).name+'-archive-receipt.json')).read_text())
 if receipt['plan_sha256']!=digest(plan_path) or receipt['archive_path']!=str(archive) or receipt['archive_sha256']!=digest(archive):raise ValueError('Restoration archive/plan SHA256 identity differs from verified receipt')
 verify_archive(archive,d);destination=pathlib.Path(destination).absolute();target=destination/pathlib.Path(d['original_path']).name
 if str(destination).startswith('/Volumes/'):raise ValueError('Restore symbolic-link trees onto the internal filesystem, never ExFAT')
 if target.exists() or target.is_symlink():raise ValueError('Restoration refuses to overwrite any existing tree')
 destination.mkdir(parents=True,exist_ok=True)
 with tarfile.open(archive,'r:') as z:
  for m in z:
   if m.name.startswith('/') or '..' in pathlib.PurePosixPath(m.name).parts:raise ValueError('Unsafe archive member')
 subprocess.run(['/usr/bin/tar','-xpf',str(archive),'-C',str(destination)],check=True)
 restored=entries(target)
 # macOS PAX restores extra metadata helper files; compare every planned
 # original member's bytes, symlink target, mode, and time, retaining extras.
 actual={r['relative_path']:r for r in restored};mismatches=[]
 for expected in d['files']:
  found=actual.get(expected['relative_path'])
  if found is None or any(found.get(k)!=expected.get(k) for k in ['kind','sha256','target','mode']):mismatches.append(expected['relative_path'])
 if mismatches:raise ValueError('Restored contents differ: '+str(mismatches[:10]))
 return {'restored_to':str(target),'planned_members_verified':len(d['files']),'overwrite':False,'symlinks_preserved':sum(r['kind']=='symlink' for r in d['files'])}

def fixture():
 import numpy as np
 from resectionlab.core import CaseData,SourceRef
 from resectionlab.workspace_bundle import save_workspace,load_workspace
 mount();base=ROOT/'testcases/data-rebuild-migration-fixture';source=base/'disposable-generated-control'
 if (source/'fixture.ressectionlab').exists():raise ValueError('Fixture native package already exists; inspect earlier evidence instead of overwriting')
 if source.exists():
  # Only this explicitly generated disposable fixture can resume creation.
  if any(p.name not in {'subdir','linked-payload.txt'} for p in source.iterdir()):raise ValueError('Unexpected fixture content; preserve it')
  shutil.rmtree(source)
 source.mkdir(parents=True);(source/'subdir').mkdir();(source/'subdir/payload.txt').write_text('generated reversible migration fixture; no patient data\n');(source/'linked-payload.txt').symlink_to('subdir/payload.txt')
 image=np.arange(64,dtype=np.float32).reshape(4,4,4);case=CaseData('GENERATED-MIGRATION-RESTORE-CONTROL',image,{'generated_target':image>50},np.eye(4),(SourceRef('generated-control','generated://migration-fixture',provenance='simulated',license='CC0-1.0'),),metadata={'is_synthetic':True,'real_patient_count':0,'purpose':'disposable archive/restore software fixture'})
 save_workspace(case,source/'fixture.ressectionlab',sources=[],imaging=None,episode=None,selection=None,artifacts={})
 pp,pl=plan(source);receipt=archive(pp);restored=restore(pp,base/'restored');loaded,_,_=load_workspace(pathlib.Path(restored['restored_to'])/'fixture.ressectionlab')
 if loaded.semantic_hash!=case.semantic_hash:raise ValueError('Restored native workspace identity differs')
 result={'fixture_only':True,'source':str(source),'archive_receipt':receipt,'restoration':restored,'production_native_reader':'pass','semantic_hash':loaded.semantic_hash,'real_local_sources_removed':False};write(CAT/'disposable-restoration-roundtrip.json',result);return result

def retire(plan_path,review_path):
 """Exact one-campaign cleanup authorized by lead after replacement app proof."""
 mount();pp=pathlib.Path(plan_path);d=json.loads(pp.read_text());tree=checked_tree(d['original_path'])
 allowed=ROOT/'testcases/performance/rehearsal-2026-10-19'
 if tree!=allowed:raise ValueError('Lead authorization covers only exact closed 2026-10-19 rehearsal campaign')
 review=json.loads(pathlib.Path(review_path).read_text());receipt_path=CAT/(tree.name+'-archive-receipt.json');receipt=json.loads(receipt_path.read_text())
 if digest(pp)!=review['plan_sha256'] or digest(pp)!=receipt['plan_sha256'] or review['unresolved_count'] or review['inconclusive_count']:raise ValueError('Reference review/archive no longer bind a resolved source plan')
 archive_path=pathlib.Path(d['archive_path'])
 if not archive_path.resolve().is_relative_to(ARCHIVE.resolve()):raise ValueError('Archive escapes canonical archive root')
 mount()
 if digest(archive_path)!=receipt['archive_sha256']:raise ValueError('Verified archive SHA256 changed')
 source=entries(tree)
 if source!=d['files']:raise ValueError('Local source snapshot changed after verified archive')
 current=references(tree)
 if current['inconclusive']:raise ValueError('Current metadata/reference scan incomplete')
 approved={(r['path'],r['kind'],r.get('member'),r['strength'],r.get('context')) for r in review['references'] if not r['inside_inventory_evidence']}
 for r in current['references']:
  if r['inside_inventory_evidence']:continue
  if r['kind'] in {'native_metadata','symlink_dependency'}:raise ValueError('An active native/symlink reference remains: '+r['path'])
  if (r['path'],r['kind'],r.get('member'),r['strength'],r.get('context')) not in approved:raise ValueError('New or changed external reference requires review: '+r['path'])
 # Refuse cleanup while any process holds an open descendant, and refuse
 # benchmark writers even between their file writes. No processes are killed.
 live=subprocess.run(['/usr/sbin/lsof','-nP','+D',str(tree)],capture_output=True,text=True)
 if live.returncode not in {0,1} or live.stdout.strip() or live.stderr.strip():raise ValueError('Live/open-file state inconclusive or in use: '+(live.stdout+live.stderr)[:1000])
 processes=subprocess.check_output(['/bin/ps','-axo','pid=,command='],text=True)
 forbidden=['benchmark_rehearsal.py','benchmark-rehearsal.cjs','benchmark-rehearsal-transport.cjs','verify-label-texture.cjs']
 active=[line for line in processes.splitlines() if any(name in line for name in forbidden) and int(line.strip().split(None,1)[0]) not in {os.getpid(),os.getppid()}]
 if active:raise ValueError('Benchmark writer process still running: '+str(active[:3]))
 mount()
 if tree.is_symlink() or tree.resolve()!=allowed:raise ValueError('Source root identity changed')
 before=shutil.disk_usage(ROOT).free
 # Only the fully hashed, archived, reference-reviewed exact campaign is
 # removed. Patient sources, active cases, history, and parent stay intact.
 shutil.rmtree(tree)
 after=shutil.disk_usage(ROOT).free
 guide=ROOT/'testcases/performance/ARCHIVED-EVIDENCE.md'
 text=('# Archived closed performance evidence\n\n'
       'The completed 2026-10-19 rehearsal benchmark campaign was archived after byte verification and dependency review. Active patient sources and native cases remain in their existing locations.\n\n'
       f'Original: `{tree}`\n\nArchive: `{archive_path}`\n\nSHA256: `{receipt["archive_sha256"]}`\n\n'
       f'Plan/hash manifest: `{pp}`\n\nRestore while the original campaign path is absent:\n\n```sh\n{receipt["restoration_command"]}\n```\n\n'
       'Restoration refuses to overwrite an existing directory. If a benchmark rerun recreated that output directory, restore to a different ignored internal parent directory and inspect the archived version there. PAX tar preserves relative symbolic links and macOS metadata; do not unpack the tar directly onto ExFAT. Benchmark output defaults may create a fresh campaign directory as before.\n')
 if guide.exists():guide.write_text(guide.read_text()+'\n'+text)
 else:guide.write_text(text)
 result={'schema':'ressectionlab-completed-reversible-migration-v1','original_path':str(tree),'archive_path':str(archive_path),'archive_sha256':receipt['archive_sha256'],'plan_path':str(pp),'plan_sha256':digest(pp),'lead_authorization':'Explicit parent goahead after hidden native replacement save/reopen proof, 2026-10-31; exact one closed campaign only','source_logical_bytes_before':sum(r['bytes'] for r in source if r['kind']=='file'),'source_allocated_lstat_bytes_before':sum(r['allocated_bytes'] for r in source),'source_path_present_after':tree.exists(),'source_logical_bytes_after':0,'source_allocated_bytes_after':0,'local_disk_free_before':before,'local_disk_free_after':after,'observed_free_change':after-before,'disk_free_caveat':'Immediate system-wide free-space measurement includes concurrent work, delayed APFS reclamation, clone sharing and snapshots; logical/allocated source removal is not a claim of exclusive physical bytes reclaimed.','source_entries_archived':len(source),'live_files_open_before_removal':False,'active_native_or_symlink_dependencies':0,'reference_scan':current['scanned'],'local_archive_guide':str(guide),'restoration_command':receipt['restoration_command'],'removed_scope':'exact closed campaign only; active sources/cases/history preserved','archive_preserved':True}
 write(CAT/(tree.name+'-migration-receipt.json'),result);return result

def main():
 ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('command',choices=['plan','archive','restore','fixture','retire']);ap.add_argument('--tree',type=pathlib.Path);ap.add_argument('--plan',type=pathlib.Path);ap.add_argument('--destination',type=pathlib.Path);ap.add_argument('--reference-review',type=pathlib.Path);a=ap.parse_args()
 if a.command=='fixture':result=fixture()
 elif a.command=='plan':p,result=plan(a.tree);result={'plan_path':str(p),'logical_regular_bytes':result['logical_regular_bytes'],'references':len(result['reference_scan']['references']),'inconclusive':len(result['reference_scan']['inconclusive']),'local_removal_authorized':False}
 elif a.command=='archive':result=archive(a.plan)
 elif a.command=='retire':result=retire(a.plan,a.reference_review)
 else:result=restore(a.plan,a.destination)
 print(json.dumps(result,indent=2))
if __name__=='__main__':main()

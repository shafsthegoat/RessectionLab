"""Read-only original-live vs authenticated canonical history; never builds an Engine.

Default execution requires the successful final primary canonical receipt first.
--preflight checks preparation only and never reads the cache key or manifest.
"""
from pathlib import Path
import argparse,collections,hashlib,hmac,json,os,stat,struct,sys
from datetime import datetime,timezone
if not __debug__:raise RuntimeError('Optimized Python refused: audit assertions must remain enabled')
HERE=Path(__file__).resolve().parent
ROOT=next(p for p in HERE.parents if (p/'src/resectionlab').is_dir())
sys.path.insert(0,str(ROOT/'src'))

def no_engine(frame,event,arg):
 if event=='call' and frame.f_globals.get('__name__')=='resectionlab.surgical_engine' and frame.f_code.co_name not in ('<module>','SurgicalEngine','_OperationCheckFailed'):
  raise RuntimeError('Engine execution prohibited by saved-history comparator')
sys.setprofile(no_engine)
from resectionlab import surgical_verified_cache as vc
from resectionlab.core import semantic_digest,array_digest
from resectionlab.workspace_bundle import load_workspace
from resectionlab.surgical_environment import load_environment

LIVE_SHA='e601eba917d6cc84ddf1e6d97659cb3db8d2b1456e2ffe374de3a1ec15ebdc80'
SOURCE_SHA='ff5a49b8706a8ccda08d8c3e82dad3e079f85dea98cb6bd40783d2262eec9838'
TOTAL=2148

def require(ok,msg):
 if not ok:raise ValueError(msg)

def sha(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()

def exact_difference(a,b,path='$'):
 if type(a) is not type(b):return path+':type'
 if type(a) is float:return None if struct.pack('!d',a)==struct.pack('!d',b) else path+':floatBits'
 if type(a) is dict:
  if a.keys()!=b.keys():return path+':keys'
  for k in a:
   error=exact_difference(a[k],b[k],path+'.'+k)
   if error:return error
  return None
 if type(a) is list:
  if len(a)!=len(b):return path+':length'
  for i,(x,y) in enumerate(zip(a,b)):
   error=exact_difference(x,y,path+'['+str(i)+']')
   if error:return error
  return None
 return None if a==b else path+':value'

def read_existing_key(directory):
 # Deliberately do NOT call _key(): that API can mkdir/chmod/create a key.
 info=directory.lstat();require(stat.S_ISDIR(info.st_mode) and not directory.is_symlink(),'Cache directory is not a real existing directory')
 key_path=directory/'key';info=key_path.lstat()
 require(stat.S_ISREG(info.st_mode) and stat.S_IMODE(info.st_mode)==0o600,'Existing cache key must be regular0600')
 require(not hasattr(os,'getuid') or info.st_uid==os.getuid(),'Cache key owner differs')
 key=vc._read(key_path,32);require(len(key)==32,'Existing cache key length differs')
 return key

def main():
 parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--preflight',action='store_true');args=parser.parse_args()
 require(callable(vc._read) and callable(vc._decode) and callable(vc.implementation_fingerprint),'Required bounded private authentication functions missing')
 script_hash=sha(Path(__file__))
 if args.preflight:
  path=HERE/'sealed-history-comparison-preparation.json'
  report={'status':'prepared-only','scriptSha256':script_hash,'canonicalReceiptPresent':(HERE/'primary-canonical.json').exists(),'authenticationPlan':'Bounded existing key+HMAC manifest read; exact source/snapshot/current implementation/2149state ancestry binding; no cache.restore/_key or Engine call.','requiredGate':'Successful forced primary canonical receipt and publication, matching final frozen artifact','cacheKeyOrManifestRead':False,'engineCalls':0,'finalComparisonPassed':False}
  require(not path.exists(),'Preserve existing preparation receipt')
  path.open('x').write(json.dumps(report,indent=2)+'\n');print(json.dumps(report));return
 # Fail before cache reads unless successful canonical publication is established.
 report_path=HERE/'primary-canonical.json';require(report_path.is_file(),'Primary canonical success receipt not yet published; no cache read attempted')
 canonical_bytes=report_path.read_bytes();canonical=json.loads(canonical_bytes)
 freeze=json.loads((HERE/'artifact-freeze.json').read_text());native=HERE/'rehearsal.ressectionlab';source=ROOT/'testcases/cases/meningioma-sub-06-standard-v2.ressectionlab'
 require(canonical.get('auditStatus')=='passed' and canonical['branch']=='full-primary' and canonical['actions']==TOTAL,'Canonical receipt is not a full successful primary')
 require(canonical['canonicalReplay']['forced'] is True and canonical['canonicalReplay']['cacheUsed'] is False and canonical['canonicalReplay']['verifiedCachePublication']['publications']==1,'Successful canonical publication not established')
 require(sha(native)==freeze['nativeSha256']==canonical['nativeSha256'],'Frozen native differs')
 require(sha(source)==SOURCE_SHA==canonical['originalSha256'],'Original source differs')
 fingerprint=vc.implementation_fingerprint();require(fingerprint is not None and fingerprint==freeze['implementationFingerprint']==canonical['implementationFingerprint'],'Current implementation differs')
 _,artifacts,_=load_workspace(native);snapshot=artifacts['surgical_toolbox']['branches']['full-primary'];snapshot_hash=semantic_digest(snapshot)
 scene=load_environment(native);scene.require_cache_source()
 require(scene.identity==canonical['sceneId']==freeze['sceneId'],'Scene binding differs')
 require(array_digest(scene.labels)==canonical['labelArrayHash'],'Original owned labels differ')
 require(snapshot['stateId']==canonical['stateId']==freeze['branches']['full-primary']['stateId'] and len(snapshot['actions'])==TOTAL,'Snapshot endpoint differs')
 directory=vc.DEFAULT_DIRECTORY
 key=read_existing_key(directory)
 entry=directory/vc.VerifiedSurgicalCache._entry_name(snapshot_hash,fingerprint)
 info=entry.lstat();require(stat.S_ISDIR(info.st_mode) and not entry.is_symlink(),'Sealed entry is not a real directory')
 compressed=vc._read(entry/'manifest.json.gz',vc.MAX_DISK_BYTES);mac=vc._read(entry/'manifest.mac',64)
 require(hmac.compare_digest(mac,hmac.digest(key,compressed,'sha256').hex().encode()),'Sealed canonical manifest authentication failed')
 manifest=vc._decode(compressed)
 require(manifest['schema']==vc.SCHEMA and manifest['snapshotHash']==snapshot_hash and manifest['sceneId']==scene.identity and manifest['implementation']==fingerprint,'Authenticated source/snapshot/implementation ancestry differs')
 history=manifest['history'];states=manifest['states'];require(len(history)==TOTAL and len(states)==TOTAL+1,'Authenticated history/state coverage differs')
 live_path=HERE/'live-primary-history.jsonl';require(sha(live_path)==LIVE_SHA,'Original live history differs')
 live=[json.loads(line) for line in live_path.read_text().splitlines()];require(len(live)==TOTAL,'Original live history coverage differs')
 differences=[];fields=collections.Counter();non_null_contacts=0;event_count=0
 for index,(original,actual) in enumerate(zip(live,history),1):
  require(original['position']==index and set(original)=={'position','outcome'},'Original live position/schema differs')
  expected=original['outcome'];require(actual['action']==snapshot['actions'][index-1],'Authenticated action differs from native snapshot')
  require(states[index-1]['stateId']==actual['stateBefore'] and states[index]['stateId']==actual['stateAfter'],'Authenticated state-chain descriptor differs')
  require(states[index-1]['file']==str(index-1)+'.json.gz' and states[index]['file']==str(index)+'.json.gz','Authenticated state descriptor position differs')
  error=exact_difference(expected,actual)
  if error:differences.append({'position':index,'difference':error})
  else:fields.update(actual.keys())
  non_null_contacts+=expected.get('contactPosition') is not None;event_count+=len(expected['events'])
 require(states[-1]['stateId']==snapshot['stateId'],'Authenticated final state differs')
 require(sha(native)==freeze['nativeSha256'] and sha(source)==SOURCE_SHA and sha(live_path)==LIVE_SHA,'Input files changed during comparison')
 require(report_path.read_bytes()==canonical_bytes and vc.implementation_fingerprint()==fingerprint and semantic_digest(snapshot)==snapshot_hash,'Canonical/fingerprint/snapshot changed during comparison')
 require(hmac.compare_digest(read_existing_key(directory),key) and vc._read(entry/'manifest.json.gz',vc.MAX_DISK_BYTES)==compressed and vc._read(entry/'manifest.mac',64)==mac,'Authenticated cache changed during comparison')
 result={'schema':'resectionlab.original-live-vs-authenticated-canonical-history.v1','status':'passed' if not differences else 'failed','utc':datetime.now(timezone.utc).isoformat(),'nativeSha256':freeze['nativeSha256'],'implementationFingerprint':fingerprint,'originalLiveHistorySha256':LIVE_SHA,'canonicalReceiptSha256':hashlib.sha256(canonical_bytes).hexdigest(),'scriptSha256':script_hash,'authenticatedManifestSha256':hashlib.sha256(compressed).hexdigest(),'snapshotHash':snapshot_hash,'sceneId':scene.identity,'actionsCompared':TOTAL,'allOutcomeFieldsCompared':True,'exactEqualOutcomeFieldCounts':dict(fields),'nonNullContactPositionsCompared':non_null_contacts,'eventsCompared':event_count,'omittedOutcomeFields':[],'mismatchCount':len(differences),'mismatches':differences[:30],'semantics':'Full recursive JSON type/value comparison; float IEEE754 bits including signed zero; list/event ordering preserved; object key order ignored.','authentication':'Existing private key0600/same owner, bounded NOFOLLOW reads, HMAC-SHA256 over complete compressed manifest; sealed source/snapshot/implementation/state-chain identity checks. No key, MAC or key digest disclosed.','engineCalls':0,'cacheWritesOrTouches':0,'limitations':['Comparison authenticates published canonical history and compares original preserved outcomes; it performs no physics replay and no direct stateblob decoding.','Local cache HMAC establishes existing local cache provenance; private key compromise is outside this evidence claim.','This supplements the separate original-live vs canonical JSONL comparison; GUI/video acceptance remains separate.']}
 output=HERE/'live-vs-authenticated-canonical-history.json';require(not output.exists(),'Preserve existing authenticated comparison report');output.open('x').write(json.dumps(result,indent=2)+'\n')
 print(json.dumps({k:result[k] for k in ('status','actionsCompared','nonNullContactPositionsCompared','omittedOutcomeFields','mismatchCount')}))
 require(not differences,'Original live vs authenticated canonical outcome mismatch')

if __name__=='__main__':main()

"""Source-record-only storage estimate. No resectionlab imports or engine use."""
from pathlib import Path
from io import BytesIO
import zipfile,json,gzip,base64,hashlib,math,collections
ROOT=Path(__file__).resolve().parents[1]
primary=ROOT/'primary-v2'
files=[primary/'checkpoints'/name for name in [
 'primary-before-large-core-100.ressectionlab',
 'primary-before-field-cleanup-director-692.ressectionlab',
 'primary-before-exact-thermal-1076.ressectionlab',
 'primary-before-exact-rollback-copy-1486.ressectionlab',
 'primary-before-bounded-left-trial-1644.ressectionlab']]+[primary/'primary.ressectionlab']
rows=[]
for path in files:
 native=path.read_bytes()
 with zipfile.ZipFile(BytesIO(native)) as z: manifest=json.loads(z.read('manifest.json'))
 enc=manifest['artifacts']['surgical_toolbox']; assert enc['uncompressedBytes']<=128*1024**2
 record=json.loads(gzip.decompress(base64.b64decode(enc['payload'])))
 branch=record['branchId']; snapshot=record['branches'][branch]; state=snapshot['checkpoint']['state']
 fields={}
 for key,value in state.items():
  raw=json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
  fields[key]={'jsonBytes':len(raw),'gzip1Bytes':len(gzip.compress(raw,compresslevel=1,mtime=0))}
 count=len(snapshot['actions']); seconds=state['elapsedSeconds']
 requested_frames=sum(math.floor((float(a['duration'])+1e-10)/.2)+2 for a in snapshot['actions'])
 frames_upper=min(2*count+math.floor((seconds+1e-8)/.2),requested_frames)
 frames_lower=max(2*count,count+math.floor((seconds+1e-8)/.2)+1)
 row={'native':str(path),'nativeSha256':hashlib.sha256(native).hexdigest(),'actions':count,'actualElapsedSeconds':seconds,
      'requestedDurationSeconds':sum(float(a['duration']) for a in snapshot['actions']),
      'estimatedFrameCountLower':frames_lower,'estimatedFrameCountUpper':frames_upper,
      'removedCells':len(state['removed']),'fields':fields,'allCompressedFieldBytes':sum(v['gzip1Bytes'] for v in fields.values())}
 rows.append(row)
 del native,manifest,record,state,enc
rows.sort(key=lambda row:row['actions'])
last=rows[-1];all_vary=drill_vary=0; previous_count=previous_elapsed=0; intervals=[]
for row in rows:
 actions=row['actions']-previous_count;elapsed=row['actualElapsedSeconds']-previous_elapsed
 frames=2*actions+math.floor((elapsed+1e-8)/.2)
 # A deliberately conservative scenario using each interval's endpoint field
 # sizes; stable flapComponents are excluded because the actual store dedups.
 varying=sum(v['gzip1Bytes'] for k,v in row['fields'].items() if k!='flapComponents')
 all_vary+=frames*varying
 drill=row['fields'].get('drillingV2',{}).get('gzip1Bytes',0);drill_vary+=frames*drill
 intervals.append({'fromActions':previous_count,'toActions':row['actions'],'elapsedSeconds':elapsed,
                  'framesUpper':frames,'endpointNonFlapFieldBytes':varying,'endpointDrillingFieldBytes':drill})
 previous_count=row['actions'];previous_elapsed=row['actualElapsedSeconds']
report={'scope':'Read-only native JSON/checkpoint records only; no engine construction, patient physics, GUI or cache publication.',
 'snapshots':rows,'frameCountEstimateMeaning':'Cadence resets per action; successful dt is <=0.1s. Count includes starts and finalized boundaries. Requested actions bound actual partial durations above; no exact frame count is claimed before observed replay.',
 'estimatedFrameCountLower':last['estimatedFrameCountLower'],'estimatedFrameCountUpper':last['estimatedFrameCountUpper'],
 'scenarioBytes':{'drillingFieldChangesEveryObservedFrame':drill_vary,'everyNonFlapFieldChangesEveryObservedFrame':all_vary,
                  'endpointNonFlapFieldBytesTimesAllFrames':(last['allCompressedFieldBytes']-last['fields']['flapComponents']['gzip1Bytes'])*last['estimatedFrameCountUpper']},
 'scenarioMeaning':'Endpoint-size envelopes over recorded intervals, not measurements or rigorous bounds of unseen intermediate states. Content dedup reduces repeated fields. Temperature cooling and removal can make large fields unique; actual cache coverage must be checked after same-pass collection.',
 'intervals':intervals,
 'currentObservedBudgetBytes':512*1024**2,'perActionReservationBytes':64*1024**2,
 'currentRetentionIssue':'At new action begin, _room(reserve,pending) may delete the current committed entry itself. Thus this branch exceeding about448MiB can evict all its already-published actions at once; publication count alone cannot prove complete retention.',
 'helperReadiness':{'accept_native.py':'One canonical independent force-verified replay; signed cache captures every action boundary, no observed microstep collector here.',
  'export_observed_telemetry.py':'Cache-only admitted baseline; --prepare-frames collects during the same observed telemetry pass. Verifies publications==actions and every action manifest remains present before export.',
  'verify_every_cached_frame.py':'Fresh-process signed admission, SHA-authenticated field blobs, every real frame restored and boundary checked; sys.setprofile refuses any physics step.'}}
out=Path(__file__).with_name('frame-disk-readiness-source-records.json');out.write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({'report':str(out),'actions':last['actions'],'elapsedSeconds':last['actualElapsedSeconds'],'frames':[report['estimatedFrameCountLower'],report['estimatedFrameCountUpper']],'scenarioGiB':{k:round(v/1024**3,3) for k,v in report['scenarioBytes'].items()},'largestFields':sorted(last['fields'].items(),key=lambda x:x[1]['gzip1Bytes'],reverse=True)[:4]},indent=2))

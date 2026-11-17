#!/usr/bin/env python3
"""Index actual saved episode frames and compact source-bound replay evidence."""
from __future__ import annotations
import argparse,hashlib,html,json,os,struct,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def file(p):
 p=Path(p).resolve(strict=True);return dict(path=str(p.relative_to(ROOT)),sha256=sha(p),bytes=p.stat().st_size)
def write_new(p,data):
 if p.exists():
  if p.read_bytes()==data:return
  raise ValueError('Preserve previous artifact; choose a new output path: '+str(p))
 p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(data)
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--episode-index',type=Path,required=True);p.add_argument('--warm-report',type=Path,required=True);p.add_argument('--cold-report',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--manifest',type=Path);a=p.parse_args()
 output=a.output.resolve();assert output.is_relative_to(ROOT/'testcases')
 sys.path.insert(0,str(ROOT/'src'))
 from resectionlab.imaging import load_case,read_case_artifacts
 from resectionlab.surgical_scene import from_case
 packet=read(a.episode_index);warm=read(a.warm_report);cold=read(a.cold_report)
 assert warm['passed'] and cold['passed'] and warm['codeStable'] and cold['codeStable'] and not warm['windowEverVisible'] and not cold['windowEverVisible']
 assert cold['sourceSha256']==warm['savedSha256'] and Path(cold['input']).resolve()==Path(warm['saved']).resolve()
 assert cold['initial']==warm['savedSnapshot']
 case=Path(cold['saved']);native=load_case(case);record=read_case_artifacts(case)['surgical_toolbox'];scene=from_case(native,record['mapping']);assert native.semantic_hash==packet['caseHash'];voxel=scene.voxel_mm3
 open_command='./scripts/launch_poc.sh --case "$PWD/'+str(case.relative_to(ROOT))+'"'
 entries=[];cards=[]
 def link(target,label):return '<a href="'+html.escape(os.path.relpath(Path(target).resolve(),output.parent),quote=True)+'">'+html.escape(label)+'</a>'
 for index,row in enumerate(packet['episodes'],1):
  frames=[]
  for phase,suffix in [('Work start','work-start'),('Native interior','interior'),('Endpoint','endpoint')]:
   name=f'episode-{index}-{suffix}.png';capture=next(c for c in cold['captures'] if c['name']==name)
   assert capture['branchId']==row['branchId']
   if suffix=='interior':assert capture['playback'] and not capture['playback']['isActionBoundary'] and .5<capture['playback']['actionElapsedSeconds']/capture['playback']['actionDurationSeconds']<.999999
   if suffix=='endpoint':assert capture['stateId']==row['finalStateId'] and capture['position']==row['nativeActions']
   path=Path(a.cold_report).parent/name;raw=path.read_bytes();assert raw[:8]==b'\x89PNG\r\n\x1a\n';width,height=struct.unpack('>II',raw[16:24]);assert width>=1000 and height>=700
   frames.append(dict(phase=phase,**file(path),branchId=capture['branchId'],stateId=capture['stateId'],position=capture['position'],interior=bool(capture['playback'] and not capture['playback']['isActionBoundary'])))
  original=row['originalEpisode'];assert sha(original['path'])==original['sha256']
  if row['checkpoint']:assert sha(row['checkpoint']['path'])==row['checkpoint']['sha256']
  residual=row['progress']['residualOriginalTargetCells']*voxel
  entry=dict(branchId=row['branchId'],method=row['method'],seed=row['seed'],nativeActions=row['nativeActions'],actualPhysicalRequests=len(row['attempts']),actualRefusals=sum(r['status']=='rejected' for r in row['attempts']),targetMm3=row['progress']['originalTargetTreatmentMm3'],residualMm3=residual,neuralMm3=row['neuralRemovedMm3'],structureRemovedMm3=row['structureRemovedMm3'],endpoint=row['endpoint']['classification'],cleanup=row['cleanup'],onlineCost=row['computation'],originalEpisode=original,checkpoint=row['checkpoint'],finalStateId=row['finalStateId'],historyHash=row['actionHistoryHash'],frames=frames)
  entries.append(entry)
  media=''.join('<figure><a href="'+html.escape(os.path.relpath(ROOT/f['path'],output.parent),quote=True)+'"><img loading="lazy" src="'+html.escape(os.path.relpath(ROOT/f['path'],output.parent),quote=True)+'" alt="'+html.escape(row['label']+' '+f['phase'],quote=True)+'"></a><figcaption>'+html.escape(f['phase'])+' · native position '+str(f['position'])+'</figcaption></figure>' for f in frames)
  cards.append('<section data-method="'+html.escape(row['method'])+'"><h2>'+html.escape(row['label'])+'</h2><p>Target '+str(entry['targetMm3'])+' mm³ · residual '+str(residual)+' mm³ · direct neural '+str(entry['neuralMm3'])+' mm³ · '+html.escape(entry['endpoint'])+'. '+str(entry['nativeActions'])+' accepted native actions; '+str(entry['actualRefusals'])+' actual refused requests.</p><p>'+link(original['path'],'Original recorded episode JSON')+(' · '+link(row['checkpoint']['path'],'Original selected checkpoint') if row['checkpoint'] else '')+'</p><div class="frames">'+media+'</div><details><summary>Bindings, costs, cleanup and original frame SHA-256</summary><pre>'+html.escape(json.dumps(entry,indent=2))+'</pre></details></section>')
 manifest=dict(schema='resectionlab.recorded-sequential-replay-delivery.v1',verificationCommit=cold.get('sourceCommit'),scope=packet['scope'],caseHash=packet['caseHash'],sourceSceneId=packet['sourceSceneId'],policyHash=packet['policyHash'],configHash=packet['configHash'],parent=packet['parent'],tier=packet['tier'],nativeCase=file(case),episodeIndex=file(a.episode_index),warmProof=file(a.warm_report),coldProof=file(a.cold_report),originalProtocol=packet['originalProtocol'],originalFreeze=packet['originalFreeze'],openCommand=open_command,episodes=entries,sourceFiles={p:sha(ROOT/p) for p in ['scripts/export_sequential_episodes.py','scripts/build_sequential_replay_index.py','src/resectionlab/surgical_recorded_sequential.py','src/resectionlab/surgical_scene.py','src/resectionlab/surgical_bridge.py','desktop/src/SequentialEpisodePanel.tsx','desktop/src/sequential-episode-data.ts','desktop/src/SurgicalWorkflow.tsx','desktop/src/SurgicalPanel.css','desktop/src/interactive-planning-data.ts','desktop/src/viewer/interactivePlanningGeometry.ts','desktop/src/surgical-comparison-context.ts','desktop/electron/verify-sequential-episodes.cjs']},unmetCriteria=[])
 document='''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Actual learned episodes · RessectionLab</title><style>body{margin:0;background:#0d1513;color:#e4ece8;font:15px/1.5 system-ui,sans-serif}main{max-width:1450px;padding:30px 24px;margin:auto}h1{font-size:32px}h2{font-size:22px}a{color:#9de3ce}section,.open{border:1px solid #344b40;border-radius:12px;padding:18px;margin:24px 0;background:#111e19}pre{white-space:pre-wrap;overflow-wrap:anywhere;font-size:12px}.frames{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:12px}figure{margin:0}img{width:100%;height:auto;border-radius:8px}figcaption{font-size:13px;color:#b4c8bd}.note{color:#b4c8bd}@media(max-width:900px){.frames{grid-template-columns:1fr}}</style><main><p class="note">Generated post-access diagnostic · patient operations: 0 · no new fitting</p><h1>Actual learned episodes beside their controls</h1><p>These are original saved episodes from learned seeds 17 and 43, greedy, constrained search and the fixed 0.4 mm tip control on the same original layout and low budget. Each three-frame sequence is captured from real native playback. Images are unchanged; there is no synthetic animation.</p><p>Earlier opening/access costs and functional motor/language consequences remain unknown. Direct generated motor/CST memberships overlap and must not be summed. Selection and replay do not establish robust learned superiority or patient eligibility.</p><div class="open"><h2>Open the saved native workspace</h2><p>'''+link(case,'Native workspace')+' · '+link(a.episode_index,'Episode bindings')+' · '+link(a.warm_report,'Real Save proof')+' · '+link(a.cold_report,'Cold process proof')+' · '+link(packet['originalProtocol']['path'],'Original protocol')+' · '+link(packet['originalFreeze']['path'],'Original checkpoint freeze')+'</p><pre>'+html.escape(open_command)+'</pre><p>In Rehearse choose Compare, select an episode, then use Replay selected actual episode or the native timeline. Open Actual accepted and refused physical requests for tool settings and refusal reasons.</p></div>'+''.join(cards)+'<details><summary>Complete compact replay manifest</summary><pre>'+html.escape(json.dumps(manifest,indent=2))+'</pre></details></main></html>'
 write_new(output,document.encode());manifest['index']=file(output)
 if a.manifest:
  target=a.manifest.resolve();assert target.is_relative_to(ROOT/'docs') and target.suffix=='.json'
  target.parent.mkdir(parents=True,exist_ok=True);target.write_text(json.dumps(manifest,indent=2)+'\n')
 write_new(output.with_suffix('.json'),(json.dumps(manifest,indent=2)+'\n').encode());print(json.dumps(dict(index=str(output),episodes=len(entries),frames=15,manifest=str(a.manifest))))
if __name__=='__main__':main()

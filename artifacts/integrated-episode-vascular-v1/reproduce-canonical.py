from pathlib import Path
import json
from resectionlab.desktop_bridge import BridgeRuntime
from resectionlab.shared_vascular_evaluation import evaluate_development_episode_vascular
out=Path(__file__).resolve().parent
events=[]
runtime=BridgeRuntime(out/'bridge-transfers',events.append)
def call(serial,op,args):
    runtime.submit({'id':str(serial),'op':op,'args':args})
    assert runtime.wait_idle(30),op
    result=[e for e in events if e['id']==str(serial)][-1]
    assert result['event']=='result',result
    return result['result']
try:
    episode=call(1,'executeDevelopmentEpisode',{'fixture':'generated-sequential-v1','selector':'scripted'})['episode']
    result=call(2,'evaluateDevelopmentEpisodeVascular',{'caseHash':episode['caseHash'],'episodeId':episode['episodeId']})
    (out/'canonical-ui-fixture.json').write_text(json.dumps({'episode':episode,'result':result},sort_keys=True,indent=2,allow_nan=False)+'\n')
    direct=evaluate_development_episode_vascular(episode=episode,output_directory=out/'retained-evaluation')
    assert direct==result['evaluation']
    print(json.dumps({'episodeId':episode['episodeId'],'evaluationId':direct['evaluationId'],'status':'canonical_bridge_and_durable_report_match','actions':len(direct['perAction']),'wholeTool':direct['wholeTool']}))
finally:
    runtime.close()

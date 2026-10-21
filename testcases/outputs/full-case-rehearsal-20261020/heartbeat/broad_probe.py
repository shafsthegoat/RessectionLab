exec(open('testcases/outputs/full-case-rehearsal-20261020/heartbeat/run.py').read().split('actions=json.load')[0])
import pickle
q=[1.,0,0,0]; entry=[-10.464996337890625,30.441162109375,-421.25]
acts=[{'slot':'left','operation':'place','duration':.2,'parameters':{'radiusMm':8.,'forceN':20.,'rpm':80000.,'duty':1.},'pose':{'position':[entry[0],entry[1],-369.25],'quaternion':q}}]
for i in range(12):
 if i: a={'slot':'left','operation':'apply','duration':30.,'pose':{'position':entry,'quaternion':q}}
 else:a=acts[0]
 t=time.perf_counter();r=engine.step(a); print(i,r['status'],r.get('reason'),engine.state['slots']['left']['position'],engine.metrics()['drillRemovedVolumeMm3'],time.perf_counter()-t,flush=True)
 if i>1 and r.get('reason')!='BONE_CUTTING_IN_PROGRESS':break
(out/'broad-probe-snapshot.json').write_text(json.dumps(engine.snapshot()))
pickle.dump(engine,open(out/'broad-probe-engine.pkl','wb'))

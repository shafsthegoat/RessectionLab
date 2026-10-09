"""No-network reproduction: advancing wall time beyond campaign during supervision."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
from unittest.mock import patch

root = Path.cwd()
sys.path.insert(0, str(root/'scripts'))
import acquire_synthrad_task1_archive as a
outdir = root/'build/synthrad-archive-runner-independent-review-v1'
qroot = outdir/'live-campaign-control-v2'
qroot.mkdir()
m = a.load_manifest()
q = a.Queue(qroot,m)
s = next(iter(q.sources.values()))
fake_wall = [time.time()]
real_popen = subprocess.Popen
launches=[]
def local_only(command, *args, **kwargs):
    assert str(command[0]) == sys.executable and command[1:3] == ['-B','-c'], 'Only explicit harmless local child allowed'
    launches.append(command)
    return real_popen(command,*args,**kwargs)
with patch.object(a.time,'time',lambda:fake_wall[0]), patch.object(a.subprocess,'Popen',local_only):
    active=a.campaign(q,create=True)
    monotonic_campaign_end=time.monotonic()+q.bounds['total_campaign_seconds']
    def launcher(command,log,**kwargs):
        on_start=kwargs['on_start']
        def started(pid):
            on_start(pid)
            # Models system sleep: wall clock crosses fixed end while monotonic
            # time barely advances. No scientific bytes or network are involved.
            fake_wall[0]=active['deadline_epoch']+1
        return a.supervise([sys.executable,'-B','-c','import time; time.sleep(0.1)'],log,
            deadline=kwargs['deadline'],grace_seconds=kwargs['grace_seconds'],
            log_bytes=kwargs['log_bytes'],on_start=started)
    result=a.run_one(q,s,{}, {'available':True},monotonic_campaign_end,active,launcher=launcher)
    trial,intent,saved=q.attempts(s)[0]
    supervision=json.loads((trial/'supervision.json').read_text())
    assert fake_wall[0]>active['deadline_epoch']
    assert supervision['state']=='completed', supervision
    assert supervision['returncode']==0 and supervision['elapsed_seconds']<2
    assert saved['status']=='interrupted'
    evidence={
      'finding_reproduced':True,
      'campaign_expired_wall_seconds':fake_wall[0]-active['deadline_epoch'],
      'supervisor_still_reported':supervision,
      'parent_closed_after_expiry_status':result['status'],
      'local_children':len(launches), 'network_requests':0, 'patient_payload_reads':0,
      'method':'Real run_one/supervise with harmless local child replacing worker; fake wall time crosses campaign end after on_start while monotonic deadline remains in future.'
    }
(outdir/'live-campaign-reproduction.json').write_text(json.dumps(evidence,indent=2)+'\n')
print(json.dumps(evidence,indent=2))

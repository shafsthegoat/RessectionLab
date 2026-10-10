from pathlib import Path
import json,os,signal,subprocess,time
root=Path.cwd();out=root/'build/learned-transfer-root-integration-v1/live-reopen';out.mkdir(exist_ok=False)
node='/Users/sayemkamal/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/bin/node'
started=time.monotonic();peak=0;samples=0;reason=None
with (out/'electron.log').open('w') as log:
 p=subprocess.Popen([node,'desktop/node_modules/electron/cli.js','desktop','--case','build/integrated-workspace-persistence-root/trained-transfer-live.ressectionlab'],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
 (out/'owner.json').write_text(json.dumps({'pid':p.pid,'max_seconds':120,'max_tree_rss_bytes':1610612736}))
 try:
  while p.poll() is None:
   rows=[tuple(map(int,x.split())) for x in subprocess.check_output(['ps','-axo','pid=,ppid=,rss='],text=True,timeout=2).splitlines()]
   owned={p.pid}
   for _ in range(12):
    expanded=owned|{pid for pid,parent,rss in rows if parent in owned}
    if expanded==owned:break
    owned=expanded
   rss=sum(r*1024 for pid,parent,r in rows if pid in owned);peak=max(peak,rss);samples+=1
   state={'elapsed_seconds':time.monotonic()-started,'tree_rss_bytes':rss,'peak_tree_rss_bytes':peak,'samples':samples,'pids':sorted(owned)}
   (out/'progress.json').write_text(json.dumps(state))
   if rss>1610612736:reason='tree_rss_limit';break
   if time.monotonic()-started>120:reason='bounded_display_session_end';break
   time.sleep(.5)
 finally:
  if p.poll() is None:
   for sig in (signal.SIGTERM,signal.SIGKILL):
    try:os.killpg(p.pid,sig)
    except ProcessLookupError:break
    try:p.wait(timeout=2);break
    except subprocess.TimeoutExpired:continue
  result={'elapsed_seconds':time.monotonic()-started,'sampled_peak_tree_rss_bytes':peak,'samples':samples,'reason':reason,'exit_code':p.poll(),'worker_reaped':p.poll() is not None,'transient_peaks_and_gpu_memory_measured':False}
  (out/'terminal.json').write_text(json.dumps(result,indent=2))
  print(json.dumps(result))

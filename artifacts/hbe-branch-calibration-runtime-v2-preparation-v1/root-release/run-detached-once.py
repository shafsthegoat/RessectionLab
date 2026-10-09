from pathlib import Path
import datetime,hashlib,json,os,subprocess,time
out=Path(__file__).resolve().parent
pre=json.loads((out/'preflight.json').read_text())
command=pre['command']+['--execute']
started=time.monotonic()
with (out/'execution.log').open('xb') as log:
    try:
        result=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT)
        terminal={'exit_code':result.returncode,'elapsed_seconds':time.monotonic()-started,'finished_utc':datetime.datetime.now(datetime.UTC).isoformat(),'log_sha256':hashlib.sha256((out/'execution.log').read_bytes()).hexdigest()}
    except BaseException as error:
        terminal={'exit_code':None,'elapsed_seconds':time.monotonic()-started,'finished_utc':datetime.datetime.now(datetime.UTC).isoformat(),'harness_error':{'type':type(error).__name__,'message':str(error)}}
with (out/'terminal.json').open('x') as f: json.dump(terminal,f,indent=2);f.write('\n')

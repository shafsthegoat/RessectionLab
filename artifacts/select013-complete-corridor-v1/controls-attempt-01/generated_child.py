import os,sys
from pathlib import Path
R=Path(sys.argv[1]);scratch=Path(sys.argv[2]).resolve()
sys.path.insert(0,str(R/'src'))
def guard(event,args):
 if event=='open' and args and isinstance(args[0],(str,bytes)):
  p=Path(os.fsdecode(args[0])).resolve()
  if p.is_relative_to(R/'data') or p.is_relative_to(R/'sources') or p.suffix.lower() in ('.dcm','.nii','.gz','.pt','.pth','.ckpt','.psckpt') or (p.suffix=='.npy' and not p.is_relative_to(scratch)) or p==R/'build/hybrid-proposal-balanced-v2/attempt-01/balanced_ridge_top8.json':
   raise PermissionError('Generated-only control blocked acquired payload: '+str(p))
 if event in ('subprocess.Popen','os.system','os.fork','socket.connect'):raise PermissionError('No child process/network in generated controls')
sys.addaudithook(guard)
import pytest
raise SystemExit(pytest.main(['-q','-p','no:cacheprovider','--basetemp='+str(scratch),str(R/'build/select013-complete-corridor-v1/tests')]))

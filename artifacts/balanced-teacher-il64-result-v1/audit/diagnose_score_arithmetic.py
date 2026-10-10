"""Scalar-only diagnosis of the inherited float32 score tolerance."""
import json,math,runpy
from pathlib import Path
A=Path(__file__).parent
h=runpy.run_path(str(A/'audit_saved.py'));rows=[]
for s in h['SUBJECTS']:
 for step in range(2 if s=='ReMIND-025' else 1):
  state=h['read'](h['P']/'attempt-01/teacher-readout'/s/f'state-{step:02d}.json')
  label=state['action_ids'].index(state['teacher_action']);mask=state['action_mask']
  for name in ['balanced_IL_64','balanced_IL_8','unweighted_IL_8']:
   row=state[name];logits=row['logits'];legal=[i for i,x in enumerate(mask) if x]
   top=max(logits[i] for i in legal);den=sum(math.exp(logits[i]-top) for i in legal)
   p=[math.exp(logits[i]-top)/den if mask[i] else 0. for i in range(len(logits))]
   ce=math.log(den)+top-logits[label]
   rows.append({'subject':s,'step':step,'endpoint':name,'teacher_CE':row['teacher_CE'],'scalar_CE':ce,'CE_absolute_error':abs(row['teacher_CE']-ce),
    'probability_max_absolute_error':max(abs(a-b) for a,b in zip(p,row['probabilities'])),'logit_min':min(logits[i] for i in legal),'logit_max':top,
    'teacher_margin_absolute_error':abs(row['teacher_minus_STOP']-(logits[label]-logits[0]))})
with (A/'score-arithmetic-diagnosis.json').open('x') as f:json.dump({'scores':rows,'input_hashes':h['seen']},f,indent=2,sort_keys=True);f.write('\n')
print(json.dumps(rows,indent=2))

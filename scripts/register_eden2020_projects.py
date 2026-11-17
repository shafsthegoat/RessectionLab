#!/usr/bin/env python3
"""Register verified local EDEN projects and refresh their private HTML index. Run from the checkout."""
from pathlib import Path
import json,time,os,html,argparse
from resectionlab.data_root import checked_root
parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',type=Path);parser.add_argument('--watch',action='store_true');args=parser.parse_args()
repo=Path.cwd();root=checked_root(args.root);base=root/'prepared/eden2020-synthetic-v1'
if not (repo/'src/resectionlab').is_dir():raise ValueError('Run from the primary checkout')
(repo/'testcases/cases').mkdir(parents=True,exist_ok=True)
while True:
 records=[json.loads(p.read_text()) for p in sorted((base/'projects').glob('EDEN2020-*.json'))] if (base/'projects').exists() else []
 for r in records:
  source=root/r['path'];link=repo/'testcases/cases'/source.name
  if not source.is_file() or source.stat().st_size!=r['bytes']:raise ValueError('Verified source package unavailable or changed')
  if link.exists() and not link.is_symlink():raise ValueError('Preserve conflicting prepared project')
  if not link.is_symlink() and not link.exists():link.symlink_to(source)
  if link.is_symlink() and link.resolve()!=source:raise ValueError('Conflicting prepared project link')
 count=sum(r['synthetic'] for r in records)
 status={'verified_native_projects':len(records),'synthetic_projects':count,'baseline_projects':len(records)-count,'source_volunteers':len(set(r['person_id'] for r in records)),'complete':count==120 and len(records)==135}
 p=base/'progress.json';tmp=p.with_name(p.name+f'.{os.getpid()}.partial');tmp.write_text(json.dumps(status,indent=2));tmp.replace(p)
 rows=[]
 for r in records:
  recipe=r['recipe'] or {};vol=sum(v['volume_mm3'] for k,v in r['compartments'].items() if k!='synthetic_edema')/1000
  rows.append('<tr><td><a href="'+html.escape(str((root/r['path']).as_uri()))+'">'+html.escape(r['case_id'])+'</a></td><td>'+('Synthetic' if r['synthetic'] else 'Acquired baseline')+'</td><td>'+('%.2f mL'%vol if r['synthetic'] else '—')+'</td><td>'+html.escape(recipe.get('hemisphere_proxy','—'))+'</td><td><a href="'+html.escape((root/r['contact_sheet']).as_uri())+'">3-plane comparison</a></td></tr>')
 title='EDEN2020 · healthy hosts & synthetic tumors'
 doc='<!doctype html><meta charset="utf-8"><title>'+title+'</title><style>body{max-width:1250px;margin:40px auto;padding:0 24px;font:16px system-ui;background:#f7f8fb;color:#152238}h1{font-size:30px}p{line-height:1.6}table{border-collapse:collapse;width:100%;background:white}th,td{padding:12px;text-align:left;border-bottom:1px solid #ddd}a{color:#2455aa}.count{font-size:22px;font-weight:600}</style><h1>'+title+'</h1><p class="count">'+str(count)+' synthetic scenarios · '+str(len(records)-count)+' baseline projects</p><p>Native RessectionLab projects on the external data drive. Open <b>Cases</b> in the checkout app and search <b>EDEN2020</b>, or use File → Open on a project below. Project files include their original T1 reference. Baseline projects include FLAIR, MRA and SWI references; large reference channels use a labeled 1 mm display derivative. Full source archives, diffusion and tractography are preserved under sources/eden2020-healthy-3994749.</p><p><b>Explicitly synthetic:</b> randomized brain-interior lesions with a generated core, rim, edema shell and T1 signal changes. These are engineering scenarios with no simulated mass effect, clinical placement validation, functional mapping or validated surgical access. The 120 variants remain 15 source people. Private noncommercial adaptations only; do not redistribute derived imaging under the source CC BY-NC-ND 4.0 terms.</p><p>Source: <a href="https://zenodo.org/records/3994749">Castellano et al., EDEN2020 healthy-volunteer MRI, DOI 10.5281/zenodo.3994749</a>. Brain envelope: SynthStrip estimate (Hoopes et al., NeuroImage 2022). Acquired baseline values outside each synthetic lesion remain exact.</p><table><thead><tr><th>Project</th><th>Type</th><th>Core + rim volume</th><th>Position proxy</th><th>QA</th></tr></thead><tbody>'+''.join(rows)+'</tbody></table>'
 tmp=base/f'index.html.{os.getpid()}.partial';tmp.write_text(doc);tmp.replace(base/'index.html')
 if status['complete'] or not args.watch:print(json.dumps(status),flush=True);break
 time.sleep(20)

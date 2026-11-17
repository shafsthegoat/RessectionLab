#!/usr/bin/env python3
"""Register verified local EDEN projects and refresh their private HTML index. Run from the checkout."""
from pathlib import Path
import json,time,os,html,argparse
from resectionlab.data_root import checked_root
parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',type=Path);parser.add_argument('--watch',action='store_true');parser.add_argument('--app-prepared-root',type=Path)
parser.add_argument('--collection',choices=['eden2020-synthetic-v1','eden2020-anatomy-v2'],default='eden2020-synthetic-v1')
parser.add_argument('--prefer-segmented-v2',action='store_true',help='After all135 v2 projects exist, replace verified v1 prepared links; preserve all HDD files.')
args=parser.parse_args()
if args.prefer_segmented_v2 and args.collection!='eden2020-anatomy-v2':raise ValueError('Preference requires the segmented v2 collection')
repo=Path.cwd();root=checked_root(args.root);base=root/'prepared'/args.collection
if not (repo/'src/resectionlab').is_dir():raise ValueError('Run from the primary checkout')
link_roots=[repo/'testcases/cases']+([args.app_prepared_root.resolve()] if args.app_prepared_root else [])
for directory in link_roots:directory.mkdir(parents=True,exist_ok=True)
while True:
 records=[json.loads(p.read_text()) for p in sorted((base/'projects').glob('EDEN2020-*.json'))] if (base/'projects').exists() else []
 for r in records:
  source=root/r['path']
  if not source.is_file() or source.stat().st_size!=r['bytes']:raise ValueError('Verified source package unavailable or changed')
  for directory in link_roots:
   link=directory/source.name
   if link.exists() and not link.is_symlink():raise ValueError('Preserve conflicting prepared project')
   if not link.is_symlink() and not link.exists():
    try:link.symlink_to(source)
    except FileExistsError:pass
   if not link.is_symlink() or link.resolve()!=source:raise ValueError('Conflicting prepared project link')
 count=sum(r['synthetic'] for r in records)
 status={'verified_native_projects':len(records),'synthetic_projects':count,'baseline_projects':len(records)-count,'source_volunteers':len(set(r['person_id'] for r in records)),'complete':count==120 and len(records)==135}
 if args.prefer_segmented_v2 and status['complete']:
  old_base=root/'prepared/eden2020-synthetic-v1/projects'
  for p in old_base.glob('EDEN2020-*.json'):
   old=json.loads(p.read_text());source=root/old['path']
   if not source.is_file()or source.stat().st_size!=old['bytes']:raise ValueError('Historical v1 project unavailable; preserve its link')
   for directory in link_roots:
    link=directory/source.name
    if link.is_symlink():
     if link.resolve()!=source:raise ValueError('Conflicting old version link')
     link.unlink()
 p=base/'progress.json';tmp=p.with_name(p.name+f'.{os.getpid()}.partial');tmp.write_text(json.dumps(status,indent=2));tmp.replace(p)
 rows=[]
 for r in records:
  recipe=r['recipe'] or {};vol=sum(v['volume_mm3'] for k,v in r['compartments'].items() if k in {'synthetic_core','synthetic_rim'})/1000
  rows.append('<tr><td><a href="'+html.escape(str((root/r['path']).as_uri()))+'">'+html.escape(r['case_id'])+'</a></td><td>'+('Synthetic' if r['synthetic'] else 'Acquired baseline')+'</td><td>'+('%.2f mL'%vol if r['synthetic'] else '—')+'</td><td>'+html.escape(recipe.get('placement_stratum',recipe.get('hemisphere_proxy','—')))+'</td><td><a href="'+html.escape((root/r['contact_sheet']).as_uri())+'">Source and shape QA</a></td></tr>')
 title='EDEN2020 · healthy hosts & synthetic tumors'
 doc='<!doctype html><meta charset="utf-8"><title>'+title+'</title><style>body{max-width:1250px;margin:40px auto;padding:0 24px;font:16px system-ui;background:#f7f8fb;color:#152238}h1{font-size:30px}p{line-height:1.6}table{border-collapse:collapse;width:100%;background:white}th,td{padding:12px;text-align:left;border-bottom:1px solid #ddd}a{color:#2455aa}.count{font-size:22px;font-weight:600}</style><h1>'+title+'</h1><p class="count">'+str(count)+' synthetic scenarios · '+str(len(records)-count)+' baseline projects</p><p>Native RessectionLab projects on the external data drive. Open <b>Cases</b> in the checkout app and search <b>EDEN2020</b>, or use File → Open on a project below. Project files include their original T1 reference. Baseline projects include eligible FLAIR, MRA and SWI references; original header conflicts are recorded and their reference views withheld; large reference channels use a labeled 1 mm display derivative. Full source archives, diffusion and tractography are preserved under sources/eden2020-healthy-3994749.</p><p><b>Explicitly synthetic:</b> randomized brain-interior lesions with a generated core, rim, edema shell and T1 signal changes. These are engineering scenarios with no simulated mass effect, clinical placement validation, functional mapping or validated surgical access. The 120 variants remain 15 source people. Private noncommercial adaptations only; do not redistribute derived imaging under the source CC BY-NC-ND 4.0 terms.</p><p>Source: <a href="https://zenodo.org/records/3994749">Castellano et al., EDEN2020 healthy-volunteer MRI, DOI 10.5281/zenodo.3994749</a>. Brain envelope: SynthStrip estimate (Hoopes et al., NeuroImage 2022). Acquired baseline values outside each synthetic lesion remain exact.</p><table><thead><tr><th>Project</th><th>Type</th><th>Core + rim volume</th><th>Position proxy</th><th>QA</th></tr></thead><tbody>'+''.join(rows)+'</tbody></table>'
 if args.collection=='eden2020-anatomy-v2':
  doc=doc.replace(title,'EDEN2020 · segmented anatomy & organic synthetic tumors')
  start=doc.index('<p>Native RessectionLab');end=doc.index('<table>',start)
  doc=doc[:start]+'''<p>Open <b>Cases</b> in RessectionLab and search <b>EDEN2020</b> or <b>SEGMENTED-v2</b>. Each native project has model-estimated full-head anatomy, an MRA vessel estimate, available published tract bundles, and the unchanged native healthy T1 plus registered FLAIR/MRA/SWI references. MRI labels include gray/white matter, deep nuclei, ventricles/CSF, brainstem/cerebellum, skull/scalp, optic nerves, arteries and veins. Full1mm model labels and all original scans remain on the drive.</p><p>Eight seeded tumor locations per volunteer span left/right, anterior/posterior and superior/inferior tissue strata. Targets use rotated, asymmetric, lobulated implicit surfaces with separate core, rim and edema. Every target is checked for connectivity, tissue containment and non-cuboid shape, and shown in the source/shape QA.</p><p><b>Engineering rehearsal only.</b> The bounded full-head rehearsal grid is explicitly resampled; exact native T1 is retained. Anatomy, registration and vascular coverage are unreviewed estimates. MRI skull labels are not CT bone measurements. No dura segmentation, complete functional mapping, mass effect, safe surgical corridor or clinical validation is supplied. Baseline tracts and vessels remain undisplaced. The120 variants remain15 source people, all DEVELOPMENT, with no training admission.</p><p>Source: <a href="https://zenodo.org/records/3994749">Castellano et al., EDEN2020 healthy-volunteer MRI, DOI10.5281/zenodo.3994749</a>. Head labels: <a href="https://surfer.nmr.mgh.harvard.edu/fswiki/SuperSynth">FreeSurfer SuperSynth</a>, pinned source/model receipts retained. MRA vessels: vesselFM base model. All originals and prior v1 projects are preserved. Private local noncommercial adaptations only; do not redistribute derived imaging under CC BY-NC-ND4.0.</p>'''+doc[end:]
  gallery=[]
  for index,person in enumerate(sorted(set(r['person_id']for r in records))):
   variants=[r for r in records if r['person_id']==person and r['synthetic']]
   if not variants:continue
   r=variants[index%len(variants)];caption=html.escape(person+' · '+r['recipe']['placement_stratum'])
   gallery.append('<figure style="margin:0"><a href="'+html.escape((root/r['contact_sheet']).as_uri())+'"><img loading="lazy" alt="'+caption+' · MRI comparison and organic target surface" style="width:100%;border-radius:8px" src="'+html.escape((root/r['contact_sheet']).as_uri())+'"></a><figcaption>'+caption+'</figcaption></figure>')
  doc=doc.replace('<table>','<div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:20px;margin:32px 0">'+''.join(gallery)+'</div><table>',1)
  evidence=[]
  for relative,label in [('README.md','Local case guide'),('qc/all15-anatomy-source-overlays.png','All15 anatomy overlays'),('final-verification.json','135-project verification'),('desktop-qa/complete.json','Packaged-app verification')]:
   p=base/relative
   if p.is_file():evidence.append('<a href="'+html.escape(p.as_uri())+'">'+label+'</a>')
  doc=doc.replace('<div style="display:grid','<p>'+(' · '.join(evidence))+'</p><div style="display:grid',1)
 tmp=base/f'index.html.{os.getpid()}.partial';tmp.write_text(doc);tmp.replace(base/'index.html')
 if status['complete'] or not args.watch:print(json.dumps(status),flush=True);break
 time.sleep(20)

"""Fetch only pinned official SynthSeg inference source, v1 weights and license.

No imaging is read. The standalone CPU environment is installed separately, so
legacy TensorFlow requirements cannot replace the application's dependencies.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import urllib.request

PIN = '2a2aa3bbfccb83f8253a51ca8b329b9938a2646d'
BASE = f'https://raw.githubusercontent.com/BBillot/SynthSeg/{PIN}/'

def fetch(destination: Path):
    request = urllib.request.Request(f'https://api.github.com/repos/BBillot/SynthSeg/git/trees/{PIN}?recursive=1', headers={'User-Agent':'RessectionLab-model-qualification'})
    tree_bytes = urllib.request.urlopen(request, timeout=30).read()
    tree = json.loads(tree_bytes)
    if tree.get('truncated') or tree.get('sha') != PIN:
        raise ValueError('Pinned complete source tree required')
    rows = []
    selected = [item for item in tree['tree'] if item['type']=='blob' and
                (item['path'].endswith('.py') or item['path'].startswith('data/labels_classes_priors/') or
                 item['path'] in ('LICENSE.txt','models/synthseg_1.0.h5'))]
    if len(selected) != 66 or sum(item['size'] for item in selected) != 53966133:
        raise ValueError('Pinned source inventory changed')
    for item in selected:
        relative = item['path']
        path = destination/relative
        if path.exists():
            content = path.read_bytes()
        else:
            content = urllib.request.urlopen(BASE+relative, timeout=60).read()
        git_blob = hashlib.sha1(b'blob '+str(len(content)).encode()+b'\0'+content).hexdigest()
        if len(content)!=item['size'] or git_blob!=item['sha']:
            raise ValueError('Official pinned blob mismatch: '+relative)
        path.parent.mkdir(parents=True,exist_ok=True)
        if not path.exists():
            path.write_bytes(content)
        rows.append({'path':relative,'bytes':len(content),'git_blob_sha1':git_blob,
                     'sha256':hashlib.sha256(content).hexdigest(),'url':BASE+relative})
    record={'schema':'resectionlab-official-model-source-v1','model':'SynthSeg 1.0',
            'repository':'https://github.com/BBillot/SynthSeg','revision':PIN,'license':'Apache-2.0',
            'license_path':'LICENSE.txt','tree_sha256':hashlib.sha256(tree_bytes).hexdigest(),
            'files':rows,'scope':'Official v1 checkpoint and source, no clinical validity claim',
            'compatibility_note':'Upstream historical Python3.6/3.8 TensorFlow2.2 dependencies are not installed into the app; a separate Python3.9 Apple Silicon CPU compatibility trial is explicitly recorded.'}
    (destination/'source-receipt.json').write_text(json.dumps(record,indent=2)+'\n')
    return {'files':len(rows),'bytes':sum(row['bytes'] for row in rows),'revision':PIN,
            'weights_sha256':next(row['sha256'] for row in rows if row['path'].endswith('.h5'))}

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--destination',type=Path,required=True)
    args=parser.parse_args()
    print(json.dumps(fetch(args.destination)))

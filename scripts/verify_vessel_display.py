"""Measure source/filtered display and residual gaps without revising anatomy."""
import argparse
import json
from pathlib import Path
import time
import tempfile
import numpy as np
from scipy import ndimage as ndi
from scipy.spatial import cKDTree
from resectionlab.patient_anatomy import PatientImagingCollection, file_hash, world, resample
from resectionlab.anatomy_intracranial import intracranial_vessels
from resectionlab.desktop_bridge import BridgeSession, _Request


def verify(patient, output):
    c = PatientImagingCollection(patient)
    before = (c.root/'patient-anatomy.json').read_bytes()
    identity = c.identity()
    r = c.layer('vessels'); mask = c.mask(r); probability = c.array(r['probability']) if 'probability' in r else None
    source_hashes = {ref['path']: file_hash(c.path(ref['path']))
                     for row in c.document['series']+c.document['layers']
                     for ref in [row['data'], *([row['probability']] if 'probability' in row else [])]}
    strict, report = intracranial_vessels(c, r)
    filtered = c.mask(strict)
    labels, components = ndi.label(mask, np.ones((3,3,3)))
    report.update(sourceVoxels=int(mask.sum()), displayedStrictVoxels=int(filtered.sum()),
                  sourceComponents=components, probabilityMaskMismatch=int(np.count_nonzero((probability>=.5)!=mask)) if probability is not None else None)
    transfers = tempfile.TemporaryDirectory(prefix='rslab-vessel-data-')
    bridge = BridgeSession(Path(transfers.name))
    def call(op, args): return bridge.execute(op,args,_Request('vessel-check'),lambda *x:None)
    opened = call('anatomyOpen',{'path':str(c.root/'patient-anatomy.json')})
    start = time.monotonic()
    result = call('anatomyView',{'collectionId':opened['collectionId'],'snapshotId':identity,
                  'seriesId':r['source_series'],'layers':['vessels'],'vesselScope':'all','previewAlignment':True})
    layer = result['compartments'][0]
    shown = np.fromfile(layer['mask']['path'],dtype=np.uint8).reshape(result['shape'])
    expected = resample(mask.astype(np.uint8),np.array(r['affine']),tuple(result['shape']),np.array(result['affine']),order=0)
    assert np.array_equal(shown,expected) and layer['displayFilter'] is None
    report.update(fullSourceLoadSeconds=time.monotonic()-start, displayedFullVoxels=int(shown.sum()), fullSourceNativeVoxels=int(mask.sum()), transferredMaskMatchesSource=True)
    transfers.cleanup()
    report['residualCandidates'] = []
    graph_path = c.root/'vessel-geometry.npz'
    if graph_path.exists() and probability is not None:
        with np.load(graph_path) as graph:
            points = graph['points_ras_mm']; edges = graph['edges']
            endpoints = points[np.bincount(edges.ravel(), minlength=len(points))==1]
        vox = world(np.linalg.inv(np.array(r['affine'])),endpoints)
        ids = labels[tuple(np.rint(vox).astype(int).T)]
        pairs = cKDTree(endpoints).query_pairs(3.0)
        pairs = sorted((np.linalg.norm(endpoints[a]-endpoints[b]),a,b) for a,b in pairs if ids[a] and ids[b] and ids[a]!=ids[b])
        image = c.array(c.series(r['source_series'])['data'])
        used = set()
        for distance,a,b in pairs:
            if a in used or b in used: continue
            line = np.linspace(vox[a],vox[b],33)
            values = ndi.map_coordinates(probability,line.T,order=1)
            if values.min()>=.5: continue
            used.update([a,b])
            entry={'endpointsRASmm':endpoints[[a,b]].tolist(),'distanceMm':float(distance),
                   'minimumProbabilityAlongLine':float(values.min()),'componentIds':ids[[a,b]].tolist(),
                   'interpretation':'Separate source components with subthreshold probability between endpoints; not evidence that these vessels should connect.'}
            report['residualCandidates'].append(entry)
            if len(report['residualCandidates'])==4: break
        # Source/probability/mask crops are review evidence, not proposed connections.
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(len(report['residualCandidates']),3,figsize=(10,3*len(report['residualCandidates'])),squeeze=False)
        for i,entry in enumerate(report['residualCandidates']):
            pair = np.array(entry['endpointsRASmm']); center = pair.mean(0)
            u = pair[1]-pair[0]; u /= np.linalg.norm(u)
            v = np.cross(u,np.eye(3)[np.argmin(abs(u))]); v /= np.linalg.norm(v)
            yy,xx = np.indices((41,41)); sample = center+(xx[...,None]-20)*.35*u+(yy[...,None]-20)*.35*v
            for j,(data,title,a) in enumerate([(image,'Original T1CE',c.series(r['source_series'])['affine']),
                                              (probability,'Retained probability',r['affine']), (mask,'Source mask',r['affine'])]):
                grid = world(np.linalg.inv(np.array(a)),sample).reshape(-1,3).T
                pixels = ndi.map_coordinates(np.asarray(data,dtype=np.float32),grid,order=0 if j==2 else 1).reshape(41,41)
                axes[i,j].imshow(pixels,origin='lower',cmap='gray' if j==0 else 'magma',**({'vmin':0,'vmax':1} if j else {}))
                axes[i,j].scatter([20-entry['distanceMm']/.7,20+entry['distanceMm']/.7],[20,20],facecolors='none',edgecolors='cyan',s=65)
                axes[i,j].set_title(f'{title} · candidate {i+1}',fontsize=10);axes[i,j].axis('off')
        fig.tight_layout();fig.savefig(output/'residual-source-review.png',dpi=140,bbox_inches='tight');plt.close(fig)
    assert (c.root/'patient-anatomy.json').read_bytes()==before and c.identity()==identity
    assert all(file_hash(c.path(p))==h for p,h in source_hashes.items())
    report.update(manifestUnchanged=True,sourceArraysUnchanged=True,identityUnchanged=True,
                  snapshotId=identity,patientId=c.document['patient_id'])
    (output/'measurements.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='residualCandidates'},indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('patient',type=Path);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    verify(args.patient,args.output)

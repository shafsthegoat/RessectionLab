#!/usr/bin/env python3
"""Hash-bound source anatomy graph for the existing partial TopBrain CTA001.

No acquisition, mask repair, registration, hemodynamics or role reassignment.
The native label map supplies vessel class; unknown anatomy stays unknown.
"""
from pathlib import Path
import argparse
import json
import platform
import resource
import sys
import time

import nibabel as nib
import numpy as np
import scipy
import skimage

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from resectionlab.anatomy_geometry import vessel_graph
from resectionlab.simulator_qualification import file_sha256

PROTOCOL = ROOT/'manifests/experiments/native-vascular-graph-v4.json'


def bind_annotation_grid(image, annotation, reference_shape, reference_affine, *, association_proven):
    """An unset label unit can inherit only an explicitly associated exact grid.

    An independently declared contradictory unit is never overridden. This binds
    array indices to their named image, not two independently positioned scans.
    """
    if (not association_proven or image.shape != tuple(reference_shape) or annotation.shape != image.shape
            or not np.allclose(image.affine,reference_affine,rtol=0.,atol=1e-7)
            or not np.allclose(annotation.affine,reference_affine,rtol=0.,atol=1e-7)
            or image.header.get_xyzt_units()[0] != 'mm'
            or annotation.header.get_xyzt_units()[0] not in ('mm','unknown')
            or image.header['sform_code'] <= 0 or annotation.header['sform_code'] <= 0):
        raise ValueError('Original CTA/annotation units, named reference or physical grid mismatch')
    return {'referenceImageUnit':'mm','annotationHeaderUnit':annotation.header.get_xyzt_units()[0],
        'binding':'Explicit named source-image grid, exact source shape and affine, no resampling',
        'physicalUnit':'mm','annotationUnitCopiedFromReference':annotation.header.get_xyzt_units()[0] == 'unknown',
        'originalHeadersModified':False}


def qualify(study, output):
    protocol = json.loads(PROTOCOL.read_text()); study = study.resolve(); output = output.resolve()
    if not output.is_relative_to(ROOT/'testcases') or output.exists():
        raise ValueError('Choose a NEW output directory under testcases/')
    # Freeze scientific code and metadata identity before opening any arrays.
    for relative, expected in protocol['source_sha256'].items():
        if file_sha256(ROOT/relative) != expected: raise ValueError('Frozen vascular source changed: '+relative)
    manifest = study/'patient-anatomy.json'
    if file_sha256(manifest) != protocol['native_manifest_sha256']:
        raise ValueError('Exact frozen existing CTA001 development manifest required before payload access')
    meta = json.loads(manifest.read_text())
    if meta['patient_id'] != protocol['patient_id'] or len(meta['series']) != 1 or meta['series'][0]['modality'] != 'CTA':
        raise ValueError('No cross-patient or cross-modality vascular semantics')
    def asset(row):
        path = (study/row['path']).resolve()
        if not path.is_relative_to(study) or file_sha256(path) != row['sha256']:
            raise ValueError('Native source identity mismatch')
        return path
    series = meta['series'][0]; artery = next(r for r in meta['layers'] if r['id'] == 'vessels')
    venous = next(r for r in meta['layers'] if r['id'] == 'venous-vessels')
    original = artery['provenance']['source_labels_original']; labelmap = artery['provenance']['labelmap_original']
    if original['sha256'] != protocol['source_labels_sha256'] or labelmap['sha256'] != protocol['labelmap_sha256']:
        raise ValueError('Frozen CTA label source mismatch')
    paths = [asset(original),asset(labelmap),asset(artery['data']),asset(venous['data']),asset(series['original'])]
    before = {str(p):file_sha256(p) for p in paths}
    output.mkdir(parents=True); started = time.perf_counter()
    image = nib.load(paths[-1]); labels_image = nib.load(paths[0]); affine = np.array(series['affine'],float)
    units = bind_annotation_grid(image,labels_image,series['data']['shape'],affine,
        association_proven=all(row['patient_id'] == meta['patient_id'] and row['source_series'] == series['id'] for row in (artery,venous)))
    labels = np.asarray(labels_image.dataobj)
    if labels.dtype.kind not in 'ui' or not set(np.unique(labels)) <= {0,*range(1,41)}:
        raise ValueError('Unexpected original categorical source values')
    classes = protocol['classes']; class_by_value = {row['value']:row for row in classes}
    inventory = []
    for value,row in class_by_value.items():
        count = int(np.count_nonzero(labels == value))
        inventory.append({**row,'sourceVoxels':count,'present':count > 0})
    graphs = {}; reports = []
    for group,values,source_array in [('arterial',list(range(1,35)),paths[2]),('venous',[35,36,37,38,39,40],paths[3])]:
        expected = np.isin(labels,values); supplied = np.load(source_array,mmap_mode='r',allow_pickle=False)
        row = artery if group == 'arterial' else venous
        if (supplied.shape != labels.shape or not np.array_equal(supplied,expected)
                or not np.array_equal(np.array(row['affine'],float),affine)):
            raise ValueError('Native binary layer is not the exact source label union on the image grid')
        graph = vessel_graph(expected,affine)
        ids = graph['source_cell_ids']; graph['source_class_values'] = labels.ravel()[ids].copy()
        graph['source_kind_codes'] = np.array([{'artery':1,'vein':2,'sinus':3}[class_by_value[int(value)]['kind']]
            for value in graph['source_class_values']],np.uint8)
        destination = output/(group+'-source-graph.npz')
        np.savez_compressed(destination,**{k:v for k,v in graph.items() if isinstance(v,(np.ndarray,int))})
        graphs[group] = graph
        reports.append({'group':group,'sourceValues':values,'graph':str(destination),'graphSha256':file_sha256(destination),
            'nodes':len(ids),'edges':len(graph['edges']),
            **{name:int(graph[name]) for name in ('source_foreground_voxels','components','source_components6',
                'skeleton_components26','source_euler26','skeleton_euler26','unrepresented_source_components26',
                'unrepresented_source_components6','unrepresented_source6_voxels','unrepresented_source26_voxels',
                'source_boundary_foreground_voxels','edge_graph_components','edge_graph_cycle_rank',
                'edge_graph_euler','voxel_skeleton_cycle_rank26','voxel_skeleton_cavities6_background',
                'source_cycle_rank26','source_cavities6_background')},
            'voxelSkeletonBettiPreserved26':bool(graph['voxel_skeleton_betti_preserved26']),
            'edgeGraphMatchesVoxelBetti':bool(graph['edge_graph_matches_voxel_betti']),
            'diagonalEdgesWithoutLocalFaceSupport':int((~graph['edge_local_face_supported']).sum()),
            'edgesAcrossSource6Components':int((~graph['edge_same_source6_component']).sum()),
            'faceAdjacentEdges':int(graph['edge_face_adjacent'].sum()),
            'boundaryLimitedDistanceNodes':int(graph['radius_limited_by_image_extent'].sum()),
            'sourceIdentityPreserved':bool(expected.ravel()[ids].all() and len(np.unique(ids)) == len(ids)),
            'h1Representative':{'status':str(graph['h1_status']),
                'edges':len(graph['h1_edges']) if 'h1_edges' in graph else None,
                'cycleRank':int(graph['h1_betti1']) if 'h1_betti1' in graph else None,
                'removedAdjacencies':int(graph['h1_boundary_rank']) if 'h1_boundary_rank' in graph else None,
                'matchesSourceBetti':bool(graph['h1_matches_source_betti']),
                'meaning':'Subset of original edges, all source nodes retained; H0/H1 GF2 inclusion certificate. Not flow/length/homotopy preservation.'},
            'nodeKinds':{kind:int(np.count_nonzero(graph['source_kind_codes'] == code)) for kind,code in [('artery',1),('vein',2),('sinus',3)]}})
    # Exact original class contacts across positive-area voxel faces. These are
    # adjacency observations, not accepted physiological vessel junctions.
    contacts = {}
    for axis in range(3):
        low=[slice(None)]*3; high=low[:]; low[axis]=slice(None,-1); high[axis]=slice(1,None)
        left,right=labels[tuple(low)],labels[tuple(high)]
        selected=(left>0)&(right>0)&(left!=right)
        pairs=np.sort(np.column_stack((left[selected],right[selected])),axis=1)
        keys,counts=np.unique(pairs,axis=0,return_counts=True)
        for pair,count in zip(keys,counts):
            key=tuple(map(int,pair)); contacts[key]=contacts.get(key,0)+int(count)
    after = {str(p):file_sha256(p) for p in paths}
    if before != after: raise ValueError('Original source modified')
    report = {'schema':'resectionlab.native-vascular-graph-qualification.v1','patientId':meta['patient_id'],
        'sourceRole':protocol['source_role'],'sourcePage':meta['source_scope']['source_page'],
        'nativeManifestSha256':file_sha256(manifest),'protocolSha256':file_sha256(PROTOCOL),
        'sourceSha256':protocol['source_sha256'],'sourceFiles':before,'sourceFilesPreserved':True,
        'noCrossPatientFusion':True,'noNewRegistration':True,'sourceGridAgreement':True,'imagePixelsDecoded':False,
        'annotationGridUnits':units,
        'sourceLabelsDecoded':True,'imageShape':list(image.shape),'affineRasMm':affine.tolist(),
        'classInventory':inventory,'graphs':reports,'classFaceContacts':[
            {'sourceValues':list(key),'sourceNames':[class_by_value[v]['name'] for v in key],'faces':count}
            for key,count in sorted(contacts.items())],
        'missing':meta['missing'],'missingClassMeaning':'Unannotated or unobserved; absence is not established',
        'reviewStatus':'unreviewed annotation; derived source geometry only','flowModelAdmitted':False,
        'boundaryConditionsMeasured':False,'territoryOrClinicalInjuryInferred':False,'gliomaEvaluationSupported':False,
        'wallSeconds':time.perf_counter()-started,'peakRssBytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform == 'darwin' else 1024),
        'versions':{'python':platform.python_version(),'numpy':np.__version__,'scipy':scipy.__version__,'skimage':skimage.__version__}}
    if report['wallSeconds'] > protocol['max_wall_seconds'] or report['peakRssBytes'] > protocol['max_peak_rss_bytes']:
        raise ValueError('Native vascular graph process exceeded frozen resource envelope')
    (output/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    if sum(p.stat().st_size for p in output.iterdir() if p.is_file()) > protocol['max_output_bytes']:
        raise ValueError('Native vascular graph output exceeded frozen envelope')
    print(json.dumps({'patientId':report['patientId'],'wallSeconds':report['wallSeconds'],'peakRssBytes':report['peakRssBytes'],'graphs':reports}))


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('study',type=Path);p.add_argument('output',type=Path);args=p.parse_args()
    qualify(args.study,args.output)

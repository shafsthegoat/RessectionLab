"""One-case source-bound TopBrain CTA packaging; no model inference or registration."""
from pathlib import Path
import json,re,shutil,hashlib
import nibabel as nib,numpy as np
from scipy import ndimage as ndi
from resectionlab.patient_anatomy import PatientImagingCollection,file_hash,world
from resectionlab.anatomy_geometry import mesh
ROOT=Path(__file__).resolve().parent
OUT=ROOT/'native-cta';QC=ROOT.parents[1]/'artifacts/desktop-renderer/medivis-fidelity-v2/source-qc'
original=ROOT/'originals';ctpath=original/'topcow_ct_001_0000.nii.gz';labpath=original/'topcow_ct_001.nii.gz'
ct=nib.load(ctpath);lab=nib.load(labpath);hu=ct.get_fdata(dtype=np.float32);labels=np.asanyarray(lab.dataobj)
assert ct.shape==lab.shape and np.array_equal(ct.affine,lab.affine)
assert np.isfinite(hu).all() and np.array_equal(labels,labels.astype(np.uint8))
names={int(m[1]):m[2] for line in (original/'labelmap_topbrain_ct.txt').read_text().splitlines() if (m:=re.match(r'\s*(\d+)\s+.*"([^"]+)"',line))}
if (OUT/'patient-anatomy.json').exists():raise ValueError('Collection exists; do not silently replace')
c=PatientImagingCollection.create(OUT,'TOPBRAIN-2025-TopCoW-CTA-001')
series=c.add_series(ctpath,series_id='cta',modality='CTA',timepoint='source-session-unknown')
series['intensity_units']='Source CT scaled intensities; nominal HU, scanner calibration not independently verified'
series['provenance']={'source':'https://zenodo.org/records/16623496','original_shape':list(ct.shape),'source_preprocessing':'Published TopCoW braincase-cropped and LPS-reoriented source; source bytes/affine preserved; this is not an uncropped original DICOM acquisition.'}
c.preview_pyramid('cta',levels=(2,4,8))
source_label=c.original(labpath);labelmap=c.original(original/'labelmap_topbrain_ct.txt')
for filename in ['README.txt','License.txt']:c.original(original/filename)
common={'source':'https://zenodo.org/records/16623496','source_author':'TopBrain Challenge Organizers','source_title':'TopBrain Challenge Data Release, Batch1 July30 2025','source_labels_original':source_label,'labelmap_original':labelmap,'license':'Noncommercial with source attribution; commercial use requires prior permission from data owner USZ.','derivation':'Exact union of supplied categorical annotation values on its original native grid; no interpolation, gap filling, inference, registration, topology edits, or component deletion.','annotation_status':'Publisher anatomical annotations; unreviewed locally; absent classes are not evidence of absent anatomy.'}
arteries=(labels>0)&(labels<35);veins=labels>=35
for layer_id,mask,label,color,scope in [('vessels',arteries,'TopBrain source arterial labels · partial patient annotation','#e85e59','Published positive labels1–34 only; missing leftICA/M1/M2/M3 label values retained as absent, not repaired.'),('venous-vessels',veins,'TopBrain source cerebral venous labels','#7689e7','Published positive labels35–40; named deep veins and dural sinuses, not a full venous tree.')]:
 row=c.add_layer(mask.astype(np.uint8),lab.affine,layer_id=layer_id,source_series='cta',kind='artery' if layer_id=='vessels' else 'vein',role='critical',color=color,label=label,provenance={**common,'anatomical_scope':scope})
 row['coverage']='Publisher annotated vessels on cropped braincase CTA; incomplete peripheral branches; image and label share exact native grid.'
bone=(hu>=700)&(labels==0)
row=c.add_layer(bone.astype(np.uint8),ct.affine,layer_id='bone',source_series='cta',kind='bone',role='context',color='#d9d1bb',label='High-attenuation CT context · threshold estimate',provenance={'method':'Native CTA scaled intensity >=700, excluding only supplied positive vessel labels. No component deletion or morphological edits.','source_ct_sha256':file_hash(ctpath),'anatomical_scope':'High-density tissue proposal, includes skull and any unannotated contrast/calcification exceeding threshold; not a vetted skull segmentation.','threshold':700,'annotated_vessel_voxels_excluded':int(np.count_nonzero((hu>=700)&(labels>0)))})
row['coverage']='Cropped source field only; not full uncropped head/neck; high-density context proposal.'
inventory=[]
for value in np.unique(labels):
 if not value:continue
 mask=labels==value;where=np.argwhere(mask);points=world(lab.affine,np.array([where.min(0),where.max(0)]))
 inventory.append({'value':int(value),'name':names[int(value)],'voxels':int(mask.sum()),'category':'artery' if value<35 else 'vein-or-sinus','voxel_bbox_inclusive':[where.min(0).tolist(),where.max(0).tolist()],'ras_bbox_mm':np.sort(points,axis=0).tolist()})
c.document['source_label_inventory']=inventory
c.document['missing']=['Local expert annotation acceptance','Full head/neck DICOM field of view; published image is braincase crop','Unannotated arterial and venous branches','LeftICA/MCA classes missing in supplied label; no cause inferred','Reviewed skull segmentation','Brain surface/parcellation; not inferred from this CTA','Tumor, nerves, functional maps and tractography','Commercial-use permission from USZ']
c.document['source_scope']={'single_patient':True,'single_modality':True,'new_model_inference':False,'new_registration':False,'cross_patient_fusion':False,'license_commercial_permission_obtained':False,'source_page':'https://zenodo.org/records/16623496','source_case':'topcow_ct_001'}
c.save('One-case public CTA with exact same-patient source vessel annotations; unreviewed native display package')
qc={'schema':'topbrain-cta001-native-source-qc-v2','native_manifest':str(OUT/'patient-anatomy.json'),'native_manifest_sha256':file_hash(OUT/'patient-anatomy.json'),'snapshot_id':c.identity(),'source_ct':{'shape':list(ct.shape),'grid_spacing_mm':nib.affines.voxel_sizes(ct.affine).tolist(),'grid_extent_mm':(np.array(ct.shape)*nib.affines.voxel_sizes(ct.affine)).tolist(),'orientation':list(nib.aff2axcodes(ct.affine)),'nifti_units':ct.header.get_xyzt_units(),'sha256':file_hash(ctpath),'affine':ct.affine.tolist()},'source_labels':{'sha256':file_hash(labpath),'same_grid':True,'max_abs_affine_difference':float(np.max(np.abs(ct.affine-lab.affine))),'positive_voxels':int(np.count_nonzero(labels)),'components26':int(ndi.label(labels>0,structure=np.ones((3,3,3)))[1]),'present_classes':len(inventory),'inventory':inventory,'absent_class_values':[v for v in range(1,41) if v not in np.unique(labels)]},'new_model_inference':False,'new_registration':False,'source_byte_hashes_preserved':True,'package_hash_checks':{},'layers':{},'limitations':c.document['missing'],'acquisition_receipt':str(ROOT/'acquisition.json'),'license':'https://zenodo.org/records/16623496; noncommercial with source attribution; commercial use requires USZ permission'}
for row in c.document['series']:
 assert file_hash(c.path(row['original']['path']))==file_hash(ctpath)
 assert np.array_equal(c.array(row['data']),hu)
 for preview in row['preview_pyramid']:c.array(preview['data'])
for row in c.document['layers']:
 mask=c.mask(row);components,n=ndi.label(mask,structure=np.ones((3,3,3)));sizes=np.bincount(components.ravel())[1:];points=np.argwhere(mask);extent=np.ptp(world(lab.affine,np.array([points.min(0),points.max(0)])),axis=0)
 qc['layers'][row['id']]={'kind':row['kind'],'voxels':int(mask.sum()),'components26':int(n),'largest_component_fraction':float(sizes.max()/mask.sum()),'positive_bbox_extent_mm':extent.tolist(),'review_status':row['review_status']}
 print('LAYER',row['id'],qc['layers'][row['id']],flush=True)
QC.mkdir(parents=True,exist_ok=True)
(ROOT/'native-cta-qc.json').write_text(json.dumps(qc,indent=2)+'\n')
for path,name in [(ROOT/'acquisition.json','topbrain-cta001-acquisition.json'),(ROOT/'native-cta-qc.json','topbrain-cta001-native-qc.json'),(original/'License.txt','TOPBRAIN-LICENSE.txt'),(original/'README.txt','TOPBRAIN-README.txt'),(original/'labelmap_topbrain_ct.txt','TOPBRAIN-LABELMAP-CT.txt')]:shutil.copyfile(path,QC/name)
print('MANIFEST',OUT/'patient-anatomy.json',flush=True)
for row in c.document['layers']:
 print('MESH start',row['id'],flush=True);geo=mesh(c,row);qc['layers'][row['id']].update(mesh_vertices=len(geo['positions']),mesh_triangles=len(geo['indices']),maximum_display_displacement_mm=float(geo['max_displacement_mm']));print('MESH done',row['id'],qc['layers'][row['id']],flush=True)
(ROOT/'native-cta-qc.json').write_text(json.dumps(qc,indent=2)+'\n');shutil.copyfile(ROOT/'native-cta-qc.json',QC/'topbrain-cta001-native-qc.json')
print('DONE',c.identity(),flush=True)

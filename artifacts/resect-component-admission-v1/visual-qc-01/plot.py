#!/usr/bin/env python3
"""One fixed Case3 visualization; reads original encoded files, never repairs them."""
from pathlib import Path
import hashlib
import itertools
import json
import os
import signal
import time

START = time.monotonic()
ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
DECLARATION_SHA = '34f02291feb4cee8d0cfbba4bdbeaa5c45c750b3bd9619c7bc7b0479da2cc21e'


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        while chunk := stream.read(1024 * 1024): h.update(chunk)
    return h.hexdigest()


def timed_out(signum, frame):
    raise TimeoutError('Fixed visualization cooperative time allowance exceeded')


def require(value, message):
    if not value: raise ValueError(message)


def main():
    # Reserve this exact attempt before importing array/image libraries.
    declaration_path = OUT/'declaration.json'
    require(sha(declaration_path) == DECLARATION_SHA, 'Declaration changed')
    with (OUT/'started.json').open('x') as stream:
        json.dump({'declaration_sha256':DECLARATION_SHA,'pid':os.getpid(),'attempts':1,
                   'array_decodes_before_declaration':0},stream)
    receipt = {'schema':'resect-case3-cavity-visual-qc.v1','status':'failed','patient_group':'RESECT:Case3',
               'role':'TRAIN','declaration_sha256':DECLARATION_SHA,'payload_array_decodes':{'image':0,'mask':0},
               'clinical_or_expert_acceptance':False,'training_admitted':False,'complete_removed_tissue':False,
               'process_hard_rss_enforced':False}
    signal.signal(signal.SIGALRM,timed_out)
    signal.setitimer(signal.ITIMER_REAL,115)  # Reserve publication time inside the120s parent cap.
    try:
        import numpy as np
        import nibabel as nib
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from matplotlib.collections import LineCollection
        from matplotlib.lines import Line2D
        declaration=json.loads(declaration_path.read_text())
        for name,binding in declaration['inputs'].items():
            path=ROOT/binding['path']
            require(path.is_file() and not path.is_symlink(),f'Invalid input path: {name}')
            require(path.stat().st_size==binding['bytes'] and sha(path)==binding['sha256'],f'Changed input: {name}')
        qc=json.loads((ROOT/declaration['inputs']['saved_qc']['path']).read_text())
        require(qc['patient_group']=='RESECT:Case3' and qc['role']=='TRAIN' and qc['status']=='completed','Saved QC not complete for TRAIN Case3')
        saved=qc['structural_qc']
        require(saved['maximum_grid_corner_difference_mm']<=.01,'Saved grid tolerance not satisfied')
        images={name:nib.load(ROOT/declaration['inputs'][name]['path']) for name in ('image','mask')}
        shape=tuple(images['image'].shape)
        require(len(shape)==3 and shape==tuple(images['mask'].shape),'Original shape mismatch')
        decoded_bytes=int(np.prod(shape,dtype=np.int64))*4*2
        require(decoded_bytes<=declaration['limits']['decoded_total_bytes'],'Decoded float32 total exceeds512MiB')
        headers={}
        for index,name in enumerate(('image','mask')):
            obj=images[name]; hdr=obj.header
            require(tuple(saved['headers'][index]['shape'])==shape,'Shape differs from saved QC')
            require(hdr.get_xyzt_units()[0]=='mm','Original grid units must remain mm')
            require(np.array_equal(obj.affine,np.asarray(saved['headers'][index]['affine_ras_mm'])),'Selected affine differs from saved QC')
            qform,qcode=obj.get_qform(coded=True); sform,scode=obj.get_sform(coded=True)
            require(int(qcode)==saved['headers'][index]['qform_code'] and int(scode)==saved['headers'][index]['sform_code'],'Coded forms differ from saved QC')
            headers[name]={'shape':list(shape),'stored_dtype':str(hdr.get_data_dtype()),'units':hdr.get_xyzt_units()[0],
                'qform_code':int(qcode),'sform_code':int(scode),'qform':None if qform is None else qform.tolist(),
                'sform':None if sform is None else sform.tolist(),'selected_affine_ras_mm':obj.affine.tolist(),
                'effective_proxy_slope':float(obj.dataobj.slope),'effective_proxy_intercept':float(obj.dataobj.inter),
                'stored_pixdim':hdr['pixdim'].tolist()}
        corners=np.array(np.meshgrid(*[(0,size-1) for size in shape],indexing='ij')).reshape(3,-1).T
        delta=nib.affines.apply_affine(images['image'].affine,corners)-nib.affines.apply_affine(images['mask'].affine,corners)
        grid_difference=float(np.max(np.linalg.norm(delta,axis=1)))
        require(grid_difference<=.01 and grid_difference==saved['maximum_grid_corner_difference_mm'],'Original grid difference changed')
        # Exactly one full payload-array decode per original. Header reads are separate.
        arrays={}
        for name in ('image','mask'):
            receipt['payload_array_decodes'][name]+=1
            arrays[name]=images[name].get_fdata(dtype=np.float32,caching='unchanged')
        image,mask=arrays['image'],arrays['mask']
        require(np.isfinite(mask).all(),'Mask contains nonfinite values')
        labels=np.unique(mask)
        require(np.array_equal(labels,np.array([0.,1.],dtype=np.float32)),'Original mask is not unchanged binary0/1')
        finite=image[np.isfinite(image)]
        require(len(finite)>0,'No finite original image intensities')
        finite_count=int(len(finite))
        window=np.percentile(finite,[1,99],method='linear',overwrite_input=True)
        del finite
        require(np.isfinite(window).all() and window[0]<window[1],'Degenerate fixed percentile display window')
        positive=mask>0  # Positive support only, with original binary labels already verified; no repair.
        bounds=[]
        for axis in range(3):
            occupied=np.flatnonzero(np.any(positive,axis=tuple(i for i in range(3) if i!=axis)))
            require(len(occupied)>0,'Empty source-manual annotation')
            bounds.append([int(occupied[0]),int(occupied[-1])])
        indices=[(lo+hi)//2 for lo,hi in bounds]
        count=int(np.count_nonzero(positive));require(count==saved['annotated_cavity_voxels'],'Mask count differs from saved QC')
        affine=images['image'].affine
        spacing=np.linalg.norm(affine[:3,:3],axis=0)
        directions=affine[:3,:3]/spacing
        plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'figure.facecolor':'white','savefig.facecolor':'white'})
        fig=plt.figure(figsize=(15.0,9.6))
        fig.text(.04,.949,'Case 3 · During-resection ultrasound and source-manual cavity outline',fontsize=19,weight='bold',color='#172839')
        fig.text(.04,.912,'TRAIN case only · visual engineering inspection · three fixed native slices, full field of view',fontsize=11.5,color='#394d60')
        fig.text(.04,.881,'Slice indices are integer midpoints of the positive-mask bounding box; no slice search, registration or grid repair.',fontsize=10,color='#526170')
        planes=[];labels_axis=('i','j','k')
        for axis in range(3):
            h,v=[a for a in range(3) if a!=axis]
            pixels=np.take(image,indices[axis],axis=axis).T
            selected=np.take(positive,indices[axis],axis=axis).T
            require(selected.any(),'Fixed bbox-midpoint plane has no positive annotation; no alternative slice permitted')
            ax=fig.add_axes([.048+axis*.321,.335,.285,.475])
            extent=(-.5,shape[h]-.5,-.5,shape[v]-.5)
            ax.imshow(pixels,origin='lower',cmap='gray',vmin=float(window[0]),vmax=float(window[1]),
                      interpolation='none',extent=extent,aspect=spacing[v]/spacing[h])
            # Exact voxel-edge outline: no interpolated contour or mask resampling.
            yedge=np.diff(np.pad(selected.astype(np.int8),((1,1),(0,0))),axis=0)!=0
            xedge=np.diff(np.pad(selected.astype(np.int8),((0,0),(1,1))),axis=1)!=0
            segments=[]
            for y,x in np.argwhere(yedge): segments.append(((x-.5,y-.5),(x+.5,y-.5)))
            for y,x in np.argwhere(xedge): segments.append(((x-.5,y-.5),(x-.5,y+.5)))
            ax.add_collection(LineCollection(segments,colors='#f5a742',linewidths=.95))
            ax.set_xlim(extent[:2]);ax.set_ylim(extent[2:]);ax.set_title(f'Fixed {labels_axis[axis]} = {indices[axis]}  |  native ({labels_axis[h]}, {labels_axis[v]}) plane',fontsize=11,weight='bold',pad=12)
            ax.set_xlabel(f'{labels_axis[h]} index →  ({spacing[h]:.6f} mm / step)',fontsize=9)
            ax.set_ylabel(f'{labels_axis[v]} index ↑  ({spacing[v]:.6f} mm / step)',fontsize=9)
            ax.tick_params(labelsize=8)
            right=directions[:,h];up=directions[:,v]
            vector=lambda value:'('+', '.join(f'{x:+.3f}' for x in value)+')'
            fig.text(.05+axis*.321,.288,f'Right +{labels_axis[h]} in RAS+: {vector(right)}\nUp +{labels_axis[v]} in RAS+:    {vector(up)}',fontsize=9,color='#394d60',linespacing=1.6)
            planes.append({'fixed_axis':axis,'fixed_index':indices[axis],'horizontal_axis':h,'vertical_axis':v,
                           'origin':'lower','extent':list(extent),'horizontal_spacing_mm':float(spacing[h]),
                           'vertical_spacing_mm':float(spacing[v]),'right_unit_ras':right.tolist(),'up_unit_ras':up.tolist(),
                           'positive_mask_pixels':int(selected.sum()),'exact_voxel_outline_segments':len(segments)})
        fig.legend(handles=[Line2D([0],[0],color='#f5a742',lw=1.5,label='Original positive-mask voxel edges (same-index overlay)')],
                   loc='lower left',bbox_to_anchor=(.043,.238),frameon=False,fontsize=10)
        fig.text(.04,.211,f'Display-only intensity window: finite original P1 = {window[0]:.3f}, P99 = {window[1]:.3f}; zeros retained. Oblique planes are not anatomical axial/coronal/sagittal views.',fontsize=9.5,color='#394d60')
        fig.text(.04,.183,f'Raw headers differ: image qform=0 / sform=1; mask qform=1 / sform=0. Max original grid-corner difference = {grid_difference:.8f} mm (overlay limit 0.01 mm).',fontsize=9.5,color='#394d60')
        fig.text(.04,.155,'No reorientation, registration, volume interpolation or threshold repair. Physical aspect follows the original image affine; the small header mismatch is retained.',fontsize=9.5,color='#394d60')
        fig.text(.04,.117,'Source annotation: manual contours + morphological interpolation + source expert review. Visible cavity may omit blood-filled regions; not complete removed tissue.',fontsize=9.5,color='#172839')
        fig.text(.04,.089,'Rights: ultrasound CC BY 4.0; annotation CC BY-NC-SA 4.0 (noncommercial). This panel grants no clinical/expert acceptance or training admission.',fontsize=9.5,color='#172839')
        fig.text(.04,.052,'Source: RESECT ultrasound (10.11582/2017.00004) · resect-seg v1 annotation (10.17605/OSF.IO/JV8BK) · Selected slices do not establish whole-volume alignment.',fontsize=8.5,color='#526170')
        output=OUT/'fixed-native-cavity-panel.png'
        fig.savefig(output,dpi=180,metadata={'Software':'RessectionLab fixed Case3 visual QC'})
        plt.close(fig)
        for name,binding in declaration['inputs'].items():
            require(sha(ROOT/binding['path'])==binding['sha256'],f'Input changed after visualization: {name}')
        require(sha(declaration_path)==DECLARATION_SHA,'Declaration changed during visualization')
        receipt.update(status='rendered_visual_review_pending',elapsed_seconds=time.monotonic()-START,
            decoded_total_bytes=decoded_bytes,decoded_total_limit_bytes=declaration['limits']['decoded_total_bytes'],
            headers=headers,raw_affine_difference=(images['image'].affine-images['mask'].affine).tolist(),
            maximum_grid_corner_difference_mm=grid_difference,saved_qc_corner_difference_mm=saved['maximum_grid_corner_difference_mm'],
            shape=list(shape),finite_image_voxels=finite_count,image_nonfinite_voxels=int(image.size-finite_count),
            display_window_percentile_values=window.tolist(),display_percentiles=[1,99],
            positive_mask_bbox_inclusive=bounds,fixed_indices=indices,source_mask_labels=labels.tolist(),
            annotated_cavity_voxels=count,planes=planes,inputs_unchanged=True,
            versions={'numpy':np.__version__,'nibabel':nib.__version__,'matplotlib':matplotlib.__version__},
            plot_source_sha256=sha(Path(__file__)),output={'path':output.name,'sha256':sha(output),'bytes':output.stat().st_size},
            decode_semantics='One get_fdata(float32) payload-array decode per original; original NIfTI scaling only; header reads separate.',
            outline_semantics='Exact voxel edges; no interpolated contours. No unobserved cavity or removed-tissue inference.')
    except Exception as exc:
        receipt.update(status='failed',failure_type=type(exc).__name__,failure=str(exc),elapsed_seconds=time.monotonic()-START)
    finally:
        signal.setitimer(signal.ITIMER_REAL,0)
        (OUT/'provenance.json').write_text(json.dumps(receipt,indent=2,sort_keys=True,allow_nan=False)+'\n')
    print(json.dumps({k:receipt[k] for k in ['status','elapsed_seconds','payload_array_decodes']},indent=2))
    if receipt['status']=='failed':
        print(receipt.get('failure'));return 1
    return 0


if __name__=='__main__': raise SystemExit(main())

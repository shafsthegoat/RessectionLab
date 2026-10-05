#!/usr/bin/env python3
"""One source-bound, separately released RHUH fixed-plane display; no inference.

Default CLI is metadata-only. Tests use constructed arrays and byte streams.
Actual rendering requires a fresh output directory and exact root release.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
from pathlib import Path
import stat
import types

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DECODER_PATH = 'scripts/inspect_rhuh_single_image.py'
DECODER_SHA = '268d23d8560c963461f633fe2d66ef581b8bd47e74e6431b89c23d7058979a6f'
SOURCE = '/RHUH-GBM_nii_v1/RHUH-0001/0/RHUH-0001_0_t1.nii.gz'
ORIGINAL_SHA = 'b3b9fa69b87221062261b8b16fbddfdac2c678104b354ba371b069fd784e3625'
REPORT = {'path': 'outputs/rhuh-single-image-inspection-v1/run-first/stdout.json',
          'sha256': '7a0bdb595034c519967c7ce044f130d2119d0a361e550d5fd424ba03fbf3c9b1'}
AUDIT = {'path': 'artifacts/rhuh-single-image-inspection-v1/independent-result-audit.json',
         'sha256': '822f50da72df1377e58c43d7b3aabd75dc9393cd6fbc9769dce3d650161df07b'}
PARENT = {'path': 'artifacts/rhuh-single-image-inspection-v1/parent-summary.json',
          'sha256': 'a28dfcf3fd96c70cbf5c0e361dc3fda9a64b935ceba07ec22fae4a50c4c0f59e'}
TERMINAL = {'path': 'artifacts/rhuh-single-image-inspection-v1/root-terminal.json',
            'sha256': '8d9287aed3301dc8e93014b970b9dcfef0034fa541b0a285566fbe6316ca0040'}
SHAPE = (240, 240, 155)
INDICES = (120, 120, 77)
AFFINE = np.array([[-1.,0.,0.,0.],[0.,-1.,0.,239.],[0.,0.,1.,0.],[0.,0.,0.,1.]])
WINDOWS = ((-8.976038932800293, 10.492466926574707), (-3., 3.))
OPPOSITE = {'R':'L','L':'R','A':'P','P':'A','S':'I','I':'S'}


class Rejected(ValueError):
    pass


def need(value, code):
    if not value:
        raise Rejected(code)


def _decoder(root):
    """Compile the exact reviewed bytes, never a stale pyc or foreign package."""
    path = root / DECODER_PATH
    need(not any(p.is_symlink() for p in (path, *path.parents)), 'decoder_symlink')
    with path.open('rb') as stream:
        raw = stream.read(262145)
    need(len(raw) <= 262144 and hashlib.sha256(raw).hexdigest() == DECODER_SHA, 'decoder_hash_changed')
    module = types.ModuleType('frozen_rhuh_inspection_for_render')
    module.__file__ = str(path)
    exec(compile(raw, str(path), 'exec'), module.__dict__)
    return module


def plane_views(volume, affine, indices):
    """Return copied, unchanged native values with explicit displayed-axis maps.

    Horizontal/vertical voxel indices increase right/up. No flips, interpolation,
    cropping, intensity transforms or orientation-dependent reindexing occur.
    Axis-aligned affine columns are required for unambiguous cardinal labels.
    """
    array = np.asarray(volume)
    matrix = np.asarray(affine)
    need(array.ndim == 3 and array.dtype.kind in 'iuf', 'scalar_3d_required')
    need(matrix.shape == (4,4) and matrix.dtype.kind in 'iuf'
         and np.isfinite(matrix).all() and np.array_equal(matrix[3], [0,0,0,1]), 'invalid_affine')
    spacing = np.linalg.norm(matrix[:3,:3], axis=0)
    need(np.all(spacing > 0), 'singular_affine')
    unit = matrix[:3,:3] / spacing
    world_axes = np.argmax(np.abs(unit), axis=0)
    need(len(set(world_axes)) == 3, 'noncardinal_or_singular_affine')
    expected = np.zeros((3,3))
    for axis, world_axis in enumerate(world_axes):
        expected[world_axis,axis] = np.sign(unit[world_axis,axis])
    need(np.allclose(unit, expected, atol=1e-12, rtol=0), 'noncardinal_affine_not_supported')
    need(len(indices) == 3 and all(type(v) is int and 0 <= v < array.shape[i]
                                 for i,v in enumerate(indices)), 'invalid_plane_indices')
    positive = [('R','L'),('A','P'),('S','I')]
    codes = [positive[w][0 if unit[w,i] > 0 else 1] for i,w in enumerate(world_axes)]
    result = []
    for fixed, index in enumerate(indices):
        horizontal, vertical = [i for i in range(3) if i != fixed]
        values = np.array(np.take(array, index, axis=fixed).T, copy=True)
        values.flags.writeable = False
        result.append({'fixed_axis':fixed,'index':index,'horizontal_axis':horizontal,'vertical_axis':vertical,
                       'pixels':values,'left':OPPOSITE[codes[horizontal]],'right':codes[horizontal],
                       'bottom':OPPOSITE[codes[vertical]],'top':codes[vertical],
                       'horizontal_step_mm':float(spacing[horizontal]),'vertical_step_mm':float(spacing[vertical]),
                       'extent':(-0.5,array.shape[horizontal]-0.5,-0.5,array.shape[vertical]-0.5)})
    return result


def build_figure(volume, affine, indices=INDICES, windows=WINDOWS, *, analytical_fixture=False):
    """Scientific display only; explicit windowing does not alter source arrays."""
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    need(len(windows)==2 and all(len(w)==2 and all(math.isfinite(v) for v in w) and w[0]<w[1] for w in windows),
         'invalid_display_windows')
    views = plane_views(volume,affine,indices)
    fig = Figure(figsize=(13.2,8.6),layout='constrained')
    FigureCanvasAgg(fig)
    axes = fig.subplots(2,3)
    letters=('i','j','k')
    for row,window in enumerate(windows):
        for col,view in enumerate(views):
            ax=axes[row,col]
            im=ax.imshow(view['pixels'],origin='lower',cmap='gray',vmin=window[0],vmax=window[1],
                         interpolation='nearest',extent=view['extent'],
                         aspect=view['vertical_step_mm']/view['horizontal_step_mm'])
            fixed,h,v=view['fixed_axis'],view['horizontal_axis'],view['vertical_axis']
            ax.set_title(f'{letters[fixed]} = {view["index"]}  |  displayed axes ({letters[h]}, {letters[v]})',fontsize=11)
            ax.set_xlabel(f'{letters[h]} (voxel index; {view["horizontal_step_mm"]:g} mm / step)',fontsize=9)
            ax.set_ylabel(f'{letters[v]} (voxel index; {view["vertical_step_mm"]:g} mm / step)',fontsize=9)
            ax.tick_params(labelsize=8)
            for x,y,label in [(0.04,0.5,view['left']),(0.96,0.5,view['right']),
                              (0.5,0.04,view['bottom']),(0.5,0.96,view['top'])]:
                ax.text(x,y,label,transform=ax.transAxes,ha='center',va='center',fontsize=10,color='#ffd166',
                        bbox={'facecolor':'black','edgecolor':'none','alpha':0.65,'pad':1.5})
        bar=fig.colorbar(im,ax=list(axes[row,:]),shrink=0.85,pad=0.02)
        label='Saved full range' if row==0 else 'Fixed display window'
        bar.set_label(f'{label}: {window[0]:.5g} to {window[1]:.5g}\nStored intensity; physical units unverified',fontsize=9)
        bar.ax.tick_params(labelsize=8)
    title='Constructed orientation fixture' if analytical_fixture else 'RHUH-0001 / visit 0 / source-labeled T1'
    fig.suptitle(title+'\nFixed sampled planes · display only · no anatomical acceptance',fontsize=14)
    fig.supxlabel('Stored-index views; edges follow qform directions. Full field of view; no masks or anatomical overlays; no MRI volume resampling.\n'
                  'Window clipping affects display only. Template/scanner provenance and unsampled anatomy remain unverified.',fontsize=9)
    return fig,views


def metadata_preflight(root):
    """Read only saved provenance/inspection/source files, never image bytes."""
    decoder=_decoder(root)
    report=decoder._json(root,REPORT)
    audit=decoder._json(root,AUDIT)
    parent=decoder._json(root,PARENT)
    terminal=decoder._json(root,TERMINAL)
    need(audit.get('schema')=='resectionlab.rhuh-saved-inspection-independent-audit.v1'
         and audit.get('accepted_for_saved_result_consistency') is True,'saved_audit_not_accepted')
    need(parent.get('accepted') is True and parent.get('stdout_sha256')==REPORT['sha256']
         and parent.get('scientific_use_released') is False and parent.get('inspector_exit_code')==0,
         'saved_parent_join')
    need(terminal.get('accepted') is True and terminal.get('exit_code')==0
         and terminal.get('parent_summary_sha256')==PARENT['sha256'],'saved_terminal_join')
    request=decoder._json(root,report['bindings']['request'])
    decoder.preflight(root,request)
    need(report['bindings']['source']==SOURCE and request['source']==SOURCE
         and report['compressed_sha256']==ORIGINAL_SHA==request['payload']['sha256']
         and report['compressed_bytes']==request['payload']['measured_compressed_bytes'],'original_saved_identity')
    q=report['geometry']['qform'];s=report['geometry']['sform']
    need(report['binary_structure_valid'] is True and report['gzip_footer_and_single_member_verified'] is True
         and report['scientific_use_released'] is False and report['clinical_validation'] is False
         and tuple(report['shape'])==SHAPE and report['dtype']=='<f4'
         and report['geometry']['spatial_units']=='mm' and report['geometry']['zooms_in_source_units']==[1.,1.,1.]
         and q['code']==1 and q['finite'] is True and q['invertible'] is True
         and np.array_equal(q['affine_in_source_units'],AFFINE) and s['present'] is False and s['code']==0,
         'fixed_grid_inspection_mismatch')
    need(report['intensity']['scaled_finite_min']==WINDOWS[0][0]
         and report['intensity']['scaled_finite_max']==WINDOWS[0][1]
         and report['intensity']['scaled_nonfinite_voxels']==0
         and report['intensity']['effective_slope']==1. and report['intensity']['effective_intercept']==0.,
         'fixed_intensity_inspection_mismatch')
    return decoder,report,request


def _decode_snapshot(decoder, compressed, saved):
    """Reuse the reviewed inspector, then recover native values with its reader.

    Two bounded passes of one immutable compressed snapshot intentionally avoid
    changing the frozen inspector's public API or introducing another parser.
    """
    observed=decoder.inspect_stream(io.BytesIO(compressed))
    need(observed=={k:v for k,v in saved.items() if k!='bindings'},'reinspection_differs_from_saved_report')
    reader=decoder._GzipReader(io.BytesIO(compressed))
    reader.exact(observed['voxel_offset'])
    data=reader.exact(observed['declared_voxel_bytes'])
    reader.finish()
    need(reader.sha.hexdigest()==ORIGINAL_SHA,'decoded_snapshot_identity')
    array=np.frombuffer(data,dtype=observed['dtype']).reshape(SHAPE,order='F')
    need(not array.flags.writeable,'source_array_must_be_readonly')
    return array


def execute(root,release_binding):
    decoder,report,request=metadata_preflight(root)
    release=decoder._json(root,release_binding)
    fields={'schema','released','action','source','renderer_sha256','decoder_sha256','inspection_sha256',
            'inspection_audit_sha256','original_compressed_sha256','output_directory'}
    source_sha=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    need(set(release)==fields and release['schema']=='resectionlab.rhuh-fixed-plane-release.v1'
         and release['released'] is True and release['action']=='fixed_three_planes_two_windows_once'
         and release['source']==SOURCE and release['renderer_sha256']==source_sha
         and release['decoder_sha256']==DECODER_SHA and release['inspection_sha256']==REPORT['sha256']
         and release['inspection_audit_sha256']==AUDIT['sha256']
         and release['original_compressed_sha256']==ORIGINAL_SHA,'render_release_mismatch')
    output=decoder._path(root,release['output_directory'])
    rel=output.relative_to(root)
    need(rel.parts[:2]==('outputs','rhuh-fixed-plane-qc-v1') and len(rel.parts)==3
         and rel.name.startswith('run-'),'output_directory_scope')
    need(not output.exists(),'fresh_output_required')
    output.parent.mkdir(parents=True,exist_ok=True)
    output.mkdir()  # Atomic one-attempt reservation, never reuse on failure.
    receipt={'schema':'resectionlab.rhuh-fixed-plane-display.v1','status':'failed','accepted':False,
             'source':SOURCE,'original_compressed_sha256':ORIGINAL_SHA,'inspection':REPORT,'inspection_audit':AUDIT,
             'release':release_binding,'renderer_sha256':source_sha,'decoder_sha256':DECODER_SHA,
             'indices':list(INDICES),'windows':[list(w) for w in WINDOWS],
             'scientific_use_released':False,'anatomical_validation':False}
    try:
        original=decoder._path(root,request['payload']['path'])
        with os.fdopen(os.open(original,os.O_RDONLY|os.O_NOFOLLOW),'rb') as stream:
            before=os.fstat(stream.fileno())
            need(stat.S_ISREG(before.st_mode) and before.st_size==request['payload']['measured_compressed_bytes']
                 and before.st_size<=decoder.MAX_COMPRESSED,'original_type_or_size')
            compressed=stream.read(decoder.MAX_COMPRESSED+1)
            after=os.fstat(stream.fileno())
        identity=lambda st:(st.st_dev,st.st_ino,st.st_size,st.st_mtime_ns,st.st_ctime_ns)
        need(identity(before)==identity(after) and len(compressed)==before.st_size,'original_changed_during_read')
        need(hashlib.sha256(compressed).hexdigest()==ORIGINAL_SHA
             and hashlib.md5(compressed,usedforsecurity=False).hexdigest()==decoder.PUBLISHED_TOKEN,'original_compressed_digest')
        array=_decode_snapshot(decoder,compressed,report)
        import matplotlib
        with matplotlib.rc_context({'font.family':'DejaVu Sans','svg.hashsalt':'rhuh-fixed-planes-v1'}):
            fig,views=build_figure(array,AFFINE,INDICES,WINDOWS)
            fig.savefig(output/'fixed-planes.png',dpi=180,metadata={'Software':'RessectionLab fixed-plane display'})
            fig.savefig(output/'fixed-planes.svg',metadata={'Date':None,'Creator':'RessectionLab fixed-plane display'})
            fig.clear()
        need(not original.is_symlink() and identity(original.stat())==identity(before),'original_changed_after_render')
        need(hashlib.sha256(Path(__file__).read_bytes()).hexdigest()==source_sha,'renderer_changed_during_run')
        _decoder(root)  # Recheck frozen decoder source after work.
        receipt.update(status='fixed_planes_rendered_review_required',accepted=True,
                       shape=list(SHAPE),affine_ras_mm=AFFINE.tolist(),numpy_version=np.__version__,
                       matplotlib_version=matplotlib.__version__,nibabel_version=decoder.nib.__version__,decoded_snapshot_passes=2,
                       planes=[{k:(list(v) if isinstance(v,tuple) else v) for k,v in plane.items() if k!='pixels'} for plane in views],
                       artifacts={name:{'bytes':(output/name).stat().st_size,
                                        'sha256':hashlib.sha256((output/name).read_bytes()).hexdigest()}
                                  for name in ('fixed-planes.png','fixed-planes.svg')})
    except Exception as exc:
        receipt['failure']=str(exc) if isinstance(exc,(Rejected,decoder.Rejected)) else 'render_failed_'+type(exc).__name__
    (output/'receipt.json').write_text(json.dumps(receipt,indent=2,allow_nan=False)+'\n')
    return receipt


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute',action='store_true')
    parser.add_argument('--release')
    parser.add_argument('--release-sha256')
    args=parser.parse_args(argv)
    try:
        if args.execute:
            need(args.release is not None and args.release_sha256 is not None,'separate_render_release_required')
            result=execute(ROOT,{'path':args.release,'sha256':args.release_sha256})
        else:
            need(args.release is None and args.release_sha256 is None,'release_only_with_execution')
            metadata_preflight(ROOT)
            result={'status':'metadata_only_no_patient_decode','accepted':False,'scientific_use_released':False}
        print(json.dumps(result,sort_keys=True,allow_nan=False))
        return 0 if not args.execute or result['accepted'] else 2
    except Exception as exc:
        print(json.dumps({'status':'rejected_no_render','failure':str(exc) if isinstance(exc,Rejected)
                          else 'render_failed_'+type(exc).__name__}))
        return 2


if __name__=='__main__':
    raise SystemExit(main())

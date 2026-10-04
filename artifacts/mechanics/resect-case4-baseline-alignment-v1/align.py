"""Frozen baseline-only diagnostic; requires a matching explicit root release."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
SNAPSHOT = ROOT/'build/resect-source-partition-6766a2a/src'
DATA = ROOT/'data/mechanics/resect-case4-v1/RESECT/NIFTI/Case4'


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        while block := stream.read(1024**2):
            h.update(block)
    return h.hexdigest()


def save(name, value):
    with (HERE/name).open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def apply(points, matrix):
    return points @ matrix[:3,:3].T + matrix[:3,3]


def rigid(source, destination):
    import numpy as np
    x, y = np.asarray(source, dtype=float), np.asarray(destination, dtype=float)
    if x.shape != y.shape or x.ndim != 2 or x.shape[1] != 3 or len(x) < 3:
        raise ValueError('Rigid fit needs matching 3D point rows')
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError('Nonfinite baseline points')
    xc, yc = x-x.mean(axis=0), y-y.mean(axis=0)
    for centered in (xc, yc):
        s = np.linalg.svd(centered, compute_uv=False)
        if s[0] <= 0 or s[1]/s[0] <= 1e-6:
            raise ValueError('Baseline rigid fit is geometrically degenerate')
    u, _, vh = np.linalg.svd(xc.T @ yc)
    correction = np.diag([1.,1.,np.linalg.det(vh.T @ u.T)])
    rotation = vh.T @ correction @ u.T
    if not np.allclose(rotation.T@rotation, np.eye(3), atol=1e-10, rtol=0) or abs(np.linalg.det(rotation)-1) > 1e-10:
        raise ValueError('Invalid proper rotation')
    matrix = np.eye(4)
    matrix[:3,:3] = rotation
    matrix[:3,3] = y.mean(axis=0)-rotation@x.mean(axis=0)
    return matrix


def errors(source, destination):
    import numpy as np
    if len(source) < 4:
        raise ValueError('At least four baseline pairs required')
    matrix = rigid(source, destination)
    raw = np.linalg.norm(source-destination,axis=1)
    fit = np.linalg.norm(apply(source,matrix)-destination,axis=1)
    loo = []
    for i in range(len(source)):
        keep = np.arange(len(source)) != i
        fold = rigid(source[keep],destination[keep])
        loo.append(float(np.linalg.norm(apply(source[i:i+1],fold)[0]-destination[i])))
    if np.sqrt(np.mean(fit**2)) > np.sqrt(np.mean(raw**2))+1e-8:
        raise ValueError('Rigid fit violates least-squares identity control')
    return matrix, raw, fit, np.asarray(loo)


def summarize(values):
    import numpy as np
    return {'rms_mm':float(np.sqrt(np.mean(values**2))), 'mean_mm':float(np.mean(values)),
            'median_mm':float(np.median(values)), 'maximum_mm':float(np.max(values))}


def sample_plane(array, source_affine, base_affine, base_shape, axis, index, world_map=None):
    import numpy as np
    from scipy.ndimage import map_coordinates
    other = [d for d in range(3) if d != axis]
    a,b = np.meshgrid(np.arange(base_shape[other[0]]),np.arange(base_shape[other[1]]),indexing='ij')
    grid = np.empty(a.shape+(3,),dtype=float)
    grid[...,axis],grid[...,other[0]],grid[...,other[1]] = index,a,b
    world = apply(grid.reshape(-1,3),base_affine)
    if world_map is not None:
        world = apply(world,world_map)
    coords = apply(world,np.linalg.inv(source_affine))
    values = map_coordinates(array,coords.T,order=1,mode='constant',cval=np.nan,prefilter=False)
    return values.reshape(a.shape)


def authenticated_declaration(declaration=None):
    """Authenticate the fixed release controls before following any supplied path."""
    release = json.loads((HERE/'root-release.json').read_text())
    pinned = {HERE/'declaration.json':release['declaration_sha256'],
              HERE/'coordinate-justification.json':release['coordinate_justification_sha256'],
              Path(__file__).resolve():release['script_sha256']}
    for path, digest in pinned.items():
        if sha(path) != digest:
            raise RuntimeError('PINNED_CONTROL_CHANGED')
    payload = (HERE/'declaration.json').read_bytes()
    if hashlib.sha256(payload).hexdigest() != release['declaration_sha256']:
        raise RuntimeError('PINNED_CONTROL_CHANGED')
    trusted = json.loads(payload)
    if declaration is not None and declaration != trusted:
        raise RuntimeError('DECLARATION_ARGUMENT_MISMATCH')
    return trusted, pinned


def make_patient_guard(allowed_paths):
    """Deny undeclared data reads, symlink aliases and every data-file write."""
    protected = Path(os.path.abspath(ROOT/'data'))
    protected_resolved = protected.resolve()
    allowed = {Path(os.path.abspath(path)) for path in allowed_paths}
    if any(path != path.resolve() or not path.is_relative_to(protected) for path in allowed):
        raise RuntimeError('INVALID_ALLOWED_PATIENT_PATH')
    write_flags = os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND
    def guard(event, args):
        if event != 'open' or not isinstance(args[0], (str, bytes, os.PathLike)):
            return
        lexical = Path(os.path.abspath(os.fsdecode(args[0])))
        resolved = lexical.resolve()
        if not (lexical.is_relative_to(protected) or resolved.is_relative_to(protected_resolved)):
            return
        mode = args[1] if len(args) > 1 else None
        flags = args[2] if len(args) > 2 else 0
        if (lexical != resolved or lexical not in allowed
            or isinstance(mode, str) and any(char in mode for char in 'wax+')
            or isinstance(flags, int) and flags & write_flags):
            raise RuntimeError('UNAUTHORIZED_PATIENT_ACCESS')
    return guard


def verify(declaration):
    declaration, pinned = authenticated_declaration(declaration)
    # Authenticate each referenced control before parsing any paths it contains.
    header = ROOT/declaration['baseline_header_receipt']
    snapshot_path = ROOT/declaration['helper_snapshot_manifest']
    for path, digest in ((header, declaration['baseline_header_receipt_sha256']),
                         (snapshot_path, declaration['helper_snapshot_manifest_sha256'])):
        if sha(path) != digest:
            raise RuntimeError('PINNED_SOURCE_OR_INPUT_CHANGED')
        pinned[path] = digest
    snapshot_payload = snapshot_path.read_bytes()
    if hashlib.sha256(snapshot_payload).hexdigest() != declaration['helper_snapshot_manifest_sha256']:
        raise RuntimeError('PINNED_SOURCE_OR_INPUT_CHANGED')
    snapshot = json.loads(snapshot_payload)
    for item in snapshot['files']:
        pinned[ROOT/item['path']] = item['sha256']
    for item in declaration['files']:
        pinned[ROOT/item['local_path']] = item['sha256']
    result = {str(path.relative_to(ROOT)):sha(path) for path in pinned}
    if any(result[str(path.relative_to(ROOT))] != digest for path,digest in pinned.items()):
        raise RuntimeError('PINNED_SOURCE_OR_INPUT_CHANGED')
    for item in declaration['files']:
        path = ROOT/item['local_path']
        md5 = hashlib.md5()
        with path.open('rb') as stream:
            while block := stream.read(1024**2):
                md5.update(block)
        if path.stat().st_size != item['bytes'] or md5.hexdigest() != item['md5']:
            raise RuntimeError('PROVIDER_IDENTITY_CHANGED')
    return result


def run_child():
    start = time.monotonic()
    declaration, _ = authenticated_declaration()
    allowed = {ROOT/item['local_path'] for item in declaration['files']}
    sys.addaudithook(make_patient_guard(allowed))
    before = verify(declaration)
    sys.path.insert(0,str(SNAPSHOT))
    import numpy as np
    import nibabel as nib
    from resectionlab import mechanics_landmarks as lm
    from resectionlab import core
    for module in (lm,core):
        if not Path(module.__file__).resolve().is_relative_to(SNAPSHOT):
            raise RuntimeError('MUTABLE_PACKAGE_IMPORT')
    def forbidden(*args,**kwargs):
        raise RuntimeError('DISPLACEMENT_MEASUREMENT_ACCESS_PROHIBITED')
    lm.build_forward_landmarks = forbidden
    lm.reveal_validation_landmarks = forbidden
    records = {Path(item['source_path']).name:item for item in declaration['files']}
    known = json.loads((ROOT/declaration['baseline_header_receipt']).read_text())
    headers = {Path(item['source_path']).name:item['header'] for item in known['files']}
    images, arrays = {},{}
    for name in ('Case4-T1.nii.gz','Case4-FLAIR.nii.gz','Case4-US-before.nii.gz'):
        image = nib.load(ROOT/records[name]['local_path'])
        header = headers[name]
        if (list(image.shape) != header['shape'] or image.header.get_xyzt_units()[0] != 'mm'
            or not np.array_equal(image.affine,np.asarray(header['selected_affine']))):
            raise RuntimeError('BASELINE_HEADER_CHANGED')
        images[name] = image
        arrays[name] = image.get_fdata(dtype=np.float32,caching='unchanged')
        if not np.isfinite(arrays[name]).all():
            raise RuntimeError('NONFINITE_BASELINE_IMAGE')
    qc = lm.LandmarkFrameQC(records['Case4-FLAIR.nii.gz']['sha256'],
        records['Case4-US-before.nii.gz']['sha256'],np.eye(4),np.eye(4),
        sha(HERE/'coordinate-justification.json'),'Creator shared NIfTI world; official RAS millimetres, anatomy alignment unaccepted')
    binding = lm.LandmarkPairBinding('RESECT:Case4',lm.REGISTRATION_ROLE,
        records['Case4-MRI-beforeUS.tag']['sha256'],qc.source_image_sha256,qc.destination_image_sha256,qc)
    registration = lm.read_registration_landmarks((ROOT/records['Case4-MRI-beforeUS.tag']['local_path']).read_bytes(),binding)
    x,y = registration.source_ras_mm,registration.observed_ras_mm
    bounds={}
    for label,points,name in (('FLAIR_reference',x,'Case4-FLAIR.nii.gz'),('beforeUS_reference',y,'Case4-US-before.nii.gz'),('MRI_reference_in_T1',x,'Case4-T1.nii.gz')):
        image=images[name]; v=apply(points,np.linalg.inv(image.affine))
        inside=np.all((v>=-.5)&(v<=np.array(image.shape)-.5),axis=1)
        bounds[label]={'inside_count':int(inside.sum()),'outside_row_ids':(np.where(~inside)[0]+1).tolist(),'voxel_coordinates':v.tolist()}
        if label != 'MRI_reference_in_T1' and not inside.all():
            raise RuntimeError('BASELINE_REFERENCE_POINT_OUTSIDE_OWN_IMAGE')
    matrix,raw,fit,loo=errors(x,y)
    summaries={key:summarize(value) for key,value in (('raw',raw),('fit',fit),('leave_one_out',loo))}
    planes=render(images,arrays,x,y,matrix)
    after=verify(declaration)
    if before != after:
        raise RuntimeError('INPUTS_CHANGED_DURING_BASELINE_QC')
    save('alignment-receipt.json',{
        'status':'baseline_diagnostic_complete_anatomical_review_pending','N':len(x),
        'source_hashes_before':before,'source_hashes_after':after,
        'coordinate_justification_sha256':sha(HERE/'coordinate-justification.json'),
        'source_world_to_RAS':np.eye(4).tolist(),'destination_world_to_RAS':np.eye(4).tolist(),
        'pair_role':lm.REGISTRATION_ROLE,'pair_binding_hash':binding.audit_hash,
        'row_ids':list(range(1,len(x)+1)),'FLAIR_reference_RAS_mm':x.tolist(),'beforeUS_reference_RAS_mm':y.tolist(),
        'FLAIR_RAS_to_beforeUS_RAS_mm':matrix.tolist(),'bounds':bounds,
        'raw_distances_mm':raw.tolist(),'fit_residuals_mm':fit.tolist(),'leave_one_out_residuals_mm':loo.tolist(),
        'summaries':summaries,'LOO_RMS_worse_than_raw':summaries['leave_one_out']['rms_mm']>summaries['raw']['rms_mm'],
        'outliers_removed':False,'alternative_fits_selected':False,'clinical_accuracy_validated':False,
        'brain_domain_available':False,'anatomical_alignment_accepted':False,
        'duringUS_or_before_during_tag_opened':False,'B_or_V_motion_accessed':False,
        'native_affines_preserved':True,'resampled_volumes_saved':False,'plots':planes,
        'runtime':{'python':sys.version,'numpy':np.__version__,'nibabel':nib.__version__,'helper_path':str(Path(lm.__file__).resolve())},
        'elapsed_seconds':time.monotonic()-start,'child_peak_RSS_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss})
    print(json.dumps({'N':len(x),'summaries':summaries}))


def render(images,arrays,x,y,matrix):
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    norms={}
    for name,array in arrays.items():
        lo,hi=np.percentile(array[::3,::3,::3],[1,99])
        if hi <= lo:
            raise RuntimeError('BASELINE_DISPLAY_CONTRAST_DEGENERATE')
        norms[name]=(float(lo),float(hi))
    def scale(a,name):
        lo,hi=norms[name]
        return np.clip((a-lo)/(hi-lo),0,1)
    outputs=[]
    for group,base_name,points in (('MRI-native','Case4-FLAIR.nii.gz',x),('US-native','Case4-US-before.nii.gz',y)):
        base=images[base_name]
        center=np.rint(np.median(apply(points,np.linalg.inv(base.affine)),axis=0)).astype(int)
        for axis in range(3):
            index=int(np.clip(center[axis],0,base.shape[axis]-1))
            base_values=np.take(arrays[base_name],index,axis=axis)
            if group=='MRI-native':
                other='Case4-T1.nii.gz'
                sampled=sample_plane(arrays[other],images[other].affine,base.affine,base.shape,axis,index)
                first=scale(base_values,base_name); second=scale(sampled,other)
                rgb=np.stack([first,np.nan_to_num(second),first],axis=-1)
                panels=[(first,'Native FLAIR'),(second,'T1 sampled using original affines'),(rgb,'Magenta FLAIR / green T1')]
            else:
                other='Case4-FLAIR.nii.gz'
                raw=sample_plane(arrays[other],images[other].affine,base.affine,base.shape,axis,index)
                fitted=sample_plane(arrays[other],images[other].affine,base.affine,base.shape,axis,index,np.linalg.inv(matrix))
                first=scale(base_values,base_name); second=scale(fitted,other)
                rgb=np.stack([np.nan_to_num(second),first,np.nan_to_num(second)],axis=-1)
                panels=[(first,'Native before-US'),(scale(raw,other),'FLAIR before rigid fit'),(second,'FLAIR after all-point fit'),(rgb,'Green US / magenta fitted FLAIR')]
            fig,axes=plt.subplots(1,len(panels),figsize=(4*len(panels),4),layout='constrained')
            for ax,(values,title) in zip(axes,panels):
                display=np.swapaxes(values,0,1)
                ax.imshow(display,origin='lower',cmap='gray',vmin=0,vmax=1,interpolation='nearest')
                ax.set_title(title,fontsize=10); ax.set_axis_off()
            fig.suptitle(f'RESECT Case4 baseline QC | {group} axis {axis}, index {index} | no anatomy approval',fontsize=11)
            path=HERE/f'{group}-axis{axis}.png'
            if path.exists():
                raise RuntimeError('IMMUTABLE_PLOT_ALREADY_EXISTS')
            fig.savefig(path,dpi=125); plt.close(fig)
            outputs.append({'path':path.name,'sha256':sha(path),'base_image':base_name,'axis':axis,'index':index,'display_limits':norms})
    return outputs


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--child',action='store_true'); parser.add_argument('--execute',action='store_true')
    args=parser.parse_args()
    if args.child:
        try:
            run_child()
        except Exception as error:
            save('failure.json',{'status':'failed_preserve_no_retry','type':type(error).__name__,'reason':str(error)})
            raise SystemExit(2)
        return
    if not args.execute or not (HERE/'root-release.json').exists():
        raise SystemExit('Requires --execute and matching explicit root-release.json; no patient access occurred')
    declaration, _ = authenticated_declaration()
    sys.addaudithook(make_patient_guard({ROOT/item['local_path'] for item in declaration['files']}))
    verify(declaration)
    if any((HERE/name).exists() for name in ('alignment-receipt.json','resource-receipt.json','failure.json','run.log')):
        raise SystemExit('Existing attempt is immutable; no automatic retry')
    start,peak,failure=time.monotonic(),0,None
    env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',VECLIB_MAXIMUM_THREADS='1')
    with (HERE/'run.log').open('x') as log:
        process=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),'--child'],stdout=log,stderr=log,env=env)
        try:
            while process.poll() is None:
                try:
                    r=subprocess.run(['ps','-o','rss=','-p',f'{os.getpid()},{process.pid}'],capture_output=True,text=True,timeout=2,check=False)
                    samples=[int(line) for line in r.stdout.splitlines() if line.strip()]
                    if r.returncode != 0 or not samples or any(value <= 0 for value in samples):
                        raise RuntimeError('RSS_PROBE_INVALID')
                    # A child may exit while ps samples; two rows are required while it is alive.
                    if len(samples) != 2 and process.poll() is None:
                        raise RuntimeError('RSS_PROBE_INCOMPLETE')
                    rss=sum(samples)*1024; peak=max(peak,rss)
                except (OSError, ValueError, RuntimeError, subprocess.SubprocessError):
                    failure='RSS_MONITOR_FAILED'; process.kill(); break
                if time.monotonic()-start > declaration['budget']['wall_seconds'] or rss > declaration['budget']['RSS_bytes']:
                    failure='WALL_OR_RSS_LIMIT'; process.kill(); break
                time.sleep(.025)
        except BaseException:
            failure='SUPERVISOR_INTERRUPTED'; process.kill()
        finally:
            if process.poll() is None:
                process.kill()
            code=process.wait(timeout=2)
    save('resource-receipt.json',{'status':'passed' if code==0 and failure is None else 'failed','exit_code':code,'failure':failure,
        'elapsed_seconds':time.monotonic()-start,'maximum_sampled_parent_child_RSS_bytes':peak,'budget':declaration['budget'],
        'sampling_caveat':'RSS monitoring is sampled, not continuous','script_sha256':sha(Path(__file__)),
        'result_sha256':sha(HERE/'alignment-receipt.json') if (HERE/'alignment-receipt.json').exists() else None,'log_sha256':sha(HERE/'run.log')})
    print(json.dumps({'exit_code':code,'failure':failure,'elapsed_seconds':time.monotonic()-start}))
    raise SystemExit(code if code else int(failure is not None))


if __name__=='__main__':
    main()

"""Generated header/import compatibility only; original archives denied."""
from pathlib import Path
import gzip, importlib.util, json, sys, tempfile
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
def guard(event,args):
    if event=='open' and str(args[0]).startswith(str(ROOT/'data/acquisition')):
        raise AssertionError('original acquisition open forbidden')
sys.addaudithook(guard)
spec=importlib.util.spec_from_file_location('bound_ixi_adapter',HERE/'launch_v2.py')
L=importlib.util.module_from_spec(spec);spec.loader.exec_module(L)
sys.addaudithook(L.deny_worker_process_and_network)
reader=L.module(L.READER,'generated_fixed_reader')
import numpy as np
import nibabel as nib
outcomes=[]
with tempfile.TemporaryDirectory(prefix='generated-header-',dir=HERE) as temp:
    for cls in (nib.Nifti1Header,nib.Nifti2Header):
        h=cls();h.set_data_shape((2,3,4));h.set_data_dtype(np.uint8);h.set_xyzt_units('mm');h.set_qform(np.eye(4),code=1);h.set_sform(np.eye(4),code=1)
        h['vox_offset']=352 if h.sizeof_hdr==348 else 544
        p=Path(temp)/'generated.nii.gz';p.write_bytes(gzip.compress(h.binaryblock+b'\0'*4+b'\1'*24,mtime=0))
        result=reader.header_only(p)
        assert result['decoded_header_bytes']==int(h['vox_offset']) and result['voxel_arrays_materialized']==0
        assert result['library_versions']=={'numpy':'2.5.3','nibabel':'5.4.2'}
        outcomes.append({'version':result['nifti_version'],'header_bytes':result['decoded_header_bytes'],'array_reads':0})
origins=sys._remind_header_verify_origins()
print(json.dumps({'status':'PASS','generated_headers':outcomes,'actual_archive_opens':0,'guard_active':True,'verified_loaded_module_count':len(origins)}))

#!/usr/bin/env python3
"""Small explicit failure fixtures, outside the accepted patient corpus."""
from pathlib import Path
from zipfile import ZipFile, ZIP_STORED
import json
from io import BytesIO
import hashlib

import numpy as np

from resectionlab.core import CaseData, SourceRef
from resectionlab.data_root import checked_root, atomic_json, sha256_file
from resectionlab.imaging import save_case
from resectionlab.workspace_bundle import load_workspace


def run(root=None):
    root=checked_root(root,required_bytes=1024**2)
    directory=root/'quarantine/negative-fixtures-v1';directory.mkdir(exist_ok=True)
    valid=directory/'generated-format-control.ressectionlab'
    if not valid.exists():
        case=CaseData('generated-format-control',np.arange(64,dtype=np.float32).reshape(4,4,4),{},np.eye(4),
            (SourceRef('format-control','generated://format-control',provenance='simulated'),),
            metadata={'is_synthetic':True,'scope':'format regression fixture; zero real people; no anatomy'})
        save_case(case,valid)
    with ZipFile(valid) as original:
        members={name:original.read(name) for name in original.namelist()}
    rows=[]
    for name in ['missing-arrays','wrong-affine','truncated']:
        path=directory/(name+'.ressectionlab')
        if not path.exists():
            if name=='truncated':path.write_bytes(valid.read_bytes()[:-32])
            else:
                altered=dict(members)
                if name=='missing-arrays':altered.pop('arrays.npz')
                else:
                    document=json.loads(altered['manifest.json'])
                    with np.load(BytesIO(altered['arrays.npz']),allow_pickle=False) as payload:
                        arrays={key:payload[key] for key in payload.files}
                    arrays['affine'][0,3]+=25.
                    stream=BytesIO();np.savez_compressed(stream,**arrays);altered['arrays.npz']=stream.getvalue()
                    document['array_sha256']=hashlib.sha256(altered['arrays.npz']).hexdigest()
                    altered['manifest.json']=json.dumps(document).encode()
                with ZipFile(path,'w',compression=ZIP_STORED) as archive:
                    for member,data in altered.items():archive.writestr(member,data)
        try:load_workspace(path)
        except Exception as error:rows.append({'path':str(path.relative_to(root)),'sha256':sha256_file(path),
            'expected':'production load rejects; no fallback anatomy','actual_error':type(error).__name__+': '+str(error),'passed':True})
        else:raise ValueError('Failure fixture was incorrectly accepted: '+name)
    load_workspace(valid)
    atomic_json(directory/'expected-failures.json',{'schema':'ressectionlab-negative-fixtures-v1','accepted_patient_count':0,'generated_control_load':'passed','fixtures':rows})
    print(json.dumps(rows,indent=2))


if __name__=='__main__':run()

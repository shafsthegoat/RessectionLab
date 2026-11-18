"""Bounded source-bound SuperSynth segmentation for local research preparation.

Loads the unchanged pinned FreeSurfer v8.2.0 backbone from the external model
cache. Only its segmentation head is evaluated. Overlapping blocks and Gaussian
logit fusion are explicit adaptations; whole-volume equivalence and clinical
accuracy are not asserted. No synthesized MRI or MNI-registration head is used.
"""
from __future__ import annotations
from datetime import datetime,timezone
from pathlib import Path
import argparse
import importlib
import itertools
import json
import sys
import time
import types
import numpy as np
from .data_root import atomic_json,sha256_file

MODEL_VERSION='FreeSurfer v8.2.0 / SuperSynth August 2025'
MODEL_SHA='7f75926757070f53e1e43c4bf31dafe2e599ef56438ce60347f12e9133bffd1d'
ASSETS={'SuperSynth_August_2025.pth':MODEL_SHA,
 'frugal_models.py':'99267132490967c19a99224f8be724082bea53a94950776ee49530109e179601',
 'utils.py':'9939ccfb9438b5d12011d5904496a162d4456c8ea3558b53e8753107cfffe006',
 'FreeSurferColorLUT.txt':'da55c6a47d316ee16d2609afd5c93cb8078e08e7e3afec5bb6283027d7c33e7b'}
PARAMETERS={'mode':'invivo','spacing_mm':1.,'patch':128,'stride':80,'blend':'Gaussian sigma0.38 normalized patch coordinates, floor0.001',
 'inference_dtype':'float16_MPS','logit_accumulator':'float16_CPU','test_time_flipping':False,
 'whole_volume_equivalence':False,'unused_heads':['MNI_registration','synthetic_T1','synthetic_T2','synthetic_FLAIR','cortical_ribbon','automated_QC']}


def tile_starts(length,patch=128,stride=80):
    if length<patch or not 0<stride<=patch:raise ValueError('Padded coverage and overlapping positive stride required')
    starts=list(range(0,length-patch+1,stride))
    if starts[-1]!=length-patch:starts.append(length-patch)
    return starts


def blend_window(patch=128):
    axis=np.linspace(-1,1,patch,dtype=np.float32);g=np.exp(-.5*(axis/.38)**2)
    return np.maximum(g[:,None,None]*g[None,:,None]*g[None,None,:],.001).astype(np.float32)


def load_assets(model_root):
    model_root=Path(model_root)
    for name,digest in ASSETS.items():
        if sha256_file(model_root/name)!=digest:raise ValueError('Pinned model/source changed: '+name)
    return model_root


def _worker(request):
    import nibabel as nib
    import torch
    from .imaging import inspect_nifti
    source=Path(request['source']);out=Path(request['output']);model_root=load_assets(request['model_root'])
    inspect_nifti(source);source_sha=sha256_file(source)
    if source_sha!=request['source_sha256']:raise ValueError('Source changed before inference')
    if not torch.backends.mps.is_available():
        raise RuntimeError('The pinned tiled SuperSynth worker requires MPS')
    torch.set_num_threads(2);torch.mps.set_per_process_memory_fraction(.5)
    module=types.ModuleType('SuperSynth');module.__path__=[str(model_root)];sys.modules['SuperSynth']=module
    utils=importlib.import_module('SuperSynth.utils');upstream=importlib.import_module('SuperSynth.frugal_models')
    labels=np.asarray(utils.get_label_lists_etc()[0],np.uint16)
    checkpoint=torch.load(model_root/'SuperSynth_August_2025.pth',map_location='cpu',weights_only=True,mmap=True)
    net=upstream.frugal_models.__new__(upstream.frugal_models)
    net.ssynth_bbone={k:v.to(device='mps',dtype=torch.float16)for k,v in checkpoint['backbone_state_dict'].items()}
    net.ssynth_final_conv_names=['seg']
    net.ssynth_final_conv_weight={'seg':checkpoint['seg_state_dict']['weight'].to(device='mps',dtype=torch.float16)}
    net.ssynth_final_conv_bias={'seg':checkpoint['seg_state_dict']['bias'].to(device='mps',dtype=torch.float16)}
    if net.ssynth_final_conv_weight['seg'].shape[0]!=len(labels):raise ValueError('Model class-order mismatch')
    del checkpoint
    image=nib.load(source);values=np.asarray(image.dataobj,dtype=np.float32)
    if not np.isfinite(values).all()or values.max()<=0:raise ValueError('Finite nonempty structural MRI required')
    # Official antialiased physical resize; Nibabel performs the corresponding
    # axis permutation/flip without the upstream NumPy1 scalar-assignment idiom.
    input,affine=utils.torch_resize(torch.tensor(values),image.affine.copy(),1.,'cpu')
    canonical=nib.as_closest_canonical(nib.Nifti1Image(input.numpy(),affine))
    values=np.asarray(canonical.dataobj,dtype=np.float32).copy();affine=canonical.affine.copy();values/=values.max()
    shape=np.array(values.shape);padded=np.maximum(PARAMETERS['patch'],((shape+15)//16)*16);offset=(padded-shape)//2
    if int(np.prod(padded)) > 28_000_000:
        raise MemoryError('Resampled SuperSynth field exceeds the local logit workspace limit')
    roi=tuple(slice(int(a),int(a+b))for a,b in zip(offset,shape))
    field=torch.zeros(tuple(padded),dtype=torch.float16,device='mps');field[roi]=torch.tensor(values,device='mps',dtype=torch.float16)
    del input,values,canonical
    scores=np.zeros((len(labels),*map(int,padded)),np.float16);coverage=np.zeros(tuple(padded),np.float32)
    patch=PARAMETERS['patch'];stride=PARAMETERS['stride']
    weight=blend_window(patch);starts=[tile_starts(int(n),patch,stride)for n in padded];tiles=list(itertools.product(*starts));started=time.monotonic()
    with torch.inference_mode():
        for index,start in enumerate(tiles):
            r=tuple(slice(int(a),int(a)+patch)for a in start)
            output=net.ssynth_inference(field[r][None,None])['seg'][0]
            for c in range(0,len(labels),4):
                prediction=output[c:c+4].float().cpu().numpy()
                destination=scores[(slice(c,c+prediction.shape[0]),*r)]
                np.add(destination,prediction*weight,out=destination,casting='unsafe')
            del prediction,destination,output
            torch.mps.empty_cache();coverage[r]+=weight
            print(json.dumps({'tile':index+1,'tiles':len(tiles)}),flush=True)
    if not np.all(coverage>0):raise ValueError('Incomplete inference field coverage')
    indices=np.empty(tuple(padded),np.uint8)
    for x in range(0,int(padded[0]),16):indices[x:x+16]=np.argmax(scores[:,x:x+16],axis=0)
    result=labels[indices[roi]];del scores,indices,field,net
    saved=nib.Nifti1Image(result,affine);saved.header.set_xyzt_units('mm');saved.set_qform(affine,1);saved.set_sform(affine,1)
    nib.save(saved,out/'labels_1mm.nii.gz')
    if sha256_file(source)!=source_sha:raise ValueError('Original image changed during inference')
    atomic_json(out/'worker.json',{'status':'succeeded','seconds':time.monotonic()-started,'source_sha256':source_sha,
        'output_sha256':sha256_file(out/'labels_1mm.nii.gz'),'shape':list(result.shape),'affine_ras_mm':affine.tolist(),
        'labels_present':list(map(int,np.unique(result))),'complete_input_FOV_coverage':True,'tiles':[list(map(int,t))for t in tiles],
        'parameters':PARAMETERS,'clinical_validation':False,'mps_driver_bytes_after_release':torch.mps.driver_allocated_memory()})


def segment_head(source, output, model_root, *, timeout=600, cancelled=lambda: False, retry=False):
    """Serialize the pinned worker. Each retry keeps a separate attempt receipt."""
    from .anatomy_models import run_bounded
    from .patient_anatomy import reconstruction_lock

    source, output, model_root = (Path(item).resolve() for item in (source, output, model_root))
    identity = {"source_sha256": sha256_file(source), "model_sha256": MODEL_SHA,
                "adapter_sha256": sha256_file(Path(__file__)), "parameters": PARAMETERS}
    with reconstruction_lock():
        if cancelled():
            raise InterruptedError("Head segmentation cancelled before inference")
        attempts = [output, *sorted(output.parent.glob(output.name + "-retry-*"))] if retry else [output]
        for attempt in attempts:
            receipt = attempt / "inference.json"
            if not receipt.exists():
                continue
            record = json.loads(receipt.read_text())
            if record.get("status") == "succeeded" and record["identity"] == identity:
                if sha256_file(attempt / "labels_1mm.nii.gz") != record["output_sha256"]:
                    raise ValueError("Cached head labels changed")
                return record
            if not retry:
                raise ValueError("Prior different or failed inference attempt exists; choose a new output directory")
        if retry and output.exists():
            output = output.with_name(output.name + "-retry-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f"))
        output.mkdir(parents=True, exist_ok=True)
        request = {**identity, "source": str(source), "output": str(output), "model_root": str(model_root)}
        atomic_json(output / "request.json", request)
        command = [sys.executable, "-u", "-m", "resectionlab.head_segmentation", "--worker", str(output / "request.json")]
        record = {"identity": identity, "source": str(source), "output": str(output),
                  "created_utc": datetime.now(timezone.utc).isoformat(), "clinical_validation": False,
                  "model_version": MODEL_VERSION, "pinned_assets": ASSETS}
        try:
            stats = run_bounded(command, output, cancelled=cancelled, timeout=timeout,
                                memory_bytes=6 * 1024**3, minimum_available_bytes=int(1.25 * 1024**3),
                                environment_overrides={"PYTHONPATH": str(Path(__file__).resolve().parents[1]),
                                                       "PYTORCH_ENABLE_MPS_FALLBACK": "1"})
            record.update(stats)
            record.update(json.loads((output / "worker.json").read_text()))
            if cancelled():
                raise InterruptedError("Head segmentation cancelled before admission")
        except (OSError, RuntimeError, ValueError, MemoryError) as error:
            record.update(status="cancelled" if isinstance(error, InterruptedError) else "failed", failure=str(error))
            atomic_json(output / "inference.json", record)
            raise
        atomic_json(output / "inference.json", record)
        return record


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--worker',type=Path,required=True);args=parser.parse_args()
    _worker(json.loads(args.worker.read_text()))

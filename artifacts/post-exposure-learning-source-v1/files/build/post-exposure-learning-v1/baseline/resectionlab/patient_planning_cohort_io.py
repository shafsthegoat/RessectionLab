"""Bounded cohort output and parameter-only checkpoint compatibility.

Extracted unchanged from the staged cohort implementation. No training runner,
patient factory, native task, source arrays or execution entry point lives here.
The legacy protocol constructor only authenticates old checkpoint metadata.
"""
import hashlib
import io
import json
import os
from pathlib import Path
import stat
import zipfile

import numpy as np
import torch

from .core import freeze_json, semantic_digest, thaw_json
from .patient_planning_learning import PREFLIGHT_PROTOCOL
from .spatial_policy import SpatialPolicy, SpatialPolicyConfig, parameter_hash

VERSION = 'fixed-four-TRAIN-patient-cohort-v1'
TRAIN = ('ReMIND-008', 'ReMIND-010', 'ReMIND-020', 'ReMIND-025')


def cohort_learning_protocol(updates=8, *, public_target_context_variant=None):
    if type(updates) is not int or not 1 <= updates <= 32:
        raise ValueError('Bounded cohort requires 1..32 declared updates per method')
    record = thaw_json(PREFLIGHT_PROTOCOL)
    record.update(version=VERSION, updates_per_method=updates, rl_episodes=4,
        scope='fixed_four_TRAIN_shared_population_pilot', population_training=True)
    if public_target_context_variant is not None:
        from .public_target_context import VERSION as TARGET_VERSION
        if public_target_context_variant!=TARGET_VERSION:raise ValueError('Unknown public target variant')
        record['public_target_context_variant']=public_target_context_variant
    return freeze_json(record)


class OutputBudget:
    """Bound known writes and reserve a small terminal record; parent is hard guard."""
    RESERVE = 65536
    def __init__(self, root, limit): self.root, self.limit = Path(root), limit
    def size(self): return sum(p.stat().st_size for p in self.root.rglob('*') if p.is_file())
    def check(self):
        if self.size() > self.limit-self.RESERVE:
            raise InterruptedError('Cohort output cap reached; no retry')
    def write_bytes(self, path, raw, *, terminal=False):
        path=Path(path)
        path.relative_to(self.root)
        if self.size()+len(raw) > self.limit-(0 if terminal else self.RESERVE):
            raise InterruptedError('Cohort output write would exceed bound')
        with path.open('xb') as stream: stream.write(raw)
    def write(self, path, value, *, terminal=False):
        raw=(json.dumps(thaw_json(freeze_json(value)),sort_keys=True,indent=2,allow_nan=False)+'\n').encode()
        self.write_bytes(path,raw,terminal=terminal)


def _save_checkpoint(policy, *, method, contexts, protocol, updates, initial_hash, output, limits):
    """Non-pickle weights plus hash-only lineage; no image, mask or observation data."""
    if updates != protocol['updates_per_method']:
        raise ValueError('Only the complete fixed endpoint can be saved as final')
    entries=[]; payloads=[]
    for index,(name,tensor) in enumerate(sorted(policy.state_dict().items())):
        array=tensor.detach().cpu().contiguous().numpy()
        if array.dtype != np.float32 or not np.isfinite(array).all():
            raise ValueError('Finite float32 checkpoint parameters required')
        stream=io.BytesIO(); np.lib.format.write_array(stream,array,version=(1,0),allow_pickle=False)
        raw=stream.getvalue(); entry=f'tensors/{index:04d}.npy'
        entries.append({'name':name,'entry':entry,'shape':list(array.shape),'dtype':'float32','sha256':hashlib.sha256(raw).hexdigest()})
        payloads.append((entry,raw))
    metadata={'version':'patient-cohort-spatial-weights-v1','kind':'fixed_endpoint',
        'method':method,'architecture_hash':policy.architecture_hash,'parameter_hash':parameter_hash(policy),
        'initial_parameter_hash':initial_hash,'learning_protocol_hash':semantic_digest(protocol),
        'completed_updates':updates,'contexts':[{'patient_group':c.patient_group,'context_hash':c.fingerprint} for c in contexts],
        'TRAIN_subjects':list(TRAIN),'SELECT_EVAL_opened':False,'private_reference_used':False,
        'tensors':entries,'format':'ZIP_STORED_JSON_FLOAT32_NPY_V1','clinical_claim':False}
    archive_bytes=io.BytesIO()
    with zipfile.ZipFile(archive_bytes,'w',compression=zipfile.ZIP_STORED) as archive:
        archive.writestr('manifest.json',json.dumps(metadata,sort_keys=True,separators=(',',':'),allow_nan=False))
        for entry,raw in payloads: archive.writestr(entry,raw)
    raw=archive_bytes.getvalue()
    if len(raw)>limits['checkpoint_bytes']: raise InterruptedError('Checkpoint cap reached')
    filename=method+'-final.psckpt'; output.write_bytes(output.root/filename,raw)
    return {'path':filename,'sha256':hashlib.sha256(raw).hexdigest(),'bytes':len(raw),
            'parameter_hash':metadata['parameter_hash'],'metadata':metadata}


def _checkpoint_json(pairs):
    result={}
    for key,value in pairs:
        if key in result: raise ValueError('Duplicate checkpoint JSON key')
        result[key]=value
    return result


def load_cohort_checkpoint(path, *, expected_sha256, expected_learning_protocol,
                           expected_context_hashes, expected_method):
    """Restore existing SpatialPolicy weights from a pinned, bounded artifact.

External pins must come from the training record. Hashes and declared lineage
are checked; this is not independent proof of optimizer execution or permission
to open a patient. The returned model is not registered for a TRAIN session.
No observation, task factory, forward, optimizer, pickle or torch.load is used.
"""
    protocol=freeze_json(expected_learning_protocol)
    if 'cohort_execution' in protocol:
        from .patient_planning_cohort_spec import validate_sequential_protocol
        validate_sequential_protocol(protocol)
    elif semantic_digest(protocol)!=semantic_digest(cohort_learning_protocol(protocol.get('updates_per_method'),
            public_target_context_variant=protocol.get('public_target_context_variant'))):
        raise ValueError('Exact cohort checkpoint protocol required')
    groups=tuple('ReMIND:'+subject.rsplit('-',1)[1] for subject in TRAIN)
    context_hashes=dict(expected_context_hashes)
    if (set(context_hashes)!=set(groups) or any(type(h) is not str or len(h)!=71
            or not h.startswith('sha256:') or any(c not in '0123456789abcdef' for c in h[7:])
            for h in context_hashes.values()) or expected_method not in ('IL','RL')):
        raise ValueError('Exact four TRAIN context pins and method required')
    if (type(expected_sha256) is not str or len(expected_sha256)!=64
            or any(c not in '0123456789abcdef' for c in expected_sha256)):
        raise ValueError('Exact checkpoint SHA256 required')
    bound=8*1024**2
    flags=os.O_RDONLY|os.O_NONBLOCK|getattr(os,'O_NOFOLLOW',0)
    fd=os.open(path,flags)
    try:
        before=os.fstat(fd)
        if not stat.S_ISREG(before.st_mode) or not 0<before.st_size<=bound:
            raise ValueError('Checkpoint must be a bounded regular file')
        with os.fdopen(fd,'rb',closefd=False) as source: payload=source.read(bound+1)
        after=os.fstat(fd)
        if ((before.st_dev,before.st_ino,before.st_size,before.st_mtime_ns,before.st_ctime_ns)
                !=(after.st_dev,after.st_ino,after.st_size,after.st_mtime_ns,after.st_ctime_ns)):
            raise ValueError('Checkpoint changed during bounded read')
    finally: os.close(fd)
    if len(payload)!=before.st_size or hashlib.sha256(payload).hexdigest()!=expected_sha256:
        raise ValueError('Checkpoint bytes/hash differ from expected artifact')
    if torch.get_default_dtype()!=torch.float32 or str(torch.get_default_device())!='cpu':
        raise ValueError('Checkpoint construction requires CPU float32 defaults')
    # Match common_patient_policies without registering new training authority.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(protocol['seed'])
        model=SpatialPolicy(SpatialPolicyConfig(**thaw_json(protocol['architecture'])),
            **({} if protocol.get('public_target_context_variant') is None else {
                'public_target_context_variant':protocol['public_target_context_variant']}))
    initial_hash=parameter_hash(model); expected=model.state_dict()
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        infos=archive.infolist(); names=[item.filename for item in infos]
        if (len(names)!=len(set(names)) or len(names)!=len(expected)+1
                or any(i.compress_type!=zipfile.ZIP_STORED or i.flag_bits&1 for i in infos)
                or sum(i.file_size for i in infos)>bound or 'manifest.json' not in names
                or archive.getinfo('manifest.json').file_size>512*1024):
            raise ValueError('Invalid bounded checkpoint archive inventory')
        metadata=json.loads(archive.read('manifest.json'),object_pairs_hook=_checkpoint_json,
            parse_constant=lambda value: (_ for _ in ()).throw(ValueError('Nonfinite checkpoint JSON')))
        keys={'version','kind','method','architecture_hash','parameter_hash','initial_parameter_hash',
            'learning_protocol_hash','completed_updates','contexts','TRAIN_subjects','SELECT_EVAL_opened',
            'private_reference_used','tensors','format','clinical_claim'}
        if (type(metadata) is not dict or set(metadata)!=keys
                or metadata['version']!='patient-cohort-spatial-weights-v1'
                or metadata['kind']!='fixed_endpoint' or metadata['method']!=expected_method
                or metadata['format']!='ZIP_STORED_JSON_FLOAT32_NPY_V1'
                or metadata['architecture_hash']!=model.architecture_hash
                or metadata['initial_parameter_hash']!=initial_hash
                or metadata['learning_protocol_hash']!=semantic_digest(protocol)
                or type(metadata['completed_updates']) is not int
                or metadata['completed_updates']!=protocol['updates_per_method']
                or metadata['contexts']!=[{'patient_group':g,'context_hash':context_hashes[g]} for g in groups]
                or metadata['TRAIN_subjects']!=list(TRAIN)
                or any(metadata[k] is not False for k in ('SELECT_EVAL_opened','private_reference_used','clinical_claim'))):
            raise ValueError('Checkpoint architecture or declared TRAIN lineage differs')
        rows=metadata['tensors']
        if (type(rows) is not list or len(rows)!=len(expected)
                or any(type(r) is not dict or set(r)!={'name','entry','shape','dtype','sha256'} for r in rows)
                or [r['name'] for r in rows]!=sorted(expected)
                or [r['entry'] for r in rows]!=[f'tensors/{i:04d}.npy' for i in range(len(expected))]
                or set(names)!={'manifest.json',*(r['entry'] for r in rows)}):
            raise ValueError('Missing, duplicate or unexpected checkpoint tensor')
        values={}
        for row in rows:
            data=archive.read(row['entry'])
            if hashlib.sha256(data).hexdigest()!=row['sha256']: raise ValueError('Tensor digest changed')
            reader=io.BytesIO(data)
            if np.lib.format.read_magic(reader)!=(1,0): raise ValueError('Only bounded NPY v1 headers accepted')
            shape,fortran,dtype=np.lib.format.read_array_header_1_0(reader,max_header_size=1024)
            wanted=tuple(expected[row['name']].shape)
            if (shape!=wanted or row['shape']!=list(wanted) or dtype!=np.dtype('<f4')
                    or row['dtype']!='float32' or fortran or len(data)-reader.tell()!=int(np.prod(wanted))*4):
                raise ValueError('Tensor header/shape/dtype differs before allocation')
            array=np.frombuffer(data,dtype='<f4',offset=reader.tell()).reshape(wanted).copy()
            if not np.isfinite(array).all(): raise ValueError('Nonfinite checkpoint tensor')
            values[row['name']]=torch.from_numpy(array)
    model.load_state_dict(values,strict=True)
    if parameter_hash(model)!=metadata['parameter_hash']:
        raise ValueError('Decoded parameter identity differs from checkpoint')
    model.eval()
    return model,freeze_json(metadata)

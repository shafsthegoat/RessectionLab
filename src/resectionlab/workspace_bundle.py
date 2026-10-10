"""Optional bounded workspace members in the existing case ZIP.

Primary CaseData and policy inputs are unchanged. Auxiliary CaseData snapshots
remain display sources. No transfer paths or renderer certification is stored.
"""
from __future__ import annotations
from copy import deepcopy
from hashlib import sha256
from io import BytesIO
import itertools
import json
import math
import os
from pathlib import Path
import re
import tempfile
from zipfile import ZIP_STORED, ZIP_DEFLATED, ZipFile
import numpy as np
from .core import semantic_digest, thaw_json
from .structural_evidence import structural_frame_hash
from .workspace_imaging import DisplaySource, MAX_ATTACHMENT_BYTES, MAX_SERIES

SCHEMA='integrated-workspace-session-v1'
MAX_JSON=4*1024**2
MAX_PRIMARY=1024**3
MAX_TOTAL=MAX_PRIMARY+MAX_ATTACHMENT_BYTES
SOURCE_KEYS={'seriesId','sourceCaseHash','sourceFrameHash','modality','annotationKind','scope'}


def encoded(value):
    return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()


def session_digest(document):
    # ZIP timestamps/compression are storage details, not workspace identity.
    logical={k:v for k,v in document.items() if k not in ('sessionHash','images')}
    logical['images']=[row['source'] for row in document['images']]
    return semantic_digest(logical)


def _object(pairs):
    result={}
    for key,value in pairs:
        if key in result:raise ValueError('Duplicate JSON field')
        result[key]=value
    return result


def _json(value):
    return json.loads(value,object_pairs_hook=_object,
        parse_constant=lambda value:(_ for _ in ()).throw(ValueError('Nonfinite JSON')))


def _members(archive):
    infos=archive.infolist();names=[x.filename for x in infos]
    if (len(names)!=len(set(names)) or any(x.flag_bits&1 or x.is_dir() or
            x.compress_type not in (ZIP_STORED,ZIP_DEFLATED) for x in infos)):
        raise ValueError('Duplicate, encrypted or unsupported workspace member')
    return names


def extension_layout(archive):
    """Metadata-only bounds also used by the legacy primary case reader."""
    names=_members(archive)
    if archive.getinfo('workspace.json').file_size>MAX_JSON:raise ValueError('Workspace metadata exceeds 4 MiB')
    if archive.getinfo('manifest.json').file_size>16*1024**2:raise ValueError('Primary manifest exceeds limit')
    raw=archive.read('workspace.json');document=_json(raw)
    required={'schema','referenceCaseHash','primaryPlanningHash','images','imagingState','episodeReplay','sessionHash'}
    if set(document)!=required or document['schema']!=SCHEMA or not isinstance(document['images'],list):
        raise ValueError('Unsupported workspace extension')
    if len(document['images'])>MAX_SERIES:raise ValueError('Too many workspace images')
    expected=['manifest.json','arrays.npz','workspace.json']
    for index,row in enumerate(document['images']):
        name=f'workspace/series-{index}.ressectionlab';expected.append(name)
        if set(row)!={'member','bytes','sha256','source'} or row['member']!=name or set(row['source'])!=SOURCE_KEYS:
            raise ValueError('Unexpected workspace image identity')
        if type(row['bytes']) is not int or not 0<row['bytes']<=MAX_ATTACHMENT_BYTES+MAX_JSON:
            raise ValueError('Workspace image size exceeds limit')
        if archive.getinfo(name).file_size!=row['bytes'] or not re.fullmatch('[a-f0-9]{64}',str(row['sha256'])):
            raise ValueError('Workspace image byte binding differs')
    if set(names)!=set(expected) or sum(i.file_size for i in archive.infolist())>MAX_TOTAL+64*1024**2:
        raise ValueError('Unlisted workspace members or total size limit')
    if session_digest(document)!=document['sessionHash']:
        raise ValueError('Workspace session digest mismatch')
    if _json(archive.read('manifest.json')).get('workspace_sha256')!=sha256(raw).hexdigest():
        raise ValueError('Workspace metadata hash mismatch')
    return document


def _case_layout(archive,limit,*,extension=False):
    """Inspect NPY sizes/dtypes before np.load can allocate a declared shape."""
    from .imaging import BUNDLE_SCHEMA
    names=_members(archive)
    if not extension and set(names)!={'manifest.json','arrays.npz'}:raise ValueError('Nested source is not a plain case bundle')
    if archive.getinfo('manifest.json').file_size>16*1024**2 or archive.getinfo('arrays.npz').file_size>limit+1024**2:
        raise ValueError('Case payload exceeds workspace limit')
    manifest=_json(archive.read('manifest.json'))
    if manifest.get('schema')!=BUNDLE_SCHEMA:raise ValueError('Unsupported auxiliary case schema')
    keys=['mri','affine']
    for group in ('compartments','source_compartments'):
        keys.extend(manifest['array_index'][group].values())
    for row in manifest.get('structural_evidence',{}).values():keys.append(row['array_key'])
    for row in manifest.get('critical_evidence',{}).values():keys.extend([row['mask_key'],row['coverage_key']])
    for row in manifest.get('prior_proposals',{}).values():keys.extend([row['data_key'],row['coverage_key']])
    if manifest.get('functional_evidence'):
        keys.extend(k for k in manifest['functional_evidence']['array_keys'].values() if k is not None)
    payload=archive.read('arrays.npz')
    if sha256(payload).hexdigest()!=manifest['array_sha256']:raise ValueError('Case array hash mismatch')
    with ZipFile(BytesIO(payload)) as arrays:
        names=_members(arrays)
        if 'brain_mask.npy' in names:keys.append('brain_mask')
        if (len(keys)!=len(set(keys)) or len(keys)>256 or set(names)!={k+'.npy' for k in keys}
                or any(not re.fullmatch(r'[A-Za-z0-9_]+\.npy',name) for name in names)):
            raise ValueError('Unlisted or aliased case arrays')
        expanded=0
        for info in arrays.infolist():
            if info.file_size>limit+65536:raise ValueError('Expanded array exceeds limit')
            with arrays.open(info) as stream:
                version=np.lib.format.read_magic(stream)
                if version==(1,0):shape,order,dtype=np.lib.format.read_array_header_1_0(stream)
                elif version==(2,0):shape,order,dtype=np.lib.format.read_array_header_2_0(stream)
                else:raise ValueError('Unsupported saved NPY header')
                if dtype.hasobject or dtype.kind not in 'buif' or not 1<=len(shape)<=4 or any(type(n) is not int or n<1 for n in shape):
                    raise ValueError('Invalid saved numeric array shape/type')
                size=math.prod(shape)*dtype.itemsize;expanded+=size
                if size+stream.tell()!=info.file_size or expanded>limit:
                    raise ValueError('Declared NPY shape/expanded size mismatch')
                if info.filename=='mri.npy' and (len(shape)!=3 or dtype.kind!='f'):
                    raise ValueError('Source image is not 3D floating point')
                if info.filename=='affine.npy' and shape!=(4,4):raise ValueError('Affine must be 4x4')
    return expanded


def _center(case):
    point=case.affine@np.array([*( (np.asarray(case.mri.shape)-1)/2 ),1.])
    return (point[:3]*([-1,-1,1] if case.frame=='LPS+' else [1,1,1])).tolist()


def view_state(primary,sources,supplied=None):
    cases={'primary':primary,**{s.identity(primary):s.case for s in sources}}
    if supplied is None:
        return {'selectedSeriesId':None,'states':{key:{'cursor':_center(case),'visibleLayers':{n:True for n in case.compartments}} for key,case in cases.items()}}
    if (not isinstance(supplied,dict) or set(supplied)!={'selectedSeriesId','states'} or
            supplied['selectedSeriesId'] not in {None,*list(cases.keys())[1:]} or
            not isinstance(supplied['states'],dict) or set(supplied['states'])!=set(cases)):
        raise ValueError('Workspace selection or per-image state identities differ')
    result=deepcopy(supplied)
    for key,case in cases.items():
        state=result['states'][key]
        if not isinstance(state,dict) or set(state)!={'cursor','visibleLayers'}:raise ValueError('Unknown image view state')
        visible=state['visibleLayers']
        if not isinstance(visible,dict) or set(visible)!=set(case.compartments) or any(type(v) is not bool for v in visible.values()):
            raise ValueError('Image visibility names/types differ from source')
        cursor=state['cursor']
        if cursor is None:continue
        if not isinstance(cursor,list) or len(cursor)!=3 or any(type(v) not in (int,float) or not math.isfinite(v) for v in cursor):
            raise ValueError('Image cursor must be finite RAS millimetres')
        corners=np.array([case.affine@np.array([*point,1.]) for point in itertools.product(*[(-.5,n-.5) for n in case.mri.shape])])[:,:3]
        if case.frame=='LPS+':corners*=np.array([-1,-1,1])
        if (np.asarray(cursor)<corners.min(0)-1e-5).any() or (np.asarray(cursor)>corners.max(0)+1e-5).any():
            raise ValueError('Image cursor is outside its native RAS display bounds')
    return result


def canonical_episode(episode):
    return encoded({k:v for k,v in episode.items() if k!='episodeId'}).decode()


def validate_episode(primary,envelope,cancelled=None):
    """Regenerate the fixed software task; a digest alone cannot certify replay."""
    if envelope is None:return None
    if not isinstance(envelope,dict) or set(envelope)!={'episode','episodeCanonicalJson'} or len(encoded(envelope))>2*1024**2:
        raise ValueError('Invalid or excessive saved episode envelope')
    episode=envelope['episode'];canonical=canonical_episode(episode)
    if canonical!=envelope['episodeCanonicalJson'] or 'sha256:'+sha256(canonical.encode()).hexdigest()!=episode.get('episodeId'):
        raise ValueError('Episode canonical identity mismatch')
    if episode.get('selector') not in ('scripted','SEARCH','RL256_ASPIRATION_TRANSFER') or episode.get('caseHash')!=primary.semantic_hash:
        raise ValueError('Saved episode does not name this fixed generated case')
    if episode['selector']=='RL256_ASPIRATION_TRANSFER':
        # Imported bytes can establish modeled geometry by fresh native replay.
        # They cannot establish that a particular checkpoint chose the actions.
        from .shared_vascular_evaluation import _preflight
        task,_,_,_=_preflight(episode,cancelled)
        from .development_episode import public_display_case
        if public_display_case(task).semantic_hash!=primary.semantic_hash:
            raise ValueError('Saved primary differs from generated source')
        return deepcopy(envelope)
    from .development_episode import execute_development_episode
    expected_case,expected=execute_development_episode(selector=episode['selector'],cancelled=cancelled)
    if expected_case.semantic_hash!=primary.semantic_hash:raise ValueError('Saved primary differs from authoritative generated source')
    actual=deepcopy(episode);expected=deepcopy(expected)
    for row in (actual,expected):
        row.pop('episodeId',None)
        for field in ('planning_seconds','policy_state_check_seconds'):
            timing=row.get('planning',{}).pop(field,None)
            if timing is not None and (type(timing) not in (int,float) or not math.isfinite(timing) or timing<0 or timing>10):
                raise ValueError('Invalid historical episode timing')
    if encoded(actual)!=encoded(expected):raise ValueError('Saved episode differs from authoritative generated history, geometry or accounting')
    return deepcopy(envelope)


def replay_state(envelope,supplied=None):
    if supplied is None:return None
    if (envelope is None or not isinstance(supplied,dict) or set(supplied)!={'episodeId','frameIndex','visible'}
            or supplied['episodeId']!=envelope['episode']['episodeId'] or type(supplied['visible']) is not bool
            or type(supplied['frameIndex']) is not int or not 0<=supplied['frameIndex']<len(envelope['episode']['replayFrames'])):
        raise ValueError('Saved replay selection is stale or malformed')
    return deepcopy(supplied)


def save_workspace(primary,path,*,sources,imaging,episode,selection,artifacts):
    from .imaging import save_case
    sources=list(sources)
    if len(sources)>MAX_SERIES or sum(s.expanded_bytes for s in sources)>MAX_ATTACHMENT_BYTES:
        raise ValueError('Workspace sources exceed display budget')
    imaging=view_state(primary,sources,imaging);selection=replay_state(episode,selection)
    if selection is not None and selection['visible'] and imaging['selectedSeriesId'] is not None:
        raise ValueError('Episode replay cannot be visible on an auxiliary native grid')
    document={'schema':SCHEMA,'referenceCaseHash':primary.semantic_hash,'primaryPlanningHash':primary.planning_hash,
              'images':[],'imagingState':imaging,'episodeReplay':None if episode is None else {'envelope':episode,'selection':selection}}
    destination=Path(path)
    with tempfile.TemporaryDirectory(prefix='.workspace-',dir=destination.parent) as tmp:
        tmp=Path(tmp);base=save_case(primary,tmp/'primary',artifacts=artifacts)
        auxiliary=[]
        for index,source in enumerate(sources):
            p=save_case(source.case,tmp/f'source-{index}')
            data=p.read_bytes();name=f'workspace/series-{index}.ressectionlab';auxiliary.append((name,data))
            document['images'].append({'member':name,'bytes':len(data),'sha256':sha256(data).hexdigest(),'source':source.manifest(primary)})
        document['sessionHash']=session_digest(document);raw=encoded(document)
        if len(raw)>MAX_JSON:raise ValueError('Workspace metadata exceeds limit')
        with ZipFile(base) as plain:
            manifest=_json(plain.read('manifest.json'));manifest['workspace_sha256']=sha256(raw).hexdigest()
            with ZipFile(tmp/'complete','w',compression=ZIP_STORED) as out:
                out.writestr('manifest.json',encoded(manifest));out.writestr('arrays.npz',plain.read('arrays.npz'))
                out.writestr('workspace.json',raw)
                for name,data in auxiliary:out.writestr(name,data)
        os.replace(tmp/'complete',destination)
    return document['sessionHash']


def load_workspace(path,cancelled=None):
    from .imaging import load_case,read_case_artifacts
    with ZipFile(path) as archive:
        names=_members(archive)
        if 'workspace.json' not in names:return load_case(archive.fp),read_case_artifacts(archive.fp),None
        document=extension_layout(archive);_case_layout(archive,MAX_PRIMARY,extension=True)
        sources=[];expanded=0
        for row in document['images']:
            if cancelled is not None and cancelled():raise ValueError('Workspace load cancelled')
            payload=archive.read(row['member'])
            if sha256(payload).hexdigest()!=row['sha256']:raise ValueError('Saved display source bytes changed')
            with ZipFile(BytesIO(payload)) as nested:expanded+=_case_layout(nested,MAX_ATTACHMENT_BYTES)
            if expanded>MAX_ATTACHMENT_BYTES:raise ValueError('Total expanded auxiliary arrays exceed display budget')
            source=row['source'];case=load_case(BytesIO(payload))
            item=DisplaySource(case,source['modality'],source['annotationKind']);sources.append(item)
        # Keep the already-open file identity; never reopen an exchanged path.
        primary=load_case(archive.fp);artifacts=read_case_artifacts(archive.fp)
    if document['referenceCaseHash']!=primary.semantic_hash or document['primaryPlanningHash']!=primary.planning_hash:
        raise ValueError('Workspace primary identity differs')
    if len({s.identity(primary) for s in sources})!=len(sources):raise ValueError('Duplicate workspace sources')
    for row,source in zip(document['images'],sources):
        if row['source']!=source.manifest(primary):raise ValueError('Saved source frame/provenance/scope differs')
    imaging=view_state(primary,sources,document['imagingState'])
    replay=document['episodeReplay'];episode=None;selection=None
    if replay is not None:
        if set(replay)!={'envelope','selection'}:raise ValueError('Unknown saved episode fields')
        episode=validate_episode(primary,replay['envelope'],cancelled)
        selection=replay_state(episode,replay['selection'])
        if selection is not None and selection['visible'] and imaging['selectedSeriesId'] is not None:
            raise ValueError('Episode replay cannot be visible on an auxiliary native grid')
    return primary,artifacts,{'sessionHash':document['sessionHash'],'sources':sources,'imagingState':imaging,'episode':episode,'selection':selection}


def session_descriptor(entry):
    sources=list(entry.display_sources.values())
    inventory=[{'seriesId':'primary','sourceCaseHash':entry.case.semantic_hash,'sourceFrameHash':structural_frame_hash(entry.case),
        'modality':str(entry.case.metadata.get('selected_modality','primary')),'sourceKind':'primary_source','displayAvailable':True,
        'planningInput':True,'readiness':'primary_case_contract','reasons':['Primary source channel only; target, support and task admission remain separate.']}]
    for source in sources:
        inventory.append({'seriesId':source.identity(entry.case),'sourceCaseHash':source.case.semantic_hash,
            'sourceFrameHash':structural_frame_hash(source.case),'modality':source.modality,
            'sourceKind':{'none':'source_image','source-provided':'source_annotation','estimated':'estimated_annotation'}[source.annotation_kind],
            'displayAvailable':True,'planningInput':False,'readiness':'display_only_registration_unreviewed',
            'reasons':['Association and registration unreviewed.','Annotation coverage and source time unknown.','Auxiliary pixels are absent from current policy channels.']})
    replay=None
    if entry.episode is not None and entry.episode_selection is not None:
        replay={**deepcopy(entry.episode),**{k:v for k,v in entry.episode_selection.items() if k!='episodeId'},
                'validation':'authoritative_generated_replay','accountingQualification':'historical_timings_not_remeasured'}
        if replay['episode']['selector']=='RL256_ASPIRATION_TRANSFER':
            from .legacy_transfer_episode import transfer_authorship
            replay['episodeAuthorship']=transfer_authorship(replay['episode'],live_backend_run=False)
            replay['accountingQualification']='imported_computational_provenance_unverified'
    state=view_state(entry.case,sources,entry.imaging_state)
    identity=entry.workspace_hash or semantic_digest({'referenceCaseHash':entry.case.semantic_hash,
        'sources':[s.manifest(entry.case) for s in sources],'imagingState':state,'episodeReplay':replay})
    return {'schema':SCHEMA,'referenceCaseHash':entry.case.semantic_hash,'sessionHash':identity,
        'displaySeries':list(entry.display_series.values()),'imagingState':state,'episodeReplay':replay,'evidenceInventory':inventory}

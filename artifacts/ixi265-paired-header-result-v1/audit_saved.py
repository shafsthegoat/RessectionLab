"""Saved IXI result audit: stdlib only, selected copies, bounded header decode."""
from pathlib import Path
import hashlib,itertools,json,math,os,stat,struct,sys,zlib
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
RUN=ROOT/'build/ixi265-paired-header-qc-v1'
PREP=ROOT/'build/ixi265-paired-header-launch-preparation-v1'
RELEASE='2e2dfa63ec1511fe5e42fe794ebf45e0493f6677e761eeeb288007763616831f'
CANDIDATE='c099b5759a0a92624c4d8ba1f1ba587c846173c202b422b731ec11b34d0e9c5c'
evidence={};original_archive_opens=[]
def guard(event,args):
    if event=='open' and isinstance(args[0],(str,bytes)):
        p=str(args[0])
        if p.startswith(str(ROOT/'data/acquisition/ixi-t1-mra-vessel-v1/verified')) or p.startswith(str(ROOT/'sources')):
            original_archive_opens.append(p);raise AssertionError('original_archive_or_synced_source_open_forbidden')
sys.addaudithook(guard)
def sha(raw):return hashlib.sha256(raw).hexdigest()
def encode(v):return (json.dumps(v,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n').encode()
def saved(path,expected=None):
    path=Path(path);assert path.is_file() and not path.is_symlink() and path.stat().st_size<4*1024**2
    raw=path.read_bytes();digest=sha(raw)
    if expected:assert digest==expected,(str(path),digest,expected)
    evidence[str(path.relative_to(ROOT))]={'sha256':digest,'bytes':len(raw)}
    return json.loads(raw) if path.suffix=='.json' else raw
release=saved(PREP/'root-release-v2.json',RELEASE)
candidate=saved(PREP/'terminal-bound-CANDIDATE-NOT-RELEASED-v2.json',CANDIDATE)
normalized={**release,'execution_released':False,'independent_review':None,'release_head':None,'released_at':None}
assert normalized==candidate and sha(encode(normalized))==CANDIDATE
assert {k for k in release if release[k]!=candidate[k]}=={'execution_released','independent_review','release_head','released_at'}
assert release['execution_released'] is True
review=saved(ROOT/release['independent_review']['path'],release['independent_review']['sha256'])
assert review['decision']=='GO_FOR_EXACT_IXI265_HEADER_LAUNCH' and review['release_candidate_sha256']==CANDIDATE
saved(PREP/'launch_v2.py',release['launcher_sha256'])
for relative,digest in release['source_sha256'].items():saved(ROOT/relative,digest)
session=saved(PREP/'root-session-terminal-v2.json');assert session['exit_code']==0 and session['root_release_sha256']==RELEASE
parent=saved(RUN/'terminal.json');worker=saved(RUN/'worker-terminal.json',parent['worker_terminal_sha256'])
report=saved(RUN/'selected/receipt.json',parent['selected_receipt_sha256'])
intent=saved(RUN/'intent.json',worker['intent_sha256']);start=saved(RUN/'worker-start.json')
protocol=saved(RUN/'child-protocol.json',intent['child_protocol_sha256'])
for v in (parent,worker,intent):assert v['release_sha256']==RELEASE
assert parent['status']=='selected_header_intake_complete_unadmitted' and parent['supervision']=='completed'
assert parent['parent_reaped_worker'] and parent['worker_exit_status']==0 and not parent['termination_requested'] and parent['cleanup_notes']==[]
assert parent['elapsed_seconds']<45 and parent['worker_self_peak_RSS_bytes']<=536870912 and parent['sampled_group_peak_RSS_bytes']<=536870912
assert start['worker_pid']==parent['worker_pid'] and start['intent_sha256']==worker['intent_sha256']
assert intent['outer_deadline_monotonic']-intent['worker_deadline_monotonic']==10
assert worker['status']=='worker_adapter_complete' and worker['reader_exit_code']==0 and worker['process_and_network_guard'] and worker['source_only_bootstrap']
assert worker['worker_peak_RSS_bytes']==parent['worker_self_peak_RSS_bytes']
assert report['status']=='selected_copies_and_header_qc_complete_unadmitted'
for v in (parent,report):
    assert v['person_group']=='IXI:265' and v['role']=='TRAIN' and v['training_admitted'] is False
    assert v['archive_read_accounting_complete'] is True and v['other_person_member_body_bytes_read']==0
assert report['protocol_sha256']==intent['child_protocol_sha256'] and report['voxel_arrays_materialized']==0
assert report['anatomical_registration_accepted'] is False and report['actor_private_reference_access_added'] is False
assert report['archive_sha256_inherited_from_transport'] and not report['whole_archives_rehashed_by_this_worker']
prepared=saved(ROOT/'build/ixi265-paired-header-preparation-v1/prepared-protocol-v2.json')
expected={**prepared,'execution_released':True,'archive_completion_receipt':release['archive_completion_receipt'],'archive_bindings':release['archive_bindings']}
assert protocol==expected
assert report['archive_completion_receipt']==release['archive_completion_receipt']
assert not (RUN/'post-publication-refusal.json').exists() and not (RUN/'terminal-publication-failure.json').exists()

# Saved transport receipt closure plus fresh publication stat only: no archive opens.
completion=saved(ROOT/release['archive_completion_receipt']['path'],release['archive_completion_receipt']['sha256'])
assert completion['status']=='all_bytes_verified' and completion['verified_files']==3 and completion['unresolved_files']==0
assert completion['cohort_sha256']==release['source_sha256']['manifests/experiments/ixi-component-person-cohort-v1.json']
rows={r['path']:r for r in completion['files']};scope=saved(ROOT/'manifests/experiments/ixi-t1-mra-vessel-byte-intake-v1.json')
entries={r['filename']:r for r in scope['files']};stat_results={}
assert set(rows)==set(entries)==set(release['archive_bindings'])==set(release['archive_stat_proofs'])
for name,binding in release['archive_bindings'].items():
    proof=release['archive_stat_proofs'][name];s=saved(ROOT/proof['path'],proof['sha256'])
    t=saved(ROOT/s['verification_receipt_path'],s['verification_receipt_sha256'])
    assert s['source_payload_bytes_read']==0 and not s['fresh_payload_rehash'] and s['partial_absent'] and s['regular_not_symlink']
    assert s['modification_and_metadata_change_times_not_after_verification']
    assert t['status']==rows[name]['status']=='byte_verified'
    assert t['sha256']==rows[name]['sha256']==binding['sha256']==s['recorded_archive_sha256']
    assert t['bytes']==rows[name]['bytes']==entries[name]['bytes']==s['expected_bytes']==binding['stat']['size']
    assert rows[name]['source_md5']==entries[name]['expected_md5']==s['recorded_archive_source_md5']
    p=ROOT/s['archive_path'];assert all(not a.is_symlink() for a in [p,*p.parents] if a.is_relative_to(ROOT))
    st=p.stat();current={k:getattr(st,'st_'+k) for k in ('dev','ino','size','mtime_ns','ctime_ns')}
    assert stat.S_ISREG(st.st_mode) and current==binding['stat']
    stat_results[name]={'unchanged_stat':True,'archive_body_bytes_read_by_audit':0}

def delta(a,b,shape):
    return max(math.sqrt(sum(sum((a[i][j]-b[i][j])*v[j] for j in range(4))**2 for i in range(3)))
               for v in [(*c,1.) for c in itertools.product(*[(-.5,n-.5) for n in shape])])
def header(path):
    decoder=zlib.decompressobj(31);pending=b'';returned=0;decoded=0
    with path.open('rb',buffering=0) as stream:
        def exact(n):
            nonlocal pending,returned,decoded
            out=b''
            while len(out)<n:
                if not pending:
                    pending=stream.read(4096);returned+=len(pending);assert pending and returned<=1024**2
                block=decoder.decompress(pending,n-len(out));pending=decoder.unconsumed_tail;out+=block
            decoded+=len(out);return out
        first=exact(4);endian='<' if struct.unpack('<i',first)[0]==348 else '>'
        assert struct.unpack(endian+'i',first)[0]==348
        h=first+exact(348);assert len(h)==352 and decoded==352 and h[344:348]==b'n+1\0' and h[348:352]==b'\0'*4
    get=lambda fmt,offset:struct.unpack_from(endian+fmt,h,offset)
    dim=get('8h',40);shape=list(dim[1:4]);assert dim[0]==3
    datatype,bitpix=get('2h',70);pix=list(get('8f',76));offset=get('f',108)[0];units=h[123]
    assert datatype==4 and bitpix==16 and units&7==2 and units&0x38==8 and offset==352
    qcode,scode=get('2h',252);assert qcode==scode==1
    b,c,d=get('3f',256);a=math.sqrt(max(0,1-b*b-c*c-d*d));x,y,z=get('3f',268)
    rot=[[a*a+b*b-c*c-d*d,2*b*c-2*a*d,2*b*d+2*a*c],
         [2*b*c+2*a*d,a*a+c*c-b*b-d*d,2*c*d-2*a*b],
         [2*b*d-2*a*c,2*c*d+2*a*b,a*a+d*d-c*c-b*b]]
    scale=[pix[1],pix[2],pix[3]*(-1 if pix[0]<0 else 1)]
    q=[[rot[i][j]*scale[j] for j in range(3)]+[[x,y,z][i]] for i in range(3)]+[[0.,0.,0.,1.]]
    s=[list(get('4f',280+i*16)) for i in range(3)]+[[0.,0.,0.,1.]]
    return {'shape':shape,'zooms':pix[1:4],'dtype':endian+'i2','units':'mm','qform':q,'sform':s,
            'qform_code':qcode,'sform_code':scode,'declared_voxel_bytes':math.prod(shape)*2,
            'decoded_header_bytes':decoded,'compressed_prefix_bytes_read':returned}

members={};copy_hash_bytes=0;prefix_bytes=0;decoded_bytes=0
assert set(report['members'])=={'T1','MRA','derived_vessel_label'} and len(report['member_attempts'])==3
for attempt in report['member_attempts']:assert attempt['status']=='selected_copy_and_header_complete'
for kind,m in report['members'].items():
    path=RUN/'selected'/m['output_path'];assert not path.is_symlink() and path.is_file()
    digest=hashlib.sha256();crc=0;size=0
    with path.open('rb',buffering=0) as stream:
        while block:=stream.read(1024**2):digest.update(block);crc=zlib.crc32(block,crc);size+=len(block)
    assert size==m['bytes'] and digest.hexdigest()==m['sha256'] and m['selected_copy_rehash_verified']
    copy_hash_bytes+=size
    if kind=='derived_vessel_label':assert crc==m['zip_crc32'] and m['zip_member_crc_verified']
    else:
        frozen=protocol['selected_person']['raw_files'][kind]
        assert all(m[k]==frozen[k] for k in ('header_offset','header_sha256')) and size==frozen['size'] and m['member']==frozen['name']
        assert m['archive_bytes_read']==512+size
    assert m['archive_sha256']==release['archive_bindings'][m['archive']]['sha256'] and m['other_member_body_bytes_read']==0
    h=header(path);saved_h=m['header_qc'];g=saved_h['geometry'];prefix_bytes+=h['compressed_prefix_bytes_read'];decoded_bytes+=h['decoded_header_bytes']
    assert h['shape']==saved_h['shape'] and h['dtype']==saved_h['dtype'] and h['declared_voxel_bytes']==saved_h['declared_voxel_bytes']
    assert h['zooms']==g['zooms_in_source_units'] and g['spatial_units']=='mm'
    for f in ('qform','sform'):
        assert g[f]['code']==1 and max(abs(h[f][i][j]-g[f]['affine_in_source_units'][i][j]) for i in range(4) for j in range(4))<1e-9
    disagreement=delta(h['qform'],h['sform'],h['shape'])
    assert abs(disagreement-g['max_full_cell_corner_difference_mm'])<1e-9 and disagreement<.001 and g['issues']==[]
    assert saved_h['acquisition_datetime'] is None and saved_h['cross_modality_time_interval'] is None
    assert saved_h['decoded_header_bytes']==352 and saved_h['voxel_arrays_materialized']==0
    assert not saved_h['gzip_footer_verified'] and not saved_h['voxel_payload_length_verified'] and not saved_h['extension_body_parsed']
    members[kind]={'bytes':size,'sha256':digest.hexdigest(),'shape':h['shape'],'zooms_mm':h['zooms'],'dtype':h['dtype'],
                   'qform_code':1,'sform_code':1,'axis_codes':g['sform']['axis_codes'],'qform_sform_max_cell_corner_delta_mm':disagreement,
                   'independent_header_bytes':h['decoded_header_bytes'],'independent_compressed_prefix_bytes':h['compressed_prefix_bytes_read'],
                   'header_matrices_reproduced':True}

mg=report['members']['MRA']['header_qc']['geometry'];lg=report['members']['derived_vessel_label']['header_qc']['geometry']
cross={f:delta(mg[f]['affine_in_source_units'],lg[f]['affine_in_source_units'],members['MRA']['shape']) for f in ('qform','sform')}
assert members['MRA']['shape']==members['derived_vessel_label']['shape'] and members['T1']['shape']!=members['MRA']['shape']
assert all(0<v<.001 for v in cross.values())
for c in report['declared_grid_comparisons'].values():assert not c['accepted_registration'] and not c['simultaneous_acquisition_established']
paths=list(RUN.rglob('*'));assert all(not p.is_symlink() for p in paths)
output=sum(p.stat().st_size for p in paths if p.is_file())
assert output<33554432 and output==parent['outer_output_bytes_before_terminal']+(RUN/'terminal.json').stat().st_size
for p in paths:
    if p.is_file() and p.suffix=='.json':saved(p)
assert original_archive_opens==[]
audit={'decision':'PASS_SAVED_HEADER_RESULT_UNADMITTED','release_sha256':RELEASE,'release_head':release['release_head'],
       'candidate_sha256':CANDIDATE,'source_and_review_bindings_verified':True,'parent_elapsed_seconds':parent['elapsed_seconds'],
       'worker_header_elapsed_seconds':report['elapsed_seconds'],'group_RSS_observations':parent['group_RSS_observations'],
       'peak_group_and_worker_RSS_bytes':parent['worker_self_peak_RSS_bytes'],'worker_pid':parent['worker_pid'],'worker_reaped_exit_zero':True,
       'outer_output_bytes':output,'selected_compressed_copy_bytes':copy_hash_bytes,'members':members,
       'source_archive_returned_bytes_recorded':sum(m['archive_bytes_read'] for m in report['members'].values()),
       'worker_selected_copy_rehash_bytes':copy_hash_bytes,'worker_compressed_header_prefix_bytes_recorded':sum(m['header_qc']['compressed_prefix_bytes_read'] for m in report['members'].values()),
       'worker_header_bytes_decoded':sum(m['header_qc']['decoded_header_bytes'] for m in report['members'].values()),
       'complete_member_attempts':3,'other_person_body_bytes_recorded':0,'voxel_arrays_materialized':0,
       'current_archive_stat_continuity':stat_results,'MRA_label_max_same_index_cell_corner_delta_mm':cross,
       'audit_io':{'selected_copy_hash_bytes':copy_hash_bytes,'compressed_header_prefix_bytes':prefix_bytes,'header_bytes_decoded':decoded_bytes,'original_archive_body_bytes':0,'voxel_arrays':0},
       'limitations':['No anatomy or registration admission; numerical header agreement is not anatomical validation.',
                      'T1/MRA grids differ; MRA/derived-label shape and declared spacing agree but qform/sform matrices are not byte-exact.',
                      'Acquisition datetime, interval and preoperative availability remain unknown; seconds unit does not establish timing.',
                      'Acquired T1/MRA and derived vessel label remain separate; unknown label background is not proven vessel-free.',
                      'Original archive fixity remains inherited from transport plus unchanged stat; no archive-body reread.',
                      'No full voxel decode, gzip-footer, full-payload-length, extension-body or anatomical validation.',
                      'Reported source-read counters are worker receipt/source evidence; this saved audit did not independently replay source reads.'],
       'evidence':evidence}
(HERE/'audit.json').write_text(json.dumps(audit,sort_keys=True,indent=2)+'\n')
print(json.dumps({k:v for k,v in audit.items() if k not in ('evidence','limitations','current_archive_stat_continuity')},sort_keys=True))

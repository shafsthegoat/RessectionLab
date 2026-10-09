#!/usr/bin/env python3
"""One IXI265 TRAIN header attempt; root release required; no broad extraction."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import stat
import subprocess
import sys
import time
import types

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
OUTPUT = ROOT/'build/ixi265-paired-header-qc-v1'
BOOT = 'build/remind-qc-adapter-preparation-v1/header_pilot_bootstrap_v2.py'
SUPERVISOR = 'build/remind-qc-adapter-preparation-v1/header_pilot_launcher_v2.py'
RUNTIME_SOURCE = 'build/remind-qc-adapter-preparation-v1/root-release-v2-r3.json'
READER = 'build/ixi265-paired-header-preparation-v1/selective_qc_v2.py'
PROTOCOL = 'build/ixi265-paired-header-preparation-v1/prepared-protocol-v2.json'
IO = 'scripts/real_intake_io.py'
RSS = 'scripts/febio_runtime.py'
COMPLETION = 'data/acquisition/ixi-t1-mra-vessel-v1/completion.json'
ARCHIVES = 'data/acquisition/ixi-t1-mra-vessel-v1/verified'
PINS = {
 BOOT:'fc6f9661997f6bde4b215aeb69ac8427b07a15d7beac345d8404a18875dfb3ff',
 SUPERVISOR:'0cdfa0e257c69da805fa44101f44cf0f7f034ebb30cc23eb50233d65c6cc7652',
 RUNTIME_SOURCE:'65c3f90df6e10028ae7c9939ed610cf332e1b368a56ed1235860a239db55aa29',
 READER:'4388145cde74e5421346391a6ecafba8ec177c6972187e5183c817fc3a361541',
 PROTOCOL:'ad53a12cdfe480ff53aacc023aebb80e653cf797a47c3df486cd46c52bc8bde8',
 IO:'496821360b1daee79459952f970d299856baacad75fcc57ee589e55115036fd8',
 RSS:'679594d7f3759f5485b9fb868e7e5543ebb242bccd24d112f9ca862bb6d2a01e',
 'build/ixi265-paired-header-independent-v2/REPORT.txt':'f5cfd4dd77c4f78fe3e1461749849db90bea5fa4efe0565a835200a7a6ee312f',
 'build/ixi265-paired-header-independent-v2/audit.json':'24003991f1266cb9b962ab5249911132c0de1af297ba4fcd5d30c1b68f21fd21',
}
LIMITS = {'worker_seconds':35,'cleanup_and_finalization_seconds':10,
          'sampled_group_RSS_bytes':512*1024**2,'outer_output_bytes':32*1024**2,
          'attempts':1,'workers':1,'receipt_bytes':64*1024}
ENV = {'PATH':'/usr/bin:/bin','LANG':'C','LC_ALL':'C','OMP_NUM_THREADS':'1',
       'OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1','VECLIB_MAXIMUM_THREADS':'1','NUMEXPR_NUM_THREADS':'1'}
if sys.platform=='darwin':ENV['__CF_USER_TEXT_ENCODING']='0x%X:0x0:0x0'%os.getuid()

class Refusal(ValueError):pass

def need(ok,reason):
    if not ok:raise Refusal(reason)
def encode(value):return (json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n').encode()
def sha(raw):return hashlib.sha256(raw).hexdigest()
def source(relative):
    path=ROOT/relative;raw=path.read_bytes();need(sha(raw)==PINS[relative],'source_pin:'+relative);return raw

def module(relative,name):
    item=types.ModuleType(name);item.__file__=str(ROOT/relative)
    exec(compile(source(relative),item.__file__,'exec',dont_inherit=True),item.__dict__);return item

def metadata(path,digest=None,cap=4*1024**2):
    path=Path(path);need(path.suffix=='.json' and not path.is_symlink(),'metadata_path')
    with path.open('rb') as f:raw=f.read(cap+1)
    need(len(raw)<=cap,'metadata_cap')
    if digest is not None:need(sha(raw)==digest,'metadata_pin')
    return json.loads(raw)

def referenced(proof):
    need(isinstance(proof,dict) and set(proof)=={'path','sha256'},'metadata_reference')
    p=ROOT/proof['path'];need(p.resolve()==p and p.is_relative_to(ROOT),'metadata_reference_path')
    return metadata(p,proof['sha256'])

def candidate():
    inherited=json.loads(source(RUNTIME_SOURCE));protocol=json.loads(source(PROTOCOL))
    pins=dict(PINS);pins.update(protocol['metadata_and_helper_sha256'])
    return {'schema':'ixi265-exact-selected-header-launch-v1','execution_released':False,
        'independent_review':None,'archive_completion_receipt':None,'archive_bindings':{},
        'archive_stat_proofs':{},'release_head':None,'released_at':None,
        'source_sha256':pins,'launcher_sha256':sha(Path(__file__).read_bytes()),
        'bootstrap_sha256':PINS[BOOT],'bounded_io_path':READER,'bounded_io_sha256':PINS[READER],
        'runtime':inherited['runtime'],'startup_contract':inherited['startup_contract'],
        'limits':LIMITS,'output_directory':str(OUTPUT),'person_group':'IXI:265','role':'TRAIN',
        'scientific_claims':{'anatomy_qc':'not_run','accepted_registration':False,'training_admitted':False,
            'simultaneity_established':False,'image_arrays_materialized':0,'derived_label_is_raw_acquisition':False}}

def check_release(path,digest,executing=False):
    release=metadata(path,digest);expected=candidate()
    for key in ('execution_released','independent_review','archive_completion_receipt','archive_bindings',
                'archive_stat_proofs','release_head','released_at'):expected[key]=release[key]
    need(release==expected,'release_scope_source_runtime_or_limit_mismatch')
    for relative,d in release['source_sha256'].items():need(sha((ROOT/relative).read_bytes())==d,'source_pin:'+relative)
    if executing:
        need(release['execution_released'] is True,'root_release_required')
        review=release['independent_review'];need(review and review['decision']=='GO_FOR_EXACT_IXI265_HEADER_LAUNCH','launcher_review_required')
        p=ROOT/review['path'];need(p.resolve()==p and p.is_relative_to(ROOT),'review_path')
        need(sha(p.read_bytes())==review['sha256'],'review_digest')
        need(isinstance(release['release_head'],str) and len(release['release_head'])==40 and release['released_at'],'root_release_record')
        startup=getattr(sys,'_remind_header_startup',None)
        need(startup and startup['bootstrap_sha256']==PINS[BOOT] and startup['environment']==ENV,
             'source_only_isolated_startup_required')
        need(sys.flags.isolated and sys.flags.no_site and sys.flags.dont_write_bytecode,'isolated_flags')
    return release

def runtime_fixity(release):
    runtime=release['runtime']
    pins=dict(runtime['stdlib_files_sha256'])
    pins.update({str(Path(runtime['prefix'])/p):d for p,d in runtime['dependency_files_sha256'].items()})
    pins[runtime['executable']]=runtime['executable_sha256']
    for p,d in pins.items():need(sha(Path(p).read_bytes())==d,'runtime_pin')
    return len(pins)

def archive_prerequisites(release,*,check_stat=True):
    proof=release['archive_completion_receipt'];need(proof and proof['path']==COMPLETION,'completion_required')
    completion=referenced(proof);scope=metadata(ROOT/'manifests/experiments/ixi-t1-mra-vessel-byte-intake-v1.json')
    entries={x['filename']:x for x in scope['files']}
    need(completion['status']=='all_bytes_verified' and completion['verified_files']==3 and completion['unresolved_files']==0,'all_three_terminal_required')
    need(completion['cohort_sha256']==release['source_sha256']['manifests/experiments/ixi-component-person-cohort-v1.json'],'cohort_mismatch')
    need(set(release['archive_bindings'])==set(entries)==set(release['archive_stat_proofs']),'three_archive_bindings_required')
    rows={x['path']:x for x in completion['files'] if x['status']=='byte_verified'}
    need(set(rows)==set(entries),'verified_archive_set')
    for name,entry in entries.items():
        binding=release['archive_bindings'][name];s=referenced(release['archive_stat_proofs'][name]);row=rows[name]
        need(s['schema']=='ixi-verified-archive-stat-binding-v1' and s['archive_path']==ARCHIVES+'/'+name,'stat_scope')
        need(s['source_payload_bytes_read']==0 and s['fresh_payload_rehash'] is False and s['regular_not_symlink'] is True
             and s['partial_absent'] is True and s['modification_and_metadata_change_times_not_after_verification'] is True,'stat_proof_status')
        mapped={k:s['stat'][v] for k,v in {'dev':'device','ino':'inode','size':'size','mtime_ns':'mtime_ns','ctime_ns':'ctime_ns'}.items()}
        need(binding=={'sha256':s['recorded_archive_sha256'],'stat':mapped},'stat_binding')
        need(binding['sha256']==row['sha256'] and row['bytes']==entry['bytes']==s['expected_bytes']==mapped['size']
             and row['source_md5']==entry['expected_md5']==s['recorded_archive_source_md5'],'archive_fixity_receipt')
        terminal=referenced({'path':s['verification_receipt_path'],'sha256':s['verification_receipt_sha256']})
        need(terminal['status']=='byte_verified' and terminal['sha256']==row['sha256'] and terminal['bytes']==entry['bytes'],'terminal_attempt')
        if check_stat:
            path=ROOT/ARCHIVES/name
            need(all(not p.is_symlink() for p in [path,*path.parents] if p.is_relative_to(ROOT)),'archive_symlink')
            st=path.stat();current={'dev':st.st_dev,'ino':st.st_ino,'size':st.st_size,'mtime_ns':st.st_mtime_ns,'ctime_ns':st.st_ctime_ns}
            need(stat.S_ISREG(st.st_mode) and current==mapped,'archive_stat_changed')
    return completion

def child_protocol(release):
    p=json.loads(source(PROTOCOL));p.update(execution_released=True,
        archive_completion_receipt=release['archive_completion_receipt'],archive_bindings=release['archive_bindings'])
    return p

def save(directory,name,value,io,supervisor):
    raw=encode(value);need(len(raw)<=LIMITS['receipt_bytes'],'receipt_cap')
    need(supervisor.output_bytes(directory)+len(raw)<=LIMITS['outer_output_bytes'],'outer_output_cap')
    io.atomic_preserve(directory/name,raw);return sha(raw)

def deny_worker_process_and_network(event,args):
    if event in ('subprocess.Popen','os.system','os.fork','os.forkpty','os.posix_spawn','os.exec','os.spawn') or event.startswith('socket.'):
        raise Refusal('worker_process_or_network_denied')

def semantic_success(report,protocol,protocol_sha,release):
    need(report['status']=='selected_copies_and_header_qc_complete_unadmitted','worker_semantic_refusal')
    need(report['person_group']=='IXI:265' and report['role']=='TRAIN' and report['protocol_sha256']==protocol_sha,'worker_scope')
    need(report['archive_completion_receipt']==release['archive_completion_receipt'],'worker_completion_binding')
    need(report['archive_read_accounting_complete'] is True and report['other_person_member_body_bytes_read']==0,'worker_read_accounting')
    for key in ('voxel_arrays_materialized','training_admitted','anatomical_registration_accepted','actor_private_reference_access_added'):
        need(report[key]==0,'worker_claims')
    need(report['archive_sha256_inherited_from_transport'] is True and report['whole_archives_rehashed_by_this_worker'] is False,'archive_hash_scope')
    names={'T1':'IXI-T1.tar','MRA':'IXI-MRA.tar','derived_vessel_label':'vessel_dataset.zip'}
    need(set(report['members'])==set(names) and len(report['member_attempts'])==3,'worker_exact_members')
    need(report['member_attempts']==[{'kind':k,'archive':a,'status':'selected_copy_and_header_complete'} for k,a in names.items()],'worker_attempts')
    for kind,name in names.items():
        m=report['members'][kind];derived=kind=='derived_vessel_label'
        need(m['archive']==name and m['archive_sha256']==release['archive_bindings'][name]['sha256'],'member_archive')
        expected_member=protocol['selected_person']['vessel_annotation']['member'] if derived else protocol['selected_person']['raw_files'][kind]['name']
        need(m['member']==expected_member and m['source_kind']==('derived_reference' if derived else 'acquired_measurement'),'member_source_kind')
        need(m['output_path']==('derived_reference' if derived else 'acquired')+'/IXI265-'+kind+'.nii.gz','member_output')
        need(m['selected_copy_rehash_verified'] is True and len(m['sha256'])==64 and all(c in '0123456789abcdef' for c in m['sha256']),'copy_fixity')
        need(m['other_member_body_bytes_read']==0 and type(m['archive_bytes_read']) is int and m['archive_bytes_read']>0,'member_read_scope')
        need(type(m['bytes']) is int and 0<m['bytes']<=(8 if derived else 32)*1024**2,'copy_size_bound')
        if not derived:
            raw=protocol['selected_person']['raw_files'][kind]
            need(m['bytes']==raw['size'] and m['header_sha256']==raw['header_sha256'] and m['header_offset']==raw['header_offset']
                 and m['archive_bytes_read']==512+raw['size'],'raw_member_identity')
        h=m['header_qc'];need(h['status']=='header_only_inspected_no_anatomical_admission' and h['voxel_arrays_materialized']==0,'header_scope')
        need(h['decoded_header_bytes'] in (352,544) and h['acquisition_datetime'] is None and h['cross_modality_time_interval'] is None,'header_time_scope')
        need(h['file_mtime_or_filename_code_interpreted_as_acquisition_time'] is False and h['extension_body_parsed'] is False
             and h['gzip_footer_verified'] is False and h['voxel_payload_length_verified'] is False,'header_limits')
        need(h['library_versions']=={k:release['runtime']['distributions'][k] for k in ('numpy','nibabel')},'header_library_versions')
    need(set(report['declared_grid_comparisons'])=={'T1_to_MRA','MRA_to_derived_label'},'grid_comparisons')
    for comparison in report['declared_grid_comparisons'].values():
        need(comparison['accepted_registration'] is False and comparison['simultaneous_acquisition_established'] is False,'no_registration_claim')

def worker(release,path,digest,intent_sha):
    intent=metadata(OUTPUT/'intent.json',intent_sha)
    need(intent['release_sha256']==digest and intent['parent_pid']==os.getppid(),'worker_owned_intent')
    need(time.monotonic()<intent['worker_deadline_monotonic'],'worker_deadline')
    io=module(IO,'ixi_atomic_worker');supervisor=module(SUPERVISOR,'ixi_worker_save')
    sys.addaudithook(deny_worker_process_and_network)
    result={'status':'worker_adapter_failed','release_sha256':digest,'intent_sha256':intent_sha,
            'process_and_network_guard':True,'source_only_bootstrap':True}
    try:
        expected=encode(child_protocol(release));actual=(OUTPUT/'child-protocol.json').read_bytes()
        need(actual==expected and sha(actual)==intent['child_protocol_sha256'],'worker_protocol')
        reader=module(READER,'ixi_fixed_reader_v2')
        sys.argv=[str(ROOT/READER),'--protocol',str(OUTPUT/'child-protocol.json'),'--protocol-sha256',sha(actual)]
        code=reader.main();result['reader_exit_code']=code
        sys._remind_header_verify_origins()
        peak=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform=='darwin' else 1024))
        result['worker_peak_RSS_bytes']=peak
        need(code==0 and peak<=LIMITS['sampled_group_RSS_bytes'] and time.monotonic()<intent['worker_deadline_monotonic'],'reader_refusal_or_bound')
        result['status']='worker_adapter_complete'
    except BaseException as error:result.update(error_type=type(error).__name__,reason=str(error) if isinstance(error,Refusal) else 'worker_adapter_refusal')
    finally:save(OUTPUT,'worker-terminal.json',result,io,supervisor)
    return 0 if result['status']=='worker_adapter_complete' else 1


def supervise_owned(command,directory,deadline,on_start,supervisor,rss_module,*,popen=subprocess.Popen):
    """Small adapter around reviewed lifecycle: retain handle for exceptional cleanup."""
    owned=[];observations=[];failure=None;result={}
    original=supervisor.subprocess
    def record(*args,**kwargs):
        process=popen(*args,**kwargs);owned.append(process);return process
    supervisor.subprocess=types.SimpleNamespace(Popen=record,DEVNULL=subprocess.DEVNULL,TimeoutExpired=subprocess.TimeoutExpired)
    def group_rss(pid):
        remaining=deadline-time.monotonic();need(remaining>0,'RSS_deadline')
        rss,members=rss_module.process_group_rss(pid,timeout_seconds=min(.5,remaining))
        if members:
            need(all(x['pid']==pid for x in members),'unexpected_worker_descendant')
            observations.append(rss)
        return rss
    try:
        result=supervisor.supervise(command,directory,deadline=deadline,rss_limit=LIMITS['sampled_group_RSS_bytes'],
            disk_cap=LIMITS['outer_output_bytes'],on_start=on_start,rss_probe=group_rss)
    except BaseException as error:
        failure=type(error).__name__;result={'supervision':'launch_or_cleanup_failed','error_type':failure}
    finally:
        supervisor.subprocess=original
        if owned:
            process=owned[0]
            # This fallback owns a direct handle; it never signals a possibly stale PGID.
            if process.poll() is None:
                try:
                    process.kill();process.wait(timeout=1)
                except BaseException as error:result['direct_cleanup_error']=type(error).__name__
            if process.poll() is not None:
                try:process.wait(timeout=0)
                except BaseException as error:result['reap_error']=type(error).__name__
            result.update(worker_pid=process.pid,worker_exit_status=process.returncode,parent_reaped_worker=process.poll() is not None)
        else:result.update(worker_pid=None,parent_reaped_worker=True,worker_exit_status=None)
    result['group_RSS_observations']=len(observations)
    result['sampled_group_peak_RSS_bytes']=max(observations) if observations else None
    if not observations:result['group_RSS_note']='No live group sample; sampled peak is unknown, worker self peak required separately.'
    return result


def parent(release,path,digest):
    # Metadata/runtime-only preflight precedes reserving the single attempt; no archive opens.
    runtime_count=runtime_fixity(release);archive_prerequisites(release)
    supervisor=module(SUPERVISOR,'ixi_reused_supervisor');io=module(IO,'ixi_atomic_parent');rss=module(RSS,'ixi_reused_group_rss')
    need(not OUTPUT.exists() and not OUTPUT.is_symlink(),'single_attempt_already_reserved')
    need(all(not p.is_symlink() for p in OUTPUT.parents if p.is_relative_to(ROOT)),'output_ancestor_symlink')
    with supervisor.lifecycle_termination() as stop:
        OUTPUT.mkdir();started=time.monotonic();deadline=started+LIMITS['worker_seconds'];outer_deadline=deadline+LIMITS['cleanup_and_finalization_seconds']
        terminal={'schema':'ixi265-header-parent-terminal-v1','status':'failed_or_incomplete_no_retry','release_sha256':digest,
          'person_group':'IXI:265','role':'TRAIN','runtime_files_verified':runtime_count,'archive_read_accounting_complete':False,
          'other_person_member_body_bytes_read':None,'anatomy_qc':'not_run','training_admitted':False,
          'registration_accepted':False,'attempts':1,'worker_pid':None,'parent_reaped_worker':True}
        try:
            raw=encode(child_protocol(release));io.atomic_preserve(OUTPUT/'child-protocol.json',raw)
            intent={'release_sha256':digest,'parent_pid':os.getpid(),'worker_deadline_monotonic':deadline,
                    'outer_deadline_monotonic':outer_deadline,'child_protocol_sha256':sha(raw),
                    'source_opens_may_occur_after_worker_start':True,'archive_read_accounting_complete':False,
                    'other_person_member_body_bytes_read':None,'no_retry':True}
            intent_sha=save(OUTPUT,'intent.json',intent,io,supervisor)
            need(not stop['requested'],'termination_requested_before_spawn')
            command=[str(Path(release['runtime']['prefix'])/'bin/python'),'-I','-S','-B',str(ROOT/BOOT),str(path),digest,str(Path(__file__).resolve()),
                     '--release',str(path),'--release-sha256',digest,'--worker','--intent-sha256',intent_sha]
            def on_start(pid):save(OUTPUT,'worker-start.json',{'worker_pid':pid,'intent_sha256':intent_sha,'unknown_source_reads':True},io,supervisor)
            result=supervise_owned(command,OUTPUT,deadline,on_start,supervisor,rss)
            terminal.update(result)
            need(result['supervision']=='completed' and result['worker_exit_status']==0 and result['parent_reaped_worker'] is True,'supervision_refusal')
            need(not stop['requested'] and time.monotonic()<outer_deadline,'termination_or_outer_deadline')
            worker_report=metadata(OUTPUT/'worker-terminal.json',cap=LIMITS['receipt_bytes'])
            need(worker_report['status']=='worker_adapter_complete' and worker_report['release_sha256']==digest
                 and worker_report['intent_sha256']==intent_sha and worker_report['worker_peak_RSS_bytes']<=LIMITS['sampled_group_RSS_bytes'],'worker_adapter_refusal')
            report=metadata(OUTPUT/'selected/receipt.json',cap=LIMITS['receipt_bytes'])
            semantic_success(report,child_protocol(release),sha(raw),release)
            archive_prerequisites(release)  # Publication identity remains unchanged; stat only.
            for member in report['members'].values():
                copy=OUTPUT/'selected'/member['output_path'];need(not copy.is_symlink() and copy.is_file() and copy.stat().st_size==member['bytes'],'selected_copy_stat')
            sys._remind_header_verify_origins()
            terminal.update(status='selected_header_intake_complete_unadmitted',archive_read_accounting_complete=True,
                other_person_member_body_bytes_read=0,worker_self_peak_RSS_bytes=worker_report['worker_peak_RSS_bytes'],
                selected_receipt_sha256=sha((OUTPUT/'selected/receipt.json').read_bytes()),worker_terminal_sha256=sha((OUTPUT/'worker-terminal.json').read_bytes()))
        except BaseException as error:
            terminal.update(error_type=type(error).__name__,reason=str(error) if isinstance(error,Refusal) else 'parent_refusal')
        finally:
            terminal['termination_requested']=stop['requested'];terminal['elapsed_seconds']=time.monotonic()-started
            if stop['requested'] or time.monotonic()>=outer_deadline:terminal['status']='failed_or_incomplete_no_retry'
            try:
                terminal['outer_output_bytes_before_terminal']=supervisor.output_bytes(OUTPUT)
                save(OUTPUT,'terminal.json',terminal,io,supervisor)
            except BaseException as error:
                # Best effort independent filename; immutable intent remains authoritative if storage fails.
                fallback={'status':'terminal_publication_failed','error_type':type(error).__name__,'release_sha256':digest,
                    'archive_read_accounting_complete':False,'other_person_member_body_bytes_read':None,
                    'worker_pid':terminal.get('worker_pid'),'parent_reaped_worker':terminal.get('parent_reaped_worker'),'no_retry':True}
                io.atomic_preserve(OUTPUT/'terminal-publication-failure.json',encode(fallback))
                raise
            if stop['requested'] or time.monotonic()>=outer_deadline:
                io.atomic_preserve(OUTPUT/'post-publication-refusal.json',encode({
                    'status':'final_publication_termination_or_deadline','release_sha256':digest,
                    'termination_requested':stop['requested'],'terminal_is_not_execution_admission':True}))
                return 1
        return 0 if terminal['status']=='selected_header_intake_complete_unadmitted' else 1

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--release',type=Path,required=True);parser.add_argument('--release-sha256',required=True)
    parser.add_argument('--check-only',action='store_true');parser.add_argument('--worker',action='store_true');parser.add_argument('--intent-sha256')
    args=parser.parse_args();args.release=args.release.resolve()
    release=check_release(args.release,args.release_sha256,executing=not args.check_only)
    if args.check_only:
        print(json.dumps({'status':'source_and_release_preparation_valid','execution_released':release['execution_released'],
                         'archive_opens':0,'archive_prerequisites_checked':False}));return 0
    if args.worker:return worker(release,args.release,args.release_sha256,args.intent_sha256)
    return parent(release,args.release,args.release_sha256)
if __name__=='__main__':raise SystemExit(main())

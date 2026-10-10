"""One bounded generated opening saturation; no source arrays or models."""
from collections import Counter
from dataclasses import asdict
import gc
import hashlib
import json
import os
from pathlib import Path
import resource
import signal
import sys
import time

ROOT=Path(__file__).resolve().parents[3]
HERE=Path(__file__).resolve().parent
OUT=HERE/'native-run-v1'
sys.path.insert(0,str(ROOT/'src'))
START=time.monotonic()
COUNTS={'previews':0,'commits':0}
ARMS=[]


def save(name,value):
    temp=OUT/(name+'.tmp'); temp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');temp.replace(OUT/name)


def capped(signum=None,frame=None):
    raise TimeoutError('generated native worker wall cap')


def check():
    if time.monotonic()-START>=55: capped()
    if COUNTS['previews']>=3000: raise RuntimeError('aggregate preview cap')
    if COUNTS['commits']>=256: raise RuntimeError('aggregate commit cap')
    if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss>2*1024**3:raise MemoryError('worker memory cap')


def main():
    from resectionlab.legacy_transfer_worker import _require_parent_lease
    _require_parent_lease()
    OUT.mkdir(exist_ok=False)
    signal.signal(signal.SIGALRM,capped);signal.signal(signal.SIGTERM,capped);signal.setitimer(signal.ITIMER_REAL,55)
    result={'status':'started','counters':COUNTS,'arms':ARMS,'patient_array_reads':0,'model_calls':0}
    states=[]
    try:
        import numpy as np
        from resectionlab.core import array_digest
        from resectionlab.geometry import AccessWindow,GENERIC_TOOLS,capsule_voxel_indices
        from resectionlab.native_resection import NativeResectionConfig,NativeResectionEngine
        from resectionlab.native_proposals import DEFAULT_COLUMN_OFFSETS,NominalCavityProposalConfig,PreparedNominalCavityProposer
        # No file ingress after imported source and one saved metadata record.
        meta_path=HERE/'result.json'; raw=meta_path.read_bytes(); metadata=json.loads(raw)
        result['spacing_metadata_sha256']=hashlib.sha256(raw).hexdigest()
        result['tools']=[asdict(t) for t in GENERIC_TOOLS]
        def audit(event,args):
            if event=='open' and args and isinstance(args[0],(str,bytes)):
                p=Path(os.fsdecode(args[0])).resolve()
                if p.suffix.lower() in ('.npy','.npz','.dcm','.pt','.pth') or p.is_relative_to(ROOT/'data'):
                    raise PermissionError('generated-only file ingress')
            if event in ('subprocess.Popen','os.system','os.fork','socket.connect','socket.bind'):
                raise PermissionError('generated-only no external work')
        sys.addaudithook(audit)
        result['algorithm']={'families':['exposed_opening','intermediate_opening'],
            'nominal_target':'all zero, no proximal/distal nominal endpoints',
            'order':'interleaved complete passes across four arms; per arm family then declared column then canonical tool',
            'freshness':'repropose for each slot, fresh native preview and immediate authenticated commit only if feasible',
            'stop':'complete no-change pass per arm or aggregate caps; no episode reward/search/horizon',
            'diagnostic':'after saturation, noncommitting central source-cell-depth probes; count toward same preview cap',
            'limits':{'worker_seconds':55,'previews':3000,'commits':256,'rss_bytes':2*1024**3}}
        def preview(engine,tool,tip,entry):
            check();COUNTS['previews']+=1
            return engine.preview_stroke(tool,tip,entry_mm=entry)
        for name in ('ReMIND-008','ReMIND-020'):
            spacing=np.array(next(r['spacing_xyz_mm'] for r in metadata['original'] if r['name']==name))
            for variant,offsets in [('current13',DEFAULT_COLUMN_OFFSETS),('plus_nearest17',DEFAULT_COLUMN_OFFSETS+((-1,0),(1,0),(0,-1),(0,1)))]:
                sx,sy,sz=spacing;air=2;layers=int(np.ceil(10/sx));half=np.ceil(8/spacing[1:]).astype(int)
                shape=(air+layers+air,2*half[0]+1,2*half[1]+1)
                tissue=np.zeros(shape,bool);tissue[air:air+layers,:,:]=True
                zero=np.zeros(shape,np.float32); affine=np.diag([*spacing,1.]);affine[:3,3]=[-(air-.5)*sx,-half[0]*sy,-half[1]*sz]
                access=AccessWindow([0,0,0],[1,0,0],6.,'generated-same-radius-axis0')
                config=NativeResectionConfig(tissue,np.zeros(shape,np.int16),affine,access,GENERIC_TOOLS,
                    'generated:'+name+':'+variant,'explicit solid generated slab, no anatomy',
                    case_id='generated-'+name+'-'+variant,max_tip_step_mm=min(.25,float(spacing.min())/2))
                engine=NativeResectionEngine(config)
                rule=NominalCavityProposalConfig(offsets_source_voxels=offsets,intermediate_opening_mm=1.)
                assert rule.max_candidates==96
                provider=PreparedNominalCavityProposer(config,zero,nominal_provenance={
                    'source_hash':config.source_hash,'nominal_target_hash':array_digest(zero),
                    'source_kind':'derived_from_scan','derivation':'generated all-zero permitted goal; opening-only diagnostic'},config=rule)
                report={'arm':name+'-'+variant,'spacing_xyz_mm':spacing.tolist(),'offsets':offsets,
                    'shape':list(map(int,shape)),'solid_slab_depth_mm':float(layers*sx),
                    'transverse_edge_distance_from_center_mm':((half+.5)*spacing[1:]).tolist(),
                    'air_layers_each_end':air,'access_radius_mm':6.,'max_candidates':96,
                    'native_config_hash':config.fingerprint,'proposal_config_hash':rule.fingerprint,
                    'status':'pending','passes':[],'commits':[],'previews':0}
                ARMS.append(report);states.append((engine,provider,report,spacing,air,half))
        while any(r['status']!='saturated_declared_opening_order' for r in ARMS):
            for engine,provider,report,spacing,air,half in states:
                if report['status']=='saturated_declared_opening_order':continue
                check();before=int(engine.removed_mask.sum()); reasons=Counter();p0=COUNTS['previews'];c0=COUNTS['commits']
                for family in ('exposed_opening','intermediate_opening'):
                    for column_index in range(len(report['offsets'])):
                        for tool in GENERIC_TOOLS:
                            check()
                            batch=provider.propose(engine,cancelled=lambda:time.monotonic()-START>=55)
                            rays=[r for r in batch.proposals if r.family==family and r.column_index==column_index and r.tool_id==tool.tool_id]
                            if not rays:continue
                            ray=rays[0];cert=preview(engine,ray.tool_id,ray.tip_mm,ray.entry_mm)
                            report['previews']+=1;reasons[cert.reason]+=1
                            if cert.feasible:
                                check();engine.commit_preview(cert);COUNTS['commits']+=1
                                report['commits'].append({'family':family,'column':report['offsets'][column_index],
                                    'tool_id':tool.tool_id,'tip_depth_mm':float(ray.tip_mm[0]),
                                    'removed_count':len(cert.removed_indices_native),'state_hash':engine.state_hash})
                after=int(engine.removed_mask.sum())
                report['passes'].append({'pass':len(report['passes'])+1,'previews':COUNTS['previews']-p0,
                    'commits':COUNTS['commits']-c0,'new_removed':after-before,'removed_total':after,'reasons':reasons})
                report['status']='saturated_declared_opening_order' if after==before else 'changed'
                save('progress.json',{'elapsed_seconds':time.monotonic()-START,'counters':COUNTS,'arms':ARMS})
        for engine,provider,report,spacing,air,half in states:
            depths=(np.arange(air,engine.remaining_mask.shape[0]-air)-air+.5)*spacing[0]
            report['central_depth_probes']={}
            for tool in GENERIC_TOOLS:
                rows=[]
                for depth in depths:
                    cert=preview(engine,tool.tool_id,[float(depth),0,0],[0,0,0]);report['previews']+=1
                    rows.append({'depth_mm':float(depth),'feasible':cert.feasible,'reason':cert.reason,'would_remove':len(cert.removed_indices_native)})
                report['central_depth_probes'][tool.tool_id]=rows
                # Shaft geometry at a fixed4mm intended endpoint, measured on
                # saved closure. Native preview retains temporal authority.
                cells=capsule_voxel_indices(engine._cell_scene,np.array([-tool.working_length_mm,0,0]),np.array([4.-tool.tip_length_mm,0,0]),tool.shaft_radius_mm)
                blocked=cells[engine.remaining_mask[tuple(cells.T)]]
                report.setdefault('central_shaft_remaining_cells_at_4mm',{})[tool.tool_id]=blocked.tolist()
            report['removed_count']=int(engine.removed_mask.sum()); report['removal_closure_hash']=array_digest(engine.removed_mask)
            removed=np.argwhere(engine.removed_mask)
            report['maximum_removed_cell_center_depth_mm']=None if not len(removed) else float((removed[:,0].max()-air+.5)*spacing[0])
            report['central_removed_axial_indices']=np.flatnonzero(engine.removed_mask[:,half[0],half[1]]).tolist()
            report['final_state_hash']=engine.state_hash
        result['status']='all_declared_opening_orders_saturated'
    except BaseException as error:
        result.update(status='terminal_cap_or_failure',error=type(error).__name__+':'+str(error))
    finally:
        signal.setitimer(signal.ITIMER_REAL,0)
        # Retain closure even if a cap interrupts a pass or final probes.
        for engine,provider,report,spacing,air,half in states:
            report['removed_count']=int(engine.removed_mask.sum())
            report['removal_closure_hash']=array_digest(engine.removed_mask)
            report['final_state_hash']=engine.state_hash
        result.update(elapsed_seconds=time.monotonic()-START,peak_rss_bytes=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
            interpretation='Saturation only for stated generated slab/families/order. No exhaustive surgery or patient reachability claim.')
        save('result.json',result)
        states.clear();gc.collect()
    return 0 if result['status']=='all_declared_opening_orders_saturated' else 1


if __name__=='__main__':raise SystemExit(main())

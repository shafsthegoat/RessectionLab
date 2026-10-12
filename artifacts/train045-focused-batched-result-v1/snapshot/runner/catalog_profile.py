"""One predeclared same-object engineering profile, never a policy-quality test."""
from contextlib import contextmanager
import statistics,time
from comparison import plain
from common import semantic


def snapshot(catalog):
    return semantic(plain({'catalog_seal':catalog._seal,'current_digest':catalog._digest(),
        'observation':catalog.observation.fingerprint,'packets':catalog.packets,
        'pages':[{'state':p._engine.state_hash,'masks':p._engine.committed_mask_digests(),
            'history':p._history,'task_seal':p._state_seal,'page':p._active_page,
            'registry':sorted((str(k),id(v),p._engine._preview_digests[k]) for k,v in p._engine._preview_records.items())}
            for p in (catalog.work,*catalog.pages)]}))


def timing_gate(rows):
    if len(rows)!=4 or any(set(r['timings'])!={'reference','batched'} for r in rows):
        raise ValueError('Exactly four complete paired timings required')
    r=[v['timings']['reference']['wall_seconds'] for v in rows];b=[v['timings']['batched']['wall_seconds'] for v in rows]
    import math
    if any(type(t) not in (int,float) or not math.isfinite(t) or t<=0 for t in r+b):raise ValueError('Finite positive timings required')
    rm,bm=statistics.median(r),statistics.median(b);saving=statistics.median(a-c for a,c in zip(r,b))
    return {'reference_median_seconds':rm,'batched_median_seconds':bm,'median_paired_saving_seconds':saving,
        'median_ratio':bm/rm,'material_gate_passed':bm<=.8*rm and saving>=.25,
        'maximum_median_ratio':.8,'minimum_median_paired_saving_seconds':.25}


def profile_catalog(catalog,*,validate,guard,counters):
    from resectionlab.native_resection import NativeResectionEngine
    from resectionlab.native_spatial_task import NativeSpatialTask
    from resectionlab.focused_paged_task import FocusedPagedObstructionTask
    from resectionlab.spatial_policy import SpatialPolicy
    import focused_policy as adapter
    initial=snapshot(catalog);rows=[];calls={'state':0,'certificate':0,'score':0}
    result={'scope':'One current public catalog; engineering timing only, no clinical or policy-quality inference',
        'state_count':1,'distinct_actions':len(catalog.identifiers),'pages':len(catalog.pages),
        'orders':[['reference','batched'],['batched','reference'],['reference','batched'],['batched','reference']],
        'before_snapshot':initial,'pairs':rows,'integrity_passed':False}
    def forbidden(*a,**k):raise RuntimeError('Authentication attempted forbidden preview/transition/observation/forward')
    attrs=[(NativeSpatialTask,'_transition'),
        (NativeSpatialTask,'observation'),(SpatialPolicy,'forward'),(adapter,'build_spatial_observation'),
        (adapter,'_feature')]
    original_assert=FocusedPagedObstructionTask._assert_frozen
    original_digest=NativeResectionEngine._result_digest;original_score=NativeSpatialTask._score_record
    def counted_assert(self):calls['state']+=1;return original_assert(self)
    def counted_digest(value):calls['certificate']+=1;return original_digest(value)
    def counted_score(self,*a,**k):calls['score']+=1;return original_score(self,*a,**k)
    # Preserve the guarded preview entry exactly: cancellation may call budget.check.
    # Native preview work is charged normally and refused by exact counter equality.
    try:
        for pair,order in enumerate(result['orders']):
            item={'pair':pair+1,'order':order,'timings':{}};rows.append(item)
            for mode in order:
                guard();before=dict(counters());originals=[(obj,name,getattr(obj,name)) for obj,name in attrs]
                calls.update(state=0,certificate=0,score=0)
                try:
                    for obj,name,_ in originals:setattr(obj,name,forbidden)
                    FocusedPagedObstructionTask._assert_frozen=counted_assert
                    NativeResectionEngine._result_digest=staticmethod(counted_digest)
                    NativeSpatialTask._score_record=counted_score
                    wall=time.perf_counter();cpu=time.process_time()
                    returned=validate(catalog,mode=mode)
                    cpu=time.process_time()-cpu;wall=time.perf_counter()-wall
                finally:
                    FocusedPagedObstructionTask._assert_frozen=original_assert
                    NativeResectionEngine._result_digest=staticmethod(original_digest)
                    NativeSpatialTask._score_record=original_score
                    for obj,name,old in originals:setattr(obj,name,old)
                guard()
                if returned is not catalog or snapshot(catalog)!=initial or counters()!=before:
                    raise ValueError('Same-catalog validator changed state/certificates/work counters')
                item['timings'][mode]={'wall_seconds':wall,'cpu_seconds':cpu,'calls':dict(calls),
                    'zero_new_preview_transition_forward':True}
            expected=len(catalog.identifiers)-1
            if any(item['timings'][mode]['calls'][key]!=expected for mode in ('reference','batched') for key in ('certificate','score')):
                raise ValueError('Batched validator omitted certificate or score checks')
        result.update(integrity_passed=True,after_snapshot=snapshot(catalog),**timing_gate(rows))
        return result
    except BaseException as error:
        error.catalog_profile_accounting=result;raise

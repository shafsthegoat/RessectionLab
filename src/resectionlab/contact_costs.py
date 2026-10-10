"""Read-only call accounting around the owned one-thread native worker process.

Counters wrap existing implementations without changing inputs/results. Timings
are inclusive and must never be summed; complete wall time is authoritative.
"""
from contextlib import contextmanager
from functools import wraps
import resource
import sys
import time

from .goal_mode_spatial_policy import GoalModeSpatialPolicy
from .native_resection import NativeResectionEngine
from .native_spatial_task import NativeSpatialTask

_ACTIVE = False


def peak_rss_bytes():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == 'darwin' else 1024)


class ContactCostMeter:
    def __init__(self):
        self.phase = 'unclassified'; self.rows = {}; self.originals = []

    def __enter__(self):
        global _ACTIVE
        if _ACTIVE: raise RuntimeError('Cost meter requires an exclusive owned worker')
        _ACTIVE = True
        def wrap(cls, name, label, predicate=None):
            original = getattr(cls, name); self.originals.append((cls, name, original))
            @wraps(original)
            def counted(instance, *args, **kwargs):
                started = time.perf_counter(); row = self.rows.setdefault(self.phase, {})
                eligible = predicate is None or predicate(instance)
                if eligible: row[label + '_calls'] = row.get(label + '_calls', 0) + 1
                try:
                    result = original(instance, *args, **kwargs)
                    if eligible: row[label + '_returned_calls'] = row.get(label + '_returned_calls', 0)+1
                    return result
                except BaseException as error:
                    if eligible:
                        row[label + '_failed_calls'] = row.get(label + '_failed_calls', 0)+1
                        if label == 'native_transition' and getattr(error, 'committed', False):
                            row['durable_transition_interrupted_calls'] = row.get('durable_transition_interrupted_calls', 0)+1
                    raise
                finally:
                    if eligible: row[label + '_inclusive_seconds'] = row.get(label + '_inclusive_seconds', 0.) + time.perf_counter()-started
            setattr(cls, name, counted)
        wrap(NativeSpatialTask, '_prepare_inventory', 'inventory_request')
        wrap(NativeSpatialTask, '_prepare_inventory', 'inventory_build_or_terminal_empty',
             predicate=lambda task: task._inventory is None)
        wrap(NativeSpatialTask, '_transition', 'native_transition')
        wrap(NativeSpatialTask, 'clone', 'task_clone')
        wrap(NativeSpatialTask, 'independent_geometry_check', 'geometry_audit')
        wrap(NativeResectionEngine, 'preview_stroke', 'geometry_preview')
        wrap(NativeResectionEngine, 'commit_preview', 'native_nonstop_commit')
        wrap(GoalModeSpatialPolicy, 'forward', 'actor_forward')
        return self

    def __exit__(self, *unused):
        global _ACTIVE
        for cls, name, method in reversed(self.originals): setattr(cls, name, method)
        _ACTIVE = False

    @contextmanager
    def scope(self, phase):
        old = self.phase; self.phase = phase; start = time.perf_counter()
        try: yield
        finally:
            row = self.rows.setdefault(phase, {})
            row['complete_wall_seconds'] = row.get('complete_wall_seconds', 0.) + time.perf_counter()-start
            row['process_lifetime_peak_rss_bytes_so_far'] = peak_rss_bytes()
            row['timing_scope'] = 'inclusive_subcalls_not_additive; complete_wall_is_authoritative'
            self.phase = old

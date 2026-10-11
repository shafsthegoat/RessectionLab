"""Bounded independent saved-geometry diagnosis, not a native/model rerun.

For this already completed axis-aligned scene, the target portion of every
executed active capsule is a subset of the final endpoint sphere. Full-leaf
containment is the farthest corner's distance to that sphere center. Analytic
z-plane clipping is rectangular height times area; no producer geometry imports.
"""
from pathlib import Path
import hashlib
import json
import math
import os
import resource
import signal
import time

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent / 'saved-cap-result-01.json'
REPORT = ROOT / 'build/native-partial-volume-v1/run-01/report.json'
REPORT_SHA = 'c9265daaf186b025403ee499a333f0e2811feab66f65e10a98a3d97f17300ee9'
ROW = 'aligned-suction-0.25mm'


def digest(data):
    return hashlib.sha256(data).hexdigest()


def main():
    started = time.monotonic()
    signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError('30 s bound')))
    signal.alarm(30)
    result = {'schema': 'saved-aligned-target-cap-diagnosis-v1', 'status': 'started'}
    checks = 0

    def need(condition, name):
        nonlocal checks
        checks += 1
        if not condition:
            raise AssertionError(name)

    try:
        data = REPORT.read_bytes()
        need(digest(data) == REPORT_SHA, 'frozen report')
        r = next(x for x in json.loads(data)['rows'] if x['row_id'] == ROW)
        gref = r['partial_evacuation']['certified_geometry']
        gp = REPORT.parent / gref['path']
        gd = gp.read_bytes()
        need(digest(gd) == gref['sha256'] and len(gd) == gref['bytes'], 'checked sparse geometry bytes')
        envelope = json.loads(gd)
        g = envelope['geometry']
        need(envelope['row_id'] == ROW and envelope['source_hash'] == r['source_hash'], 'geometry source')
        need(envelope['history_hash'] == r['native']['committed_history_hash'], 'geometry history')
        need(g['geometry_digest'] == r['partial_evacuation']['checker_geometry_digest'], 'accepted geometry digest')
        need(r['independent_audit']['certificate']['feasible'] is True, 'executed history accepted')
        need(r['native']['complete_sequence'] is True and len(r['actions']) == 10, 'complete ten strokes')
        need(all(a['status'] == 'accepted' and a['failure_tip_mm'] is None for a in r['actions']), 'all planned strokes executed')
        for a, b in zip(r['actions'], r['actions'][1:]):
            need(a['result_state_hash'] == b['source_state_hash'], 'state chain')
        affine = r['affine']
        expected = [[.25, 0., 0., -5.5], [0., .25, 0., -5.5], [0., 0., .25, -5.5], [0., 0., 0., 1.]]
        need(affine == expected and g['native_affine'] == affine, 'aligned grid')
        need(g['cut_union_spec']['max_depth'] == 3, 'unchanged depth3')
        n = 8
        h = .25
        leaf = h/n
        origin = -5.5-h/2
        plane = 2.8
        tool = r['fixed_tool']
        radius = tool['tip_radius_mm']
        tip = r['actions'][-1]['tip_mm']
        need(radius == 1. and tip == [0., 0., 2.2], 'exact target sphere')
        need(all(a['tip_mm'][2]+radius < plane for a in r['actions'][:-1]), 'preparations cannot reach target')
        need(all(a['entry_mm'][:2] == a['tip_mm'][:2] for a in r['actions']), 'all executed axes vertical')
        need(plane > tip[2]+leaf, 'every target-crossing leaf entirely distal to capsule axis endpoint')
        penetration = tip[2]+radius-plane
        exact = math.pi*penetration**2*(radius-penetration/3)
        need(math.isclose(exact, r['exact_continuous_target_cap_mm3'], rel_tol=0, abs_tol=1e-14), 'executed analytic cap comparator')
        full = {tuple(p) for p in g['whole_removed_indices']}
        partial = {tuple(p['index']): int(p['covered_bits_hex'], 16) for p in g['partial_cells']}
        need(not full.intersection(partial), 'disjoint final full/partial ledger')
        fullmask = (1 << 512)-1

        def mask(p):
            return fullmask if p in full else partial.get(p, 0)

        def target_height(global_z):
            lo = origin+global_z*leaf
            return max(0., lo+leaf-max(lo, plane))

        # First recompute actual target volume directly from saved mask columns.
        actual_terms = []
        actual_target_leaves = 0
        for p in full | partial.keys():
            bits = mask(p)
            for lx in range(n):
                for ly in range(n):
                    for lz in range(n):
                        if bits >> (lx*n*n+ly*n+lz) & 1:
                            height = target_height(p[2]*n+lz)
                            if height:
                                actual_target_leaves += 1
                                actual_terms.append(leaf*leaf*height)
        actual = math.fsum(actual_terms)
        need(math.isclose(actual, r['partial_evacuation']['physical_target_evacuated_mm3'], rel_tol=0, abs_tol=2e-13), 'independent saved target volume')

        # Generous index box containing all target-intersecting sphere leaves.
        ranges = [range(math.floor((tip[d]-radius-origin)/leaf)-1,
                        math.ceil((tip[d]+radius-origin)/leaf)+1) for d in range(3)]
        ranges[2] = range(math.floor((plane-origin)/leaf)-1, ranges[2].stop)
        terms, missing, rows = [], [], {}
        candidates = contained = inside_saved = boundary_margin_effect = 0
        for ix in ranges[0]:
            xlo = origin+ix*leaf
            dx2 = max(abs(xlo-tip[0]), abs(xlo+leaf-tip[0]))**2
            for iy in ranges[1]:
                ylo = origin+iy*leaf
                dxy2 = dx2+max(abs(ylo-tip[1]), abs(ylo+leaf-tip[1]))**2
                for iz in ranges[2]:
                    height = target_height(iz)
                    if not height:
                        continue
                    candidates += 1
                    zhi = origin+(iz+1)*leaf
                    distance2 = dxy2+(zhi-tip[2])**2
                    geometric = distance2 <= (radius-1e-10)**2
                    boundary_margin_effect += int((distance2 <= radius**2) != geometric)
                    p = (ix//n, iy//n, iz//n)
                    bit = (ix % n)*n*n+(iy % n)*n+(iz % n)
                    present = bool(mask(p) >> bit & 1)
                    need(not present or geometric, 'every target-bearing saved leaf inside final executed capsule')
                    if not geometric:
                        continue
                    contained += 1
                    volume = leaf*leaf*height
                    terms.append(volume)
                    if present:
                        inside_saved += 1
                        continue
                    missing.append(volume)
                    rec = rows.setdefault(p, {'parent': list(p), 'missing_target_leaves': 0,
                                             'missing_target_mm3': 0., 'has_any_certified_bits': bool(mask(p)),
                                             'final_whole_parent_face_exposed': any(tuple(p[j]+(s if j == d else 0) for j in range(3)) in full
                                                                                   for d in range(3) for s in (-1, 1))})
                    rec['missing_target_leaves'] += 1
                    rec['missing_target_mm3'] += volume
        ceiling = math.fsum(terms)
        missing_volume = math.fsum(missing)
        need(inside_saved == actual_target_leaves, 'all saved target leaves covered by enumeration')
        need(math.isclose(ceiling-actual, missing_volume, rel_tol=0, abs_tol=2e-13), 'gap partition')
        need(actual <= ceiling <= exact, 'geometric and actual bounds')
        by_exposure = {str(flag): {'parents': sum(q['final_whole_parent_face_exposed'] == flag for q in rows.values()),
                                 'missing_target_mm3': math.fsum(q['missing_target_mm3'] for q in rows.values() if q['final_whole_parent_face_exposed'] == flag)}
                       for flag in (False, True)}
        result.update(status='completed_saved_geometry_only', row_id=ROW,
            inputs={'report': {'path': str(REPORT.relative_to(ROOT)), 'sha256': REPORT_SHA},
                    'geometry': {'path': str(gp.relative_to(ROOT)), 'sha256': gref['sha256']},
                    'native_resection_source_sha256': digest((ROOT/'src/resectionlab/native_resection.py').read_bytes()),
                    'native_cut_union_source_sha256': digest((ROOT/'src/resectionlab/native_cut_union.py').read_bytes())},
            complete_executed_strokes=10, final_target_stroke=r['actions'][-1],
            target_plane_z_mm=plane, leaf_side_mm=leaf,
            exact_executed_target_cap_mm3=exact,
            actual_certified_target_mm3=actual,
            final_capsule_contained_leaf_target_ceiling_mm3=ceiling,
            residual_total_mm3=exact-actual,
            depth3_full_leaf_boundary_loss_mm3=exact-ceiling,
            contained_but_uncertified_target_mm3=missing_volume,
            geometric_loss_fraction_of_residual=(exact-ceiling)/(exact-actual),
            uncertified_contained_fraction_of_residual=missing_volume/(exact-actual),
            final_exposure_partition=by_exposure,
            missing_parents=list(rows.values()),
            work={'candidate_target_leaves': candidates, 'contained_target_leaves': contained,
                  'actual_target_leaves': actual_target_leaves, 'radius_margin_changed_leaf_tests': boundary_margin_effect},
            interpretation='Evaluator-only optimistic leaf coverage: removes exposure eligibility but changes neither actual saved geometry nor credited outcome. Final exposure is retrospective, not a reconstructed microstep eligibility certificate.',
            no_model_or_native_calls=True)
    except BaseException as error:
        result.update(status='failed', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        result.update(checks=checks, seconds=time.monotonic()-started,
                      peak_process_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                      script_sha256=digest(Path(__file__).read_bytes()),
                      thread_environment={k: os.environ.get(k) for k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS')})
        with OUT.open('x') as f:
            json.dump(result, f, indent=2, allow_nan=False)
            f.write('\n')
        signal.alarm(0)
        print(json.dumps({k: result[k] for k in ('status', 'checks', 'seconds')}))


if __name__ == '__main__':
    main()

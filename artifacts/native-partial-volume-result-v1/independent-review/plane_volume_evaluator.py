"""Evaluator-only affine-box clipping of an independently certified leaf union.

No actor, producer, native geometry, target-sampling or NumPy imports. Analytic
planes are reference truth for the fixed generated experiment only. Whole parent
boxes are counted once; leaf traversal is limited to plane-straddling partial
parents. Coverage proof/exposure/shaft ordering must be verified separately.
"""
from itertools import product
import hashlib
import json
import math
import struct

CORNERS = tuple(product((0, 1), repeat=3))
FACES = ((0, 1, 3, 2), (4, 6, 7, 5), (0, 4, 5, 1),
         (2, 3, 7, 6), (0, 2, 6, 4), (1, 5, 7, 3))


class EvaluationLimit(ValueError):
    """No complete physical-volume result may be claimed after this refusal."""


def _det(a, b, c):
    return (a[0]*(b[1]*c[2]-b[2]*c[1])
            - a[1]*(b[0]*c[2]-b[2]*c[0])
            + a[2]*(b[0]*c[1]-b[1]*c[0]))


def _sub(a, b):
    return tuple(x-y for x, y in zip(a, b))


def float32_fraction_enclosure(value):
    """Enclose the pre-rounding [0,1] value for stored IEEE binary32 nearest f.

    Midpoints include both ties, then expand one float64 ULP outward. This is a
    quantization interval, not a tuned physical tolerance. It does not claim the
    upstream analytic fraction calculation was itself exact arithmetic.
    """
    f = float(value)
    if not math.isfinite(f) or not 0 <= f <= 1:
        raise ValueError('Finite declared fractional membership required')
    if f == 0.: f = 0.  # Normalize signed zero before IEEE neighbor ordering.
    bits = struct.unpack('!I', struct.pack('!f', f))[0]
    if struct.unpack('!f', struct.pack('!I', bits))[0] != f:
        raise ValueError('Expected an exactly representable stored float32')
    previous = struct.unpack('!f', struct.pack('!I', bits-1))[0] if f else 0.
    following = struct.unpack('!f', struct.pack('!I', bits+1))[0]
    return (max(0., math.nextafter((previous+f)/2, -math.inf)),
            min(1., math.nextafter((f+following)/2, math.inf)))


def target_allocation_interval(parent_volume, evacuated_source_volume, mass_bounds):
    """Sharp cumulative allocation bound; never multiply leaf volume by f."""
    v, r = float(parent_volume), float(evacuated_source_volume)
    lo, hi = map(float, mass_bounds)
    if not all(math.isfinite(x) for x in (v, r, lo, hi)) or not 0 <= r <= v or not 0 <= lo <= hi <= v:
        raise ValueError('Incoherent parent/source-volume/target-mass declaration')
    return max(0., lo-(v-r)), min(r, hi)


def _affine(value):
    a = tuple(tuple(float(x) for x in row) for row in value)
    if len(a) != 4 or any(len(row) != 4 for row in a) or not all(math.isfinite(x) for row in a for x in row) or a[3] != (0., 0., 0., 1.):
        raise ValueError('Finite affine with homogeneous last row required')
    determinant = _det(a[0][:3], a[1][:3], a[2][:3])
    v = abs(determinant)
    if not math.isfinite(v) or v <= 0:
        raise ValueError('Nonsingular physical cell required')
    return a, v


def _vertices(a, center, scale):
    return tuple(tuple(math.fsum(a[d][j]*(center[j]+scale*(c[j]-.5)) for j in range(3))+a[d][3]
                       for d in range(3)) for c in CORNERS)


def _clip_volume(vertices, volume, plane):
    """Clip six convex faces, add horizontal cap, triangulate about interior.

    Every face triangle forms a tetrahedron with a common interior point. The
    absolute determinant sum avoids depending on imported face orientation.
    This is independent of the existing weighted-uniform-sum CDF integration.
    """
    if min(p[2] for p in vertices) >= plane:
        return volume
    if max(p[2] for p in vertices) <= plane:
        return 0.
    faces, crossings = [], set()
    for ids in FACES:
        polygon = [vertices[i] for i in ids]
        output = []
        for start, end in zip(polygon, polygon[1:]+polygon[:1]):
            inside_start, inside_end = start[2] >= plane, end[2] >= plane
            if inside_start:
                output.append(start)
            if inside_start != inside_end:
                # Same undirected edge always uses same arithmetic ordering.
                low, high = sorted((start, end), key=lambda p: (p[2], p[0], p[1]))
                t = (plane-low[2])/(high[2]-low[2])
                cross = (low[0]+t*(high[0]-low[0]), low[1]+t*(high[1]-low[1]), plane)
                output.append(cross); crossings.add(cross)
        if len(output) >= 3:
            faces.append(output)
    if len(crossings) >= 3:
        center = tuple(math.fsum(p[d] for p in crossings)/len(crossings) for d in range(3))
        faces.append(sorted(crossings, key=lambda p: math.atan2(p[1]-center[1], p[0]-center[0])))
    unique = set(p for face in faces for p in face)
    if len(unique) < 4:
        return 0.
    center = tuple(math.fsum(p[d] for p in unique)/len(unique) for d in range(3))
    tetrahedra = []
    for face in faces:
        for i in range(1, len(face)-1):
            tetrahedra.append(abs(_det(_sub(face[0], center), _sub(face[i], center), _sub(face[i+1], center)))/6)
    clipped = math.fsum(tetrahedra)
    if not math.isfinite(clipped) or clipped < 0 or clipped > volume*(1+2e-12):
        raise ValueError('Clipping volume outside finite parent box')
    return min(volume, clipped)


def affine_box_halfspace_volume(affine, center, scale, plane_z_mm):
    a, v = _affine(affine)
    if not (math.isfinite(scale) and 0 < scale <= 1 and math.isfinite(plane_z_mm)):
        raise ValueError('Finite positive box scale and plane required')
    return _clip_volume(_vertices(a, tuple(center), scale), v*scale**3, plane_z_mm)


def evaluate_certified_union(affine, max_depth, removed_indices, partial_cells,
                             target_fraction_bounds, *, tissue_plane_z_mm=0.,
                             target_plane_z_mm=2.8, cancelled=None,
                             max_parent_cells=65536, max_partial_cells=4096,
                             max_leaf_visits=2097152, max_clips=131072):
    """Summarize checked final full parents + final partial masks, without IDs.

    partial_cells maps (i,j,k) to nonzero int bitset. Full masks must be retired
    into removed_indices. Bounds mapping covers precisely these parents, with
    fractions [lo,hi] in [0,1]; it contains NO subcell reference truth. Returned
    physical target containment is an outcome, not a source assertion.
    """
    a, v = _affine(affine)
    if type(max_depth) is not int or not 1 <= max_depth <= 3:
        raise ValueError('Declared fixed partition depth required')
    if not all(type(x) is int and x > 0 for x in (max_parent_cells, max_partial_cells, max_leaf_visits, max_clips)):
        raise ValueError('Explicit finite work budgets required')
    if not all(math.isfinite(x) for x in (tissue_plane_z_mm, target_plane_z_mm)) or target_plane_z_mm < tissue_plane_z_mm:
        raise ValueError('Evaluator target plane must be nested inside tissue')
    n = 2**max_depth; leaves = n**3; full = (1 << leaves)-1
    removed = [tuple(x) for x in removed_indices]; partial = {tuple(k): b for k, b in partial_cells.items()}
    if len(removed) != len(set(removed)) or set(removed) & set(partial):
        raise ValueError('Duplicate full/partial parent geometry')
    if len(removed)+len(partial) > max_parent_cells or len(partial) > max_partial_cells:
        raise EvaluationLimit('Parent/partial resident budget')
    keys = set(removed) | set(partial)
    if set(target_fraction_bounds) != keys:
        raise ValueError('Explicit target mass bounds for exactly all evacuated parents required')
    if any(len(k) != 3 or any(type(i) is not int or i < 0 for i in k) for k in keys):
        raise ValueError('Nonnegative integral parent index required')
    if any(type(b) is not int or not 0 < b < full for b in partial.values()):
        raise ValueError('Only proper nonempty partial bitsets allowed')
    counts = {'full_parent_boxes':len(removed), 'partial_parent_boxes':len(partial),
              'partial_leaf_bits':sum(b.bit_count() for b in partial.values()),
              'leaf_visits':0, 'face_clips':0, 'one_sided_partial_plane_counts':0}
    def check():
        if cancelled is not None and cancelled():
            raise EvaluationLimit('Evaluator cancelled; no complete volume claim')
    def clipped(center, scale, plane):
        vertices = _vertices(a, center, scale)
        low, high = min(p[2] for p in vertices), max(p[2] for p in vertices)
        volume = v*scale**3
        if low >= plane:return volume
        if high <= plane:return 0.
        if counts['face_clips'] >= max_clips:raise EvaluationLimit('Independent clipping budget')
        counts['face_clips'] += 1
        return _clip_volume(vertices, volume, plane)
    source, physical, target, lower, upper, full_physical = [], [], [], [], [], []
    serialized = []; mass_violations = []; mass_violation_count = 0
    for q, key in enumerate(sorted(keys)):
        if q % 64 == 0:check()
        mask = partial.get(key, full); amount = v*(mask.bit_count()/leaves)
        lo, hi = map(float, target_fraction_bounds[key])
        if not all(math.isfinite(x) for x in (lo, hi)) or not 0 <= lo <= hi <= 1:
            raise ValueError('Invalid declared fractional mass bounds')
        interval = target_allocation_interval(v, amount, (v*lo, v*hi))
        # Independent parent mass check makes the physical-target interval premise explicit.
        analytic_parent_target = clipped(key, 1., target_plane_z_mm)
        violation = max(0., v*lo-analytic_parent_target, analytic_parent_target-v*hi)
        mass_violations.append(violation)
        mass_violation_count += violation > 64*math.ulp(max(1.,v))
        source.append(amount);lower.append(interval[0]);upper.append(interval[1])
        serialized.append([list(key),format(mask,'x')])
        if mask == full:
            tissue = clipped(key, 1., tissue_plane_z_mm)
            physical.append(tissue);full_physical.append(tissue)
            target.append(analytic_parent_target)
            continue
        corners = _vertices(a,key,1.); low=min(p[2] for p in corners);high=max(p[2] for p in corners)
        values = []; mixed = []
        for plane in (tissue_plane_z_mm,target_plane_z_mm):
            if low >= plane:values.append(amount);counts['one_sided_partial_plane_counts']+=1
            elif high <= plane:values.append(0.);counts['one_sided_partial_plane_counts']+=1
            else:values.append(None);mixed.append(plane)
        if mixed:
            per_plane={plane:[] for plane in mixed};bits=mask
            while bits:
                if counts['leaf_visits'] >= max_leaf_visits:raise EvaluationLimit('Independent leaf traversal budget')
                counts['leaf_visits']+=1
                if counts['leaf_visits'] % 256 == 0:check()
                bit=bits & -bits;leaf=bit.bit_length()-1;bits^=bit
                offset=(leaf//(n*n),(leaf//n)%n,leaf%n)
                center=tuple(key[j]-.5+(offset[j]+.5)/n for j in range(3))
                for plane in mixed:per_plane[plane].append(clipped(center,1/n,plane))
            values=[math.fsum(per_plane[p]) if x is None else x for p,x in zip((tissue_plane_z_mm,target_plane_z_mm),values)]
        physical.append(values[0]);target.append(values[1])
    check()
    src,phys,targ,lo,hi=map(math.fsum,(source,physical,target,lower,upper))
    tolerance=64*math.ulp(max(1.,src))
    if not -tolerance<=targ<=phys+tolerance or not phys<=src+tolerance:
        raise ValueError('Nested evaluator/reference volume ordering failed')
    return {'schema':'independent-plane-evacuated-union-v1','complete':True,
        'source_evacuated_mm3':src,'physical_tissue_evacuated_lower_mm3':phys,
        'physical_target_evacuated_mm3':targ,'whole_cell_physical_lower_mm3':math.fsum(full_physical),
        'source_outer_voxelization_excess_mm3':src-phys,
        'cumulative_target_interval_source_mm3':[lo,hi],
        'physical_target_within_source_interval':lo-tolerance<=targ<=hi+tolerance,
        'parent_target_mass_enclosure_pass':mass_violation_count==0,
        'parent_target_mass_enclosure_violation_count':mass_violation_count,
        'maximum_parent_target_mass_enclosure_violation_mm3':max(mass_violations,default=0.),
        'arithmetic_comparison_tolerance_mm3':tolerance,'work':counts,
        'certified_union_sha256':hashlib.sha256(json.dumps(serialized,separators=(',',':')).encode()).hexdigest(),
        'scope':'Evaluator-only planes; certified subset of idealized evacuation, not measured biological removal. Source target intervals use source R, not clipped physical R.'}

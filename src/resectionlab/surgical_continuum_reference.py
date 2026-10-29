"""Offline, fixed-topology contact-compliance reference for the voxel continuum.

With K_j = alpha_j K_0, zero initial fields, fixed contacts and no body load,
all displacement/memory fields remain in span(K_0^-1 C^T). This reduction is
algebraic, not a learned or approximate rollout. It is not a live engine mode:
changing topology, nonproportional spectra or unobserved supports needs another
model. Large reference domains never enlarge the live engine's cell/DOF caps.
"""
from __future__ import annotations

from dataclasses import dataclass
import heapq
import time

import numpy as np
from scipy import sparse
from scipy.linalg import lu_factor, lu_solve
from scipy.sparse.linalg import splu

from . import surgical_continuum as continuum
from . import surgical_deformation_v2 as brain


def source_domain(engine, radius_mm, *, max_cells):
    if type(radius_mm) not in (int, float) or not np.isfinite(radius_mm) or radius_mm <= 0:
        raise ValueError('Positive physical reference radius required')
    if type(max_cells) is not int or max_cells < 1: raise ValueError('Explicit source-cell cap required')
    removed = set(engine.state['removed']) | brain.rigid_cells(engine)
    seeds = sorted({c for h in engine.state['holds'].values() if 'target' in h for c in h['cells']})
    if not seeds or any(c in removed or brain._material(engine, c) not in continuum.SOFT for c in seeds):
        raise ValueError('Retained soft source contacts required')
    if len(seeds)>max_cells: raise ValueError('REFERENCE_SOURCE_CELL_CAP: contact seeds exceed domain budget')
    blocked = brain._blocked(engine); distance = {c: 0. for c in seeds}; queue = [(0., c) for c in seeds]
    heapq.heapify(queue)
    while queue:
        d, cell = heapq.heappop(queue)
        if d != distance[cell]: continue
        for other, length in brain._neighbors(engine, cell):
            next_d = d + length
            if (next_d > radius_mm or other in removed or tuple(sorted((cell, other))) in blocked
                    or brain._material(engine, other) not in continuum.SOFT): continue
            if next_d < distance.get(other, float('inf')):
                if other not in distance and len(distance) >= max_cells:
                    raise ValueError('REFERENCE_SOURCE_CELL_CAP: no truncated reference domain')
                distance[other] = next_d; heapq.heappush(queue, (next_d, other))
    return sorted(distance)


def equilibrium_matrix(scene, mesh, materials, *, check=lambda: None, batch_elements=256):
    """Bound temporary COO storage; do not alter canonical engine summation."""
    shear, lame, _ = continuum.element_operators(scene.affine)
    shape = (len(mesh.nodes)*3,)*2; matrix = sparse.csr_matrix(shape)
    for start in range(0, len(mesh.cells), batch_elements):
        check(); cells = mesh.cells[start:start+batch_elements]; dofs = mesh.element_dofs[start:start+batch_elements]
        m = [materials[str(int(scene.labels.flat[c]))] for c in cells]
        mu = np.array([r['shearPa'] for r in m])*1e-6
        nu = np.array([r['poissonRatio'] for r in m]); lam = 2*mu*nu/(1-2*nu)
        values = mu[:, None, None]*shear + lam[:, None, None]*lame
        rows = np.repeat(dofs, 24, axis=1).ravel(); cols = np.tile(dofs, (1, 24)).ravel()
        matrix = matrix + sparse.coo_matrix((values.ravel(), (rows, cols)), shape=shape).tocsr()
    return matrix


def proportional_spectrum(materials, times):
    if any(type(t) not in (int,float) or not np.isfinite(t) or t <= 0 for t in times):
        raise ValueError('Explicit positive relaxation times required')
    ratios = []
    for m in materials.values():
        mu = m['shearPa']; values = m['relaxationShearPa']
        if mu <= 0 or len(values) != len(times): raise ValueError('Consistent positive equilibrium material required')
        ratios.append(np.asarray(values, float)/mu)
    if not ratios or any(not np.isfinite(v).all() or (v < 0).any() for v in ratios):
        raise ValueError('Finite nonnegative branch ratios required')
    if any(not np.allclose(v, ratios[0], rtol=1e-13, atol=0.) for v in ratios[1:]):
        raise ValueError('REFERENCE_NONPROPORTIONAL_SPECTRUM: a contact-only reduction would lose distributed history')
    return ratios[0]


@dataclass
class ContactReference:
    mesh: continuum.Mesh
    stiffness: sparse.csr_matrix
    contact: sparse.csr_matrix
    free: np.ndarray
    basis: np.ndarray
    compliance: np.ndarray
    spring: np.ndarray
    contact_cells: tuple
    alpha: np.ndarray
    times: np.ndarray
    diagnostics: dict


def build_reference(engine, cells, *, max_dofs, check=lambda: None):
    engine.scene.require_cache_source()
    p = engine.scene.properties[continuum.PROPERTY]
    if 'patientPosition' in engine.state: raise ValueError('REFERENCE_BODY_LOAD_UNSUPPORTED')
    state = engine.state.get(continuum.PROPERTY, {})
    for field in [state.get('cornerDisplacementMm', {}), *state.get('memoryDisplacementMm', [])]:
        if any(np.any(np.asarray(v) != 0.) for v in field.values()):
            raise ValueError('REFERENCE_REQUIRES_ZERO_INITIAL_FIELDS: arbitrary history is not contact-only')
    if any(np.any(np.asarray(v) != 0.) for v in engine.state['deformation'].values()):
        raise ValueError('REFERENCE_REQUIRES_ZERO_INITIAL_FIELDS')
    if any('target' not in h for h in engine.state['holds'].values()): raise ValueError('REFERENCE_RIGID_HOLD_UNSUPPORTED')
    if set(cells) & (set(engine.state['removed']) | brain.rigid_cells(engine)):
        raise ValueError('Reference cells must be retained source tissue')
    alpha = proportional_spectrum(p['materials'], p['relaxationSeconds'])
    blocked = brain._blocked(engine); mesh = continuum.voxel_mesh(engine.scene, cells, blocked, max_dofs=max_dofs)
    active = set(mesh.cells); held = {c for h in engine.state['holds'].values() for c in h['cells']}
    if not held or not held <= active: raise ValueError('Every declared contact needs reference support')
    anchors = {c for c in active if c not in held and any(n not in active and n not in engine.state['removed']
        and brain._material(engine,n) in continuum.SOFT and tuple(sorted((c,n))) not in blocked
        for n,_ in brain._neighbors(engine,c))}
    fixed = continuum._fixed_nodes(engine, mesh, anchors, blocked)
    # K_0 itself must be supported; springs cannot invent skull fixation.
    continuum.check_support(mesh, fixed, set())
    free = np.ones(len(mesh.nodes)*3, bool)
    for node in fixed: free[3*node:3*node+3] = False
    indexes = {c:i for i,c in enumerate(mesh.cells)}; contact_cells = []; k = []
    for key, h in sorted(engine.state['holds'].items()):
        for cell in sorted(h['cells']):
            contact_cells.append((key,cell)); k.append(p['handleStiffnessNPerMm']/len(h['cells']))
    if len(k)*3 > 64: raise ValueError('REFERENCE_CONTACT_RANK_CAP')
    rows=[]; cols=[]
    for i,(_,cell) in enumerate(contact_cells):
        for axis in range(3):
            rows.extend([3*i+axis]*8); cols.extend(3*mesh.element_nodes[indexes[cell]]+axis)
    contact = sparse.coo_matrix((np.full(len(rows),1/8.),(rows,cols)),shape=(len(k)*3,len(free))).tocsr()
    started=time.perf_counter(); matrix=equilibrium_matrix(engine.scene,mesh,p['materials'],check=check)
    check(); kff=matrix[free][:,free].tocsc()
    factor=splu(kff,permc_spec='MMD_AT_PLUS_A',options={'SymmetricMode':True})
    check(); basis=np.zeros((len(free),contact.shape[0])); basis[free]=factor.solve(contact[:,free].T.toarray())
    compliance=np.asarray(contact@basis)
    residual=float(np.max(np.abs(kff@basis[free]-contact[:,free].T.toarray()),initial=0.))
    asymmetry=float(np.max(np.abs(compliance-compliance.T),initial=0.))
    if not np.isfinite(basis).all() or residual > 1e-9 or asymmetry > 1e-8:
        raise ValueError('REFERENCE_COMPLIANCE_RESIDUAL')
    diagnostics={'freeDofs':int(free.sum()),'fullDofs':len(free),'contactCoordinates':contact.shape[0],
        'stiffnessNnz':matrix.nnz,'factorNnz':factor.L.nnz+factor.U.nnz,
        'basisResidualN':residual,'complianceAsymmetryMmPerN':asymmetry,
        'buildWallSeconds':time.perf_counter()-started}
    return ContactReference(mesh,matrix,contact,free,basis,compliance,np.repeat(k,3),tuple(contact_cells),
                            alpha,np.asarray(p['relaxationSeconds']),diagnostics)


def _schedule(reference, targets, h):
    values=np.asarray(targets)
    if (values.dtype.kind not in 'fiu' or values.ndim!=2 or values.shape[1]!=len(reference.spring)
            or not 1<=len(values)<=10000 or not np.isfinite(values).all()
            or type(h) not in (int,float) or not np.isfinite(h) or h<=0):
        raise ValueError('Finite, bounded contact-target history and physical step required')
    return values.astype(float,copy=False)


def integrate(reference, targets, h, *, check=lambda:None):
    targets=_schedule(reference,targets,h); m=len(reference.spring); b=len(reference.alpha)
    r=reference.times/(reference.times+h); beta=1+float(r@reference.alpha)
    a=beta*np.eye(m)+reference.spring[:,None]*reference.compliance
    factor=lu_factor(a); q=np.zeros(m); z=np.zeros((b,m)); coordinates=[]; memories=[]
    max_residual=0.; max_balance=0.; dissipated=0.; numerical=0.
    s=reference.compliance; d=reference.spring
    for target in targets:
        check(); previous=q.copy(); old_z=z.copy()
        rhs=d*target+np.sum((r*reference.alpha)[:,None]*z,axis=0)
        q=lu_solve(factor,rhs); z=(reference.times[:,None]*z+h*q)/(reference.times[:,None]+h)
        delta=q-previous; cq=s@q; cp=s@previous
        energy=.5*q@(s@q)+.5*cq@(d*cq); old_energy=.5*previous@(s@previous)+.5*cp@(d*cp)
        viscous=0.; loss=.5*delta@(s@delta)+.5*(cq-cp)@(d*(cq-cp))
        for j,alpha in enumerate(reference.alpha):
            stretch=q-z[j]; old_stretch=previous-old_z[j]; dz=z[j]-old_z[j]
            energy+=.5*alpha*stretch@(s@stretch); old_energy+=.5*alpha*old_stretch@(s@old_stretch)
            viscous+=alpha*reference.times[j]/h*dz@(s@dz)
            loss+=.5*alpha*(stretch-old_stretch)@(s@(stretch-old_stretch))
        balance=target@(d*(cq-cp))-(energy-old_energy)-viscous-loss
        max_residual=max(max_residual,float(np.max(np.abs(a@q-rhs),initial=0.)))
        max_balance=max(max_balance,abs(float(balance))); dissipated+=float(viscous); numerical+=float(loss)
        if not np.isfinite(q).all() or max_residual>1e-9 or max_balance>1e-8:
            raise ValueError('REFERENCE_REDUCED_STEP_RESIDUAL')
        coordinates.append(q.copy()); memories.append(z.copy())
    return {'coordinates':np.asarray(coordinates),'memoryCoordinates':np.asarray(memories),
        'maxCoefficientResidualN':max_residual,'maxEnergyBalanceResidualNmm':max_balance,
        'dissipatedEnergyNmm':dissipated,'numericalDissipationNmm':numerical}


def prepare_full(reference, h):
    if len(reference.free)>8000: raise ValueError('REFERENCE_DETAILED_CONTROL_CAP')
    if type(h) not in (int,float) or not np.isfinite(h) or h<=0: raise ValueError('Positive detailed step required')
    r=reference.times/(reference.times+h); beta=1+float(r@reference.alpha)
    c=reference.contact; a=beta*reference.stiffness+c.T@sparse.diags(reference.spring)@c
    factor=splu(a[reference.free][:,reference.free].tocsc(),permc_spec='MMD_AT_PLUS_A',options={'SymmetricMode':True})
    return {'reference':reference,'h':h,'factor':factor}


def integrate_full(reference, targets, h, *, check=lambda:None, prepared=None):
    """Detailed disagreement control, bounded to small reference domains."""
    targets=_schedule(reference,targets,h); n=len(reference.free); b=len(reference.alpha)
    if n>8000 or len(targets)*n*(b+1)*8>64*1024**2:
        raise ValueError('REFERENCE_DETAILED_CONTROL_CAP')
    r=reference.times/(reference.times+h)
    c=reference.contact; k=reference.stiffness; free=reference.free
    prepared=prepare_full(reference,h) if prepared is None else prepared
    if prepared.get('reference') is not reference or prepared.get('h')!=h:
        raise ValueError('Detailed factorization belongs to another reference or physical step')
    factor=prepared['factor']
    u=np.zeros(n); z=np.zeros((b,n)); fields=[]; memories=[]
    for target in targets:
        check(); rhs=np.asarray(c.T@(reference.spring*target)).ravel()
        for j,alpha in enumerate(reference.alpha): rhs+=r[j]*alpha*(k@z[j])
        u=np.zeros(n); u[free]=factor.solve(rhs[free]); z=(reference.times[:,None]*z+h*u)/(reference.times[:,None]+h)
        fields.append(u.copy()); memories.append(z.copy())
    return {'displacement':np.asarray(fields),'memories':np.asarray(memories)}

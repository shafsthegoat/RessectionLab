"""Offline exact exterior condensation for proportional Maxwell equations.

Inputs contain already supported free DOFs. This module neither changes live
domains nor transfers source history between meshes. Episode fields are caller
owned; only fixed operators/factors are retained. No contact-created support,
history projection, matrix symmetrization or approximate modal basis is used.
"""
from __future__ import annotations

import time
import weakref

import numpy as np
from scipy import sparse
from scipy.linalg import cho_factor, cho_solve, lu_factor, lu_solve
from scipy.sparse.linalg import splu

EPS = 64 * np.finfo(float).eps


def _finite(value, shape, name):
    a = np.asarray(value)
    if a.dtype.kind not in 'fiu' or a.shape != shape or not np.isfinite(a).all():
        raise ValueError('EXTERIOR_' + name)
    return a.astype(float, copy=False)


def _same_sparse_bits(left, right):
    a = left.tocsr(copy=True); b = right.tocsr(copy=True)
    for matrix in (a,b):matrix.sum_duplicates();matrix.eliminate_zeros();matrix.sort_indices()
    return (a.shape == b.shape and np.array_equal(a.indptr,b.indptr)
            and np.array_equal(a.indices,b.indices)
            and np.array_equal(a.data.view(np.uint64),b.data.view(np.uint64)))


def source_stiffness(geometry, *, source_order=None, check=lambda: None):
    """Assemble the declared source-cell equilibrium law in bounded batches.

    These are already exposed offline export arrays, not a new image adapter.
    Verify each input's own source-affine corner identity before using its
    mechanical coefficients. No unknown material or support is supplied.

    An optional frozen full-source order pins the 256-owner assembly buckets
    across deletions, with explicit source-owner reduction of duplicate matrix
    entries inside each bucket. Unchanged block contributions then retain their
    summation order instead of shifting when a different block loses an owner.
    It neither keeps removed stiffness nor certifies factor reuse. The default
    assembly path remains identical to the earlier fixed-domain witness.
    """
    from .surgical_continuum_partition import _validate
    from .surgical_continuum import CORNERS, element_operators
    cells, shape, nodes, elements, affine, free = _validate(geometry)
    ijk = np.column_stack(np.unravel_index(cells, tuple(shape)))
    expected = (ijk[:, None, :]+CORNERS-.5)@affine[:3, :3].T+affine[:3, 3]
    if (not np.isfinite(expected).all() or np.max(np.abs(expected-nodes[elements]), initial=0.) > 1e-10
            or np.linalg.det(affine[:3, :3]) == 0):
        raise ValueError('EXTERIOR_SOURCE_AFFINE_CORNERS')
    mu = _finite(geometry['muPa'], (len(cells),), 'SOURCE_MU')
    nu = _finite(geometry['nu'], (len(cells),), 'SOURCE_NU')
    if np.any(mu <= 0) or np.any(nu < 0) or np.any(nu > .45):
        raise ValueError('EXTERIOR_SOURCE_MATERIAL')
    shear, lame, _ = element_operators(affine)
    matrix = sparse.csr_matrix((nodes.size, nodes.size))
    if source_order is None:
        batches = (slice(start,start+256) for start in range(0,len(cells),256))
    else:
        order = np.asarray(source_order)
        if (order.ndim != 1 or order.dtype.kind not in 'iu' or not len(cells) <= len(order) <= 16000
                or np.any(order < 0) or np.any(order >= np.prod(shape)) or np.any(order[1:] <= order[:-1])):
            raise ValueError('EXTERIOR_SOURCE_ASSEMBLY_ORDER')
        positions = np.searchsorted(order,cells)
        if np.any(positions >= len(order)) or not np.array_equal(order[positions],cells):
            raise ValueError('EXTERIOR_SOURCE_ASSEMBLY_ORDER')
        source_rows = np.argsort(positions,kind='stable'); positions = positions[source_rows]
        batches = (source_rows[np.searchsorted(positions,start):np.searchsorted(positions,start+256)]
                   for start in range(0,len(order),256))
    for batch in batches:
        check(); en = elements[batch]
        if not len(en): continue
        dofs = (3*en[:, :, None]+np.arange(3)).reshape((-1, 24))
        m = mu[batch]*1e-6; v = nu[batch]
        values = m[:, None, None]*shear+(2*m*v/(1-2*v))[:, None, None]*lame
        rows = np.repeat(dofs, 24, axis=1).ravel(); columns = np.tile(dofs, (1, 24)).ravel()
        if source_order is None:
            part = sparse.coo_matrix((values.ravel(), (rows, columns)), shape=matrix.shape).tocsr()
        else:
            # COO->CSR duplicate summation does not promise source-owner order:
            # deleting an I/I contribution can reorder untouched I/E ties.
            # Reduce each coefficient in an explicit row/column/owner order
            # before constructing CSR, leaving no duplicates for it to sum.
            permutation = np.lexsort((np.repeat(cells[batch],576),columns,rows))
            keys = rows[permutation]*nodes.size+columns[permutation]
            starts = np.r_[0,np.flatnonzero(keys[1:]!=keys[:-1])+1]
            reduced = np.add.reduceat(values.ravel()[permutation],starts)
            part = sparse.coo_matrix((reduced,(rows[permutation][starts],columns[permutation][starts])),
                                     shape=matrix.shape).tocsr()
        matrix = matrix+part
    check()
    return matrix


class ExteriorOperator:
    """One sparse exterior factor; at most 64 harmonic-response columns at once."""
    def __init__(self, stiffness, core_dofs, *, retained=None, free_parent=None, check=lambda: None):
        started = time.perf_counter()
        self.k = sparse.csr_matrix(stiffness, dtype=float, copy=True)
        self.k.sum_duplicates(); self.k.eliminate_zeros(); self.k.sort_indices()
        n = self.k.shape[0]
        if (self.k.shape != (n, n) or not 1 < n <= 60000
                or not np.isfinite(self.k.data).all()):
            raise ValueError('EXTERIOR_STIFFNESS')
        scale = float(np.max(np.abs(self.k.data), initial=0.))
        if scale == 0 or np.max(np.abs((self.k-self.k.T).data), initial=0.) > EPS*scale:
            raise ValueError('EXTERIOR_SYMMETRY')
        core = np.asarray(core_dofs)
        if (core.ndim != 1 or core.dtype.kind not in 'iu' or not 0 < len(core) < n
                or len(core) > 6000 or len(np.unique(core)) != len(core)
                or np.any(core < 0) or np.any(core >= n)):
            raise ValueError('EXTERIOR_CORE_PARTITION')
        self.core = core.astype(np.int64, copy=True)
        self.exterior = np.setdiff1d(np.arange(n), self.core)
        self.kii = self.k[self.core][:, self.core].tocsc()
        self.kie = self.k[self.core][:, self.exterior].tocsr()
        self.kei = self.k[self.exterior][:, self.core].tocsc()
        self.kee = self.k[self.exterior][:, self.exterior].tocsc()
        self.interface = np.flatnonzero(np.diff(self.kei.indptr))
        self.stats = {'freeDofs': n, 'coreFreeDofs': len(self.core),
                      'exteriorFreeDofs': len(self.exterior), 'coupledCoreDofs': len(self.interface)}
        self._retained_operator = None
        self._factor_positions = np.arange(len(self.exterior))
        self._factor_identity = True
        if retained is not None:
            self._retain(retained,free_parent)
            self.stats.update(exteriorFactorSeconds=0.,exteriorFactorNnz=retained.stats['exteriorFactorNnz'],
                retainedExteriorFactor=True,exactExteriorAndBothCouplings=True,
                partitionAndFactorSeconds=time.perf_counter()-started)
            self._freeze(); check(); return
        if free_parent is not None: raise ValueError('EXTERIOR_RETAINED_PARENT_WITHOUT_OPERATOR')
        check(); begin = time.perf_counter()
        try:
            # With symmetric ordering and no diagonal pivoting, positive LDL
            # pivots certify exterior support. A spring is never part of K.
            self.factor = splu(self.kee, permc_spec='MMD_AT_PLUS_A', diag_pivot_thresh=0.,
                               options={'SymmetricMode': True})
        except RuntimeError as error:
            raise ValueError('EXTERIOR_UNSUPPORTED_EXTERIOR') from error
        upper = self.factor.U
        diagonal = upper.diagonal()
        factor_scale = float(np.max(np.abs(diagonal), initial=0.))
        if (not np.array_equal(self.factor.perm_r, self.factor.perm_c)
                or not np.isfinite(diagonal).all() or np.min(diagonal) <= EPS*factor_scale):
            raise ValueError('EXTERIOR_UNSUPPORTED_EXTERIOR')
        # A complete U-D L.T temporary can rival the sparse factor itself.
        # Certify the identical coefficients in bounded column blocks instead.
        lower_rows = self.factor.L.tocsr()
        for start in range(0, len(diagonal), 64):
            check(); stop = start+64
            dl = lower_rows[start:stop].T.multiply(diagonal[:,None]).tocsc()
            defect = upper[:,start:stop]-dl
            if np.max(np.abs(defect.data), initial=0.) > EPS*factor_scale:
                raise ValueError('EXTERIOR_FACTOR_SYMMETRY')
        del lower_rows, dl, defect
        self.stats.update(exteriorFactorSeconds=time.perf_counter()-begin,
            exteriorFactorNnz=self.factor.L.nnz+self.factor.U.nnz,
            partitionAndFactorSeconds=time.perf_counter()-started)
        self._freeze(); check()

    def _freeze(self):
        # Factors and source blocks must describe the same admitted operator
        # when another topology retains them. No episode fields live here.
        for matrix in (self.k,self.kii,self.kie,self.kei,self.kee):
            for array in (matrix.data,matrix.indices,matrix.indptr):array.flags.writeable=False
        for array in (self.core,self.exterior,self.interface,self._factor_positions):array.flags.writeable=False

    def _retain(self, old, free_parent):
        if not isinstance(old,ExteriorOperator):raise ValueError('EXTERIOR_RETAINED_OPERATOR')
        parents = np.asarray(free_parent)
        if (parents.ndim!=1 or parents.dtype.kind not in 'iu' or len(parents)!=self.k.shape[0]
                or len(np.unique(parents))!=len(parents) or np.any(parents<0) or np.any(parents>=old.k.shape[0])):
            raise ValueError('EXTERIOR_RETAINED_FREE_LINEAGE')
        # The retained fast path refuses splits/joins or lost E coordinates.
        # Such a graph still has the ordinary full construction path.
        ei = np.full(old.k.shape[0],-1,dtype=np.int64); ei[old.exterior]=np.arange(len(old.exterior))
        ci = np.full(old.k.shape[0],-1,dtype=np.int64); ci[old.core]=np.arange(len(old.core))
        e = ei[parents[self.exterior]]; i = ci[parents[self.core]]
        if (len(e)!=len(old.exterior) or np.any(e<0) or len(np.unique(e))!=len(e) or np.any(i<0)):
            raise ValueError('EXTERIOR_RETAINED_PARTITION_LINEAGE')
        if not _same_sparse_bits(self.kee,old.kee[e][:,e]):
            raise ValueError('EXTERIOR_RETAINED_EXTERIOR_COEFFICIENTS')
        if (not _same_sparse_bits(self.kie,old.kie[i][:,e])
                or not _same_sparse_bits(self.kei,old.kei[e][:,i])):
            raise ValueError('EXTERIOR_RETAINED_COUPLING_COEFFICIENTS')
        self.factor = old.factor
        self._factor_positions = old._factor_positions[e].copy()
        self._factor_identity = np.array_equal(self._factor_positions,np.arange(len(e)))
        self._retained_core_rows = i.copy(); self._retained_core_rows.flags.writeable=False
        # Retaining a factor must not keep every prior K/Schur/history alive.
        self._retained_operator = weakref.ref(old)

    def retained_schur(self, previous, *, check=lambda: None):
        if (not isinstance(previous,ExteriorCoreFactor) or self._retained_operator is None
                or self._retained_operator() is not previous.operator):
            raise ValueError('EXTERIOR_RETAINED_SCHUR_ORIGIN')
        begin=time.perf_counter();rows=previous._correction_rows[self._retained_core_rows]
        schur=self.kii.toarray()
        # Keep the original measured correction explicitly. Reconstructing it
        # as KIIold-Sold can lose it to cancellation when KIIold is much larger.
        # Only a 64-row indexed block is materialized; root correction storage
        # is immutable/shared, never an E-by-I harmonic or episode-field basis.
        for start in range(0,len(rows),64):
            check();selected=rows[start:start+64]
            correction=previous._correction[np.ix_(selected,rows)]
            schur[start:start+len(selected)]-=correction
        scale=float(np.max(np.abs(schur),initial=0.))
        asymmetry=float(np.max(np.abs(schur-schur.T),initial=0.))
        if not np.isfinite(schur).all() or asymmetry>EPS*scale:
            raise ValueError('EXTERIOR_SCHUR_SYMMETRY')
        self.stats.update(retainedSchurSeconds=time.perf_counter()-begin,schurColumnsSeconds=0.,
            schurAsymmetryNPerMm=asymmetry,denseSchurBytes=schur.nbytes,
            retainedSchurCoefficientProof=True,retainedSchurBatchRows=64)
        check();return schur

    def solve_exterior(self, rhs):
        rhs = np.asarray(rhs, float)
        if rhs.ndim not in (1, 2) or rhs.shape[0] != len(self.exterior) or not np.isfinite(rhs).all():
            raise ValueError('EXTERIOR_RHS')
        if rhs.ndim == 2 and rhs.shape[1] > 64:
            raise ValueError('EXTERIOR_BATCH_COLUMNS')
        if self._factor_identity:
            result = self.factor.solve(rhs)
        else:
            factor_rhs=np.empty_like(rhs);factor_rhs[self._factor_positions]=rhs
            result=self.factor.solve(factor_rhs)[self._factor_positions]
        if not np.isfinite(result).all():
            raise ValueError('EXTERIOR_NONFINITE_SOLVE')
        return result

    def columns(self, columns, *, retain_correction=False):
        columns = np.asarray(columns)
        if (columns.ndim != 1 or columns.dtype.kind not in 'iu' or not 1 <= len(columns) <= 64
                or np.any(columns < 0) or np.any(columns >= len(self.core))):
            raise ValueError('EXTERIOR_BATCH_COLUMNS')
        rhs = self.kei[:, columns].toarray()
        response = self.solve_exterior(rhs)
        residual = float(np.max(np.abs(self.kee@response-rhs), initial=0.))
        if residual > 1e-9:
            raise ValueError('EXTERIOR_BASIS_RESIDUAL')
        correction=self.kie@response
        result=self.kii[:,columns].toarray()-correction
        return (result,residual,correction) if retain_correction else (result,residual)

    def apply(self, core_field):
        value = _finite(core_field, (len(self.core),), 'CORE_FIELD')
        return self.kii@value-self.kie@self.solve_exterior(self.kei@value)

    def dense_schur(self, *, batch_columns=64, retain_correction=False, check=lambda: None):
        if type(batch_columns) is not int or not 1 <= batch_columns <= 64:
            raise ValueError('EXTERIOR_BATCH_COLUMNS')
        begin = time.perf_counter(); schur = self.kii.toarray(); maximum = 0.
        correction=np.zeros_like(schur) if retain_correction else None
        for start in range(0, len(self.interface), batch_columns):
            check(); columns = self.interface[start:start+batch_columns]
            if retain_correction:
                schur[:,columns],residual,correction[:,columns]=self.columns(columns,retain_correction=True)
            else:
                schur[:, columns], residual = self.columns(columns)
            maximum = max(maximum, residual)
        scale = float(np.max(np.abs(schur), initial=0.))
        asymmetry = float(np.max(np.abs(schur-schur.T), initial=0.))
        if not np.isfinite(schur).all() or asymmetry > EPS*scale:
            raise ValueError('EXTERIOR_SCHUR_SYMMETRY')
        self.stats.update(schurColumnsSeconds=time.perf_counter()-begin,
            maximumExteriorBasisResidualN=maximum, schurAsymmetryNPerMm=asymmetry,
            batchColumns=batch_columns, denseSchurBytes=schur.nbytes)
        if retain_correction:self.stats['retainedCorrectionBytes']=correction.nbytes
        check()
        return (schur,correction) if retain_correction else schur


class ExteriorCoreFactor:
    """Fixed tissue operators shared by contact bindings, without history."""
    def __init__(self, operator, *, retained=None, check=lambda: None):
        self.operator = operator
        if retained is None:
            self.schur,self._correction=operator.dense_schur(retain_correction=True,check=check)
            self._correction_rows=np.arange(len(operator.core))
        else:
            self.schur=operator.retained_schur(retained,check=check)
            self._correction=retained._correction
            self._correction_rows=retained._correction_rows[operator._retained_core_rows].copy()
        begin = time.perf_counter()
        try:
            self.factor = cho_factor(self.schur, lower=True, check_finite=False)
        except np.linalg.LinAlgError as error:
            raise ValueError('EXTERIOR_UNSUPPORTED_CORE') from error
        diagonal = np.diag(self.factor[0])
        if np.min(diagonal)**2 <= EPS*np.max(np.abs(self.schur)):
            raise ValueError('EXTERIOR_UNSUPPORTED_CORE')
        self.schur.flags.writeable=False;self.factor[0].flags.writeable=False
        self._correction.flags.writeable=False;self._correction_rows.flags.writeable=False
        self._body_force = None; self._body_field = None
        operator.stats['coreFactorSeconds'] = time.perf_counter()-begin
        check()

    def rebuild(self, stiffness, core_dofs, free_parent, *, check=lambda: None):
        """Retain proven fixed E/couplings, with a fresh core support factor.

        free_parent maps every new supported-free DOF into this operator's
        coordinates. No displacement, memory or removal energy is transferred.
        A refusal leaves this preparation unchanged; full construction remains
        required for changed exterior/couplings or non-bijective E lineage.
        """
        operator=ExteriorOperator(stiffness,core_dofs,retained=self.operator,free_parent=free_parent,check=check)
        return type(self)(operator,retained=self,check=check)

    def body_field(self, body):
        """Exact full equilibrium field; invalidate on every changed load value."""
        op = self.operator
        if self._body_force is None or not np.array_equal(body, self._body_force):
            field = np.zeros(op.k.shape[0])
            if np.any(body):
                outside = op.solve_exterior(body[op.exterior])
                field[op.core] = cho_solve(self.factor, body[op.core]-op.kie@outside, check_finite=False)
                field[op.exterior] = op.solve_exterior(body[op.exterior]-op.kei@field[op.core])
                if np.max(np.abs(op.k@field-body),initial=0.) > 1e-9:
                    raise ValueError('EXTERIOR_BODY_FIELD_RESIDUAL')
            self._body_force = body.copy(); self._body_field = field
        return self._body_field


class ExteriorHistorySolve:
    """Full distributed-history BE advancement through an exact exterior map.

    All branch matrices are declared alpha_j*K0. Source adapters must verify
    this spectrum before construction. Contact changes require a separately
    constructed binding and their energy jump is outside a fixed-binding step.
    """
    def __init__(self, operator, contact, spring, alpha, times, *, prepared=None, check=lambda: None):
        self.operator = operator
        self.alpha = np.asarray(alpha, float).copy(); self.times = np.asarray(times, float).copy()
        if (self.alpha.ndim != 1 or len(self.alpha) > 2 or self.times.shape != self.alpha.shape
                or not np.isfinite(self.alpha).all() or not np.isfinite(self.times).all()
                or np.any(self.alpha < 0) or np.any(self.times <= 0)):
            raise ValueError('EXTERIOR_PROPORTIONAL_SPECTRUM')
        self.c = sparse.csr_matrix(contact, dtype=float, copy=True); self.c.eliminate_zeros()
        m, n = self.c.shape
        if n != operator.k.shape[0] or m > 64 or not np.isfinite(self.c.data).all():
            raise ValueError('EXTERIOR_CONTACT')
        self.spring = _finite(spring, (m,), 'SPRING').copy()
        if np.any(self.spring <= 0) or self.c[:, operator.exterior].nnz:
            raise ValueError('EXTERIOR_CONTACT_OUTSIDE_CORE_OR_NONPOSITIVE')
        self.ci = self.c[:, operator.core]
        if prepared is not None and (not isinstance(prepared, ExteriorCoreFactor) or prepared.operator is not operator):
            raise ValueError('EXTERIOR_PREPARED_FOR_ANOTHER_OPERATOR')
        self.prepared = ExteriorCoreFactor(operator,check=check) if prepared is None else prepared
        self.schur = self.prepared.schur; self.core_factor = self.prepared.factor
        begin = time.perf_counter()
        self.basis = cho_solve(self.core_factor, self.ci.T.toarray(), check_finite=False)
        # Recover a complete contact compliance basis once per binding. The
        # distributed source memory remains full-dimensional in every step;
        # it is never fitted to this contact basis.
        self.full_basis = np.zeros((operator.k.shape[0],m)); self.full_basis[operator.core] = self.basis
        if m:
            self.full_basis[operator.exterior] = -operator.solve_exterior(operator.kei@self.basis)
        self.compliance = np.asarray(self.ci@self.basis)
        defect = operator.k@self.full_basis-self.c.T.toarray()
        if np.max(np.abs(defect), initial=0.) > 1e-9:
            raise ValueError('EXTERIOR_CORE_BASIS_RESIDUAL')
        self.binding_seconds = time.perf_counter()-begin
        operator.stats['contactBindingSeconds'] = self.binding_seconds
        operator.stats['fullContactBasisBytes'] = self.full_basis.nbytes
        check()

    def rebind(self, contact, spring, *, check=lambda: None):
        """Change only contacts; a changed tissue/topology needs a new operator."""
        return type(self)(self.operator,contact,spring,self.alpha,self.times,
                          prepared=self.prepared,check=check)

    def contact_energy(self, displacement, target):
        u = _finite(displacement, (self.operator.k.shape[0],), 'DISPLACEMENT')
        target = _finite(target, (len(self.spring),), 'TARGET')
        stretch = self.c@u-target
        return float(.5*stretch@(self.spring*stretch))

    def step(self, displacement, memories, body_force, target, h, *, previous_target):
        op = self.operator; n = op.k.shape[0]; m = len(self.spring)
        before = _finite(displacement, (n,), 'DISPLACEMENT')
        old_z = _finite(memories, (len(self.alpha), n), 'MEMORY')
        body = _finite(body_force, (n,), 'BODY_FORCE')
        target = _finite(target, (m,), 'TARGET')
        previous_target = _finite(previous_target, (m,), 'PREVIOUS_TARGET')
        if type(h) not in (int, float) or not np.isfinite(h) or h <= 0:
            raise ValueError('EXTERIOR_PHYSICAL_STEP')
        ratios = self.times/(self.times+h); beta = 1+float(ratios@self.alpha)
        history = np.sum((ratios*self.alpha)[:, None]*old_z, axis=0)
        base = (self.prepared.body_field(body)+history)/beta
        a = beta*np.eye(m)+self.spring[:, None]*self.compliance
        rhs = self.spring*(target-self.c@base)
        q = lu_solve(lu_factor(a), rhs) if m else np.empty(0)
        u = base+self.full_basis@q
        z = (self.times[:, None]*old_z+h*u)/(self.times[:, None]+h)
        delta = u-before; cu = self.c@u; cb = self.c@before
        external = body+self.c.T@(self.spring*target)
        ku = op.k@u; kb = op.k@before; kd = ku-kb
        residual = ku+self.c.T@(self.spring*cu)-external
        stretch_contact = cu-target; old_contact = cb-previous_target
        contact_increment = stretch_contact-old_contact
        contact_energy = .5*stretch_contact@(self.spring*stretch_contact)
        energy = .5*u@ku+contact_energy
        old_energy = .5*before@kb+.5*old_contact@(self.spring*old_contact)
        numerical = .5*delta@kd+.5*contact_increment@(self.spring*contact_increment); viscous = 0.
        for j, alpha in enumerate(self.alpha):
            stretch = u-z[j]; old_stretch = before-old_z[j]; dz = z[j]-old_z[j]
            ks = op.k@stretch; kos = op.k@old_stretch
            residual += alpha*ks
            energy += .5*alpha*stretch@ks; old_energy += .5*alpha*old_stretch@kos
            viscous += alpha*self.times[j]/h*dz@(op.k@dz)
            numerical += .5*alpha*(stretch-old_stretch)@(ks-kos)
        # Use the actual spring potential and explicit target increment. This
        # is model spring/actuator work, not a measured instrument-work claim.
        body_work = float(body@delta)
        target_work = float((-self.spring*stretch_contact)@(target-previous_target))
        work = body_work+target_work
        balance = work-float(energy-old_energy)-float(viscous)-float(numerical)
        maximum = float(np.max(np.abs(residual), initial=0.))
        if (not np.isfinite(u).all() or not np.isfinite(z).all()
                or maximum > 1e-9 or not np.isfinite(balance) or abs(balance) > 1e-8
                or viscous < -1e-12 or numerical < -1e-12):
            raise ValueError('EXTERIOR_STEP_RESIDUAL_OR_ENERGY')
        return {'displacement': u, 'memories': z, 'maxFreeResidualN': maximum,
                'energyNmm': float(energy), 'oldEnergyNmm': float(old_energy),
                'contactEnergyNmm': float(contact_energy), 'bodyWorkNmm': body_work,
                'targetWorkNmm': target_work,
                'dissipatedEnergyNmm': float(viscous), 'numericalDissipationNmm': float(numerical),
                'workNmm': work, 'energyBalanceResidualNmm': balance,
                'contactReactionN': self.spring*(target-cu)}

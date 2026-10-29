"""Exact retained exterior/coupling proof and independently solved new cores."""
import gc
import weakref

import numpy as np
import pytest
from scipy import sparse

from resectionlab.surgical_continuum_exterior import ExteriorOperator,ExteriorCoreFactor,ExteriorHistorySolve


def control(seed=151):
    rng=np.random.default_rng(seed);a=rng.normal(size=(19,19));k=a.T@a+3*np.eye(19)
    core=np.array([13,4,18,1,7,10,0]);base=ExteriorCoreFactor(ExteriorOperator(k,core))
    # Drop two old I coordinates, reorder both I and E, and change only KII.
    parents=np.array([x for x in rng.permutation(19) if x not in (4,10)])
    new_core=np.array([int(np.flatnonzero(parents==x)[0]) for x in [18,0,7,13,1]])
    new=k[np.ix_(parents,parents)].copy();new[np.ix_(new_core,new_core)]+=np.diag([.3,.7,1.2,.2,.6])
    return rng,k,core,base,parents,new_core,new


@pytest.mark.parametrize('seed',[151,152,153,154])
def test_retained_permuted_fields_and_two_memories_match_direct_full_solve(seed,monkeypatch):
    rng,_,_,old,parents,core,k=control(seed)
    def forbidden(**kwargs):raise AssertionError('Retained update rebuilt all Schur columns')
    monkeypatch.setattr(old.operator,'dense_schur',forbidden)
    rebuilt=old.rebuild(k,core,parents);op=rebuilt.operator
    assert op.factor is old.operator.factor and rebuilt.factor is not old.factor
    assert op.stats['retainedExteriorFactor'] and op.stats['schurColumnsSeconds']==0
    e=op.exterior
    expected=k[np.ix_(core,core)]-k[np.ix_(core,e)]@np.linalg.solve(k[np.ix_(e,e)],k[np.ix_(e,core)])
    np.testing.assert_allclose(rebuilt.schur,expected,atol=2e-13,rtol=1e-12)
    rhs=rng.normal(size=(len(e),3))
    np.testing.assert_allclose(op.solve_exterior(rhs),np.linalg.solve(k[np.ix_(e,e)],rhs),atol=1e-13)
    c=np.zeros((3,len(k)));c[np.arange(3),core[:3]]=1.;d=np.array([.2,.3,.4]);alpha=np.array([.5,.2]);times=np.array([.4,.9])
    solve=ExteriorHistorySolve(op,c,d,alpha,times,prepared=rebuilt)
    u=rng.normal(size=len(k))*.01;z=rng.normal(size=(2,len(k)))*.02;previous=np.zeros(3)
    for h in (.01,.037,.1,.04):
        target=rng.normal(size=3)*.03;body=rng.normal(size=len(k))*.002;r=times/(times+h)
        matrix=(1+alpha@r)*k+c.T@np.diag(d)@c
        force=body+c.T@(d*target)+sum((aj*rj*(k@memory) for aj,rj,memory in zip(alpha,r,z)),np.zeros(len(k)))
        direct=np.linalg.solve(matrix,force);result=solve.step(u,z,body,target,h,previous_target=previous)
        direct_z=(times[:,None]*z+h*direct)/(times[:,None]+h)
        np.testing.assert_allclose(result['displacement'],direct,atol=1e-13,rtol=1e-11)
        np.testing.assert_allclose(result['memories'],direct_z,atol=1e-13,rtol=1e-11)
        energy=lambda x,memories,t:.5*x@k@x+.5*(c@x-t)@(d*(c@x-t))+sum(.5*aj*(x-m)@k@(x-m) for aj,m in zip(alpha,memories))
        assert abs(result['energyNmm']-energy(direct,direct_z,target))<1e-13
        assert abs(result['oldEnergyNmm']-energy(u,z,previous))<1e-13
        assert abs(result['energyBalanceResidualNmm'])<1e-13
        u,z=result['displacement'],result['memories'];previous=target


def test_one_bit_exterior_or_either_coupling_change_refuses_and_old_factor_is_unchanged():
    _,_,_,old,parents,core,k=control();e=np.setdiff1d(np.arange(len(k)),core)
    original=old.schur.copy();factor=old.factor
    for row,col,pattern in [(e[0],e[0],'EXTERIOR_COEFFICIENTS'),(core[0],e[0],'COUPLING_COEFFICIENTS'),(e[0],core[0],'COUPLING_COEFFICIENTS')]:
        changed=k.copy();changed[row,col]=np.nextafter(changed[row,col],np.inf)
        with pytest.raises(ValueError,match=pattern):old.rebuild(changed,core,parents)
    np.testing.assert_array_equal(old.schur,original);assert old.factor is factor


def test_invalid_lineage_and_partition_cannot_retain_an_exterior_factor():
    _,_,_,old,parents,core,k=control()
    cases=[parents.astype(float),parents[:-1],np.r_[parents[:-1],parents[0]],parents+30]
    for mapping in cases:
        with pytest.raises(ValueError,match='FREE_LINEAGE'):old.rebuild(k,core,mapping)
    e=np.setdiff1d(np.arange(len(k)),core)
    for changed_core in (core[:-1],np.r_[core[:-1],e[0]]):
        with pytest.raises(ValueError,match='PARTITION_LINEAGE'):old.rebuild(k,changed_core,parents)


def test_new_unsupported_core_refuses_even_with_unchanged_exterior_and_couplings():
    _,_,_,old,parents,core,k=control();bad=k.copy();bad[np.ix_(core,core)]=-100*np.eye(len(core))
    with pytest.raises(ValueError,match='UNSUPPORTED_CORE'):old.rebuild(bad,core,parents)


def test_small_exterior_correction_is_preserved_under_large_old_core_cancellation():
    old=ExteriorCoreFactor(ExteriorOperator([[1e16,1.],[1.,1.]],[0]))
    # KIIold-Sold rounds to zero, although the correction is exactly one.
    assert float(old.operator.kii.toarray()[0,0]-old.schur[0,0])==0.
    rebuilt=old.rebuild([[2.,1.],[1.,1.]],[0],[0,1])
    assert rebuilt.schur[0,0]==1. and rebuilt._correction is old._correction
    with pytest.raises(ValueError,match='UNSUPPORTED_CORE'):
        old.rebuild([[.5,1.],[1.,1.]],[0],[0,1])


def test_wrong_schur_origin_and_accidental_matrix_mutation_refuse():
    _,_,_,old,parents,core,k=control()
    op=ExteriorOperator(k,core,retained=old.operator,free_parent=parents)
    unrelated=ExteriorCoreFactor(ExteriorOperator(old.operator.k,old.operator.core))
    with pytest.raises(ValueError,match='SCHUR_ORIGIN'):ExteriorCoreFactor(op,retained=unrelated)
    for value in (old.operator.k.data,old.operator.kee.data,old.operator.kie.indices,old.schur,old.factor[0]):
        with pytest.raises(ValueError):value.flat[0]=0


def test_rebuild_has_fresh_load_cache_and_does_not_keep_previous_dense_factors_alive():
    _,_,_,old,parents,core,k=control()
    body=np.ones(old.operator.k.shape[0])*.001;old.body_field(body)
    rebuilt=old.rebuild(k,core,parents)
    assert rebuilt._body_field is None and rebuilt._body_force is None
    ref=weakref.ref(old);old_operator=weakref.ref(old.operator);old_schur=weakref.ref(old.schur)
    del old;gc.collect()
    assert ref() is None and old_operator() is None and old_schur() is None
    actual=rebuilt.body_field(np.ones(len(k))*.002)
    np.testing.assert_allclose(actual,np.linalg.solve(k,np.ones(len(k))*.002),atol=1e-13)


def test_chained_reuse_composes_root_exterior_permutations_and_owns_inputs():
    rng,_,_,old,parents,core,k=control();prepared=old.rebuild(k,core,parents)
    correction=prepared._correction;factor=prepared.operator.factor
    for _ in range(4):
        mapping=rng.permutation(len(k));next_k=k[np.ix_(mapping,mapping)].copy()
        next_core=np.flatnonzero(np.isin(mapping,core))[::-1].copy()
        next_k[next_core,next_core]+=.1
        expected_k=next_k.copy();expected_core=next_core.copy()
        prepared=prepared.rebuild(next_k,next_core,mapping)
        assert prepared._correction is correction and prepared.operator.factor is factor
        next_k[:]=0.;next_core[:]=0;mapping[:]=0  # Caller arrays are not retained.
        np.testing.assert_array_equal(prepared.operator.k.toarray(),expected_k)
        np.testing.assert_array_equal(prepared.operator.core,expected_core)
        body=rng.normal(size=len(k))*.002
        np.testing.assert_allclose(prepared.body_field(body),np.linalg.solve(expected_k,body),atol=1e-13)
        k,core=expected_k,expected_core

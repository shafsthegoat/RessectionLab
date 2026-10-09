"""Algebra, immutable geometry and metadata controls; no native response fixtures."""
from copy import deepcopy
import hashlib
import math
from pathlib import Path
import subprocess
import time
import xml.etree.ElementTree as ET
import numpy as np
import pytest
from scripts import mechanics_hbe_branch_calibration as c
from scripts import mechanics_hbe_branch_calibration_readout as r
from scripts import mechanics_hbe_branch_calibration_experiment as runner
from scripts import mechanics_hbe_readout as original

ROOT=Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def forbid_execution(monkeypatch):
    def forbidden(*a,**kw):raise AssertionError('No native/phase/member access in pure controls')
    for owner,name in [(subprocess,'Popen'),(c.old,'solve'),(c.runtime,'supervise'),
                       (c.access.ReleasedStudy,'_read_selected'),(runner,'launch'),(runner,'worker'),(runner,'prepare')]:
        monkeypatch.setattr(owner,name,forbidden)


@pytest.fixture(scope='module')
def study():return c.declaration(ROOT)


@pytest.fixture(scope='module')
def context(study):
    reg=c.Registry(ROOT);refs=c.qualify(ROOT,study,reg)
    return {'root':ROOT,'study':study,'registry':reg,'references':refs,
            'directory':ROOT/study['output_root']/'experiment'}


def test_exact_frozen_manifest_sources_and_caps(study):
    assert len(study['inherited_sources'])==26
    for binding in study['inherited_sources'].values():
        assert hashlib.sha256((ROOT/binding['path']).read_bytes()).hexdigest()==binding['sha256']
    assert study['budgets']['aggregate_seconds']==3600
    assert study['budgets']['maximum_native_calls']==2
    assert study['budgets']['native_seconds_by_branch']=={'compression':2100,'tension':420}


def test_real_qualifications_preserve_negatives(context):
    assert set(context['references'])==set(c.AXIAL+c.TORSION)
    for branch,N in [('compression',36),('tension',24),('torsion_neg',12),('torsion_pos',12)]:
        assert context['references'][branch]['mesh_N']==N
    assert context['registry'].bound(context['study']['legacy']['outcome'])['failed_aggregate_criteria']


@pytest.mark.parametrize('branch',['compression','tension'])
def test_wrong_axial_identity_refused(context,branch):
    row=deepcopy(context['references'][branch]);row['mesh_N']=12
    with pytest.raises(ValueError,match='Reference branch'):c.checked_reference(branch,row)


def test_missing_qualification_refused_before_reader(study):
    altered=deepcopy(study);altered['axial_references']['tension']['artifact_bindings']['result.json']['sha256']='0'*64
    with pytest.raises(ValueError,match='hash'):c.qualify(ROOT,altered,c.Registry(ROOT))


def test_exact_caps_survive_metadata_registration_and_snapshot(monkeypatch,study):
    reg=c.Registry(ROOT);p=study['axial_references']['compression']['native_primitives']
    for binding in p.values():reg.add(binding)
    reg.primitives('compression',p)
    calls=[]
    monkeypatch.setattr(c.access,'verify_binding',lambda root,binding,**kw:calls.append((binding['path'],kw['maximum_bytes'])))
    reg.verify_all()
    assert (p['nodes']['path'],768*1024**2) in calls
    assert (p['elements']['path'],512*1024**2) in calls
    reg.add(p['nodes']);reg.verify_all()
    assert reg.snapshot()==reg.snapshot()
    assert reg.entries[p['nodes']['path']]['maximum_bytes']==768*1024**2
    with pytest.raises(ValueError,match='Unregistered'):reg.read({'path':'unknown','sha256':'a'*64})
    with pytest.raises(ValueError,match='Conflicting exact'):reg.add(p['nodes'],2**40,kind='unknown')


@pytest.mark.parametrize('branch',['compression','tension'])
def test_saved_decks_scalar_only(context,branch):
    source,deck,loading=c.fitted_contents(context['registry'],context['study'],branch,1750.)
    assert loading['mu_Pa']==1750.
    assert loading['min_residual_N2']==(1e-10*1750.*.004**2)**2
    assert loading['K_Pa']==149*1750./3
    assert 'role' in loading
    reference=(ROOT/context['study']['axial_references'][branch]['source_deck']['path']).read_text()
    c.old.check_fitted_deck(reference,source,1750.)
    for path in ['Material/material/c1','Control/time_steps','Boundary/bc/value']:
        tree=ET.fromstring(source);node=tree.find(path)
        assert node is not None
        node.text=str(float(node.text)+1)
        with pytest.raises(ValueError):c.old.check_fitted_deck(reference,ET.tostring(tree,encoding='unicode'),1750.)


@pytest.fixture(scope='module')
def model(context):return r.model_for(context,'tension')


def test_actual_saved_geometry_at_reference_mu_exact_math(model):
    # Pure displaced-geometry algebra, not a native prediction or patient record.
    current=model.Xh.copy();raw=np.zeros_like(current);raw[0]=[1e-8,2e-8,3e-8]
    a=model.half_frame(current,raw,full_displacement_m=0.)
    b=r.half_frame(model,current,raw,full_displacement_m=0.,mu=1000.)
    assert a['ratios']==b['ratios']
    assert np.array_equal(a['cell_energy_J'],b['cell_energy_J'])
    assert a['energy_J']==b['energy_J']


def test_actual_mu_changes_physics_and_floors(model):
    current=model.Xh.copy();current[:,0]*=1.001;raw=np.zeros_like(current);raw[-1]=[1e-7,0.,0.]
    a=r.half_frame(model,current,raw,full_displacement_m=0.,mu=1000.)
    b=r.half_frame(model,current,raw,full_displacement_m=0.,mu=2500.)
    assert b['energy_J']==pytest.approx(2.5*a['energy_J'],rel=1e-12)
    assert b['ratios']['free_dof_reaction']==pytest.approx(a['ratios']['free_dof_reaction']/2.5)
    assert b['ratios']['prescribed_motion']==a['ratios']['prescribed_motion']


@pytest.mark.parametrize('mu',[0.,-1.,float('inf'),float('nan'),True])
def test_bad_actual_mu_rejected(mu):
    with pytest.raises(ValueError):r.ScaleReduction(mu)


def test_streaming_scale_matches_original_algebra():
    # Explicit small algebra arrays only; never serialized as solver response logs.
    X=np.array([[0.,0.,0.],[.001,.001,.002]])
    x1=np.repeat(X[None,:,:],121,axis=0);x1[:,1,2]+=np.linspace(0,1e-4,121)
    x2=x1.copy();x2[67,1,0]+=1e-10
    raw=np.ones_like(x1)*np.linspace(0,1e-3,121)[:,None,None];scaled=raw*1.75;scaled[42,0,1]+=1e-8
    t1=np.linspace(0,1e-6,121);t2=1.75*t1;t2[20]+=1e-12
    class Mesh:
        fingerprint='explicit-algebra';rest_nodes_m=X;radius_m=.004
    a={'mesh':Mesh(),'current_nodes_m':x1,'raw_reactions_N':raw}
    b={'mesh':Mesh(),'current_nodes_m':x2,'raw_reactions_N':scaled}
    expected=original._scale_group({'mu_Pa':1000.,'applied_torque_Nm':t1},{'mu_Pa':1750.,'applied_torque_Nm':t2},a,b)
    reduced=r.ScaleReduction(1750.)
    for i in range(121):
        reduced.update({'index':i,'current':x1[i],'raw':raw[i],'torque':t1[i]},
                       {'index':i,'current':x2[i],'raw':scaled[i],'torque':t2[i]},X)
    assert reduced.finish()==expected
    assert expected['reaction']['actual']>0


def test_incomplete_or_reordered_pair_rejected():
    reducer=r.ScaleReduction(1500.)
    with pytest.raises(ValueError,match='Complete121'):reducer.finish()
    with pytest.raises(ValueError,match='out-of-order'):reducer.update({'index':1},{'index':1},np.zeros((1,3)))


def test_missing_confirmation_cannot_freeze(context):
    with pytest.raises(ValueError,match='Both actual'):r.paired_evidence(context,{},1200.,time.monotonic()+60)


def test_unregistered_large_hash_and_no_reveal_before_freeze(context):
    reg=context['registry']
    b=context['study']['axial_references']['compression']['native_primitives']['nodes']
    with pytest.raises(ValueError,match='changed'):reg.read(dict(b,sha256='0'*64),json_value=False)


@pytest.mark.parametrize('deadline,cap,now',[(5,600,6),(float('inf'),600,1),(5,0,1)])
def test_nested_budget_exhaustion(deadline,cap,now):
    with pytest.raises((ValueError,TimeoutError)):c.common.remaining_seconds(deadline,cap,now=now)


def test_one_corrupted_nodal_component_fails_original_scale_tolerance():
    rest=np.zeros((2,3));reducer=r.ScaleReduction(1750.)
    for i in range(121):
        reference={'index':i,'current':rest.copy(),'raw':rest.copy(),'torque':0.}
        fitted=deepcopy(reference)
        if i==60:fitted['raw'][1,2]=1e-5
        reducer.update(reference,fitted,rest)
    metrics=reducer.finish()
    assert metrics['reaction']['actual']>metrics['reaction']['limit']


def test_no_unconfirmed_freeze_or_unaudited_reveal(monkeypatch,study):
    released=object.__new__(c.BranchReleasedStudy)
    released._calibration=None
    monkeypatch.setattr(released,'_events',lambda:[])
    with pytest.raises(ValueError,match='prior calibration'):
        released.freeze_predictions(fit_binding={},prediction_binding={},fitted_runs={},output_path='unused')
    released.context={'release':{'execution':{'csv_schemas':study['csv_schemas']}}}
    monkeypatch.setattr(released,'_release',lambda:None)
    with pytest.raises(ValueError,match='Audited durable freeze'):
        released.evaluate_held_out(freeze_binding={},schemas={b:study['csv_schemas'][b] for b in c.TORSION})
    monkeypatch.setattr(released,'_events',lambda:[{'phase':'held_out_attempt'}])
    with pytest.raises(ValueError,match='seals subsequent'):
        released.read_calibration({})


def test_full_frame_scales_at_actual_mu(model):
    current=model.X.copy();current[:,0]*=1.001;raw=np.zeros_like(current);raw[100]=[1e-7,0,0]
    first=model.full.read_frame(current,raw,mu_Pa=1000.,branch='tension',fraction=0.)
    second=model.full.read_frame(current,raw,mu_Pa=2500.,branch='tension',fraction=0.)
    first['current_nodes_m']=current;second['current_nodes_m']=current
    assert r.frame_ratios(model.full,first,raw,mu=1000.)==c.inherited._frame_ratios(model.full,first,raw)
    assert second['energy_J']==pytest.approx(2.5*first['energy_J'],rel=1e-12)
    assert r.frame_ratios(model.full,second,raw,mu=2500.)['free_node_reaction']==pytest.approx(
        r.frame_ratios(model.full,first,raw,mu=1000.)['free_node_reaction']/2.5)


@pytest.mark.parametrize('mu',[1e-300,1e308])
def test_unrepresentable_fitted_scalars_refused_before_preparation(context,mu):
    with pytest.raises((ValueError,OverflowError)):
        c.fitted_contents(context['registry'],context['study'],'tension',mu)

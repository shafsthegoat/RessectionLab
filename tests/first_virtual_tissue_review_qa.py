"""Independent opt-in strict-tissue admission audit using tiny constructed arrays.

No patient files, upstream kernel, native fitting or growth solver are used.
The CLI requires final source/test pins before executing generated checks.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import replace
import hashlib
from itertools import product
import json
import math
from pathlib import Path
import subprocess
import types
from unittest.mock import patch

import numpy as np
from resectionlab.core import CaseData, SourceRef, array_digest, semantic_digest, thaw_json
from resectionlab.structural_evidence import structural_frame_hash
import resectionlab.virtual_glioma as vg

ROOT = Path(__file__).resolve().parents[1]
BASE = 'artifacts/virtual-glioma-tissue-review-independent-v1'
OLD_SOURCE_SHA = 'ac6bf026b204edb3ea4f853aa6749c0385eb7f16b48d0ef2b1e8d5c469802555'
SCHEMA = 'resectionlab.fk-categorical-tissue-pair-review.v1'
SCOPE = 'research_categorical_GM_WM_support_for_FK'
CHECKS = []


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def source_bytes(relative):
    assert Path(relative).suffix in {'.py', '.json', '.md', '.txt'}
    path = ROOT / relative
    assert path.stat().st_size < 2_000_000
    return path.read_bytes()


def metadata(case, mutate):
    value = thaw_json(case.metadata)
    mutate(value)
    return replace(case, metadata=value)


def fixture(generated=False):
    shape = (7,6,5)
    coords = np.indices(shape)
    brain = coords[0] > 0
    unknown = np.zeros(shape,bool)
    unknown[2,3,4] = True
    support = brain & ~unknown
    compartments = {'wm':(coords[0]>=4)&support,'gm':(coords[0]>=2)&(coords[0]<4)&support,
                    'classified':support,'unknown':unknown}
    angle=.43
    affine=np.eye(4)
    affine[:3,:3]=np.array([[math.cos(angle),-math.sin(angle),0.],
                           [math.sin(angle),math.cos(angle),0.],[0.,0.,1.]])@np.diag([-2.,1.5,2.5])
    affine[:3,3]=[3.,-4.,8.]
    refs = tuple(SourceRef(name,'generated://qa-'+name,sha256=digest,provenance='simulated' if generated else 'observed',native_frame='RAS+')
                 for name,digest in [('image','1'*64),('method-receipt','2'*64)])
    case = CaseData('qa-constructed-host', (10+coords[0]*20+coords[1]+coords[2]).astype(np.float32),
        compartments,affine,refs,brain_mask=brain,
        metadata={'role':'TRAIN','patient_group':'qa:constructed','is_synthetic':generated})
    layers = {}
    for name,kind in [('wm','white_matter'),('gm','gray_matter'),('classified','classification_support'),('unknown','unknown')]:
        layers[name]={'kind':'brain_tissue','tissue_class':kind,'provenance':'estimated',
            'source_role':'generated_numerical_control' if generated else 'unreviewed_intensity_class',
            'source_sha256':'1'*64,'run_sha256':'2'*64,'source_image_hash':array_digest(case.mri),
            'source_frame_hash':structural_frame_hash(case),'method':'tiny constructed QA fixture, no real person or model'}
    return metadata(case,lambda m:m.update(anatomy_layers=layers))


def health(generated=False):
    return {'status':'generated_numerical_control' if generated else 'source_reported_healthy',
        'patient_group':'qa:constructed','source_url':'https://example.invalid/generated-audit-fixture',
        'source_sha256':'3'*64,'statement':'Constructed QA object only; this is not a real health or anatomy review.'}


def seal(case):
    """Independently construct the documented evidence, not the production helper."""
    layers = {}
    for kind,name in [('white_matter','wm'),('gray_matter','gm'),('classification_support','classified'),('unknown','unknown')]:
        layers[kind]={'name':name,'array_hash':array_digest(case.compartments[name]),
                      'provenance':thaw_json(case.metadata['anatomy_layers'][name])}
    evidence={'case_id':case.case_id,'patient_group':'qa:constructed','source_image_hash':array_digest(case.mri),
        'source_frame_hash':structural_frame_hash(case),'source_refs_hash':semantic_digest([r.to_dict() for r in case.source_refs]),
        'brain_envelope_hash':array_digest(case.brain_mask),'classification_support_layer':'classified','unknown_layer':'unknown',
        'representation':'disjoint_binary_tissue_classes','physical_partial_volume_claim':False,'layers':layers}
    digest=semantic_digest(evidence)
    record={'schema':SCHEMA,'evidence':evidence,'evidence_hash':digest,
        'review':{'evidence_hash':digest,'reviewer_id':'generated-independent-test-only',
            'reviewed_at':'2026-10-16T00:00:00+00:00','decision':'accepted','rationale':'Constructed positive control, no patient approval.',
            'scope':SCOPE}}
    return metadata(case,lambda m:m.update(fk_tissue_pair_review=record))


def qualify(case, *, module=vg, uniform=False, evidence=None, affine=None):
    return module.qualify_host(case,patient_group='qa:constructed',role='TRAIN',
        healthy_source_evidence=health(case.metadata.get('is_synthetic') is True) if evidence is None else evidence,
        qualified_by='independent-generated-fixture-only',grid_shape=list(case.mri.shape),grid_affine=case.affine if affine is None else affine,
        tissue_mode=module.UNIFORM_DIFFUSION_DOMAIN if uniform else module.GM_WM_FRACTIONS,
        source_wm_layer=None if uniform else 'wm',source_gm_layer=None if uniform else 'gm')


def reject(name,case,evidence=None):
    # Stale and missing reviews must stop before spatial interpolation.
    with patch.object(vg,'_resample',side_effect=AssertionError('resampling occurred before rejection')):
        try:
            qualify(case,evidence=evidence)
        except (ValueError,TypeError):
            CHECKS.append(name)
        else:
            raise AssertionError('admitted '+name)


def audit(old):
    raw=fixture()
    reject('real_ordered_classes_without_review',raw)
    for field in ('accepted','tissue_review_accepted','anatomical_review_accepted','reviewed','host_accepted'):
        reject('sham_'+field,metadata(raw,lambda m,k=field:m.update({k:True})))
    reject('sham_nested_accepted',metadata(raw,lambda m:m.update(fk_tissue_pair_review={'accepted':True})))
    case=seal(raw)
    accepted=qualify(case)
    restored=vg._validate_qualification(case,accepted)
    assert restored[-1] == accepted['mapping']
    assert np.array_equal(restored[3],case.compartments['wm']|case.compartments['gm'])
    assert not restored[3][2,3,4]
    assert accepted['mapping']['tissuePairReview']==thaw_json(case.metadata['fk_tissue_pair_review'])
    CHECKS.append('exact_generated_review_contract_roundtrip')
    legacy=qualify(raw,module=old)
    with patch.object(vg,'_resample',side_effect=AssertionError('legacy unreviewed receipt reached resampling')):
        try: vg._validate_qualification(raw,legacy)
        except ValueError: CHECKS.append('legacy_unreviewed_strict_receipt_rejected')
        else: raise AssertionError('legacy unreviewed strict receipt accepted')
    for name in ('wm','gm','classified','unknown'):
        masks=dict(case.compartments); value=np.array(masks[name],copy=True);value[3,2,1]=~value[3,2,1];masks[name]=value
        reject('stale_'+name+'_array',replace(case,compartments=masks))
    image=np.array(case.mri,copy=True);image[3,2,1]+=1
    reject('stale_image',replace(case,mri=image))
    affine=np.array(case.affine,copy=True);affine[0,3]+=.25
    reject('stale_frame',replace(case,affine=affine))
    brain=np.array(case.brain_mask,copy=True);brain[1,0,0]=False
    reject('stale_brain',replace(case,brain_mask=brain))
    reject('stale_case_id',replace(case,case_id='another-constructed-host'))
    reject('stale_source_refs',replace(case,source_refs=case.source_refs+(SourceRef('extra','generated://extra',sha256='4'*64),)))
    for name in ('wm','gm','classified','unknown'):
        for key,value in {'method':'changed method','source_role':'changed role','run_sha256':'4'*64,
                          'source_sha256':'4'*64,'source_image_hash':'sha256:'+'4'*64,'source_frame_hash':'sha256:'+'4'*64}.items():
            reject('stale_'+name+'_'+key,metadata(case,lambda m,n=name,k=key,v=value:m['anatomy_layers'][n].__setitem__(k,v)))
    for key,value in {'decision':'pending','reviewer_id':'','rationale':'','reviewed_at':'2026-10-16T00:00:00',
                      'scope':'clinical_acceptance','evidence_hash':'sha256:'+'4'*64}.items():
        reject('bad_review_'+key,metadata(case,lambda m,k=key,v=value:m['fk_tissue_pair_review']['review'].__setitem__(k,v)))
    reject('stale_review_seal',metadata(case,lambda m:m['fk_tissue_pair_review'].__setitem__('evidence_hash','sha256:'+'4'*64)))
    reject('PVE_claim',metadata(case,lambda m:m['fk_tissue_pair_review']['evidence'].__setitem__('physical_partial_volume_claim',True)))
    for name in ('wm','gm'):
        altered=metadata(raw,lambda m,n=name:m['anatomy_layers'][n].__setitem__('tissue_class','ordered_intensity_class'))
        reject('freshly_sealed_wrong_'+name+'_class_meaning',seal(altered))
    for key in ('source_sha256','run_sha256'):
        altered=metadata(raw,lambda m,k=key:m['anatomy_layers']['wm'].__setitem__(k,'4'*64))
        reject('freshly_sealed_detached_'+key,seal(altered))
    for problem in ('overlap','unknown_in_support','missing_coverage','support_outside_brain','wm_in_unknown'):
        masks={n:np.array(v,copy=True) for n,v in raw.compartments.items()}
        if problem=='overlap': masks['wm'][3,2,1]=True
        elif problem=='unknown_in_support': masks['classified'][2,3,4]=True
        elif problem=='missing_coverage': masks['classified'][1,0,0]=False
        elif problem=='support_outside_brain': masks['classified'][0,0,0]=True
        elif problem=='wm_in_unknown': masks['wm'][2,3,4]=True
        reject('freshly_sealed_invalid_'+problem,seal(replace(raw,compartments=masks)))
    gen=fixture(True)
    for synthetic,refs,health_generated,wm_generated,gm_generated in product((False,True),repeat=5):
        changed=metadata(gen,lambda m:m.update(is_synthetic=synthetic))
        if not refs:
            changed=replace(changed,source_refs=changed.source_refs+(SourceRef('observed','generated://laundering-test',sha256='5'*64),))
        for name,flag in (('wm',wm_generated),('gm',gm_generated)):
            if not flag:
                changed=metadata(changed,lambda m,n=name:m['anatomy_layers'][n].__setitem__('source_role','estimated_intensity_class'))
        flags=(synthetic,refs,health_generated,wm_generated,gm_generated)
        if all(flags):
            record=qualify(changed,evidence=health(True))
            assert 'tissuePairReview' not in record['mapping']
            CHECKS.append('generated_exemption_all_five_conditions')
        else:
            reject('generated_exemption_reject_'+''.join(str(int(v)) for v in flags),changed,health(health_generated))
    # Protected admission must precede even image hashing, not merely resampling.
    for role in ('SELECT','EVAL','EXTERNAL_TEST'):
        protected=metadata(case,lambda m,r=role:m.update(role=r))
        with patch.object(vg,'array_digest',side_effect=AssertionError('protected array touched')):
            reject('protected_role_'+role,protected)
    # Compare frozen implementation on generated inputs without loading FK assets.
    continuity=[]
    for label,host,uniform in [('real_schema_uniform',raw,True),('generated_uniform',gen,True),('generated_strict',gen,False)]:
        before=qualify(host,module=old,uniform=uniform)
        after=qualify(host,uniform=uniform)
        assert before==after
        old_values=old._validate_qualification(host,before)
        new_values=vg._validate_qualification(host,before)
        for a,b in zip(old_values,new_values):
            assert np.array_equal(a,b) if isinstance(a,np.ndarray) else a==b
        continuity.append({'fixture':label,'qualification_hash':after['qualificationHash'],'qualification_and_reconstruction_equal':True})
    # Even malformed tissue review metadata is irrelevant to unchanged uniform mode.
    uniform_bad_review=metadata(raw,lambda m:m.update(fk_tissue_pair_review={'accepted':True}))
    assert qualify(uniform_bad_review,uniform=True)==qualify(uniform_bad_review,module=old,uniform=True)
    CHECKS.append('uniform_mode_ignores_unused_tissue_review_as_before')
    coverage=[]
    for axis,shift in [(0,.25),(1,-.25),(2,-.2)]:
        affine=np.array(case.affine,copy=True)
        affine[:3,3]+=case.affine[:3,axis]*shift
        q=qualify(case,affine=affine)
        values=vg._validate_qualification(case,q)
        prior=old._validate_qualification(raw,qualify(raw,module=old,affine=affine))
        # Independent point-by-point nearest coverage oracle, no resampling helper.
        mapped_support=np.zeros(case.mri.shape,bool);mapped_unknown=np.zeros_like(mapped_support)
        matrix=np.linalg.inv(case.affine)@affine
        for index in np.ndindex(case.mri.shape):
            source=(matrix@np.array([*index,1.]))[:3]
            if np.any(source<0) or np.any(source>np.array(case.mri.shape)-1): continue
            nearest=tuple(np.floor(source+.5).astype(int))
            mapped_support[index]=case.compartments['classified'][nearest]
            mapped_unknown[index]=case.compartments['unknown'][nearest]
        expected=prior[3]&mapped_support&~mapped_unknown
        assert np.array_equal(values[3],expected)
        assert not np.any(values[3]&mapped_unknown)
        assert np.array_equal(values[4],prior[4]) and np.array_equal(values[5],prior[5])
        assert q['mapping']['reviewedCoverageMapping']=={
            'interpolation':'nearest_neighbor_order0_constant0_voxel_centers',
            'classificationSupportHash':array_digest(mapped_support),'unknownHash':array_digest(mapped_unknown),
            'domainRule':'brain_and_known_support_and_not_UNKNOWN_and_GM_plus_WM_at_least_threshold'}
        assert q['mapping']['domainRule']==q['mapping']['reviewedCoverageMapping']['domainRule']
        assert q['mapping']['admittedDomainHash']==array_digest(expected)
        point=(2,3,4)
        assert mapped_unknown[point] and prior[3][point] and not values[3][point]
        coverage.append({'source_axis':axis,'shift_source_voxels':shift,'nearest_unknown_target':list(point),
            'uncorrected_domain_included_target':bool(prior[3][point]),'final_domain_included_target':bool(values[3][point]),
            'independent_mapped_support_hash':array_digest(mapped_support),'independent_mapped_unknown_hash':array_digest(mapped_unknown),
            'final_domain_hash':array_digest(expected),'linear_weights_unchanged':True,'mapping':q['mapping']})
    CHECKS.append('shifted_oblique_reflected_UNKNOWN_coverage_excluded_all_three_axes')
    return {'checks':CHECKS,'uniform_and_generated_strict_continuity':continuity,'reviewed_real_mapped_coverage':coverage,
        'positive_fixture':{'shape':list(case.mri.shape),'brain_cells':int(case.brain_mask.sum()),
            'classification_support_cells':int(case.compartments['classified'].sum()),'unknown_cells':int(case.compartments['unknown'].sum()),
            'GM_WM_domain_cells':int(restored[3].sum()),'review_created_for_actual_patient':False}}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-sha',required=True)
    parser.add_argument('--tests-sha',required=True)
    args=parser.parse_args()
    pins={'src/resectionlab/virtual_glioma.py':args.source_sha,
          'tests/test_virtual_glioma_tissue_review.py':args.tests_sha,
          BASE+'/initial-coverage-finding.json':'671a0167409745c567fb7080fc8c80af0e8a30c1472b984621140cf981403e83'}
    for name,pin in pins.items(): assert sha(source_bytes(name))==pin,name
    previous=subprocess.run(['git','show','HEAD:src/resectionlab/virtual_glioma.py'],cwd=ROOT,capture_output=True,check=True).stdout
    assert sha(previous)==OLD_SOURCE_SHA
    old=types.ModuleType('resectionlab._qa_previous_virtual_glioma');old.__package__='resectionlab'
    # Dataclass processing looks up the defining module; no external assets run.
    import sys
    sys.modules[old.__name__]=old
    exec(compile(previous,'<frozen-before-tissue-review>','exec'),old.__dict__)
    with (patch.object(vg,'load_pinned_kernel',side_effect=AssertionError('solver not permitted')),
          patch.object(vg,'ensure_source_assets',side_effect=AssertionError('solver assets not permitted'))):
        results=audit(old)
    for name,pin in pins.items(): assert sha(source_bytes(name))==pin,name
    result={'schema':'independent-virtual-tissue-review-admission-v1','status':'pass_generated_admission_only',
        'source_pins_unchanged_before_after':pins,'previous_uniform_source_sha256':OLD_SOURCE_SHA,
        'auditor_sha256':sha(source_bytes(str(Path(__file__).relative_to(ROOT)))),**results,
        'real_patient_review_created':False,'patient_arrays_opened':False,'fitting_or_growth_executed':False,
        'retained_initial_finding':{'path':BASE+'/initial-coverage-finding.json',
            'source_sha256':'d8ad944f31dbb26fd1bd4dda749aad7bc02701e3c17d030ea0436f9b8820ce46',
            'initial_source_snapshot_retained':False,
            'limitation':'Only the exact source hash and generated numerical witness were retained; original source bytes had changed before snapshot request.'},
        'limitations':['The review is a declared, hash-bound research record, not authentication of a human reviewer or anatomical correctness.',
            'Positive controls use only constructed arrays and fictional evidence; no IXI tissue, host, growth or training acceptance is created.',
            'Categorical tissue occupancy after interpolation is not a validated physical partial-volume estimate.']}
    target=ROOT/BASE/'independent-qa.json';target.parent.mkdir(parents=True,exist_ok=True)
    with target.open('x') as stream: json.dump(result,stream,indent=2);stream.write('\n')
    print(json.dumps({'status':result['status'],'checks':len(CHECKS),'sha256':sha(target.read_bytes()),'report':str(target.relative_to(ROOT))},indent=2))


if __name__=='__main__': main()

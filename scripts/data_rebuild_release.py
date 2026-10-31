#!/usr/bin/env python3
"""Publish independently accepted local packages into the mounted app catalog.

No network publication. Candidate bytes stay immutable for resumable preparation;
the linked review sidecar is authoritative acceptance, bound to exact package bytes.
"""
from __future__ import annotations

import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import re
import shutil

from resectionlab.core import thaw_json
from resectionlab.data_root import atomic_json,checked_root,safe_path,sha256_file
from resectionlab.workspace_bundle import load_workspace

ACCEPTED={'accepted_for_stated_capabilities','accepted_with_explicit_limitations'}


def publish(root=None,*,freeze=None):
    root=checked_root(root,required_bytes=10*1024**3)
    decision_path=root/'qc/independent/release-decisions-v1.json';raw=decision_path.read_bytes()
    decisions=json.loads(raw)
    if decisions.get('schema')!='ressectionlab-independent-release-decisions-v1' or not decisions.get('reviewer'):
        raise ValueError('Independent release decisions are required')
    digest=hashlib.sha256(raw).hexdigest()
    snapshots=root/'qc/release-review-snapshots';snapshots.mkdir(exist_ok=True)
    snapshot=snapshots/(digest+'.json')
    if not snapshot.exists():snapshot.write_bytes(raw)
    case_reviews=root/'qc/case-decisions';case_reviews.mkdir(exist_ok=True)
    repair_index=root/'qc/proxy-unit-repairs/index.json'
    proxy_repairs=json.loads(repair_index.read_text())['cases'] if repair_index.exists() else []
    rows=[];ids=set()
    for decision in decisions['cases']:
        if decision['decision'] not in ACCEPTED:continue
        case_id=decision['case_id']
        if not re.fullmatch(r'[A-Za-z0-9_.:-]{1,200}',case_id) or case_id in ids:raise ValueError('Invalid/duplicate stable case identity')
        ids.add(case_id)
        source=Path(decision['path'])
        if not source.is_absolute():source=safe_path(root,source)
        if not source.resolve().is_relative_to(root) or sha256_file(source)!=decision['sha256']:raise ValueError('Reviewed candidate bytes changed')
        case,_,_=load_workspace(source)
        contract=thaw_json(case.metadata).get('data_case',{})
        if contract.get('schema')!='ressectionlab-data-case-v1':raise ValueError('Native preparation provenance missing')
        family=contract.get('family_id',contract.get('person_id'))
        if family!=decision.get('family_id') or contract.get('role')!=decision.get('role'):raise ValueError('Independent family/role binding differs')
        classification=contract.get('classification',contract.get('case_type','real_patient_partial'))
        category='synthetic' if classification.startswith('synthetic') else 'controls' if 'control' in classification else 'real'
        destination=root/'cases'/category/source.name
        checked_root(root,required_bytes=source.stat().st_size*2)
        if destination.exists():
            if sha256_file(destination)!=decision['sha256']:raise ValueError('Preserve different existing release package')
        else:
            temporary=destination.with_name(destination.name+'.partial')
            shutil.copyfile(source,temporary)
            if sha256_file(temporary)!=decision['sha256']:raise ValueError('Package copy checksum failed')
            temporary.replace(destination)
        reopened,_,_=load_workspace(destination)
        if reopened.semantic_hash!=case.semantic_hash:raise ValueError('Production final-location reopen changed anatomy')
        missing=contract.get('missing_anatomy',[])
        if category=='synthetic':missing=['acquired_tumor_observation' if name=='tumor' else name for name in missing]
        sidecar={'schema':'ressectionlab-case-acceptance-v1','case_id':case_id,'native_case_id':case.case_id,
            'package_path':str(destination.relative_to(root)),'package_sha256':decision['sha256'],
            'person_id':family,'family_id':family,'role':decision['role'],'classification':classification,
            'source':contract.get('source'),'source_version':contract.get('source_version'),
            'license':contract.get('license',contract.get('source_license')),'exposure':contract.get('exposure'),
            'components':contract.get('components'),'observations':contract.get('observations',contract.get('source_files')),
            'missing_anatomy':missing,'capabilities':decision['capabilities'],'limitations':decision['limitations'],
            'independent_decision':decision,'source_decision_snapshot':str(snapshot.relative_to(root)),
            'source_decision_sha256':digest,'production_final_location_reopen':'passed',
            'generated_proxy_unit_repairs':[row for row in proxy_repairs if row['person_id']==family],
            'native_preparation_status_is_historical':True,'clinical_validation':False}
        sidecar_name=case_id.replace(':','-')+'-'+decision['sha256'][:12]+'-'+digest[:12]+'.json'
        sidecar_path=case_reviews/sidecar_name
        atomic_json(sidecar_path,sidecar)
        rows.append({'case_id':case_id,'native_case_id':case.case_id,'path':str(destination.relative_to(root)),
            'sha256':decision['sha256'],'bytes':destination.stat().st_size,'person_id':family,'role':decision['role'],
            'classification':classification,'source':contract.get('source','External'),
            'decision':decision['decision'],'capabilities':decision['capabilities'],'limitations':decision['limitations'],
            'missing_anatomy':missing,'review_path':str(sidecar_path.relative_to(root)),'review_sha256':sha256_file(sidecar_path)})
    document={'schema':'ressectionlab-data-release-v1','release_id':freeze or 'development-reviewed-preview',
        'built_utc':datetime.now(timezone.utc).isoformat(),'independent_review_snapshot':str(snapshot.relative_to(root)),
        'clinical_validation':False,'cases':rows,'case_count':len(rows),'person_family_count':len({row['person_id'] for row in rows}),
        'synthetic_variant_count':sum(row['classification'].startswith('synthetic') for row in rows),
        'exact_acquired_tumor_MRI_skull_CT_dedicated_angiography_subjects':0}
    if freeze:
        if not re.fullmatch(r'[A-Za-z0-9_.-]{1,100}',freeze):raise ValueError('Safe release name required')
        releases=root/'catalog/releases';releases.mkdir(exist_ok=True);frozen=releases/(freeze+'.json')
        if frozen.exists():raise ValueError('Frozen release already exists; choose a new version')
        atomic_json(frozen,document)
    atomic_json(root/'catalog/release.json',document)
    print(json.dumps({key:value for key,value in document.items() if key!='cases'},indent=2))
    return document


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--freeze');args=parser.parse_args()
    publish(freeze=args.freeze)

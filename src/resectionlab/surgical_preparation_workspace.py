"""Native study/application ownership for source preparation alternatives.

Admissions belong to installed preparation branches. The legacy single admission
stays unchanged. A report, seal or branch label alone never grants authority.
"""
from __future__ import annotations

import copy
from collections import OrderedDict

from .core import semantic_digest
from .surgical_engine import SurgicalEngine
from .surgical_task_admission import bind_admission
from .surgical_interactive_workspace import _seal, _checked_seal
from . import surgical_preparation_planning as planning

SCHEMA='resectionlab.source-preparation-workspace.v1'
_APPLICATIONS=OrderedDict()


def unused(record,prefix):
    result=prefix+str(len(record['branches']))
    while result in record['branches']:result+='-next'
    return result


def validate_workspace(scene,record,case_hash,*,check=lambda:None):
    check();history=record.get('preparationComparisons',[]);applications=record.get('preparationApplications',[])
    planning.require(type(history) is list and len(history)<=4 and type(applications) is list and len(applications)<=12,
        'preparation history exceeds bounded schema')
    if history or applications:
        for start in record['branches']:
            check();seen=set();branch=start
            while branch is not None:
                planning.require(branch not in seen and branch in record['branches'],'invalid or cyclic native branch ancestry')
                seen.add(branch);parent=record.get('parents',{}).get(branch)
                if parent is None:break
                planning.require(type(parent) is dict and type(parent.get('branchId')) is str and
                    parent['branchId'] in record['branches'] and type(parent.get('position')) is int and
                    0<=parent['position']<=len(record['branches'][parent['branchId']]['actions']),
                    'native branch parent or position differs')
                branch=parent['branchId']
    reports={};roots={};study_branches=set()
    for revision,item in enumerate(history,1):
        check();_checked_seal(item,'workspaceSeal')
        planning.require(set(item)=={'schema','revision','caseHash','sourceBranchId','sourceSnapshotHash','configuredBranchId','report','workspaceSeal'}
            and item['schema']==SCHEMA and type(item['revision']) is int and item['revision']==revision and item['caseHash']==case_hash,
            'preparation study identity differs')
        snapshot=record['branches'].get(item['sourceBranchId']);configured=record['branches'].get(item['configuredBranchId'])
        planning.require(item['sourceBranchId']!=item['configuredBranchId'] and not
            ({item['sourceBranchId'],item['configuredBranchId']} & study_branches),'preparation study branch ownership repeats')
        study_branches.update((item['sourceBranchId'],item['configuredBranchId']))
        planning.require(snapshot is not None and semantic_digest(snapshot)==item['sourceSnapshotHash']==item['report']['sourceSnapshotHash']
            and semantic_digest(configured)==semantic_digest(item['report']['configuredSnapshot']),'preparation source branches differ')
        root=SurgicalEngine.restore(scene,snapshot,check=check)
        planning.validate_report(root,item['report'],check=check)
        timeline=record.get('timeline',{}).get(item['sourceBranchId'])
        planning.require(timeline is None or semantic_digest(timeline)==semantic_digest(root.history),'preparation source receipts differ')
        reports[revision]=item;roots[revision]=root
    application_branches=set()
    for app in applications:
        check();_checked_seal(app,'applicationSeal')
        planning.require(set(app)=={'revision','reportSeal','candidateId','phase','branchId','preparationBranchId',
            'snapshotHash','preparedSnapshotHash','receipt','applicationSeal'} and type(app['revision']) is int,
            'preparation application schema differs')
        item=reports.get(app['revision']);snapshot=record['branches'].get(app['branchId']);prepared=record['branches'].get(app['preparationBranchId'])
        planning.require(item is not None and snapshot is not None and prepared is not None and item['report']['reportSeal']==app['reportSeal']
            and semantic_digest(snapshot)==app['snapshotHash'] and semantic_digest(prepared)==app['preparedSnapshotHash'],
            'preparation application identity differs')
        owned={app['branchId'],app['preparationBranchId']}
        planning.require(not (owned & (application_branches|study_branches)),'preparation application branch ownership repeats')
        application_branches.update(owned)
        row=next((r for r in item['report']['rows'] if r['id']==app['candidateId']),None)
        planning.require(row is not None and row['preparedSnapshot'] is not None
            and semantic_digest(row['preparedSnapshot'])==semantic_digest(prepared),'installed preparation differs from its selected row')
        parent=record.get('parents',{}).get(app['preparationBranchId']);source=record['branches'][item['sourceBranchId']]
        parent_snapshot=record['branches'].get(parent.get('branchId')) if type(parent) is dict else None
        planning.require(parent_snapshot is not None and type(parent.get('position')) is int and parent['position']==len(source['actions'])
            and semantic_digest(parent_snapshot['actions'][:parent['position']])==semantic_digest(source['actions'])
            and parent_snapshot.get('sceneId')==source.get('sceneId')
            and semantic_digest(parent_snapshot.get('protectionContract'))==semantic_digest(source.get('protectionContract')),
            'installed preparation source ancestry differs')
        if app['phase']=='prepared':planning.require(app['branchId']==app['preparationBranchId'],'prepared application branch differs')
        else:planning.require(app['phase']=='withdraw' and record.get('parents',{}).get(app['branchId'])==
            {'branchId':app['preparationBranchId'],'position':len(prepared['actions'])},'complete preparation plan ancestry differs')
        key=(scene.identity,semantic_digest(item),semantic_digest(app),semantic_digest(snapshot),semantic_digest(prepared))
        if key not in _APPLICATIONS:
            candidate,receipt=planning.materialize(roots[app['revision']],item['report'],app['candidateId'],app['phase'],check=check)
            planning.require(semantic_digest(candidate.snapshot())==semantic_digest(snapshot) and semantic_digest(receipt)==semantic_digest(app['receipt']),
                'installed preparation native program/receipt differs')
            check();_APPLICATIONS[key]=True
        else:_APPLICATIONS.move_to_end(key)
    while len(_APPLICATIONS)>32:_APPLICATIONS.popitem(last=False)
    return history


def install_report(engine,record,case_hash,report,*,check=lambda:None):
    history=validate_workspace(engine.scene,record,case_hash,check=check)
    planning.require(len(history)<4,'save this preparation study before adding another comparison')
    planning.validate_report(engine,report,check=check)
    staged=copy.deepcopy(record);source=unused(staged,'preparation-source-')
    staged['branches'][source]=engine.snapshot()
    staged.setdefault('parents',{})[source]={'branchId':record['branchId'],'position':record['position']}
    staged.setdefault('timeline',{})[source]=copy.deepcopy(engine.history)
    configured=unused(staged,'preparation-configured-')
    staged['branches'][configured]=copy.deepcopy(report['configuredSnapshot'])
    configured_engine=SurgicalEngine.restore(engine.scene,report['configuredSnapshot'],check=check)
    staged['parents'][configured]={'branchId':source,'position':len(engine.actions)}
    staged['timeline'][configured]=copy.deepcopy(configured_engine.history)
    item=_seal({'schema':SCHEMA,'revision':len(history)+1,'caseHash':case_hash,'sourceBranchId':source,
        'sourceSnapshotHash':semantic_digest(engine.snapshot()),'configuredBranchId':configured,'report':copy.deepcopy(report)},'workspaceSeal')
    staged.setdefault('preparationComparisons',[]).append(item)
    check();record.clear();record.update(staged)
    return item


def current_report(engine,record,case_hash,seal,*,check=lambda:None):
    history=validate_workspace(engine.scene,record,case_hash,check=check)
    planning.require(bool(history) and history[-1]['report']['reportSeal']==seal,'preparation comparison changed')
    item=history[-1]
    planning.validate_report(engine,item['report'],check=check)
    return item


def install_application(engine,record,case_hash,item,candidate_id,phase,*,check=lambda:None):
    planning.require(current_report(engine,record,case_hash,item['report']['reportSeal'],check=check)==item,'preparation source comparison changed')
    planning.require(len(record.get('preparationApplications',[]))<12,'save the existing preparation branches before installing more')
    candidate,receipt=planning.materialize(engine,item['report'],candidate_id,phase,check=check)
    row=next(r for r in item['report']['rows'] if r['id']==candidate_id)
    prepared=SurgicalEngine.restore(engine.scene,row['preparedSnapshot'],check=check)
    staged=copy.deepcopy(record);prepared_branch=unused(staged,'preparation-')
    staged['branches'][prepared_branch]=prepared.snapshot()
    staged.setdefault('parents',{})[prepared_branch]={'branchId':record['branchId'],'position':record['position']}
    staged.setdefault('timeline',{})[prepared_branch]=copy.deepcopy(prepared.history)
    branch=prepared_branch
    if phase=='withdraw':
        branch=unused(staged,'preparation-plan-');staged['branches'][branch]=candidate.snapshot()
        staged['parents'][branch]={'branchId':prepared_branch,'position':len(prepared.actions)}
        staged['timeline'][branch]=copy.deepcopy(candidate.history)
    staged.update(branchId=branch,position=len(candidate.actions))
    app=_seal({'revision':item['revision'],'reportSeal':item['report']['reportSeal'],'candidateId':candidate_id,'phase':phase,
        'branchId':branch,'preparationBranchId':prepared_branch,'snapshotHash':semantic_digest(candidate.snapshot()),
        'preparedSnapshotHash':semantic_digest(prepared.snapshot()),'receipt':receipt},'applicationSeal')
    staged.setdefault('preparationApplications',[]).append(app)
    check();record.clear();record.update(staged)
    return candidate,receipt


def _admissions(record,history):
    reports={item['revision']:item for item in history};result={}
    for app in record.get('preparationApplications',[]):
        row=next(r for r in reports[app['revision']]['report']['rows'] if r['id']==app['candidateId'])
        result[app['preparationBranchId']]=row['preparedAdmission']
    return result


def admission_for_branch(scene,record,case_hash,branch_id,*,check=lambda:None):
    history=validate_workspace(scene,record,case_hash,check=check)
    admissions=_admissions(record,history);seen=set();branch=branch_id
    while branch is not None:
        planning.require(branch not in seen,'cyclic native branch ancestry');seen.add(branch)
        if branch in admissions:return admissions[branch]
        parent=record.get('parents',{}).get(branch)
        if isinstance(parent,dict) and parent.get('reset') is True:break
        branch=parent.get('branchId') if isinstance(parent,dict) else None
    return record.get('taskAdmission')


def require_owned_admission(scene,record,case_hash,value,*,check=lambda:None):
    # Preserve the legacy equality fast path and its hot replay/hash behavior.
    if value==record.get('taskAdmission'):return
    history=validate_workspace(scene,record,case_hash,check=check)
    planning.require(any(semantic_digest(v)==semantic_digest(value) for v in _admissions(record,history).values()),
        'task admission is not owned by an installed native preparation')


def bind_for_branch(engine,record,case_hash,*,check=lambda:None):
    if record.get('preparationComparisons') or record.get('preparationApplications'):
        value=admission_for_branch(engine.scene,record,case_hash,record['branchId'],check=check)
    else:value=record.get('taskAdmission')
    if value is None:engine.__dict__.pop('_planning_task_admission',None)
    else:bind_admission(engine,value)


def planning_view(engine,record,case_hash,*,check=lambda:None):
    history=validate_workspace(engine.scene,record,case_hash,check=check)
    if not history:return None
    item=history[-1];report=item['report']
    reports={entry['revision']:entry['report'] for entry in history};applications=[]
    for app in record.get('preparationApplications',[]):
        row=next(r for r in reports[app['revision']]['rows'] if r['id']==app['candidateId'])
        selected=row['preparation']['publicPlan']['decision']['selectedIndex']
        applications.append({**copy.deepcopy(app),'widthMm':row['widthMm'],
            'preparationSummary':copy.deepcopy(row['preparation']['publicPlan']['attempts'][selected]['summary'])})
    return {**copy.deepcopy(report),'revision':item['revision'],'sourceBranchId':item['sourceBranchId'],
        'sourcePosition':report['prefixActions'],'current':semantic_digest(engine.snapshot())==report['sourceSnapshotHash'],
        'applications':applications}


def protected_branches(record):
    return {b for item in record.get('preparationComparisons',[]) for b in (item['sourceBranchId'],item['configuredBranchId'])} | {
        b for app in record.get('preparationApplications',[]) for b in (app['branchId'],app['preparationBranchId'])}

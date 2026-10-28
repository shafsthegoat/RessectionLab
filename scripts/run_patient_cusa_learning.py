#!/usr/bin/env python3
"""Fit and execute a bounded observed CUSA continuation on a saved DEVELOPMENT branch.

The fixed 30 mm3 / two-stroke task is declared before fitting. Training and
frozen execution use the same source branch and public physics. This is a
within-case engineering demonstration, not clinical validation or a held-out
generalization estimate. Source archives are never overwritten.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import platform
import sys
import traceback

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
import torch
import numpy
import scipy
import resectionlab.patient_cusa_learning as policy

from resectionlab.core import semantic_digest
from resectionlab.imaging import save_case
from resectionlab.patient_cusa_learning import (SCHEMA, Actor, Critic, Budget, ACTION_IDS,
    CONTROLLER_REVISION, APPLY_RETRY_DURATIONS_SECONDS,
    TARGET_GOAL_MM3, NON_TARGET_CAP_MM3, MIN_ADMITTED_ACTIONS, DECISION_CAP,
    actor_choice, execute, fit_bc, parameter_sha256, ppo_update, prepare,
    scripted_choice, sha256)
from resectionlab.surgical_engine import SurgicalEngine
from resectionlab.surgical_planning import validate_saved_comparison
from resectionlab.surgical_verified_cache import implementation_fingerprint
from resectionlab.workspace_bundle import load_workspace, save_workspace


def write_json(path,value):
    Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def producer_identity():
    """Bind the producer sources and the engine's loaded implementation seal."""
    engine = implementation_fingerprint()
    if engine is None:
        raise ValueError('Authoritative surgical engine fingerprint unavailable')
    return {'policyModuleSourceSha256':sha256(policy.__file__),
            'runnerSourceSha256':sha256(__file__),
            'authoritativeEngineFingerprint':engine,
            'runtime':{'python':sys.version,'pythonCacheTag':sys.implementation.cache_tag,
                       'platform':platform.platform(),'numpy':numpy.__version__,
                       'scipy':scipy.__version__,'torch':str(torch.__version__)}}


def physics_audit(scene):
    profile=scene.properties.get('tissuePhysicsV2')
    if not isinstance(profile,dict) or profile.get('schema')!='resectionlab.tissue-physics.v2':
        raise ValueError('CUSA continuation requires explicit tissue physics v2')
    return {'schema':profile['schema'],'profileHash':semantic_digest(profile),
            'sceneId':scene.identity}


def development_declaration(prepared, *, registry_root=None):
    """Bind a saved source patient/revision to its pre-pixel role assignment."""
    adapter=prepared['case'].metadata.get('surgical_adapter',{})
    manifest=adapter.get('sourceManifest',{})
    patient=adapter.get('sourcePatient')
    series=manifest.get('series',())
    revisions={row.get('source_mirror_revision') for row in series}
    if (not isinstance(patient,str) or not patient or
            manifest.get('patient_id')!=patient or not series or
            any(row.get('patient_id')!=patient for row in series) or
            len(revisions)!=1 or None in revisions or
            prepared['record']['planning'].get('sourceSceneId')!=prepared['scene'].identity):
        raise ValueError('Saved source lacks a consistent DEVELOPMENT patient/revision binding')
    root=Path(registry_root) if registry_root is not None else ROOT/'testcases/downloaded-datasets'
    matches=[]
    for path in sorted(root.glob('*/development-declaration.json')):
        declaration=json.loads(path.read_text())
        if patient not in declaration.get('cases',()):
            continue
        if (declaration.get('schema')!='resectionlab.fresh-source-development-declaration.v1'
                or declaration.get('role')!='DEVELOPMENT'
                or declaration.get('mirror_revision')!=next(iter(revisions))
                or not declaration.get('selection_rule') or not declaration.get('created_at')):
            raise ValueError('Matching saved source has a conflicting DEVELOPMENT declaration')
        matches.append((path,declaration))
    if len(matches)!=1:
        raise ValueError('Saved source has no unique pre-pixel DEVELOPMENT declaration')
    path,declaration=matches[0]
    return {'schema':declaration['schema'],'role':'DEVELOPMENT',
            'patientId':patient,'mirrorRepository':declaration['mirror_repository'],
            'mirrorRevision':declaration['mirror_revision'],
            'declarationPath':str(path.resolve()),'declarationSha256':sha256(path),
            'selectionRule':declaration['selection_rule'],
            'savedComparisonSeal':prepared['record']['planning']['comparisonSeal']}


def frozen_actor(path,expected,*,scene=None):
    checkpoint=torch.load(path,map_location='cpu',weights_only=True)
    proof=checkpoint.get('trainingReceipt')
    current_producer=producer_identity()
    if checkpoint.get('producerIdentity')!=current_producer or not isinstance(proof,dict) or \
            proof.get('producerIdentity')!=current_producer:
        raise ValueError('Frozen CUSA policy, runner, engine, or runtime implementation differs')
    fit_physics=checkpoint.get('fitPhysicsAudit')
    if (not isinstance(fit_physics,dict) or fit_physics.get('schema')!='resectionlab.tissue-physics.v2'
            or not isinstance(fit_physics.get('profileHash'),str)
            or not isinstance(checkpoint.get('sourceBinding'),dict)
            or fit_physics.get('sceneId')!=checkpoint['sourceBinding'].get('sceneId')
            or proof.get('fitPhysicsAudit')!=fit_physics):
        raise ValueError('Frozen CUSA fitting physics audit is missing or inconsistent')
    if scene is not None:
        current_physics=physics_audit(scene)
        if (current_physics['schema']!=fit_physics['schema'] or
                current_physics['profileHash']!=fit_physics['profileHash']):
            raise ValueError('Frozen CUSA physics profile differs from the fitted source')
    if (checkpoint['schema']!=SCHEMA or checkpoint['actorSha256']!=expected or
            checkpoint.get('controllerRevision')!=CONTROLLER_REVISION or
            proof.get('controllerRevision')!=CONTROLLER_REVISION or
            not isinstance(checkpoint.get('sourceBinding'),dict) or
            not isinstance(proof,dict) or not proof.get('actualPublicTransitions') or
            checkpoint.get('observationSchema')!='surgical-rl.v2-tool-frame-task-v1' or
            checkpoint.get('actionIds')!=ACTION_IDS):
        raise ValueError('Frozen CUSA actor metadata differs')
    if proof.get('sourceBinding')!=checkpoint['sourceBinding'] or checkpoint.get('taskGate')!={
            'incrementalTargetGoalMm3':TARGET_GOAL_MM3,
            'minAdmittedNonStopActions':MIN_ADMITTED_ACTIONS,
            'incrementalNonTargetCapMm3':NON_TARGET_CAP_MM3}:
        raise ValueError('Frozen CUSA task/source binding differs')
    if checkpoint['stage']=='bc':
        fit=proof.get('bcFit',{})
        demo=proof.get('demonstration',{})
        if (fit.get('finalActorSha256')!=expected or fit.get('initialActorSha256')==expected or
                demo.get('status')!='success' or demo.get('admittedNonStopActions',0)<MIN_ADMITTED_ACTIONS):
            raise ValueError('Frozen BC lacks actual fitted source actions')
    elif checkpoint['stage']=='bc-initialized-ppo':
        updates=proof.get('optimizerUpdates',[])
        episodes=proof.get('sampledEpisodes',[])
        if (not updates or not episodes or updates[-1].get('actorAfterSha256')!=expected or
                any(not update.get('actorChanged') or
                    update.get('actorBeforeSha256')==update.get('actorAfterSha256')
                    for update in updates)):
            raise ValueError('Frozen PPO lacks actual actor updates and sampled source actions')
    else:
        raise ValueError('Unknown frozen CUSA fitting stage')
    actor=Actor();actor.load_state_dict(checkpoint['state_dict'],strict=True);actor.eval()
    if parameter_sha256(actor)!=expected:
        raise ValueError('Frozen CUSA actor parameters differ')
    return actor


def seal_actor(actor,path,stage,source_binding,training_receipt,producer,fit_physics,scene):
    identity=parameter_sha256(actor)
    torch.save({'schema':SCHEMA,'stage':stage,'actorSha256':identity,
                'controllerRevision':CONTROLLER_REVISION,
                'state_dict':actor.state_dict(),
                'sourceBinding':source_binding,
                'producerIdentity':producer,'fitPhysicsAudit':fit_physics,
                'observationSchema':'surgical-rl.v2-tool-frame-task-v1',
                'actionIds':ACTION_IDS,
                'taskGate':{'incrementalTargetGoalMm3':TARGET_GOAL_MM3,
                            'minAdmittedNonStopActions':MIN_ADMITTED_ACTIONS,
                            'incrementalNonTargetCapMm3':NON_TARGET_CAP_MM3},
                'trainingReceipt':training_receipt},path)
    frozen_actor(path,identity,scene=scene)
    return {'stage':stage,'path':str(path.resolve()),'fileSha256':sha256(path),
            'actorSha256':identity,
            'trainingReceiptSha256':semantic_digest(training_receipt),
            'producerIdentity':producer,'fitPhysicsAudit':fit_physics,
            'fitSourceSceneId':source_binding['sceneId'],
            'fitPrefixStateId':source_binding['stateId'],
            'optimizerUpdates':len(training_receipt.get('optimizerUpdates',[]))}


def verify_episode(prepared,episode,budget):
    native=episode['engine']
    budget.reserve(len(native.actions))
    replay=SurgicalEngine.restore(prepared['scene'],native.snapshot())
    if (replay.state_id!=native.state_id or replay.state!=native.state or
            native.actions[:prepared['binding']['position']]!=prepared['prefix'].actions):
        raise ValueError('Learned continuation differs from independent full native replay')
    return {'exactNativeReplay':True,'stateId':replay.state_id,
            'actions':len(replay.actions)}


def episode_failure_reasons(episode):
    """Keep a missed task gate distinct from failed checked cleanup."""
    if episode['status']=='success':
        if not episode['taskGateMet'] or not episode['cleanupComplete']:
            raise ValueError('Positive CUSA outcome contradicts task or cleanup evidence')
        return []
    if episode['status'] not in ('abstained','incomplete'):
        raise ValueError('Unknown evaluated CUSA episode status')
    reasons=[]
    if episode['deltaTargetMm3']<TARGET_GOAL_MM3:
        reasons.append('target_progress_below_declared_30_mm3_goal')
    if episode['admittedNonStopActions']<MIN_ADMITTED_ACTIONS:
        reasons.append('fewer_than_two_admitted_non_stop_actions')
    if not episode['cleanupComplete']:
        reasons.append('checked_cleanup_incomplete')
    if episode['status']=='incomplete':
        reasons.append('episode_incomplete_'+str(episode['stopReason']))
    return reasons or ['declared_task_gate_unpassed']


def persist_derived(prepared,variants,output,budget,filename):
    case=prepared['case']
    artifacts=copy.deepcopy(prepared['artifacts'])
    record=artifacts['surgical_toolbox']
    entries=[]
    for stage,episode,actor_receipt in variants:
        outcome=episode['receipt']
        failures=episode_failure_reasons(outcome)
        identity='learned-cusa-'+stage+('-unpassed-' if failures else '-')+actor_receipt['actorSha256'][:10]
        if identity in record['branches']:
            raise ValueError('Derived learned branch already exists')
        record['branches'][identity]=episode['engine'].snapshot()
        record.setdefault('parents',{})[identity]={'branchId':prepared['binding']['branchId'],
                                                   'position':prepared['binding']['position']}
        entries.append({'schema':SCHEMA,'method':stage,'branchId':identity,
            'controllerRevision':CONTROLLER_REVISION,
            'parentBranchId':prepared['binding']['branchId'],
            'parentPosition':prepared['binding']['position'],
            'sourceStateId':prepared['binding']['stateId'],
            'sourceSceneId':prepared['binding']['sceneId'],
            'sourceArchiveSha256':prepared['binding']['sourceSha256'],
            'sourceRoleDeclaration':prepared['developmentDeclaration'],
            'modelSha256':actor_receipt['actorSha256'],
            'checkpointFileSha256':actor_receipt['fileSha256'],
            'trainingReceiptSha256':actor_receipt['trainingReceiptSha256'],
            'producerIdentitySha256':semantic_digest(actor_receipt['producerIdentity']),
            'fitPhysicsAudit':actor_receipt['fitPhysicsAudit'],
            'fitSourceSceneId':actor_receipt['fitSourceSceneId'],
            'fitPrefixStateId':actor_receipt['fitPrefixStateId'],
            'observationSchema':'surgical-rl.v2-tool-frame-task-v1',
            'actionIds':ACTION_IDS,
            'status':outcome['status'],
            'taskGateMet':outcome['taskGateMet'],
            'failureReasons':failures,
            'promotionStatus':('no-clinical-promotion-development-only' if not failures
                else 'not-promoted-unpassed-development-episode'),
            'stopReason':outcome['stopReason'],
            'deltaTargetMm3':outcome['deltaTargetMm3'],
            'deltaNonTargetMm3':outcome['deltaNonTargetMm3'],
            'admittedNonStopActions':outcome['admittedNonStopActions'],
            'cleanupComplete':outcome['cleanupComplete'],
            'finalStateId':outcome['finalStateId'],
            'actorDecisions':[row['actionId'] for row in outcome['decisions']],
            'recoveryAttempts':[
                {'actionId':row.get('actionId',row.get('action',{}).get('operation')),
                 'attempts':row['attempts']}
                for section in ('decisions','cleanup') for row in outcome[section]
                if row.get('attempts') and len(row['attempts'])>1],
            'optimizerUpdates':actor_receipt['optimizerUpdates'],
            'limitations':['Unreviewed brain estimate; source enhancing tumor annotation',
                           'Skull, vascular, functional and clinical completeness unknown',
                           'Within-case development demonstration, no superiority or clinical claim']})
    record['learnedExecutions']=entries
    record['learnedExecution']=entries[0]
    validate_saved_comparison(prepared['scene'],record,record['planning'])
    destination=output/filename
    workspace=prepared['workspace']
    if workspace is None:
        save_case(case,destination,artifacts=artifacts)
    else:
        save_workspace(case,destination,sources=workspace['sources'],imaging=workspace['imagingState'],
            episode=workspace['episode'],selection=workspace['selection'],
            originals=workspace['originals'],artifacts=artifacts)
    reopened_case,reopened_artifacts,_=load_workspace(destination)
    reopened_record=reopened_artifacts['surgical_toolbox']
    validate_saved_comparison(prepared['scene'],reopened_record,reopened_record['planning'])
    for (stage,episode,_),entry in zip(variants,entries):
        branch_id=entry['branchId']
        budget.reserve(len(reopened_record['branches'][branch_id]['actions']))
        reopened=SurgicalEngine.restore(prepared['scene'],reopened_record['branches'][branch_id])
        if reopened_case.semantic_hash!=case.semantic_hash or reopened.state_id!=episode['engine'].state_id:
            raise ValueError('Derived '+stage+' native branch changed on save/reopen')
    return {'workspacePath':str(destination.resolve()),'sha256':sha256(destination),
            'branches':entries,'parentBranchId':prepared['binding']['branchId'],
            'parentPosition':prepared['binding']['position'],
            'reopenedExact':True}


def run(source,branch,output,*,seed=17,max_wall_seconds=600.,max_calls=200,
        max_rss_bytes=3*1024**3,profile_only=False,frozen_model=None,source_role):
    if source_role!='DEVELOPMENT':
        raise ValueError('This prototype requires an explicit DEVELOPMENT source role')
    budget=Budget(max_wall_seconds,max_calls,max_rss_bytes)
    receipt={'schema':SCHEMA,'controllerRevision':CONTROLLER_REVISION,
        'revisionContext':'Post-frozen-0005-failure DEVELOPMENT controller revision; v4 frozen failures remain the prespecified transfer evidence.',
        'status':'running',
        'source':{'path':str(Path(source).resolve()),'sha256':sha256(source)},
        'task':{'incrementalTargetGoalMm3':TARGET_GOAL_MM3,
                'diagnosticTargetGoalMm3':5.,
                'minAdmittedNonStopActions':MIN_ADMITTED_ACTIONS,
                'incrementalNonTargetCapMm3':NON_TARGET_CAP_MM3,
                'decisionCap':DECISION_CAP,'role':source_role,
                'baseTransitionCap':160,'extraRecoveryAndVerificationCalls':40,
                'totalPhysicalTransitionCap':max_calls,
                'applyShaftCollisionRetryDurationsSeconds':APPLY_RETRY_DURATIONS_SECONDS,
                'withdrawShaftCollisionRetry':'same endpoint and quaternion at min(30, max(initial, distance/2)) then 30 seconds',
                'policyInput':'published SurgicalRLEnv local/tool/metric observation plus declared task budget'},
        'trials':[], 'optimizerUpdates':0,
        'limitations':['Same DEVELOPMENT case supplies demonstration, policy fitting and live execution; no held-out performance claim.',
            'Source enhancing tumor label is an annotation; the brain layer is an unreviewed reconstruction estimate.',
            'Missing skull, vessels, functional anatomy and calibrated tissue mechanics remain unknown.']}
    output=Path(output)
    try:
        prepared=prepare(source,branch,budget=budget,seed=seed)
        receipt['prefix']=prepared['binding']
        prepared['developmentDeclaration']=development_declaration(prepared)
        receipt['roleDeclaration']=prepared['developmentDeclaration']
        producer=producer_identity()
        fit_physics=physics_audit(prepared['scene'])
        receipt['producerIdentity']=producer
        receipt['physicsAudit']=fit_physics
        if frozen_model is not None:
            checkpoint=torch.load(frozen_model,map_location='cpu',weights_only=True)
            identity=checkpoint['actorSha256']
            frozen=frozen_actor(frozen_model,identity,scene=prepared['scene'])
            asset={'stage':checkpoint['stage'],'controllerRevision':CONTROLLER_REVISION,
                   'path':str(Path(frozen_model).resolve()),
                   'fileSha256':sha256(frozen_model),'actorSha256':identity,
                   'trainingReceiptSha256':semantic_digest(checkpoint['trainingReceipt']),
                   'fitSourceSceneId':checkpoint['sourceBinding']['sceneId'],
                   'fitPrefixStateId':checkpoint['sourceBinding']['stateId'],
                   'producerIdentity':checkpoint['producerIdentity'],
                   'fitPhysicsAudit':checkpoint['fitPhysicsAudit'],
                   'optimizerUpdates':len(checkpoint['trainingReceipt'].get('optimizerUpdates',[])),
                   'fitSource':'separate DEVELOPMENT case; no update on this source'}
            receipt['model']=asset
            trial=execute(prepared,actor_choice(frozen),budget=budget,method='frozen-transfer-live')
            receipt['trials'].append(trial['receipt'])
            receipt['execution']=trial['receipt']
            receipt['exactReplay']=verify_episode(prepared,trial,budget)
            stage='ppo' if checkpoint['stage']=='bc-initialized-ppo' else 'bc'
            receipt['derived']=persist_derived(prepared,[(stage,trial,asset)],output,
                budget,'frozen-transfer.ressectionlab')
            receipt['status']='evaluated'
            return receipt
        demonstration=execute(prepared,scripted_choice,budget=budget,method='fixed-tool-frame-script')
        receipt['trials'].append(demonstration['receipt'])
        receipt['profile']={'status':demonstration['receipt']['status'],
            'deltaTargetMm3':demonstration['receipt']['deltaTargetMm3'],
            'deltaNonTargetMm3':demonstration['receipt']['deltaNonTargetMm3'],
            'publicPreviewCommitChecked':True}
        if profile_only:
            receipt['profile']['exactReplay']=verify_episode(prepared,demonstration,budget)
            receipt['status']='profiled'
            return receipt
        if demonstration['receipt']['status']!='success':
            raise ValueError('Frozen generic script did not meet the declared pre-fit task gate')
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            untrained=Actor();untrained.eval()
        receipt['untrainedActorSha256']=parameter_sha256(untrained)
        baseline=execute(prepared,actor_choice(untrained),budget=budget,method='untrained-same-architecture')
        receipt['trials'].append(baseline['receipt'])
        actor,bc_fit=fit_bc(demonstration,seed=seed)
        receipt['bcFit']=bc_fit
        bc_training={'actualPublicTransitions':demonstration['receipt']['nativeActions'],
            'sourceBinding':prepared['binding'],'demonstration':demonstration['receipt'],
            'producerIdentity':producer,'fitPhysicsAudit':fit_physics,
            'controllerRevision':CONTROLLER_REVISION,
            'bcFit':bc_fit,'role':'same-case DEVELOPMENT behavior cloning'}
        bc_asset=seal_actor(actor,output/'bc-policy.pt','bc',prepared['binding'],bc_training,
            producer,fit_physics,prepared['scene'])
        receipt['bcModel']=bc_asset
        frozen=frozen_actor(output/'bc-policy.pt',bc_asset['actorSha256'],scene=prepared['scene'])
        bc=execute(prepared,actor_choice(frozen),budget=budget,method='frozen-bc-live')
        receipt['trials'].append(bc['receipt'])
        receipt['bcReplay']=verify_episode(prepared,bc,budget)
        if bc['receipt']['status']!='success':
            raise ValueError('Frozen BC failed the same live physical task gate')
        receipt['execution']=bc['receipt']
        receipt['model']=bc_asset
        receipt['bcDerived']=persist_derived(prepared,[('bc',bc,bc_asset)],output,budget,
            'bc-continuation.ressectionlab')
        if sha256(source)!=prepared['binding']['sourceSha256']:
            raise ValueError('Source archive changed during learning')
        # BC-initialized PPO is additional within-case evidence. The frozen BC
        # branch is retained as a successful comparator, with no promotion claim.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            critic=Critic()
        generator=torch.Generator().manual_seed(seed)
        for update in range(2):
            episodes=[]
            for episode_index in range(2):
                sampled=execute(prepared,actor_choice(actor,sample=True,generator=generator),
                    budget=budget,method=f'ppo-sampled-{update}-{episode_index}')
                receipt['trials'].append(sampled['receipt'])
                episodes.append(sampled)
            update_receipt=ppo_update(actor,critic,episodes,seed=seed+update)
            receipt.setdefault('ppoUpdates',[]).append(update_receipt)
            receipt['optimizerUpdates']+=1
        ppo_training={'actualPublicTransitions':sum(t['nativeActions'] for t in receipt['trials']
            if t['method'].startswith('ppo-sampled-')),
            'sourceBinding':prepared['binding'], 'bcCheckpointSha256':bc_asset['fileSha256'],
            'producerIdentity':producer,'fitPhysicsAudit':fit_physics,
            'controllerRevision':CONTROLLER_REVISION,
            'sampledEpisodes':[t for t in receipt['trials'] if t['method'].startswith('ppo-sampled-')],
            'optimizerUpdates':receipt['ppoUpdates'],
            'role':'same-case DEVELOPMENT BC-initialized PPO; actor hash separately verified'}
        ppo_asset=seal_actor(actor,output/'ppo-policy.pt','bc-initialized-ppo',
            prepared['binding'],ppo_training,producer,fit_physics,prepared['scene'])
        receipt['ppoModel']=ppo_asset
        ppo_frozen=frozen_actor(output/'ppo-policy.pt',ppo_asset['actorSha256'],scene=prepared['scene'])
        ppo=execute(prepared,actor_choice(ppo_frozen),budget=budget,method='frozen-ppo-live')
        receipt['trials'].append(ppo['receipt'])
        receipt['ppoReplay']=verify_episode(prepared,ppo,budget)
        receipt['ppoExecution']=ppo['receipt']
        receipt['derived']=persist_derived(prepared,[('bc',bc,bc_asset),('ppo',ppo,ppo_asset)],
            output,budget,'learned-continuations.ressectionlab')
        if sha256(source)!=prepared['binding']['sourceSha256']:
            raise ValueError('Source archive changed after cold learned branch export')
        receipt['status']='saved'
        return receipt
    except BaseException as error:
        receipt['status']='failed'
        receipt['failure']={'type':type(error).__name__,'message':str(error),
                            'traceback':traceback.format_exc(limit=8)}
        raise
    finally:
        receipt['budget']=budget.report()
        receipt['source']['unchanged']=sha256(source)==receipt['source']['sha256']
        write_json(output/'receipt.json',receipt)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--branch',required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--seed',type=int,default=17)
    parser.add_argument('--source-role',choices=['DEVELOPMENT'],required=True,
                        help='Explicit predeclared source role; no clinical or held-out claim')
    parser.add_argument('--profile-only',action='store_true')
    parser.add_argument('--frozen-model',type=Path,
                        help='Evaluate an existing frozen actor without fitting or updating it')
    args=parser.parse_args()
    output=args.output.resolve()
    if 'testcases' not in output.parts or output.exists():
        parser.error('Choose a new output directory under ignored testcases/')
    output.mkdir(parents=True)
    if args.profile_only and args.frozen_model:
        parser.error('Choose profile-only or frozen-model, not both')
    result=run(args.source,args.branch,output,seed=args.seed,
               profile_only=args.profile_only,frozen_model=args.frozen_model,
               source_role=args.source_role)
    print(json.dumps({'status':result['status'],'task':result['task'],
        'bc':result.get('execution',{}).get('status'),
        'ppo':result.get('ppoExecution',{}).get('status'),
        'budget':result['budget']}))


if __name__=='__main__':main()

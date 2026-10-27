"""Read-only modeled consequences from authoritative surgical engine states.

All geometry stays in the saved RAS+ source frame. These are simulated local
effects and source annotation volumes, not calibrated injury, neurological
outcome, vascular perfusion, or clinical risk estimates.
"""
from __future__ import annotations

from pathlib import Path
from typing import Sequence
import hashlib
import math

import numpy as np

from .core import array_digest, semantic_digest
from .surgical_engine import SurgicalEngine
from .surgical_transition_meter import MeteredEngine
from .surgical_environment import load_environment
from .surgical_planning import validate_saved_comparison
from .workspace_bundle import load_workspace

SCHEMA = 'resectionlab.surgical-modeled-outcomes.v1'
MAX_TARGET_CELLS = 2_000_000
MAX_BRANCHES = 8
MAX_TOTAL_ACTIONS = 1024


def _measured(value, unit):
    number = float(value)
    if not math.isfinite(number):
        raise ValueError('Nonfinite surgical outcome metric')
    return {'status':'measured','value':number,'unit':unit}


def _unavailable(reason, unit=None, *, status='unavailable'):
    return {'status':status,'value':None,'unit':unit,'reason':reason}


def _vector(points):
    if not len(points):
        return None
    return [float(x) for x in np.mean(points,axis=0)]


def _bounds(points):
    if not len(points):
        return None
    return {'minRasMm':[float(x) for x in np.min(points,axis=0)],
            'maxRasMm':[float(x) for x in np.max(points,axis=0)]}


def _target_distribution(scene, target_ids, removed, baseline_removed):
    if not target_ids:
        return _unavailable('no_source_target_region','mm3')
    flat = scene.labels.ravel()
    ids = np.flatnonzero(np.isin(flat, target_ids))
    if len(ids) > MAX_TARGET_CELLS:
        raise ValueError('Target spatial summary exceeds bounded cell limit')
    if not len(ids):
        return _unavailable('mapped_target_region_has_no_source_cells','mm3')
    points = scene.points(ids)
    center = np.mean(points,axis=0)
    removed_mask = np.isin(ids,np.fromiter(removed,dtype=np.int64,count=len(removed)))
    baseline_mask = np.isin(ids,np.fromiter(baseline_removed,dtype=np.int64,count=len(baseline_removed)))
    if np.any(baseline_mask & ~removed_mask):
        raise ValueError('Baseline target removal is absent from endpoint')
    octant = ((points[:,0]>=center[0]).astype(np.int8)*4 +
              (points[:,1]>=center[1]).astype(np.int8)*2 +
              (points[:,2]>=center[2]).astype(np.int8))
    source_counts = np.bincount(octant,minlength=8)
    removed_counts = np.bincount(octant[removed_mask],minlength=8)
    baseline_counts = np.bincount(octant[baseline_mask],minlength=8)
    octants=[]
    for code in range(8):
        octants.append({'region':('R' if code&4 else 'L')+'-'+
            ('A' if code&2 else 'P')+'-'+('S' if code&1 else 'I'),
            'sourceMm3':float(source_counts[code]*scene.voxel_mm3),
            'removedMm3':float(removed_counts[code]*scene.voxel_mm3),
            'incrementalRemovedMm3':float((removed_counts[code]-baseline_counts[code])*scene.voxel_mm3),
            'residualMm3':float((source_counts[code]-removed_counts[code])*scene.voxel_mm3)})
    residual_points=points[~removed_mask]
    removed_points=points[removed_mask]
    total=float(len(ids)*scene.voxel_mm3)
    removed_volume=float(np.count_nonzero(removed_mask)*scene.voxel_mm3)
    return {'status':'measured','sourceVolumeMm3':total,
        'removedMm3':removed_volume,
        'incrementalRemovedMm3':float((np.count_nonzero(removed_mask)-
            np.count_nonzero(baseline_mask))*scene.voxel_mm3),
        'residualMm3':total-removed_volume,
        'fractionRemoved':removed_volume/total,
        'sourceCentroidRasMm':[float(x) for x in center],
        'removedCentroidRasMm':_vector(removed_points),
        'residualCentroidRasMm':_vector(residual_points),
        'sourceBoundsRasMm':_bounds(points),
        'residualBoundsRasMm':_bounds(residual_points),
        'octantOrigin':'source target centroid in RAS+ mm; ties go to positive axis',
        'octants':octants}


def _count_by_label(scene, cells):
    if not cells:
        return {}
    ids=np.fromiter(cells,dtype=np.int64,count=len(cells))
    if np.any(ids<0) or np.any(ids>=scene.labels.size):
        raise ValueError('Removed cell lies outside the source scene')
    values,counts=np.unique(scene.labels.ravel()[ids],return_counts=True)
    return {int(value):int(count) for value,count in zip(values,counts)}


def _metrics_group(metrics, baseline_metrics, names, unit):
    return {name:{'endpoint':_measured(metrics[name],unit),
                  'baseline':_measured(baseline_metrics[name],unit),
                  'delta':_measured(metrics[name]-baseline_metrics[name],unit)}
            for name in names}


def _source_regions(scene):
    layers=(scene.source.get('adapter') or {}).get('sourceLayers',{})
    rows=[]
    for region in scene.regions:
        layer=layers.get(region['name'],{})
        provenance=layer.get('provenance') or {}
        origin=('source_annotation' if provenance.get('kind')=='observed_source_annotation'
            else 'model_estimate' if 'inference' in provenance else 'unclassified_source_layer')
        rows.append({'labelId':region['id'],'name':region['name'],
            'material':region['material'],'target':bool(region.get('target')),
            'origin':origin,'reviewStatus':layer.get('reviewStatus','unknown'),
            'sampledMaskHash':layer.get('sampledMaskHash')})
    return rows


def summarize_engine_outcomes(engine: SurgicalEngine, *, baseline: SurgicalEngine | None = None) -> dict:
    """Summarize one actual state; baseline defaults to intact same-scene state.

    Input is an engine, never a patient mask or a claimed planner score. A
    supplied baseline must share the scene and precede the endpoint removals.
    Scalar ledgers are labeled with their simulator units; unsupported anatomy
    is null with an explicit reason rather than a reassuring zero.
    """
    # The joint planner meters public transitions on a plain delegating proxy.
    # This summary is read-only; unwrap only that exact proxy type so the
    # authoritative engine type check below keeps rejecting arbitrary objects.
    if type(engine) is MeteredEngine:
        engine=engine.unwrap()
    if type(baseline) is MeteredEngine:
        baseline=baseline.unwrap()
    if not isinstance(engine,SurgicalEngine):
        raise TypeError('An authoritative SurgicalEngine is required')
    engine.scene.require_cache_source()
    base = baseline if baseline is not None else engine.fresh()
    if not isinstance(base,SurgicalEngine) or base.scene.identity!=engine.scene.identity:
        raise ValueError('Outcome baseline must share the exact source scene')
    base.scene.require_cache_source()
    if (len(engine.actions)!=len(engine.history) or
            len(base.actions)!=len(base.history) or
            engine.actions[:len(base.actions)]!=base.actions or
            (engine.actions and engine.state_id!=engine.history[-1]['stateAfter']) or
            (base.actions and base.state_id!=base.history[-1]['stateAfter']) or
            (base.actions and base.state_id!=engine.history[len(base.actions)-1]['stateAfter']) or
            (not base.actions and base.state_id!=engine.fresh().state_id) or
            (not engine.actions and engine.state_id!=base.state_id)):
        raise ValueError('Outcome baseline must be an exact native action prefix')
    scene=engine.scene
    removed=set(map(int,engine.state['removed']))
    baseline_removed=set(map(int,base.state['removed']))
    if len(removed)!=len(engine.state['removed']) or not baseline_removed<=removed:
        raise ValueError('Endpoint removal must extend the baseline without duplicate cells')
    counts=_count_by_label(scene,removed)
    base_counts=_count_by_label(scene,baseline_removed)
    source_labels,source_counts=np.unique(scene.labels,return_counts=True)
    source_count={int(key):int(value) for key,value in zip(source_labels,source_counts)}
    target_ids=tuple(r['id'] for r in scene.regions if r.get('target'))
    non_target=[]
    mapped_materials=set()
    for region in scene.regions:
        label=region['id']
        if not source_count.get(label):
            continue
        mapped_materials.add(region['material'])
        if region.get('target'):
            continue
        source_volume=source_count[label]*scene.voxel_mm3
        removed_volume=counts.get(label,0)*scene.voxel_mm3
        non_target.append({'labelId':label,'name':region['name'],
            'material':region['material'],
            'sourceMm3':float(source_volume),
            'removedMm3':float(removed_volume),
            'incrementalRemovedMm3':float((counts.get(label,0)-base_counts.get(label,0))*scene.voxel_mm3),
            'residualMm3':float(source_volume-removed_volume)})
    metrics=engine.metrics()
    baseline_metrics=base.metrics()
    target=_target_distribution(scene,target_ids,removed,baseline_removed)
    if target['status']=='measured' and not math.isclose(target['removedMm3'],
            metrics['targetRemovedMm3'],abs_tol=1e-7):
        raise ValueError('Target source cells disagree with authoritative engine metrics')
    non_volume=sum(row['removedMm3'] for row in non_target)
    if not math.isclose(non_volume,metrics['nonTargetRemovedMm3'],abs_tol=1e-7):
        raise ValueError('Non-target source cells disagree with authoritative engine metrics')
    exposure=metrics['exposureByMaterialCellSeconds']
    baseline_exposure=baseline_metrics['exposureByMaterialCellSeconds']
    contact={material:{'endpoint':_measured(exposure.get(material,0.),'cell*s'),
        'baseline':_measured(baseline_exposure.get(material,0.),'cell*s'),
        'delta':_measured(exposure.get(material,0.)-baseline_exposure.get(material,0.),'cell*s')}
        for material in sorted(mapped_materials)}
    mapped={'brain':'soft_tissue' in mapped_materials,
            'skull':'bone' in mapped_materials,
            'vessels':'vessel' in mapped_materials,
            'target':target['status']=='measured'}
    guard=engine.guard_report()
    protected_cells=sum(int(row.get('sourceCells',0)) for row in guard.get('structures',[]))
    effective_protection=bool(guard['configured'] and not guard['missingRequired'] and protected_cells)
    protection={'status':'available_model_only' if effective_protection else 'unavailable',
        'value':guard['contractId'] if effective_protection else None,
        'reason':None if effective_protection
            else 'required_protection_labels_missing_or_no_nonempty_declared_structure',
        'contractId':guard.get('contractId'),
        'missingRequired':list(guard['missingRequired']),
        'declaredProtectedSourceCells':protected_cells,
        'unknownStructures':list(guard.get('unknownStructures',[]))}
    v2=bool(scene.properties.get('tissuePhysicsV2'))
    if mapped['brain'] and v2:
        deformation={name:_measured(metrics[name],unit) for name,unit in (
            ('brainActiveNodes','cells'),('brainMaxDisplacementMm','mm'),
            ('brainElasticEnergyNmm','N*mm'),
            ('brainMaxNeighborDisplacementRatio','ratio'))}
        deformation['status']='measured_model_only'
    else:
        deformation={'status':'unavailable','reason':'mapped_brain_and_tissue_physics_v2_required',
                     'value':None}
    if mapped['skull'] and v2:
        drill={name:_measured(metrics[name],unit) for name,unit in (
            ('drillInputEnergyJ','J'),('drillCuttingWorkJ','J'),
            ('drillUnusedEnergyJ','J'),('drillHeatGeneratedJ','J'),
            ('drillCoolingLossJ','J'),('drillPendingWorkJ','J'),
            ('drillRemovedVolumeMm3','mm3'),('drillMaxTemperatureC','degC'))}
        drill['status']='measured_model_only'
    else:
        drill={'status':'unavailable','reason':'mapped_skull_and_tissue_physics_v2_required',
               'value':None}
    if 'cusaV3' in engine.state:
        cusa={name:_measured(metrics[name],'mm3') for name in (
            'cusaInputCapacityMm3','cusaAspiratedSolidMm3','cusaPendingCapacityMm3',
            'cusaUnusedCapacityMm3','cusaDiscardedCapacityMm3')}
        cusa['pendingCells']=_measured(metrics['cusaPendingCells'],'cells')
        cusa['status']='measured_model_only'
    else:
        cusa={'status':'unavailable','reason':'cusa_contact_capacity_v3_not_enabled',
              'value':None}
    fluid=_metrics_group(metrics,baseline_metrics,(
        'fluidAddedMm3','fluidRemovedMm3','fluidPresentMm3','fluidEscapedMm3',
        'debrisMm3','debrisRemovedMm3','debrisEscapedMm3'),'mm3')
    if mapped['vessels'] and v2:
        vessel={name:_measured(metrics['vesselPhysicsV2'][name],unit) for name,unit in (
            ('woundCount','count'),('damagedFaceCount','count'),
            ('woundAreaMm2','mm2'),('activeWounds','count'),
            ('vascularRemainingMm3','mm3'),('refilledMm3','mm3'),
            ('generatedMm3','mm3'),('emittedMm3','mm3'),
            ('blockedOutletReservoirMm3','mm3'),
            ('conservationResidualMm3','mm3'))}
        vessel['severedVesselConnections']=_measured(metrics['severedVesselConnections'],'count')
        vessel['vesselEncounters']=_measured(metrics['vesselEncounters'],'cells')
        vessel['status']='measured_mapped_local_model_only'
    else:
        vessel={'status':'unavailable','reason':'mapped_vessels_and_tissue_physics_v2_required',
                'value':None}
    return {'schema':SCHEMA,'sceneId':scene.identity,'sourceCaseHash':scene.source.get('caseHash'),
        'sourceLabelsHash':array_digest(scene.labels),
        'frame':'RAS+','stateId':engine.state_id,'baselineStateId':base.state_id,
        'actionCount':len(engine.actions),'baselineActionCount':len(base.actions),
        'actionHistoryHash':semantic_digest(engine.actions),
        'voxelVolumeMm3':float(scene.voxel_mm3),
        'sourceRegions':_source_regions(scene),
        'mappedAnatomy':mapped,'protection':protection,
        'target':target,'nonTargetByLabel':non_target,
        'nonTargetRemovedMm3':_measured(non_volume,'mm3'),
        'incrementalNonTargetRemovedMm3':_measured(sum(row['incrementalRemovedMm3'] for row in non_target),'mm3'),
        'contact':{'workingToolCellSecondsByMaterial':contact,
            'retainedHoldOrDissectionCellSeconds':_measured(metrics['contactCellSeconds'],'cell*s'),
            'meaning':'Engine-recorded acting-tool working cells times step duration; retained contact is only holds/elevation/dissection. This is not full bimanual or swept-shaft contact.'},
        'mechanics':{'deformation':deformation,'drillWork':drill,
                     'cusaCapacity':cusa,
                     'travelMm':_metrics_group(metrics,baseline_metrics,('travelMm',),'mm')['travelMm'],
                     'elapsedSeconds':_metrics_group(metrics,baseline_metrics,('elapsedSeconds',),'s')['elapsedSeconds']},
        'thermal':{'coagulationPowerTimeExposure':_measured(metrics['thermalExposure'],'power_fraction*cell*s'),
            'meaning':'Uncalibrated coagulation exposure proxy; not a temperature or injury dose.'},
        'fluid':fluid,'localVessels':vessel,
        'globalPerfusion':_unavailable('no_anatomical_perfusion_network',status='unknown'),
        'neurologicalRisk':_unavailable('no_calibrated_neurological_outcome_model',status='unknown'),
        'limitations':['Whole-cell removal and source-estimate anatomy; missing structures stay unknown.',
            'Mechanical, thermal, fluid and vessel values are uncalibrated simulation outputs.',
            'Recorded material exposure omits comprehensive stationary-other-tool and swept-shaft contact.',
            'No patient risk, perfusion or neurological deficit probability is estimated.']}


def compare_saved_access_outcomes(path: str | Path,
    branch_ids: Sequence[str] | None = None, *, max_total_actions: int = MAX_TOTAL_ACTIONS) -> dict:
    """Independently replay saved access branches and summarize their endpoints."""
    path=Path(path).resolve()
    if type(max_total_actions) is not int or not 1<=max_total_actions<=MAX_TOTAL_ACTIONS:
        raise ValueError('Access outcome replay action bound is invalid')
    case,artifacts,_=load_workspace(path)
    scene=load_environment(path)
    record=artifacts.get('surgical_toolbox')
    if not record or record.get('planning',{}).get('task')!='anatomy-access':
        raise ValueError('A saved native anatomy-access comparison is required')
    comparison=record['planning']
    validate_saved_comparison(scene,record,comparison)
    plans={row['branchId']:row for row in comparison['alternatives']}
    selected=list(plans) if branch_ids is None else list(branch_ids)
    if (not selected or len(selected)>MAX_BRANCHES or len(set(selected))!=len(selected)
            or any(branch not in plans or branch not in record['branches'] for branch in selected)):
        raise ValueError('Select distinct saved access comparison branches')
    total=sum(len(record['branches'][branch]['actions']) for branch in selected)
    if total>max_total_actions:
        raise ValueError('Access outcome replay exceeds action bound')
    alternatives=[]
    for branch in selected:
        engine=SurgicalEngine.restore(scene,record['branches'][branch])
        endpoint=summarize_engine_outcomes(engine)
        plan=plans[branch]
        if (endpoint['stateId']!=plan['finalStateId'] or
                endpoint['baselineStateId']!=plan['startStateId']):
            raise ValueError('Replayed access outcome differs from saved branch states')
        alternatives.append({'branchId':branch,'method':plan['method'],
            'nativeAxis':plan['route']['nativeAxis'],
            'accessCompleteness':plan['route']['accessCompleteness'],
            'stopReason':plan['stopReason'],'outcome':endpoint})
    with path.open('rb') as stream:
        archive_hash=hashlib.file_digest(stream,'sha256').hexdigest()
    return {'schema':SCHEMA,'source':{'archivePath':str(path),
        'archiveSha256':archive_hash,'caseHash':case.semantic_hash,
        'sceneId':scene.identity,'comparisonSeal':comparison['comparisonSeal'],
        'sourceLabelsHash':array_digest(scene.labels),
        'frame':'RAS+','voxelVolumeMm3':float(scene.voxel_mm3),
        'sourceRegions':_source_regions(scene)},
        'replay':{'branchCount':len(alternatives),'totalActions':total,
                  'maxTotalActions':max_total_actions,'independentNativeRestore':True},
        'alternatives':alternatives}

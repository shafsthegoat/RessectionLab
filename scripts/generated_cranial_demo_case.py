"""Independent generated source case for the desktop sequential demonstration.

Geometry and source-bound stage declarations are ported from the repository's
closed-shell engineering controls. This runnable builder imports no test code.
Motor memberships are stylized declarations supplied before any search; they
are not patient anatomy, encounter evidence, or functional-deficit estimates.
"""
from __future__ import annotations
import copy
from dataclasses import replace
import numpy as np
from scipy.spatial.transform import Rotation
from resectionlab.core import CaseData, SourceRef, semantic_digest, thaw_json
from resectionlab.geometry import GeometryScene, capsule_voxel_indices
from resectionlab.surgical_scene import SurgicalScene, from_case
from resectionlab.surgical_engine import SurgicalEngine
from resectionlab.surgical_craniotomy import PROPERTY, SCHEMA, binding, contract_hash
from resectionlab import surgical_cranial_stages as stages


def pose(x=32., y=32., z=55.):
    return {'position': [x, y, z], 'quaternion': [0., 0., 0., 1.]}


def request(operation, *, slot='left', **kwargs):
    return {'slot': slot, 'operation': operation, 'duration': .1, **kwargs}


def shell(*, proof_status='reviewed', obstacle=None, source_gap=False, protected=False, attachments=()):
    """Freeze the source-cell contour before executing any native command."""
    labels = np.zeros((64, 64, 96), np.uint16)
    labels[20:45, 20:45, 60] = 1
    labels[20:45, 20:45, 75] = 1
    labels[20, 20:45, 60:76] = labels[44, 20:45, 60:76] = 1
    labels[20:45, 20, 60:76] = labels[20:45, 44, 60:76] = 1
    labels[31:34, 31:34, 67:70] = 2
    if source_gap:
        labels[38:43, 30:35, 60] = 0
    if obstacle is not None:
        labels[obstacle] = 3
    regions = [{'id': 1, 'name': 'Idealized shell', 'material': 'bone'},
               {'id': 2, 'name': 'Generated target', 'material': 'tumor', 'target': True}]
    if obstacle is not None:
        regions.append({'id': 3, 'name': 'Transport obstacle', 'material': 'soft_tissue', 'protected': protected})
    source = {'kind': 'generated closed-shell craniotomy control; not a patient', 'evidenceClass': 'generated-engineering'}
    anchor = int(np.ravel_multi_index((20, 20, 75), labels.shape))
    properties = {'attachments': copy.deepcopy(list(attachments)), 'boneAnchors': [anchor]}
    base = SurgicalScene(labels, np.eye(4), regions, source, properties)
    contour = [pose(x, y, 58.3) for x, y in ((26, 26), (38, 26), (38, 38), (26, 38), (26, 26))]
    grid = GeometryScene(np.zeros(labels.shape, bool), np.eye(4))
    kerf = set()
    for start, end in zip(contour, contour[1:]):
        indices = capsule_voxel_indices(grid, start['position'], end['position'], 2.)
        ids = np.ravel_multi_index(indices.T, labels.shape)
        kerf.update(map(int, ids[labels.ravel()[ids] == 1]))
    flap = set(map(int, np.flatnonzero(labels.ravel()))) & set(
        int(np.ravel_multi_index((x, y, 60), labels.shape)) for x in range(29, 36) for y in range(29, 36))
    bone = set(map(int, np.flatnonzero(labels.ravel() == 1)))
    proof = {'status': proof_status, 'provenance': 'Explicit idealized generated shell, no patient qualification',
             'evidenceSha256': '1'*64 if proof_status == 'reviewed' else None}
    contract = {'schema': SCHEMA, **binding(base), 'windowCells': sorted(flap | kerf),
        'flapCells': sorted(flap), 'kerfCells': sorted(kerf),
        'retainedSkullCells': sorted(bone-flap-kerf), 'boneAnchorCells': [anchor],
        'entry': {'centerRasMm': [32., 32., 59.5], 'normalRasUnit': [0., 0., 1.],
                  'uRasUnit': [1., 0., 0.], 'halfExtentsMm': [10., 10.],
                  'maximumInwardMm': 30., 'maximumAngleDegrees': 30.},
        'exit': {'axis': 2, 'side': -1, 'planeVoxelBoundary': float(min(np.nonzero(labels)[2])-1),
                 'minimumVoxelBoundary': [-.5, -.5, -.5],
                 'maximumVoxelBoundary': [63.5, 63.5, 95.5]},
        'evidence': {'scope': 'idealized-engineering', **{name: copy.deepcopy(proof)
                      for name in ('window', 'support', 'attachments', 'exit')}}}
    contract['contentHash'] = contract_hash(contract)
    properties[PROPERTY] = contract
    scene = SurgicalScene(labels, np.eye(4), regions, source, properties)
    if protected:
        from resectionlab.surgical_rl_protection import GuardConfig, ProtectionSet
        # Isolate actual whole-flap contact with zero additional clearance,
        # declared before any source action; no patient settings are changed.
        engine = SurgicalEngine(scene, protection=ProtectionSet(scene, clearance_mm=0., required=()),
                                guard_config=GuardConfig(require_critical_coverage=False))
    else:
        engine = SurgicalEngine(scene)
    return engine, contour, flap, kerf


def stage_source(*,unknown=False,attachment=False,anchors=True,obstacle=None,protected=False,curved=False):
    engine,contour,_,_=shell(obstacle=obstacle)
    membrane_id=max(x['id'] for x in engine.scene.regions)+1
    labels=engine.scene.labels.copy();labels[25:40,25:40,64]=membrane_id
    if curved:
        labels[labels==membrane_id]=0
        for x in range(25,40):
            for y in range(25,40):
                if 29<=x<36 and 29<=y<36:
                    height=64+max(0,max(abs(x-32),abs(y-32))-1)
                    labels[x,y,64:height+1]=membrane_id
                else:labels[x,y,66]=membrane_id
    p=tuple(int(np.ravel_multi_index((x,y,z),labels.shape)) for x in range(29,36) for y in range(29,36) for z in range(64,67) if labels[x,y,z]==membrane_id)
    all_mem=set(map(int,np.flatnonzero(labels==membrane_id)));r=tuple(sorted(all_mem-set(p)))
    regions=engine.scene.regions+[{'id':membrane_id,'name':'Generated membrane surrogate','material':'membrane'}]
    props=copy.deepcopy(engine.scene.properties);raw=props.pop(PROPERTY)
    if attachment:props['attachments'].append({'id':'generated-soft-membrane-attachment','kind':'soft','cells':[p[len(p)//2]]})
    base=SurgicalScene(labels,np.eye(4),regions,engine.scene.source,props)
    raw.update(binding(base));raw['contentHash']=contract_hash(raw);props[PROPERTY]=raw
    scene=SurgicalScene(labels,np.eye(4),regions,engine.scene.source,props)
    if protected:
        from resectionlab.surgical_rl_protection import ProtectionAnnotation,ProtectionSet,GuardConfig
        annotation=ProtectionAnnotation.from_mask(scene,'generated-protected-obstacle',labels==3,provenance='Generated hard exclusion outside the membrane holder path; no patient coverage')
        protection=ProtectionSet(scene,[annotation],clearance_mm=0.,required=())
        source=SurgicalEngine(scene,protection=protection,guard_config=GuardConfig(require_critical_coverage=False))
    else:source=SurgicalEngine(scene)
    proof={'status':'unknown' if unknown else 'reviewed','provenance':'Generated categorical membrane transport control, not clinical dura/scalp evidence','evidenceSha256':None if unknown else '2'*64}
    evidence={k:copy.deepcopy(proof) for k in stages.PROOFS}
    w=tuple(map(int,np.flatnonzero(labels==2)))
    contract=stages.freeze_cranial_stage_contract(source,membrane_cells=p,retained_membrane_cells=r,membrane_anchor_cells=(r[0],) if anchors else (),work_cells=w,evidence=evidence)
    return source,contract,contour,p,r


def joint_source(**kwargs):
    source,contract,contour,p,r=stage_source(**kwargs)
    engine=SurgicalEngine(source.scene,protection=source.protection,guard_config=source.guard_config,cranial_stages=contract)
    return engine,contract,contour,p,r


def bone_actions(contour):
    a=[request('configure',mechanicsRevision=2,clockRevision=1),request('place',pose=pose(26,26,55))]
    a += [request('apply',pose=p) for p in contour]
    a += [request('withdraw',pose=pose(26,26,55)),request('exchange',instrument='bone_flap_holder'),
        request('place',pose=pose(),parameters={'jawOpeningMm':0.}),request('grasp',pose=pose(z=59.5)),
        request('move',pose=pose(z=54.5)),request('extract'),request('withdraw',pose=pose(z=54.5))]
    return a


def membrane_cut_actions(engine,contract):
    """Finite source-edge inventory, fixed stock tool; no outcome-tuned poses."""
    a=[request('exchange',instrument='dural_scissors')]
    for i,j in contract.boundary_edges:
        u,v=engine.scene.points([i,j]);mid=(u+v)/2
        axis=int(np.argmax(np.abs(v-u)));q=Rotation.from_euler('z',90,degrees=True).as_quat().tolist() if axis==0 else [0.,0.,0.,1.]
        outside={'position':[float(mid[0]),float(mid[1]),55.],'quaternion':q}
        cutting={'position':[float(mid[0]),float(mid[1]),float(mid[2]-.2)],'quaternion':q}
        a += [request('place',pose=outside,parameters={'jawOpeningMm':0.,'depthMm':contract.scissors_depth_mm}),request('apply',pose=cutting),request('withdraw',pose=outside)]
    return a


def membrane_grasp_actions(hand='left'):
    return [request('exchange',slot=hand,instrument='membrane_forceps'),request('place',slot=hand,pose=pose(),parameters={'jawOpeningMm':0.}),request('grasp',slot=hand,pose=pose(z=63.5))]


def preparation_program(engine, contract, contour):
    return (bone_actions(contour)+membrane_cut_actions(engine, contract)+
            membrane_grasp_actions()+[request('move', pose=pose(z=54.5)),
                request('extract'), request('withdraw', pose=pose(z=54.5))])


def cleanup_program():
    """Collect actual kerf debris with stock suction and explicit corner dwell.

    Centerline sweeps cover the straight kerf. Offset corner poses cover its
    source-cell corner footprint. Dwell is real modeled time at each pose.
    """
    actions = [request('exchange', instrument='frazier_suction'),
        request('place', pose=pose(26.,26.,55.))]
    for x,y in ((26.,26.),(38.,26.),(38.,38.),(26.,38.),(26.,26.)):
        actions.append(request('apply', pose=pose(x,y,60.), duration=4.))
    for x,y in ((25.8,25.8),(38.2,25.8),(38.2,38.2),(25.8,38.2)):
        actions.extend([request('apply', pose=pose(x,y,60.), duration=1.),
            request('apply', pose=pose(x,y,60.), duration=1.)])
    actions.append(request('withdraw', pose=pose(25.8,38.2,55.)))
    return actions


def build_generated_cranial_case(neural_limit=0., *, cerebellar_shortcut=False):
    """Prospective anatomy/route inventory; no expected outcome enters search."""
    from resectionlab.surgical_engine import SurgicalEngine
    from resectionlab.surgical_craniotomy import binding, contract_hash
    from resectionlab.surgical_cranial_stages import freeze_cranial_stage_contract
    from resectionlab.surgical_cranial_planning_roots import candidate_scene, create_registry, source_binding, FIELD
    from resectionlab.surgical_procedure_policy import PROPERTY, make_procedure_policy
    template, stage, contour, _, _ = joint_source()
    labels = template.scene.labels.copy()
    labels[33, 32, 66] = labels[22, 22, 70] = labels[23, 22, 70] = 4
    neural_position=(34,32,67) if cerebellar_shortcut else (33,32,66)
    if cerebellar_shortcut:
        # Keep this generated shortcut below the supplied membrane checkpoint;
        # source preparation itself must remain clear of the excluded anatomy.
        labels[33,32,66]=0
        labels[neural_position]=4
    regions = template.scene.regions + [dict(id=4, name='Declared generated neural tissue', material='soft_tissue')]
    props = copy.deepcopy(template.scene.properties)
    opening = props.pop('cranialAccessV1')
    case = CaseData(case_id='generated-structure-aware-approaches', mri=labels.astype(np.float32),
        compartments={r['name']: labels == r['id'] for r in regions}, affine=template.scene.affine,
        source_refs=(SourceRef('generated', 'synthetic://structure-aware-approaches', provenance='simulated'),),
        metadata={'generated': True, 'is_synthetic': True,
            'surgical_adapter': {'type': 'generated-engineering-control'}, 'surgical_properties': props,
            'anatomy_layers': {r['name']: {'kind': r['material']} for r in regions}})
    base = from_case(case, regions)
    flat = lambda xyz: int(np.ravel_multi_index(xyz, labels.shape))
    intracranial = [flat((x,y,z)) for x in range(21,44) for y in range(21,44) for z in range(61,75)]
    evidence = dict(provenance='Independent generated closed-shell anatomy and superior entry declaration before search; no acquired anatomy',
        evidenceSha256=semantic_digest({'intracranial': intracranial, 'source': base.source}).split(':')[-1],
        origin='independent-source-declaration', reviewStatus='generated-engineering')
    structures = {name: dict(evidence, kind='neural', cells=[flat(xyz)], coverageCells=intracranial)
        for name, xyz in [('brainstem',(22,22,70)), ('cerebellum',(23,22,70)), ('supratentorial-neural',neural_position)]}
    # Declared independently of route execution. The shared cell deliberately
    # preserves overlapping cortex/CST/supratentorial memberships.
    structures['motor-cortex'] = dict(evidence, kind='neural', cells=[flat(neural_position)],
        displayLabel='Generated motor cortex', coverageCells=intracranial)
    structures['corticospinal-tract'] = dict(evidence, kind='neural', cells=[flat(neural_position)],
        displayLabel='Generated corticospinal tract', coverageCells=intracranial)
    for name in ('motor-cortex','corticospinal-tract'):
        declaration = structures[name]
        declaration['provenance'] = ('Independent stylized generated ' + name +
            ' membership supplied before planning; shared source-cell overlap is deliberate; no patient functional evidence')
        declaration['evidenceSha256'] = semantic_digest({'source':base.source,
            'structure':name,'cells':declaration['cells'],'coverageCells':intracranial}).split(':')[-1]
    critical = {flat((22,22,70)), flat((23,22,70))}
    if cerebellar_shortcut:
        # Distinct generated source anatomy supplied before any search.
        structures['cerebellum']['cells'].append(flat(neural_position))
        structures['cerebellum']['cells'].sort()
        structures['supratentorial-neural']['cells']=[]
        critical.add(flat(neural_position))
    permitted = sorted(set(intracranial)-critical)
    entry = [flat((x,y,61)) for x in range(24,41) for y in range(24,41)]
    raw = make_procedure_policy(base, policy_id='generated-supratentorial-harm-v1', evidence=evidence,
        structures=structures, intracranial_cells=intracranial, observed_cells=range(labels.size),
        entry_regions={'superior-entry': dict(evidence,cells=entry)},
        permitted_compartments={'supratentorial-work': dict(evidence,cells=permitted),
            'bone-preparation':dict(evidence,cells=list(map(int,np.flatnonzero(labels==1)))),
            'membrane-preparation':dict(evidence,cells=list(map(int,np.flatnonzero(labels==3))))},
        work_domain=dict(evidence, minimumVoxelBoundary=[-.5]*3, maximumVoxelBoundary=(np.asarray(labels.shape)-.5).tolist()),
        tissue_permissions={r['material']: {'remove':True,'cut':True} for r in regions},
        neural_max_mm3=neural_limit, bone_max_mm3=285., preparation_nonbone_max_mm3=49., modeled_time_max_seconds=240.,
        structure_max_mm3={'brainstem':0.,'cerebellum':0.,'supratentorial-neural':neural_limit,'motor-cortex':neural_limit,'corticospinal-tract':neural_limit}, start_stage='post-scalp')
    metadata = thaw_json(case.metadata); metadata['surgical_properties'][PROPERTY] = raw
    case = replace(case, metadata=metadata)
    base = from_case(case, regions)
    opening.update(binding(base)); opening['contentHash'] = contract_hash(opening)
    scene = candidate_scene(base,'cranialAccessV1',opening)
    native = SurgicalEngine(scene)
    contract = freeze_cranial_stage_contract(native, membrane_cells=stage.membrane_cells,
        retained_membrane_cells=stage.retained_membrane_cells, membrane_anchor_cells=stage.membrane_anchor_cells,
        work_cells=sorted([*stage.work_cells,flat(neural_position)]), evidence=stage.as_dict()['evidence'])
    native = SurgicalEngine(scene, protection=native.protection, guard_config=native.guard_config, cranial_stages=contract)
    opening_actions = bone_actions(contour)
    prep = preparation_program(native,contract,contour)[len(opening_actions):] + cleanup_program() + [request('exchange',instrument='cusa'), request('place',pose=pose())]
    center = base.points(sorted(contract.original_target_cells)).mean(axis=0)
    routes = [{'id':name, 'motion':request('apply',pose=pose(float(center[0]+offset),float(center[1]),66.5),duration=30.)}
        for name,offset in [('central',0.),('lateral',-1.5)]]
    planning = {'limits': {'max_bone_mm3':285.,'max_preparation_nonbone_mm3':49.,'max_work_nonbone_mm3':20.,
        'max_native_actions':300,'max_public_calls':1800,'max_wall_seconds':240.,
        'max_bone_contact_replans':6,'max_tissue_contact_replans':8,'export_replay_reserve_calls':350},
        'openingSkill':{'id':'source-opening','kind':'opening','actions':opening_actions}, 'preparationActions':prep,
        'workRoutes':routes,'outsidePoses':{}, 'qualification':{'status':'reviewed','sourceBinding':source_binding(base),
            'evidenceSha256':evidence['evidenceSha256'],'blockers':[], 'provenance':'Generated engineering declaration only; no patient qualification'}}
    roots = [{'id':'superior-opening','label':'Declared superior approaches','openingProperty':'cranialAccessV1',
        'openingContract':opening,'sourceSnapshot':native.snapshot(),'planning':planning}]
    common = SurgicalEngine(base).snapshot()
    record = {'mapping':regions,'branches':{'main':common},'branchId':'main','position':0,
        'parents':{'main':None},'timeline':{'main':[]},FIELD:create_registry(base,roots)}
    return case, base, record

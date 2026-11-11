"""Portable generated acceleration controls; never acquired or patient evidence.

Promoted from the existing bone/membrane/stage test controls. Geometry and stock
programs are fixed independently of backend outcomes; no pytest imports.
"""
import copy
import numpy as np
from scipy.spatial.transform import Rotation
from .core import CaseData, SourceRef
from .geometry import GeometryScene, capsule_voxel_indices
from .surgical_craniotomy import PROPERTY, SCHEMA, binding, contract_hash
from .surgical_engine import SurgicalEngine
from .surgical_scene import SurgicalScene, from_case
from . import surgical_cranial_stages as stages


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
    source = {'kind': 'generated closed-shell craniotomy control; not a patient'}
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


def cranial_generate_fixture(*, physics='declared_legacy'):
    """Fresh source + one qualified generated root consumed by real Generate.

    Legacy physics is the unchanged successful engineering control. Standardized
    physics is a separate negative control, with its original inner-table stop.
    """
    if physics not in ('declared_legacy', 'standardized'):
        raise ValueError('Unknown generated physics control')
    from .surgical_environment import standardize, SCHEMA as ENVIRONMENT_SCHEMA
    from .surgical_cranial_planning_roots import FIELD, candidate_scene, create_registry, source_binding
    template, stage, contour, _, _ = joint_source()
    props = copy.deepcopy(template.scene.properties)
    opening = props.pop(PROPERTY)
    masks = {r['name']: template.scene.labels == r['id'] for r in template.scene.regions}
    # Nonconstant display data passes the real image validator. Values have no
    # anatomical meaning and never determine material/stage declarations.
    x, y, z = np.indices(template.scene.shape, dtype=np.float32)
    image = (x + 2*y + 3*z + 8*np.sin(x/7)*np.cos(y/9)).astype(np.float32)
    case = CaseData(case_id='generated-cpp-acceleration-v1-' + physics,
        mri=image, compartments=masks, affine=template.scene.affine,
        source_refs=(SourceRef('generated', 'synthetic://cpp-acceleration-v1', provenance='simulated'),),
        metadata={'generated': True, 'is_synthetic': True, 'surgical_properties': props,
            'anatomy_layers': {r['name']: {'kind': r['material']} for r in template.scene.regions}})
    base = from_case(case, template.scene.regions)
    if physics == 'standardized':
        base = standardize(base)
    opening.update(binding(base)); opening['contentHash'] = contract_hash(opening)
    scene = candidate_scene(base, PROPERTY, opening)
    bound = stages.freeze_cranial_stage_contract(SurgicalEngine(scene),
        membrane_cells=stage.membrane_cells, retained_membrane_cells=stage.retained_membrane_cells,
        membrane_anchor_cells=stage.membrane_anchor_cells, work_cells=stage.work_cells,
        evidence=stage.as_dict()['evidence'])
    source = SurgicalEngine(scene, cranial_stages=bound)
    preparation = (membrane_cut_actions(source, bound) + membrane_grasp_actions() +
        [request('move', pose=pose(z=54.5)), request('extract'), request('withdraw', pose=pose(z=54.5)),
         request('exchange', instrument='cusa'), request('place', pose=pose())])
    opening_actions = bone_actions(contour)
    if physics == 'standardized':
        for action in opening_actions:
            if action['operation'] == 'apply':
                action['duration'] = 15.
    planning = {'limits': {'max_bone_mm3': 285., 'max_preparation_nonbone_mm3': 49.,
        'max_work_nonbone_mm3': 20., 'max_native_actions': 300, 'max_public_calls': 1400,
        'max_wall_seconds': 600., 'max_bone_contact_replans': 6,
        'max_tissue_contact_replans': 8, 'export_replay_reserve_calls': 300},
        'openingSkill': {'id': 'opening', 'kind': 'opening', 'actions': opening_actions},
        'preparationActions': preparation, 'outsidePoses': {},
        'qualification': {'status': 'reviewed', 'sourceBinding': source_binding(base),
            'evidenceSha256': '3'*64, 'blockers': [],
            'provenance': 'Fixed generated software control only; no patient qualification'},
        'workRoutes': [
            {'id': 'unreached-long-work', 'motion': request('apply', pose=pose(z=68.), duration=.05)},
            {'id': 'target-work', 'motion': request('apply', pose=pose(z=66.5), duration=30.)}]}
    values = [{'id': 'opening-0', 'label': 'Generated complete joint sequence',
        'openingProperty': PROPERTY, 'openingContract': opening,
        'sourceSnapshot': source.snapshot(), 'planning': planning}]
    record = {'mapping': base.regions,
        'branches': {'source': SurgicalEngine(base).snapshot()}, 'branchId': 'source', 'position': 0,
        FIELD: create_registry(base, values)}
    if 'tissuePhysicsV2' in base.properties:
        record['environmentSchema'] = ENVIRONMENT_SCHEMA
        record['physicsProfile'] = base.properties['tissuePhysicsV2']
    return case, base, values, record


def continuum_fixture(n=13):
    """Bounded genuine viscoelastic coupon with hold, exposure and topology work."""
    from .surgical_continuum import PROPERTY as CONTINUUM_PROPERTY, research_profile
    from .surgical_physics_v2 import default_profile
    from . import surgical_deformation_v2 as brain
    labels = np.ones((n, n, n), np.uint16)
    material = {'shearPa': 1000., 'poissonRatio': .3, 'relaxationShearPa': [600.],
        'source': 'Generated numerical control coefficient; not a patient measurement'}
    scene = SurgicalScene(labels, np.eye(4), [{'id': 1, 'name': 'Generated continuum coupon',
        'material': 'soft_tissue'}], {'kind': 'generated numerical continuum control'},
        {'tissuePhysicsV2': default_profile(), CONTINUUM_PROPERTY:
            research_profile({'1': material}, relaxation_seconds=[.5])})
    engine = SurgicalEngine(scene)
    cell = int(np.ravel_multi_index((n//2, n//2, n//2), labels.shape))
    center = [float(n//2)] * 3
    brain.begin_hold(engine, 'left', [cell], center)
    brain.set_target(engine, 'left', [center[0] + .01, *center[1:]])
    return engine, cell

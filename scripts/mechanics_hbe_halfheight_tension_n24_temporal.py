"""Exact retained N24 tension/S120 adapter, not a generic study selector."""
from copy import deepcopy
import hashlib
import xml.etree.ElementTree as ET
import numpy as np
from scripts import mechanics_hbe_halfheight_spatial as spatial
from scripts import mechanics_hbe_readout as original_readout
from scripts import mechanics_hbe_halfheight_temporal_common as common

access, BASE, backend = spatial.access, spatial.BASE, spatial.backend
DECLARATION_PATH = 'manifests/experiments/hbe-01-03-halfheight-tension-n24-temporal-v1.json'
DECLARATION_SHA256 = 'eefc84557f68dce12f6dd41b72600ae60da1eae6d1dab63dc3de9561a6be8d8e'
FINGERPRINT = '7fca6d38400eda3593a8950ea808e6a1d3f5cb6db4b742a318084e98f06ad9c1'
RUN_ID = 'tension:N24:S120:reference'
TIMES = common.TIMES


def require_study(study):
    if spatial.original.fingerprint(study) != FINGERPRINT:
        raise ValueError('Exact frozen tension N24 temporal declaration required')


def declaration(root, binding):
    if binding != {'path': DECLARATION_PATH, 'sha256': DECLARATION_SHA256}:
        raise ValueError('Exact tension temporal declaration binding required')
    study = access.verify_binding(root, binding, maximum_bytes=1024**2, read_json=True)
    require_study(study)
    return study


def geometry(root, study):
    require_study(study)
    b = study['baseline']
    old = spatial.declaration(root, b['declaration'])
    extraction = spatial.load_extraction(root, old, 24)
    half = access.verify_binding(root, b['primitive_bindings']['mesh'], read_json=True)
    wrapper = access.verify_binding(root, b['reconstruction'], read_json=True)
    expected = spatial.reconstruction(b['declaration'], b['source_bindings'], extraction,
                                       b['full_mesh'], b['primitive_bindings']['mesh'])
    if half != extraction['mesh'] or access.canonical_json(wrapper) != access.canonical_json(expected):
        raise ValueError('Original N24 geometry/reconstruction provenance differs')
    for key in ('halfheight_spatial', 'halfheight_mesh'):
        access.verify_binding(root, b['source_bindings'][key], maximum_bytes=1024**2)
    full = access.verify_binding(root, b['full_mesh'], read_json=True)
    protocol = access.verify_binding(root, b['original_protocol'], read_json=True)
    spatial.require_protocol(protocol)
    return protocol, extraction, wrapper, full


def contents(root, study, protocol, extraction):
    """Original generator plus accepted axial half constraints; only S120 changes."""
    require_study(study)
    xml, loading = BASE.specimen_deck(extraction['mesh'], 'tension', 120, 1000., protocol)
    tree = ET.fromstring(xml)
    boundary, retained = tree.find('Boundary'), 0
    for bc in list(boundary):
        if bc.attrib['node_set'].startswith('top_node_'):
            if bc.findtext('dof') in ('x','y'):
                boundary.remove(bc)
            elif bc.findtext('dof') == 'z':
                bc.find('value').text = '0.5'
                retained += 1
            else:
                raise ValueError('Unexpected midplane DOF')
    if retained != 1777:
        raise ValueError('Exact N24 midplane count required')
    ET.indent(tree, space='  ')
    skyline = ET.tostring(tree, encoding='unicode', xml_declaration=True)+'\n'
    deck = backend.transform_deck(skyline.encode())
    b = study['baseline']
    execution = access.verify_binding(root, b['execution'], read_json=True)
    for binding, new in ((execution['backend_source_deck'], skyline), (b['primitive_bindings']['deck'], deck)):
        access.verify_binding(root, binding, maximum_bytes=16*1024**2)
        common.verify_schedule_only_change(access.local_path(root, binding['path']).read_bytes(), new)
    loading.update(temporal_declaration_sha256=DECLARATION_SHA256, baseline_declaration=b['declaration'],
        protocol_sha256=b['original_protocol']['sha256'], mesh_sha256=b['primitive_bindings']['mesh']['sha256'],
        full_mesh_sha256=b['full_mesh']['sha256'], reconstruction_sha256=b['reconstruction']['sha256'],
        symmetry_model='lower_half_height_axial', full_height_m=common.H, retained_height_m=common.H/2,
        prescribed_dofs={'bottom':'xyz','midplane':'z'}, artificial_midplane_tangential_dofs='free',
        full_reaction_scale=1., full_energy_scale=2., deck_sha256=hashlib.sha256(deck.encode()).hexdigest())
    return skyline, deck, loading


def verify_primitives(root, primitives):
    if set(primitives) != common.inherited.PRIMITIVE_KEYS:
        raise ValueError('Exact six native temporal primitive bindings required')
    for key, binding in primitives.items():
        limit = 256*1024**2 if key in ('nodes','elements') else 16*1024**2
        access.verify_binding(root, binding, maximum_bytes=limit)


def read_run(root, study_binding, primitives):
    primitives = deepcopy(primitives)
    study = declaration(root, study_binding)
    verify_primitives(root, primitives)
    b = study['baseline']
    if primitives['mesh'] != b['primitive_bindings']['mesh']:
        raise ValueError('Exact accepted N24 mesh required')
    protocol, extraction, wrapper, full = geometry(root, study)
    _, deck, loading = contents(root, study, protocol, extraction)
    if (access.local_path(root, primitives['deck']['path']).read_bytes() != deck.encode()
            or access.verify_binding(root, primitives['loading'], read_json=True) != loading):
        raise ValueError('Temporal deck/loading changed original physics')
    model = common.inherited.HalfHeightReconstruction(full, extraction['mesh'], wrapper['mapping'])
    if len(model.Xh) != 12439 or model.half.element_count != 10368:
        raise ValueError('Exact N24 native dimensions required')
    report = common.read_frames(root, primitives, model, branch='tension')
    verify_primitives(root, primitives)
    for binding in (study_binding,b['full_mesh'],b['reconstruction'],b['original_protocol'],
                    wrapper['mesh_source'],wrapper['extraction_source']):
        access.verify_binding(root,binding)
    return dict(report, schema='hbe-tension-n24-temporal-readout-v1', mesh_N=24,
        protocol_sha256=b['original_protocol']['sha256'], declaration=study_binding,
        reconstruction=b['reconstruction'], primitive_bindings=primitives)


def metric_pass(metric):
    return metric['actual'] < metric['limit'] if metric.get('comparison') == 'lt' else metric['actual'] <= metric['limit']


def compare(rows, fine, study, old_comparison):
    require_study(study)
    if set(rows) != {8,12,16,24}:
        raise ValueError('Exactly four authenticated tension S60 references required')
    for N, row, steps in [(N,row,60) for N,row in rows.items()]+[(24,fine,120)]:
        F, U = np.asarray(row['applied_force_N']), np.asarray(row['probe_displacements_m'])
        coordinate = np.asarray(row['load_coordinate_m' if N>=16 else 'load_coordinate'])
        times = np.linspace(0.,1.,steps+1)
        expected = times*.15*common.H if N>=16 else times*(.15*common.H)
        if ((row.get('branch'),row.get('mesh_N'),row.get('steps'),row.get('mu_Pa'),row.get('frame_count'))
                != ('tension',N,steps,1000.,steps+1) or row.get('passed') is not True
                or row.get('protocol_sha256') != study['baseline']['original_protocol']['sha256']
                or F.shape!=(steps+1,) or U.shape!=(steps+1,75,3)
                or not np.isfinite(F).all() or not np.isfinite(U).all()
                or not np.array_equal(coordinate,expected)):
            raise ValueError('Complete finite exact-case tension common-state inputs required')
    if (fine.get('schema')!='hbe-tension-n24-temporal-readout-v1'
            or fine.get('declaration')!={'path':DECLARATION_PATH,'sha256':DECLARATION_SHA256}
            or fine.get('reconstruction')!=study['baseline']['reconstruction']
            or fine.get('full_native_equivalence_evaluated') is not False
            or any(fine.get(k,{}).get('passed') is not True for k in
                   ('half_native','reconstructed_full','reconstruction_consistency'))):
        raise ValueError('Complete native/reconstructed S120 receipt required')
    if not np.array_equal(np.asarray(fine['load_coordinate_m'])[::2],rows[24]['load_coordinate_m']):
        raise ValueError('Exact61 shared load coordinates required')
    temporal=original_readout._refinement_group(rows[24],rows[24],fine,kind='step')
    passed=all(metric_pass(v) for v in temporal.values())
    mixed,changes={},[]
    for first in (12,8):
        name=f'tension:N{first}-N16-N24'
        # Reproduce old tension metrics from compact accepted receipts first.
        old=original_readout._refinement_group(rows[first],rows[16],rows[24],kind='mesh')
        if access.canonical_json(old)!=access.canonical_json(old_comparison['groups'][name]):
            raise ValueError('Original tension spatial comparison does not reproduce')
        new=original_readout._refinement_group(rows[first],rows[16],fine,kind='mesh')
        mixed[name]={'original':old,'S120_substitution':new}
        changes.extend(name+'/'+key for key in new if metric_pass(new[key])!=metric_pass(old[key]))
    delta=np.asarray(fine['applied_force_N'])[::2]-np.asarray(rows[24]['applied_force_N'])
    candidate=passed and not changes
    return {'schema':'hbe-tension-n24-temporal-comparison-v1','study_sha256':DECLARATION_SHA256,
        'baseline_comparison':study['baseline']['comparison'],'common_state_count':61,
        'original_temporal_criteria':temporal,'original_temporal_checks_passed':passed,
        'signed_S120_minus_S60_force_N':delta.tolist(),'maximum_force_shift_state':int(np.argmax(np.abs(delta))),
        'mixed_step_groups':mixed,'mixed_step_classification_changes':changes,
        'mixed_step_scope':'Only N24 uses S120; older levels stay S60. Not a homogeneous spatial convergence study.',
        'status':'original_temporal_criteria_failed' if not passed else
                 'temporal_pass_spatial_classification_fragile' if changes else 'tension_temporal_candidate',
        'tension_temporal_candidate':candidate,'calibration_released':False,'spatial_convergence_accepted':False,
        'physical_validation_pass':None,'measured_data_accessed':False,'automatic_next_run_permitted':False,
        'historical_global_spatial_failure_preserved':True,'preserved_failures':study['preserved_failures']}

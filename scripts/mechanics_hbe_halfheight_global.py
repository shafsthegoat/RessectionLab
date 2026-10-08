"""Exact N32 geometry/deck adapter; native meshing requires released supervision.

Uses the original five-block generator, quality mathematics and half extraction.
Importing this module does not import Gmsh or prepare any geometry.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import importlib.util
from pathlib import Path
import sys
import time
import xml.etree.ElementTree as ET

from scripts import mechanics_hbe_halfheight_spatial as spatial

original, access, backend, BASE = spatial.original, spatial.access, spatial.backend, spatial.BASE
DECLARATION_PATH = 'manifests/experiments/hbe-01-03-halfheight-global-n32-v1.json'
DECLARATION_SHA256 = '902286e195a673acb6384a864f0bb57f747eb0ba450be44d2451dc221cd7d1ce'
DECLARATION_FINGERPRINT = 'd18f8b06ce10e02977a709c1ff63f5b736c93afebb60d48c3665ce60c0f73c01'
RUN_ID = 'compression:N32:S60:reference'
ORDERED_RUNS = (RUN_ID,)
COUNTS = {'full_nodes': 53329, 'full_elements': 49152, 'half_nodes': 28233,
          'half_elements': 24576, 'midplane_nodes': 3137}


def require_study(study):
    if original.fingerprint(study) != DECLARATION_FINGERPRINT:
        raise ValueError('Exact frozen single N32 diagnostic required')


def declaration(root, binding):
    if binding != {'path': DECLARATION_PATH, 'sha256': DECLARATION_SHA256}:
        raise ValueError('Exact separately frozen N32 declaration required')
    study = access.verify_binding(root, binding, maximum_bytes=1024**2, read_json=True)
    require_study(study)
    return study


def geometry_protocol(protocol, study):
    """Only the prospective mesh-level selection changes, never physics."""
    require_study(study)
    spatial.require_protocol(protocol)
    result = deepcopy(protocol)
    result['mesher']['levels'] = [{'N': 32, 'expected_nodes': 53329, 'nominal_hex8_cells': 49152}]
    return result


def extract_halfheight(full, protocol, study):
    prepared_protocol = geometry_protocol(protocol, study)
    if (type(full.get('mesh_N')) is not int or full['mesh_N'] != 32
            or full.get('geometry') != protocol['geometry']
            or (len(full['rest_nodes_m']), len(full['elements_hex8'])) != (53329, 49152)):
        raise ValueError('Exact newly prepared uniform full N32 geometry required')
    quality = BASE.mesh_quality(full, prepared_protocol)
    if (quality['passed'] is not True or quality['node_count'] != 53329
            or quality['connected_cell_count'] != 49152):
        raise ValueError('Original finest geometry gates failed')
    result = original._extract_verified(full, protocol)
    if any(result['verification'][key] != value for key, value in COUNTS.items()):
        raise ValueError('Exact N32 half/reflection counts differ')
    return dict(result, full_mesh=full, full_quality=quality)


def generate_full_mesh(root, study, protocol, output):
    """One native Gmsh call, only inside the separately released phase worker."""
    adapted = geometry_protocol(protocol, study)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    record = {'schema': 'hbe-global-n32-mesh-receipt-v1', 'status': 'starting', 'mesh_N': 32,
        'gmsh_generation_calls': 0, 'solver_calls': 0, 'measured_data_accessed': False,
        'declaration_sha256': DECLARATION_SHA256, 'protocol_sha256': study['original_protocol']['sha256'],
        'runtime_receipt': study['gmsh_runtime'], 'source_sha256': BASE.sha256(__file__),
        'generator_source_sha256': study['inherited_source_sha256']['mesh_deck']}
    BASE.write_json(output/'receipt.json', record)
    gmsh, initialized, inserted = None, False, False
    try:
        runtime = BASE.verify_runtime(access.local_path(root, study['gmsh_runtime']['path']),
                                      study['gmsh_runtime']['sha256'], protocol)
        if BASE.sha256(BASE.__file__) != record['generator_source_sha256']:
            raise ValueError('Original mesher mathematics changed')
        if 'gmsh' in sys.modules:
            raise RuntimeError('Ambient Gmsh import is forbidden')
        spec = importlib.util.spec_from_file_location('gmsh', runtime['module']['path'])
        if spec is None or spec.loader is None:
            raise RuntimeError('Private Gmsh module unavailable')
        gmsh = importlib.util.module_from_spec(spec)
        sys.modules['gmsh'] = gmsh
        inserted = True
        spec.loader.exec_module(gmsh)
        if (Path(gmsh.__file__).resolve() != Path(runtime['module']['path']).resolve()
                or Path(gmsh.lib._name).resolve() != Path(runtime['library']['path']).resolve()
                or gmsh.__version__ != runtime['version']):
            raise RuntimeError('Loaded Gmsh identity differs from frozen runtime')
        gmsh.initialize([], readConfigFiles=False, run=False)
        initialized = True
        if gmsh.option.getString('General.Version') != runtime['version']:
            raise RuntimeError('Initialized Gmsh version differs')
        for option in ('General.NumThreads', 'Mesh.MaxNumThreads1D', 'Mesh.MaxNumThreads2D', 'Mesh.MaxNumThreads3D'):
            gmsh.option.setNumber(option, 1)
        gmsh.option.setNumber('Mesh.ElementOrder', 1)
        gmsh.option.setNumber('Mesh.Binary', 1)
        geometry = BASE.build_five_block_geometry(gmsh, 32, adapted)
        record.update(gmsh_generation_calls=1, status='generating')
        BASE.write_json(output/'receipt.json', record)
        gmsh.model.mesh.generate(3)
        full = BASE.extract_mesh(gmsh, 32, adapted, geometry)
        BASE.write_json(output/'mesh.json', full)
        gmsh.write(str(output/'specimen.msh'))
        extraction = extract_halfheight(full, protocol, study)
        gmsh.finalize()
        initialized = False
        BASE.verify_runtime(access.local_path(root, study['gmsh_runtime']['path']), study['gmsh_runtime']['sha256'], protocol)
        if (BASE.sha256(__file__) != record['source_sha256']
                or BASE.sha256(BASE.__file__) != record['generator_source_sha256']):
            raise RuntimeError('Preparation source changed')
        record.update(status='prepared_not_solved', quality=extraction['full_quality'],
            full_mesh_sha256=BASE.sha256(output/'mesh.json'), native_mesh_sha256=BASE.sha256(output/'specimen.msh'),
            runtime_unchanged=True)
        return extraction
    except BaseException as error:
        record.update(status='failed_or_incomplete', error={'type': type(error).__name__, 'message': str(error)})
        raise
    finally:
        if initialized:
            try:
                gmsh.finalize()
            except BaseException as error:
                record['finalize_error'] = str(error)
        if inserted:
            sys.modules.pop('gmsh', None)
        record['elapsed_seconds'] = time.monotonic()-started
        BASE.write_json(output/'receipt.json', record)


def halfheight_deck(extraction, protocol, study):
    require_study(study)
    spatial.require_protocol(protocol)
    if any(extraction['verification'][key] != value for key, value in COUNTS.items()):
        raise ValueError('Exact prepared N32 half geometry required')
    mesh, mapping = extraction['mesh'], extraction['mapping']
    if (type(mesh.get('mesh_N')) is not int or mesh['mesh_N'] != 32
            or original.fingerprint(mesh) != mapping['half_mesh_fingerprint']
            or original.fingerprint(extraction['full_mesh']) != mapping['full_mesh_fingerprint']
            or mesh['full_geometry'] != protocol['geometry']
            or mesh['full_height_m'] != protocol['geometry']['height_m']
            or mesh['geometry'] != dict(protocol['geometry'], height_m=protocol['geometry']['height_m']/2)):
        raise ValueError('Exact N32 original-physics half-height case required')
    xml, loading = BASE.specimen_deck(extraction['mesh'], 'compression', 60, 1000., protocol)
    root = ET.fromstring(xml)
    boundaries, retained = root.find('Boundary'), 0
    for bc in list(boundaries):
        if bc.attrib['node_set'].startswith('top_node_'):
            if bc.findtext('dof') in ('x', 'y'):
                boundaries.remove(bc)
            elif bc.findtext('dof') == 'z':
                bc.find('value').text = '0.5'
                retained += 1
            else:
                raise ValueError('Unexpected artificial-midplane DOF')
    if retained != 3137:
        raise ValueError('Incomplete midplane normal constraint')
    ET.indent(root, space='  ')
    xml = ET.tostring(root, encoding='unicode', xml_declaration=True)+'\n'
    loading.update(symmetry_model='lower_half_height_axial',
        full_height_m=protocol['geometry']['height_m'], retained_height_m=protocol['geometry']['height_m']/2,
        halfheight_global_declaration_sha256=DECLARATION_SHA256,
        prescribed_dofs={'bottom': 'xyz', 'midplane': 'z'}, artificial_midplane_tangential_dofs='free',
        full_reaction_scale=1., full_energy_scale=2.,
        full_mesh_fingerprint=original.fingerprint(extraction['full_mesh']),
        half_mesh_fingerprint=original.fingerprint(extraction['mesh']),
        reconstruction_mapping_fingerprint=original.fingerprint(extraction['mapping']),
        deck_sha256=hashlib.sha256(xml.encode()).hexdigest())
    return xml, loading


def reconstruction(study_binding, sources, extraction, level):
    return {'schema': 'hbe-halfheight-global-reconstruction-v1', 'mapping': extraction['mapping'],
        'verification': extraction['verification'], 'full_quality': extraction['full_quality'],
        'full_mesh': level['full_mesh'], 'half_mesh': level['mesh'], 'declaration': study_binding,
        'mesh_source': sources['halfheight_global'], 'extraction_source': sources['halfheight_mesh']}


def case_contents(study, protocol, extraction, level):
    xml, loading = halfheight_deck(extraction, protocol, study)
    adapted = backend.transform_deck(xml.encode())
    loading.update(protocol_sha256=study['original_protocol']['sha256'], mesh_sha256=level['mesh']['sha256'],
        full_mesh_sha256=level['full_mesh']['sha256'], reconstruction_sha256=level['reconstruction']['sha256'],
        deck_sha256=hashlib.sha256(adapted.encode()).hexdigest())
    return xml, adapted, loading


def verify_prepared_case(root, study_binding, primitives, reconstruction_binding):
    study = declaration(root, study_binding)
    wrapper = access.verify_binding(root, reconstruction_binding, maximum_bytes=32*1024**2, read_json=True)
    if (set(wrapper) != {'schema', 'mapping', 'verification', 'full_quality', 'full_mesh', 'half_mesh',
                        'declaration', 'mesh_source', 'extraction_source'}
            or wrapper['schema'] != 'hbe-halfheight-global-reconstruction-v1'
            or wrapper['declaration'] != study_binding or wrapper['half_mesh'] != primitives['mesh']
            or wrapper['extraction_source']['sha256'] != study['inherited_source_sha256']['halfheight_mesh']
            or access.local_path(root, wrapper['mesh_source']['path']) != Path(__file__).resolve()
            or access.local_path(root, wrapper['extraction_source']['path']) != Path(original.__file__).resolve()):
        raise ValueError('Prepared N32 source/provenance differs')
    for name in ('mesh_source', 'extraction_source'):
        access.verify_binding(root, wrapper[name], maximum_bytes=1024**2)
    full = access.verify_binding(root, wrapper['full_mesh'], maximum_bytes=32*1024**2, read_json=True)
    protocol = access.verify_binding(root, study['original_protocol'], maximum_bytes=1024**2, read_json=True)
    extraction = extract_halfheight(full, protocol, study)
    level = {'full_mesh': wrapper['full_mesh'], 'mesh': primitives['mesh'], 'reconstruction': reconstruction_binding}
    expected = reconstruction(study_binding, {'halfheight_global': wrapper['mesh_source'],
        'halfheight_mesh': wrapper['extraction_source']}, extraction, level)
    half = access.verify_binding(root, primitives['mesh'], maximum_bytes=32*1024**2, read_json=True)
    if access.canonical_json(wrapper) != access.canonical_json(expected) or half != extraction['mesh']:
        raise ValueError('Prepared mapping/half geometry differs from complete extraction')
    _, deck, loading = case_contents(study, protocol, extraction, level)
    access.verify_binding(root, primitives['deck'], maximum_bytes=32*1024**2)
    if access.local_path(root, primitives['deck']['path']).read_bytes() != deck.encode():
        raise ValueError('Prepared N32 deck changed original physics')
    actual_loading = access.verify_binding(root, primitives['loading'], maximum_bytes=1024**2, read_json=True)
    if access.canonical_json(actual_loading) != access.canonical_json(loading):
        raise ValueError('Prepared N32 loading changed')
    return study, wrapper, half, full

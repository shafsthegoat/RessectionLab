"""One S120 deck on the immutable accepted N36 mesh; no mesher or solver.

Historical declarations and reconstruction source origins remain unchanged.
Only the separately declared static load-increment schedule changes.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
from scripts import mechanics_hbe_halfheight_global_n36 as previous

access, BASE, backend = previous.access, previous.BASE, previous.backend
original = previous.original
DECLARATION_PATH = 'manifests/experiments/hbe-01-03-halfheight-global-n36-temporal-v1.json'
DECLARATION_SHA256 = 'c726bc9299ed0cbb394a4889586df9487aa26cf6b932bae9ddc5b1db5914991f'
DECLARATION_FINGERPRINT = '09541913546b0592b17f256cc7a3f8f13e1973501c7991593ec103f1d7f3314c'
RUN_ID = 'compression:N36:S120:reference'
TIMES = np.linspace(0., 1., 121)
COUNTS = previous.COUNTS


def require_study(study):
    if original.fingerprint(study) != DECLARATION_FINGERPRINT:
        raise ValueError('Exact frozen N36 S120 temporal declaration required')


def declaration(root, binding):
    if binding != {'path': DECLARATION_PATH, 'sha256': DECLARATION_SHA256}:
        raise ValueError('Exact temporal declaration binding required')
    result = access.verify_binding(root, binding, maximum_bytes=1024**2, read_json=True)
    require_study(result)
    return result


def baseline_geometry(root, study):
    """Recheck saved geometry using the original pure extraction mathematics."""
    require_study(study)
    baseline = study['baseline']
    old_study = previous.declaration(root, baseline['declaration'])
    protocol = access.verify_binding(root, baseline['original_protocol'], read_json=True)
    previous.spatial.require_protocol(protocol)
    wrapper = access.verify_binding(root, baseline['reconstruction'], maximum_bytes=32*1024**2, read_json=True)
    full = access.verify_binding(root, baseline['full_mesh'], maximum_bytes=32*1024**2, read_json=True)
    half = access.verify_binding(root, baseline['primitive_bindings']['mesh'], maximum_bytes=32*1024**2, read_json=True)
    for key in ('halfheight_global_n36', 'halfheight_mesh'):
        access.verify_binding(root, baseline['source_bindings'][key], maximum_bytes=1024**2)
    extraction = previous.extract_halfheight(full, protocol, old_study)
    level = {'full_mesh': baseline['full_mesh'], 'mesh': baseline['primitive_bindings']['mesh'],
             'reconstruction': baseline['reconstruction']}
    expected = previous.reconstruction(baseline['declaration'], baseline['source_bindings'], extraction, level)
    if access.canonical_json(wrapper) != access.canonical_json(expected) or half != extraction['mesh']:
        raise ValueError('Accepted N36 mesh/reconstruction or historical source identity differs')
    return protocol, extraction, wrapper


def verify_schedule_only_change(old_xml, new_xml):
    """Compare all XML except the four time controls and load-curve samples."""
    old, new = ET.fromstring(old_xml), ET.fromstring(new_xml)
    paths = {'Control/time_steps': ('60', '120'),
             'Control/step_size': (format(1/60, '.17g'), format(1/120, '.17g')),
             'Control/time_stepper/dtmin': (format(1/60, '.17g'), format(1/120, '.17g')),
             'Control/time_stepper/dtmax': (format(1/60, '.17g'), format(1/120, '.17g'))}
    for path, (before, after) in paths.items():
        a, b = old.findall(path), new.findall(path)
        if len(a) != 1 or len(b) != 1 or a[0].text != before or b[0].text != after:
            raise ValueError('Only the exact S60 to S120 fixed schedule is allowed')
        b[0].text = a[0].text
    a_curves, b_curves = old.findall('LoadData/load_controller'), new.findall('LoadData/load_controller')
    if len(a_curves) != 2 or len(b_curves) != 2:
        raise ValueError('Original two affine axial load controllers required')
    for a, b in zip(a_curves, b_curves):
        ap, bp = a.find('points'), b.find('points')
        if ap is None or bp is None or len(ap) != 61 or len(bp) != 121:
            raise ValueError('Complete S60 and S120 load schedules required')
        av = np.array([[float(x) for x in pt.text.split(',')] for pt in ap])
        bv = np.array([[float(x) for x in pt.text.split(',')] for pt in bp])
        if (av.shape != (61, 2) or bv.shape != (121, 2)
                or not np.isfinite(av).all() or not np.isfinite(bv).all()
                or not np.array_equal(av, bv[::2])
                or not np.array_equal(av[:, 0], np.linspace(0., 1., 61))
                or not np.array_equal(bv[:, 0], TIMES)
                or not np.allclose(bv[:, 1], TIMES*bv[-1, 1], rtol=0., atol=2e-19)):
            raise ValueError('Load coordinates/endpoints or affine path changed')
        b.remove(bp)
        b.append(deepcopy(ap))
    # Whitespace is not physical content; every tag, attribute and value is.
    def canonical(node):
        return (node.tag, tuple(sorted(node.attrib.items())), (node.text or '').strip(),
                tuple(canonical(child) for child in node))
    if canonical(old) != canonical(new):
        raise ValueError('Non-temporal deck content changed')


def case_contents(root, study, protocol, extraction):
    require_study(study)
    baseline = study['baseline']
    xml, loading = BASE.specimen_deck(extraction['mesh'], 'compression', 120, 1000., protocol)
    tree = ET.fromstring(xml)
    boundaries, retained = tree.find('Boundary'), 0
    for bc in list(boundaries):
        if bc.attrib['node_set'].startswith('top_node_'):
            if bc.findtext('dof') in ('x', 'y'):
                boundaries.remove(bc)
            elif bc.findtext('dof') == 'z':
                bc.find('value').text = '0.5'
                retained += 1
            else:
                raise ValueError('Unexpected artificial-midplane DOF')
    if retained != 3961:
        raise ValueError('Exact N36 normal midplane constraints required')
    ET.indent(tree, space='  ')
    skyline = ET.tostring(tree, encoding='unicode', xml_declaration=True)+'\n'
    deck = backend.transform_deck(skyline.encode())
    old_execution = access.verify_binding(root, baseline['execution'], read_json=True)
    for binding in (old_execution['backend_source_deck'], baseline['primitive_bindings']['deck']):
        access.verify_binding(root, binding, maximum_bytes=32*1024**2)
    verify_schedule_only_change(access.local_path(root, old_execution['backend_source_deck']['path']).read_bytes(), skyline)
    verify_schedule_only_change(access.local_path(root, baseline['primitive_bindings']['deck']['path']).read_bytes(), deck)
    loading.update(symmetry_model='lower_half_height_axial', full_height_m=protocol['geometry']['height_m'],
        retained_height_m=protocol['geometry']['height_m']/2,
        temporal_declaration_sha256=DECLARATION_SHA256, baseline_declaration=baseline['declaration'],
        prescribed_dofs={'bottom': 'xyz', 'midplane': 'z'}, artificial_midplane_tangential_dofs='free',
        full_reaction_scale=1., full_energy_scale=2.,
        protocol_sha256=baseline['original_protocol']['sha256'],
        mesh_sha256=baseline['primitive_bindings']['mesh']['sha256'],
        full_mesh_sha256=baseline['full_mesh']['sha256'], reconstruction_sha256=baseline['reconstruction']['sha256'],
        full_mesh_fingerprint=original.fingerprint(extraction['full_mesh']),
        half_mesh_fingerprint=original.fingerprint(extraction['mesh']),
        reconstruction_mapping_fingerprint=original.fingerprint(extraction['mapping']),
        deck_sha256=hashlib.sha256(deck.encode()).hexdigest())
    return skyline, deck, loading


def verify_prepared_case(root, study_binding, primitives, reconstruction_binding):
    study = declaration(root, study_binding)
    if (set(primitives) != {'mesh', 'deck', 'loading', 'nodes', 'elements', 'solver'}
            or primitives['mesh'] != study['baseline']['primitive_bindings']['mesh']
            or reconstruction_binding != study['baseline']['reconstruction']):
        raise ValueError('Exact retained N36 geometry and six temporal primitives required')
    protocol, extraction, wrapper = baseline_geometry(root, study)
    _, deck, loading = case_contents(root, study, protocol, extraction)
    access.verify_binding(root, primitives['deck'], maximum_bytes=32*1024**2)
    actual = access.verify_binding(root, primitives['loading'], maximum_bytes=1024**2, read_json=True)
    if (access.local_path(root, primitives['deck']['path']).read_bytes() != deck.encode()
            or access.canonical_json(actual) != access.canonical_json(loading)):
        raise ValueError('Prepared temporal physics/loading identity differs')
    return study, wrapper, extraction['mesh'], extraction['full_mesh']

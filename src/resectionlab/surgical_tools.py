"""Generic research instrument geometry; no calibrated clinical mechanics."""

def _tool(name, category, geometry, materials, effect, radius=1.):
    return dict(name=name, category=category, geometry=geometry, materials=materials,
                effect=effect, radiusMm=radius, shaftRadiusMm=.5)

TOOLS = {
    'cranial_drill': _tool('Cranial drill', 'Bone', 'burr', ['bone'], 'remove', 2.),
    'cusa': _tool('CUSA / ultrasonic aspirator', 'Removal', 'aspirator', ['soft_tissue', 'tumor'], 'remove', 1.5),
    'brain_retractor': _tool('Brain retractor', 'Manipulation', 'blade', ['soft_tissue', 'tumor'], 'retract', 3.),
    'scalpel': _tool('Scalpel', 'Cutting', 'blade', ['soft_tissue', 'tumor'], 'incise'),
    'periosteal_elevator': _tool('Periosteal elevator', 'Separation', 'elevator', ['bone', 'soft_tissue'], 'elevate', 2.),
    'craniotome': _tool('Craniotome', 'Bone', 'kerf', ['bone'], 'kerf'),
    'dural_scissors': _tool('Dural scissors', 'Cutting', 'scissors', ['membrane'], 'sheet_cut'),
    'dissector': _tool('Microsurgical dissector', 'Separation', 'dissector', ['soft_tissue', 'tumor', 'membrane'], 'dissect'),
    'frazier_suction': _tool('Frazier suction', 'Fluid', 'suction', [], 'suction', 1.5),
    'bipolar_forceps': _tool('Bipolar forceps', 'Energy', 'forceps', ['soft_tissue', 'tumor', 'vessel', 'membrane'], 'coagulate'),
    'micro_scissors': _tool('Micro scissors', 'Cutting', 'scissors', ['vessel', 'soft_tissue', 'tumor'], 'transect'),
    'tumor_forceps': _tool('Tumor forceps', 'Manipulation', 'forceps', ['tumor'], 'grasp', 1.5),
    'irrigation': _tool('Irrigation', 'Fluid', 'irrigator', [], 'irrigate', 1.5),
    'cortical_probe': _tool('Cortical stimulation probe', 'Mapping', 'probe', ['soft_tissue', 'functional_region'], 'probe'),
    'subcortical_probe': _tool('Subcortical stimulation probe', 'Mapping', 'probe', ['soft_tissue', 'tumor', 'functional_region'], 'probe'),
}


def catalog():
    return [dict(v, id=k, workingLengthMm=40., approximation='generic research geometry') for k, v in TOOLS.items()]

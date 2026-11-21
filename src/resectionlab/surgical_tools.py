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
    'bone_flap_holder': _tool('Bone flap holder', 'Bone', 'forceps', ['bone'], 'grasp', 1.5),
    'irrigation': _tool('Irrigation', 'Fluid', 'irrigator', [], 'irrigate', 1.5),
    'cortical_probe': _tool('Cortical stimulation probe', 'Mapping', 'probe', ['soft_tissue', 'functional_region'], 'probe'),
    'subcortical_probe': _tool('Subcortical stimulation probe', 'Mapping', 'probe', ['soft_tissue', 'tumor', 'functional_region'], 'probe'),
    'membrane_forceps': _tool('Membrane forceps', 'Manipulation', 'forceps', ['membrane'], 'grasp', 1.5),
}


from .surgical_tool_profiles import PROFILE_INDEX, profile_segments


OPERATIVE_TOOLS = {**TOOLS,
    'raney_clip_applier': _tool('Raney scalp clip applier', 'Hemostasis', 'forceps', ['soft_tissue'], 'apply_clip'),
    'dressing_forceps': _tool('Dressing forceps', 'Field materials', 'forceps', ['soft_tissue', 'tumor', 'membrane', 'vessel'], 'place_dressing'),
    'needle_driver': _tool('Needle driver', 'Closure', 'forceps', ['soft_tissue', 'membrane'], 'approximate'),
    'fixation_driver': _tool('Bone fixation driver', 'Closure', 'driver', ['bone'], 'fixate'),
}
for _name, _radius, _shaft in (('craniotome',.75,.85), ('cranial_drill',2.,.9),
                               ('frazier_suction',1.5,1.5), ('irrigation',.55,.55)):
    OPERATIVE_TOOLS[_name] = {**TOOLS[_name], 'radiusMm':_radius, 'shaftRadiusMm':_shaft}


def tools_for_operative_profile(enabled):
    return OPERATIVE_TOOLS if enabled else TOOLS


def operative_tool_parameters(instrument, params):
    return {**params, 'radiusMm': OPERATIVE_TOOLS[instrument]['radiusMm'],
            'operativeToolProfileV1': PROFILE_INDEX[instrument]}


def catalog(operative=False):
    if operative:
        return [dict(tool, id=name, workingLengthMm=40., approximation='fixed distal research profile in mm',
                     physicalProfile={'revision':1, 'parts':[part[4] for part in profile_segments(PROFILE_INDEX[name], 2.)]})
                for name, tool in OPERATIVE_TOOLS.items()]
    return [dict(v, id=k, workingLengthMm=40., approximation='generic research geometry') for k, v in TOOLS.items()]

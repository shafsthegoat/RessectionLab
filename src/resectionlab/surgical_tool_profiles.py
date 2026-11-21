"""Fixed distal instrument profiles for the generated operative-exposure scene.

All coordinates are local mm; +Z points towards tissue. The two native contact
roles are deliberately separate from the render forms: guards and insulated
limbs obstruct motion, while only the exposed blades/tips produce contacts.
"""
from __future__ import annotations

from functools import lru_cache

# Stable explicit indices are replay data. Never derive them from a dictionary's order.
PROFILE_IDS = (
    'cranial_drill', 'cusa', 'brain_retractor', 'scalpel', 'periosteal_elevator',
    'craniotome', 'dural_scissors', 'dissector', 'frazier_suction', 'bipolar_forceps',
    'micro_scissors', 'tumor_forceps', 'bone_flap_holder', 'irrigation', 'cortical_probe',
    'subcortical_probe', 'membrane_forceps', 'raney_clip_applier', 'dressing_forceps',
    'needle_driver', 'fixation_driver',
)
PROFILE_INDEX = {name: index for index, name in enumerate(PROFILE_IDS, 1)}


@lru_cache(maxsize=256)
def profile_segments(index, aperture_mm):
    """Return (local A, local B, radius, contact role, render form) capsules.

    Each endpoint is affine in aperture. Interpolation at the middle aperture
    with a quarter aperture-change inflation encloses the entire jaw sweep.
    """
    if type(index) is not int or not 1 <= index <= len(PROFILE_IDS):
        raise ValueError('UNKNOWN_OPERATIVE_TOOL_PROFILE')
    name = PROFILE_IDS[index-1]
    half = aperture_mm/2
    parts = []
    def part(a, b, radius, form, role='working'):
        parts.append((tuple(a), tuple(b), radius, role, form))
    def shaft(end, radius=.5):
        part((0,0,-40), (0,0,end), radius, 'shaft', 'shaft')
    def axial(start, end, radius, form, role='working'):
        part((0,0,start), (0,0,end), radius, form, role)

    if name == 'cranial_drill':
        shaft(-4, .9)
        axial(-4,-1.7,.8,'chuck','shaft')
        axial(0,0,2.,'fluted-burr')
    elif name == 'craniotome':
        shaft(-5, .85)
        axial(-5,0,.75,'side-cutter')
        part((1.55,0,-5),(1.55,0,.7),.34,'guard-post','shaft')
        part((1.55,0,.7),(-.55,0,.7),.36,'guard-foot','shaft')
    elif name == 'cusa':
        shaft(0)
        axial(0,0,1.5,'aspiration-lumen')
    elif name in ('frazier_suction','irrigation'):
        radius=1.5 if name=='frazier_suction' else .55
        shaft(-4,radius)
        axial(-4,0,radius,'suction-lumen' if name=='frazier_suction' else 'irrigation-nozzle')
    elif name == 'scalpel':
        shaft(-5.8)
        # Two joined bevels provide a continuous cutting edge in the local X/Z plane.
        part((0,0,-5.8),(-.55,0,-2.4),.48,'blade-heel')
        part((-.55,0,-2.4),(0,0,0),.42,'blade-point')
    elif name in ('dural_scissors','micro_scissors'):
        length=5. if name=='dural_scissors' else 3.5
        shaft(-length)
        axial(-length,-length,.52,'pivot','shaft')
        for side in (-1,1):
            part((0,side*.09,-length),(side*half,side*.09,0),.3,'scissor-edge')
        if name=='dural_scissors':
            part((-half,-.09,0),(-half,-.09,0),.38,'blunt-toe')
    elif name in ('brain_retractor','periosteal_elevator'):
        broad=name=='brain_retractor'
        width, length=(6.,5.) if broad else (3.,3.)
        shaft(-length)
        # Each strip encloses a longitudinal slice through the thin paddle.
        strips=6 if broad else 3
        for i in range(strips):
            x=width*((i+.5)/strips-.5)
            part((x,0,-length),(x,0,0),.54,'paddle-strip')
    elif name in ('dissector','subcortical_probe','fixation_driver'):
        shaft(-3)
        end=(.6,0,0) if name=='dissector' else (0,0,0)
        form={'dissector':'blunt-hook','subcortical_probe':'probe-tip','fixation_driver':'driver-bit'}[name]
        part((0,0,-3),end,.48,form)
    elif name=='cortical_probe':
        shaft(-4)
        for side in (-1,1):
            part((0,0,-4),(side*.68,0,-1),.24,'insulated-arm','shaft')
            part((side*.68,0,-1),(side*.68,0,0),.32,'electrode')
    else:
        shaft(-8)
        tip_radius, form={
            'bipolar_forceps':(.25,'electrode'),
            'tumor_forceps':(.35,'serrated-jaw'),
            'bone_flap_holder':(.54,'gripping-cup'),
            'membrane_forceps':(.28,'fine-jaw'),
            'raney_clip_applier':(.55,'clip-seat'),
            'dressing_forceps':(.36,'dressing-jaw'),
            'needle_driver':(.32,'needle-jaw'),
        }[name]
        for side in (-1,1):
            part((0,0,-8),(side*(.8+half*.5),0,-3),.38,
                 'insulated-arm' if name=='bipolar_forceps' else 'forceps-arm','shaft')
            part((side*(.8+half*.5),0,-3),(side*half,0,0),tip_radius,form)
    return tuple(parts)

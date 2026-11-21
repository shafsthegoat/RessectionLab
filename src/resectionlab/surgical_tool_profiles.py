"""Fixed distal instrument profiles for the generated operative-exposure scene.

All coordinates are local mm; +Z points towards tissue. The two native contact
roles are deliberately separate from the render forms: guards and insulated
limbs obstruct motion, while only the exposed blades/tips produce contacts.
"""
from __future__ import annotations

from functools import lru_cache
from hashlib import sha256
import json

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


@lru_cache(maxsize=len(PROFILE_IDS))
def profile_geometry_hash(index):
    """Bind the complete piecewise rigid geometry and its linear aperture law."""
    endpoints = [profile_segments(index, jaw) for jaw in (0.,12.)]
    contract = ['operative-distal-segments-mm-v1', PROFILE_IDS[index-1], [0.,12.], endpoints]
    return 'sha256:'+sha256(json.dumps(contract,separators=(',',':'),allow_nan=False).encode()).hexdigest()

# Total outer axial extent and the depth of the shaft-to-handle junction in mm.
# Sources and dimensional assumptions are identified in the surgical toolbox API.
# Radius bounds the transverse grip body (individual arms in a paired spring grip).
ASSEMBLIES = {
    'cranial_drill':(172.,70.,9.,'motor'),
    'craniotome':(172.,70.,9.,'motor'),
    'cusa':(194.,80.,10.,'ultrasonic'),
    'frazier_suction':(180.,100.,4.,'suction'),
    'irrigation':(155.,75.,3.2,'suction'),
    'scalpel':(143.,43.,4.2,'flat'),
    'brain_retractor':(185.,82.,4.2,'flat'),
    'periosteal_elevator':(165.,65.,4.3,'flat'),
    'dissector':(175.,76.,3.,'pencil'),
    'cortical_probe':(165.,72.,3.8,'pencil'),
    'subcortical_probe':(165.,72.,3.8,'pencil'),
    'fixation_driver':(136.,48.,6.5,'driver'),
    'bipolar_forceps':(195.,75.,2.1,'spring'),
    'tumor_forceps':(160.,62.,1.8,'spring'),
    'bone_flap_holder':(165.,65.,2.5,'ring'),
    'membrane_forceps':(150.,60.,1.6,'spring'),
    'raney_clip_applier':(165.,55.,2.6,'ring'),
    'dressing_forceps':(155.,62.,1.8,'spring'),
    'needle_driver':(180.,65.,2.1,'spring'),
    'dural_scissors':(170.,65.,2.1,'ring'),
    'micro_scissors':(155.,62.,1.7,'spring'),
}


@lru_cache(maxsize=32)
def proximal_segments(index):
    name = PROFILE_IDS[index-1]
    length, neck, radius, style = ASSEMBLIES[name]
    shaft_radius = profile_segments(index, 0.)[0][2]
    result = []
    def member(a, b, r, form):
        result.append((a,b,r,'shaft',form))
    member((0,0,-40),(0,0,-neck),shaft_radius,'proximal-shank')
    if style in ('spring','ring'):
        width = 4. if style=='spring' else 8.5
        if style=='ring':
            pivot = length-29
            for sign in (-1,1):
                member((0,0,-neck),(sign*width,0,-pivot),radius,'lever-arm')
                # A filled capsule encloses each transverse finger loop and
                # its central void; collision remains deliberately conservative.
                member((sign*width,0,-pivot),(sign*width,0,-length+7.5),7.5,'finger-loop')
        else:
            for sign in (-1,1):
                bend=(sign*width,0,-(neck+length)/2)
                member((0,0,-neck),bend,radius,'spring-grip')
                member(bend,(0,0,-length+radius),radius,'spring-grip')
    elif style=='suction':
        # Slight bend and a thickened proximal valve/hose connector. Tubing
        # starts after the terminal socket and is outside this rigid body.
        bend=(5.,0.,-neck-13)
        member((0,0,-neck),bend,shaft_radius,'angled-neck')
        member(bend,(5.,0.,-length+radius),radius,'valve-grip')
    else:
        form={'motor':'motor-handpiece','ultrasonic':'ultrasonic-handpiece',
              'flat':'flat-handle','pencil':'pencil-grip','driver':'driver-grip'}[style]
        member((0,0,-neck),(0,0,-length+radius),radius,form)
    return tuple(result)


def assembly_segments(index, aperture_mm, revision):
    distal=profile_segments(index,aperture_mm)
    if revision==1:
        return distal
    if revision!=2:
        raise ValueError('UNKNOWN_OPERATIVE_ASSEMBLY_REVISION')
    return distal+proximal_segments(index)


@lru_cache(maxsize=32)
def assembly_geometry_hash(index):
    endpoints=[assembly_segments(index,jaw,2) for jaw in (0.,12.)]
    contract=['operative-full-device-segments-mm-v2',PROFILE_IDS[index-1],[0.,12.],endpoints]
    return 'sha256:'+sha256(json.dumps(contract,separators=(',',':'),allow_nan=False).encode()).hexdigest()

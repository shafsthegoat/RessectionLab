"""Small development-only, existing-physics challenge to a constant tip choice.

No learning admission or new physics. One independently declared unguarded work
route is displaced to2.2mm from its target column. Wide-tip work can reach it;
other routes retain source neural gates that require the narrow tip. These
claims must be established by physical receipts, not by this declaration.
"""
from __future__ import annotations
import copy
from dataclasses import asdict
import numpy as np
from resectionlab.surgical_sequential_task import generated_task,GeneratedTask,Skill,TASK_PROPERTY
from resectionlab.surgical_procedure_policy import procedure_source_memberships
from resectionlab.surgical_scene import SurgicalScene
from resectionlab.core import semantic_digest

DEVELOPMENT_PARENTS=(4000,4004)
LATER_GROUPS={'train':list(range(8000,8032)),'validation':list(range(8100,8108)),
              'assessment':list(range(8200,8216))}

def variation_task(parent):
    # Only these two predeclared development geometries are executable now.
    if parent not in DEVELOPMENT_PARENTS:
        raise ValueError('Development-only geometry; future groups are a proposal, not admitted evaluation')
    base=generated_task(parent*4)
    neural=base.scene.points(sorted(procedure_source_memberships(base.scene)['neural']))
    def separation(skill):
        a,b=np.asarray(skill.route);d=b-a
        return float(np.min(np.linalg.norm(neural-(a+np.clip((neural-a)@d/(d@d),0,1)[:,None]*d),axis=1)))
    widened=max((s for s in base.skills if s.kind=='work'),key=separation)
    skills=[]
    for s in base.skills:
        if s is not widened:skills.append(s);continue
        commands=copy.deepcopy(s.actions)
        for a in commands:a['pose']['position'][0]+=1.8
        route=tuple(tuple(a['pose']['position']) for a in commands[:2])
        skills.append(Skill(semantic_digest(commands)[7:19],'work',commands,route=route))
    props=copy.deepcopy(base.scene.properties)
    props[TASK_PROPERTY]['inventory']=[asdict(s) for s in skills]
    props[TASK_PROPERTY]['developmentVariation']=dict(schema='resectionlab.state-dependent-tip-development.v1',
        parent=parent,baseParent=parent*4,displacedRoute=skills[list(base.skills).index(widened)].id,
        originalOffsetMm=.4,displacedOffsetMm=2.2,anatomyUnchanged=True,
        meaning='Development-only source route geometry; original engine, labels, memberships, limits and endpoint unchanged')
    scene=SurgicalScene(base.scene.labels,base.scene.affine,base.scene.regions,base.scene.source,props)
    audit=dict(parent=parent,pattern=0,geometryHash=semantic_digest(dict(labels=scene.labels.tolist(),
        affine=scene.affine.tolist(),inventory=props[TASK_PROPERTY]['inventory'])),
        sourceSceneId=scene.identity,configHash=base.config.fingerprint,developmentOnly=True)
    return GeneratedTask(scene,tuple(skills),audit,base.config)


def geometry_state_selector(features,mask,stochastic=False):
    """Source/current-state rule; no preview, fitted parameter or outcome label.

    While wide, take remaining work whose observed source-neural separation is
    at least3mm; when none remains, exchange to narrow. With a narrow tool take
    remaining work. This is a feasibility witness, not a proposed learned model.
    """
    work=[i for i,x in enumerate(features) if mask[i] and x[1] and x[9]>0]
    wide=features[0,3]>.4/1.5+1e-6
    safe=[i for i in work if features[i,10]*30>=3.-1e-6] if wide else work
    if safe:return max(safe,key=lambda i:features[i,9]),{}
    if wide and work:
        prepare=[i for i,x in enumerate(features) if mask[i] and x[0] and abs(x[4]*1.5-.4)<1e-6]
        if prepare:return prepare[0],{}
    return next(i for i,x in enumerate(features) if mask[i] and x[2]),{}

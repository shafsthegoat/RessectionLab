"""Bounded streaming parser for declared FEBio specimen primitives.

It requires the actual initial state and every converged state. It never fills
missing records from an analytical solution and does not launch FEBio.
"""
from __future__ import annotations

import math
import re

import numpy as np


def iter_data_records(lines, *, expected_times, item_count, field_count, record_name,
                      maximum_bytes=256*1024**2):
    times=tuple(float(t) for t in expected_times)
    if len(times)<2 or times[0]!=0 or times[-1]!=1 or not all(math.isfinite(t) for t in times) or any(b<=a for a,b in zip(times,times[1:])):
        raise ValueError('Declared complete rest-to-load time grid required')
    if not isinstance(item_count,int) or not 1<=item_count<=20000 or not isinstance(field_count,int) or not 1<=field_count<=16:
        raise ValueError('Invalid bounded output dimensions')
    count=0; total_bytes=0; record=None

    def finish(value,index):
        if value is None or index>=len(times) or value['step']!=index or value['name']!=record_name or value['time'] is None:
            raise ValueError('Missing, extra or unbound output state')
        # Tagged default output uses 12 significant digits, so tolerate only print rounding.
        if not math.isfinite(value['time']) or abs(value['time']-times[index])>5e-12:
            raise ValueError('Unexpected physical load time')
        if not value['seen'].all():
            raise ValueError('Missing primitive node/element ID')
        return {'step':index,'time':value['time'],'declared_time':times[index],
                'name':record_name,'values':value['values']}

    for raw in lines:
        if not isinstance(raw,str):raise ValueError('Strict UTF-8 text lines required')
        total_bytes+=len(raw.encode('utf-8'))
        if total_bytes>maximum_bytes or len(raw)>4096:
            raise ValueError('Primitive output exceeds declared byte/line bound')
        line=raw.strip()
        if not line:continue
        if line.startswith('*Step'):
            if record is not None:
                yield finish(record,count);count+=1
            match=re.fullmatch(r'\*Step\s*=\s*(\d+)',line)
            if not match:raise ValueError('Malformed step header')
            record={'step':int(match[1]),'time':None,'name':None,
                    'values':np.empty((item_count,field_count)), 'seen':np.zeros(item_count,dtype=bool)}
        elif line.startswith('*Time'):
            match=re.fullmatch(r'\*Time\s*=\s*(\S+)',line)
            if not match or record is None or record['time'] is not None:raise ValueError('Missing/duplicate time header')
            record['time']=float(match[1])
        elif line.startswith('*Data'):
            match=re.fullmatch(r'\*Data\s*=\s*(.+)',line)
            if record is None or record['name'] is not None or not match:raise ValueError('Missing/duplicate/malformed data header')
            record['name']=match[1].strip()
        else:
            if record is None or record['time'] is None or record['name']!=record_name:
                raise ValueError('Unbound primitive row')
            fields=line.split(',')
            if len(fields)!=field_count+1 or not fields[0].isdigit():raise ValueError('Malformed primitive columns')
            i=int(fields[0])-1
            if not 0<=i<item_count or record['seen'][i]:raise ValueError('Duplicate/out-of-range primitive ID')
            values=np.array([float(v) for v in fields[1:]])
            if not np.isfinite(values).all():raise ValueError('Nonfinite primitive')
            record['seen'][i]=True;record['values'][i]=values
    if record is not None:
        yield finish(record,count);count+=1
    if count!=len(times):raise ValueError('Missing final or initial state; no analytical substitution')


def check_solver_records(lines, *, expected_times, residual_floor_N2, maximum_bytes=16*1024**2):
    """Check actual final residual evidence at each nonzero declared load time."""
    times=tuple(float(t) for t in expected_times)
    if len(times)<2 or times[0]!=0 or times[-1]!=1 or not all(math.isfinite(t) for t in times) or any(b<=a for a,b in zip(times,times[1:])) or not math.isfinite(residual_floor_N2) or residual_floor_N2<=0:
        raise ValueError('Finite declared residual floor and complete time grid required')
    # FESolidSolver2 prints status time with %lg (six significant digits), unlike
    # the 12-digit primitive recorder. Match that pinned source format exactly.
    printed_times={float(format(t,'.6g')):i for i,t in enumerate(times) if i>0}
    if len(printed_times)!=len(times)-1:raise ValueError('Time grid aliases in solver status precision')
    seen={};active=None;active_has_residual=False;normal=False;byte_count=0
    for raw in lines:
        byte_count+=len(raw.encode('utf-8'))
        if byte_count>maximum_bytes:raise ValueError('Solver log bound exceeded')
        banner=re.sub(r'[^A-Za-z]','',raw).upper()
        if banner=='NORMALTERMINATION':normal=True
        if banner in ('ERRORTERMINATION','ABNORMALTERMINATION') or re.search(r'\bnan\b|negative jacobian',raw,re.I):
            raise ValueError('Solver failure evidence')
        match=re.search(r'Nonlinear solution status:\s*time=\s*(\S+)',raw)
        if match:
            if active is not None and not active_has_residual:raise ValueError('Iteration missing its residual evidence')
            t=float(match[1]);nearest=printed_times.get(t)
            if nearest is None or (active is not None and nearest<active):raise ValueError('Undeclared or backtracked solver load state/cutback')
            active=nearest;active_has_residual=False
        fields=raw.split()
        if fields and fields[0]=='residual':
            if active is None or active_has_residual or len(fields)!=4:raise ValueError('Unbound/duplicate residual norm')
            values=[float(v) for v in fields[1:]]
            if not all(math.isfinite(v) and v>=0 for v in values):raise ValueError('Invalid residual norm')
            required=1e-8*values[0]
            if abs(values[2]-required)>2e-5*max(required,values[2]):
                raise ValueError('Reported required residual disagrees with declared rtol')
            seen[active]=values;active_has_residual=True
    if not normal or not active_has_residual or set(seen)!=set(range(1,len(times))):raise ValueError('Incomplete converged termination/residual evidence')
    checks=[{'fraction':times[i],'initial_N2':v[0],'actual_N2':v[1],
             'reported_required_N2':v[2], 'declared_rtol':1e-8,
             'limit_N2':max(1e-8*v[0],residual_floor_N2)*(1+2e-5),
             'passed':v[1]<=max(1e-8*v[0],residual_floor_N2)*(1+2e-5)} for i,v in sorted(seen.items())]
    return {'passed':all(c['passed'] for c in checks),'states':checks,'normal_termination':normal}

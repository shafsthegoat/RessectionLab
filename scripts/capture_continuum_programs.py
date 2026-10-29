#!/usr/bin/env python3
"""Capture committed inputs from qualified native programs, never larger-domain admission."""
from pathlib import Path
import argparse
import gzip
import json
import os
import resource
import subprocess
import sys
import time

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from resectionlab.workspace_bundle import load_workspace
from resectionlab.surgical_scene import from_case
from resectionlab.surgical_environment import standardize
from resectionlab.surgical_continuum_capture import MechanicsCaptureEngine,capture_mechanics_inputs
from resectionlab import surgical_continuum as c
from resectionlab.simulator_qualification import removed_accounting
from profile_continuum_exterior import bound,sha,write
from febio_runtime import supervise

PROTOCOL=ROOT/'manifests/experiments/continuum-program-capture-v1.json'
SOURCES=[Path(__file__),ROOT/'scripts/profile_continuum_exterior.py',ROOT/'scripts/febio_runtime.py',
    *[ROOT/'src/resectionlab'/name for name in ('surgical_continuum_capture.py','surgical_continuum.py',
      'surgical_engine.py','surgical_continuum_solve.py','surgical_deformation_v2.py','surgical_cusa_v3.py',
      'surgical_physics_v2.py','surgical_motion.py','surgical_tools.py','surgical_scene.py','core.py','workspace_bundle.py')]]


def worker(output,launch):
    output.mkdir(parents=True,exist_ok=False);p=json.loads(PROTOCOL.read_text());identity=json.loads(launch.read_text())
    started=time.perf_counter();branches={};summary=[]
    report={'schema':'resectionlab.continuum-program-capture-result.v1','status':'started','phase':'admission',
        'sourceCommit':identity['commit'],'sourceHashes':identity['sourceHashes'],'protocolSha256':sha(PROTOCOL),
        'sourceRole':p['sourceRole'],'physicalValidationPassed':False,'physiologicalRL':'hold','largeScaleRL':'hold',
        'largerDomainPublicActionsAdmitted':False,'newAcquiredDatasetBytes':0,'trainingUpdates':0}
    def check():
        if time.perf_counter()-started>p['caps']['wallSeconds']-10:raise TimeoutError('Native input capture wall cap')
        if sum(f.stat().st_size for f in output.rglob('*') if f.is_file())>p['caps']['outputBytes']-65536:
            raise ValueError('Native input capture output cap')
    try:
        if identity['protocolSha256']!=sha(PROTOCOL):raise ValueError('Launch protocol changed')
        for path,digest in identity['sourceHashes'].items():
            if sha(ROOT/path)!=digest:raise ValueError('Launch source changed')
        case,artifacts,_=load_workspace(bound(p['qualifiedNativeCase']));record=artifacts['surgical_toolbox']
        scene=standardize(from_case(case,record['mapping']),physics_profile=record['physicsProfile'])
        if scene.identity!=p['sourceSceneId']:raise ValueError('Qualified native scene differs')
        receipts=json.loads(gzip.decompress(bound(p['qualifiedActionTraces']).read_bytes()))
        with np.load(bound(p['coreGeometry']),allow_pickle=False) as raw:core_cells=set(map(int,raw['cells']))
        with np.load(bound(p['referenceGeometry']),allow_pickle=False) as raw:reference_cells=set(map(int,raw['cells']))
        for branch in p['branches']:
            begin=time.perf_counter();engine=MechanicsCaptureEngine(scene);prefix=None;outcomes=[]
            report.update(phase='native-capture',activeBranch=branch);write(output/'report.json',report)
            with capture_mechanics_inputs():
                for index,action in enumerate(record['branches'][branch]['actions']):
                    result=engine.step(action,check=check)
                    if result!=receipts[branch]['proportional-history-compliance'][index]:
                        raise ValueError('Capture changed canonical public action receipt')
                    outcomes.append(result)
                    if index+1==p['commonActions']:
                        state=engine.state[c.PROPERTY]
                        for field in [state['cornerDisplacementMm'],*state['memoryDisplacementMm']]:
                            if any(np.any(np.asarray(v)!=0.) for v in field.values()):
                                raise ValueError('Post-exposure activation requires explicit zero initial history')
                        prefix={'actions':p['commonActions'],'timeSeconds':engine.state['elapsedSeconds'],
                            'stateId':engine.state_id,'nativeRemainderSeconds':engine.state['deformationModel']['remainderSeconds'],
                            'fullSourceAccounting':removed_accounting(scene,engine.state),'removedSourceIds':engine.state['removed'][:],
                            'continuumFieldsZero':True}
            if engine.snapshot()!=record['branches'][branch]:raise ValueError('Captured native full snapshot differs')
            if prefix is None or prefix['nativeRemainderSeconds']!=0.:raise ValueError('Prefix/clock admission differs')
            captured=engine.mechanics_inputs()
            if len(captured['ticks'])>p['caps']['maximumTicksPerBranch']:raise ValueError('Capture row cap')
            suffix=[row for row in captured['ticks'] if row['actionIndex']>=p['commonActions']]
            elapsed=sum(row['durationSeconds'] for row in suffix)
            expected_elapsed=engine.state['elapsedSeconds']-prefix['timeSeconds']
            steps=sum(row['nativeFixedSteps'] for row in suffix)
            if abs(elapsed-expected_elapsed)>1e-9 or steps!=p['expectedSuffixFixedSteps'][branch]:
                raise ValueError('Committed native mechanics clock incomplete')
            added=sorted(set(engine.state['removed'])-set(prefix['removedSourceIds']))
            outside=set(added)-reference_cells
            if outside:raise ValueError('Suffix removes owners outside the frozen reference; no fake zero energy')
            accounting=removed_accounting(scene,engine.state)
            if branch!='stop-at-model-limit':
                if (accounting['targetRemovedMm3']-prefix['fullSourceAccounting']['targetRemovedMm3']!=8.
                        or accounting['nonTargetRemovedMm3']-prefix['fullSourceAccounting']['nonTargetRemovedMm3']!=0.
                        or len(set(added)&core_cells)!=7 or len(set(added)-core_cells)!=1):
                    raise ValueError('Original source removal ownership not preserved')
            row={'branch':branch,'actions':len(outcomes),'canonicalEveryActionExact':True,'canonicalFullSnapshotExact':True,
                'committedTicks':len(captured['ticks']),'suffixTicks':len(suffix),'suffixFixedSteps':steps,
                'suffixDurationSeconds':elapsed,'topologies':len(captured['topologies']),
                'additionalRemovedSourceIds':added,'additionalCoreOwners':len(set(added)&core_cells),
                'additionalExteriorOwners':len(set(added)-core_cells),'fullSourceAccounting':accounting,
                'captureSeconds':time.perf_counter()-begin}
            branches[branch]={**captured,'prefixActivation':prefix,'actions':record['branches'][branch]['actions'],
                              'actionReceipts':outcomes,'summary':row}
            summary.append(row)
            data=gzip.compress(json.dumps(branches[branch],sort_keys=True,allow_nan=False).encode(),mtime=0)
            destination=output/(branch+'-inputs.json.gz');destination.write_bytes(data)
            row.update(inputFile=destination.name,inputSha256=sha(destination))
            report.update(branches=summary);write(output/'report.json',report)
            print(json.dumps({'branch':branch,'suffixFixedSteps':steps,'elapsedWallSeconds':time.perf_counter()-started}),flush=True)
        for path,digest in identity['sourceHashes'].items():
            if sha(ROOT/path)!=digest:raise ValueError('Source changed during capture')
        for entry in [p['qualifiedNativeCase'],p['qualifiedActionTraces'],p['coreGeometry'],p['referenceGeometry']]:bound(entry)
        report.update(status='completed',phase='completed',branches=summary,
            suffixFixedSteps=sum(row['suffixFixedSteps'] for row in summary),
            prefixInterpretation='All native prefix actions/receipts/source costs retained. Full20 mesh is post-exposure only; no zero energy is assigned to unmodeled prefix removal.',
            sourceInputsUnchanged=True,scope='Prescribed inputs from qualified8mm native world; not20mm action/collision acceptance')
    except BaseException as error:
        report.update(status='invalid_or_incomplete',error=type(error).__name__+': '+str(error));raise
    finally:
        report.update(wallSeconds=time.perf_counter()-started,
            peakRssBytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform=='darwin' else 1024))
        write(output/'report.json',report)
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('output',type=Path)
    parser.add_argument('--worker',action='store_true');parser.add_argument('--launch',type=Path)
    args=parser.parse_args();output=args.output.resolve()
    if not output.is_relative_to(ROOT/'testcases'):raise ValueError('Evidence stays under testcases/')
    if args.worker:
        if args.launch is None:raise ValueError('Supervised launch required')
        worker(output,args.launch);return
    if output.exists():raise ValueError('Use a fresh output; no automatic retry')
    if subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip():raise ValueError('Commit prospective source/capture declaration first')
    launch=output.with_name(output.name+'-launch.json')
    if launch.exists():raise ValueError('Launch already exists')
    write(launch,{'protocolSha256':sha(PROTOCOL),'sourceHashes':{str(f.relative_to(ROOT)):sha(f) for f in SOURCES},
                  'commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()})
    p=json.loads(PROTOCOL.read_text());env=os.environ.copy();env.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',VECLIB_MAXIMUM_THREADS='1')
    receipt=supervise([sys.executable,str(Path(__file__)),'--worker','--launch',str(launch),str(output/'result')],
        output,cwd=ROOT,environment=env,seconds=p['caps']['wallSeconds'],rss_bytes=p['caps']['rssBytes'])
    valid=receipt['status']=='completed'
    if valid:
        r=json.loads((output/'result/report.json').read_text())
        valid=r['status']=='completed' and len(r['branches'])==3 and r['suffixFixedSteps']==4218
        if sum(f.stat().st_size for f in output.rglob('*') if f.is_file())>p['caps']['outputBytes']:valid=False
    write(output/'acceptance.json',{'nativeInputCaptureComplete':valid,'largerDomainPublicActionsAdmitted':False,
        'physicalValidationPassed':False,'physiologicalRL':'hold','supervisionStatus':receipt['status']})
    if not valid:raise RuntimeError('Native mechanics capture failed/incomplete; evidence retained')
    print(json.dumps({'wallSeconds':r['wallSeconds'],'suffixFixedSteps':r['suffixFixedSteps']}))


if __name__=='__main__':main()

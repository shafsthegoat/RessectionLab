#!/usr/bin/env python3
"""Audit every local patient file without modifying source anatomy or sessions.
Mesh caches and checkpointed receipts are written only below --output.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
import time
from types import SimpleNamespace
import numpy as np
from resectionlab.anatomy_geometry import mesh
from resectionlab.anatomy_qa import layer_quality, mask_quality
from resectionlab.patient_anatomy import PatientImagingCollection, file_hash
from resectionlab.anatomy_package import open_package
from resectionlab.imaging import load_case


def discover(root):
    if root.is_file(): return [root]
    process = subprocess.run(['rg', '--files', '--hidden', '--no-ignore', '--follow', '-g', '*.ressectionlab', '-g', '*.rsworkspace', '-g', 'patient-anatomy.json', '-g', '!node_modules/**', '-g', '!.git/**', '-g', '!.venv/**', '-g', '!.tools/**', '-g', '!build/**'], cwd=root, capture_output=True, text=True)
    if process.returncode not in (0, 1): raise RuntimeError(process.stderr)
    return [root/line for line in process.stdout.splitlines()]


def audit(path, output, meshes, history=False):
    before = file_hash(path); started = time.perf_counter()
    row = {'path': str(path), 'sha256': before, 'status': 'running', 'layers': [], 'anatomicalAccuracy': 'unmeasured'}
    try:
        with tempfile.TemporaryDirectory(prefix='resection-display-audit-') as scratch:
            manifest = open_package(path, scratch) if path.suffix == '.rsworkspace' else path
            if manifest.name == 'patient-anatomy.json':
                c = PatientImagingCollection(manifest.parent)
                row.update(patientId=c.document['patient_id'], sourceIdentity=c.identity(), format='native', scans=[{'id':s['id'], 'modality':s['modality'], 'shape':s['data']['shape']} for s in c.document['series']])
                cache_root = output/'derivatives'/before; cache_root.mkdir(parents=True, exist_ok=True)
                proxy = SimpleNamespace(root=cache_root, mask=c.mask, array=c.array)
                for layer in c.document['layers']:
                    if layer['kind'] not in {'vessels', 'vessel', 'artery', 'vein', 'sinus', 'tumor'}: continue
                    entry = {'name':layer['id'], 'kind':layer['kind'], 'qa':layer_quality(c, layer)}
                    if meshes:
                        t = time.perf_counter(); geometry = mesh(proxy, layer)
                        entry['surface'] = {'vertices':len(geometry['source']), 'triangles':len(geometry['indices']),
                            'maxDisplacementMm':float(geometry['max_displacement_mm']), 'restoredComponents':int(geometry.get('restored_components', 0)),
                            'policy':str(geometry.get('policy_version', 'empty-source')), 'preparationSeconds':time.perf_counter()-t}
                    row['layers'].append(entry)
            else:
                case = load_case(path)
                row.update(patientId=case.case_id, sourceIdentity=case.semantic_hash, format='saved-case', shape=list(case.mri.shape))
                for name, mask in case.compartments.items():
                    metadata = case.metadata.get('anatomy_layers', {}).get(name, {})
                    row['layers'].append({'name':name, 'kind':metadata.get('kind', 'unknown'), 'qa':mask_quality(mask, case.semantic_hash+':'+name)})
                if history:
                    from resectionlab.imaging import read_case_artifacts
                    from resectionlab.surgical_scene import from_case
                    from resectionlab.surgical_environment import standardize, SCHEMA
                    from resectionlab.surgical_engine import SurgicalEngine
                    record = read_case_artifacts(path).get('surgical_toolbox')
                    row['savedHistory'] = []
                    if record:
                        scene = from_case(case, record['mapping'])
                        if record.get('environmentSchema') == SCHEMA:
                            scene = standardize(scene, physics_profile=record['physicsProfile'])
                        for branch, snapshot in record['branches'].items():
                            engine = SurgicalEngine.restore(scene, snapshot)
                            position = record['position'] if branch == record['branchId'] else len(engine.actions)
                            selected = engine.branch(position)
                            row['savedHistory'].append({'branch':branch, 'steps':len(engine.actions),
                                'selectedPosition':position, 'stateId':selected.state_id,
                                'removedCells':len(selected.state['removed']), 'replay':'passed'})
            row['status'] = 'passed_with_findings' if any(e['qa']['issues'] for e in row['layers']) else 'passed'
    except Exception as exc:
        row.update(status='failed', error=str(exc))
    row['sourceUnchanged'] = file_hash(path) == before
    if not row['sourceUnchanged']: row['status']='failed'
    row['seconds'] = time.perf_counter()-started
    return row


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('roots', nargs='+', type=Path); parser.add_argument('--output', type=Path, required=True); parser.add_argument('--meshes', action='store_true');parser.add_argument('--history',action='store_true')
    args=parser.parse_args(); args.output.mkdir(parents=True, exist_ok=True)
    paths=sorted({p.resolve() for root in args.roots for p in discover(root.resolve())})
    result={'schema':'resectionlab-display-quality-v1', 'files':[], 'patients':[], 'unavailable':[], 'accuracy':'unmeasured'}
    readiness=Path('outputs/standard-environments/batch-2026-10-18/final-export/readiness.json')
    if readiness.exists(): result['unavailable']=json.loads(readiness.read_text()).get('unavailable_cases', [])
    receipt=args.output/'audit.json'
    for path in paths:
        row=audit(path,args.output,args.meshes,args.history);result['files'].append(row)
        # Identity from source metadata; saved research suffixes denote versions.
        patients={r.get('patientId', '').removesuffix('-surgical-research') for r in result['files'] if r.get('patientId') and not r.get('patientId','').startswith(('instrument-', 'synthetic'))}
        result['patients']=sorted(patients)
        receipt.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
        print(row['status']+' '+str(path),flush=True)
    print(json.dumps({'files':len(result['files']), 'independentPatientIds':result['patients'], 'failures':sum(r['status']=='failed' for r in result['files'])}))


if __name__=='__main__': main()

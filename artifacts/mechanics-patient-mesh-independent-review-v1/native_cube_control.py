#!/usr/bin/env python3
"""Single root-authorized tiny analytic API control. Never opens patient data."""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time


def sha(path):
    with Path(path).open('rb') as stream: return hashlib.file_digest(stream, 'sha256').hexdigest()


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    obj = importlib.util.module_from_spec(spec); spec.loader.exec_module(obj); return obj


def write(path, value):
    temp = path.with_suffix('.tmp'); temp.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n'); temp.replace(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--settings', type=Path, required=True)
    parser.add_argument('--sha256', required=True)
    parser.add_argument('--worker', action='store_true')
    args = parser.parse_args()
    assert sha(args.settings) == args.sha256
    settings = json.loads(args.settings.read_text()); root = Path(settings['root']); out = Path(settings['output'])
    assert sha(__file__) == settings['script_sha256']
    for name, expected in settings['source_bindings'].items(): assert sha(root/name) == expected, name
    if not args.worker:
        runtime = load(root/'scripts/febio_runtime.py', 'analytic_supervisor')
        env = runtime.private_environment({'caps': {'thread_environment': settings['thread_environment']}})
        for name in list(env):
            if name.startswith(('PYTHON', 'DYLD_')) or name in ('LD_LIBRARY_PATH', 'LD_PRELOAD'): env.pop(name)
        env.update(settings['thread_environment'])
        command = [str(root/'.venv/bin/python'), '-B', str(Path(__file__).resolve()), '--settings', str(args.settings.resolve()), '--sha256', args.sha256, '--worker']
        result = runtime.supervise(command, out/'attempt-01', cwd=root, environment=env,
            seconds=settings['caps']['seconds'], rss_bytes=settings['caps']['rss_bytes'])
        print(json.dumps(result))
        return 0 if result['status']=='completed' else 1
    receipt = out/'native-control.json'
    if receipt.exists(): raise FileExistsError('No retry allowed')
    result = {'status':'running', 'phase':'imports', 'patient_array_reads':0, 'solver_calls':0, 'native_generation_calls':0,
              'script_sha256':sha(__file__), 'settings_sha256':args.sha256, 'started_at':datetime.now(timezone.utc).isoformat()}
    write(receipt,result); start=time.monotonic(); gmsh=None
    try:
        import numpy as np
        from skimage.measure import marching_cubes
        from vtkmodules.vtkCommonCore import vtkSMPTools
        import importlib.metadata
        for name, expected in settings['package_versions'].items(): assert importlib.metadata.version(name)==expected,name
        assert vtkSMPTools.SetBackend('Sequential') and vtkSMPTools.GetBackend()=='Sequential'
        helper=load(root/'scripts/mechanics_patient_mesh.py','frozen_analytic_mesh_helper')
        protocol=json.loads((root/'manifests/experiments/resect-case4-patient-mesh-v1.json').read_text())
        # This explicit analytic fixture is the ONLY array input.
        mask=np.zeros(settings['fixture']['shape'],dtype=np.uint8)
        lo,hi=settings['fixture']['positive_box']; mask[lo:hi,lo:hi,lo:hi]=1
        affine=np.asarray(settings['fixture']['affine_mm'])
        result['phase']='actual_marching_cubes';write(receipt,result)
        X,F,surface=helper.extract_native_surface(mask,affine,marching_cubes)
        result['source_surface']=surface
        np.savez_compressed(out/'cube-source.npz',vertices_m=X,triangles=F)
        result['phase']='gmsh_import_and_parameterization';write(receipt,result)
        gmsh=helper.import_file(protocol['gmsh_runtime']['module_path'],'gmsh')
        assert gmsh.__version__=='4.15.2'
        assert Path(gmsh.lib._name).resolve()==Path(protocol['gmsh_runtime']['library_path']).resolve()
        gmsh.initialize([],readConfigFiles=False,run=False)
        def charge():
            result['native_generation_calls']+=1;result['phase']='actual_tet10_generation';write(receipt,result)
        nodes,cells,origin=helper.generate_gmsh_level(gmsh,X,F,settings['level'],protocol,charge)
        result.update(returned_nodes=len(nodes),returned_elements=len(cells),origin=origin)
        result['phase']='tet10_quality';write(receipt,result)
        np.savez_compressed(out/'cube-tet10.npz',nodes_m=nodes,tet10_indices=cells)
        gmsh.write(str(out/'cube-tet10.msh'))
        quality,boundary,faces=helper.validate_tet10(nodes,cells,settings['level'],protocol)
        result['quality']=quality
        result['phase']='actual_VTK_bidirectional_distance';write(receipt,result)
        # Exercise bulk EvaluateFunction and retain both numerical cover bounds.
        forward=helper.directed_surface_bound(X,F,helper.vtk_distance_function(boundary,faces),protocol['surface_fidelity'])
        result['source_to_mesh']=forward;write(receipt,result)
        reverse=helper.directed_surface_bound(boundary,faces,helper.vtk_distance_function(X,F),protocol['surface_fidelity'])
        result['mesh_to_source']=reverse
        result['relative_volume_error']=abs(quality['volume_m3']/surface['enclosed_volume_m3']-1)
        write(receipt,result)
        assert quality['boundary']['euler_characteristic']==surface['euler_characteristic']
        assert max(forward['full_surface_upper_bound_m'],reverse['full_surface_upper_bound_m'])<=protocol['surface_fidelity']['maximum_distance_m']
        assert result['relative_volume_error']<=protocol['surface_fidelity']['maximum_relative_volume_error']
        result.update(status='passed_analytic_API_control_only',phase='complete')
    except BaseException as error:
        result.update(status='failed_or_incomplete',error={'type':type(error).__name__,'message':str(error)})
    finally:
        if gmsh is not None:
            try: gmsh.finalize()
            except BaseException as error: result.update(status='failed_or_incomplete',finalize_error=str(error))
        result['elapsed_seconds']=time.monotonic()-start
        result['sources_after']={name:sha(root/name)==expected for name,expected in settings['source_bindings'].items()}
        if not all(result['sources_after'].values()):result['status']='failed_or_incomplete'
        result['files']={p.name:sha(p) for p in out.iterdir() if p.is_file() and p!=receipt}
        write(receipt,result)
    print(json.dumps({'status':result['status'],'phase':result['phase'],'seconds':result['elapsed_seconds']}))
    return 0 if result['status']=='passed_analytic_API_control_only' else 1


if __name__=='__main__':raise SystemExit(main())

"""Read-only dependency/import audit; does not install or remove packages."""
from pathlib import Path
from datetime import datetime, timezone
from importlib import metadata
import ast, hashlib, json, os, subprocess, sys, tempfile, tomllib
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

ROOT=Path(__file__).resolve().parents[2]
OUT=Path(__file__).parent
legacy={"pyside6","pyside6-addons","pyside6-essentials","shiboken6","vtk"}
def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def pins(path):
 return [Requirement(line) for line in path.read_text().splitlines() if line.strip() and not line.startswith(("#","-r"))]
current=pins(ROOT/"requirements-lock.txt")
optional=pins(ROOT/"requirements-legacy-qt-lock.txt")
assert not {canonicalize_name(r.name) for r in current}&legacy
assert {canonicalize_name(r.name) for r in optional}==legacy
versions={}
for r in current+optional:
 version=metadata.version(r.name)
 assert r.specifier.contains(version,prereleases=True),(str(r),version)
 versions[r.name]=version
source=tomllib.loads((ROOT/"pyproject.toml").read_text())
assert not {canonicalize_name(Requirement(r).name) for r in source["project"]["dependencies"]}&legacy
qt_dependencies=[]
for d in metadata.distributions():
 if canonicalize_name(d.metadata["Name"]) in legacy: continue
 for raw in d.requires or []:
  r=Requirement(raw)
  if canonicalize_name(r.name) in legacy and (r.marker is None or r.marker.evaluate({"extra":""})):
   item={"distribution":d.metadata["Name"],"version":d.version,"requirement":str(r)}
   if item not in qt_dependencies: qt_dependencies.append(item)
assert all(canonicalize_name(x["distribution"])=="resectionlab" for x in qt_dependencies)
imports=[]
for folder in ["src","scripts","tests","packaging"]:
 for file in sorted((ROOT/folder).rglob("*.py")):
  if "__pycache__" in file.parts: continue
  for node in ast.walk(ast.parse(file.read_text())):
   names=[x.name for x in node.names] if isinstance(node,ast.Import) else [node.module] if isinstance(node,ast.ImportFrom) and node.module else []
   for name in names:
    if name.split(".")[0] in {"PySide6","PyQt6","vtk","vtkmodules","shiboken6"}:
     imports.append({"file":str(file.relative_to(ROOT)),"line":node.lineno,"module":name})
blocker="""import importlib.abc, sys
class NoLegacyGui(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'PySide6','PyQt6','vtk','vtkmodules','shiboken6'}:
            raise ModuleNotFoundError('Legacy GUI deliberately unavailable for dependency probe: '+fullname, name=fullname)
sys.meta_path.insert(0, NoLegacyGui())
"""
commands={}
with tempfile.TemporaryDirectory(prefix="resectionlab-no-qt-") as folder:
 path=Path(folder);(path/"sitecustomize.py").write_text(blocker)
 environment=dict(os.environ,PYTHONPATH=os.pathsep.join([str(path),str(ROOT/"src")]),PYTEST_DISABLE_PLUGIN_AUTOLOAD="1",OMP_NUM_THREADS="2")
 gui={"desktop","scene","slice_view","training_view","workers"}
 modules=[]
 for file in sorted((ROOT/"src"/"resectionlab").rglob("*.py")):
  if file.parent.name=="app" and file.stem in gui: continue
  parts=file.relative_to(ROOT/"src").with_suffix("").parts
  modules.append(".".join(parts[:-1] if parts[-1]=="__init__" else parts))
 import_probe="import importlib,json,sys; names="+repr(modules)+"; [importlib.import_module(n) for n in names]; assert not any(n.split('.')[0] in {'PySide6','vtk','vtkmodules','shiboken6','PyQt6'} for n in sys.modules); print(json.dumps({'imported_modules':names,'legacy_gui_loaded':False}))"
 selections={"import_all_numerical_modules":[sys.executable,"-c",import_probe],
 "default_collection":[sys.executable,"-m","pytest","--collect-only","-q"],
 "focused_engine_tests":[sys.executable,"-m","pytest","-q","tests/test_app_workers.py","tests/test_app_annotation.py","tests/test_app_reslice.py","tests/test_app_refinement.py::test_replay_uses_history_increments_not_accessibility","tests/test_desktop_bridge.py::test_case_binary_roundtrip_versions_and_null_clinical_fields","tests/test_desktop_bridge.py::test_generate_save_reopen_retains_real_evaluator_records","tests/test_desktop_bridge.py::test_jsonl_process_stdout_is_protocol_only_and_imports_no_gui"]}
 for name,command in selections.items():
  result=subprocess.run(command,cwd=ROOT,env=environment,capture_output=True,text=True,timeout=90)
  (OUT/(name+".log")).write_text(result.stdout+result.stderr)
  commands[name]={"returncode":result.returncode,"last_output_lines":(result.stdout+result.stderr).splitlines()[-5:]}
  print(name,result.returncode,commands[name]["last_output_lines"][-1:] if commands[name]["last_output_lines"] else [])
  assert result.returncode==0,name
 legacy_result=subprocess.run([sys.executable,"-m","pytest","tests/test_app_workers.py","-q"],cwd=ROOT,env=dict(os.environ,PYTEST_DISABLE_PLUGIN_AUTOLOAD="1"),capture_output=True,text=True,timeout=30)
 (OUT/"legacy_worker_tests.log").write_text(legacy_result.stdout+legacy_result.stderr)
 assert legacy_result.returncode==0
 commands["installed_legacy_worker_tests"]={"returncode":legacy_result.returncode,"last_output_lines":legacy_result.stdout.splitlines()[-2:]}
report={"created_at":datetime.now(timezone.utc).isoformat(),"python_version":sys.version.split()[0],"default_pin_count":len(current),"legacy_pin_count":len(optional),"pinned_versions_match_installed":True,"versions":versions,"file_hashes":{str(p.relative_to(ROOT)):sha(p) for p in [ROOT/"requirements-lock.txt",ROOT/"requirements-legacy-qt-lock.txt",ROOT/"pyproject.toml",ROOT/"tests/test_app_workers.py"]},"static_legacy_imports":imports,"nonlegacy_installed_metadata_requiring_legacy_gui":qt_dependencies,"stale_editable_metadata_note":"Installed resectionlab metadata predates optional Qt split; current pyproject declares Qt/VTK only under legacy-qt. Environment was not regenerated.","commands":commands,"clean_environment_install_performed":False,"environment_mutations":False,"new_packages_or_upgrades":False,"scope":"Source/dependency metadata plus subprocess import blockers; no clean environment or rebuilt package claim."}
report["probe_sha256"]=sha(Path(__file__))
report["packaging_builder_sha256"]=sha(ROOT/"scripts/build_electron_sidecar.py")
report["packaging_exclusions"]=["PySide6","PyQt6","vtk","vtkmodules"]
internal=ROOT/"desktop"/"sidecar"/"ressectionlab-engine"/"_internal"
if internal.is_dir():
 paths=list(internal.rglob("*"))
 matches=[str(p.relative_to(internal)) for p in paths if any(part.startswith(("PySide6","PyQt6","shiboken6","libshiboken","vtk","libvtk","libQt6","Qt6")) for part in p.relative_to(internal).parts)]
 assert not matches
 report["existing_frozen_engine_payload_scan"]={"root":str(internal.relative_to(ROOT)),"entries_scanned":len(paths),"legacy_toolkit_payload_matches":matches,"scope":"Existing frozen engine only; not a rebuild of the separated lock; retained legacy source is not runtime toolkit binaries."}
(OUT/"audit.json").write_text(json.dumps(report,indent=2)+"\n")
print(json.dumps({"default_pin_count":len(current),"legacy_pin_count":len(optional),"clean_install":False}))

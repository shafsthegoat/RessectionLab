# Native arm64, onedir bundle. Build via scripts/build_macos.py.
from importlib.metadata import distribution, PackageNotFoundError
from pathlib import Path
import os
import tomllib

from PyInstaller.utils.hooks import collect_data_files, collect_submodules, copy_metadata

root = Path(SPECPATH).parent
config = tomllib.loads((root / "pyproject.toml").read_text())
version = config["project"]["version"]
datas = collect_data_files("resectionlab", excludes=["**/__pycache__/*"])
datas.append((str(root / "BUILD_INPUT_MANIFEST.json"), "build_info"))

# Include the packages' own attribution and license material with the runtime.
for package in ("resectionlab", "numpy", "scipy", "nibabel", "torch", "PySide6", "PySide6-Essentials", "shiboken6", "vtk", "dipy"):
    try:
        datas += copy_metadata(package)
        dist = distribution(package)
        for item in dist.files or []:
            if "license" in str(item).lower() or "copying" in str(item).lower():
                source = Path(dist.locate_file(item))
                if source.is_file():
                    datas.append((str(source), "licenses/" + package + "/" + str(Path(item).parent)))
    except PackageNotFoundError:
        if package != "resectionlab":
            raise

for document in ("LICENSE", "THIRD_PARTY_NOTICES.md"):
    source = root / document
    if source.is_file():
        datas.append((str(source), "licenses"))

a = Analysis(
    [str(root / "packaging" / "macos_entry.py")],
    pathex=[str(root / "src")],
    binaries=[],
    datas=datas,
    hiddenimports=collect_submodules("resectionlab", on_error="raise") + [
        "vtkmodules.qt.QVTKRenderWindowInteractor",
        "vtkmodules.vtkInteractionStyle",
        "vtkmodules.vtkRenderingOpenGL2",
        "vtkmodules.vtkRenderingFreeType",
        "vtkmodules.vtkRenderingAnnotation",
        "vtkmodules.vtkInteractionWidgets",
    ],
    hookspath=[],
    hooksconfig={"matplotlib": {"backends": ["Agg"]}},
    runtime_hooks=[str(root / "packaging" / "runtime_logging.py")],
    # DIPY's optional FURY I/O branch imports the whole legacy vtk facade even
    # when FURY is absent. The app uses vtkmodules directly; this unused branch
    # otherwise expands the bundle by hundreds of MB. Keep actual DIPY I/O and
    # reconstruction APIs, which are exercised by the frozen diagnostic check.
    excludes=["vtk", "fury", "PyQt5", "PyQt6", "PySide2", "tkinter", "IPython", "jupyter", "notebook", "pytest"],
    noarchive=False,
    module_collection_mode={"resectionlab": "pyz+py"},
)
pyz = PYZ(a.pure)
exe = EXE(
    pyz, a.scripts, [],
    exclude_binaries=True,
    name="RessectionLab",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    argv_emulation=False,
    target_arch="arm64",
    codesign_identity=None,
    entitlements_file=None,
)
coll = COLLECT(exe, a.binaries, a.datas, strip=False, upx=False, name="RessectionLab")
app = BUNDLE(
    coll,
    name="RessectionLab.app",
    icon=os.environ["RESSECTIONLAB_ICON_PATH"],
    bundle_identifier="org.ressectionlab.desktop",
    version=version,
    info_plist={
        "CFBundleDisplayName": "RessectionLab",
        "CFBundleShortVersionString": version,
        "NSHighResolutionCapable": True,
        "LSApplicationCategoryType": "public.app-category.medical",
        "LSMinimumSystemVersion": "14.0",
        "NSHumanReadableCopyright": "RessectionLab contributors. Research software; clinical use is not validated.",
    },
)

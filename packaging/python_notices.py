"""Copy installed upstream notices for the modules/files in a frozen engine.

This inventories software, not permission to distribute the project. It never
imports numerical packages or modifies an engine. Run against the matching
PyInstaller work directory and immutable source snapshot, not a new environment.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
from importlib import metadata
import json
from pathlib import Path
import re
import shutil
import sys
import sysconfig
import types


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def notice_path(path: Path) -> bool:
    """Keep whole upstream license directories, including h5py's hdf5.txt."""
    if "__pycache__" in path.parts or path.suffix.lower() in {".py", ".pyc", ".pyo"}:
        return False
    return any(part.lower() in {"license", "licenses", "licence", "licences"}
               for part in path.parts[:-1]) or bool(
        re.match(r"^(licen[cs]e|copying|copyright|notice|authors)([._-]|$)", path.name, re.I)
    )


def toc_rows(value):
    if isinstance(value, (list, tuple)):
        if len(value) == 3 and all(isinstance(x, str) for x in value):
            yield value
        else:
            for child in value:
                yield from toc_rows(child)


def safe_relative(path: Path) -> Path:
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"Unsafe upstream notice path: {path}")
    return path


def copy_notice(source: Path, destination: Path, *, origin: str) -> dict:
    before = sha256(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)
    if sha256(source) != before or sha256(destination) != before:
        raise RuntimeError(f"Upstream notice changed during capture: {source}")
    return {"source": origin, "sha256": before, "bytes": destination.stat().st_size}


def locked_versions(snapshot: Path) -> dict[str, str]:
    result = {}
    for line in (snapshot / "requirements-lock.txt").read_text().splitlines():
        match = re.fullmatch(r"([A-Za-z0-9_.-]+)==([^\s;]+)", line.strip())
        if match:
            result[canonical(match[1])] = match[2]
    return result


def code_matches(first, second) -> bool:
    """Compare executable code and constants, ignoring archive filename rewrites.

    Compare fields recursively instead of accepting only a module name or an
    installed version declaration. PyInstaller rewrites code filenames.
    """
    if isinstance(first, types.CodeType) and isinstance(second, types.CodeType):
        fields = ("co_code", "co_names", "co_varnames", "co_flags", "co_argcount",
                  "co_posonlyargcount", "co_kwonlyargcount", "co_nlocals",
                  "co_stacksize", "co_firstlineno", "co_linetable", "co_exceptiontable",
                  "co_freevars", "co_cellvars", "co_name", "co_qualname")
        return all(getattr(first, name) == getattr(second, name) for name in fields) and code_matches(first.co_consts, second.co_consts)
    if isinstance(first, tuple) and isinstance(second, tuple):
        return len(first) == len(second) and all(code_matches(a, b) for a, b in zip(first, second))
    return type(first) is type(second) and first == second


def verify_archive_modules(engine: Path, rows: list[tuple]) -> dict:
    # This reads bytecode metadata; it does not execute a bundled module.
    from PyInstaller.archive.readers import CArchiveReader
    archive = CArchiveReader(str(engine))
    names = [name for name, item in archive.toc.items() if item[-1] == "z"]
    if len(names) != 1:
        raise ValueError("Expected exactly one embedded Python archive")
    embedded = archive.open_embedded_archive(names[0])
    expected = {name for name, _, kind in rows if kind.startswith("PYMODULE")}
    if expected != set(embedded.toc):
        raise ValueError("PyInstaller PYZ table does not match this engine's actual modules")
    # Version pins alone cannot prove that the installed sources match the old
    # engine. Compare every corresponding Python code object without importing.
    matched = 0
    for name, source, kind in rows:
        if not kind.startswith("PYMODULE") or not source:
            continue
        frozen = embedded.extract(name)
        if frozen is None:  # namespace package: no module bytes
            continue
        path = Path(source)
        if not path.is_file():
            raise ValueError(f"Frozen module source is unavailable: {name}")
        optimization = int(kind.rsplit('-', 1)[-1]) if kind[-1:].isdigit() else 0
        current = compile(path.read_bytes(), frozen.co_filename, "exec", optimize=optimization, dont_inherit=True)
        if not code_matches(current, frozen):
            raise ValueError(f"Installed module differs from frozen code: {name}")
        matched += 1
    return {"module_count": len(expected), "matching_source_code_objects": matched,
            "module_names_sha256": hashlib.sha256(json.dumps(sorted(expected)).encode()).hexdigest()}


def collected_payload(engine: Path, rows: list[tuple]) -> dict:
    """Bind notice ownership to actual collected files, including native binaries."""
    root = engine.parent.resolve()
    result = {}
    for name, _, kind in rows:
        relative = safe_relative(Path(name))
        filename = root / relative if kind == "EXECUTABLE" else root / "_internal" / relative
        if not filename.is_file():
            raise ValueError(f"Collected runtime file is missing: {name}")
        if not filename.resolve().is_relative_to(root):
            raise ValueError(f"Collected runtime file escapes engine: {name}")
        record = {"kind": kind, "sha256": sha256(filename), "bytes": filename.stat().st_size}
        if filename.is_symlink():
            record["symlink"] = str(filename.readlink())
        result[filename.relative_to(root).as_posix()] = record
    return result


def collect(engine: Path, work: Path, snapshot: Path, output: Path) -> dict:
    if output.exists():
        raise FileExistsError("Use a new notice output directory; prior inventories are immutable")
    captured = json.loads((snapshot / "BUILD_INPUT_MANIFEST.json").read_text())
    embedded_manifest = engine.parent / "_internal/build_info/BUILD_INPUT_MANIFEST.json"
    if sha256(embedded_manifest) != sha256(snapshot / "BUILD_INPUT_MANIFEST.json"):
        raise ValueError("Source snapshot does not match the frozen engine build manifest")
    tables = {name: work / name for name in ("PYZ-00.toc", "COLLECT-00.toc", "EXE-00.toc")}
    rows = {name: list(toc_rows(ast.literal_eval(path.read_text()))) for name, path in tables.items()}
    archive = verify_archive_modules(engine, rows["PYZ-00.toc"])
    payload = collected_payload(engine, rows["COLLECT-00.toc"])
    expected_versions = locked_versions(snapshot)
    included = {}
    for group in rows.values():
        for name, source, kind in group:
            if source and Path(source).is_absolute() and Path(source).is_file():
                included.setdefault(Path(source).resolve(), []).append({"name": name, "kind": kind})

    # RECORD file ownership disambiguates import names such as PIL/skimage and
    # bundled vendor modules such as setuptools._vendor. No top-level-name guess.
    distributions = []
    owned = set()
    unresolved = []
    files = {}
    output.mkdir(parents=True)
    for dist in sorted(metadata.distributions(), key=lambda d: canonical(d.metadata.get("Name", ""))):
        records = list(dist.files or ())
        actual = [(item, Path(dist.locate_file(item)).resolve()) for item in records]
        members = [(item, source) for item, source in actual if source in included]
        if not members:
            continue
        name, version = dist.metadata["Name"], dist.version
        if expected_versions.get(canonical(name)) != version:
            raise ValueError(f"Installed {name}=={version} does not match the captured build lock")
        owned.update(source for _, source in members)
        prefix = Path("python") / f"{canonical(name)}-{version}"
        notices = []
        for item, source in actual:
            relative = Path(str(item))
            if not notice_path(relative) and relative.name != "METADATA":
                continue
            safe_relative(relative)
            if not source.is_file():
                raise ValueError(f"Declared upstream notice is missing: {name}/{relative}")
            target = prefix / relative
            files[target.as_posix()] = copy_notice(source, output / target, origin=f"{name}=={version}/{relative.as_posix()}")
            if relative.name != "METADATA":
                notices.append(target.as_posix())
        if not notices:
            unresolved.append({"component": f"{name}=={version}", "reason": "No upstream notice file in installed distribution RECORD"})
        distributions.append({"name": name, "version": version,
            "declared_license_expression": dist.metadata.get("License-Expression"),
            "declared_license": dist.metadata.get("License"),
            "license_files": notices,
            "included_members": [{"installed_path": str(item), "frozen_entries": included[source]}
                                 for item, source in sorted(members, key=lambda x: str(x[0]))]})

    runtime_root = Path(sys.base_prefix).resolve()
    stdlib = Path(sysconfig.get_path("stdlib"))
    python_license = stdlib / "LICENSE.txt"
    files["python-runtime/LICENSE.txt"] = copy_notice(python_license, output / "python-runtime/LICENSE.txt", origin=f"CPython {sys.version.split()[0]}/lib/python{sys.version_info.major}.{sys.version_info.minor}/LICENSE.txt")
    runtime_sources = []
    for source, entries in sorted(included.items(), key=lambda x: str(x[0])):
        if source in owned:
            continue
        if source.is_relative_to(runtime_root) and "site-packages" not in source.parts:
            runtime_sources.append({"path": str(source.relative_to(runtime_root)), "frozen_entries": entries})
        elif "site-packages" in source.parts:
            unresolved.append({"component": str(source), "reason": "Frozen third-party file has no installed distribution owner"})
        elif not source.is_relative_to(snapshot) and not source.is_relative_to(work):
            unresolved.append({"component": str(source), "reason": "Unclassified frozen source outside captured project/build/runtime"})
    # This runtime statically links vendors; its installed PSF notice is not a
    # substitute for absent vendor attribution. Keep the gap machine-readable.
    runtime_config = stdlib / f"{sysconfig._get_sysconfigdata_name()}.py"
    if runtime_config.is_file():
        files["python-runtime/build-config.py"] = copy_notice(runtime_config, output / "python-runtime/build-config.py", origin=f"CPython {sys.version.split()[0]}/{runtime_config.name}")
    unresolved.append({"component": f"CPython {sys.version.split()[0]} static vendor libraries",
        "reason": "No verified static-vendor notice/version manifest has been configured for this exact runtime. Its upstream build provenance must supply that closure; copying CPython LICENSE.txt alone does not resolve the gap."})
    result = {"schema_version": 1, "scope": "Actual frozen archive modules and collected file owners; package-provided notices may also cover optional upstream components.",
        "collector_source_sha256": sha256(Path(__file__)),
        "engine_sha256": sha256(engine), "engine_source_digest": captured["source_digest"],
        "engine_manifest_sha256": sha256(embedded_manifest),
        "source_table_sha256": {name: sha256(path) for name, path in tables.items()},
        "captured_lock_sha256": sha256(snapshot / "requirements-lock.txt"),
        "archive_verification": archive, "distributions": distributions,
        "collected_payload": payload,
        "native_version_basis": "Captured build lock and COLLECT source ownership. Actual relocated/signed frozen payload bytes are hashed here; they are not expected to equal upstream wheel binaries after PyInstaller relocation/signing.",
        "python_runtime": {"version": sys.version.split()[0], "included_members": runtime_sources},
        "files": files, "unresolved": unresolved,
        "notice_source_completeness": "incomplete" if unresolved else "complete_for_detected_components",
        "project_license": "Not selected by this inventory; third-party terms do not license the project."}
    (output / "inventory.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", required=True, type=Path)
    parser.add_argument("--work", required=True, type=Path)
    parser.add_argument("--snapshot", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = collect(args.engine.resolve(), args.work.resolve(), args.snapshot.resolve(), args.output.resolve())
    print(json.dumps({"distributions": len(result["distributions"]), "notice_files": len(result["files"]),
                      "unresolved": result["unresolved"], "engine_sha256": result["engine_sha256"]}))


if __name__ == "__main__":
    main()

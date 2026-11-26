"""Explicit local build of the optional ``resectionlab._accel_cpp`` extension.

Run ``python -m resectionlab.build_accel_native`` from a source checkout. It
compiles ``accel_native.cpp`` in place next to this file and writes
``_accel_cpp.build.json``. ``setup.py``, ``_geometry_cpp`` and its build
manifest are not touched, so the qualified native backend identity is
unchanged. Nothing compiles when the simulator imports a module.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import shutil
import sysconfig
import tempfile

SCHEMA = 'resectionlab-accel-cpp-build-v1'
MODULE = 'resectionlab._accel_cpp'
SOURCE = 'accel_native.cpp'
# Same arithmetic contract as the main extension.
FLAGS = ('-O3', '-fno-fast-math', '-ffp-contract=off')


def _digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build(*, verbose=False):
    """Compile in place and return the manifest that was written."""
    import pybind11
    from pybind11.setup_helpers import Pybind11Extension, build_ext
    from setuptools import Distribution

    package = Path(__file__).resolve().parent
    source = package / SOURCE
    extension = Pybind11Extension(MODULE, [str(source)], cxx_std=17, extra_compile_args=list(FLAGS))
    distribution = Distribution({'name': 'resectionlab-accel', 'ext_modules': [extension],
                                 'package_dir': {'': str(package.parent)}})
    distribution.verbose = verbose
    with tempfile.TemporaryDirectory() as scratch:
        command = build_ext(distribution)
        command.inplace = True
        command.build_temp = scratch
        command.ensure_finalized()
        command.run()
        binary = Path(command.get_ext_fullpath(MODULE)).resolve()
        compiler = command.compiler.compiler_cxx[0]
    manifest = {'schema': SCHEMA, 'module': MODULE, 'sourceSha256': _digest(source),
                'recipeSha256': _digest(__file__), 'binaryName': binary.name, 'binarySha256': _digest(binary),
                'flags': list(FLAGS), 'cppStandard': 17, 'pybind11': pybind11.__version__,
                'compiler': shutil.which(compiler) or compiler,
                'python': {'version': platform.python_version(), 'abi': sysconfig.get_config_var('SOABI')},
                'platform': {'system': platform.system(), 'machine': platform.machine()}}
    (package / '_accel_cpp.build.json').write_text(json.dumps(manifest, sort_keys=True, indent=2) + '\n')
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--verbose', action='store_true')
    arguments = parser.parse_args(argv)
    print(json.dumps(build(verbose=arguments.verbose), indent=2))


if __name__ == '__main__':
    main()

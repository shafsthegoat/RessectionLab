"""The installed simulator's single, deterministic C++17 extension."""
import hashlib
import json
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import sysconfig

import pybind11
from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup

ROOT = Path(__file__).parent
SOURCES = ('geometry_native.cpp', 'state_native.hpp', 'executor_native.hpp',
           'json_native.hpp', 'motion_native.hpp', 'validation_native.hpp')
FLAGS = ['-O3', '-fno-fast-math', '-ffp-contract=off']


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class NativeBuild(build_ext):
    def build_extensions(self):
        compiler = self.compiler.compiler_cxx[0]
        compiler_path = shutil.which(compiler) or compiler
        identity = {
            'schema': 'resectionlab-geometry-cpp-v1',
            'sources': {name: digest(ROOT / 'src/resectionlab' / name) for name in SOURCES},
            'recipeSha256': digest(ROOT / 'setup.py'),
            'compiler': {'path': compiler_path, 'version': subprocess.check_output(
                [compiler_path, '--version'], text=True).splitlines()[0]},
            'flags': list(self.extensions[0].extra_compile_args), 'cppStandard': 17, 'pybind11': pybind11.__version__,
            'python': {'version': platform.python_version(), 'abi': sysconfig.get_config_var('SOABI')},
            'platform': {'system': platform.system(), 'machine': platform.machine()},
        }
        self.native_identity = identity
        encoded = json.dumps(identity, sort_keys=True, separators=(',', ':'))
        for extension in self.extensions:
            extension.define_macros.append(('RSL_BUILD_IDENTITY', json.dumps(encoded)))
        super().build_extensions()

    def run(self):
        super().run()
        for extension in self.extensions:
            binary = Path(self.get_ext_fullpath(extension.name))
            manifest = {'schema': 'resectionlab-geometry-cpp-build-v1',
                        'buildIdentity': self.native_identity,
                        'binaryName': binary.name, 'binarySha256': digest(binary)}
            binary.with_name('_geometry_cpp.build.json').write_text(
                json.dumps(manifest, sort_keys=True, indent=2) + '\n')


setup(ext_modules=[Pybind11Extension('resectionlab._geometry_cpp',
    ['src/resectionlab/geometry_native.cpp'], cxx_std=17,
    depends=['src/resectionlab/' + name for name in SOURCES[1:]] + ['setup.py'],
    extra_compile_args=FLAGS)], cmdclass={'build_ext': NativeBuild})

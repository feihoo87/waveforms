import os

from Cython.Build import cythonize
from setuptools import Extension, find_packages, setup


def module_name(dirpath, filename):
    l = [os.path.splitext(filename)[0]]
    head = dirpath
    while True:
        head, tail = os.path.split(head)
        l.append(tail)
        if not head:
            break
    return '.'.join(reversed(l))


def get_extensions():
    #from numpy.distutils.misc_util import get_numpy_include_dirs
    #from pathlib import Path

    extensions = [
        Extension('waveforms.sys.net._kcp', ['src/kcp.c', 'src/ikcp.c'],
                  include_dirs=['src']),
        # Extension(
        #     'waveforms.math.npufunc',
        #     ['src/multi_type_logit.c'],
        #     include_dirs=get_numpy_include_dirs(),
        #     library_dirs=[str(Path(p).parent / 'lib') for p in get_numpy_include_dirs()],
        #     libraries=["npymath"],
        #     # extra_compile_args=['-std=c99'],
        # )
    ]

    for dirpath, dirnames, filenames in os.walk('waveforms'):
        for filename in filenames:
            if filename.endswith('.pyx'):
                extensions.append(
                    Extension(module_name(dirpath, filename),
                              [os.path.join(dirpath, filename)]))

    return extensions


setup(packages=find_packages(),
      ext_modules=cythonize(get_extensions(), build_dir=f"build"))
